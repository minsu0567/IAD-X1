import argparse, json, time, csv, os, math

import huggingface_hub
if not hasattr(huggingface_hub, 'is_offline_mode'):
    from huggingface_hub.constants import HF_HUB_OFFLINE
    huggingface_hub.is_offline_mode = lambda: bool(HF_HUB_OFFLINE)

from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

GENERAL_QUESTION_PROMPT = (
    'You are an expert in detecting industrial anomalies in images. '
    'You will be provided with two images, a reference image (ref_img) as a normal standard and a query image (query_img) for inspection. '
    'Your reasoning and response must strictly follow these constraints based on the specific tags. '
    '\n1. Analyze only the ref_img and define the characteristics of a normal state such as structural integrity, surface texture, and component completeness. '
    '2. Inspect the query_img for any anomalies by comparing it against the ref_img baseline. '
    '\nIf you find anomalies in the query image, respond with <type>...</type><location>...</location><answer>Yes</answer>'
    '\nIf no anomalies are detected in the query image, respond with <answer>No</answer> '
)

FIELDNAMES = ['batch', 'benchmark', 'batch_size',
              'inference_time_s', 'decode_time_s',
              'ttft_s', 'e2e_latency_s',
              'prompt_tokens', 'output_tokens',
              'output_throughput_tok_s', 'total_throughput_tok_s', 'latency_per_sample_s']
MEAN_COLS = [c for c in FIELDNAMES if c not in ('batch', 'benchmark')]

DECODE_COLS = ('decode_time_s', 'output_throughput_tok_s')


def is_degenerate(rec):
    return math.isnan(rec['output_throughput_tok_s'])


def build_prompt(processor):
    conversation = [
        {'role': 'user', 'content': [
            {'type': 'text', 'text': GENERAL_QUESTION_PROMPT + '\nThe first image is a normal reference sample. '},
            {'type': 'image'},
            {'type': 'text', 'text': '\n'},
            {'type': 'image'},
            {'type': 'text', 'text': '\nAre there any defects in the query image?'},
        ]},
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    prompt = prompt.rstrip()
    if prompt.endswith('<think>'):
        prompt = prompt[:-len('<think>')]
    return prompt


def load_image(path, resize_to):
    img = Image.open(path).convert('RGB')
    if resize_to is not None and max(img.size) > resize_to:
        scale = resize_to / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    return img


def chunks(lst, m):
    for i in range(0, len(lst), m):
        yield lst[i:i + m]


def count_output_tokens(outs, eos_id):
    total = 0
    for o in outs:
        ids = list(o.outputs[0].token_ids)
        n = len(ids)
        if o.outputs[0].finish_reason == 'stop' and eos_id is not None and (not ids or ids[-1] != eos_id):
            n += 1
        total += n
    return total


def mean_row(rows, label, benchmark=''):
    row = {'batch': label, 'benchmark': benchmark}
    ok = [r for r in rows if not is_degenerate(r)]
    for c in MEAN_COLS:
        src = ok if c in DECODE_COLS else rows
        vals = [r[c] for r in src if not math.isnan(r[c])]
        row[c] = (sum(vals) / len(vals)) if vals else float('nan')
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--hf-token', default=None)
    ap.add_argument('--manifest-json', default=None,
                    help='throughput_manifest_180.json; required unless --demo-only')
    ap.add_argument('--limit-per-benchmark', type=int, default=0,
                    help='cap samples per benchmark for a quick smoke test; 0 = use all')
    ap.add_argument('--batch-size', type=int, default=1,
                    help='1 matches the HF notebooks; raise it to exercise vLLM batching '
                         '(then the numbers are no longer HF-comparable)')
    ap.add_argument('--warmup-batches', type=int, default=1)
    ap.add_argument('--resize-to', type=int, default=512, help='<=0 to disable resize')
    ap.add_argument('--max-model-len', type=int, default=32768)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--gpu-mem', type=float, default=0.9)
    ap.add_argument('--enforce-eager', action='store_true')
    ap.add_argument('--csv', default='/content/batch_inference_benchmark_vllm_qwen35.csv')
    ap.add_argument('--demo-ref', default=None)
    ap.add_argument('--demo-query', default=None)
    ap.add_argument('--demo-only', action='store_true',
                    help='run only the (demo-ref, demo-query) inference, print it, and exit (no benchmark)')
    args = ap.parse_args()

    if args.demo_only:
        if not (args.demo_ref and args.demo_query):
            raise SystemExit('--demo-only requires both --demo-ref and --demo-query.')
        if not (os.path.isfile(args.demo_ref) and os.path.isfile(args.demo_query)):
            raise SystemExit(f'demo image not found: {args.demo_ref} | {args.demo_query}')
    elif not args.manifest_json:
        raise SystemExit('--manifest-json is required unless --demo-only is set.')

    if args.hf_token:
        huggingface_hub.login(token=args.hf_token)

    resize_to = args.resize_to if args.resize_to and args.resize_to > 0 else None

    llm = LLM(
        model=args.model_id,
        dtype='bfloat16',
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={'image': 2},
        enable_prefix_caching=False,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    eos_id = processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else None
    eos_ids = [eos_id] if eos_id is not None else None
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, stop_token_ids=eos_ids,
    )
    ttft_sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1, ignore_eos=True, stop_token_ids=None,
    )
    prompt_text = build_prompt(processor)
    print('Model loaded.')
    print('-' * 60)
    print(prompt_text)
    print('-' * 60)

    if args.demo_ref and args.demo_query and os.path.isfile(args.demo_ref) and os.path.isfile(args.demo_query):
        out = llm.generate(
            [{'prompt': prompt_text,
              'multi_modal_data': {'image': [load_image(args.demo_ref, resize_to),
                                             load_image(args.demo_query, resize_to)]}}],
            sampling_params=sampling_params,
        )
        print('=' * 60); print('[Demo Response]'); print('=' * 60)
        print(out[0].outputs[0].text)

    if args.demo_only:
        print('demo-only mode: benchmark skipped.')
        return

    with open(args.manifest_json, encoding='utf-8') as f:
        manifest = json.load(f)
    print(f'Loaded {len(manifest)} samples from {args.manifest_json}')

    by_bench = {}
    for item in manifest:
        ref_path, query_path = item['ref_path'], item['query_path']
        if not (os.path.isfile(ref_path) and os.path.isfile(query_path)):
            print(f'  [skip] missing image: {ref_path} | {query_path}')
            continue
        bucket = by_bench.setdefault(item['benchmark'], [])
        if args.limit_per_benchmark and len(bucket) >= args.limit_per_benchmark:
            continue
        bucket.append({'prompt': prompt_text,
                       'multi_modal_data': {'image': [load_image(ref_path, resize_to),
                                                      load_image(query_path, resize_to)]}})
    n_prepared = sum(len(v) for v in by_bench.values())
    print(f'Prepared {n_prepared} valid requests across {len(by_bench)} benchmarks.')
    if not n_prepared:
        raise SystemExit('No valid requests (check the manifest / Drive mount).')

    warm = next(iter(by_bench.values()))[:args.batch_size]
    for _ in range(args.warmup_batches):
        _ = llm.generate(warm, sampling_params=ttft_sampling_params)
        _ = llm.generate(warm, sampling_params=sampling_params)

    records = []
    bidx = 0
    for bench, reqs in by_bench.items():
        for batch in chunks(reqs, args.batch_size):
            bidx += 1
            t0 = time.perf_counter()
            _ = llm.generate(batch, sampling_params=ttft_sampling_params)
            t_ttft = time.perf_counter() - t0
            t1 = time.perf_counter()
            outs = llm.generate(batch, sampling_params=sampling_params)
            t_full = time.perf_counter() - t1

            out_tok = count_output_tokens(outs, eos_id)
            in_tok = sum(len(o.prompt_token_ids) for o in outs)

            decode_time = t_full - t_ttft
            decode_steps = out_tok - len(batch)
            if decode_time <= 0 or decode_steps <= 0:
                print(f'  [warn] batch {bidx}: t_ttft ({t_ttft:.3f}s) >= t_full ({t_full:.3f}s) or '
                      f'out_tok<=batch_size - measurement noise / degenerate output; '
                      f'throughput set to NaN and this batch is dropped from the decode aggregates.')
                decode_thr = float('nan')
            else:
                decode_thr = decode_steps / decode_time
            ttft = t_ttft
            e2e = t_full

            records.append({
                'batch': bidx, 'benchmark': bench, 'batch_size': len(batch),
                'inference_time_s': t_full, 'decode_time_s': decode_time,
                'ttft_s': ttft, 'e2e_latency_s': e2e,
                'prompt_tokens': in_tok, 'output_tokens': out_tok,
                'output_throughput_tok_s': decode_thr,
                'total_throughput_tok_s': (in_tok + out_tok) / t_full,
                'latency_per_sample_s': e2e / len(batch),
            })
            print(f'[batch {bidx}/{bench}] n={len(batch)} e2e={e2e:.3f}s '
                  f'ttft={ttft:.3f}s decode={decode_time:.3f}s '
                  f'in_tok={in_tok} out_tok={out_tok} out_decode={decode_thr:.1f} tok/s '
                  f'total={(in_tok + out_tok) / t_full:.1f} tok/s lat/sample={e2e / len(batch):.3f}s')

    n_batches = len(records)
    ok_records = [r for r in records if not is_degenerate(r)]
    n_bad = n_batches - len(ok_records)
    bad_by_bench = {}
    for r in records:
        if is_degenerate(r):
            bad_by_bench[r['benchmark']] = bad_by_bench.get(r['benchmark'], 0) + 1

    total_time = sum(r['inference_time_s'] for r in records)
    total_ttft = sum(r['ttft_s'] for r in records)
    total_decode = sum(r['decode_time_s'] for r in ok_records)
    total_e2e = sum(r['e2e_latency_s'] for r in records)
    total_out = sum(r['output_tokens'] for r in records)
    total_in = sum(r['prompt_tokens'] for r in records)
    total_n = sum(r['batch_size'] for r in records)
    total_steps = sum(r['output_tokens'] - r['batch_size'] for r in ok_records)
    gen_thr = total_steps / total_decode if total_decode > 0 else float('nan')
    total_thr = (total_in + total_out) / total_time
    samples_per_sec = total_n / total_e2e
    avg_sample_lat = total_e2e / total_n
    avg_ttft = total_ttft / n_batches

    bench_order = list(dict.fromkeys(r['benchmark'] for r in records))
    bench_rows = [mean_row([r for r in records if r['benchmark'] == b], 'BENCH_AVERAGE', b)
                  for b in bench_order]
    avg_row = mean_row(records, 'AVERAGE', 'ALL')

    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in records:
            w.writerow(r)
        for r in bench_rows:
            w.writerow(r)
        w.writerow({'batch': 'SUMMARY', 'benchmark': 'ALL', 'batch_size': total_n,
                    'inference_time_s': total_time, 'decode_time_s': total_decode,
                    'ttft_s': total_ttft, 'e2e_latency_s': total_e2e,
                    'prompt_tokens': total_in, 'output_tokens': total_out,
                    'output_throughput_tok_s': gen_thr, 'total_throughput_tok_s': total_thr,
                    'latency_per_sample_s': avg_sample_lat})
        w.writerow(avg_row)

    print('=' * 60)
    print(f'[Batch Benchmark] N={total_n} batch_size={args.batch_size} '
          f'batches={n_batches} warmup={args.warmup_batches}')
    print(f'  total end-to-end time        : {total_e2e:.3f} s  (Pass B)')
    print(f'  total TTFT time (Pass A)     : {total_ttft:.3f} s  (preprocess+prefill, engine-internal)')
    print(f'  total decode-only time       : {total_decode:.3f} s  (over {len(ok_records)}/{n_batches} batches)')
    print(f'  avg TTFT / batch             : {avg_ttft:.3f} s')
    print(f'  avg e2e latency / sample     : {avg_sample_lat:.3f} s')
    print(f'  output throughput (decode)   : {gen_thr:.1f} tok/s')
    print(f'  total throughput (prompt+out): {total_thr:.1f} tok/s')
    print(f'  samples per second (e2e)     : {samples_per_sec:.2f} samples/s')
    print(f'  avg prompt tokens / sample   : {total_in / total_n:.1f}')
    print(f'  avg output tokens / sample   : {total_out / total_n:.1f}')
    if n_bad:
        print(f'  !! degenerate batches        : {n_bad}/{n_batches} excluded from decode '
              f'aggregates (decode_time_s, output_throughput_tok_s) -> {bad_by_bench}')
        print(f'     Their ttft/e2e/token columns still count. If this is more than a couple of '
              f'batches the run is too noisy to quote - re-run it.')
    print('-' * 60)
    print('[Per-benchmark AVERAGE]')
    print(f"  {'benchmark':10s} {'n':>3s} {'e2e_s':>8s} {'ttft_s':>8s} {'decode_s':>9s} "
          f"{'out_tok':>8s} {'decode_tok/s':>13s} {'bad':>4s}")
    for b, r in zip(bench_order, bench_rows):
        cnt = sum(x['batch_size'] for x in records if x['benchmark'] == b)
        print(f"  {b:10s} {cnt:3d} {r['e2e_latency_s']:8.3f} {r['ttft_s']:8.3f} "
              f"{r['decode_time_s']:9.3f} {r['output_tokens']:8.1f} "
              f"{r['output_throughput_tok_s']:13.1f} {bad_by_bench.get(b, 0):4d}")
    print('-' * 60)
    print(f'[Per-batch AVERAGE] (mean over {n_batches} batches, '
          f'{len(ok_records)} for decode columns)')
    for c in MEAN_COLS:
        print(f'  {c:28s} : {avg_row[c]:.3f}')
    print(f'  CSV saved -> {args.csv}')
    print('=' * 60)


if __name__ == '__main__':
    main()
