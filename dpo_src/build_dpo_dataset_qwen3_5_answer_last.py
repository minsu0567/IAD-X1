import argparse
import json
import os
import re
import shutil

import huggingface_hub
if not hasattr(huggingface_hub, 'is_offline_mode'):
    from huggingface_hub.constants import HF_HUB_OFFLINE
    huggingface_hub.is_offline_mode = lambda: bool(HF_HUB_OFFLINE)

from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

GENERAL_QUESTION_PROMPT = (
    'You are an expert in detecting industrial anomalies in images. '
    'You will be provided with two images, a reference image (ref_img) as a normal standard and a query image (query_img) for inspection. '
    'Your reasoning and response must strictly follow these constraints based on the specific tags. '
    '\n1. Analyze only the ref_img and define the characteristics of a normal state such as structural integrity, surface texture, and component completeness. '
    '2. Inspect the query_img for any anomalies by comparing it against the ref_img baseline. '
    '\nIf you find anomalies in the query image, respond with <type>...</type><location>...</location><answer>Yes</answer>'
    '\nIf no anomalies are detected in the query image, respond with <answer>No</answer> '
)

TAG_RE = {
    'answer': re.compile(r'<answer>\s*(.*?)\s*</answer>', re.IGNORECASE | re.DOTALL),
    'location': re.compile(r'<location>\s*(.*?)\s*</location>', re.IGNORECASE | re.DOTALL),
    'type': re.compile(r'<type>\s*(.*?)\s*</type>', re.IGNORECASE | re.DOTALL),
}

FORMAT_NO_RE = re.compile(r'^\s*<answer>\s*no\s*</answer>\s*$', re.IGNORECASE | re.DOTALL)
FORMAT_YES_RE = re.compile(
    r'^\s*<type>.*?</type>\s*<location>.*?</location>\s*<answer>\s*yes\s*</answer>\s*$',
    re.IGNORECASE | re.DOTALL,
)

STRICT_FORMAT = True


def build_prompt(processor):
    conversation = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': GENERAL_QUESTION_PROMPT + '\nThe first image is a normal reference sample. '},
            {'type': 'image'},
            {'type': 'text', 'text': '\n'},
            {'type': 'image'},
            {'type': 'text', 'text': '\nAre there any defects in the query image?'},
        ],
    }]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    prompt = prompt.rstrip()
    if prompt.endswith('<think>'):
        prompt = prompt[: -len('<think>')]
    return prompt


def load_resized(path, target=512):
    img = Image.open(path).convert('RGB')
    if target and max(img.size) > target:
        scale = target / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    return img


def extract_tag(text, name):
    if text is None:
        return None
    m = TAG_RE[name].search(text)
    return m.group(1).strip().lower() if m else None


def matches_format(text, gt_answer):
    if text is None:
        return False
    if not STRICT_FORMAT:
        need = ('answer',) if gt_answer == 'no' else ('type', 'location', 'answer')
        return all(TAG_RE[t].search(text) is not None for t in need)
    if gt_answer == 'no':
        return FORMAT_NO_RE.match(text) is not None
    if gt_answer == 'yes':
        return FORMAT_YES_RE.match(text) is not None
    return False


def is_hard_sample(gt_content, pred_content):
    gt_answer = extract_tag(gt_content, 'answer')
    pred_answer = extract_tag(pred_content, 'answer')

    if gt_answer == 'no':
        if pred_answer == 'yes':
            return True
        if not matches_format(pred_content, 'no'):
            return True
        return False

    if gt_answer == 'yes':
        for tag in ('type', 'location', 'answer'):
            if extract_tag(gt_content, tag) != extract_tag(pred_content, tag):
                return True
        if not matches_format(pred_content, 'yes'):
            return True
        return False

    return False


def run_debug(llm, sampling_params, prompt_text, samples, resize_to):
    reqs = [{'prompt': prompt_text,
             'multi_modal_data': {'image': [load_resized(s['image'][0], resize_to),
                                            load_resized(s['image'][1], resize_to)]}}
            for s in samples]
    outs = llm.generate(reqs, sampling_params=sampling_params)
    for s, o in zip(samples, outs):
        pred = o.outputs[0].text
        gt = s['solution']
        print('=' * 70)
        print(repr(pred))
        print('-' * 70)
        print('GT   :', gt)
        print('PRED : answer=', extract_tag(pred, 'answer'),
              '| type=', extract_tag(pred, 'type'),
              '| location=', extract_tag(pred, 'location'))
        print('has <think> :', '<think>' in pred or '</think>' in pred)
        print('HARD SAMPLE :', is_hard_sample(gt, pred))


def main():
    global STRICT_FORMAT

    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--hf-token', default=None)
    ap.add_argument('--input-json', required=True)
    ap.add_argument('--output-json', required=True)
    ap.add_argument('--local-json', default='/content/dpo_hard_samples_answer_last.json')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--checkpoint-every', type=int, default=10)
    ap.add_argument('--resize-to', type=int, default=512)
    ap.add_argument('--max-model-len', type=int, default=8192)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--gpu-mem', type=float, default=0.9)
    ap.add_argument('--no-strict-format', action='store_true')
    ap.add_argument('--debug-n', type=int, default=3)
    ap.add_argument('--debug-only', action='store_true')
    args = ap.parse_args()

    STRICT_FORMAT = not args.no_strict_format
    resize_to = args.resize_to if args.resize_to > 0 else None

    if args.hf_token:
        huggingface_hub.login(token=args.hf_token)

    os.environ['VLLM_LOGGING_LEVEL'] = 'INFO'
    os.environ['VLLM_TRACE_FUNCTION'] = '0'

    llm = LLM(
        model=args.model_id,
        dtype='bfloat16',
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=False,
        limit_mm_per_prompt={'image': 2},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    prompt_text = build_prompt(processor)
    print('Model loaded.')
    print(prompt_text[-400:])

    with open(args.input_json, encoding='utf-8') as f:
        grpo_samples = json.load(f)
    print(f'Loaded {len(grpo_samples)} samples')

    if args.debug_n > 0:
        run_debug(llm, sampling_params, prompt_text, grpo_samples[:args.debug_n], resize_to)

    if args.debug_only:
        return

    dpo_samples = []
    n_pass = n_hard = n_error = n_blank = 0

    def save_local():
        with open(args.local_json, 'w', encoding='utf-8') as f:
            json.dump(dpo_samples, f, ensure_ascii=False, indent=2)

    for batch_idx, start in enumerate(tqdm(range(0, len(grpo_samples), args.batch_size))):
        chunk = grpo_samples[start:start + args.batch_size]

        reqs, valid = [], []
        for j, s in enumerate(chunk):
            try:
                ref, query = s['image'][0], s['image'][1]
                reqs.append({'prompt': prompt_text,
                             'multi_modal_data': {'image': [load_resized(ref, resize_to),
                                                            load_resized(query, resize_to)]}})
                valid.append(j)
            except Exception as e:
                n_error += 1
                print(f'[{start + j}] image load ERROR: {e}')

        try:
            if reqs:
                outs = llm.generate(reqs, sampling_params=sampling_params)
                for j, o in zip(valid, outs):
                    s = chunk[j]
                    pred = o.outputs[0].text
                    gt = s['solution']

                    if not is_hard_sample(gt, pred):
                        n_pass += 1
                        continue

                    if not pred.strip():
                        n_blank += 1
                        continue

                    dpo_samples.append({
                        'images': s['image'],
                        'messages': [{'role': 'user', 'content': s['problem']}],
                        'chosen': {'role': 'assistant', 'content': gt},
                        'rejected': {'role': 'assistant', 'content': pred},
                    })
                    n_hard += 1
        except Exception as e:
            print(f'[batch {batch_idx} @ start {start}] generate/score ERROR: {e}')

        save_local()
        if (batch_idx + 1) % args.checkpoint_every == 0:
            shutil.copy(args.local_json, args.output_json)
            print(f'  [checkpoint] batch {batch_idx + 1} -> {args.output_json} ({len(dpo_samples)} pairs)')

    save_local()
    shutil.copy(args.local_json, args.output_json)

    print(f'\nTotal : {len(grpo_samples)}')
    print(f'Pass  : {n_pass}')
    print(f'Hard  : {n_hard}')
    print(f'Blank : {n_blank}')
    print(f'Error : {n_error}')
    print(f'Saved -> local: {args.local_json} | drive: {args.output_json}')


if __name__ == '__main__':
    main()
