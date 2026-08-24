import argparse
import json
import logging
import os
import re
import sys
from typing import List, Optional, Tuple

import huggingface_hub
if not hasattr(huggingface_hub, "is_offline_mode"):
    try:
        from huggingface_hub.constants import HF_HUB_OFFLINE
        huggingface_hub.is_offline_mode = lambda: bool(HF_HUB_OFFLINE)
    except ImportError:
        pass

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

BENCHMARKS = ["VisA", "SDD", "MPDD", "DTD", "DS-MVTec", "DAGM"]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

GENERAL_QUESTION_PROMPT = (
    'You are an expert in detecting industrial anomalies in images. '
    'You will be provided with two images, a reference image (ref_img) as a normal standard and a query image (query_img) for inspection. '
    'Your reasoning and response must strictly follow these constraints based on the specific tags. '
    '\n1. Analyze only the ref_img and define the characteristics of a normal state such as structural integrity, surface texture, and component completeness. '
    '2. Inspect the query_img for any anomalies by comparing it against the ref_img baseline. '
    '\nIf you find anomalies in the query image, respond with <type>...</type><location>...</location><answer>Yes</answer>'
    '\nIf no anomalies are detected in the query image, respond with <answer>No</answer> '
)


def parse_conversation(text_gt: dict) -> Tuple[List[dict], List[str]]:
    questions, answers = [], []
    for key in text_gt.keys():
        if not key.startswith("conversation"):
            continue
        for i, QA in enumerate(text_gt[key]):
            options_items = list(QA["Options"].items())
            options_text = ""
            new_answer_key = None
            for new_key, (original_key, value) in enumerate(options_items):
                options_text += f"{chr(65 + new_key)}. {value}\n"
                if QA["Answer"] == original_key:
                    new_answer_key = chr(65 + new_key)
            option_dict = {chr(65 + nk): v for nk, (_, v) in enumerate(options_items)}
            questions.append({
                "type": "text",
                "text": f"Question {i + 1}: {QA['Question']} \n{options_text}",
                "options": option_dict,
            })
            if new_answer_key is None:
                raise ValueError("Answer key not found.")
            answers.append(new_answer_key)
        break
    return questions, answers


def get_ans(response_text: str, options: Optional[dict] = None) -> str:
    try:
        ans_match = re.search(r"<answer>(.*?)</answer>", response_text)
        gpt_answer = ans_match.group(1).strip().lower()
        if options is None:
            return gpt_answer
        for key, value in options.items():
            if gpt_answer == value.lower().strip("."):
                return key
        for key, value in options.items():
            option_clean = value.lower().strip(".").strip()
            if gpt_answer in option_clean or option_clean in gpt_answer:
                return key
        return "E"
    except (AttributeError, TypeError):
        return "E"


def build_prompt(processor) -> str:
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": GENERAL_QUESTION_PROMPT + "\nThe first image is a normal reference sample. "},
            {"type": "image"},
            {"type": "text", "text": "\n"},
            {"type": "image"},
            {"type": "text", "text": "\nAre there any defects in the query image?"},
        ],
    }]
    return processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)


_GOOD_DIR_CACHE = {}


def _list_good_dir(good_dir_abs: str) -> List[str]:
    if good_dir_abs not in _GOOD_DIR_CACHE:
        if not os.path.isdir(good_dir_abs):
            _GOOD_DIR_CACHE[good_dir_abs] = []
        else:
            files = [f for f in os.listdir(good_dir_abs) if f.lower().endswith(IMAGE_EXTS)]
            files.sort(reverse=True)
            _GOOD_DIR_CACHE[good_dir_abs] = files
    return _GOOD_DIR_CACHE[good_dir_abs]


def resolve_reference_path(
    benchmark: str,
    query_rel_path: str,
    text_gt: dict,
    data_root: str,
) -> Optional[str]:

    parts = query_rel_path.split("/")
    if len(parts) < 3:
        return None

    good_dir_rel = "/".join(parts[:-2] + ["good"])
    good_dir_abs = os.path.join(data_root, good_dir_rel)

    files = _list_good_dir(good_dir_abs)
    if not files:
        return None

    query_filename = parts[-1]
    query_is_good = parts[-2] == "good"
    chosen = files[0]
    if query_is_good and chosen == query_filename:
        if len(files) < 2:
            return None
        chosen = files[1]
    return os.path.join(good_dir_abs, chosen)


def build_batch(chat_ad: dict, args, existing_images: set):
    batch_data = []
    skipped_no_ref = 0
    for image_path in tqdm(chat_ad.keys(), desc="Preparing data"):
        if image_path in existing_images and not args.reproduce:
            continue
        text_gt = chat_ad[image_path]

        ref_rel_or_abs = resolve_reference_path(
            benchmark=args.benchmark,
            query_rel_path=image_path,
            text_gt=text_gt,
            data_root=args.data_root,
        )
        if ref_rel_or_abs is None:
            skipped_no_ref += 1
            continue

        query_abs = os.path.join(args.data_root, image_path)
        batch_data.append({
            "image_path": query_abs,
            "ref_path": ref_rel_or_abs,
            "text_gt": text_gt,
            "original_image_path": image_path,
        })
    if skipped_no_ref:
        print(f"[warn] skipped {skipped_no_ref} samples (no reference image found)")
    return batch_data


def summarize_references(batch_data, data_root: str) -> None:
    per_cat = {}
    for item in batch_data:
        cat = item["original_image_path"].split("/")[1] if "/" in item["original_image_path"] else "-"
        ref = os.path.relpath(item["ref_path"], data_root).replace(os.sep, "/")
        per_cat.setdefault(cat, {}).setdefault(ref, 0)
        per_cat[cat][ref] += 1
    print(f"[ref] resolved references for {len(per_cat)} categories:")
    for cat in sorted(per_cat):
        refs = per_cat[cat]
        head = ", ".join(f"{r} (x{n})" for r, n in sorted(refs.items(), key=lambda kv: -kv[1])[:3])
        more = "" if len(refs) <= 3 else f" ... +{len(refs) - 3} more"
        expected = len(refs) == 1 or (len(refs) == 2 and min(refs.values()) == 1)
        flag = "" if expected else "  [!] reference did not resolve uniformly"
        print(f"  {cat:<14} {head}{more}{flag}")


def run_batch(llm: LLM, processor, batch, sampling_params, prompt_text: str):
    inputs, metadata = [], []
    for item in batch:
        questions, answers = parse_conversation(item["text_gt"])
        if not questions or not answers:
            continue
        questions = questions[0:1]
        answers = answers[0:1]
        try:
            ref_img = Image.open(item["ref_path"]).convert("RGB").resize((512, 512))
            query_img = Image.open(item["image_path"]).convert("RGB").resize((512, 512))
        except Exception as e:
            print(f"[warn] failed to load images for {item['original_image_path']}: {e}")
            continue
        inputs.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": [ref_img, query_img]},
        })
        metadata.append({
            "image_path": item["image_path"],
            "original_image_path": item["original_image_path"],
            "questions": questions,
            "answers": answers,
            "text_gt": item["text_gt"],
        })

    if not inputs:
        return []

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    results = []
    for output, meta in zip(outputs, metadata):
        response = output.outputs[0].text
        gpt_answer = get_ans(response, meta["questions"][0]["options"])
        if not gpt_answer:
            gpt_answer = response
            logging.error(f"No matching answer at {meta['image_path']}: {meta['questions']}")
        results.append({
            "original_image_path": meta["original_image_path"],
            "questions": meta["questions"],
            "answers": meta["answers"],
            "gpt_answers": [gpt_answer],
            "text_gt": meta["text_gt"],
            "raw_response": response,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True, choices=BENCHMARKS)
    parser.add_argument("--model_path", type=str, default="minsu0567/Uni-IAD-R2-Qwen3.5")
    parser.add_argument("--data_root", type=str, default="/content/drive/MyDrive/Uni-IAD_eval_dataset")
    parser.add_argument("--output_dir", type=str, default="/content/eval_results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Disable CUDA Graph/torch.compile. Default (flag absent) keeps them ON "
                             "for faster decode; pass this flag for max stability on the hybrid Mamba path.")
    parser.add_argument("--reproduce", action="store_true",
                        help="Re-run inference on samples that already exist in the output JSON.")
    parser.add_argument("--iad_r1_root", type=str, default="/content/drive/MyDrive/IAD-R1-main",
                        help="Path to IAD-R1-main (for helper/summary.py import).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Resolve references and print the per-category summary, then exit "
                             "without loading vLLM. Use this to verify the reference rule.")
    args = parser.parse_args()

    torch.manual_seed(6666)

    if os.path.isdir(args.iad_r1_root):
        sys.path.append(args.iad_r1_root)
    try:
        from helper.summary import caculate_accuracy_mmad
    except ImportError:
        print(f"[warn] could not import helper.summary from {args.iad_r1_root}; final metric step will be skipped.")
        caculate_accuracy_mmad = None

    json_filename = f"test_{args.benchmark}_format.json"
    json_path = os.path.join(args.data_root, args.benchmark, json_filename)
    assert os.path.isfile(json_path), f"Benchmark JSON not found: {json_path}"
    with open(json_path, "r") as f:
        chat_ad = json.load(f)
    print(f"[info] loaded {len(chat_ad)} samples from {json_path}")

    out_subdir = os.path.join(args.output_dir, args.benchmark)
    os.makedirs(out_subdir, exist_ok=True)
    model_tag = os.path.basename(args.model_path.rstrip("/"))
    answers_json_path = os.path.join(out_subdir, f"answers_{model_tag}_vllm_fixedref.json")
    print(f"[info] results -> {answers_json_path}")

    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as f:
            all_answers_json = json.load(f)
    else:
        all_answers_json = []
    existing_images = {a["image"] for a in all_answers_json}

    batch_data = build_batch(chat_ad, args, existing_images)
    summarize_references(batch_data, args.data_root)
    print(f"[info] {len(batch_data)} samples to process (resume mode = {not args.reproduce})")

    if args.dry_run:
        print("[info] --dry_run set; exiting before vLLM init.")
        return

    print("[info] initializing vLLM ...")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={"image": 2},
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop_token_ids=[processor.tokenizer.eos_token_id] if hasattr(processor, "tokenizer") else None,
    )
    prompt_text = build_prompt(processor)
    print("[info] vLLM ready.")

    total_batches = (len(batch_data) + args.batch_size - 1) // args.batch_size
    for i in tqdm(range(0, len(batch_data), args.batch_size),
                  desc="Processing batches", total=total_batches):
        batch = batch_data[i:i + args.batch_size]
        results = run_batch(llm, processor, batch, sampling_params, prompt_text)

        for result in results:
            questions = result["questions"]
            answers = result["answers"]
            gpt_answers = result["gpt_answers"]
            text_gt = result["text_gt"]
            original_path = result["original_image_path"]

            if gpt_answers is None or len(gpt_answers) != len(answers):
                print(f"[warn] error at {original_path}")
                continue

            questions_type = [c["type"] for c in text_gt["conversation"]]
            for q, a, ga, qt in zip(questions, answers, gpt_answers, questions_type):
                all_answers_json.append({
                    "image": original_path,
                    "question": q,
                    "question_type": qt,
                    "correct_answer": a,
                    "gpt_answer": ga,
                })

        with open(answers_json_path, "w") as f:
            json.dump(all_answers_json, f, indent=4)

    if caculate_accuracy_mmad is not None:
        print("[info] computing accuracy ...")
        caculate_accuracy_mmad(answers_json_path, show_overkill_miss=True)
    else:
        correct = sum(1 for a in all_answers_json
                      if a["question_type"] == "Anomaly Detection"
                      and a["correct_answer"] == a["gpt_answer"])
        total = sum(1 for a in all_answers_json if a["question_type"] == "Anomaly Detection")
        if total:
            print(f"[fallback] Anomaly Detection accuracy: {correct}/{total} = {correct/total*100:.2f}%")


if __name__ == "__main__":
    main()
