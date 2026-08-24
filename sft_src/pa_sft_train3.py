import json
import os
import sys

from llamafactory.train.tuner import run_exp

# ANSWER-LAST order: <type> -> <location> -> <answer> (matches merged_reordered_answer_last.json)
GENERAL_QUESTION_PROMPT = (
    'You are an expert in detecting industrial anomalies in images. '
    'You will be provided with two images, a reference image (ref_img) as a normal standard and a query image (query_img) for inspection. '
    'Your reasoning and response must strictly follow these constraints based on the specific tags. '
    '\n1. Analyze only the ref_img and define the characteristics of a normal state such as structural integrity, surface texture, and component completeness. '
    '2. Inspect the query_img for any anomalies by comparing it against the ref_img baseline. '
    '\nIf you find anomalies in the query image, respond with <type>...</type><location>...</location><answer>Yes</answer>'
    '\nIf no anomalies are detected in the query image, respond with <answer>No</answer> '
)


def _arg(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def _set_arg(flag, value):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        sys.argv[i + 1] = value
    else:
        sys.argv.extend([flag, value])


def inject_prompt():
    """Build a prompt-injected copy of the dataset and rewire dataset_info.json.

    Returns the path to a temporary dataset_info.json so the caller can clean
    it up after training (the injected dataset JSON sits next to the original).
    """
    dataset_name = _arg("--dataset")
    dataset_dir = _arg("--dataset_dir")
    if not dataset_name or not dataset_dir:
        return None

    info_path = os.path.join(dataset_dir, "dataset_info.json")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    if dataset_name not in info:
        return None

    entry = info[dataset_name]
    src_json = entry["file_name"]
    if not os.path.isabs(src_json):
        src_json = os.path.join(dataset_dir, src_json)

    with open(src_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sample in data:
        for msg in sample.get("messages", []):
            if msg.get("role") == "user":
                msg["content"] = GENERAL_QUESTION_PROMPT + "\n\n" + msg["content"]

    src_dir, src_name = os.path.split(src_json)
    injected_json = os.path.join(src_dir, f"_prompt_injected_{src_name}")
    with open(injected_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    new_entry = dict(entry)
    new_entry["file_name"] = injected_json
    new_info = dict(info)
    new_info[dataset_name] = new_entry

    tmp_info_dir = os.path.join(dataset_dir, "_prompt_injected")
    os.makedirs(tmp_info_dir, exist_ok=True)
    tmp_info_path = os.path.join(tmp_info_dir, "dataset_info.json")
    with open(tmp_info_path, "w", encoding="utf-8") as f:
        json.dump(new_info, f, ensure_ascii=False)

    _set_arg("--dataset_dir", tmp_info_dir)

    print(f"[pa_sft_train3] Injected GENERAL_QUESTION_PROMPT (answer-last) into {len(data)} samples")
    print(f"[pa_sft_train3] Injected dataset JSON: {injected_json}")
    print(f"[pa_sft_train3] Rewired --dataset_dir to: {tmp_info_dir}")
    return injected_json, tmp_info_path


def main():
    inject_prompt()
    run_exp()


if __name__ == "__main__":
    main()
