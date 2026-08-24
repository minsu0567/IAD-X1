# IAD-X1

Unified Model for Industrial Anomaly Detection — Qwen3.5-4B.

Given a **reference** image (known good) and a **query** image, the model decides whether the
query is defective and, if so, reports the defect type and location.

```
defective : <type>scratch</type><location>top</location><answer>Yes</answer>
normal    : <answer>No</answer>
```

Tag order is fixed to **answer-last** (`type → location → answer`). The prompt, the reward
functions and the datasets all follow that order.

## IAD-X1 Model

| Stage  | Hugging Face |
|---|---|
| Base | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| SFT | [minsu0567/IAD-X1-SFT-answer-last](https://huggingface.co/minsu0567/IAD-X1-SFT-answer-last) |
| SFT + GRPO | [minsu0567/IAD-X1-GRPO-answer-last-no-hard](https://huggingface.co/minsu0567/IAD-X1-GRPO-answer-last-no-hard) |

A DPO stage (notebooks 3–4) also starts from the SFT model, training on preference pairs
collected from its own wrong answers.

## Benchmark dataset

Evaluation (notebooks 5–6) runs on
**[yanhui01/Industrial_test](https://huggingface.co/datasets/yanhui01/Industrial_test)** —
VisA, SDD, MPDD, DTD, DS-MVTec and DAGM.

The dataset is **gated**: open the page above, submit an access request and wait for the
author's approval before downloading. Unpack it to `MyDrive/Uni-IAD_eval_dataset/`, the
path the evaluation notebooks read.

## Layout

```
IAD-X1/
├── sft_src/
│   └── pa_sft_train3.py                        answer-last prompt injection -> LlamaFactory
├── grpo_src/
│   ├── reward_c_qwen_answer_last.py            accuracy + consistency rewards
│   └── grpo_pipeline.py                        dataset + trainer assembly
├── dpo_src/
│   ├── build_dpo_dataset_qwen3_5_answer_last.py   collects wrong answers as preference pairs
│   └── dpo_pipeline.py                         Qwen-VL patches for trl 0.24
├── eval_src/
│   ├── eval_benchmark_qwen3_5.py               per-benchmark accuracy
│   ├── e2e_latency_qwen3_5.py                  TTFT / decode throughput / e2e latency
│   └── build_throughput_manifest.py            shared 180-sample benchmark manifest
└── notebooks/                                  Colab drivers: install, model load, run
```

## Google Drive

The notebooks run on Colab and read everything from Drive. Copy this repository to
`MyDrive/IAD-X1/`; these siblings must exist:

```
MyDrive/
├── IAD-X1/                                  this repository
├── IAD-R1-main/                             reward_process, helper/summary.py, data/dataset_info.json
├── PA-SFT_dataset_2/                        SFT images
├── GRPO_dataset3/                           GRPO / DPO images and JSON
├── Uni-IAD_eval_dataset/                    VisA, SDD, MPDD, DTD, DS-MVTec, DAGM
├── merged_reordered_answer_last.json        SFT training set
└── grpo_no_hard_samples_answer_last.json    GRPO training set
```

## Notebooks

| # | Notebook | Purpose |
|---|---|---|
| 1 | `SFT/run_pa_sft_qwen3_5_4b.ipynb` | full fine-tune via pip-installed LlamaFactory |
| 2 | `GRPO/run_grpo_qwen3_5_4b_answer_last.ipynb` | SC-GRPO with C-format rewards |
| 3 | `DPO/build_dpo_dataset_vllm_qwen3_5_4b_answer_last.ipynb` | build the DPO preference set |
| 4 | `DPO/run_dpo_qwen3_5_4b_answer_last.ipynb` | DPO on those pairs |
| 5 | `Evaluation/run_eval_qwen3_5.ipynb` | accuracy on one benchmark |
| 6 | `Evaluation/run_demo_vllm_qwen3_5_4b.ipynb` | TTFT / throughput / latency |

## Environments

Training and evaluation use incompatible stacks — run each in a fresh Colab runtime.

| | Training (1–4) | Evaluation (5–6) |
|---|---|---|
| stack | unsloth + trl 0.24 + transformers 5.2 | vllm 0.17 + transformers 5.3 |

Every notebook installs its own stack in its first cell and requires a runtime restart
afterwards. A100 40/80GB is recommended.

## HuggingFace token

The token is read from Colab Secrets, never stored in the source:

```python
HF_TOKEN = userdata.get('HF_TOKEN')
```

Add `HF_TOKEN` in the Colab sidebar (🔑) and enable notebook access for it.

## Reference

This project builds on **[IAD-R1](https://github.com/Yanhui-Lee/IAD-R1)**:

- `grpo_src/` calls its `reward_process` for type / location scoring
- evaluation reuses its `helper/summary.py` accuracy protocol
- SFT reads its `data/dataset_info.json`

`IAD-R1-main/` in the Drive layout above is a clone of that repository.
