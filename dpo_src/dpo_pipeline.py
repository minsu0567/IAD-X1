import json
import os

import torch
from datasets import Dataset
from PIL import Image
from tqdm.auto import tqdm
from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import DataCollatorForPreference

GENERAL_QUESTION_PROMPT = (
    'You are an expert in detecting industrial anomalies in images. '
    'You will be provided with two images, a reference image (ref_img) as a normal standard and a query image (query_img) for inspection. '
    'Your reasoning and response must strictly follow these constraints based on the specific tags. '
    '\n1. Analyze only the ref_img and define the characteristics of a normal state such as structural integrity, surface texture, and component completeness. '
    '2. Inspect the query_img for any anomalies by comparing it against the ref_img baseline. '
    '\nIf you find anomalies in the query image, respond with <type>...</type><location>...</location><answer>Yes</answer>'
    '\nIf no anomalies are detected in the query image, respond with <answer>No</answer> '
)

EXPECTED_PROBLEM = ('The first image is a normal reference sample. <image>\n<image>\n'
                    'Are there any defects in the query image?')

PROMPT_PARTS = [
    {'type': 'text', 'text': GENERAL_QUESTION_PROMPT + '\nThe first image is a normal reference sample. '},
    {'type': 'image'},
    {'type': 'text', 'text': '\n'},
    {'type': 'image'},
    {'type': 'text', 'text': '\nAre there any defects in the query image?'},
]

VISION_EXTRA_KEYS = ('image_grid_thw',)

_VISION_SLOT = {}


def strip_think_primer(tokenizer):
    if getattr(tokenizer, '_no_think_wrapped', False):
        return tokenizer
    orig = tokenizer.apply_chat_template

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)

        def strip(x):
            if isinstance(x, str):
                t = x.rstrip()
                if t.endswith('<think>'):
                    return t[: -len('<think>')]
            return x

        if isinstance(out, str):
            return strip(out)
        if isinstance(out, list) and out and isinstance(out[0], str):
            return [strip(x) for x in out]
        return out

    tokenizer.apply_chat_template = wrapped
    tokenizer._no_think_wrapped = True
    return tokenizer


def load_resized(path, target=512):
    local = path.replace('/content/drive/MyDrive/', '/content/')
    if os.path.exists(local):
        path = local
    img = Image.open(path).convert('RGB')
    if target and max(img.size) > target:
        s = target / max(img.size)
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.LANCZOS)
    return img


def build_dataset(dpo_json, img_resolution=512, max_samples=None):
    with open(dpo_json, encoding='utf-8') as f:
        raw = json.load(f)
    if max_samples:
        raw = raw[:max_samples]

    mismatch = [i for i, s in enumerate(raw) if s['messages'][0]['content'] != EXPECTED_PROBLEM]
    assert not mismatch, f'unexpected problem string at indices {mismatch[:5]}'

    records = []
    for s in tqdm(raw, desc='loading images'):
        records.append({
            'images': [load_resized(p, img_resolution) for p in s['images']],
            'prompt': [{'role': 'user', 'content': PROMPT_PARTS}],
            'chosen': [{'role': 'assistant', 'content': s['chosen']['content']}],
            'rejected': [{'role': 'assistant', 'content': s['rejected']['content']}],
        })
    return Dataset.from_list(records)


class QwenVLPreferenceCollator(DataCollatorForPreference):

    _DROP_FOR_BASE = ('images', 'prompt', 'pixel_values')

    def __init__(self, pad_token_id, processor):
        super().__init__(pad_token_id=pad_token_id)
        self.processor = processor

    def torch_call(self, examples):
        output = super().torch_call([
            {k: v for k, v in ex.items() if k not in self._DROP_FOR_BASE} for ex in examples
        ])

        pixel_values = []
        extras = {k: [] for k in VISION_EXTRA_KEYS}
        for ex in examples:
            proc = self.processor(images=ex['images'], text=ex['prompt'], add_special_tokens=False)
            pixel_values.append(torch.as_tensor(proc['pixel_values']))
            for k in VISION_EXTRA_KEYS:
                if k in proc:
                    extras[k].append(torch.as_tensor(proc[k]))

        output['pixel_values'] = torch.cat(pixel_values, dim=0)
        for k, vals in extras.items():
            if vals:
                output[k] = torch.cat(vals, dim=0)
        return output


class QwenVLDPOTrainer(DPOTrainer):

    @staticmethod
    def process_row(features, processing_class, max_prompt_length=None,
                    max_completion_length=None, add_special_tokens=True):
        processor, tokenizer = processing_class, processing_class.tokenizer
        processed = processor(images=features['images'], text=features['prompt'],
                              add_special_tokens=False)

        prompt_input_ids = processed['input_ids'][0]
        chosen_input_ids = tokenizer(features['chosen'], add_special_tokens=False)['input_ids']
        rejected_input_ids = tokenizer(features['rejected'], add_special_tokens=False)['input_ids']

        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        if max_prompt_length is not None and len(prompt_input_ids) > max_prompt_length:
            raise ValueError(
                f'prompt exceeds max_prompt_length ({max_prompt_length}): '
                f'{len(prompt_input_ids)} tokens. Lower IMG_RESOLUTION or raise MAX_PROMPT_LEN.'
            )
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        return {
            'prompt_input_ids': prompt_input_ids,
            'chosen_input_ids': chosen_input_ids,
            'rejected_input_ids': rejected_input_ids,
        }

    def concatenated_inputs(self, batch, padding_value):
        output = super().concatenated_inputs(batch, padding_value)
        for k in VISION_EXTRA_KEYS:
            if k in batch:
                output[k] = torch.cat([batch[k], batch[k]], dim=0)
        _VISION_SLOT.clear()
        _VISION_SLOT.update({k: output[k] for k in VISION_EXTRA_KEYS if k in output})
        return output


def attach_vision_kwargs(module):
    if module is None or getattr(module, '_qwen_dpo_vision_wrapped', False):
        return False
    orig = module.forward

    def forward(*args, **kwargs):
        for k, v in _VISION_SLOT.items():
            kwargs.setdefault(k, v)
        return orig(*args, **kwargs)

    module.forward = forward
    module._qwen_dpo_vision_wrapped = True
    return True


def patch_warnings_issued(model):
    targets = [model, getattr(model, 'base_model', None)]
    base = getattr(model, 'base_model', None)
    if base is not None:
        targets.append(getattr(base, 'model', None))
    for m in targets:
        if m is not None and not hasattr(m, 'warnings_issued'):
            try:
                m.warnings_issued = {}
            except Exception:
                pass


def build_trainer(model, tokenizer, train_dataset, output_dir,
                  learning_rate=1e-5, beta=0.1, per_device_bs=1, grad_accum=4,
                  max_steps=247, save_steps=247, max_prompt_len=8192,
                  max_completion_len=640, max_len=8832,
                  adam_beta1=0.9, adam_beta2=0.99, weight_decay=0.0, warmup_ratio=0.1,
                  lr_scheduler='cosine', max_grad_norm=1.0, optim='adamw_8bit', seed=42):
    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()

    patch_warnings_issued(model)

    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        learning_rate=learning_rate,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler,
        max_grad_norm=max_grad_norm,
        optim=optim,
        logging_steps=1,
        save_steps=save_steps,
        save_total_limit=2,
        seed=seed,
        bf16=True,
        report_to='none',
        beta=beta,
        max_length=max_len,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        remove_unused_columns=False,
    )

    trainer = QwenVLDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.data_collator = QwenVLPreferenceCollator(
        pad_token_id=trainer.pad_token_id,
        processor=tokenizer,
    )

    for module in (getattr(trainer, 'model', None),
                   getattr(trainer, 'model_wrapped', None),
                   model):
        attach_vision_kwargs(module)

    if 'pixel_values' in trainer.train_dataset.column_names:
        trainer.train_dataset = trainer.train_dataset.remove_columns(['pixel_values'])

    return trainer


def save_log(trainer, hf_repo_out, beta, learning_rate,
             log_dir='/content/drive/MyDrive/DPO_logs'):
    import pandas as pd
    df = pd.DataFrame(trainer.state.log_history)
    variant = hf_repo_out.split('/')[-1]
    path = f'{log_dir}/{variant}__beta{beta}_lr{learning_rate}.csv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path
