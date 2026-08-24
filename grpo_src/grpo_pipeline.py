import os

from datasets import load_dataset
from PIL import Image
from trl import GRPOConfig, GRPOTrainer

from reward_c_qwen_answer_last import (
    accuracy_reward_c_qwen_answer_last,
    consistency_reward_c_qwen_answer_last,
)

GENERAL_QUESTION_PROMPT = (
    'You are an expert in detecting industrial anomalies in images. '
    'You will be provided with two images, a reference image (ref_img) as a normal standard and a query image (query_img) for inspection. '
    'Your reasoning and response must strictly follow these constraints based on the specific tags. '
    '\n1. Analyze only the ref_img and define the characteristics of a normal state such as structural integrity, surface texture, and component completeness. '
    '2. Inspect the query_img for any anomalies by comparing it against the ref_img baseline. '
    '\nIf you find anomalies in the query image, respond with <type>...</type><location>...</location><answer>Yes</answer>'
    '\nIf no anomalies are detected in the query image, respond with <answer>No</answer> '
)

REWARD_FUNCS = [accuracy_reward_c_qwen_answer_last, consistency_reward_c_qwen_answer_last]

GEN_DROP = ('mm_token_type_ids',)


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


def make_example(ex):
    prompt = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': GENERAL_QUESTION_PROMPT + '\nThe first image is a normal reference sample. '},
            {'type': 'image'},
            {'type': 'text', 'text': '\n'},
            {'type': 'image'},
            {'type': 'text', 'text': '\nAre there any defects in the query image?'},
        ],
    }]
    return {'prompt': prompt, 'image_paths': ex['image'], 'solution': ex['solution']}


def build_dataset(data_json, img_resolution=512, max_samples=None):
    raw = load_dataset('json', data_files=data_json, split='train')
    if max_samples:
        raw = raw.select(range(min(max_samples, len(raw))))
    dataset = raw.map(make_example, remove_columns=raw.column_names)

    def attach_images(batch):
        batch['images'] = [[load_resized(p, img_resolution) for p in paths]
                           for paths in batch['image_paths']]
        return batch

    return dataset.with_transform(attach_images)


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


def wrap_generate(module):
    if module is None or getattr(module, '_uns_gen_wrapped', False):
        return False
    orig = module.generate

    def generate(*args, **kwargs):
        for key in GEN_DROP:
            kwargs.pop(key, None)
        return orig(*args, **kwargs)

    module.generate = generate
    module._uns_gen_wrapped = True
    return True


def build_trainer(model, tokenizer, train_dataset, output_dir,
                  learning_rate=1e-5, num_generations=4, max_steps=704, save_steps=704,
                  max_prompt_len=8192, max_completion_len=640, beta=0.01):
    training_args = GRPOConfig(
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        optim='adamw_8bit',
        logging_steps=1,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        temperature=0.9,
        top_p=0.9,
        top_k=50,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        max_steps=max_steps,
        save_steps=save_steps,
        max_grad_norm=1.0,
        report_to='none',
        remove_unused_columns=False,
        output_dir=output_dir,
        loss_type='grpo',
        beta=beta,
    )
    training_args.generation_batch_size = None

    patch_warnings_issued(model)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        reward_funcs=REWARD_FUNCS,
        train_dataset=train_dataset,
    )

    args = trainer.args
    if args.steps_per_generation is None:
        args.steps_per_generation = args.gradient_accumulation_steps
    if args.generation_batch_size is None:
        args.generation_batch_size = (args.per_device_train_batch_size
                                      * trainer.accelerator.num_processes
                                      * args.steps_per_generation)
    assert args.generation_batch_size % args.num_generations == 0, (
        args.generation_batch_size, args.num_generations)

    for candidate in (getattr(trainer, 'model', None),
                      getattr(trainer, 'model_wrapped', None),
                      model):
        wrap_generate(candidate)

    return trainer
