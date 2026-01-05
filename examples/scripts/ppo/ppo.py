# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

import os
import shutil

import sys
from pathlib import Path
print(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # 添加项目根目录到 Python 路径
sys.path.insert(0, "/workspace/trl")

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig, ScriptArguments, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.experimental.ppo import PPOConfig, PPOTrainer


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


"""
python -i examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir pythia-1b-deduped-descriptiveness-sentiment-trl-style-ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir pythia-1b-deduped-descriptiveness-sentiment-trl-style-ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0
"""


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    shutil.rmtree(training_args.output_dir, ignore_errors=True) # remove output_dir if exists

    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config




    # 价值模型和奖励模型 - 各自独立
    value_model = AutoModelForSequenceClassification.from_pretrained(training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code, num_labels=1, **model_kwargs,)
    reward_model = AutoModelForSequenceClassification.from_pretrained(training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code, num_labels=1, **model_kwargs,)

    # 策略模型和参考模型 - 基于 peft 方案加载
    policy = AutoModelForCausalLM.from_pretrained(training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs)
    peft_config = get_peft_config(model_args)  #基于 peft-lora 方案 - Ref model: 可以设置为 None，训练器会通过禁用 adapter 来使用一个与 policy model 相同的基础模型作为参考
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs)
    else:
        ref_policy = None

    """ 其中 policy model 和 value moel 会被组合在一起 """

    ################
    # Dataset
    ################

    # # 生成模型务必使用做填充： 生成式模型是根据之前的词预测下一个词。如果 [PAD] 填充在右边，模型在生成时会首先看到一堆无意义的填充符，导致预测混乱。左填充能保证模型看到的序列末尾永远是真实的文本内容。
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"}) # "[PAD]": 这是一个通用的填充占位符。执行这一行后，分词器会将这个新词加入词表，并分配一个唯一的 ID。
    # model.resize_token_embeddings(len(tokenizer)) # # 调整模型的嵌入层大小，以匹配增加了新 token 的分词器  -   genimi 建议的代码，但是项目本身并没有

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split)
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = "prompt"  # 这行代码指定了数据集中要处理的字段名称为 "prompt"
    # PPO 数据集只需要 prompt 字段，因为训练流程是：从 prompt 生成响应 → 奖励模型评分 → 更新策略。这与 DPO 或 SFT 训练不同，后者需要 chosen/rejected 或 completion 字段。

    def prepare_dataset(dataset, tokenizer):
        """ pre-tokenize the dataset before training; only collate during training """

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # 这段代码使用 PartialState().local_main_process_first() 上下文管理器，确保只在主进程上执行数据预处理，其他进程等待主进程完成后直接使用缓存结果
    # 1. 避免重复计算：多个进程不会重复执行相同的 tokenization 操作
    # 2. 加速数据处理：其他进程直接加载主进程处理好的结果
    # 3. 节省内存：不需要在每个进程中都保存原始数据
    # 这些预处理后的数据集会被传递给 PPOTrainer，训练器会使用 DataCollatorWithPadding 在训练时动态进行 padding
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()
