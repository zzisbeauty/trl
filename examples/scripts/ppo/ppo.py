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


""" 这个脚本是一个基础的 PPO 训练示例。 它旨在展示如何使用 PPOTrainer 来微调模型，以提高其生成具有积极情感或物理描述性语言的能力
除了本地的一个 reward model，还有一个 "EleutherAI/pythia-160m", 没有使用验证
"""

import os
import shutil

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


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
 
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation, # none
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args) # 这个方法用于根据 ModelConfig 参数生成量化配置,以便在加载模型时使用 4-bit 或 8-bit 量化来减少显存占用。
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    tokenizer = AutoTokenizer.from_pretrained( # policy model
        # model_args.model_name_or_path, 
        '/workspace/models/Qwen3-4B-Instruct-2507', # model path； 一般是从一个基础模型加载，即 base model； 后面的 plicy model 会从其 sft model 加载
        padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"}) # 保证 tokenizer 中的 pad token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    value_model = AutoModelForSequenceClassification.from_pretrained(
        # training_args.reward_model_path, 
        '/workspace/models/qwen3-0.6B', # 可以小一点
        trust_remote_code=model_args.trust_remote_code, num_labels=1, device_map="auto"
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        # training_args.reward_model_path, 
        '/workspace/models/qwen3-0.6B', # 可以小一点，但是和 value model 彼此独立
        trust_remote_code=model_args.trust_remote_code, num_labels=1, device_map="auto"
    )
    policy = AutoModelForCausalLM.from_pretrained(
        # training_args.sft_model_path, \
        "/workspace/models/qwen2.5-1.5b-instruct",
        # '/workspace/models/Qwen3-4B-Instruct-2507', # model path； 实际加载 policy model 的权重，通常指向一个已经过 SFT（监督微调）的模型
        trust_remote_code=model_args.trust_remote_code, device_map="auto"
    )

    
    # PEFT 设置方式一 创建 peft 参数，不单独加载 reference model
    from peft import LoraConfig 
    peft_config = LoraConfig(  
        r=16,  # LoRA 矩阵的秩（rank），决定了 adapter 的大小
        lora_alpha=32,  
        lora_dropout=0.05,  
        bias="none",  
        task_type="CAUSAL_LM",  
    ) 

    # peft 设置方式二
    # model_args.use_peft = True  
    # model_args.lora_r = 16  
    # model_args.lora_alpha = 32  
    # model_args.lora_dropout = 0.05  
    # peft_config = get_peft_config(model_args)

    if peft_config is None: # None 意味着不使用 PEFT: 需要显式加载参考模型
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code,device_map="auto"
        )
    else: # 即 peft 配置不为空
        ref_policy = None # 使用 PEFT: 不单独加载 reference model

    ################
    # Dataset
    ################

    training_args.per_device_train_batch_size = 1  
    training_args.gradient_accumulation_steps = 2  
    training_args.num_mini_batches = 1  
    training_args.local_rollout_forward_batch_size = 32 

    dataset = load_dataset(
        # script_args.dataset_name, 
        "trl-internal-testing/descriptiveness-sentiment-trl-style", # dataset name
        name=script_args.dataset_config, 
        # 参数决定了加载哪个子集的数据，每个 split 包含不同的训练样本，用于不同的训练目标，在 PPO 训练中，这些样本都只需要 prompt 字段
        split="sentiment",
        # split=script_args.dataset_train_split
    )
    eval_samples = 100 # 将数据集分割为训练集和评估集,最后 100 个样本作为评估集
    train_dataset = dataset.select(range(len(dataset) - eval_samples)) # 选择数据集的前 N-100 个样本作为训练集
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset))) # 选择数据集的最后 100 个样本作为评估集
    # 对于 PPO 训练,数据集只需要 prompt 字段
    # 因为:PPO 训练从 prompt 生成响应，生成的响应由奖励模型评分，不需要预先准备的标准答案。
    # 这与 SFT 或 DPO 训练不同,后者需要 chosen/rejected 或 completion 字段
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer):
        """ pre-tokenize the dataset before training; only collate during training
        """
        def tokenize(element):
            outputs = tokenizer(element[dataset_text_field], padding=False,)
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
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
