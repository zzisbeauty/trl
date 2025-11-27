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

"""
# Full training
```bash
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns
```

# LoRA:
```bash
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```
"""

import argparse
import os
from typing import Optional

import torch
from accelerate import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from trl import (
    DatasetMixtureConfig,
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def main(script_args, training_args, model_args, dataset_args):
    ################
    # Model
    ###################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype) # 数据类型(dtype)配置
    model_kwargs = dict( # 构建模型加载参数
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args) # 量化配置处理， get_quantization_config() 检查是否需要量化
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map() # 当量化配置存在时,还需要设置 device_map
        model_kwargs["quantization_config"] = quantization_config # get_kbit_device_map() 返回当前进程的设备映射,确保量化模型正确分配到 GPU

    model = AutoModelForCausalLM.from_pretrained( # 加载策略模型(Policy Model) 这是被训练的模型,DPO 训练会更新它的权重，最终保存的就是这个模型
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    """
    下方两个选项的工作原理： 根据 trl/trainer/dpo_trainer.py 中 DPO_trainer 的具体实现：
    工作原理: 当使用 LoRA 等 PEFT 方法时,可以通过关闭适配器来将策略模型当作参考模型使用,从而节省内存。
        策略模型 = 基础模型 + LoRA 适配器(开启)
        参考模型 = 基础模型 + LoRA 适配器(关闭)
    这样只需要加载一个基础模型,通过切换适配器状态来模拟两个模型。
    在 DPOTrainer 初始化时,这两个模型会被传入； trl/scripts/dpo.py
    """
    peft_config = get_peft_config(model_args) # PEFT 配置检查与参考模型处理；  参考模型 (Reference Model) - ref_model
    if peft_config is None: # 情况 A：不使用 PEFT (peft_config is None)； 作用: 加载一个独立的参考模型,用于计算 DPO 损失中的基线概率。
        ref_model = AutoModelForCausalLM.from_pretrained( # 为什么需要: DPO 算法需要计算策略模型和参考模型之间的 KL 散度,防止模型偏离太远。参考模型的权重保持冻结,不会被训练更新。
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
        )
    else:
        # 情况 B: 使用 PEFT (peft_config is not None)； 作用: 不加载独立的参考模型,而是利用 PEFT 机制。
        # 工作原理: 当使用 LoRA 等 PEFT 方法时,可以通过关闭适配器来将策略模型当作参考模型使用,从而节省内存。
        ref_model = None

    # DDP 优化设置 (可选)：作用: 这是一个分布式训练优化。 为什么需要: 在使用 PyTorch DDP(Distributed Data Parallel)时,布尔类型的 buffer 可能导致同步问题。通过忽略这些 buffer,可以避免潜在的错误。
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Load the dataset
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning("Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the dataset and `dataset_name` will be ignored.")
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")


    import logging  
    logging.basicConfig(level=logging.INFO)  
    # 或者使用 logger 的设置  
    logger.setLevel(logging.INFO)

    # 添加数据日志  
    if script_args.dataset_train_split in dataset:  
        train_dataset = dataset[script_args.dataset_train_split]  
        logger.info(f"Training dataset size: {len(train_dataset)}")  
        # 打印第一个样本  
        first_example = train_dataset[0]  
        logger.info(f"First example structure: {list(first_example.keys())}")  
        logger.info(f"First example content: {first_example}")  
    else:  
        logger.warning(f"Dataset split '{script_args.dataset_train_split}' not found. Available splits: {list(dataset.keys())}")

    # Initialize the DPO trainer
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,
    )

    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run the DPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    ...
    ...
    main(script_args, training_args, model_args, dataset_args)
