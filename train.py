#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import copy
import gc
import hashlib
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import (
    LoraLoaderMixin,
    text_encoder_lora_state_dict,
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from model import VisionTransformer
import copy


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.19.0.dev0")

logger = get_logger(__name__)

def save_progress(learned_embeds, accelerator, args, save_path):
    print("Saving embeddings")
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)

def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    train_text_encoder=False,
    prompt=str,
    repo_folder=None,
    pipeline: DiffusionPipeline = None,
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
instance_prompt: {prompt}
tags:
- {'stable-diffusion' if isinstance(pipeline, StableDiffusionPipeline) else 'if'}
- {'stable-diffusion-diffusers' if isinstance(pipeline, StableDiffusionPipeline) else 'if-diffusers'}
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA DreamBooth - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_image_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the source images.",
    )
    parser.add_argument(
        "--instance_mask_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the foreground masks of intended concepts.",
    )
    parser.add_argument(
        "--instance_name",
        type=str,
        default=None,
        required=True,
        help="Specify the source image.",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default=None,
        required=True,
        help="Specify the class of the intended concept.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=("The resolution for input images"),
    )
    parser.add_argument(
        "--cond_resolution",
        type=int,
        default=224,
        help=("The resolution for condition images"),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        default=True,
        help="Whether to train the text encoder. If true, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--s1_train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader (stage I)."
    )
    parser.add_argument(
        "--s2_train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader (stage II)."
    )
    parser.add_argument(
        "--s1_train_steps", type=int, default=50, help="Stage I iterations."
    )
    parser.add_argument(
        "--s2_train_steps", type=int, default=50, help="Stage II iterations."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--s1_learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for the inversion stage (stage I).",
    )
    parser.add_argument(
        "--s2_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate for the finetuning stage (stage II).",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--class_labels_conditioning",
        required=False,
        default=None,
        help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="_*_",
        help=("The placeholder token representing the intended concept's identity."),
    )
    parser.add_argument(
        "--bg_weight",
        type=float,
        default=1.0,
        help=("The weight of the Background Loss."),
    )
    parser.add_argument(
        "--sm_weight",
        type=float,
        default=1.0,
        help=("The weight of the Semantic Loss."),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    # Set max_train_steps
    args.max_train_steps = args.s1_train_steps + args.s2_train_steps

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class SingleInsertDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for SingleInsert pipeline.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_image_root,
        instance_mask_root,
        instance_name,
        class_name,
        placeholder_token,
        tokenizer,
        train_batch_size,
        size=512,
        cond_size=224,
        tokenizer_max_length=None,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length

        self.device = self.tokenizer

        self.instance_image_path = os.path.join(instance_image_root, instance_name)
        if not Path(self.instance_image_path).exists():
            raise ValueError("Instance image root doesn't exists.")
        self.instance_mask_path = os.path.join(instance_mask_root, instance_name)

        self.instance_name = instance_name
        self.instance_prompt = ' '.join([placeholder_token, class_name])
        self._length = train_batch_size

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        self.cond_transforms = transforms.Compose(
            [
                transforms.Resize(cond_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.mask_transforms = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(), 
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_image_path)
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_image"] = self.image_transforms(instance_image)

        example["cond_image"] = self.cond_transforms(instance_image)

        instance_mask = Image.open(self.instance_mask_path)
        example["instance_mask"] = self.mask_transforms(instance_mask)

        text_inputs = tokenize_prompt(
            self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
        )
        example["instance_prompt_ids"] = text_inputs.input_ids[0]
        example["instance_attention_mask"] = text_inputs.attention_mask[0]


        return example


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    r"""
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (sayakpaul): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    try:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
    except OSError:
        # IF does not have a VAE so let's just set it to None
        # We don't have to error out here
        vae = None

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Copy UNet to calculate the background loss and semantic loss in the finetuning stage
    unet_ori = copy.deepcopy(unet)
    
    # Image-to-text encoder
    I2TNet = VisionTransformer(
            input_resolution=224,
            patch_size=32,
            width=768,
            layers=12,
            heads=12,
            output_dim=768 # v1-5
            # output_dim=1024 # v2-1
        )

    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # We only train the additional adapter LoRA layers and I2TNet
    if vae is not None:
        vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    I2TNet.requires_grad_(True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and the original unet to device and cast to weight_dtype
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    unet_ori.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device)
    text_encoder.to(accelerator.device)
    I2TNet.to(accelerator.device)
    device = accelerator.device

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x up blocks) = 18
    # => 32 layers

    # Set correct lora layers
    unet_lora_attn_procs = {}
    unet_lora_parameters = []
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = (
                LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
            )

        module = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=args.rank
        )
        unet_lora_attn_procs[name] = module
        unet_lora_parameters.extend(module.parameters())

    unet.set_attn_processor(unet_lora_attn_procs)

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_parameters = LoraLoaderMixin._modify_text_encoder(text_encoder, dtype=torch.float32, rank=args.rank)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = unet_attn_processors_state_dict(model)
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                    text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                text_encoder_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)
        LoraLoaderMixin.load_lora_into_text_encoder(
            lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_
        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    '''
    params_to_optimize = (
        itertools.chain(unet_lora_parameters, text_lora_parameters, I2TNet.parameters())
        if args.train_text_encoder
        else itertools.chain(unet_lora_parameters, I2TNet.parameters())
    )
    '''
    params_to_optimize = I2TNet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.s1_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    # Stage I
    train_dataset_s1 = SingleInsertDataset(
        instance_image_root=args.instance_image_dir,
        instance_mask_root=args.instance_mask_dir,
        instance_name=args.instance_name,
        class_name=args.class_name,
        placeholder_token=args.placeholder_token,
        tokenizer=tokenizer,
        train_batch_size=args.s1_train_batch_size,
        size=args.resolution,
        cond_size=args.cond_resolution,
        tokenizer_max_length=args.tokenizer_max_length,
    )

    train_dataloader_s1 = torch.utils.data.DataLoader(
        train_dataset_s1,
        batch_size=args.s1_train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    # Stage II
    train_dataset_s2 = SingleInsertDataset(
        instance_image_root=args.instance_image_dir,
        instance_mask_root=args.instance_mask_dir,
        instance_name=args.instance_name,
        class_name=args.class_name,
        placeholder_token=args.placeholder_token,
        tokenizer=tokenizer,
        train_batch_size=args.s2_train_batch_size,
        size=args.resolution,
        cond_size=args.cond_resolution,
        tokenizer_max_length=args.tokenizer_max_length,
    )

    train_dataloader_s2 = torch.utils.data.DataLoader(
        train_dataset_s2,
        batch_size=args.s2_train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    # Stage I initialization
    train_dataloader = train_dataloader_s1

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, unet_ori, text_encoder, I2TNet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, unet_ori, text_encoder, I2TNet, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, unet_ori, optimizer, I2TNet, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, unet_ori, optimizer, I2TNet, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.s1_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {1}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader_s1)}")
    logger.info(f"  S1 epochs = {args.s1_train_steps}")
    logger.info(f"  S2 epochs = {args.s2_train_steps}")
    logger.info(f"  S1 batch size per device = {args.s1_train_batch_size}")
    logger.info(f"  S2 batch size per device = {args.s2_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    class_prompt = args.class_name
    uncond_tokens = tokenizer(
                class_prompt,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0] # [77]
    uncond_tokens = uncond_tokens.unsqueeze(0).repeat(args.s1_train_batch_size, 1).to(accelerator.device)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        # Stage I (Inversion Stage)
        if global_step == args.s1_train_steps:
            params_to_optimize = (
                itertools.chain(unet_lora_parameters, text_lora_parameters, I2TNet.parameters())
                if args.train_text_encoder
                else itertools.chain(unet_lora_parameters, I2TNet.parameters())
            )
            del optimizer
            optimizer = optimizer_class(
                params_to_optimize,
                lr=args.s2_learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )
            del lr_scheduler
            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.max_train_steps * accelerator.num_processes,
                num_cycles=args.lr_num_cycles,
                power=args.lr_power,
            )
            optimizer, lr_scheduler = accelerator.prepare(
                optimizer, lr_scheduler
            )
            train_dataloader = train_dataloader_s2

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["instance_image"].to(unet.device).to(dtype=weight_dtype)

                if vae is not None:
                    # Convert images to latent space
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                else:
                    model_input = pixel_values

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Initialise the newly added placeholder token with the embeddings of the initializer token
                token_embeds = text_encoder.get_input_embeddings().weight.data

                ###
                cond_img = batch["cond_image"].to(device)
                cond_embeds = I2TNet(cond_img)

                # Get initial text embeddings
                inputs_embeds = F.embedding(batch["instance_prompt_ids"].to(device), token_embeds)
                # Replace the id embeddings with predicted ones
                inputs_embeds[:, 1] = cond_embeds # "{} <class>"

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Get the text embedding for conditioning
                '''
                encoder_hidden_states = encode_prompt(
                    text_encoder,
                    batch["input_ids"],
                    batch["attention_mask"],
                    text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                )
                '''
                encoder_hidden_states = text_encoder(batch["instance_prompt_ids"].to(device), inputs_embeds=inputs_embeds)[0] 
                # Corresponding embeddings for prompt: "<class>"
                uncond_hidden_states = text_encoder(uncond_tokens)[0].to(dtype=weight_dtype) # Unconditional embeddings

                if accelerator.unwrap_model(unet).config.in_channels == channels * 2:
                    noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                if args.class_labels_conditioning == "timesteps":
                    class_labels = timesteps
                else:
                    class_labels = None
                
                tt = torch.cat([timesteps] * 2)
                if global_step < args.s1_train_steps:
                    # Predict the noise residual
                    model_pred = unet_ori(
                        torch.cat([noisy_model_input] * 2), tt, torch.cat([encoder_hidden_states, uncond_hidden_states[:bsz]])
                    ).sample
                    model_pred, model_pred_uncond = model_pred.chunk(2)

                    model_pred_uncond_ori = model_pred_uncond
                else: 
                    # Predict the noise residual
                    model_pred = unet(
                        torch.cat([noisy_model_input] * 2), tt, torch.cat([encoder_hidden_states, uncond_hidden_states[:bsz]])
                    ).sample
                    model_pred, model_pred_uncond = model_pred.chunk(2)

                    with torch.no_grad():
                        model_pred_uncond_ori = unet_ori(
                            noisy_model_input, timesteps, uncond_hidden_states[:bsz], class_labels=class_labels
                        ).sample

                # if model predicts variance, throw away the prediction. we will only train on the
                # simplified training objective. This means that all schedulers using the fine tuned
                # model must be configured to use one of the fixed variance variance types.
                if model_pred.shape[1] == 6:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)
                    model_pred_uncond, _ = torch.chunk(model_pred_uncond, 2, dim=1)
                    model_pred_uncond_ori, _ = torch.chunk(model_pred_uncond_ori, 2, dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Get x0 from noise predictions 
                def extract_into_tensor(a, t, x_shape):
                    b, *_ = t.shape
                    out = a.gather(-1, t)
                    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
                def Getx0FromPred(xt, t, pred, scheduler):
                    sqrt_alphas_cumprod = torch.sqrt(scheduler.alphas_cumprod).to(xt.device)
                    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - scheduler.alphas_cumprod).to(xt.device)
                    w1 = extract_into_tensor(sqrt_alphas_cumprod, t, xt.shape)
                    w2 = extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, xt.shape) 
                    x0 = (xt - w2 * pred) / w1
                    return x0
                
                # Get x0
                x0_from_noisy = Getx0FromPred(noisy_model_input, timesteps, model_pred, noise_scheduler)
                x0_from_noisy_uncond = Getx0FromPred(noisy_model_input, timesteps, model_pred_uncond, noise_scheduler)
                x0_from_noisy_uncond_ori = Getx0FromPred(noisy_model_input, timesteps, model_pred_uncond_ori, noise_scheduler)
                
                masks = batch["instance_mask"].to(device).mean(dim=1, keepdims=True)
                # Compute the Foreground Loss
                loss_fg = (F.mse_loss(x0_from_noisy.float(), model_input.float(), reduction="none") * masks).mean()
                # Compute the Background Loss
                loss_bg = (F.mse_loss(x0_from_noisy.float(), x0_from_noisy_uncond_ori.float(), reduction='none') * (1. - masks)).mean()
                if global_step < args.s1_train_steps:
                    loss = loss_fg + args.bg_weight * loss_bg
                else:
                    # Compute the Semantic Loss
                    loss_sm = F.mse_loss(x0_from_noisy_uncond.float(), x0_from_noisy_uncond_ori.float(), reduction='mean')
                    loss = loss_fg + args.bg_weight * loss_bg + args.sm_weight * loss_sm

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if global_step < args.s1_train_steps:
                        params_to_clip = I2TNet.parameters()
                    else:
                        params_to_clip = (
                            itertools.chain(unet_lora_parameters, text_lora_parameters, I2TNet.parameters())
                            if args.train_text_encoder
                            else itertools.chain(text_lora_parameters, I2TNet.parameters())
                        )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # save embedding
        save_path = os.path.join(args.output_dir, f"learned_embeds.bin")
        save_progress(cond_embeds[0], accelerator, args, save_path)
        # save lora layers
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = unet_attn_processors_state_dict(unet)

        if text_encoder is not None and args.train_text_encoder:
            text_encoder = accelerator.unwrap_model(text_encoder)
            text_encoder = text_encoder.to(torch.float32)
            text_encoder_lora_layers = text_encoder_lora_state_dict(text_encoder)
        else:
            text_encoder_lora_layers = None

        LoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
        )

        # Final inference
        # Load previous pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype
        )

        # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
        scheduler_args = {}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.load_lora_weights(args.output_dir, weight_name="pytorch_lora_weights.safetensors")

        # run inference
        # images = []
        # if args.validation_prompt and args.num_validation_images > 0:
        #     generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
        #     images = [
        #         pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
        #         for _ in range(args.num_validation_images)
        #     ]

        #     for tracker in accelerator.trackers:
        #         if tracker.name == "tensorboard":
        #             np_images = np.stack([np.asarray(img) for img in images])
        #             tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
        #         if tracker.name == "wandb":
        #             tracker.log(
        #                 {
        #                     "test": [
        #                         wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
        #                         for i, image in enumerate(images)
        #                     ]
        #                 }
        #             )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                train_text_encoder=args.train_text_encoder,
                prompt=args.instance_prompt,
                repo_folder=args.output_dir,
                pipeline=pipeline,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    # accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)