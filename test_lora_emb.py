### Test finetune

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from einops import rearrange
import clip
import torchvision.transforms as T

import shutil
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from typing import Optional, Union, Mapping
from torch import Tensor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

def add_tokens_to_model(learned_embeds: Mapping[str, Tensor], text_encoder: CLIPTextModel, 
        tokenizer: CLIPTokenizer, override_token: Optional[Union[str, dict]] = None) -> None:
    r"""Adds tokens to the tokenizer and text encoder of a model."""
    
    # Loop over learned embeddings
    new_tokens = []
    for token, embedding in learned_embeds.items():
        embedding = embedding.to(text_encoder.get_input_embeddings().weight.dtype)
        if override_token is not None:
            token = override_token if isinstance(override_token, str) else override_token[token]
        
        # Add the token to the tokenizer
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError((f"The tokenizer already contains the token {token}. Please pass a "
                               "different `token` that is not already in the tokenizer."))
  
        # Resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))
  
        # Get the id for the token and assign the embeds
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight.data[token_id] = embedding    
        new_tokens.append(token)

    print(f'Added {len(new_tokens)} tokens to tokenizer and text embedding: {new_tokens}')

def add_tokens_to_model_from_path(learned_embeds_path: str, text_encoder: CLIPTextModel, 
        tokenizer: CLIPTokenizer, override_token: Optional[Union[str, dict]] = None) -> None:
    r"""Loads tokens from a file and adds them to the tokenizer and text encoder of a model."""
    learned_embeds: Mapping[str, Tensor] = torch.load(learned_embeds_path, map_location='cpu')
    add_tokens_to_model(learned_embeds, text_encoder, tokenizer, override_token)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='runwayml/stable-diffusion-v1-5',
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default='_*_ face',
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--ncol",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--bsz",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default='output',
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default='test',
    )
    parser.add_argument(
        "--tte",
        action="store_true",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    device = 'cuda'
    dtype = torch.float16

    ### v1-5
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    add_tokens_to_model_from_path(
            args.exp_dir + '/learned_embeds.bin', text_encoder, tokenizer
        )

    lora_path = args.exp_dir
    
    args.tte = True
    if args.tte: # train_text_encoder
        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(lora_path)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet)
        LoraLoaderMixin.load_lora_into_text_encoder(
            lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder
        )
    else:
        unet.load_attn_procs(lora_path)

    vae.to(device, dtype=dtype)
    text_encoder.to(device, dtype=dtype)
    unet = unet.to(device, dtype=dtype)

    with torch.no_grad():
        prompt = [args.prompt]
        generator = torch.manual_seed(1000)
        bsz = args.bsz

        text_input = tokenizer(prompt, padding="max_length", return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        text_embeddings_ori = text_embeddings
    
        num_inference_steps = 50
        noise_scheduler.set_timesteps(num_inference_steps)
        
        txt = ''
        uncond_input = tokenizer(txt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings.repeat(bsz, 1, 1), text_embeddings_ori.repeat(bsz, 1, 1)])
        guidance_scale = 7.5
        noisy_latent = torch.randn([bsz, 4, 64, 64]).to(dtype).to(device)

        latent_n = noisy_latent
        
        for t in tqdm(noise_scheduler.timesteps):
            noisy_latent = torch.cat([latent_n] * 2)
            noise_pred = unet(noisy_latent, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latent_n = noise_scheduler.step(noise_pred, t, latent_n).prev_sample
        
        x0_from_noise = latent_n
        x0_from_noise = x0_from_noise.to(dtype)

        x0_from_noise = 1 / 0.18215 * x0_from_noise
        with torch.no_grad():
            x0_from_noise = vae.decode(x0_from_noise).sample
        x0_from_noise = torch.clamp((x0_from_noise + 1.0) / 2.0, min=0.0, max=1.0)

    grid = x0_from_noise
    for i in range(bsz):
        img = 255. * grid[i].permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img.astype(np.uint8))
    grid = make_grid(grid, nrow=args.nrow)
    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    img = Image.fromarray(grid.astype(np.uint8))
    img.save(os.path.join(args.out_dir, 'output.jpg'))

if __name__ == "__main__":
    main()