from PIL import Image
from lang_sam import LangSAM
from torchvision.utils import save_image
import torch
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Image to mask using LangSAM.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/masks",
    )
    parser.add_argument(
        "--input_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model = LangSAM()
    image_pil = Image.open(os.path.join(args.input_dir, args.input_name)).convert("RGB")
    text_prompt = args.prompt
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)

    mask = masks[:1].float()
    save_image(mask, os.path.join(args.output_dir, args.input_name))

if __name__ == "__main__":
    main()