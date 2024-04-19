# SingleInsert: Inserting New Concepts from a Single Image to Text-to-Image Models for Flexible Editing

[**Project**](https://jarrentwu1031.github.io/SingleInsert-web/) | [**Arxiv**](https://arxiv.org/abs/2310.08094) 

If you find our work useful for your research, please star this repo and cite our paper. Thanks!
```
@article{wu2023singleinsert,
    author = {Wu, Zijie and Yu, Chaohui and Zhu, Zhen and Wang, Fan and Bai, Xiang.},
    title  = {SingleInsert: Inserting New Concepts from a Single Image into Text-to-Image Models for Flexible Editing},
    journal = {arxiv:2310.08094},
    year   = {2023},
```

<div align=center>
    <img src="https://github.com/JarrentWu1031/SingleInsert/blob/main/teaser.png" width=85%>
</div>
  
### Installation

```
conda create -n singleinsert python=3.9
conda activate singleinsert

pip install -r requirements.txt

# Install LangSAM for foreground segmentation
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git

# Please replace the modeling_clip.py in transformers with the provided one 
cp modeling_clip.py /YOUR/PATH/TO/ENV/singleinsert/lib/python3.9/site-packages/transformers/models/clip/
```

### Mask Preparation

Use command like below for forground mask preparation:
```
python img2mask.py --input_dir ./data/images --output_dir ./data/masks --input_name 066.jpg --prompt face
```
Please specify the class name of the intended foreground concept correctly.

### Train

Use command like below for training:
```
python train.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --instance_image_dir ./data/images --instance_mask_dir ./data/masks --instance_name 066.jpg --class_name "face"
```
We set the training iterations as 50 for stage 1 and stage 2 by default. For non facial instances, the training iterations could be more or less for better quality.

### Test

Use command like below for inference:
```
python test_lora_emb.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --prompt "A man with red hair, _*_ face"
```

### Acknowledgments

The code is based on project [Dreambooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth). We sincerely thank them for their great work!
