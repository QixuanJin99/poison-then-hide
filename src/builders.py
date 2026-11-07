from torchvision import models
import torch
import torch.nn as nn

import timm
import copy
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from clip_backbone import ClipVisualWrapper, TimmBackboneWrapper

# Pretrained CIFAR10 ResNet-18 
def base_builder_cifar10_resnet18():
    return timm.create_model("resnet18_cifar10", pretrained=True, num_classes=10)

# Pretrained ResNet50 for ImageNet base 
def base_builder_imagenet_resnet50():
    return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Pretrained CLIP + ImageNet fc base 
def base_builder_timm_clip_imnet(device, *, return_head: bool = True, return_transforms: bool = True):
    name = "vit_base_patch16_clip_224.openai_ft_in12k_in1k"

    full = timm.create_model(name, pretrained=True).to(device).eval()
    imagenet_head = None
    if return_head:
        clf = full.get_classifier()
        assert isinstance(clf, nn.Linear)
        imagenet_head = copy.deepcopy(clf).to(device)   # <- deepcopy here

    feat = timm.create_model(name, pretrained=True, num_classes=0).to(device).eval()
    backbone = TimmBackboneWrapper(feat)

    cfg = resolve_model_data_config(full)  # <- authoritative
    tf_train = tf_val = None
    if return_transforms:
        tf_train = create_transform(**cfg, is_training=True)
        tf_val   = create_transform(**cfg, is_training=False)

    # return cfg so downstream can read cfg['mean'], cfg['std'], cfg['input_size']...
    return backbone, imagenet_head, tf_train, tf_val, cfg


# Pretrained ImageNet ViT 
def base_builder_timm_vit_imnet(
    device,
    *,
    name: str = "vit_base_patch16_224.augreg_in21k_ft_in1k",
    return_head: bool = True,
    return_transforms: bool = True,
):
    """
    Non-CLIP ViT for ImageNet: returns (backbone, imagenet_head, tf_train, tf_val, cfg)
    - backbone: features-only wrapper (num_classes=0) -> [B, D]
    - imagenet_head: nn.Linear(D,1000) copied from the pretrained classifier
    - cfg: use cfg['mean'], cfg['std'], cfg['input_size'] for all pipelines
    """
    # 1) Load full classifier model (to copy its trained head)
    full = timm.create_model(name, pretrained=True).to(device).eval()

    imagenet_head = None
    if return_head:
        clf = full.get_classifier()
        assert isinstance(clf, nn.Linear), f"Unexpected classifier type for {name}: {type(clf)}"
        imagenet_head = copy.deepcopy(clf).to(device)

    # 2) Features-only variant (num_classes=0)
    feat = timm.create_model(name, pretrained=True, num_classes=0).to(device).eval()
    backbone = TimmBackboneWrapper(feat)  # your wrapper (unchanged)

    # 3) Transforms & data config
    cfg = resolve_model_data_config(full)
    tf_train = tf_val = None
    if return_transforms:
        tf_train = create_transform(**cfg, is_training=True)
        tf_val   = create_transform(**cfg, is_training=False)

    return backbone, imagenet_head, tf_train, tf_val, cfg
