# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import segmentation_models_pytorch as smp
from util.misc import NestedTensor, is_main_process
import matplotlib.pyplot as plt
from .position_encoding import build_position_encoding
from panns_inference import AudioTagging, SoundEventDetection
from panns_inference.models import Cnn14_DecisionLevelMax, Cnn14
from transformers import ASTFeatureExtractor, ASTModel
import torchvggish, os
# import tensorflow_hub as hub
import numpy as np

from torch_vggish_yamnet.input_proc import *
from torch_vggish_yamnet import yamnet
import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

def replace_bn_with_frozen_bn(module):
    """
    递归替换 module 中的所有 nn.BatchNorm2d 为 FrozenBatchNorm2d。
    你可以根据需求来保留或修改。
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            # 用 FrozenBatchNorm2d 的通道数替换
            num_features = child.num_features
            frozen = FrozenBatchNorm2d(num_features)
            setattr(module, name, frozen)
        else:
            replace_bn_with_frozen_bn(child)
    return module

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:  # return_interm_layers = False
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels


    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        # print("is_main_process() is",is_main_process())
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    # for attr in vars(args):
    #     print(f"{attr}: {getattr(args, attr)}")
    # exit()
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0   # train_backbone = 1
    return_interm_layers = args.masks       # return_interm_layers = False
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation) # dilation = False
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


# ✅ 正确的 VGGishBackbone 继承 `nn.Module`
class VGGishBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vggish = torchvggish.vggish()  # ✅ 正确调用 VGGish

    def forward(self, x):
        # print(self.vggish.features)
        x = self.vggish.features[:13](x)  # ✅ 提取 `conv4` (512 通道)
        # print(f"conv4 output shape: {x.shape}")  # ✅ 打印 `conv4` 输出尺寸
        return {"conv4": x}


class LightAudioBackbone(nn.Module):
    def __init__(self, output_shape=(512, 15, 20)):
        super().__init__()
        self.output_shape = output_shape
        pooled_dim = 4  # Global pooling后是 (B, 1, 4, 4) = 16 dim
        self.pool = nn.AdaptiveAvgPool2d((pooled_dim, pooled_dim))  # [B, 1, 4, 4]

        self.mlp = nn.Sequential(
            nn.Flatten(),  # [B, 16]
            nn.Linear(pooled_dim * pooled_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_shape[0] * output_shape[1] * output_shape[2])
        )

    def forward(self, x):
        x = self.pool(x)  # Shape: [B, 1, 4, 4]
        x = self.mlp(x)
        x = x.view(x.size(0), *self.output_shape)
        return {"mlp_out": x}



class MLPBackbone(nn.Module):
    def __init__(self, input_shape=(1, 80, 194), output_shape=(512, 15, 20)):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        input_dim = input_shape[0] * input_shape[1] * input_shape[2]  # 1*80*194 = 15520
        output_dim = output_shape[0] * output_shape[1] * output_shape[2]  # 512*15*20 = 153600

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, output_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = x.view(x.size(0), *self.output_shape)  # reshape to (B, 512, 15, 20)
        return {"mlp_out": x}



def build_sonic_backbone_vggish(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0  # 是否训练 backbone

    backbone = VGGishBackbone()  # 自动下载 `vggish.pth`

    if not train_backbone:  # 只冻结 backbone
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()  # 设为推理模式（关闭 Dropout / BatchNorm）
    else:
        backbone.train()  # 允许训练（如果你要更新 VGGish）

    # ✅ 组合 `backbone` 和 `position_embedding`
    model = Joiner(backbone, position_embedding)
    model.num_channels = 512  # `conv4` 具有 512 个通道
    return model



def build_sonic_backbone_mlp(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0  # 是否训练 backbone

    backbone = LightAudioBackbone()  # 自动下载 `vggish.pth`

    if not train_backbone:  # 只冻结 backbone
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()  # 设为推理模式（关闭 Dropout / BatchNorm）
    else:
        backbone.train()  # 允许训练（如果你要更新 VGGish）

    # ✅ 组合 `backbone` 和 `position_embedding`
    model = Joiner(backbone, position_embedding)
    model.num_channels = 512  # `conv4` 具有 512 个通道
    return model

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.global_avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.global_max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2  # To maintain the same spatial size
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Channel-wise max pooling
        out = torch.cat([avg_out, max_out], dim=1)  # Concatenate along the channel dimension
        out = self.conv(out)
        return self.sigmoid(out)
