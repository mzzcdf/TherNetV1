# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import pdb
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.ops import CrissCrossAttention
from einops import repeat

from IPython import embed

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp=768*2, oup=768*2, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        #pdb.set_trace()
        out = identity * a_w * a_h

        return out


class CoordAtt1(nn.Module):
    def __init__(self, inp=768*2, oup=768*2, reduction=32):
        super(CoordAtt1, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        #pdb.set_trace()
        out = identity * a_w * a_h

        return out


class Mater_block_TIC(nn.Module):
    def __init__(self):
        super(Mater_block_TIC, self).__init__()

        self.epi = torch.tensor([0, 0.92, 0.95, 0.98, 0.98, 0.05, 0.05,
                    0.1, 0.1, 0.93, 0.9, 0.93, 0.1, 0.3,
                    0.3, 0.9, 0.92, 0])
        self.atten = CoordAtt(inp=2*18, oup=2*18)
        self.ac = nn.GELU()
        self.linear_fuse = ConvModule(
            in_channels=18 * 2,
            out_channels=18,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

    def Mater_Mat_Gen(self, x, epi):
        B, C, M, N = x.shape
        mat = torch.Tensor(B,18,M,N)
        for i in range(18):
            nn.init.constant_(mat[:,i,:,:] , epi[i])
        return mat

    def forward(self, x):
        epi = self.epi.to(x.device)
        Mater_Mat = self.Mater_Mat_Gen(x, epi).to(x.device)
        x = torch.cat([x, Mater_Mat], dim=1)
        x = self.atten(x)
        x = self.linear_fuse(x)
        x = self.ac(x)
        return x

        
        
class Mater_block_SODA(nn.Module):
    def __init__(self):
        super(Mater_block_SODA, self).__init__()

        self.epi = torch.tensor([0.0, 0.98, 0.9, 0.8, 0.9, 0.9,
                   0.5, 0.8, 0.7, 0.8, 0.9, 0.9, 
                   0.85, 0.9, 0.7, 0.9, 0.95, 0.9, 0.0, 0.95, 0.8])
        self.atten = CoordAtt(inp=2*21, oup=2*21)

        self.ac = nn.GELU()
        self.linear_fuse = ConvModule(
            in_channels=21 * 2,
            out_channels=21,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

    def Mater_Mat_Gen(self, x, epi):
        B, C, M, N = x.shape
        mat = torch.Tensor(B,21,M,N)
        for i in range(21):
            nn.init.constant_(mat[:,i,:,:] , epi[i])
        return mat

    def forward(self, x):
        epi = self.epi.to(x.device)
        Mater_Mat = self.Mater_Mat_Gen(x, epi).to(x.device)
        x = torch.cat([x, Mater_Mat], dim=1)
        x = self.atten(x)
        x = self.linear_fuse(x)
        x = self.ac(x)
        return x



class Mater_block_SCUT(nn.Module):
    def __init__(self):
        super(Mater_block_SCUT, self).__init__()
        self.epi = torch.tensor([0, 0.92, 0.9, 0.7, 0.8, 1, 1, 0.9, 0.9, 0.7])
        self.atten = CoordAtt(inp=2*10, oup=2*10)
        self.ac = nn.GELU()
        self.linear_fuse = ConvModule(
            in_channels=2 * 10,
            out_channels=10,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

    def Mater_Mat_Gen(self, x, epi):
        B, C, M, N = x.shape
        mat = torch.Tensor(B,10,M,N)
        for i in range(10):
            nn.init.constant_(mat[:,i,:,:] , epi[i])
        return mat

    def forward(self, x):
        epi = self.epi.to(x.device)
        Mater_Mat = self.Mater_Mat_Gen(x, epi).to(x.device)
        x = torch.cat([x, Mater_Mat], dim=1)
        x = self.atten(x)
        x = self.linear_fuse(x)
        x = self.ac(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



@HEADS.register_module()
class Thernetv1Head_TIC(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, feature_strides, **kwargs):
        super(Thernetv1Head_TIC, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels


        embedding_dim = 768

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.feature_layer1 = ConvModule(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.feature_layer2 = ConvModule(
            in_channels=16,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.attn = Attention(
            dim=16
        )
        self.attn1 = CoordAtt(inp=embedding_dim * 5, oup=embedding_dim * 5)


        self.feature_layer3 = ConvModule(
            in_channels=256,
            out_channels=768,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_fuse1 = ConvModule(
            in_channels=embedding_dim * 5,
            out_channels=embedding_dim * 4,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.Mater_block = Mater_block_TIC()

    def forward(self, inputs):
        feature_map = inputs[-1]
        inputs = inputs[:-1]
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        feature_map = feature_map.unsqueeze(dim=1)

        feature_map = self.feature_layer1(feature_map)

        feature_map = self.feature_layer2(feature_map)


        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _f = self.feature_layer3(feature_map)

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c0 = self.attn1(torch.cat([_c4, _c3, _c2, _c1, _f], dim=1))
        _c0 = self.linear_fuse1(_c0)
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1) + _c0)

        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = self.Mater_block(x) + x
        return x
        
        
@HEADS.register_module()
class TherNetV1Head_SODA(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, feature_strides, **kwargs):
        super(TherNetV1Head_SODA, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.feature_layer1 = ConvModule(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.feature_layer2 = ConvModule(
            in_channels=16,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.attn = Attention(
            dim=16
        )
        self.attn1 = CoordAtt(inp=embedding_dim * 5, oup=embedding_dim * 5)


        self.feature_layer3 = ConvModule(
            in_channels=256,
            out_channels=768,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_fuse1 = ConvModule(
            in_channels=embedding_dim * 5,
            out_channels=embedding_dim * 4,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.Mater_block = Mater_block_SODA()

    def forward(self, inputs):
        feature_map = inputs[-1]
        inputs = inputs[:-1]
        x = self._transform_inputs(inputs) 
        c1, c2, c3, c4 = x
        feature_map = feature_map.unsqueeze(dim=1)
        feature_map = self.feature_layer1(feature_map)
        feature_map = self.feature_layer2(feature_map)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _f = self.feature_layer3(feature_map)  # .permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c0 = self.attn1(torch.cat([_c4, _c3, _c2, _c1, _f], dim=1))
        _c0 = self.linear_fuse1(_c0)
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1) + _c0)

        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = self.Mater_block(x) + x
        return x
        
        
@HEADS.register_module()
class TherNetV1Head_SCUT(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, feature_strides, **kwargs):
        super(TherNetV1Head_SCUT, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.feature_layer1 = ConvModule(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.feature_layer2 = ConvModule(
            in_channels=16,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.attn = Attention(
            dim=16
        )
        self.attn1 = CoordAtt(inp=embedding_dim * 5, oup=embedding_dim * 5)


        self.feature_layer3 = ConvModule(
            in_channels=256,
            out_channels=768,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_fuse1 = ConvModule(
            in_channels=embedding_dim * 5,
            out_channels=embedding_dim * 4,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.Mater_block = Mater_block_SCUT()

    def forward(self, inputs):
        feature_map = inputs[-1]
        inputs = inputs[:-1]
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        feature_map = feature_map.unsqueeze(dim=1)
        feature_map = self.feature_layer1(feature_map)
        feature_map = self.feature_layer2(feature_map)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _f = self.feature_layer3(feature_map)  # .permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c0 = self.attn1(torch.cat([_c4, _c3, _c2, _c1, _f], dim=1))
        _c0 = self.linear_fuse1(_c0)
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1) + _c0)

        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = self.Mater_block(x) + x
        return x