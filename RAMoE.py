# @Time    : 2025/3/7 15:16
# @Author  : Nan Xiao
# @File    : RAMoE.py

import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import numbers


# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# --------------------------Main------------------------------- #
class RAMoEN(nn.Module):
    def __init__(self, hsi_channels, msi_channels, upscale_factor, dim_moe=32, num_experts=8, num_shared=1, residual_type='3conv', n_feats=96, deepth=3, value_case='sum', act_func='gelu',ffn_mode='RAMoE'):
        super(RAMoEN, self).__init__()
        kernel_size = 3
        self.up_factor = upscale_factor
        self.residual_type = residual_type
        self.headX = nn.Conv2d(hsi_channels, n_feats, kernel_size, stride=1, padding=3 // 2)
        self.headY = nn.Sequential(
            nn.Conv2d(msi_channels, 64, kernel_size, stride=1, padding=3 // 2),
            nn.ReLU(),
            nn.Conv2d(64, n_feats, kernel_size, stride=1, padding=3 // 2)
        )

        self.body1 = CMFM(dim=n_feats, num_heads=n_feats//16, deepth=deepth, value_case=value_case, dim_moe=dim_moe, num_experts=num_experts, num_shared=num_shared, ffn_mode=ffn_mode, act_func=act_func)
        self.body2 = CMFM(dim=n_feats, num_heads=n_feats//16, deepth=deepth, value_case=value_case, dim_moe=dim_moe, num_experts=num_experts, num_shared=num_shared, ffn_mode=ffn_mode, act_func=act_func)
        self.body3 = CMFM(dim=n_feats, num_heads=n_feats//16, deepth=deepth, value_case=value_case, dim_moe=dim_moe, num_experts=num_experts, num_shared=num_shared, ffn_mode=ffn_mode, act_func=act_func)

        self.fe_conv1 = torch.nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=3 // 2)
        self.fe_conv2 = torch.nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=3 // 2)
        self.fe_conv3 = torch.nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=3 // 2)

        self.final = nn.Conv2d(n_feats, hsi_channels, kernel_size, stride=1, padding=3 // 2)

        if self.residual_type == '3conv':
            self.SSRM = nn.Sequential(
                nn.Conv2d(hsi_channels + msi_channels, (hsi_channels + msi_channels) * 2, kernel_size, stride=1, padding=1),
                nn.Conv2d((hsi_channels + msi_channels) * 2, (hsi_channels + msi_channels) * 2, kernel_size, stride=1,
                          padding=1),
                nn.Conv2d((hsi_channels + msi_channels) * 2, hsi_channels, kernel_size, stride=1, padding=1)
            )
        elif self.residual_type == 'UPLRHSI':
            self.SSRM = nn.Sequential(
                nn.Conv2d(hsi_channels, (hsi_channels + msi_channels) * 2, kernel_size, stride=1,
                          padding=1),
                nn.Conv2d((hsi_channels + msi_channels) * 2, (hsi_channels + msi_channels) * 2, kernel_size, stride=1,
                          padding=1),
                nn.Conv2d((hsi_channels + msi_channels) * 2, hsi_channels, kernel_size, stride=1, padding=1)
            )
        elif self.residual_type == 'HRMSI':
            self.SSRM = nn.Sequential(
                nn.Conv2d(msi_channels, (hsi_channels + msi_channels) * 2, kernel_size, stride=1,
                          padding=1),
                nn.Conv2d((hsi_channels + msi_channels) * 2, (hsi_channels + msi_channels) * 2, kernel_size, stride=1,
                          padding=1),
                nn.Conv2d((hsi_channels + msi_channels) * 2, hsi_channels, kernel_size, stride=1, padding=1)
            )


    def forward(self, x, y):
        x = torch.nn.functional.interpolate(x, scale_factor=self.up_factor, mode='bicubic', align_corners=False)
        bicubic_x = x
        hrmsi = y
        final_residual = torch.cat([x, y], dim=1)

        x = self.headX(x)
        res = x
        y = self.headY(y)

        x = self.body1(x, y)
        x1 = self.fe_conv1(x)

        x2 = self.body2(x1, y)
        x2 = self.fe_conv2(x2)

        x3 = self.body3(x2, y)
        x3 = self.fe_conv3(x3)

        x_out = res + x3
        if self.residual_type == '3conv':
            x_out = self.final(x_out) + self.SSRM(final_residual)
        elif self.residual_type == 'UPLRHSI':
            x_out = self.final(x_out) + self.SSRM(bicubic_x)
        elif self.residual_type == 'none':
            x_out = self.final(x_out)
        elif self.residual_type == 'HRMSI':
            x_out = self.final(x_out) + self.SSRM(hrmsi)
        else:
            x_out = x_out
        return x_out


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
class CMFM(nn.Module):
    def __init__(self, dim=48, num_heads=1, dim_moe=32, num_experts=8, num_shared=1, ffn_mode='RAMoE', bias=False, LayerNorm_type='WithBias', deepth=3, value_case='mix', act_func='relu'):
        super(CMFM, self).__init__()
        self.blocks = nn.ModuleList([
            Cross_TransformerBlock(dim=dim, num_heads=num_heads, dim_moe=dim_moe, num_experts=num_experts, num_shared=num_shared, ffn_mode=ffn_mode, bias=bias, LayerNorm_type=LayerNorm_type, value_case=value_case, act_func=act_func)
            for i in range(deepth)])
        self.conv = nn.Conv2d(dim, dim, 3, 1,1)

    def forward(self, x1, x2):
        short_cut = x1
        for blk in self.blocks:
            x1 = blk(x1, x2)
        x1 =  short_cut + self.conv(x1)
        return x1


##########################################################################
class Expert(nn.Module):
    def __init__(self, dim: int, dim_moe: int, act_func: str, bias: bool):
        super(Expert, self).__init__()
        self.act_func = act_func
        self.project_in = nn.Conv2d(dim, dim_moe * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim_moe * 2, dim_moe * 2, kernel_size=3, stride=1, padding=1, groups=dim_moe * 2, bias=bias)
        self.project_out = nn.Conv2d(dim_moe, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        if self.act_func == 'gelu':
            x = F.gelu(x1) * x2
        elif self.act_func == 'silu':
            x = F.silu(x1) * x2
        else:
            x = F.relu(x1) * x2
        x = self.project_out(x)
        return x


class Gate(nn.Module):
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Conv2d(dim, dim * num_experts, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b,c,h,w = x.size()
        scores = self.w1(x)
        weights = scores.reshape(self.num_experts, b, c, h, w)
        return weights


class RAMoE(nn.Module):
    def __init__(self, dim: int, dim_moe: int, num_RAE: int, num_IWE: int, act_func: str, bias: bool):
        super().__init__()
        self.dim = dim
        self.num_experts = num_RAE
        self.num_shared = num_IWE
        self.gate = Gate(dim, num_RAE)
        self.experts = nn.ModuleList([Expert(dim, dim_moe, act_func, bias) for i in range(num_RAE)])
        if num_IWE == 1:
            self.shared_experts = Expert(dim, dim_moe, act_func, bias)
        elif num_IWE > 1:
            self.shared_experts = nn.ModuleList([Expert(dim, dim_moe, act_func, bias) for i in range(num_IWE)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.gate(x)
        y = torch.zeros_like(x)
        for i in range(self.num_experts):
          expert_output = self.experts[i](x) * weights[i]
          y += expert_output
        if self.num_shared == 1:
            z = self.shared_experts(x)
            return y+z
        elif self.num_shared > 1:
            for i in range(self.num_shared):
                z = self.shared_experts[i](x)
                y += z
            return y
        else:
            return y


def normalize_band(band):
    band_min, band_max = torch.min(band), torch.max(band)
    band = torch.clip((band - band_min) / (band_max - band_min + 1e-6), 0, 1)
    return band


##########################################################################
class Cross_TransformerBlock(nn.Module):
    def __init__(self, dim=48, num_heads=1, dim_moe=32, num_experts=8, num_shared=1, ffn_mode='RAMoE', bias=False, LayerNorm_type='WithBias', value_case='mix', act_func='relu'):
        super(Cross_TransformerBlock, self).__init__()

        self.norm11 = LayerNorm(dim, LayerNorm_type)
        self.norm12 = LayerNorm(dim, LayerNorm_type)
        self.attn = ASM(dim, num_heads, bias, value_case)
        self.norm2 = LayerNorm(dim*2, LayerNorm_type)
        if ffn_mode == 'RAMoE':
            self.ffn = RAMoE(dim=dim*2, dim_moe=dim_moe, num_RAE=num_experts, num_IWE=num_shared, act_func=act_func, bias=bias)
        elif ffn_mode == 'FFN':
            self.ffn = Expert(dim=dim*2, dim_moe=dim_moe, act_func=act_func, bias=bias)
        self.conv = nn.Conv2d(dim*2, dim, 3,1,1)

    def forward(self, x1, x2):
        x = self.attn(self.norm11(x1), self.norm12(x2))
        x = x + self.ffn(self.norm2(x))
        x = self.conv(x)
        return x


##########################################################################
class ASM(nn.Module):
    def __init__(self, dim, num_heads, bias, value_case):
        super(ASM, self).__init__()
        self.num_heads = num_heads
        self.value_case = value_case
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv1 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv2 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.alpha = nn.parameter.Parameter(torch.ones(num_heads, 1, 1))


    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        qkv1 = self.qkv_dwconv1(self.qkv1(x1))
        q1, k1, v1 = qkv1.chunk(3, dim=1)

        qkv2 = self.qkv_dwconv2(self.qkv2(x2))
        q2, k2, v2 = qkv2.chunk(3, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)

        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature1
        attn1 = attn1.softmax(dim=-1)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature2
        attn2 = attn2.softmax(dim=-1)

        if self.value_case == 'normal':
            out1 = (attn1 @ v1)
            out2 = (attn2 @ v2)
        elif self.value_case == 'sum':
            v3 = self.alpha * v1 + (1-self.alpha) * v2
            out1 = (attn1 @ v3)
            out2 = (attn2 @ v3)
        else:
            out1 = (attn1 @ v2)
            out2 = (attn2 @ v1)

        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = torch.cat([out1+x1, out2+x2], dim=1)
        return out
