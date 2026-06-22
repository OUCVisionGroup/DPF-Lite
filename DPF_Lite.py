import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.conv(x)

class MLP(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, out_dim=128):
        super().__init__()
        self.param1_branch = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim))

        self.param2_branch = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim))

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.out_dim = out_dim

    def forward(self, param1, param2):
        p1 = self.param1_branch(param1) * self.alpha
        p2 = self.param2_branch(param2) * self.beta
        return p1 + p2  # 特征融合

class DegradationAwareBlock(nn.Module):
    def __init__(self, in_dim=3, out_dim=128, reduction_ratio=1):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=reduction_ratio, stride=reduction_ratio),
            nn.Conv2d(in_dim, out_dim // 4, kernel_size=3, padding=1))

        self.mlp = MLP(in_dim=out_dim // 4, out_dim=out_dim)

        self.out_conv = nn.Conv2d(out_dim, out_dim, kernel_size=1)

    def forward(self, param1, param2):
        p1 = self.downsample(param1)
        p2 = self.downsample(param2)

        B, C, H, W = p1.shape
        p1 = p1.permute(0, 2, 3, 1)
        p2 = p2.permute(0, 2, 3, 1)
        feat = self.mlp(p1, p2)
        feat = feat.permute(0, 3, 1, 2)

        return self.out_conv(feat)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2)
            self.conv_out = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_channels // 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        if hasattr(self, 'conv_out'):
            x = self.conv_out(self.conv(x))
        else:
            x = self.conv(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, num_filter):
        super(ResBlock, self).__init__()
        body = []
        for i in range(2):
            body.append(nn.ReflectionPad2d(1))
            body.append(nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=0))
            if i == 0:
                body.append(nn.Mish())
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        x = res + x
        return x

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.in_conv = nn.Sequential(ResBlock(in_ch),
                                     nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                     ResBlock(out_ch),
                                     nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

    def forward(self, x):
        return self.in_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.out_conv = nn.Sequential(ResBlock(in_ch),
                                      nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                      ResBlock(out_ch),
                                      nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.out_conv(x)
        return self.sigmoid(x)

class DegradationAwareXCA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_fusion = nn.Linear(2 * dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.proj = nn.Linear(dim, dim)

        self.attn_weights = None

    def forward(self, x, deg):
        B, N, C = x.shape
        q = self.q_fusion(torch.cat([x, deg], dim=-1))
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        self.attn_weights = attn.detach()

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

class DegradationAwareLGFI(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 num_heads=8, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim

        self.norm_xca = LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = DegradationAwareXCA(self.dim, num_heads=num_heads)

        self.norm = LayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn_weights = None

    def forward(self, x, deg):
        input_ = x
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, H * W).permute(0, 2, 1)
        deg_flat = deg.reshape(B, C, H * W).permute(0, 2, 1)
        x_norm = self.norm_xca(x_flat)
        deg_norm = self.norm_xca(deg_flat)

        x_attn = self.xca(x_norm, deg_norm)
        self.attn_weights = self.xca.attn_weights

        x_attn = x_attn.permute(0, 2, 1).reshape(B, C, H, W)
        x = x + self.gamma_xca[None, :, None, None] * x_attn
        x = x.reshape(B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input_ + self.drop_path(x)
        return x

class UNet_lite(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, isTrain=True):
        super(UNet_lite, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.inc = InConv(input_nc, 16)
        self.outc = OutConv(16, output_nc)
        self.isTrain = isTrain

        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)

        self.bridge = ResBlock(256)

        self.up1 = Up(256 + 128, 128)
        self.up2 = Up(128 + 64, 64)
        self.up3 = Up(64 + 32, 32)
        self.up4 = Up(32 + 16, 16)

        self.deg_down1 = DegradationAwareBlock(3, 16, 1)
        self.deg_down2 = DegradationAwareBlock(3, 32, 2)
        self.deg_down3 = DegradationAwareBlock(3, 64, 4)

        self.lgfi_down1 = DegradationAwareLGFI(16)
        self.lgfi_down2 = DegradationAwareLGFI(32)
        self.lgfi_down3 = DegradationAwareLGFI(64)

    def forward(self, x, B, T):
        x1 = self.inc(x)
        x1 = self.lgfi_down1(x1, self.deg_down1(B, T))
        x2 = self.down1(x1)
        x2 = self.lgfi_down2(x2, self.deg_down2(B, T))
        x3 = self.down2(x2)
        x3 = self.lgfi_down3(x3, self.deg_down3(B, T))
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.bridge(x5)

        x = self.up1(x6, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

