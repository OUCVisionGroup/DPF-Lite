import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from . import litemono_encoder
from . import litemono_decoder


class DWTModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lp = torch.tensor([1., 1.]) / 2.0
        self.hp = torch.tensor([-1., 1.]) / 2.0

        self.register_buffer('kernel_LL', self.lp.view(1, 1, 1, 2) @ self.lp.view(1, 1, 2, 1))
        self.register_buffer('kernel_LH', self.lp.view(1, 1, 1, 2) @ self.hp.view(1, 1, 2, 1))
        self.register_buffer('kernel_HL', self.hp.view(1, 1, 1, 2) @ self.lp.view(1, 1, 2, 1))
        self.register_buffer('kernel_HH', self.hp.view(1, 1, 1, 2) @ self.hp.view(1, 1, 2, 1))

    def forward(self, x):
        B, C, H, W = x.shape

        LL = F.conv2d(x, self.kernel_LL.repeat(C, 1, 1, 1), stride=2, groups=C)
        LH = F.conv2d(x, self.kernel_LH.repeat(C, 1, 1, 1), stride=2, groups=C)
        HL = F.conv2d(x, self.kernel_HL.repeat(C, 1, 1, 1), stride=2, groups=C)
        HH = F.conv2d(x, self.kernel_HH.repeat(C, 1, 1, 1), stride=2, groups=C)

        return LL, LH, HL, HH

class IDWTModule(nn.Module):
    def __init__(self):
        super().__init__()
        ilp = torch.tensor([1., 1.]) / math.sqrt(2)
        ihp = torch.tensor([1., -1.]) / math.sqrt(2)
        self.register_buffer('ikernel_LL', torch.outer(ilp, ilp).view(1, 1, 2, 2))
        self.register_buffer('ikernel_LH', torch.outer(ilp, ihp).view(1, 1, 2, 2))
        self.register_buffer('ikernel_HL', torch.outer(ihp, ilp).view(1, 1, 2, 2))
        self.register_buffer('ikernel_HH', torch.outer(ihp, ihp).view(1, 1, 2, 2))

    def forward(self, LL, LH, HL, HH):

        B, C, H, W = LL.shape
        up_LL = F.conv_transpose2d(LL, self.ikernel_LL.repeat(C, 1, 1, 1), stride=2, groups=C)
        up_LH = F.conv_transpose2d(LH, self.ikernel_LH.repeat(C, 1, 1, 1), stride=2, groups=C)
        up_HL = F.conv_transpose2d(HL, self.ikernel_HL.repeat(C, 1, 1, 1), stride=2, groups=C)
        up_HH = F.conv_transpose2d(HH, self.ikernel_HH.repeat(C, 1, 1, 1), stride=2, groups=C)
        return up_LL + up_LH + up_HL + up_HH

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_norm=True):
        super(DoubleConv, self).__init__()
        if with_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

class FrequencyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DWTModule()
        self.idwt = IDWTModule()
        self.attention = nn.Sequential(
            nn.Conv2d(4 * 3, 4 * 3 // 2, 1),
            nn.GELU(),
            nn.Conv2d(4 * 3 // 2, 4 * 3, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.attention[-2].bias[9:12], -1)

    def forward(self, x):
        LL, LH, HL, HH = self.dwt(x)
        attn = self.attention(torch.cat([LL, LH, HL, HH], dim=1))
        return self.idwt(LL * attn[:, 0:3, :, :], LH * attn[:, 3:6, :, :], HL * attn[:, 6:9, :, :], HH * attn[:, 9:12, :, :])

class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1)
        self.linear_1 = nn.Conv2d(dim, dim, 1)
        self.linear_2 = nn.Conv2d(dim, dim, 1)

        self.lde = DMlp(dim, 2)
        self.dw_conv = nn.Conv2d(dim, dim, 3, groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 2
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))
        self.inception = EnhancedInceptionDWConv(dim)

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)

        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)

        x_l = x * F.interpolate(
            self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)),
            size=(h, w), mode='nearest'
        )

        y_d = self.lde(y)
        y_i = self.linear_2(x_l + y_d)
        return self.inception(y_i)

class EnhancedInceptionDWConv(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3,
                 band_kernel_size=11, branch_ratio=0.125, img_size=None):
        super().__init__()

        if img_size and min(img_size) < 64:
            branch_ratio = max(0.0625, branch_ratio * 0.5)

        gc = int(in_channels * branch_ratio)

        self.dwconv_hw = nn.Sequential(
            nn.Conv2d(gc, gc, square_kernel_size,
                      padding=square_kernel_size // 2, groups=gc),
            nn.GELU()
        )
        self.dwconv_w = nn.Conv2d(gc, gc, (1, band_kernel_size),
                                  padding=(0, band_kernel_size // 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, (band_kernel_size, 1),
                                  padding=(band_kernel_size // 2, 0), groups=gc)

        self.split_indexes = [in_channels - 3 * gc, gc, gc, gc]

        self.fuse = nn.Conv2d(in_channels, in_channels, 1)
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x_out = torch.cat([
            x_id,
            self.dwconv_hw(x_hw),
            self.dwconv_w(x_w),
            self.dwconv_h(x_h)
        ], dim=1)
        return self.norm(self.fuse(x_out))

class ParameterInteraction(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.p_gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.Sigmoid()
        )
        self.norm = nn.BatchNorm2d(channels)
        self.res_scale = nn.Parameter(torch.tensor(0.6))

    def forward(self, f1_p, f1_p1, f1_p2):
        p1_to_p = self.p_gate(torch.cat([f1_p, f1_p1], dim=1)) * f1_p1
        p2_to_p = self.p_gate(torch.cat([f1_p, f1_p2], dim=1)) * f1_p2
        f2_p = f1_p + self.res_scale * (p1_to_p + p2_to_p)

        return self.norm(f2_p)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
                    DoubleConv(in_ch, in_ch),
                    EnhancedInceptionDWConv(in_ch),
                    nn.Conv2d(in_ch, in_ch//2, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_ch//2, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = nn.Sequential(
                    DoubleConv(in_ch, out_ch//2),
                    EnhancedInceptionDWConv(out_ch//2),
                    DoubleConv(out_ch//2, out_ch)
        )
    def forward(self, x):
        return self.conv(x)

class GradientConsistencyLoss(nn.Module):
    def __init__(self, mode='l1'):
        super().__init__()
        self.mode = mode

    def compute_gradients(self, x):
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]  # (B,C,H-1,W)
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]  # (B,C,H,W-1)
        return dy, dx

    def normalize(self, pred):
        d_max = torch.max(pred).item()
        d_min = torch.min(pred).item()
        normalized_d = (pred-d_min)/(d_max-d_min)
        return normalized_d

    def forward(self, target, pred):
        normalized_pred = self.normalize(pred)
        pred_dy, pred_dx = self.compute_gradients(normalized_pred)
        normalized_target = self.normalize(target)
        target_dy, target_dx = self.compute_gradients(normalized_target)

        if self.mode == 'l2':
            grad_diff_y = (pred_dy - target_dy) ** 2
            grad_diff_x = (pred_dx - target_dx) ** 2
        else:
            grad_diff_y = torch.abs(pred_dy - target_dy)
            grad_diff_x = torch.abs(pred_dx - target_dx)
        # 加权平均
        loss_y = grad_diff_y.mean()
        loss_x = grad_diff_x.mean()

        return (loss_y + loss_x) / 2

class MainNet(nn.Module):
    def __init__(self, mid_ch=32, watermono_path='./DPEM/watermono_checkpoint'):
        super(MainNet, self).__init__()

        self.current_stage = 0
        self.mid_ch = mid_ch
        self.watermono_path = watermono_path
        self.norm = nn.BatchNorm2d(self.mid_ch)
        self.grad_loss = GradientConsistencyLoss()

        self.B_inConv = InConv(6, self.mid_ch)
        self.beatD_inConv = InConv(3, self.mid_ch)
        self.beatB_inConv = InConv(3, self.mid_ch)

        self.B_smfa = SMFA(self.mid_ch)
        self.betaD_smfa = SMFA(self.mid_ch)
        self.betaB_smfa = SMFA(self.mid_ch)

        self.B_gate = ParameterInteraction(self.mid_ch)
        self.betaD_gate = ParameterInteraction(self.mid_ch)
        self.betaB_gate = ParameterInteraction(self.mid_ch)

        self.B_outConv = OutConv(self.mid_ch, 3)
        self.betaD_outConv = OutConv(self.mid_ch, 3)
        self.betaB_outConv = OutConv(self.mid_ch, 3)

        self.wm_encoder = litemono_encoder.LiteMono(height=256, width=256)
        encoder_dict = torch.load(os.path.join(self.watermono_path, 'encoder.pth'))
        self.wm_encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in self.wm_encoder.state_dict()})

        self.wm_decoder = litemono_decoder.DepthDecoder(self.wm_encoder.num_ch_enc)
        decoder_dict = torch.load(os.path.join(self.watermono_path, 'depth.pth'))
        self.wm_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in self.wm_decoder.state_dict()})

        self.depth_scaler = nn.Parameter(torch.tensor(1.0))  # 初始缩放因子
        self.depth_shift = nn.Parameter(torch.tensor(0.0))  # 初始偏移量

        self.depth_inConv = DoubleConv(4, self.mid_ch)
        self.depth_smfa1 = SMFA(self.mid_ch)
        self.depth_midConv = DoubleConv(self.mid_ch, self.mid_ch*2)
        self.depth_smfa2 = SMFA(self.mid_ch*2)
        self.depth_norm = nn.BatchNorm2d(self.mid_ch*2)
        self.depth_outConv = OutConv(self.mid_ch*2, 1)

    def get_train_parameters(self, lr=0.00005):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.betaD_outConv.parameters():
            param.requires_grad = True
        for param in self.betaB_outConv.parameters():
            param.requires_grad = True
        parameters = [
            {'params': self.betaD_outConv.parameters(), 'lr': lr},
            {'params': self.betaB_outConv.parameters(), 'lr': lr},
            {'params': self.B_outConv.parameters(), 'lr': lr},
            {'params': self.betaD_smfa.parameters(), 'lr': lr},
            {'params': self.betaB_smfa.parameters(), 'lr': lr},
            {'params': self.B_smfa.parameters(), 'lr': lr}
        ]
        return parameters

    def forward(self, x, pre_B):
        xB = torch.cat((x, pre_B), dim=1)
        f1_B = self.B_inConv(xB)
        f1_betaD = self.beatD_inConv(x)
        f1_betaB = self.beatB_inConv(x)

        f2_B = self.B_smfa(f1_B)
        f2_betaD = self.betaD_smfa(f1_betaD)
        f2_betaB = self.betaB_smfa(f1_betaB)

        B = self.B_outConv(self.norm(f1_B + self.B_gate(f2_B, f2_betaD, f2_betaB)))
        betaD = self.betaD_outConv(self.norm(f1_betaD + self.betaD_gate(f2_betaD, f2_B, f2_betaB)))
        betaB = self.betaB_outConv(self.norm(f1_betaB + self.betaB_gate(f2_betaB, f2_B, f2_betaD)))

        d_features = self.wm_encoder(x)
        relDepth = self.wm_decoder(d_features)
        f1_d = self.depth_inConv(torch.cat((x, relDepth), dim=1))
        f2_d = self.depth_smfa1(f1_d)
        f3_d = self.depth_midConv(self.norm(f1_d + f2_d))
        f4_d = self.depth_smfa2(f3_d)
        d = self.depth_outConv(self.depth_norm(f3_d + f4_d))

        return B*255.0, betaD, betaB, d * self.depth_scaler + self.depth_shift
