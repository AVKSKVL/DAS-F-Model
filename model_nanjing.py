import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat
import math
from mamba_ssm.ops.selective_scan_interface import SelectiveScanFn, selective_scan_fn
from mamba_ssm import Mamba
from causal_conv1d import causal_conv1d_fn

device = torch.device("cuda:0")

class upms(nn.Module): 
    def __init__(self):
        super(upms, self).__init__()

    def forward(self, x):

        return F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

class R1(nn.Module): 
    def __init__(self):
        super(R1, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=1) 

    def forward(self, x):

        return self.conv(x)

class Classifier(nn.Module):
    def __init__(self, Classes):
        super().__init__()
        self.classifier_1 = nn.Linear(4 * 64 * 64, 1024)
        self.classifier_2 = nn.Linear(1024, Classes)

    def forward(self, x_in):
        x = self.classifier_1(x_in)
        return self.classifier_2(x)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs

        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

class MAMBA(nn.Module):
    def __init__(self, d_model, d_state=24, d_conv=3, expand=1, dt_rank='auto', bias=False, device=None, dtype=None, ):
        super(MAMBA, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.d_model * self.expand)
        self.d_conv = d_conv
        # self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == 'auto' else dt_rank
        self.dt_rank = 24
        # new
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=False)
        self.act = nn.SiLU()
        self.conv2d = nn.Conv2d(in_channels=4*expand, out_channels=4*expand, groups=4*expand, bias=True,
                                kernel_size=d_conv, padding=(d_conv - 1) // 2)
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(4)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
            dt_init_floor=1e-4, k_group=4,
        )
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.patch = nn.Sequential(
            nn.Conv2d(in_channels=4*expand, out_channels=64*expand, kernel_size=4, stride=4, bias=True),
            nn.BatchNorm2d(self.d_inner))

    def forward(self, x_in):  # in:B,16,16,64
        x_ = x_in.clone()
        b, p, q, d = x_.shape
        z = self.in_proj(x_)
        z = self.act(z)
        x_ = x_.reshape(b,16,16,4*self.expand,4,4).permute(0,3,1,4,2,5).reshape(b,4*self.expand,64,64)

        x_ = self.conv2d(x_)  # B,8,64,64
        x_ = self.act(x_)
        x = self.patch(x_)  # B,128,16,16

        x_proj_bias = getattr(self, "x_proj_bias", None)
        R, N = self.dt_rank, self.d_state
        x_hwwh = torch.stack([x.view(b, 64*self.expand, 256), torch.transpose(x, dim0=2, dim1=3).contiguous().view(b, 64*self.expand, 256)],
                             dim=1).view(b, 2, 64*self.expand, 256)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # b,k,d,l
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        xs = xs.view(b, 64*self.expand * 4, 256)  # (b, k * d, l)
        dts = dts.contiguous().view(b, -1, 256)  # (b, k * d, l)
        Bs = Bs.contiguous()  # (b, k, d_state, l)
        Cs = Cs.contiguous()  # (b, k, d_state, l)
        As = -self.A_logs.float().exp()  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = []
        for i in range(4):
            yi = selective_scan_fn(
                xs.view(b, 4, -1, 256)[:, i], dts.view(b, 4, -1, 256)[:, i],
                As.view(4, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(4, -1)[i],
                delta_bias=dt_projs_bias.view(4, -1)[i],
                delta_softplus=True,
            ).view(b, -1, 256)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(b, 2, -1, 256)
        wh_y = torch.transpose(out_y[:, 1].view(b, -1, 16, 16), dim0=2, dim1=3).contiguous().view(b, -1, 256)
        invwh_y = torch.transpose(inv_y[:, 1].view(b, -1, 16, 16), dim0=2, dim1=3).contiguous().view(b, -1, 256)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, 256, 128)
        y = self.out_norm(y).view(b, 16, 16, -1)
        y = y * z
        out = self.out_proj(y)

        return out

class CROMAMBA(nn.Module):
    def __init__(self, d_model, d_state=24, d_conv=3, expand=1, dt_rank='auto', bias=False, ):
        super(CROMAMBA, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.d_model * self.expand)
        self.d_conv = d_conv
        self.dt_rank = 24
        # new
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=False)
        self.act = nn.SiLU()
        self.conv2d = nn.Conv2d(in_channels=4*expand, out_channels=4*expand, groups=4*expand, bias=True,
                                kernel_size=d_conv, padding=(d_conv - 1) // 2)
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(4)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
            dt_init_floor=1e-4, k_group=4,
        )
        self.out_norm = nn.LayerNorm(64*self.expand)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        # pan
        self.in_proj_pan = nn.Linear(self.d_model, self.d_inner, bias=False)
        self.conv2d_pan = nn.Conv2d(in_channels=4*expand, out_channels=4*expand, groups=4*expand, bias=True,
                                    kernel_size=d_conv, padding=(d_conv - 1) // 2)
        self.x_proj_pan = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(4)
        ]
        self.x_proj_weight_pan = nn.Parameter(torch.stack([t.weight for t in self.x_proj_pan], dim=0))
        del self.x_proj_pan
        self.A_logs_pan, self.Ds_pan, self.dt_projs_weight_pan, self.dt_projs_bias_pan = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
            dt_init_floor=1e-4, k_group=4,
        )
        self.out_proj_pan = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.patch_ms = nn.Sequential(
            nn.Conv2d(in_channels=4*expand, out_channels=64*expand, kernel_size=4, stride=4, bias=True),
            nn.BatchNorm2d(64*expand))
        self.patch_pan = nn.Sequential(
            nn.Conv2d(in_channels=4*expand, out_channels=64*expand, kernel_size=4, stride=4, bias=True),
            nn.BatchNorm2d(64*expand))

    def forward(self, x_in, pan_in):
        # ms
        b, p, q, d = x_in.shape
        z = self.in_proj(x_in)
        z = self.act(z)
        x_ = x_in.reshape(b, 16, 16, 4*self.expand, 4, 4).permute(0, 3, 1, 4, 2, 5).reshape(b, 4*self.expand, 64, 64)

        x_ = self.conv2d(x_) 
        x_ = self.act(x_)
        x = self.patch_ms(x_) 

        x_proj_bias = getattr(self, "x_proj_bias", None)
        R, N = self.dt_rank, self.d_state
        x_hwwh = torch.stack([x.view(b, 64*self.expand, 256), torch.transpose(x, dim0=2, dim1=3).contiguous().view(b, 64*self.expand, 256)],
                             dim=1).view(b, 2, 64*self.expand, 256)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # b,k,d,l
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(b, 64*self.expand * 4, 256)  # (b, k * d, l)
        dts = dts.contiguous().view(b, -1, 256)  # (b, k * d, l)
        Bs = Bs.contiguous()  # (b, k, d_state, l)
        Cs = Cs.contiguous()  # (b, k, d_state, l)
        As = -self.A_logs.float().exp()  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # pan
        z1 = self.in_proj_pan(pan_in)
        z1 = self.act(z1)

        x1_ = pan_in.reshape(b, 16, 16, 4*self.expand, 4, 4).permute(0, 3, 1, 4, 2, 5).reshape(b, 4*self.expand, 64, 64)
        x1_ = self.conv2d_pan(x1_)  # B,128,64,64
        x1_ = self.act(x1_)
        x1 = self.patch_pan(x1_)  # B,16,16,128

        x_proj_bias_pan = getattr(self, "x_proj_bias_pan", None)
        x_hwwh_pan = torch.stack(
            [x1.view(b, 64*self.expand, 256), torch.transpose(x1, dim0=2, dim1=3).contiguous().view(b, 64*self.expand, 256)], dim=1).view(b,
                                                                                                                    2,
                                                                                                                    64*self.expand,
                                                                                                                    256)
        xs1 = torch.cat([x_hwwh_pan, torch.flip(x_hwwh_pan, dims=[-1])], dim=1)  # b,k,d,l
        x_dbl1 = torch.einsum("b k d l, k c d -> b k c l", xs1, self.x_proj_weight_pan)
        dts1, Bs1, Cs1 = torch.split(x_dbl1, [R, N, N], dim=2)
        dts1 = torch.einsum("b k r l, k d r -> b k d l", dts1, self.dt_projs_weight_pan)
        xs1 = xs1.view(b, 64*self.expand * 4, 256)  # (b, k * d, l)
        dts1 = dts1.contiguous().view(b, -1, 256)  # (b, k * d, l)
        Bs1 = Bs1.contiguous()  # (b, k, d_state, l)
        Cs1 = Cs1.contiguous()  # (b, k, d_state, l)
        As1 = -self.A_logs_pan.float().exp()  # (k * d, d_state)
        Ds1 = self.Ds_pan.float()  # (k * d)
        dt_projs_bias_pan = self.dt_projs_bias_pan.float().view(-1)  # (k * d)

        # ms
        out_y = []
        for i in range(4):
            yi = selective_scan_fn(
                xs.view(b, 4, -1, 256)[:, i], dts1.view(b, 4, -1, 256)[:, i],
                As1.view(4, -1, N)[i], Bs1[:, i].unsqueeze(1), Cs1[:, i].unsqueeze(1), Ds1.view(4, -1)[i],
                delta_bias=dt_projs_bias_pan.view(4, -1)[i],
                delta_softplus=True,
            ).view(b, -1, 256)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(b, 2, -1, 256)
        wh_y = torch.transpose(out_y[:, 1].view(b, -1, 16, 16), dim0=2, dim1=3).contiguous().view(b, -1, 256)
        invwh_y = torch.transpose(inv_y[:, 1].view(b, -1, 16, 16), dim0=2, dim1=3).contiguous().view(b, -1, 256)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, 256, 128)
        y = self.out_norm(y).view(b, 16, 16, -1)
        y = y * z
        out = self.out_proj(y)

        # pan
        out_y_pan = []
        for j in range(4):
            yi_pan = selective_scan_fn(
                xs1.view(b, 4, -1, 256)[:, j], dts.view(b, 4, -1, 256)[:, j],
                As.view(4, -1, N)[j], Bs[:, j].unsqueeze(1), Cs[:, j].unsqueeze(1), Ds.view(4, -1)[j],
                delta_bias=dt_projs_bias.view(4, -1)[j],
                delta_softplus=True,
            ).view(b, -1, 256)
            out_y_pan.append(yi_pan)
        out_y_pan = torch.stack(out_y_pan, dim=1)
        inv_y_pan = torch.flip(out_y_pan[:, 2:4], dims=[-1]).view(b, 2, -1, 256)
        wh_y_pan = torch.transpose(out_y_pan[:, 1].view(b, -1, 16, 16), dim0=2, dim1=3).contiguous().view(b, -1, 256)
        invwh_y_pan = torch.transpose(inv_y_pan[:, 1].view(b, -1, 16, 16), dim0=2, dim1=3).contiguous().view(b, -1, 256)
        y_pan = out_y_pan[:, 0] + inv_y_pan[:, 0] + wh_y_pan + invwh_y_pan
        y_pan = y_pan.transpose(dim0=1, dim1=2).contiguous()  # (B, 256, 128)
        y_pan = self.out_norm(y_pan).view(b, 16, 16, -1)
        y_pan = y_pan * z1
        out_pan = self.out_proj_pan(y_pan)

        return out, out_pan

class Mamba(nn.Module):  # in:B,16,16,64  out:B,16,16,64
    def __init__(self):
        super(Mamba, self).__init__()
        self.norm = nn.LayerNorm(64)
        self.mamba = MAMBA(64) 
        self.mlp = nn.Sequential(nn.Linear(64, 64 * 4), nn.GELU(), nn.Linear(64 * 4, 64))

    def forward(self, x):
        result = x + self.mamba(self.norm(x)) 
        result1 = result + self.mlp(self.norm(result))  

        return result1

class Cromamba(nn.Module):  # in:B,16,16,64  out:B,16,16,64
    def __init__(self):
        super(Cromamba, self).__init__()
        self.norm = nn.LayerNorm(64)
        self.cromamba = CROMAMBA(64) 
        self.mlp_ms = nn.Sequential(nn.Linear(64, 64 * 4), nn.GELU(), nn.Linear(64 * 4, 64))
        self.mlp_pan = nn.Sequential(nn.Linear(64, 64 * 4), nn.GELU(), nn.Linear(64 * 4, 64))

    def forward(self, ms_in, pan_in): 
        ms_, pan_ = self.cromamba(self.norm(ms_in), self.norm(pan_in)) 
        msc_, panc_ = ms_ + ms_in, pan_ + pan_in
        msc = msc_ + self.mlp_ms(self.norm(msc_))  
        panc = panc_ + self.mlp_pan(self.norm(panc_))

        return msc, panc

class Get_G(nn.Module):
    def __init__(self):
        super(Get_G, self).__init__()
        self.mlp = nn.Linear(16*2, 16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A: torch.tensor, B: torch.tensor): 
        combined = torch.cat((A, B), 2)

        G = self.sigmoid(self.mlp(combined))

        return G

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.get_G = Get_G()

    def forward(self, A_in, B):  # B,L,D  B,H*W,C

        G = self.get_G(A_in, B)  # G:B,L,D
        A = A_in * G

        outer_product = torch.einsum('bci,bcj->bcij', A, B) 

        # Step 2: 
        pooled_outer_product = torch.mean(outer_product, dim=2)  

        return pooled_outer_product

class Block2(nn.Module):  # in:B,L,D  out:B,L,D
    def __init__(self):
        super(Block2, self).__init__()
        self.fusion_1 = Fusion()
        self.fusion_2 = Fusion()
        self.norm = nn.LayerNorm(16)

    def forward(self, mss, pans, fu):  # B,16,16,64
        x = self.fusion_1(self.norm(mss), self.norm(pans))  # in:B,L,D  out:B,L,D
        result = self.fusion_2(self.norm(x), self.norm(fu))

        return result

class Net(nn.Module):
    def __init__(self,channel_ms,channel_pan, Classes):
        super(Net, self).__init__()

        self.upms = upms()
        self.R1 = R1()
        self.scan_161664_ms = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=4, stride=4, bias=True), Permute(0, 2, 3, 1),
            nn.LayerNorm(64))
        self.scan_161664_pan = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=4, stride=4, bias=True), Permute(0, 2, 3, 1),
            nn.LayerNorm(64))
        self.mamba_ms_1 = Mamba() 
        self.mamba_ms_2 = Mamba()
        self.mamba_pan_1 = Mamba()
        self.mamba_pan_2 = Mamba()
        self.cromamba_1 = Cromamba()
        self.cromamba_2 = Cromamba()
        self.FusionBlock3 = Block2() 
        self.FusionBlock2 = Block2()
        self.classifier = Classifier(Classes=Classes)

    def forward(self, ms_in, pan_in):
        ms = self.upms(ms_in) 
        pan = self.R1(pan_in) 
        b, c, h, w = ms.shape
        ms_patch = self.scan_161664_ms(ms) 
        pan_patch = self.scan_161664_pan(pan)

        msf1 = self.mamba_ms_1(ms_patch)
        panf1 = self.mamba_pan_1(pan_patch)
        msc1, panc1 = self.cromamba_1(ms_patch, pan_patch)
        mss1, pans1 = (msf1 - msc1).reshape(b,16,16,16,2,2).permute(0,1,4,2,5,3).reshape(b, 32,32, 16).reshape(b,32*32,16), (panf1 - panc1).reshape(b,16,16,16,2,2).permute(0,1,4,2,5,3).reshape(b, 32,32, 16).reshape(b,32*32,16) # B,4,64,64 -> B,32*32,16

        msf2 = self.mamba_ms_2(msc1)
        panf2 = self.mamba_pan_2(panc1)
        msc2, panc2 = self.cromamba_2(msc1, panc1)
        mss2, pans2 = (msf2 - msc2).reshape(b,16,16,16,2,2).permute(0,1,4,2,5,3).reshape(b, 32,32, 16).reshape(b,32*32,16), (panf2 - panc2).reshape(b,16,16,16,2,2).permute(0,1,4,2,5,3).reshape(b, 32,32, 16).reshape(b,32*32,16)

        fu = ((msc2 + panc2) / 2).reshape(b,16,16,16,2,2).permute(0,1,4,2,5,3).reshape(b, 32,32, 16).reshape(b,32*32,16)
        fu2 = self.FusionBlock2(mss2, pans2, fu)
        fu3 = self.FusionBlock3(mss1, pans1, fu2)
        fu4 = fu3.contiguous().view(b, 16 * 16 * 64)
        output = self.classifier(fu4)

        return output


if __name__ == "__main__":
    device = torch.device("cuda:0")
    net = Net(4,1,Classes=11)
    net = net.to(device)
    x1 = torch.randn(1, 4, 16, 16, device=device)
    x2 = torch.randn(1, 1, 64, 64, device=device)

    output = net(x1, x2)
    print(output.shape)
    print(type(output))
    print(f"Number of parameters: {sum(p.numel() for p in net.parameters()):.2f}")
    # print("结束")
