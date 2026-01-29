
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from einops import repeat
    EMVM_AVAILABLE = True
    print("✅ EMVM 依赖已加载")
except ImportError as e:
    EMVM_AVAILABLE = False
    print(f"⚠️  EMVM 不可用: {e}")

class DynamicFusionGate(nn.Module):

    def __init__(self, channels, reduction=16, init_ema_weight=0.6):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 2, 1, bias=False)
        )

        self.register_buffer('init_weight', torch.tensor([init_ema_weight, 1 - init_ema_weight]))

    def forward(self, x):

        avg_feat = self.avg_pool(x)  # [B, C, 1, 1]
        max_feat = self.max_pool(x)  # [B, C, 1, 1]

        gate_input = torch.cat([avg_feat, max_feat], dim=1)  # [B, 2C, 1, 1]
        gate_weights = self.fc(gate_input)  # [B, 2, 1, 1]

        gate_weights = F.softmax(gate_weights, dim=1)
        gate_weights = 0.8 * gate_weights + 0.2 * self.init_weight.view(1, 2, 1, 1)

        return gate_weights


class EfficientFusionModule(nn.Module):
    def __init__(
        self,
        channels: int,
        ema_factor: int = 32,
        d_state: int = 16,
        expand: float = 2.0,
        drop_path: float = 0.1,
        reduction: int = 16,
        use_ema: bool = True,
        use_emvm: bool = True,
        use_adaptive_gate: bool = True,
        fixed_ema_weight: float = 0.5,
        use_light_emvm: bool = False,
        **kwargs
    ):
        super().__init__()
        self.channels = channels

        global EMVM_AVAILABLE
        self.use_ema = use_ema
        self.use_emvm = use_emvm and EMVM_AVAILABLE
        self.use_adaptive_gate = use_adaptive_gate
        self.fixed_ema_weight = fixed_ema_weight
        self.use_light_emvm = use_light_emvm


        if not self.use_ema and not self.use_emvm:
            raise ValueError("至少需要启用 EMA 或 EMVM 中的一个！")


        if self.use_ema:
            self.ema_branch = EMA(channels, factor=ema_factor)

        # EMVM
        if self.use_emvm:
            if use_light_emvm:

                self.emvm_branch = VSSBlockLight(
                    hidden_dim=channels,
                    d_state=d_state,
                    expand=expand,  #  1.0
                    drop_path=drop_path,
                    **kwargs
                )
            else:
                # expand=2.0
                self.emvm_branch = VSSBlock(
                    hidden_dim=channels,
                    d_state=d_state,
                    expand=expand,
                    drop_path=drop_path,
                    **kwargs
                )


        if self.use_ema and self.use_emvm:
            if self.use_adaptive_gate:

                self.fusion_gate = DynamicFusionGate(channels, reduction)
            else:

                self.register_buffer(
                    'fixed_weights',
                    torch.tensor([fixed_ema_weight, 1.0 - fixed_ema_weight])
                )


        self._print_config()


        self.norm = nn.LayerNorm(channels)

    def _print_config(self):

        mode_parts = []
        if self.use_ema:
            mode_parts.append("EMA")
        if self.use_emvm:
            if self.use_light_emvm:
                mode_parts.append("EMVM-Light")
            else:
                mode_parts.append("EMVM")

        mode = "+".join(mode_parts)

        if len(mode_parts) == 2:
            if self.use_adaptive_gate:
                mode += " (自适应门控)"
            else:
                mode += f" (固定权重 {self.fixed_ema_weight:.1f}:{1-self.fixed_ema_weight:.1f})"

        print(f"✅ EfficientFusion 配置: {mode}")

    def forward(self, x):

        B, C, H, W = x.shape


        if self.use_ema:
            ema_out = self.ema_branch(x)  # [B, C, H, W]


        if self.use_emvm:
            emvm_out = self.emvm_branch(x)  # [B, C, H, W]


        if self.use_ema and self.use_emvm:

            if self.use_adaptive_gate:

                gate_weights = self.fusion_gate(x)  # [B, 2, 1, 1]
                w_ema = gate_weights[:, 0:1, :, :]   # [B, 1, 1, 1]
                w_emvm = gate_weights[:, 1:2, :, :]  # [B, 1, 1, 1]
            else:

                w_ema = self.fixed_weights[0]
                w_emvm = self.fixed_weights[1]

            fused = w_ema * ema_out + w_emvm * emvm_out

        elif self.use_ema:

            fused = ema_out

        else:

            fused = emvm_out


        output = x + fused


        output = output.permute(0, 2, 3, 1)  # [B, H, W, C]
        output = self.norm(output)
        output = output.permute(0, 3, 1, 2)  # [B, C, H, W]

        return output

    def get_params_and_flops(self, input_shape):

        B, C, H, W = input_shape


        ema_params = sum(p.numel() for p in self.ema_branch.parameters())

        total_params = ema_params
        total_flops = ema_params * H * W

        if self.use_emvm:
            emvm_params = sum(p.numel() for p in self.emvm_branch.parameters())
            gate_params = sum(p.numel() for p in self.fusion_gate.parameters())
            total_params += emvm_params + gate_params
            total_flops += emvm_params * H * W

        return {
            'params': total_params,
            'flops': total_flops,
            'params_M': total_params / 1e6,
            'flops_G': total_flops / 1e9
        }

#EMA
class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


#EMVM block

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SS2D_Block(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            ss2d_type=None,

    ):
        super().__init__()
        # if ss2d_type == None:
        #     raise 'ss2d_type undefine'
        if ss2d_type is None:
            raise ValueError('ss2d_type undefined')
        self.ss2d_type = ss2d_type
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,

        )
        self.act = nn.SiLU()
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False) for _ in range(ss2d_type)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         ) for _ in range(ss2d_type)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.ss2d_type, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.ss2d_type, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        if self.ss2d_type != 1:
            b1_stride = 2
            stride = 1
            d_conv = 3
            self.conv2d1_b1 = nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                groups=d_model,
                # bias=conv_bias,
                kernel_size=7,
                stride=b1_stride,
                padding=3,
                # **factory_kwargs,
            )
            self.upsample = nn.ConvTranspose2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=d_model,
                bias=conv_bias,
                output_padding=1,
            )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                ):
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
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        # K = 3
        K = self.ss2d_type
        if K == 3:
            x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                                 dim=1).view(B, 2, -1, L)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
            xs = xs[:, 1:, :, :]
        else:
            xs = x

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        if K == 3:
            inv_y = torch.flip(out_y[:, 1:3], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 0].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 0].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            seq = (inv_y[:, 0], wh_y, invwh_y)
        else:
            seq = out_y[:, 0]
        return seq

    def forward(self, x: torch.Tensor, **kwargs):
        #new 1023
        original_shape = None
        if self.ss2d_type == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
            #new 1023
            original_shape = (x.shape[2],x.shape[3])
            x = self.conv2d1_b1(x).permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        if self.ss2d_type == 3:
            y1, y2, y3 = self.forward_core(x)
            y = y1 + y2 + y3
        elif self.ss2d_type == 1:
            y = self.forward_core(x)

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        # if self.ss2d_type == 3:
        #     out = out.permute(0, 3, 1, 2).contiguous()
        #     out = self.upsample(out).permute(0, 2, 3, 1).contiguous()
        if self.ss2d_type == 3:
            out = out.permute(0, 3, 1, 2).contiguous()
            out = self.upsample(out)  # 可能变成 20×20

            if original_shape is not None:
                out = F.interpolate(out, size=original_shape, mode='bilinear',
                                    align_corners=False)
            out = out.permute(0, 2, 3, 1).contiguous()

        if self.dropout is not None:
            out = self.dropout(out)

        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.0,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention_1 = SS2D_Block(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate,
                                           ss2d_type=1, **kwargs)
        self.self_attention_3 = SS2D_Block(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate,
                                           ss2d_type=3, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = Mlp(hidden_dim, hidden_dim, hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input):
        B, C, H, W = input.shape
        input = input.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x_1 = self.self_attention_1(x)
        x_3 = self.self_attention_3(x)

        x = x_1 + x_3
        x = input * self.skip_scale + self.drop_path(x)

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x



class VSSBlockLight(nn.Module):

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 1.0,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)

        self.self_attention = SS2D_Block(
            d_model=hidden_dim,
            d_state=d_state,
            expand=expand,
            dropout=attn_drop_rate,
            ss2d_type=1,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input):
        B, C, H, W = input.shape
        input = input.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = input * self.skip_scale + self.drop_path(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x



EMA_EMVM_Fusion = EfficientFusionModule


if __name__ == "__main__":
    # test
    print(f"EMVM 可用: {EMVM_AVAILABLE}")


    B, C, H, W = 2, 256, 20, 20


    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ 使用 GPU 进行测试")
    else:
        device = torch.device('cpu')
        print(f"⚠️  GPU 不可用，将使用 CPU (EMVM 会被禁用)")

    x = torch.randn(B, C, H, W).to(device)


    fusion = EfficientFusionModule(
        channels=C,
        ema_factor=32,
        d_state=16,
        expand=2.0,
        drop_path=0.1,
        use_emvm=EMVM_AVAILABLE and torch.cuda.is_available()
    ).to(device)


    out = fusion(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")

    stats = fusion.get_params_and_flops(x.shape)
    print(f"参数量: {stats['params_M']:.2f}M")
    print(f"FLOPs: {stats['flops_G']:.2f}G")
