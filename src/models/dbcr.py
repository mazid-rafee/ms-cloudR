import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t, max_period=10000.0):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, device=t.device) / half
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        dw_channel = channels * 2
        ffn_channel = channels * 2

        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = nn.GroupNorm(1, channels)
        self.conv4 = nn.Conv2d(channels, ffn_channel, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        return x + y * self.gamma


class SFBlock(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x_opt, x_sar):
        b, c, h, w = x_opt.shape
        q = self.q(x_opt).reshape(b, self.heads, c // self.heads, h * w)
        k = self.k(x_sar).reshape(b, self.heads, c // self.heads, h * w)
        v = self.v(x_sar).reshape(b, self.heads, c // self.heads, h * w)

        q = q / math.sqrt(c // self.heads)
        attn = torch.softmax(
            torch.einsum("bhcn,bhdn->bhcd", q, k),
            dim=-1,
        )
        fused = torch.einsum("bhcn,bhcd->bhdn", v, attn).reshape(b, c, h, w)
        out = x_opt + self.proj(fused)
        out = out + self.mlp(out)
        return out


class DBCRNet(nn.Module):
    def __init__(
        self,
        in_opt=13,
        in_sar=2,
        widths=(22, 44, 88, 176),
        enc_blocks=(1, 1, 1, 28),
        dec_blocks=(1, 1, 1, 1),
        heads=(1, 1, 2, 4),
        time_dim=128,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.time_to_channels = nn.ModuleList(
            [nn.Linear(time_dim, c) for c in widths]
        )

        self.opt_stem = nn.Conv2d(in_opt, widths[0], 1)
        self.sar_stem = nn.Conv2d(in_sar, widths[0], 1)

        self.opt_enc = nn.ModuleList()
        self.sar_enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.fuse = nn.ModuleList()

        for i, c in enumerate(widths):
            self.opt_enc.append(nn.Sequential(*[NAFBlock(c) for _ in range(enc_blocks[i])]))
            self.sar_enc.append(nn.Sequential(*[NAFBlock(c) for _ in range(enc_blocks[i])]))
            self.fuse.append(SFBlock(c, heads[i]))
            if i < len(widths) - 1:
                self.downs.append(nn.Conv2d(c, widths[i + 1], 2, stride=2))

        self.mid = nn.Sequential(*[NAFBlock(widths[-1]) for _ in range(1)])

        self.ups = nn.ModuleList()
        self.opt_dec = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2))
            self.opt_dec.append(nn.Sequential(*[NAFBlock(widths[i - 1]) for _ in range(dec_blocks[len(widths) - 1 - i])]))

        self.head = nn.Conv2d(widths[0], in_opt, 1)

    def forward(self, x_t, t, z):
        b = x_t.size(0)
        t = t.float()
        t_embed = self.time_mlp(t)

        opt = self.opt_stem(x_t)
        sar = self.sar_stem(z)

        skips = []
        for i in range(len(self.opt_enc)):
            bias = self.time_to_channels[i](t_embed).view(b, -1, 1, 1)
            opt = opt + bias
            opt = self.opt_enc[i](opt)
            sar = self.sar_enc[i](sar)
            opt = self.fuse[i](opt, sar)
            skips.append(opt)
            if i < len(self.downs):
                opt = self.downs[i](opt)
                sar = self.downs[i](sar)

        opt = self.mid(opt)

        for i, up in enumerate(self.ups):
            opt = up(opt)
            skip = skips[-(i + 2)]
            opt = opt + skip
            opt = self.opt_dec[i](opt)

        return self.head(opt)


# Sentinel-2 optical channel count used by spectral mean-reverting bridges.
NUM_S2_BANDS = 13


def alpha_schedule(t, T):
    """Original DB-CR sinusoidal bridge schedule. alpha(0)=0, alpha(T)=1."""
    return torch.sin((t / T) * math.pi / 2)


def _as_rate_tensor(rate, *, dtype, device):
    """Normalize rate to a 0-D (scalar) or 1-D (spectral) float tensor."""
    if torch.is_tensor(rate):
        rate_t = rate.detach().to(dtype=dtype, device=device).float().reshape(-1)
    elif isinstance(rate, (list, tuple)):
        rate_t = torch.as_tensor(list(rate), dtype=dtype, device=device).float().reshape(-1)
    else:
        rate_t = torch.as_tensor(float(rate), dtype=dtype, device=device).float().reshape(-1)

    if rate_t.numel() == 1:
        if float(rate_t.item()) <= 0.0:
            raise ValueError(f"mean_reversion_rate must be > 0, got {float(rate_t.item())}")
        return rate_t.reshape(())  # 0-D scalar tensor

    if rate_t.numel() != NUM_S2_BANDS:
        raise ValueError(
            f"spectral mean_reversion rates must have length {NUM_S2_BANDS}, "
            f"got {rate_t.numel()}"
        )
    if bool((rate_t <= 0).any().item()):
        raise ValueError(
            f"all spectral mean_reversion rates must be > 0, got {rate_t.tolist()}"
        )
    return rate_t  # [13]


def mean_reverting_alpha_schedule(t, T, rate=3.0):
    """Deterministic mean-reverting bridge schedule with exact endpoints.

    Uses normalized time s = t / T and
        alpha(t) = (1 - exp(-rate * s)) / (1 - exp(-rate))
    so that alpha(0) = 0 and alpha(T) = 1 exactly (up to floating-point).

    Args:
        t: scalar or tensor timesteps (same device as returned alpha).
        T: total diffusion steps (positive scalar).
        rate: mean-reversion rate; must be > 0. Either a scalar or a length-13
            vector (list/tuple/1D tensor) for spectrally anisotropic schedules.

    Returns:
        Scalar path (scalar rate):
            * t shaped [B] -> alpha shaped [B]
            * t scalar / 0-D -> alpha scalar / 0-D
        Spectral path (length-13 rate):
            * t shaped [B] -> alpha shaped [B, 13]
            * t scalar / 0-D -> alpha shaped [13]
    """
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32)
    t = t.float()

    rate_t = _as_rate_tensor(rate, dtype=t.dtype, device=t.device)

    # Scalar path: preserve historical broadcasting (rate is 0-D).
    if rate_t.ndim == 0:
        s = t / float(T)
        # expm1 improves accuracy for small rate; mathematically equal to
        # (1 - exp(-rate * s)) / (1 - exp(-rate)).
        return torch.expm1(-rate_t * s) / torch.expm1(-rate_t)

    # Spectral path: rate_t is [13].
    s = t / float(T)
    if t.ndim == 0:
        # s scalar, rate [13] -> alpha [13]
        return torch.expm1(-rate_t * s) / torch.expm1(-rate_t)
    # s [B] -> [B, 1], rate [13] -> alpha [B, 13]
    s = s.reshape(-1, 1)
    return torch.expm1(-rate_t.unsqueeze(0) * s) / torch.expm1(-rate_t)


def reshape_alpha_for_broadcast(alpha, *, as_channels: bool = False):
    """Reshape schedule output for broadcasting against [B, C, H, W].

    Supported mappings:
      * 0-D            -> [1, 1, 1, 1]     (scalar inference)
      * 1-D [B]        -> [B, 1, 1, 1]     (scalar train; as_channels=False)
      * 1-D [C]        -> [1, C, 1, 1]     (spectral inference; as_channels=True)
      * 2-D [B, C]     -> [B, C, 1, 1]     (spectral train)

    ``as_channels`` disambiguates 1-D tensors: use True only for spectral
    alpha produced from a scalar timestep (inference / reverse).
    """
    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(alpha)
    if alpha.ndim == 0:
        return alpha.reshape(1, 1, 1, 1)
    if alpha.ndim == 2:
        return alpha.unsqueeze(-1).unsqueeze(-1)
    if alpha.ndim == 1:
        if as_channels:
            return alpha.reshape(1, -1, 1, 1)
        return alpha.reshape(-1, 1, 1, 1)
    raise ValueError(
        f"reshape_alpha_for_broadcast expects 0-D, 1-D, or 2-D alpha, got shape {tuple(alpha.shape)}"
    )


def get_alpha_schedule(
    bridge_schedule="original",
    mean_reversion_rate=3.0,
    spectral_mean_reversion_rates=None,
):
    """Return the alpha(t, T) callable for the selected bridge schedule.

    If ``spectral_mean_reversion_rates`` is provided, it must be a length-13
    sequence of positive rates and ``bridge_schedule`` must be
    ``\"mean_reverting\"``. The spectral vector then overrides the scalar
    ``mean_reversion_rate``. When absent, scalar behavior is unchanged.
    """
    name = str(bridge_schedule).lower().replace("-", "_")
    if spectral_mean_reversion_rates is not None:
        if name != "mean_reverting":
            raise ValueError(
                "spectral_mean_reversion_rates requires bridge_schedule='mean_reverting'"
            )
        rates = list(spectral_mean_reversion_rates)
        if len(rates) != NUM_S2_BANDS:
            raise ValueError(
                f"spectral_mean_reversion_rates must have length {NUM_S2_BANDS}, got {len(rates)}"
            )
        if any(float(r) <= 0.0 for r in rates):
            raise ValueError(
                f"all spectral_mean_reversion_rates must be > 0, got {rates}"
            )

        def spectral_schedule(t, T):
            return mean_reverting_alpha_schedule(t, T, rate=rates)

        return spectral_schedule

    if name == "original":
        return alpha_schedule
    if name == "mean_reverting":
        rate = float(mean_reversion_rate)

        def schedule(t, T):
            return mean_reverting_alpha_schedule(t, T, rate=rate)

        return schedule
    raise ValueError(
        f"Unknown bridge_schedule '{bridge_schedule}'. "
        "Expected 'original' or 'mean_reverting'."
    )
