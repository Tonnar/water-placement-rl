"""Neural networks shared by the agents.

Every network reads the same state: the density grid, all water positions
and (for continuous control) the index of the focus water. The core
:class:`StateEncoder` combines three views of it:

* a **local density crop** around each water, sampled with trilinear
  interpolation, so the policy can see which direction the nearest density
  peak lies in (globally pooled density features cannot tell it that);
* a **global density token** from a strided 3D CNN;
* **water tokens** from an MLP over coordinates scaled by the box size.

Water tokens attend over ``[global token, water tokens]`` with multi-head
attention. Networks are size-agnostic: the same weights work for any number
of waters, so an agent can be trained across proteins.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


def local_density_crops(
    density: torch.Tensor, centres: torch.Tensor, half_extent: float, voxel_size: float, size: int
) -> torch.Tensor:
    """Sample (size^3) density patches centred on each point.

    Args:
        density: (B, G, G, G) grid indexed [x, y, z].
        centres: (B, M, 3) positions in Angstrom.
    Returns:
        (B, M, size, size, size) patches; zero outside the grid.
    """
    b, m, _ = centres.shape
    offsets = (torch.arange(size, device=centres.device, dtype=centres.dtype) - (size - 1) / 2) * voxel_size
    ox, oy, oz = torch.meshgrid(offsets, offsets, offsets, indexing="ij")
    offset_grid = torch.stack([ox, oy, oz], dim=-1)  # (s, s, s, 3) in x, y, z
    pts = centres[:, :, None, None, None, :] + offset_grid  # (B, M, s, s, s, 3)
    # grid_sample wants normalised (W, H, D) = (z, y, x) order in [-1, 1].
    norm = (pts / half_extent).flip(-1).reshape(b, m * size, size, size, 3)
    out = F.grid_sample(
        density.unsqueeze(1), norm, mode="bilinear", padding_mode="zeros", align_corners=False
    )  # (B, 1, M*s, s, s)
    return out.view(b, m, size, size, size)


class StateEncoder(nn.Module):
    """Encode (density, waters) into one feature vector per water."""

    def __init__(
        self,
        half_extent: float,
        voxel_size: float,
        embed_dim: int = 128,
        num_heads: int = 4,
        local_size: int = 9,
    ):
        super().__init__()
        self.half_extent = float(half_extent)
        self.voxel_size = float(voxel_size)
        self.local_size = local_size

        self.global_cnn = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim),
        )
        self.local_cnn = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * local_size**3, embed_dim),
        )
        # Coordinates are divided by the box half-width rather than passed
        # through LayerNorm(3), which would standardise x, y, z against each
        # other and throw away where the water actually is.
        self.point = mlp(3, embed_dim, embed_dim)
        self.fuse = nn.Linear(2 * embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(
        self, density: torch.Tensor, waters: torch.Tensor, query_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            density: (B, G, G, G).
            waters: (B, N, 3).
            query_idx: optional (B,) long. If given, only that water is
                encoded and the result is (B, E); otherwise (B, N, E).
        """
        b, n, _ = waters.shape
        point_tokens = self.point(waters / self.half_extent)  # (B, N, E)
        global_token = self.global_cnn(density.unsqueeze(1)).unsqueeze(1)  # (B, 1, E)
        context = torch.cat([global_token, point_tokens], dim=1)

        if query_idx is None:
            centres, q_points = waters, point_tokens
        else:
            gather = query_idx.view(b, 1, 1).expand(b, 1, 3)
            centres = waters.gather(1, gather)
            q_points = point_tokens.gather(1, query_idx.view(b, 1, 1).expand(b, 1, self.embed_dim))
        m = centres.shape[1]
        crops = local_density_crops(density, centres, self.half_extent, self.voxel_size, self.local_size)
        local = self.local_cnn(crops.reshape(b * m, 1, *crops.shape[2:])).view(b, m, -1)
        query = self.fuse(torch.cat([q_points, local], dim=-1))
        attended, _ = self.attn(query, context, context, need_weights=False)
        out = self.norm(query + attended)
        return out.squeeze(1) if query_idx is not None else out


def _init_output(layer: nn.Linear, scale: float = 3e-3) -> None:
    nn.init.uniform_(layer.weight, -scale, scale)
    nn.init.uniform_(layer.bias, -scale, scale)


class Actor(nn.Module):
    """Deterministic policy for TD3: state -> action in [-1, 1]^action_dim."""

    def __init__(self, encoder: StateEncoder, action_dim: int, hidden: int = 256):
        super().__init__()
        self.encoder = encoder
        self.head = mlp(encoder.embed_dim, hidden, action_dim)
        _init_output(self.head[-1])

    def forward(self, density, waters, focus) -> torch.Tensor:
        return torch.tanh(self.head(self.encoder(density, waters, focus)))


class TwinCritic(nn.Module):
    """Two Q-heads on a shared encoder (clipped double-Q learning)."""

    def __init__(self, encoder: StateEncoder, action_dim: int, hidden: int = 256):
        super().__init__()
        self.encoder = encoder
        e = encoder.embed_dim
        self.action_proj = nn.Linear(action_dim, e)
        self.q1 = mlp(2 * e, hidden, 1)
        self.q2 = mlp(2 * e, hidden, 1)

    def forward(self, density, waters, focus, action):
        x = torch.cat([self.encoder(density, waters, focus), F.relu(self.action_proj(action))], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, density, waters, focus, action):
        x = torch.cat([self.encoder(density, waters, focus), F.relu(self.action_proj(action))], dim=-1)
        return self.q1(x).squeeze(-1)


class GaussianActorCritic(nn.Module):
    """PPO policy and value function on a shared encoder.

    The policy is a diagonal Gaussian with a state-independent learned
    log-std; actions are sampled unbounded and clipped by the environment.
    """

    def __init__(self, encoder: StateEncoder, action_dim: int, hidden: int = 256, init_log_std: float = -0.5):
        super().__init__()
        self.encoder = encoder
        self.mean_head = mlp(encoder.embed_dim, hidden, action_dim)
        self.value_head = mlp(encoder.embed_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))
        _init_output(self.mean_head[-1], 1e-2)

    def forward(self, density, waters, focus):
        h = self.encoder(density, waters, focus)
        mean = self.mean_head(h)
        dist = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        return dist, self.value_head(h).squeeze(-1)


class QNetwork(nn.Module):
    """Q-values for every (water, move) pair: output shape (B, N * n_moves).

    With ``dueling=True`` the head splits into a state value V(s) and
    advantages A(s, a), combined as ``V + A - mean(A)``.
    """

    def __init__(self, encoder: StateEncoder, n_moves: int = 7, hidden: int = 256, dueling: bool = True):
        super().__init__()
        self.encoder = encoder
        self.n_moves = n_moves
        self.dueling = dueling
        self.advantage = mlp(encoder.embed_dim, hidden, n_moves)
        if dueling:
            self.value = mlp(encoder.embed_dim, hidden, 1)

    def forward(self, density, waters) -> torch.Tensor:
        h = self.encoder(density, waters)  # (B, N, E)
        adv = self.advantage(h).flatten(1)  # (B, N * n_moves)
        if not self.dueling:
            return adv
        value = self.value(h.mean(dim=1))  # (B, 1)
        return value + adv - adv.mean(dim=1, keepdim=True)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


__all__ = [
    "StateEncoder",
    "Actor",
    "TwinCritic",
    "GaussianActorCritic",
    "QNetwork",
    "local_density_crops",
    "count_parameters",
]
