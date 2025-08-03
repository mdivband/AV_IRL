import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from imitation.rewards.reward_nets import RewardNet
from imitation.util.networks import RunningNorm


class SlotAttention(nn.Module):
    """
    Original Slot Attention (Locatello et al., 2020) – single-head version.
    """

    def __init__(self, num_slots: int = 4, dim: int = 32, iters: int = 3):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x = self.norm_inputs(x)

        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand_as(mu)
        slots = mu + torch.randn_like(mu) * sigma

        for _ in range(self.iters):
            slots_prev = slots

            q = self.to_q(self.norm_slots(slots))
            k = self.to_k(x)
            v = self.to_v(x)

            attn_logits = torch.einsum("bkd,bnd->bkn", q, k) * self.scale
            attn = attn_logits.softmax(dim=-1) + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bkn,bnd->bkd", attn, v)

            slots = self.gru(
                updates.reshape(-1, self.dim),
                slots_prev.reshape(-1, self.dim)
            )
            slots = slots.view(B, self.num_slots, self.dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots.reshape(B, self.num_slots * self.dim)


class SlotRewardNet(RewardNet):
    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        *,
        num_slots: int = 4,
        slot_dim: int = 32,
        hidden: int = 64,
        feature_dim: int = 5,
        iters: int = 3,
        use_running_norm: bool = True,
    ):
        super().__init__(obs_space, act_space)

        obs_dim = int(np.prod(obs_space.shape))
        assert obs_dim % feature_dim == 0, (
            f"Observation length {obs_dim} must be multiple of feature_dim={feature_dim}"
        )
        self.N = obs_dim // feature_dim
        self.F = feature_dim

        if isinstance(act_space, gym.spaces.Box):
            act_dim = int(np.prod(act_space.shape))
        elif isinstance(act_space, gym.spaces.Discrete):
            act_dim = 1
        else:
            raise TypeError(f"Unsupported action space {act_space}")

        self.encoder = SlotAttention(
            num_slots=num_slots, dim=slot_dim, iters=iters
        )

        mlp_in = num_slots * slot_dim + act_dim
        layers = [nn.Linear(mlp_in, hidden), nn.ReLU(), nn.Linear(hidden, 1)]
        self.mlp = nn.Sequential(*layers)

        self.use_running_norm = use_running_norm
        if use_running_norm:
            from imitation.util.networks import RunningNorm
            self.norm = RunningNorm(mlp_in)

    def _process(self, obs: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        x = obs.view(B, self.N, self.F)
        slots = self.encoder(x)
        feat = torch.cat([slots, acts], dim=-1)
        if self.use_running_norm:
            feat = self.norm(feat)
        return feat

    def forward(self, obs: torch.Tensor, acts: torch.Tensor, **kwargs):
        return self.mlp(self._process(obs, acts))

    def reward(self, obs: torch.Tensor, acts: torch.Tensor, **kwargs):
        return self.forward(obs, acts)

    def logits(self, obs: torch.Tensor, acts: torch.Tensor, **kwargs):
        return self.forward(obs, acts)

