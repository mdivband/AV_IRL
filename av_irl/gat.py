from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from imitation.rewards.reward_nets import RewardNet



class GATLayer(nn.Module):
    """Single-head GAT (attention over pair-wise nodes)."""
    def __init__(self, in_features: int, out_features: int, alpha: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.attn_fc = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h: torch.Tensor) -> torch.Tensor: # h:(B,N,F)
        B, N, _ = h.shape
        Wh = self.W(h)  # (B,N,H)

        a_input = torch.cat(
            [Wh.unsqueeze(2).repeat(1, 1, N, 1),
             Wh.unsqueeze(1).repeat(1, N, 1, 1)],
            dim=-1)  # (B,N,N,2H)

        e = self.leakyrelu(self.attn_fc(a_input).squeeze(-1))  # (B,N,N)
        attn = F.softmax(e, dim=-1)
        h_prime = torch.matmul(attn, Wh)
        return h_prime.mean(dim=1)


class GATRewardNet(RewardNet):
    def __init__(
        self,
        obs_space: gym.spaces.Box,
        act_space: gym.spaces.Space,
        feature_dim: int = 5,
        hidden_gat: int = 32,
        hidden_mlp: int = 64,
        use_running_norm: bool = True,
    ):
        super().__init__(obs_space, act_space)

        obs_dim = int(np.prod(obs_space.shape))
        if obs_dim % feature_dim != 0:
            raise ValueError(f"obs_dim {obs_dim} not divisible by feature_dim {feature_dim}")

        self.F = feature_dim
        self.N = obs_dim // self.F

        if isinstance(act_space, gym.spaces.Discrete):
            self.discrete = True
            self.act_dim = act_space.n
        elif isinstance(act_space, gym.spaces.Box):
            self.discrete = False
            self.act_dim = int(np.prod(act_space.shape))
        else:
            raise TypeError(f"Unsupported action space: {act_space}")

        self.use_running_norm = use_running_norm
        if use_running_norm:
            from imitation.util.networks import RunningNorm
            self.norm = RunningNorm(obs_dim)

        self.gat  = GATLayer(self.F, hidden_gat)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden_gat + self.act_dim, hidden_mlp),
            nn.ReLU(),
            nn.Linear(hidden_mlp, 1),
        )
        

    def _obs_to_graph(self, flat_obs: torch.Tensor) -> torch.Tensor:
        B = flat_obs.size(0)
        return flat_obs.view(B, self.N, self.F)

    def _act_to_vec(self, acts: torch.Tensor, B: int, device) -> torch.Tensor:
        if self.discrete:
            
            if acts.shape == (B, self.act_dim):
                return acts.float().to(device)
            
            if acts.ndim == 1 or (acts.ndim == 2 and acts.shape[1] == 1):
                if acts.ndim > 1:
                    acts = acts.squeeze(-1)
                acts = acts.long().to(device)
                onehot = F.one_hot(acts, num_classes=self.act_dim).float()
                result = onehot.view(B, -1)
                return result
            
            else:
                result = acts.view(B, -1).float().to(device)
                return result
        else:
            result = acts.view(B, -1).float().to(device)
            return result


    def forward(self, obs: torch.Tensor, acts: torch.Tensor, *_, **__):
        B = obs.shape[0]
        device = obs.device

        flat_obs = obs.view(B, -1) if obs.ndim == 3 else obs
        if self.use_running_norm:
            flat_obs = self.norm(flat_obs)

        graph_in = self._obs_to_graph(flat_obs)  # (B,N,F)
        graph_repr = self.gat(graph_in)  # (B,H)

        act_vec = self._act_to_vec(acts, B, device)  # (B, act_dim)
        
        
        if torch.isnan(graph_repr).any() or torch.isnan(act_vec).any():
            print(f"Warning: NaN detected in graph_repr or act_vec")
        
        cat = torch.cat([graph_repr, act_vec], dim=-1)  # (B, H+act_dim)
        
        
        expected_dim = self.gat.W.out_features + self.act_dim
        if cat.shape[1] != expected_dim:
            print(f"Dimension mismatch details:")
            print(f"  graph_repr actual dim: {graph_repr.shape[1]}")
            print(f"  act_vec actual dim: {act_vec.shape[1]}")
            print(f"  total actual dim: {cat.shape[1]}")
            print(f"  expected graph dim: {self.gat.W.out_features}")
            print(f"  expected act dim: {self.act_dim}")
            print(f"  expected total dim: {expected_dim}")
            raise RuntimeError(f"Dimension mismatch: got {cat.shape[1]}, expected {expected_dim}")
            
        reward = self.mlp(cat)  # (B,1)
        return reward.squeeze(-1)

    def reward(self, obs, acts, **kwargs):
        return self.forward(obs, acts)
