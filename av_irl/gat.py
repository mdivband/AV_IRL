import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from imitation.rewards.reward_nets import RewardNet

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.attn_fc = nn.Linear(2 * out_features, 1, bias=False)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h):
        B, N, _ = h.size()
        Wh = self.W(h)

        a_input = torch.cat([
            Wh.unsqueeze(2).repeat(1, 1, N, 1),
            Wh.unsqueeze(1).repeat(1, N, 1, 1)
        ], dim=-1)

        e = self.leakyrelu(self.attn_fc(a_input).squeeze(-1))
        attention = F.softmax(e, dim=-1)
        h_prime = torch.matmul(attention, Wh)

        return h_prime.mean(dim=1)


class GATRewardNet(RewardNet):
    def __init__(self, obs_space: gym.spaces.Box, act_space: gym.spaces.Space,
                 in_features=5, hidden=32, out_dim=64):
        super().__init__(obs_space, act_space)

        obs_dim = obs_space.shape[-1]
        assert obs_dim % in_features == 0, "Observation shape mismatch"
        self.N = obs_dim // in_features
        self.F = in_features

        if isinstance(act_space, gym.spaces.Box):
            act_dim = int(np.prod(act_space.shape))
        elif isinstance(act_space, gym.spaces.Discrete):
            act_dim = 1
        else:
            raise TypeError("Unsupported action space")

        self.gat = GATLayer(in_features=self.F, out_features=hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs: torch.Tensor, acts: torch.Tensor, **kwargs):
        B = obs.shape[0]
        x = obs.view(B, self.N, self.F)
        graph_repr = self.gat(x)
        cat = torch.cat([graph_repr, acts], dim=-1)
        return self.mlp(cat)

    def reward(self, obs, acts, **kwargs):
        return self.forward(obs, acts)
