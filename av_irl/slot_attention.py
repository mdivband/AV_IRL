import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from imitation.rewards.reward_nets import RewardNet
from imitation.util.networks import RunningNorm


class SlotAttention(nn.Module):
    def __init__(self, num_slots=4, dim=32, iters=3):
        super().__init__()
        self.num_slots, self.dim, self.iters = num_slots, dim, iters
        self.scale = dim ** -0.5
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.n_x = nn.LayerNorm(dim)
        self.n_s = nn.LayerNorm(dim)
        self.n_ff = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, _ = x.shape
        x = self.n_x(x)
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        std = self.slots_logsigma.exp().expand_as(mu)
        slots = mu + torch.randn_like(mu) * std
        for _ in range(self.iters):
            q = self.to_q(self.n_s(slots))
            k, v = self.to_k(x), self.to_v(x)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(-1) + 1e-8
            attn = attn / attn.sum(-1, keepdim=True)
            upd = attn @ v
            slots = self.gru(
                upd.reshape(-1, self.dim), slots.reshape(-1, self.dim)
            ).view(B, self.num_slots, self.dim)
            slots = slots + self.mlp(self.n_ff(slots))
        return slots.reshape(B, -1)


class SlotRewardNet(RewardNet):
    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        num_slots=4,
        slot_dim=32,
        hidden=64,
        feature_dim=5,
        iters=3,
        use_running_norm=True,
        num_discrete_actions=5,  # Add parameter for number of discrete actions
    ):
        super().__init__(obs_space, act_space)

        obs_len = int(np.prod(obs_space.shape))
        if obs_len % feature_dim:
            raise ValueError("obs length not divisible by feature_dim")
        self.N = obs_len // feature_dim
        self.F = feature_dim

        # Determine action space properties
        self.discrete = isinstance(act_space, gym.spaces.Discrete)
        self.multidiscrete = isinstance(act_space, gym.spaces.MultiDiscrete)
        
        # Store the expected number of discrete actions
        self.num_discrete_actions = num_discrete_actions
        
        if self.discrete:
            self.act_dim = act_space.n
            # Always expect num_discrete_actions actions, each one-hot encoded
            action_vec_dim = self.act_dim * num_discrete_actions
        elif self.multidiscrete:
            # MultiDiscrete has multiple discrete actions
            self.act_dims = act_space.nvec
            self.num_discrete_actions = len(self.act_dims)
            action_vec_dim = sum(self.act_dims)
        else:
            # Continuous action space
            self.act_dim = int(np.prod(act_space.shape))
            action_vec_dim = self.act_dim

        self.proj = nn.Linear(feature_dim, slot_dim)
        self.encoder = SlotAttention(num_slots, slot_dim, iters)

        # Store these values for later use
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.hidden = hidden
        
        self.use_norm = use_running_norm
        if use_running_norm:
            self.norm = RunningNorm(obs_len)
        
        # Calculate the total feature dimension
        slots_output_dim = num_slots * slot_dim
        mlp_input_dim = slots_output_dim + action_vec_dim
        
        # Initialize MLP with the correct dimension
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, 1)
        )
        
        print(f"SlotRewardNet initialized with:")
        print(f"  Observation space: {obs_space}")
        print(f"  Action space: {act_space}")
        print(f"  Discrete: {self.discrete}, MultiDiscrete: {self.multidiscrete}")
        if self.discrete:
            print(f"  Action dimension: {self.act_dim}")
            print(f"  Expected number of discrete actions: {num_discrete_actions}")
            print(f"  Total action vector dimension: {action_vec_dim}")
        elif self.multidiscrete:
            print(f"  Action dimensions: {self.act_dims}")
        else:
            print(f"  Continuous action dimension: {self.act_dim}")
        print(f"  MLP input dimension: {mlp_input_dim}")

    def _action_vec(self, acts, B, device):
        """Convert actions to vector representation."""
        if isinstance(acts, np.ndarray):
            acts = torch.as_tensor(acts, dtype=torch.float32)
        acts = acts.to(device)
        
        if self.discrete:
            # Ensure acts is 2D
            if acts.dim() == 1:
                # Single action per sample - need to expand to expected number of actions
                # This is likely a case where only the first action is provided
                # We'll repeat it to match expected dimensions
                acts = acts.view(B, 1)
                # Expand to expected number of actions by repeating the same action
                # NOTE: This might not be the correct behavior - it depends on the environment
                # A better approach might be to pad with zeros or a default action
                if self.num_discrete_actions > 1:
                    # For now, we'll just use the first action and pad with zeros
                    # This assumes that when we get a single action, the other actions should be 0
                    acts_expanded = torch.zeros(B, self.num_discrete_actions, device=device, dtype=torch.long)
                    acts_expanded[:, 0] = acts[:, 0].long()
                    acts = acts_expanded
            elif acts.dim() == 2:
                # Check if we have the expected number of actions
                if acts.shape[1] != self.num_discrete_actions:
                    if acts.shape[1] == 1 and self.num_discrete_actions > 1:
                        # Single action provided, need to pad
                        acts_expanded = torch.zeros(B, self.num_discrete_actions, device=device, dtype=torch.long)
                        acts_expanded[:, 0] = acts[:, 0].long()
                        acts = acts_expanded
                    elif acts.shape[1] > self.num_discrete_actions:
                        # Too many actions, truncate
                        acts = acts[:, :self.num_discrete_actions]
                    else:
                        # Fewer actions than expected, pad with zeros
                        acts_expanded = torch.zeros(B, self.num_discrete_actions, device=device, dtype=torch.long)
                        acts_expanded[:, :acts.shape[1]] = acts.long()
                        acts = acts_expanded
            else:
                acts = acts.view(B, -1)
                # Ensure we have the right number of actions
                if acts.shape[1] != self.num_discrete_actions:
                    acts_expanded = torch.zeros(B, self.num_discrete_actions, device=device, dtype=torch.long)
                    acts_expanded[:, :min(acts.shape[1], self.num_discrete_actions)] = acts[:, :min(acts.shape[1], self.num_discrete_actions)].long()
                    acts = acts_expanded
            
            # Convert to long and ensure within bounds
            acts = acts.long()
            acts = acts.clamp(0, self.act_dim - 1)
            
            # Create one-hot encoding for each action dimension
            one_hots = []
            for i in range(self.num_discrete_actions):
                one_hot = F.one_hot(acts[:, i], num_classes=self.act_dim).float()
                one_hots.append(one_hot)
            
            # Concatenate all one-hot vectors
            action_vec = torch.cat(one_hots, dim=1)
            return action_vec
            
        elif self.multidiscrete:
            # Handle MultiDiscrete action space
            if acts.dim() == 1:
                acts = acts.view(B, -1)
            
            acts = acts.long()
            one_hots = []
            for i, n in enumerate(self.act_dims):
                act_i = acts[:, i].clamp(0, n - 1)
                one_hot = F.one_hot(act_i, num_classes=n).float()
                one_hots.append(one_hot)
            
            return torch.cat(one_hots, dim=1)
        else:
            # Continuous actions
            return acts.view(B, -1).float()

    def _feat(self, obs, acts):
        """Extract features from observations and actions."""
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = obs.to(next(self.parameters()).device)
        B = obs.shape[0]
        flat = obs.view(B, -1)
        
        if self.use_norm:
            flat = self.norm(flat)
        
        # Ensure the observation has the correct shape for reshaping
        if flat.shape[1] != self.N * self.F:
            raise ValueError(f"Observation dimension mismatch. Expected {self.N * self.F}, got {flat.shape[1]}")
            
        x = self.proj(flat.view(B, self.N, self.F))
        slots = self.encoder(x)
        
        # Ensure slots has the expected shape
        expected_slots_dim = self.num_slots * self.slot_dim
        if slots.shape[1] != expected_slots_dim:
            slots = slots.view(B, expected_slots_dim)
        
        action_vec = self._action_vec(acts, B, slots.device)
        
        result = torch.cat([slots, action_vec], -1)
        
        # Verify dimension matches what MLP expects
        expected_dim = self.mlp[0].in_features
        if result.shape[1] != expected_dim:
            raise RuntimeError(
                f"Feature dimension mismatch. MLP expects {expected_dim} but got {result.shape[1]}.\n"
                f"  Slots output: {slots.shape[1]} dims\n"
                f"  Action vector: {action_vec.shape[1]} dims\n"
                f"  Acts shape before processing: {acts.shape if torch.is_tensor(acts) else 'numpy array'}"
            )
        
        return result

    def forward(self, obs, acts, next_obs=None, dones=None, **kw):
        features = self._feat(obs, acts)
        return self.mlp(features).squeeze(-1)

    def reward(self, obs, acts, next_obs=None, dones=None, **kw):
        with torch.no_grad():
            return self.forward(obs, acts).cpu().numpy().astype(np.float32)

    def logits(self, obs, acts, next_obs=None, dones=None, **kw):
        return self.forward(obs, acts)
