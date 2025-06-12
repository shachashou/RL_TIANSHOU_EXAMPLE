import numpy as np
from tianshou.policy import RainbowPolicy
from tianshou.utils.net.discrete import NoisyLinear
from torch import nn

from settings import *


class SimpleRainbowCNN(nn.Module):
    def __init__(self, c, h, w, action_shape, num_atoms, noisy_std, is_dueling=True, is_noisy=True):
        super().__init__()
        self.action_num = int(np.prod(action_shape))
        self.num_atoms = num_atoms
        self._is_dueling = is_dueling

        self.feature_net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        with torch.no_grad():
            feature_dim = self.feature_net(torch.zeros(1, c, h, w)).shape[1]

        def linear_layer(in_features, out_features):
            if is_noisy:
                return NoisyLinear(in_features, out_features, noisy_std)
            else:
                return nn.Linear(in_features, out_features)

        self.advantage_head = nn.Sequential(
            linear_layer(feature_dim, 512),
            nn.ReLU(inplace=True),
            linear_layer(512, self.action_num * self.num_atoms)
        )

        if self._is_dueling:
            self.value_head = nn.Sequential(
                linear_layer(feature_dim, 512),
                nn.ReLU(inplace=True),
                linear_layer(512, self.num_atoms)
            )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=DEVICE, dtype=torch.float32)
        else:
            obs = obs.to(DEVICE, dtype=torch.float32)

        obs = obs / 255.0

        features = self.feature_net(obs)

        adv = self.advantage_head(features)

        adv = adv.view(-1, self.action_num, self.num_atoms)

        if self._is_dueling:
            val = self.value_head(features)
            val = val.view(-1, 1, self.num_atoms)
            logits = val + adv - adv.mean(dim=1, keepdim=True)
        else:
            logits = adv

        probs = logits.softmax(dim=2)

        return probs, state


def get_policy(
        env,
        lr,
        discount_factor,
        num_atoms,
        v_min,
        v_max,
        estimation_step,
        target_update_freq,
        noisy_std,
        is_dueling,
        is_noisy,
):
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    net = SimpleRainbowCNN(
        *state_shape,
        action_shape,
        num_atoms,
        noisy_std,

        is_dueling=is_dueling,
        is_noisy=is_noisy,
    )
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    policy = RainbowPolicy(
        model=net,
        optim=optim,
        discount_factor=discount_factor,
        action_space=env.action_space,
        num_atoms=num_atoms,
        v_min=v_min,
        v_max=v_max,
        estimation_step=estimation_step,
        target_update_freq=target_update_freq,
    )
    return policy
