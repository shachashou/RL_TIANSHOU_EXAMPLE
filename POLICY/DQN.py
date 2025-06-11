import numpy as np
import tianshou as ts
from torch import nn

from settings import *


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=DEVICE)
        else:
            obs.to(DEVICE)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


def get_policy(env, lr, discount_factor, estimation_step, target_update_freq):
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape)
    net.to("cuda")
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    policy = ts.policy.DQNPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=discount_factor,
        estimation_step=estimation_step,
        target_update_freq=target_update_freq
    )
    return policy
