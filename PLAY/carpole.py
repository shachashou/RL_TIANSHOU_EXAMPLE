import gymnasium as gym
import tianshou as ts
import torch

from POLICY.DQN import get_policy

ENV_ID = 'CartPole-v1'

POLICY_TRAIN_EPS = 0.1
POLICY_TEST_EPS = POLICY_TRAIN_EPS / 10
POLICY_LR = 1e-3
POLICY_DISCOUNT_FACTOR = 0.9
POLICY_ESTIMATION_STEP = 3
POLICY_TARGET_UPDATE_FREQ = 320

policy = get_policy(
    gym.make(ENV_ID),
    POLICY_LR,
    POLICY_DISCOUNT_FACTOR,
    POLICY_ESTIMATION_STEP,
    POLICY_TARGET_UPDATE_FREQ,
)
policy.load_state_dict(torch.load(f"STATE/{ENV_ID}.pth"))
policy.to("cuda")

policy.eval()
policy.set_eps(POLICY_TEST_EPS)

envs = ts.env.DummyVectorEnv(
    [lambda: gym.make(ENV_ID, render_mode="human")]
)
collector = ts.data.Collector(
    policy,
    envs,
    exploration_noise=False,
)

result_stats = collector.collect(
    n_episode=1,
    render=1 / 75,
    reset_before_collect=True,
)
print(f"本次收集的 Episode 奖励为: {result_stats.returns[0]}")
