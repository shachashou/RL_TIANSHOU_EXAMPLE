import gymnasium as gym
import tianshou as ts
import torch

from POLICY.DQN import get_policy

ENV_ID = 'MountainCar-v0'

POLICY_TRAIN_EPS = 0.5  # 探索率
POLICY_TEST_EPS = POLICY_TRAIN_EPS / 10
POLICY_LR = 1e-3  # 学习率
POLICY_DISCOUNT_FACTOR = 0.99  # 远见程度， 越接近1越有远见
POLICY_ESTIMATION_STEP = 3  # 往后看多步答案 ，简单任务就是3,迷宫的话，可能是10或者15
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
    # reset_before_collect=True,
)
print(f"本次收集的 Episode 奖励为: {result_stats['rew']}")
