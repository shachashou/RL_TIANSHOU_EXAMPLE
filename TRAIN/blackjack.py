import tianshou as ts
import torch

from ENV.blackjack import BlackjackEnvWrapper
from POLICY.DQN import get_policy

ENV_ID = 'Blackjack-v1'
ENV_TRAIN_NUM = 10
ENV_TEST_NUM = 100

COLLECTOR_SIZE = 20000

TRAINER_MAX_EPOCH = 100
TRAINER_STEP_PER_EPOCH = 5000  # 一轮做题数
TRAINER_STEP_PER_COLLECT = 1000  # 多少题学习一次
TRAINER_UPDATE_PER_STEP = 0.1  # 反思强度
TRAINER_EPISODE_PER_TEST = 100
TRAINER_BATCH_SIZE = 64
TRAINER_EPS_TRAIN_START = 1.0  # 训练开始时的探索率 (完全随机)
TRAINER_EPS_TRAIN_END = 0.05  # 训练结束时的最低探索率 (接近贪心)
TRAINER_EPS_DECAY_STEPS = 100000

POLICY_TEST_EPS = 0.005
POLICY_LR = 1e-3  # 学习率
POLICY_DISCOUNT_FACTOR = 0.99  # 远见程度， 越接近1越有远见
POLICY_ESTIMATION_STEP = 5  # 往后看多步答案 ，简单任务就是3,迷宫的话，可能是10或者15
POLICY_TARGET_UPDATE_FREQ = 500

env = BlackjackEnvWrapper()
train_envs = ts.env.DummyVectorEnv([lambda: BlackjackEnvWrapper() for _ in range(ENV_TRAIN_NUM)])
test_envs = ts.env.DummyVectorEnv([lambda: BlackjackEnvWrapper() for _ in range(ENV_TEST_NUM)])

policy = get_policy(
    env,
    POLICY_LR,
    POLICY_DISCOUNT_FACTOR,
    POLICY_ESTIMATION_STEP,
    POLICY_TARGET_UPDATE_FREQ,
)
policy.to("cuda")

train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(COLLECTOR_SIZE, ENV_TRAIN_NUM), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

result = ts.trainer.OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=TRAINER_MAX_EPOCH,
    step_per_epoch=TRAINER_STEP_PER_EPOCH,
    step_per_collect=TRAINER_STEP_PER_COLLECT,
    update_per_step=TRAINER_UPDATE_PER_STEP,
    episode_per_test=TRAINER_EPISODE_PER_TEST,
    batch_size=TRAINER_BATCH_SIZE,
    train_fn=lambda epoch, env_step: policy.set_eps(
        TRAINER_EPS_TRAIN_START - min(1.0, env_step / TRAINER_EPS_DECAY_STEPS) * (TRAINER_EPS_TRAIN_START - TRAINER_EPS_TRAIN_END)
    ),
    test_fn=lambda epoch, env_step: policy.set_eps(POLICY_TEST_EPS),
    stop_fn=lambda mean_rewards: mean_rewards >= env.reward_threshold
).run()

torch.save(policy.state_dict(), f"STATE/{ENV_ID}.pth")

print(f'Finished training! Use {result}')
