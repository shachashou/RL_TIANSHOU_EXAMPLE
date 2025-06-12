import tianshou as ts
import torch

from ENV.atari import AtariWrapper
from POLICY.RainbowDQN import get_policy

# 环境参数
ENV_ID = 'ALE/AirRaid-v5'
ENV_TRAIN_NUM = 10
ENV_TEST_NUM = 10
REWARD_THRESHOLD = 1500

POLICY_LR = 1e-4
POLICY_DISCOUNT_FACTOR = 0.99
POLICY_ESTIMATION_STEP = 3
POLICY_TARGET_UPDATE_FREQ = 8000

POLICY_NUM_ATOMS = 51
POLICY_V_MIN = 0.0
POLICY_V_MAX = 200.0
POLICY_NOISY_STD = 0.5
POLICY_IS_DUELING = True
POLICY_IS_NOISY = True

PER_ALPHA = 0.5
PER_BETA = 0.4
PER_BETA_FINAL = 1.0
PER_BETA_ANNEAL_STEPS = 20_000_000  # 在多少步内完成beta退火

COLLECTOR_SIZE = 300_000
TRAINER_MAX_EPOCH = 200
TRAINER_STEP_PER_EPOCH = 100_000
TRAINER_STEP_PER_COLLECT = 4000
TRAINER_UPDATE_PER_STEP = 1.0 / 4.0  # 每4步更新一次
TRAINER_EPISODE_PER_TEST = ENV_TEST_NUM
TRAINER_BATCH_SIZE = 32

env = AtariWrapper(ENV_ID)
train_envs = ts.env.DummyVectorEnv([lambda: AtariWrapper(ENV_ID) for _ in range(ENV_TRAIN_NUM)])
test_envs = ts.env.DummyVectorEnv([lambda: AtariWrapper(ENV_ID) for _ in range(ENV_TEST_NUM)])

policy = get_policy(
    env,
    POLICY_LR,
    POLICY_DISCOUNT_FACTOR,
    POLICY_NUM_ATOMS,
    POLICY_V_MIN,
    POLICY_V_MAX,
    POLICY_ESTIMATION_STEP,
    POLICY_TARGET_UPDATE_FREQ,
    POLICY_NOISY_STD,
    POLICY_IS_DUELING,
    POLICY_IS_NOISY,
)
try:
    policy.load_state_dict(torch.load(f"STATE/{ENV_ID}.pth"))
except Exception as e:
    print(e)
policy.to("cuda")

buffer = ts.data.PrioritizedVectorReplayBuffer(
    total_size=COLLECTOR_SIZE,
    buffer_num=ENV_TRAIN_NUM,
    alpha=PER_ALPHA,
    beta=PER_BETA
)
train_collector = ts.data.Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)


def train_callback(epoch, env_step):
    """在每个训练步骤中被调用，用于更新 PER 的 beta 值。"""
    if env_step <= PER_BETA_ANNEAL_STEPS:
        # 线性增加 beta 的值
        beta = PER_BETA + (PER_BETA_FINAL - PER_BETA) * (env_step / PER_BETA_ANNEAL_STEPS)
    else:
        # 退火结束后，beta 保持最终值
        beta = PER_BETA_FINAL
    # 将计算出的 beta 值设置到 buffer 中
    buffer.set_beta(beta)


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
    train_fn=train_callback,
    stop_fn=lambda mean_rewards: mean_rewards >= REWARD_THRESHOLD
).run()

torch.save(policy.state_dict(), f"STATE/{ENV_ID}.pth")
print(f'Finished training! Use {result}')
