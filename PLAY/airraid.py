import tianshou as ts
import torch

from ENV.atari import AtariWrapper
from POLICY.RainbowDQN import get_policy

ENV_ID = 'ALE/AirRaid-v5'

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

policy = get_policy(
    AtariWrapper(ENV_ID),
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
policy.load_state_dict(torch.load(f"STATE/{ENV_ID}.pth"))
policy.to("cuda")

policy.eval()
policy.set_eps(0)

envs = ts.env.DummyVectorEnv(
    [lambda: AtariWrapper(ENV_ID, render_mode="human")]
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
