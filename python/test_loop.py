from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env import MiniMetroEnv, MiniMetroFrameStack
from models import MetroFeatureExtractor
from policy import MetroPolicy
from train import ConditionalMaskRolloutBuffer


def test_training_loop_with_conditional_mask():
    env = DummyVecEnv([lambda: MiniMetroEnv(port=8811, managed=True)])
    env.action_masks = lambda: [env.envs[0].action_masks()]
    env = MiniMetroFrameStack(env, n_stack=4)
    try:
        model = MaskablePPO(
            MetroPolicy,
            env,
            n_steps=64,
            batch_size=64,
            n_epochs=1,
            rollout_buffer_class=ConditionalMaskRolloutBuffer,
            policy_kwargs={
                "features_extractor_class": MetroFeatureExtractor,
                "net_arch": [128, 128],
            },
            verbose=0,
        )

        model.learn(total_timesteps=64)
    finally:
        env.close()
