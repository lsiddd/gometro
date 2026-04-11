from stable_baselines3.common.vec_env import DummyVecEnv
from env import MiniMetroEnv
from models import MetroFeatureExtractor
from sb3_contrib import MaskablePPO

def make_env():
    return MiniMetroEnv(port=8811, managed=True)

print("Creating env...")
env = DummyVecEnv([make_env])
print("Creating model...")
model = MaskablePPO(
    "MlpPolicy",
    env,
    n_steps=64,
    batch_size=64,
    n_epochs=1,
    policy_kwargs={
        "features_extractor_class": MetroFeatureExtractor,
        "net_arch": [128, 128],
    },
    verbose=1,
)

print("Training a few steps to verify functionality...")
model.learn(total_timesteps=64)
print("Success!")
