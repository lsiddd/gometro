import sys
sys.path.append('python')
from env import MiniMetroEnv
import numpy as np

print("Initializing env...")
env = MiniMetroEnv(port=8123, managed=True)
obs, info = env.reset()
print(f"Obs shape: {obs.shape}")

print("Taking random actions...")
for i in range(10):
    mask = env.action_masks()
    valid_actions = np.where(mask)[0]
    action = np.random.choice(valid_actions)
    obs, reward, done, trunc, info = env.step(action)
    print(f"Step {i}: reward={reward:.3f}, done={done}, score={info.get('score', 0)}")
    if done:
        break
        
env.close()
print("Done")
