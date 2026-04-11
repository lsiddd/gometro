import numpy as np
from sb3_contrib import MaskablePPO
from env import MiniMetroEnv

print("Loading model...")
model = MaskablePPO.load("checkpoints/best_model.zip")
env = MiniMetroEnv(port=8124, managed=True)

scores = []
for ep in range(5):
    obs, info = env.reset()
    done = False
    ep_reward = 0
    steps = 0
    while not done and steps < 2000:
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))
        ep_reward += reward
        steps += 1
    scores.append(info.get('score', 0))
    print(f"Episode {ep} finished: steps={steps}, score={info.get('score', 0)}, passengers={info.get('passengers_delivered', 0)}")

print(f"Average score: {np.mean(scores)}")
env.close()
