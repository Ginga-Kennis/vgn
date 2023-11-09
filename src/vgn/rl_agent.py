from stable_baselines3 import PPO

from vgn.rl_env import Env

env = Env()
env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)
    print(_, reward,done,truncated)
    if done == True or truncated == True:
        break