import gymnasium as gym
import walker2d
import hopper
import time

import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure



xml_file = "/home/master-andreas/gym-mjc-v5-model-validation/walker2d_v5.xml"
eval_env = gym.wrappers.TimeLimit(walker2d.Walker2dEnv(xml_file=xml_file, render_mode='human'), max_episode_steps=1000)

model = TD3.load(path='/home/master-andreas/gym-mjc-v5-model-validation/results/walker2d_v5/run_0/best_model.zip', env=eval_env, device='cpu')

vec_env = model.get_env()
obs = vec_env.reset()
#obs = eval_env.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    #obs, reward, terminal, truncated, info = eval_env.step(action)
    obs, reward, done, info = vec_env.step(action)
    time.sleep(0.010)

    #vec_env.render("human")

