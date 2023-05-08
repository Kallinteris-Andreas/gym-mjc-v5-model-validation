import gymnasium as gym
import walker2d
import hopper

import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
#from stable_baselines3.common.callbacks import EvalCallback
from my_eval import EvalCallback

RUNS = 10  # Number of Statistical Runs
TOTAL_TIMESTEPS = 200_000
EVAL_SEED = 1234


for run in range(5, RUNS):
    #env = gym.wrappers.TimeLimit(walker2d_v4.Walker2dEnv(), max_episode_steps=1000)
    #eval_env = gym.wrappers.TimeLimit(walker2d_v4.Walker2dEnv(), max_episode_steps=1000)
    #env = gym.wrappers.TimeLimit(walker2d_v4_fixed.Walker2dEnv(), max_episode_steps=1000)
    #eval_env = gym.wrappers.TimeLimit(walker2d_v4_fixed.Walker2dEnv(), max_episode_steps=1000)
    #env = gym.wrappers.TimeLimit(hopper_v4.HopperEnv(), max_episode_steps=1000)
    #eval_env = gym.wrappers.TimeLimit(hopper_v4.HopperEnv(), max_episode_steps=1000)
    #env = gym.wrappers.TimeLimit(hopper_v5.HopperEnv(), max_episode_steps=1000)
    #eval_env = gym.wrappers.TimeLimit(hopper_v5.HopperEnv(), max_episode_steps=1000)
    #env = gym.wrappers.TimeLimit(hopper_v5_gen.HopperEnv(), max_episode_steps=1000)
    #eval_env = gym.wrappers.TimeLimit(hopper_v5_gen.HopperEnv(), max_episode_steps=1000)
    #env = gym.wrappers.TimeLimit(hopper_v5_t2.HopperEnv(), max_episode_steps=1000)
    #eval_env = gym.wrappers.TimeLimit(hopper_v5_t2.HopperEnv(), max_episode_steps=1000)

    xml_file = "/home/master-andreas/gym-mjc-v5-model-validation/hopper.xml"
    xml_file = "/home/master-andreas/gym-mjc-v5-model-validation/hopper_new.xml"
    xml_file = "/home/master-andreas/gym-mjc-v5-model-validation/hopper_new_gen.xml"
    #xml_file = "/home/master-andreas/gym-mjc-v5-model-validation/hopper_saran_t_trans.xml"
    xml_file = "/home/master-andreas/gym-mjc-v5-model-validation/hopper_saran_t_trans2.xml"
    #xml_file = "/home/master-andreas/gym-mjc-v5-model-validation/walker2d.xml"
    #xml_file = "/home/master-andreas/gym-mjc-v5-model-validation/walker2d_fixed.xml"
    #xml_file = "/home/master-andreas/gym-mjc-v5-model-validation/walker2d_new.xml"

    env = gym.wrappers.TimeLimit(hopper.HopperEnv(xml_file=xml_file), max_episode_steps=1000)
    eval_env = gym.wrappers.TimeLimit(hopper.HopperEnv(xml_file=xml_file), max_episode_steps=1000)
    #env = gym.wrappers.TimeLimit(walker2d.Walker2dEnv(xml_file=xml_file), max_episode_steps=1000)
    #eval_env = gym.wrappers.TimeLimit(walker2d.Walker2dEnv(xml_file=xml_file), max_episode_steps=1000)

    #eval_path = 'results/walker2d_v4/run_' + str(run)
    #eval_path = 'results/walker2d_v4_fixed/run_' + str(run)
    #eval_path = 'results/hopper_v4/run_' + str(run)
    #eval_path = 'results/hopper_v5/run_' + str(run)
    #eval_path = 'results/hopper_v5_gen/run_' + str(run)
    #eval_path = 'results/hopper_v5_t2/run_' + str(run)
    #eval_path = 'results/hopper_v5_t3/run_' + str(run)
    eval_path = 'results/' + xml_file[49:-4] + '/run_' + str(run)
    #breakpoint()
    #eval_path = 'results/temp/run_' + str(run)

    eval_callback = EvalCallback(eval_env, seed=EVAL_SEED, best_model_save_path=eval_path, log_path=eval_path, n_eval_episodes=10, eval_freq=1000, deterministic=True, render=False, verbose=True)

    model = TD3("MlpPolicy", env, seed=run, verbose=1, device='cuda')
    #model.set_logger(configure(eval_path, ["stdout", "csv"]))
    model.set_logger(configure(eval_path, ["csv"]))

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

