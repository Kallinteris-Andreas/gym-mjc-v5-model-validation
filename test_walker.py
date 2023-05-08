from numpy.testing import assert_array_equal

#import hopper
import walker2d
import numpy as np

from icecream import ic

xml_file_global = "/home/master-andreas/gym-mjc-v5-model-validation/walker2d.xml"
xml_file_local = "/home/master-andreas/gym-mjc-v5-model-validation/walker2d_v5.xml"
global_env = walker2d.Walker2dEnv(exclude_current_positions_from_observation=False, xml_file=xml_file_global, render_mode=None)
local_env = walker2d.Walker2dEnv(exclude_current_positions_from_observation=False, xml_file=xml_file_local, render_mode=None)

NUM_STEPS = 10000

np.set_printoptions(precision=10000, floatmode='maxprec')

for SEED in range(10000):
    ic(SEED)
    initial_obs_1, initial_info_1 = global_env.reset(seed=SEED)
    initial_obs_2, initial_info_2 = local_env.reset(seed=SEED)
    assert_array_equal(initial_obs_1, initial_obs_2)

    global_env.action_space.seed(SEED)

    for time_step in range(NUM_STEPS):
        # We don't evaluate the determinism of actions
        action = global_env.action_space.sample()

        obs_1, rew_1, terminated_1, truncated_1, info_1 = global_env.step(action)
        obs_2, rew_2, terminated_2, truncated_2, info_2 = local_env.step(action)

        assert_array_equal(obs_1, obs_2, f"[{time_step}] ")
        assert_array_equal(rew_1, rew_2, f"[{time_step}] ")
        assert_array_equal(terminated_1, terminated_2, f"[{time_step}] ")
        assert_array_equal(truncated_1, truncated_2, f"[{time_step}] ")
        # assert info_1['x_position'] == info_2['x_position']

global_env.close()
local_env.close()
