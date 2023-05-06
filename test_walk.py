from numpy.testing import assert_array_equal

#import hopper_v4
#import hopper_v4_new
import walker2d_v4
import walker2d_v4_new
import numpy as np

from icecream import ic

global_env = walker2d_v4.Walker2dEnv(exclude_current_positions_from_observation=False, reset_noise_scale=0, render_mode=None)
local_env = walker2d_v4_new.Walker2dEnv(exclude_current_positions_from_observation=False, reset_noise_scale=0, render_mode='human')
"""
for t in dir(global_env.unwrapped.model):
    if isinstance(getattr(global_env.unwrapped.model, t), np.ndarray) and getattr(global_env.unwrapped.model, t).size > 0:
        if (getattr(global_env.unwrapped.model, t) != getattr(local_env.unwrapped.model, t)).all():
            print(t)
    #elif callable(getattr(global_env.unwrapped.model, t)):
        #None
    else:
        if getattr(global_env.unwrapped.model, t) != getattr(local_env.unwrapped.model, t):
            print(t)
"""

NUM_STEPS = 1000

max_error = 0
np.set_printoptions(precision=10000, floatmode='maxprec')

for SEED in range(1):
    ic(SEED)
    initial_obs_1, initial_info_1 = global_env.reset(seed=SEED)
    initial_obs_2, initial_info_2 = local_env.reset(seed=SEED)
    assert_array_equal(initial_obs_1, initial_obs_2)

    global_env.action_space.seed(SEED)

    for time_step in range(1000):
        # We don't evaluate the determinism of actions
        action = global_env.action_space.sample()

        obs_1, rew_1, terminated_1, truncated_1, info_1 = global_env.step(action)
        obs_2, rew_2, terminated_2, truncated_2, info_2 = local_env.step(action)

        error = obs_1-obs_2
        #assert((error== [-1.1492543028346347e-17,  0.0000000000000000e+00, -2.0816681711721685e-16, -2.9837243786801082e-16, 1.1552174147833050e-16, -2.7755575615628914e-17, -2.8796409701214998e-16,  9.3675067702747583e-17, -1.3877787807814457e-17, -1.9984014443252818e-15, -1.3877787807814457e-16, -3.4194869158454821e-14, -4.8849813083506888e-14,  1.7656015538491943e-14, 0.0000000000000000e+00, -4.5297099404706387e-14, 9.7699626167013776e-15,  0.0000000000000000e+00]).all())
        #breakpoint()
        ic(error)
        #breakpoint()
        """
        if (terminated_1 or truncated_1 or terminated_1 or terminated_2):
            ic(time_step)
            ic(error)
            ic((terminated_1, terminated_2, truncated_1))
            max_error = max(max_error, max(error))
            break
        """
    """
    assert_array_equal(obs_1, obs_2, f"[{time_step}] ")
    assert global_env.observation_space.contains(
        obs_1
    )  # obs_2 verified by previous assertion

    assert rew_1 == rew_2, f"[{time_step}] reward 1={rew_1}, reward 2={rew_2}"
    assert (
        terminated_1 == terminated_2
    ), f"[{time_step}] done 1={terminated_1}, done 2={terminated_2}"
    assert (
        truncated_1 == truncated_2
    ), f"[{time_step}] done 1={truncated_1}, done 2={truncated_2}"
    # assert_equals(info_1, info_2, f"[{time_step}] ")

    if (
        terminated_1 or truncated_1
    ):  # terminated_2, truncated_2 verified by previous assertion
        global_env.reset(seed=SEED)
        local_env.reset(seed=SEED)
    """
ic(max_error)

global_env.close()
local_env.close()
