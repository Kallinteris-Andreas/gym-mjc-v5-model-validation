from numpy.testing import assert_array_equal

import hopper_v4
import hopper_v4_new
import numpy as np

from icecream import ic

global_env = hopper_v4.HopperEnv(exclude_current_positions_from_observation=False, reset_noise_scale=0, render_mode=None)
local_env = hopper_v4_new.HopperEnv(exclude_current_positions_from_observation=False, reset_noise_scale=0, render_mode=None)
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
#breakpoint()

SEED = 1234
NUM_STEPS = 1000

max_error = 0
np.set_printoptions(precision=10000, floatmode='maxprec')
#np.set_printoptions(floatmode='maxprec_equal')

for SEED in range(1):
    ic(SEED)
    initial_obs_1, initial_info_1 = global_env.reset(seed=SEED)
    initial_obs_2, initial_info_2 = local_env.reset(seed=SEED)
    assert_array_equal(initial_obs_1, initial_obs_2)

    global_env.action_space.seed(SEED)

    for time_step in range(1):
        # We don't evaluate the determinism of actions
        action = global_env.action_space.sample()

        obs_1, rew_1, terminated_1, truncated_1, info_1 = global_env.step(action)
        obs_2, rew_2, terminated_2, truncated_2, info_2 = local_env.step(action)

        error = obs_1-obs_2
        ic(error)
        assert((error== [ 1.6263032587282567e-19, 0.0000000000000000e+00, 5.4210108624275222e-19, 3.2526065174565133e-19, 8.6736173798840355e-19, -8.6736173798840355e-19, 1.3877787807814457e-17, 0.0000000000000000e+00, 1.1102230246251565e-16, 4.1633363423443370e-17, 0.0000000000000000e+00, -2.2204460492503131e-16]).all())
        #breakpoint()
        #breakpoint()
        """
        if (terminated_1 or truncated_1 or terminated_1 or terminated_2):
            ic(time_step)
            ic(error)
            ic((terminated_1, terminated_2, truncated_1))
            max_error = max(max_error, max(error))
            break000
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
