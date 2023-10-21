import gymnasium
from gymnasium.utils.env_match import check_environments_match


current_env = gymnasium.make('Swimmer-v3', render_mode='rgb_array')
new_env = gymnasium.make('Swimmer-v3', xml_file="./new_swim.xml", render_mode='rgb_array')

check_environments_match(current_env, new_env, num_steps=int(1e6), skip_render=True)

