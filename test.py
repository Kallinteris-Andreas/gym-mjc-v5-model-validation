import gymnasium
import hopper

xml_file = "/home/master-andreas/gym-mjc-v5-model-validation/hopper.xml"
#xml_file = "/home/master-andreas/gym-mjc-v5-model-validation/hopper_saran_t_man.xml"
env = hopper.HopperEnv(xml_file=xml_file, reset_noise_scale=0, render_mode='human')


for _ in range(10000):
    #env.step(env.action_space.sample())
    env.step([0,0,0])
