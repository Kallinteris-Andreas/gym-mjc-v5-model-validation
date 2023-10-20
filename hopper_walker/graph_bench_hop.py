import numpy as np
import matplotlib.pyplot as plt

RUNS = 10  # Number of statistical runs

steps = np.load('results/hopper_new/run_0/evaluations.npz')['timesteps']
returns_hop4 = np.average(np.array([np.load('results/hopper/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_hop5 = np.average(np.array([np.load('results/hopper_new/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_hop5_gen = np.average(np.array([np.load('results/hopper_new_gen/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
#returns_hop5_saran_t_trans = np.average(np.array([np.load('results/hopper_saran_t_trans/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_hop5_saran_t_trans2 = np.average(np.array([np.load('results/hopper_saran_t_trans2/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
#returns_hop5_t2 = np.average(np.array([np.load('results/hopper_v5_t2/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
#returns_hop5_t3 = np.average(np.array([np.load('results/hopper_v5_t3/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
#returns_hop_temp = np.average(np.array([np.load('results/temp/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)


breakpoint()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(steps, np.average(returns_hop4, axis=0), label='Hopper-v4')
#ax.plot(steps, np.average(returns_hop5, axis=0), label='Hopper-v5')
#ax.plot(steps, np.average(returns_hop5_gen, axis=0), label='Hopper-v5_gen')
#ax.plot(steps, np.average(returns_hop5_saran_t_trans, axis=0), label='Hopper @saran_t manually transcribed')
ax.plot(steps, np.average(returns_hop5_saran_t_trans2, axis=0), label='Hopper @saran_t manually transcribed (2)')
ax.fill_between(steps, np.min(returns_hop4, axis=0), np.max(returns_hop4, axis=0), alpha=0.2)
#ax.fill_between(steps, np.min(returns_hop5, axis=0), np.max(returns_hop5, axis=0), alpha=0.2)
#ax.fill_between(steps, np.min(returns_hop5_gen, axis=0), np.max(returns_hop5_gen, axis=0), alpha=0.2)
#ax.fill_between(steps, np.min(returns_hop5_saran_t_trans, axis=0), np.max(returns_hop5_saran_t_trans, axis=0), alpha=0.2)
ax.fill_between(steps, np.min(returns_hop5_saran_t_trans2, axis=0), np.max(returns_hop5_saran_t_trans2, axis=0), alpha=0.2)

ax.set_title('SB3/TD3 on MuJoCo/Hopper, for ' + str(RUNS) + ' Runs')
ax.legend()

fig.show()
breakpoint()

