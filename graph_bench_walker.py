import numpy as np
import matplotlib.pyplot as plt

#a = np.load('results/walker2d_v4/run_0/evaluations.npz')
#b = np.load('results/walker2d_v4_fixed/run_0/evaluations.npz')

#assert a == b
RUNS = 1  # Number of statistical runs

steps = np.load('results/walker2d/run_0/evaluations.npz')['timesteps']
returns_walker4 = np.average(np.array([np.load('results/walker2d/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_walker4_fixed = np.average(np.array([np.load('results/walker2d_fixed/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_walker5 = np.average(np.array([np.load('results/walker2d_new/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)


#breakpoint()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(steps, np.average(returns_walker4, axis=0), label='Walker v4')
ax.plot(steps, np.average(returns_walker4_fixed, axis=0), label='Walker (v4) coeff=2')
ax.plot(steps, np.average(returns_walker5, axis=0), label='Walker v5')
#ax.fill_between(steps, np.min(returns_TD3_v4_w_ctn , axis=0), np.max(returns_TD3_v4_w_ctn , axis=0), alpha=0.2)

ax.set_title('SB3/TD3 on MuJoCo/Walker2d, for ' + str(RUNS) + ' Runs')
ax.legend()

fig.show()
breakpoint()

