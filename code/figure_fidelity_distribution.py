"""
Script used to plot Fig.6 of [arXiv:2012.01459]
"""

import pickle
import os
import numpy as np

from qutip import Bloch as Bloch

import scipy
from scipy.integrate import cumtrapz, simps

from qc_floquet import *

from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression

from scipy import stats

linear = lambda x, a, b: a + b*x

import sys
sys.path.append('../') 

from matplotlib import pyplot as plt
from matplotlib.pyplot import rc


burnt_orange = 1.1*np.array([191., 87., 0.]) / 256
persian_orange = np.array([217., 144., 88.]) / 256
rust = np.array([183., 65., 14.]) / 256

nice_blue = 0.9*np.array([94., 138., 210.]) / 256
nice_green = 1.3*np.array([82., 112., 63.]) / 256


myMacRatio = 1680/1280  # this is to get the figure to render properly on my scaled mac screen.
singleColumnWidth = myMacRatio * (3. + 3/8)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size" : 10
})


def load_obj_local(filename ):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# omit 1.7 and 2.3
filenames = ['data/data_2020-10-30/target_h_t20_m0-6_omega0-125_90pc', \
            'data/data_2020-10-18/target_h_t20_m0-8_omega0-125_90pc', \
            'data/data_2020-10-22/target_h_t20_m1_omega0-125_90pc', \
            'data/data_2020-10-28/target_h_t20_m1-2_omega0-125_90pc', \
            'data/data_2020-10-17/target_h_t20_m1-4_omega0-125_90pc', \
            'data/data_2020-10-21/target_h_t20_m2-6_omega0-125_90pc', \
            'data/data_2020-10-18/target_h_t20_m2-8_omega0-125_90pc', \
            'data/data_2020-10-27/target_h_t20_m3_omega0-125_90pc', \
            'data/data_2020-11-06/target_h_t20_m3-2_omega0-125_90pc', \
            'data/data_2020-10-28/target_h_t20_m3-4_omega0-125_90pc']


fidelity_total = np.array([])

for ii in range(len(filenames)):

    real_data = load_obj_local(filenames[ii] + '_real')


    dt = real_data['dt']
    hs = real_data['h']
    max_drive_strength = real_data['max_drive_strength']

    #setupArmonkBackend_real(verbose = True)

    m = real_data['h parameters']['m']
    eta = real_data['h parameters']['eta']
    omega1 = real_data['h parameters']['omega1']
    omega2 = real_data['h parameters']['omega2']

    print("M: ",m)


    #### simulation for comparison ###########


    # set the initial state to be considered
    psi0 = instantaneous_eigenstate(hs[:,0])
    sim_results = get_expectation_values( hs, dt, psi0)

    drive_samples = [get_closest_multiple_of_16(x * us /dt) for x in real_data['drive_lengths']]
    t = np.array(drive_samples)*dt/us
    t_sim = np.array([x*dt / us for x in range(len(sim_results[0]))])
    drive_samples[-1] = t_sim.shape[0] - 1

    #### plot real 1 #########################

    correct_results_real = real_data['corrected results']
    pure_results_real = pure_results(correct_results_real)


    fidelity = ((np.array(pure_results_real['x'])+sim_results[0][drive_samples])**2 
                        +(np.array(pure_results_real['y'])+sim_results[1][drive_samples])**2 
                        +(np.array(pure_results_real['z'])+sim_results[2][drive_samples])**2) / 4

    fidelity_total = np.append(fidelity_total, fidelity)

    print(f"M={m}, average fidelity = ",np.average(fidelity))


    if ii == 2:
        fidelity_example = fidelity





print(f"total average fidelity = {np.average(fidelity_total)}")
print(f"using average = 0.971")



fig, ax1 = plt.subplots()
fig.set_size_inches(singleColumnWidth, singleColumnWidth/1.6)
left, bottom, width, height = [0.37, 0.4, 0.6, 0.55]

plt.ylabel("probability density")
plt.xlabel("1 - fidelity")
plt.xlim([-0.003,0.153])

ax2 = fig.add_axes([left, bottom, width, height])

plt.xlabel("1 - fidelity")
plt.xlim([-0.002,0.102])


error_mean = 1.-0.971

num_bins = 400
ax1.hist(1-fidelity_total, bins=num_bins, density=True, range=(0,1), color=burnt_orange, histtype='stepfilled', facecolor=burnt_orange,
               alpha=0.25)
ax1.hist(1-fidelity_total, bins=num_bins, density=True, range=(0,1), color=burnt_orange, histtype='step')
x = np.array([j/1000 for j in range(1001)])
distr = 1/error_mean*np.exp(-x/error_mean)
ax1.plot(x, distr, color=nice_blue)


error_mean = 1.-np.average(fidelity_example)
ax2.hist(1-fidelity_example, bins=num_bins, density=True, range=(0,1), color=burnt_orange, histtype='stepfilled', facecolor=burnt_orange,
               alpha=0.25)
ax2.hist(1-fidelity_example, bins=num_bins, density=True, range=(0,1), color=burnt_orange, histtype='step')
x = np.array([j/1000 for j in range(1001)])
distr = 1/error_mean*np.exp(-x/error_mean)
print(f"length = {len(fidelity_example)}")
ax2.plot(x, distr, color=nice_blue)

ax2.legend(["Model","Data"],prop={'size': 10},loc='upper right')

plt.tight_layout(pad=0.15)
plt.show()
