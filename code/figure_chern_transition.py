"""
Script used to plot Fig.2 of [arXiv:2012.01459]
"""

import pickle
import os
import numpy as np

from scipy.integrate import cumtrapz

from qc_floquet import *

from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit

from scipy import stats

linear = lambda x, a, b: a + b*x

import sys
sys.path.append('../') 

from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size" : 10
})

def load_obj_local(filename ):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


burnt_orange = 1.1*np.array([191., 87., 0.]) / 256
nice_blue = 0.9*np.array([94., 138., 210.]) / 256
nice_green = 1.3*np.array([82., 112., 63.]) / 256
white = np.array([1., 1., 1.])


filenames = ['data/data_2020-10-17/target_h_t20_m1-4_omega0-125_90pc', \
            'data/data_2020-10-18/target_h_t20_m0-8_omega0-125_90pc', \
            'data/data_2020-10-18/target_h_t20_m2-8_omega0-125_90pc', \
            'data/data_2020-10-21/target_h_t20_m2-6_omega0-125_90pc', \
            'data/data_2020-10-22/target_h_t20_m1_omega0-125_90pc', \
            'data/data_2020-10-27/target_h_t20_m3_omega0-125_90pc', \
            'data/data_2020-10-28/target_h_t20_m1-2_omega0-125_90pc', \
            'data/data_2020-10-28/target_h_t20_m3-4_omega0-125_90pc', \
            'data/data_2020-10-29/target_h_t20_m2-3_omega0-125_90pc', \
            'data/data_2020-10-30/target_h_t20_m0-6_omega0-125_90pc', \
            'data/data_2020-11-05/target_h_t20_m1-7_omega0-125_90pc', \
            'data/data_2020-11-06/target_h_t20_m3-2_omega0-125_90pc' ]

m_vals = [1.4, 0.8, 2.8, 2.6, 1., 3., 1.2, 3.4, 2.3, 0.6, 1.7, 3.2]


plt.plot([0,2,2,4],[-1,-1,0,0],'k--', label='exact')  # exact step function transition

# load simulation data for the chern transition
sim_data = load_obj_local('data/chern_sim_data/chern_simulation_data_800_omega_0-125_length_20')
m_sim = np.array(sim_data['m'])
C_sim = np.array(sim_data['C'])

C_sim_sample = np.array(sim_data['C sampled'])

plt.plot(m_sim, C_sim_sample, '.', color=nice_blue, label='sim')


for ii in range(len(filenames)):

    real_data = load_obj_local(filenames[ii]+ '_real')

    dt = real_data['dt']
    max_drive_strength = real_data['max_drive_strength']
    num_points = real_data['num_points']
    drive_length_max = real_data['drive_length_max']  # drive time 
    drive_lengths = real_data['drive_lengths'] 
    total_samples = get_closest_multiple_of_16(drive_length_max * us /dt)

    hs = real_data['h']


    #### simulation for comparison ###########
    psi0 = instantaneous_eigenstate(hs[:,0])
    sim_results = get_expectation_values( hs, dt, psi0)

    correct_results_real = real_data['corrected results']
    pure_results_real = pure_results(correct_results_real)


    m = real_data['h parameters']['m']
    eta = real_data['h parameters']['eta']
    omega1 = real_data['h parameters']['omega1']
    omega2 = real_data['h parameters']['omega2']

    factor = omega1*omega2*max_drive_strength**2/(2*np.pi)


    #### real!! #############

    times = np.array(real_data['drive_lengths'])*us 

    pure_results_array = np.array([pure_results_real['x'],pure_results_real['y'],pure_results_real['z']])

    hs_1_dot = h1_dot(m, eta, omega1, np.pi/10, times, max_drive_strength, ramp_time=real_data['ramp_time'])
    hs_1_dot = np.array([hs_1_dot[0],hs_1_dot[1],hs_1_dot[2]])

    hs_2_dot = h2_dot(m, eta, omega2, 0, times, max_drive_strength, ramp_time=real_data['ramp_time'])
    hs_2_dot = np.array([hs_2_dot[0],hs_2_dot[1],hs_2_dot[2]])

    E1_dot = np.sum(hs_1_dot * pure_results_array, axis=0)
    E2_dot = np.sum(hs_2_dot * pure_results_array, axis=0)

    W1 = cumtrapz(E1_dot, x=times, initial=0)
    W2 = cumtrapz(E2_dot, x=times, initial=0)

    slope = (omega1*omega2*max_drive_strength**2/(2*np.pi))
    b1, a1, _, _, std_err_1 = stats.linregress(times, W1 / slope)
    b2, a2, _, _, std_err_2 = stats.linregress(times, W2 / slope)

    C_real = (b1-b2)/2
    C_real_error = 1.96*(std_err_1 + std_err_2)/2

    print("real C for m = {} is {} ± {}".format(m_vals[ii], C_real ,C_real_error))

    if ii == 0:
        label = 'real'
    else:
        label = '_nolegend_'
    plt.errorbar(m_vals[ii], C_real, yerr=C_real_error, fmt='o', capsize=4, color=burnt_orange, label=label)


plt.xlabel('$M$')
plt.ylabel('frequency conversion')
plt.xlim([0,4])
plt.ylim([-1.2,0.45])
plt.legend(loc='upper left', ncol=3, prop={'size': 9},)


myMacRatio = 1680/1280  # this is to get the figure to render properly on my scaled mac screen.
singleColumnWidth = myMacRatio * (3. + 3/8)

fig = plt.gcf()
fig.set_size_inches(singleColumnWidth, singleColumnWidth/1.6)
plt.tight_layout(pad=0.1)
plt.show()
