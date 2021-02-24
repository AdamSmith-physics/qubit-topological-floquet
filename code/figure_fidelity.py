"""
Script used to plot Fig.8 of [arXiv:2012.01459]
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

from math import log10, floor

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


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size" : 10
})


def load_obj_local(filename ):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


def round_sig(x, sig=3):
    factor = sig-int(floor(log10(abs(x))))-1
    answer = round(x, factor)
    if factor <= 0:
        answer = int(answer)
    return answer


filenames = ['data/data_2020-10-30/target_h_t20_m0-6_omega0-125_90pc', \
            'data/data_2020-10-18/target_h_t20_m0-8_omega0-125_90pc', \
            'data/data_2020-10-22/target_h_t20_m1_omega0-125_90pc', \
            'data/data_2020-10-28/target_h_t20_m1-2_omega0-125_90pc', \
            'data/data_2020-10-17/target_h_t20_m1-4_omega0-125_90pc', \
            'data/data_2020-11-05/target_h_t20_m1-7_omega0-125_90pc', \
            'data/data_2020-10-29/target_h_t20_m2-3_omega0-125_90pc', \
            'data/data_2020-10-21/target_h_t20_m2-6_omega0-125_90pc', \
            'data/data_2020-10-18/target_h_t20_m2-8_omega0-125_90pc', \
            'data/data_2020-10-27/target_h_t20_m3_omega0-125_90pc', \
            'data/data_2020-11-06/target_h_t20_m3-2_omega0-125_90pc', \
            'data/data_2020-10-28/target_h_t20_m3-4_omega0-125_90pc']


fig, axs = plt.subplots(len(filenames))

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

    print(f"M={m}, average fidelity = ",np.average(fidelity))


    axs[ii].scatter(t,fidelity, marker='.', color= tuple(nice_blue))

    ans = scipy.optimize.curve_fit(lambda x,a,b: a*np.exp(-b*x),  t,  fidelity,  p0=(1, 0))
    print(ans)

    axs[ii].plot(t, ans[0][0]*np.exp(-ans[0][1]*t),'--',color= tuple(0.7*nice_blue))


    axs[ii].legend([f"M={m} fit [{round_sig(ans[0][0])},{round_sig(1/ans[0][1])}]"],prop={'size': 9},loc='lower left')

    if ii < len(filenames) - 1:
        axs[ii].xaxis.set_ticklabels([])
    axs[ii].locator_params(axis='y', nbins=3)
    axs[ii].set(ylabel=r"Fidelity")



myMacRatio = 1680/1280  # this is to get the figure to render properly on my scaled mac screen.
singleColumnWidth = myMacRatio * (3. + 3/8)

plt.xlabel("Drive length [$\mu$s]")

fig = plt.gcf()
fig.set_size_inches(singleColumnWidth, singleColumnWidth/1.6 * (len(filenames)-2)/3)
plt.tight_layout(pad=0.1)
plt.show()
