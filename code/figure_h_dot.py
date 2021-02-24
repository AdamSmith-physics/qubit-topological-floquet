"""
Script used to plot Fig.5 of [arXiv:2012.01459]
"""

import pickle
import os
import numpy as np

from qutip import Bloch as Bloch

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


path = 'data/data_2020-10-22'
master_filename = 'target_h_t20_m1_omega0-125_90pc'

real_data = load_obj_local(path + '/'+ master_filename + '_real')


dt = real_data['dt']
hs = real_data['h']
max_drive_strength = real_data['max_drive_strength']

m = real_data['h parameters']['m']
eta = real_data['h parameters']['eta']
omega1 = real_data['h parameters']['omega1']
omega2 = real_data['h parameters']['omega2']


#### simulation for comparison ###########

# set the initial state to be considered
psi0 = instantaneous_eigenstate(hs[:,0])
sim_results = get_expectation_values( hs, dt, psi0)

fig, axs = plt.subplots(2)

#### plot simulation #####################

t = np.array(real_data['drive_lengths']) 
t_sim = np.array([x*dt / us for x in range(len(sim_results[0]))])

factor = 1 #omega1*omega2*max_drive_strength**2/(2*np.pi)

hs_1_dot = h1_dot(m, eta, omega1, np.pi/10, t_sim*us, max_drive_strength, ramp_time=real_data['ramp_time'])
hs_1_dot = np.array([hs_1_dot[0],hs_1_dot[1],hs_1_dot[2]])
E1_dot = np.sum(hs_1_dot * sim_results, axis=0)

axs[0].plot(t_sim, E1_dot /factor, '-', c = tuple(0.8*nice_blue))

hs_2_dot = h2_dot(m, eta, omega2, 0, t_sim*us, max_drive_strength, ramp_time=real_data['ramp_time'])
hs_2_dot = np.array([hs_2_dot[0],hs_2_dot[1],hs_2_dot[2]])
E2_dot = np.sum(hs_2_dot * sim_results, axis=0)

axs[1].plot(t_sim, E2_dot /factor, '-', c = tuple(0.8*burnt_orange))

#### plot real #########################

correct_results_real = real_data['corrected results']
pure_results_real = pure_results(correct_results_real)

times = np.array(real_data['drive_lengths'])*us 

pure_results = np.array([pure_results_real['x'],pure_results_real['y'],pure_results_real['z']])

hs_1_dot = h1_dot(m, eta, omega1, np.pi/10, times, max_drive_strength, ramp_time=real_data['ramp_time'])
hs_1_dot = np.array([hs_1_dot[0],hs_1_dot[1],hs_1_dot[2]])
E1_dot = np.sum(hs_1_dot * pure_results, axis=0)

axs[0].scatter(times/us, E1_dot /factor , marker='.', c = tuple(nice_blue))
axs[0].set(ylabel=r"$\langle d h_1 / dt \rangle$")


hs_2_dot = h2_dot(m, eta, omega2, 0, times, max_drive_strength, ramp_time=real_data['ramp_time'])
hs_2_dot = np.array([hs_2_dot[0],hs_2_dot[1],hs_2_dot[2]])
E2_dot = np.sum(hs_2_dot * pure_results, axis=0)

axs[1].scatter(times/us, E2_dot /factor , marker='.', c = tuple(burnt_orange))
axs[1].set(ylabel=r"$\langle d h_2 / dt \rangle$")


plt.xlabel("Drive length [$\mu$s]")

for i in range(2):
    #axs[i].set_ylim([-1.05,1.05])
    axs[i].legend(['ED', 'real'],prop={'size': 9},loc='upper left')

myMacRatio = 1680/1280  # this is to get the figure to render properly on my scaled mac screen.
singleColumnWidth = myMacRatio * (3. + 3/8)

fig = plt.gcf()
fig.set_size_inches(singleColumnWidth, singleColumnWidth/1.6)
plt.tight_layout(pad=0.1)
plt.show()