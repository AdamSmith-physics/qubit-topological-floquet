"""
Script used to plot Fig.4 of [arXiv:2012.XXXXX]
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

setupArmonkBackend_real(verbose = True)

m = real_data['h parameters']['m']
eta = real_data['h parameters']['eta']
omega1 = real_data['h parameters']['omega1']
omega2 = real_data['h parameters']['omega2']


#### simulation for comparison ###########

# set the initial state to be considered
psi0 = instantaneous_eigenstate(hs[:,0])
sim_results = get_expectation_values( hs, dt, psi0)


#### plot simulation 1 #####################

t = np.array(real_data['drive_lengths']) 
t_sim = np.array([x*dt / us for x in range(len(sim_results[0]))])

factor = 1 #omega1*omega2*max_drive_strength**2/(2*np.pi)

fig, axs = plt.subplots(3)


axs[0].plot(t_sim, np.real(sim_results[0]), color= tuple(0.8*burnt_orange)) # plot real part of Rabi values
axs[1].plot(t_sim, np.real(sim_results[1]), color= tuple(0.8*nice_green)) # plot real part of Rabi values
axs[2].plot(t_sim, np.real(sim_results[2]), color= tuple(0.8*nice_blue)) # plot real part of Rabi values

#### plot real 1 #########################

correct_results_real = real_data['corrected results']
pure_results_real = pure_results(correct_results_real)

# raw results
axs[0].scatter(t,correct_results_real['x'], marker='.', color= tuple(burnt_orange)) # plot real part of Rabi values
axs[1].scatter(t,correct_results_real['y'], marker='.', color= tuple(nice_green)) # plot real part of Rabi values
axs[2].scatter(t,correct_results_real['z'], marker='.', color= tuple(nice_blue)) # plot real part of Rabi values


axs[0].set(ylabel=r"$\langle \sigma_x \rangle$")
axs[1].set(ylabel=r"$\langle \sigma_y \rangle$")
axs[2].set(ylabel=r"$\langle \sigma_z \rangle$")
plt.xlabel("Drive length [$\mu$s]")


for i in range(3):
    axs[i].set_ylim([-1.05,1.05])
    axs[i].legend(['ED', 'real'],prop={'size': 9},loc='upper left')

myMacRatio = 1680/1280  # this is to get the figure to render properly on my scaled mac screen.
singleColumnWidth = myMacRatio * (3. + 3/8)

fig = plt.gcf()
fig.set_size_inches(singleColumnWidth, singleColumnWidth/1.6)
plt.tight_layout(pad=0.1)
plt.show()