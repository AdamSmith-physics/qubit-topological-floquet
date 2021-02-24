"""
Script used to plot Fig.1 of [arXiv:2012.01459]
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


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size" : 10
})


def load_obj_local(filename ):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


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


#### plot simulation 1 #####################

drive_samples = [get_closest_multiple_of_16(x * us /dt) for x in real_data['drive_lengths']]
t = np.array(drive_samples)*dt/us
t_sim = np.array([x*dt / us for x in range(len(sim_results[0]))])
drive_samples[-1] = t_sim.shape[0] - 1


factor = 1 #omega1*omega2*max_drive_strength**2/(2*np.pi)

fig, axs = plt.subplots(5)


axs[0].plot(t_sim, np.real(sim_results[0]), color= tuple(0.7*nice_blue)) # plot real part of Rabi values
axs[1].plot(t_sim, np.real(sim_results[1]), color= tuple(0.7*nice_blue)) # plot real part of Rabi values
axs[2].plot(t_sim, np.real(sim_results[2]), color= tuple(0.7*nice_blue)) # plot real part of Rabi values

#axs[0].scatter(t_sim[drive_samples], np.real(sim_results[0][drive_samples]), color= tuple(0.8*burnt_orange)) # plot real part of Rabi values
#axs[1].scatter(t_sim[drive_samples], np.real(sim_results[1][drive_samples]), color= tuple(0.8*nice_green)) # plot real part of Rabi values
#axs[2].scatter(t_sim[drive_samples], np.real(sim_results[2][drive_samples]), color= tuple(0.8*nice_blue)) # plot real part of Rabi values

#### plot real 1 #########################

correct_results_real = real_data['corrected results']
pure_results_real = pure_results(correct_results_real)

# raw results
axs[0].scatter(t,correct_results_real['x'], marker='.', color= tuple(nice_blue)) # plot real part of Rabi values
axs[1].scatter(t,correct_results_real['y'], marker='.', color= tuple(nice_blue)) # plot real part of Rabi values
axs[2].scatter(t,correct_results_real['z'], marker='.', color= tuple(nice_blue)) # plot real part of Rabi values


axs[0].xaxis.set_ticklabels([])
axs[1].xaxis.set_ticklabels([])
axs[2].xaxis.set_ticklabels([])
axs[0].set(ylabel=r"$\langle \sigma_x \rangle$")
axs[1].set(ylabel=r"$\langle \sigma_y \rangle$")
axs[2].set(ylabel=r"$\langle \sigma_z \rangle$")
plt.xlabel("Drive length [$\mu$s]")


for i in range(1):
    axs[i].set_ylim([-1.05,1.05])
    axs[i].legend(['ED', 'real'],prop={'size': 9},loc='upper left')

myMacRatio = 1680/1280  # this is to get the figure to render properly on my scaled mac screen.
singleColumnWidth = myMacRatio * (3. + 3/8)

"""
fig = plt.gcf()
fig.set_size_inches(singleColumnWidth, singleColumnWidth/1.6)
plt.tight_layout(pad=0.1)
plt.show()"""



fidelity = ((np.array(pure_results_real['x'])+sim_results[0][drive_samples])**2 
                    +(np.array(pure_results_real['y'])+sim_results[1][drive_samples])**2 
                    +(np.array(pure_results_real['z'])+sim_results[2][drive_samples])**2) / 4

purity = 1/2 + 1/2*(np.array(correct_results_real['x'])**2 
                    +np.array(correct_results_real['y'])**2 
                    +np.array(correct_results_real['z'])**2)
purity[purity>1.]=1.  # this is equivalent to maximum likelihood fitting!!

axs[3].scatter(t,purity, marker='.', color= tuple(nice_blue))
axs[4].scatter(t,fidelity, marker='.', color= tuple(nice_blue))

#plt.ylim([0,1.1])


ans = scipy.optimize.curve_fit(lambda x,a,b: 1/2 + a*np.exp(b*x),  t,  purity,  p0=(1, 0))
print(ans)

axs[3].plot(t, 1/2+ans[0][0]*np.exp(ans[0][1]*t),'--',color= tuple(0.7*nice_blue))

ans = scipy.optimize.curve_fit(lambda x,a,b: a*np.exp(b*x),  t,  fidelity,  p0=(1, 0))
print(ans)

axs[4].plot(t, ans[0][0]*np.exp(ans[0][1]*t),'--',color= tuple(0.7*nice_blue))


axs[3].legend([r"fit"],prop={'size': 9},loc='lower left')

axs[3].xaxis.set_ticklabels([])
axs[3].set_ylim([0.48,1.02])
axs[3].yaxis.set_ticks([0.5,0.75,1.0])
#axs[3].yaxis.set_ticklabels([0.6,0.8,1.0])
axs[4].yaxis.set_ticks([0.9,0.95,1.0])
#axs[4].set_ylim([0])
#axs[4].yaxis.set_ticklabels([0.9,0.95,1.0])
axs[3].set(ylabel=r"Purity")
axs[4].set(ylabel=r"Fidelity")
plt.xlabel("Drive length [$\mu$s]")

fig = plt.gcf()
fig.set_size_inches(singleColumnWidth, singleColumnWidth/1.1)
plt.tight_layout(pad=0.1)
plt.show()
