"""
Script used to plot Fig.1 and the subfigures in Fig.6 of [arXiv:2012.XXXXX]
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

# specify font for plotting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size" : 10
})

def load_obj_local(filename ):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


# load data for plotting
path = 'data/data_2020-11-06'
master_filename = 'target_h_t20_m3-2_omega0-125_90pc'

real_data = load_obj_local(path + '/'+ master_filename + '_real')


dt = real_data['dt']
hs = real_data['h']
max_drive_strength = real_data['max_drive_strength']

setupArmonkBackend_real(verbose = True)  # get parameters from device

m = real_data['h parameters']['m']
eta = real_data['h parameters']['eta']
omega1 = real_data['h parameters']['omega1']
omega2 = real_data['h parameters']['omega2']


#### exact numerical simulation for comparison ###########

# set the initial state to be considered
psi0 = instantaneous_eigenstate(hs[:,0])
sim_results = get_expectation_values( hs, dt, psi0)


#### plot exact numerical simulation 1 #####################

t = np.array(real_data['drive_lengths']) 
t_sim = np.array([x*dt / us for x in range(len(sim_results[0]))])

factor = 1 #omega1*omega2*max_drive_strength**2/(2*np.pi)

hs_1_dot = h1_dot(m, eta, omega1, np.pi/10, t_sim*us, max_drive_strength, ramp_time=real_data['ramp_time'])
hs_1_dot = np.array([hs_1_dot[0],hs_1_dot[1],hs_1_dot[2]])
E1_dot = np.sum(hs_1_dot * sim_results, axis=0)
W1 = cumtrapz(E1_dot, x=t_sim*us, initial=0)

plt.plot(t_sim, W1 /factor, '-', c = tuple(0.8*nice_blue))

#### plot real 1 #########################

correct_results_real = real_data['corrected results']
pure_results_real = pure_results(correct_results_real)

times = np.array(real_data['drive_lengths'])*us 

pure_results = np.array([pure_results_real['x'],pure_results_real['y'],pure_results_real['z']])

hs_1_dot = h1_dot(m, eta, omega1, np.pi/10, times, max_drive_strength, ramp_time=real_data['ramp_time'])
hs_1_dot = np.array([hs_1_dot[0],hs_1_dot[1],hs_1_dot[2]])
E1_dot = np.sum(hs_1_dot * pure_results, axis=0)
W1 = cumtrapz(E1_dot, x=times, initial=0)

plt.plot(times/us, W1 /factor , 'o-', c = tuple(nice_blue), markevery=1, markersize=2)

#### plot fits 1 #################

slope = omega1*omega2*max_drive_strength**2/(2*np.pi) / factor

# 1.96*std_err give 95% confidence interval
b1, a1, _, _, std_err_1 = stats.linregress(times, W1 /factor)
text_res = "Best fit parameters W1:\na = {}\nb = {} ± {}".format(a1, b1, 1.96*std_err_1)
print(text_res)

plt.plot(times/us, linear(times, a1, b1) ,  '--', color=0.6*nice_blue)
bound_upper = linear(times, a1, b1 + 1.96*std_err_1)
bound_lower = linear(times, a1, b1 - 1.96*std_err_1)
# plotting the confidence intervals
plt.fill_between(times/us, bound_lower, bound_upper,
                 color = 0.6*nice_blue, alpha = 0.15)


#### plot exact numerical simulation 2 #####################

hs_2_dot = h2_dot(m, eta, omega2, 0, t_sim*us, max_drive_strength, ramp_time=real_data['ramp_time'])
hs_2_dot = np.array([hs_2_dot[0],hs_2_dot[1],hs_2_dot[2]])

E2_dot = np.sum(hs_2_dot * sim_results, axis=0)
W2 = cumtrapz(E2_dot, x=t_sim*us, initial=0)

plt.plot(t_sim, W2 /factor, '-', c = tuple(0.8*burnt_orange))

#### plot real 2 #########################

hs_2_dot = h2_dot(m, eta, omega2, 0, times, max_drive_strength, ramp_time=real_data['ramp_time'])
hs_2_dot = np.array([hs_2_dot[0],hs_2_dot[1],hs_2_dot[2]])
E2_dot = np.sum(hs_2_dot * pure_results, axis=0)
W2 = cumtrapz(E2_dot, x=times, initial=0)
plt.plot(times/us, W2 /factor, 'o-', c = tuple(burnt_orange), markevery=1, markersize=2)


#### plot fits 2 #################

b2, a2, _, _, std_err_2 = stats.linregress(times, W2 /factor)
text_res = "Best fit parameters W2:\na = {}\nb = {} ± {}".format(a2, b2, 1.96*std_err_2)
print(text_res)

print("\nC = {} ± {}".format((b1-b2)/2 ,(1.96*std_err_1 + 1.96*std_err_2)/2))

plt.plot(times/us, linear(times, a2, b2) , '--', color=0.6*burnt_orange)
bound_upper = linear(times, a2, b2 + 1.96*std_err_2)
bound_lower = linear(times, a2, b2 - 1.96*std_err_2)
# plotting the confidence intervals
plt.fill_between(times/us, bound_lower, bound_upper,
                 color = 0.6*burnt_orange, alpha = 0.15)


plt.plot(times/us, (-slope*times + a1), '--',c='k')
plt.plot(times/us, (slope*times + a2),'--',c='k')


plt.ylabel(r"Work done")
plt.xlabel("Drive length [$\mu$s]")
plt.legend(['$W_1$ sim', '$W_1$ real', '$W_1$ fit', '$W_2$ sim','$W_2$ real','$W_2$ fit'],prop={'size': 9}, loc='upper left',ncol=2)

date = path.replace("data/data_","")
plt.title(f"m = {m}, omega = {omega1}, date = {date}", fontweight="bold")

myMacRatio = 1680/1280  # this is to get the figure to render properly on my scaled mac screen.
singleColumnWidth = myMacRatio * (3. + 3/8)

fig = plt.gcf()
fig.set_size_inches(singleColumnWidth, singleColumnWidth/1.6)
plt.tight_layout(pad=0.1)
plt.show()