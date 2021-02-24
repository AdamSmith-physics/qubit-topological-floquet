"""
Numerical simulation of the simple heuristic error model
"""

import pickle
import os
import numpy as np

from scipy.integrate import cumtrapz, simps

from qc_floquet import *

from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit

linear = lambda x, a, b: a + b*x

import sys
sys.path.append('../') 

from matplotlib import pyplot as plt

burnt_orange = 1.1*np.array([191., 87., 0.]) / 256
persian_orange = np.array([217., 144., 88.]) / 256
rust = np.array([183., 65., 14.]) / 256

nice_blue = 0.9*np.array([94., 138., 210.]) / 256
nice_green = 1.3*np.array([82., 112., 63.]) / 256


def load_obj_local(filename ):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)

# values for M to simulate
m_vals = np.array([0.05 + 0.1*x for x in range(40)])
num_samples = 500
error_rate = 0.029  # average value extracted from fidelity of experimental data

# load some real data to extract experimental parameters
real_data = load_obj_local('data/data_2020-10-22/target_h_t20_m1_omega0-125_90pc' + '_real')

omega_vals = [0.125]
omega1 = omega_vals[0]

drive_length_max = 20

print(f"omega = {omega1}, drive length = {drive_length_max}")

dt = real_data['dt']
max_drive_strength = real_data['max_drive_strength']

num_points = 800
ramp_time = 2000*dt

eta = 1
omega2 = omega1*(1+np.sqrt(5))/2

factor = omega1*omega2*max_drive_strength**2/(2*np.pi)

drive_lengths = np.linspace(0, drive_length_max, num_points)
total_samples = get_closest_multiple_of_16(drive_length_max * us /dt)
times = np.array([x*dt for x in range(total_samples)])

drive_samples = [get_closest_multiple_of_16(x * us /dt) for x in drive_lengths]
drive_samples[-1] = times.shape[0] - 1


for m in m_vals:

    C_array_sample = []
    C_error_array_sample = []

    hs = target_h(m, eta, omega1, omega2, np.pi/10, 0, times, max_drive_strength, ramp_time=ramp_time)
    hs = np.array([hs[0],hs[1],hs[2]])

    #### Exact simulation ###########
    psi0 = instantaneous_eigenstate(hs[:,0])
    sim_results = get_expectation_values( hs, dt, psi0)

    sampled_sim_results = np.real(np.array(sim_results)[:,drive_samples])

    for sample in range(num_samples):
        # Generate random errors with exponential distribution for 1-fidelity
        new_sim_results = sampled_sim_results.copy()

        for jj in range(sampled_sim_results.shape[1]):
            v = np.array([[sampled_sim_results[0,jj]],[sampled_sim_results[1,jj]],[sampled_sim_results[2,jj]]])
            v = v / np.linalg.norm(v)
            n = np.random.normal(size=(3,1))
            n = n / np.linalg.norm(n)  # vector on unit sphere

            n = n - (n.T @ v)[0,0] * v
            n = n / np.linalg.norm(n)  # vector on circle orthogonal to v

            sampled_x = np.min([np.random.exponential(scale=error_rate),1])
            theta = np.arccos(1-2*sampled_x)

            C = np.array([[0, -n[2,0], n[1,0]],[n[2,0], 0, -n[0,0]],[-n[1,0], n[0,0], 0]])

            new_v = (v + np.sin(theta)* C @ v + (1-np.cos(theta))* C @ C @ v).flatten()
            new_sim_results[0,jj] = new_v[0]
            new_sim_results[1,jj] = new_v[1]
            new_sim_results[2,jj] = new_v[2]


        #### simulation sampled with errors!! #############
        # this code samples the exact numerical simulation at 800 points to match the experimental data
        # fitting this data more accurately reflects the fitting we do to experimental data (e.g. similar error bars)

        results_sample_x = new_sim_results[0,:]
        results_sample_y = new_sim_results[1,:]
        results_sample_z = new_sim_results[2,:]
        results_sample = np.array([results_sample_x,results_sample_y,results_sample_z])
        t_sim = np.array([x*dt / us for x in range(len(sim_results[0]))])
        t_sample = np.array(t_sim[drive_samples])

        hs_1_dot = h1_dot(m, eta, omega1, np.pi/10, t_sample*us, max_drive_strength, ramp_time=real_data['ramp_time'])
        hs_1_dot = np.array([hs_1_dot[0],hs_1_dot[1],hs_1_dot[2]])

        hs_2_dot = h2_dot(m, eta, omega2, 0, t_sample*us, max_drive_strength, ramp_time=real_data['ramp_time'])
        hs_2_dot = np.array([hs_2_dot[0],hs_2_dot[1],hs_2_dot[2]])

        E1_dot = np.sum(hs_1_dot * results_sample, axis=0)
        E2_dot = np.sum(hs_2_dot * results_sample, axis=0)

        W1 = cumtrapz(E1_dot, x=t_sample, initial=0)
        W2 = cumtrapz(E2_dot, x=t_sample, initial=0)

        best_fit_ab1, covar1 = curve_fit(linear, t_sample, W1/(omega1*omega2*max_drive_strength**2/(2*np.pi)))
        best_fit_ab2, covar2 = curve_fit(linear, t_sample, W2/(omega1*omega2*max_drive_strength**2/(2*np.pi)))
        sigma_ab1 = np.sqrt(np.diagonal(covar1))
        sigma_ab2 = np.sqrt(np.diagonal(covar2))

        C_sim_sample = (best_fit_ab1[1]-best_fit_ab2[1])/2
        C_sim_error_sample = (sigma_ab1[1]+sigma_ab2[1])/2

        C_array_sample.append(C_sim_sample)
        C_error_array_sample.append(C_sim_error_sample)

    C_mean = np.average(C_array_sample)
    C_sd = np.sqrt(1/(num_samples-1) * np.sum( (np.array(C_array_sample) - C_mean)**2 ) )

    print("sampled simulation C for m = {}, is C_mean = {}, C_sd = {}".format(m, C_mean ,C_sd))


    #### save data to file ###########################

    path = "data/chern_error_sim_data/"

    create_directory(path)

    filename_ext = f'_omega_{omega1}_length_{drive_length_max}_error_{error_rate}_m_{m:.2f}'.replace('.','-')
    filename = "chern_error_simulation_data_800" + filename_ext
    h_parameters = {'eta': eta, 'omega1': omega1, 'omega2': omega2, 'error_rate': error_rate}
    data = {   'Hamiltonian' : h_parameters,
            'ramp_time': ramp_time,
            'dt': dt,
            'max_drive_strength': max_drive_strength,
            'm': m_vals,
            'C sampled': C_array_sample,
            'C_errors_sampled': C_error_array_sample,
            'C_mean': C_mean,
            'C_sd': C_sd  }

    saveDataToFile(data, path, filename)

