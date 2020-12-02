"""
Script used to generate the simulation data plotted in figure_chern_transition.py
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


def load_obj_local(filename ):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)

# values for M to simulate
m_vals = np.linspace(0,4,401,endpoint=True)

# load some real data to extract experimental parameters
real_data = load_obj_local('data/data_2020-10-22/target_h_t20_m1_omega0-125_90pc' + '_real')

omega_vals = [0.125]
drive_length_vals = np.array([20])
drive_length_vals = list(drive_length_vals)

for ii in range(len(omega_vals)):

    omega1 = omega_vals[ii]
    drive_length_max = drive_length_vals[ii]

    C_array = []
    C_error_array = []

    C_array_sample = []
    C_error_array_sample = []

    for ii in range(len(m_vals)):

        print(f"omega = {omega1}, drive length = {drive_length_max}")

        dt = real_data['dt']
        max_drive_strength = real_data['max_drive_strength']
        

        num_points = 800
        ramp_time = 2000*dt
        m = m_vals[ii]
        eta = 1
        omega2 = omega1*(1+np.sqrt(5))/2

        drive_lengths = np.linspace(0, drive_length_max, num_points)
        total_samples = get_closest_multiple_of_16(drive_length_max * us /dt)
        times = np.array([x*dt for x in range(total_samples)])

        hs = target_h(m, eta, omega1, omega2, np.pi/10, 0, times, max_drive_strength, ramp_time=ramp_time)
        hs = np.array([hs[0],hs[1],hs[2]])


        #### simulation for comparison ###########
        psi0 = instantaneous_eigenstate(hs[:,0])
        sim_results = get_expectation_values( hs, dt, psi0)


        factor = omega1*omega2*max_drive_strength**2/(2*np.pi)

        #### simulation!! #############

        t_sim = np.array([x*dt / us for x in range(len(sim_results[0]))])

        hs_1_dot = h1_dot(m, eta, omega1, np.pi/10, t_sim*us, max_drive_strength, ramp_time=real_data['ramp_time'])
        hs_1_dot = np.array([hs_1_dot[0],hs_1_dot[1],hs_1_dot[2]])

        hs_2_dot = h2_dot(m, eta, omega2, 0, t_sim*us, max_drive_strength, ramp_time=real_data['ramp_time'])
        hs_2_dot = np.array([hs_2_dot[0],hs_2_dot[1],hs_2_dot[2]])

        E1_dot = np.sum(hs_1_dot * sim_results, axis=0)
        E2_dot = np.sum(hs_2_dot * sim_results, axis=0)

        W1 = cumtrapz(E1_dot, x=t_sim, initial=0)
        W2 = cumtrapz(E2_dot, x=t_sim, initial=0)


        # fit the exact numerical simulation in the same way we fit the experimental data!
        best_fit_ab1, covar1 = curve_fit(linear, t_sim, W1/(omega1*omega2*max_drive_strength**2/(2*np.pi)))
        best_fit_ab2, covar2 = curve_fit(linear, t_sim, W2/(omega1*omega2*max_drive_strength**2/(2*np.pi)))
        sigma_ab1 = np.sqrt(np.diagonal(covar1))
        sigma_ab2 = np.sqrt(np.diagonal(covar2))

        C_sim = (best_fit_ab1[1]-best_fit_ab2[1])/2
        C_sim_error = (sigma_ab1[1]+sigma_ab2[1])/2

        print("simulation C for m = {} is {} ± {}".format(m_vals[ii], C_sim ,C_sim_error))

        C_array.append(C_sim)
        C_error_array.append(C_sim_error)


        #### simulation sampled!! #############
        # this code samples the exact numerical simulation at 800 points to match the experimental data
        # fitting this data more accurately reflects the fitting we do to experimental data (e.g. similar error bars)

        results_sample_x = [sim_results[0,min(get_closest_multiple_of_16(x* us /dt),len(sim_results[0,:])-1)] for x in drive_lengths]
        results_sample_y = [sim_results[1,min(get_closest_multiple_of_16(x* us /dt),len(sim_results[0,:])-1)] for x in drive_lengths]
        results_sample_z = [sim_results[2,min(get_closest_multiple_of_16(x* us /dt),len(sim_results[0,:])-1)] for x in drive_lengths]
        results_sample = np.array([results_sample_x,results_sample_y,results_sample_z])
        t_sample = np.array([t_sim[min(get_closest_multiple_of_16(x* us /dt),len(sim_results[0,:])-1)] for x in drive_lengths])

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

        print("sampled simulation C for m = {} is {} ± {}".format(m_vals[ii], C_sim_sample ,C_sim_error_sample))

        C_array_sample.append(C_sim_sample)
        C_error_array_sample.append(C_sim_error_sample)


    #### save data to file ###########################

    path = "data/chern_sim_data/"

    create_directory(path)

    filename_ext = f'_omega_{omega1}_length_{drive_length_max}'.replace('.','-')
    filename = "chern_simulation_data_800" + filename_ext
    h_parameters = {'eta': eta, 'omega1': omega1, 'omega2': omega2}
    data = {   'Hamiltonian' : h_parameters,
                'ramp_time': ramp_time,
                'dt': dt,
                'max_drive_strength': max_drive_strength,
                'm': m_vals,
                'C': C_array,
                'C_errors': C_error_array,
                'C sampled': C_array_sample,
                'C_errors sampled': C_error_array_sample  }

    saveDataToFile(data, path, filename)