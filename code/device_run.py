"""
This is the main script for running the experiment on the IBM devices.
"""

import numpy as np
from qc_floquet_qiskit import *
from qc_floquet import *

import datetime

from matplotlib import pyplot as plt


#### parameters ###########################
# change the name and m

real_backend, freq_est, drive_strength_est, dt = setupArmonkBackend_real()

num_points = 800
drive_length_max = 20  # drive time in us
ramp_time = 2000*dt
eta = 1
omega1 = 0.125
omega2 = omega1*(1+np.sqrt(5))/2


m_vals = [1.2]
filenames = ['target_h_t20_m1-2_omega0-125_90pc']


for kk in range(len(m_vals)):

    m = m_vals[kk]
    master_filename = filenames[kk]

    print(f"\nRun for m = {m}\n")

    #### load device details ###########################

    real_backend, freq_est, drive_strength_est, dt = setupArmonkBackend_real(verbose=True)

    drive_strength_est = drive_strength_est
    drive_strength_factor = 0.9 / np.sqrt(2)  # 100% plus 1/sqrt(2) scaling to normalise drive
    max_drive_strength = drive_strength_factor*drive_strength_est  # should really be called reference drive strength


    #### setup the drive ###########################

    drive_lengths = np.linspace(0, drive_length_max, num_points)
    total_samples = get_closest_multiple_of_16(drive_length_max * us /dt)

    drive_samples = [get_closest_multiple_of_16(x * us /dt) for x in drive_lengths]

    # real field!!
    times = np.array([x*dt for x in range(total_samples)])
    hx, hy, hz = target_h(m, eta, omega1, omega2, np.pi/10, 0, times, max_drive_strength, ramp_time=ramp_time)
    hs = np.array([hx,hy,hz])

    psi0 = instantaneous_eigenstate(hs[:,0])

    drive = get_drive_from_hs(hx, hy, hz, dt, max_drive_strength, strength=drive_strength_factor)


    #### run ###########################

    # run on device!
    results_real, counts_real, Rabi_freqs = qiskit_simulation(drive/drive_strength_est, drive_lengths, dt, psi0, real_backend, 
                                                        realDevice=True, num_shots=8192)

    # correct for rotating frame around z-axis (doesn't incorporate changing Rabi freqs) can be redone after
    corrected_results_real, rotation_z = corrected_results(results_real, drive_lengths, hz, dt)


    #### save data to file ###########################

    date_object = datetime.date.today()
    path = "data/data_" + str(date_object)

    create_directory(path)

    filename = master_filename + '_real'
    counts_dict = counts_real
    raw_results_dict = results_real
    corrected_results_dict = corrected_results_real
    h_parameters = {'m': m, 'eta': eta, 'omega1': omega1, 'omega2': omega2}
    data = {   'num_points': num_points, 
                    'drive_lengths': drive_lengths,
                    'drive_length_max': drive_length_max,
                    'ramp_time': ramp_time,
                    'dt': dt,
                    'rabi freqency': drive_strength_est,
                    'max_drive_strength': max_drive_strength,
                    'backend': 'ibmq_armonk', 
                    'counts':  counts_dict, 
                    'raw results': raw_results_dict,
                    'corrected results': corrected_results_dict,
                    'h': hs,
                    'h parameters': h_parameters,
                    'drive': drive,
                    'rotation z': rotation_z,
                    'Rabi_freqs': Rabi_freqs}
    saveDataToFile(data, path, filename)