"""
    This code was written for qiskit version 0.19.6
    It may need updating for future versions of qiskit

    If you use any part of this code please cite [arXiv:2012.01459]

    This file augments qc_floquet.py with functions that are dependent on qiskit

    This file contains the following functions:

    -- gate pulse sequences --
    hadamard_pulse
    u3_pulse
    u2_pulse
    rx_pulse
    ry_pulse
    s_pulse
    s_dag_pulse

    -- device setup --
    setupArmonkBackend_real
    setupArmonkBackend_fake

    -- run simulation --
    qiskit_simulation

"""


#### imports #####################################

import numpy as np
import itertools
import pickle
import math
import qutip as qt
from scipy import integrate

from qiskit.providers.aer import PulseSimulator  # The pulse simulator
from qiskit.providers.aer.pulse import PulseSystemModel  # Object for representing physical models
from qiskit.test.mock.backends.armonk.fake_armonk import FakeArmonk  # Mock Armonk backend
from qiskit.providers.ibmq.managed import IBMQJobManager

# main class for accessing the real IBM quantum computers
from qiskit import IBMQ

# required functions for pulse control
from qiskit import pulse            
from qiskit.pulse import Play
from qiskit.pulse import pulse_lib  
from qiskit.pulse import DriveChannel
from qiskit.scheduler import measure
from qiskit import assemble
from qiskit.tools.monitor import job_monitor
from qiskit import execute

from matplotlib import pyplot as plt

# This is the 3D plotting toolkit
from mpl_toolkits.mplot3d import Axes3D

import os
import time


#### global variables ###############################

# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds

bases = ['x','y','z']
paulis = np.array([ [[0,1],[1,0]], [[0,-1j],[1j,0]] , [[1,0],[0,-1]] ])


#### gate pulse sequences ############################

def hadamard_pulse(backend_defaults, drive_chan):
    """ Hadamard pulse using H=Ry(pi/2) """
    return ry_pulse(np.pi/2, backend_defaults, drive_chan)

def u3_pulse(theta, phi, lmbda, backend_defaults, drive_chan):
    """ u3 gate using the device instruction schedule map """
    inst_map = backend_defaults.instruction_schedule_map
    this_schedule = inst_map.get('u3', 0, P0=theta, P1=phi, P2=lmbda)

    return this_schedule

def u2_pulse(phi, lmbda, backend_defaults, drive_chan):
    """ u3 gate using the device instruction schedule map """
    inst_map = backend_defaults.instruction_schedule_map
    this_schedule = inst_map.get('u2', 0, P0=phi, P1=lmbda)

    return this_schedule

def rx_pulse(theta, backend_defaults, drive_chan):
    # 𝑈3(𝜃,−𝜋/2,𝑝𝑖/2)=𝑅𝑋(𝜃)
    return u3_pulse(theta, -np.pi/2, np.pi/2, backend_defaults, drive_chan)

def ry_pulse(theta, backend_defaults, drive_chan):
    # 𝑈3(𝜃,0,0)=𝑅𝑌(𝜃)
    return u3_pulse(theta, 0, 0, backend_defaults, drive_chan)

def s_pulse(drive_chan):
    return pulse.ShiftPhase(np.pi/2, drive_chan)

def s_dag_pulse(drive_chan):
    return pulse.ShiftPhase(-np.pi/2, drive_chan)


#### device setup ##################################

def setupArmonkBackend_real(verbose = False):
    """ Setup the real armonk backend. 
    loads IBMQ account if not already loaded and extracts qubit and Rabi frequencies from device.
    
    Returns
    -------
    real_backend : backend for the real armonk device
    freq_est: qubit : frequency
    drive_stength_est : Rabi frequency for the qubit
    dt : time resolution for the pulse samples
    """

    backend_name = 'ibmq_armonk'

    if IBMQ.active_account() is None:
        IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    real_backend = provider.get_backend(backend_name)

    real_backend_config = real_backend.configuration()
    real_backend_defaults = real_backend.defaults()
    assert real_backend_config.open_pulse, "Backend doesn't support Pulse"

    freq_est = real_backend_defaults.qubit_freq_est[0] 
    drive_strength_est = getattr(real_backend_config, 'hamiltonian')['vars']['omegad0'] * GHz
    dt = real_backend_config.dt

    if verbose:
        print(f"omega_0 = {freq_est}")
        print(f"Rabi frequency = {drive_strength_est}")
        print(f"Sampling time: {dt*1e9} ns") 

    print("Real backend loaded!")

    return real_backend, freq_est, drive_strength_est, dt

def setupArmonkBackend_fake(fromReal = True):
    """ Setup the fake armonk backend. 
    loads IBMQ account if not already loaded and extracts real qubit and Rabi frequencies from device.
    The real device information is used to define a simulation model for the armonk device
    
    Returns
    -------
    fake_backend : backend for the fake armonk device
    freq_est: qubit : frequency
    drive_stength_est : Rabi frequency for the qubit
    dt : time resolution for the pulse samples
    armonk_model : model of the armonk device for qiskit simulation runs
    """

    if IBMQ.active_account() is None:
        IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    real_backend = provider.get_backend('ibmq_armonk')

    fake_backend = FakeArmonk()

    getattr(fake_backend.configuration(), 'hamiltonian')['vars']['omegad0'] \
        = getattr(real_backend.configuration(), 'hamiltonian')['vars']['omegad0'] * GHz 
    drive_strength_est = getattr(fake_backend.configuration(), 'hamiltonian')['vars']['omegad0']

    getattr(fake_backend.configuration(), 'hamiltonian')['vars']['wq0'] \
        = getattr(real_backend.configuration(), 'hamiltonian')['vars']['wq0'] * GHz
    freq_est = getattr(fake_backend.configuration(), 'hamiltonian')['vars']['wq0'] / (2*np.pi)

    fake_backend.defaults().qubit_freq_est = [freq_est]

    fake_backend_config = fake_backend.configuration()
    fake_backend_defaults = fake_backend.defaults()
    assert fake_backend_config.open_pulse, "Backend doesn't support Pulse"
    armonk_model = PulseSystemModel.from_backend(fake_backend)

    dt = real_backend.configuration().dt

    print("Fake backend loaded!")

    return fake_backend, freq_est, drive_strength_est, dt, armonk_model


#### run simulation #####################################

def qiskit_simulation(drive, drive_lengths, dt, psi0, backend, realDevice=True, armonk_model=None, num_shots = 8192):
    # Note maximum (combined) number of samples for a job is 258144 !! (round to 258000)
    # also limit of 75 circuits

    # preliminaries
    num_points = len(drive_lengths)
    drive_chan = DriveChannel(0)  # set up drive channel for single qubit
    theta, phi, lmbda = psi0_rotation(psi0)  # get angles to rotate to initial instantaneous eigenstate


    #### Construct the circuits #########

    measure_duration = 16000  # Found empirically, may change!
    
    # create an array of schedule for different bases and simulation lengths
    schedules = []
    track_basis = []
    track_steps = []
    for basis in bases:
        for ii in range(num_points):
                drive_length = drive_lengths[ii]
                drive_samples = get_closest_multiple_of_16(drive_length * us / dt)   # The truncating parameter in units of dt
                
                this_schedule = pulse.Schedule(name=f"drive length = {drive_length}, basis = {basis}")  # create new schedule

                # rotate to instantaneous eigenstate (if it is not [1,0])
                if not psi0[0] == 1:
                    this_schedule += u3_pulse(theta, phi, lmbda, backend.defaults(), drive_chan)

                # apply the time dependent drive corresponding to the Hamiltonian
                if drive_length > 0:
                    rabi_pulse = pulse_lib.SamplePulse(drive[0:drive_samples]) 
                    this_schedule += Play(rabi_pulse, drive_chan)
                
                # basis change
                if basis == 'x':
                    this_schedule += hadamard_pulse(backend.defaults(), drive_chan)
                elif basis == 'y':
                    this_schedule += s_dag_pulse(drive_chan)
                    this_schedule += hadamard_pulse(backend.defaults(), drive_chan)
                elif basis == 'z':
                    pass  # do nothing!
                else:
                    raise ValueError("Invalid basis!!")

                # Add the measure schedule
                this_schedule += measure([0], backend) << this_schedule.duration
                schedules.append(this_schedule)
                track_basis.append(basis)
                track_steps.append(ii)

    
    #### Group and assemble the schedule ########
    
    """
    Here we assemble to schedules ("circuits") and group them into jobs.
    These jobs are grouped such that we maximise the total number of samples (approx 258000)
    while staying within the 75 circuit limit for a job.
    """

    num_circuits = 0
    total_samples = 0
    circuit_list = []
    temp_circuit_dict = {'circuits': [], 'info': []}
    for ii in range(len(schedules)):

        if ((num_circuits + 1) > 75) \
                or (total_samples + schedules[ii].duration - measure_duration > 258000):
            # limits reached. Put grouped schedules into job
            print(f"Splitting with num_circuits = {num_circuits}, total_samples = {total_samples}")
            assembled_circuits = assemble(temp_circuit_dict['circuits'],
                                        backend=backend,
                                        meas_level=2,
                                        meas_return='avg',
                                        shots=num_shots)
            temp_circuit_dict['circuits'] = assembled_circuits
            circuit_list.append(temp_circuit_dict)

            # reset temp arrays / dicts
            temp_circuit_dict = {'circuits': [], 'info': []}
            num_circuits = 0
            total_samples = 0
        
        # add the next schedule to the current group
        temp_circuit_dict['circuits'].append(schedules[ii])
        temp_circuit_dict['info'].append({'basis': track_basis[ii], 'duration': schedules[ii].duration, 'time step': track_steps[ii]})
        num_circuits += 1
        total_samples += schedules[ii].duration - measure_duration

    # assemble final schedule and job
    assembled_circuits = assemble(temp_circuit_dict['circuits'],
                                        backend=backend,
                                        meas_level=2,
                                        meas_return='avg',
                                        shots=num_shots)
    temp_circuit_dict['circuits'] = assembled_circuits
    circuit_list.append(temp_circuit_dict)
    print(f"Final set! Splitting with num_circuits = {num_circuits}, total_samples = {total_samples}")
    
    total_num_jobs = len(circuit_list)
    print(f"\nJob has been split into {total_num_jobs} jobs")


    #### Run the circuits #############
    
    """
    Here we use our own job manager to run the jobs, 
    constantly maximising number of jobs in the queue.
    We submit as many simulataneous jobs to the queue as possible then wait for results. 
    Whenever we obtain results from a job we add a new job to the queue.
    """

    # run in series on simulator
    if realDevice:
        max_jobs = 5
    else:
        max_jobs = 1

    # create arrays for keeping track of unsubmitted, submitted and finished jobs
    running_circuits = []
    finished_circuits = []
    while len(circuit_list) > 0 or len(running_circuits) > 0:
        print(f"\nNumber of circuits left: {len(circuit_list)}")
        print(f"Number of running circuits: {len(running_circuits)}")
        print(f"Number of finished circuits: {len(finished_circuits)}")

        # Get new Rabi frequency in case device config changes!
        _, _, drive_strength_est, _ = setupArmonkBackend_real()

        if realDevice:
            remaining_jobs = backend.remaining_jobs_count()
        else:
            remaining_jobs = 1 - len(running_circuits)

        if (remaining_jobs == 0 and len(running_circuits) > 0) or (len(running_circuits) == max_jobs) or (len(circuit_list)==0):
            # stop and track existing jobs if job limit is reached!
            print(f"Tracking job {len(finished_circuits)+1} / {total_num_jobs}")
            try:
                # try to get the results from IBM
                # try / except clause used due to occasional connection issues with IBM server
                job_monitor(running_circuits[0]['job'])
                results = running_circuits[0]['job'].result(timeout=120)
                del running_circuits[0]['job']
                running_circuits[0]['results'] = results
                finished_circuits.append(running_circuits.pop(0))
                print("Job successful!")
            except:
                # if there is an error put the circuit back in the list of circuits to be run!
                print("Job failed! Trying again!")
                del running_circuits[0]['job']
                circuit_list.insert(0,running_circuits.pop(0))

        # in case we are running other jobs on the same account
        elif remaining_jobs == 0 and len(running_circuits) == 0:
            print("Something else is filling the queue!")
            time.sleep(120)  # wait 2 minutes.

        if len(circuit_list) > 0:
            # run another job if there's more in the circuit list.
            if realDevice:
                try:
                    running_circuits.append(circuit_list.pop(0))
                    running_circuits[-1]['job'] = backend.run(running_circuits[-1]['circuits'])
                    running_circuits[-1]['Rabi frequency'] = drive_strength_est
                except:
                    print("Submitting job failed! Trying again in 1 minute!")
                    circuit_list.insert(0,running_circuits.pop(-1))  # put circuit back in circuit_list
                    time.sleep(60)  # wait 1 minute.

            else:
                if armonk_model is not None:
                    backend_sim = PulseSimulator()
                    running_circuits.append(circuit_list.pop(0))
                    running_circuits[-1]['job'] = backend_sim.run(running_circuits[-1]['circuits'],armonk_model)
                    
                else:
                    raise ValueError("No armonk model provided!!!")


    #### process results #############################

    """
    Because of the circuits potentially being run out of order
    we must extract and reorganise the data from the experiments.
    """

    counts = {'x': [{}]*num_points, 'y': [{}]*num_points, 'z': [{}]*num_points}
    expectation = {'x': [0.]*num_points, 'y': [0.]*num_points, 'z': [0.]*num_points}
    Rabi_freqs = {'x': [0.]*num_points, 'y': [0.]*num_points, 'z': [0.]*num_points}

    for ii in range(len(finished_circuits)):
        for jj in range(len(finished_circuits[ii]['info'])):

            temp_counts = finished_circuits[ii]['results'].get_counts(jj)

            sigma_z = 0

            if '0' in temp_counts.keys():
                sigma_z += temp_counts['0']/num_shots
            if '1' in temp_counts.keys():
                sigma_z -= temp_counts['1']/num_shots
            
            currentBasis = finished_circuits[ii]['info'][jj]['basis']
            currentStep = finished_circuits[ii]['info'][jj]['time step']
            expectation[currentBasis][currentStep] = np.array(sigma_z)
            counts[currentBasis][currentStep] = temp_counts
            Rabi_freqs[currentBasis][currentStep] = finished_circuits[ii]['Rabi frequency']

    return expectation, counts, Rabi_freqs

