"""
    This code was written for qiskit version 0.19.6
    It may need updating for future versions of qiskit

    If you use any part of this code please cite [arXiv:2012.XXXXX]

    This file contains the following functions:
    
    -- miscellaneous functions --
    save_obj
    load_obj
    create_directory
    saveDataToFile
    get_closest_multiple_of_16

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

    -- drive samples and effective Hamiltonian --
    random_h
    constant_h
    create_omega_ramp
    target_h
    h1
    h2
    h1_dot
    h2_dot
    target_h_t
    get_drive_from_hs
    get_experimental_hs (deprecated)
    instantaneous_eigenstate
    psi0_rotation

    -- run simulation --
    qiskit_simulation

    -- exact simulation --
    exph
    get_expectation_values

    -- save to file --
    corrected_results
    pure_results

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


#### miscellaneous functions ##################################

# Functions to save and load python type objects to file using pickle.
def save_obj(data, filename ):
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=4)  # protocol 4 needed for compatability with python3.6
def load_obj(filename ):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)

def create_directory(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed. Might already exist!" % path)
    else:
        print ("Successfully created the directory %s " % path)

def saveDataToFile(data, path, filename):
    save_obj(data, path + '/' + filename)

    print("Data saved to file!")

# samples need to be multiples of 16
def get_closest_multiple_of_16(num):
    if num == 0:
        return 0
    else:
        return max(int(num + 8 ) - (int(num + 8 ) % 16), 64)


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


#### drive samples and effective Hamiltonian ######################

def random_h( num_points_random, total_samples, max_drive_strength, seed=None ):
    """ Constructs a piecewise linear random magentic field for the qubit.

    Parameters
    ----------
    num_points_random : number of piecewise linear segments
    total_samples : specifies total number of samples for the drive
    max_drive_strength : max drive strength given as a fraction times the Rabi frequency (in Hz)
    seed : sets the seed for random number generator to allow reproducable results

    Returns
    -------
    hx, hy, hz :  the x, y, and z components of the random field each as numpy arrays of length total samples
    """

    step_size = math.ceil(total_samples/num_points_random)

    # h uniformly distributed in unit sphere
    if seed is not None:
        np.random.seed(seed) # make it reproducible
    U = np.random.rand(num_points_random+1)**(1/3)
    h = np.random.normal(0,1,size=(3,num_points_random+1))
    h = U * h / np.linalg.norm(h,axis=0) * max_drive_strength

    hx = []; hy = []; hz = []
    for ii in range(num_points_random):
        hx += list(np.linspace(h[0,ii], h[0,ii+1], num=step_size, endpoint=False))
        hy += list(np.linspace(h[1,ii], h[1,ii+1], num=step_size, endpoint=False))
        hz += list(np.linspace(h[2,ii], h[2,ii+1], num=step_size, endpoint=False))
    
    # force arrays to have correct length!
    hx = np.array(hx)[:total_samples]; hy = np.array(hy)[:total_samples]; hz = np.array(hz)[:total_samples]

    return hx, hy, hz

def constant_h(vector, total_samples, max_drive_strength ):
    """ Constant magnetic field along a given vector

    Parameters
    ----------
    vector : unnormalised vector (list) specifying direction of magnetic field
    total_samples : specifies total number of samples for the drive
    max_drive_strength : max drive strength given as a fraction times the Rabi frequency (in Hz)

    Returns
    -------
    hx, hy, hz : the x, y, and z components of the magnetic field each as numpy arrays of length total samples
    """
    vector = np.array(vector)
    vector = vector/np.linalg.norm(vector)  # h has unit length

    hx = np.array([vector[0]]*total_samples) * max_drive_strength
    hy = np.array([vector[1]]*total_samples) * max_drive_strength
    hz = np.array([vector[2]]*total_samples) * max_drive_strength

    return hx, hy, hz

def create_omega_ramp(omega, times, ramp_time):
    """ creates an array of omega values that ramp up linearly over ramp time

    Parameters
    ----------
    omega : the target value for omega
    times : the sample times
    ramp_time : the time over which to linearly ramp up the value of omega

    Returns
    -------
    omega : a numpy array same length as times that has a linear ramp from 0 to omega over ramp_time
    """
    ramp_sample = np.sum(times < ramp_time)
    omega_ramped = np.array([1.]*len(times))
    theRamp = np.linspace(0.,1.,ramp_sample,endpoint=False)
    omega_ramped[:len(theRamp)] = theRamp
    omega = omega*omega_ramped

    return omega

def target_h(m, eta, omega1, omega2, phi1, phi2, times, max_drive_strength, ramp_time=None):
    """ The time dependent magnetic field corresponding to the Hamiltonian 
    in [PRX 7, 041008 (2017)] Eq.(28)
    and [arXiv:2012.XXXXX] Eq.(7)

    Includes a linear ramp of omega1 and omega2 to reduce transient effects from sudden quench

    Parameters
    ----------
    m : constant sigma_z field
    eta : overall scale factor of magnetic field (keep as 1)
    omega1 : frequency of drive 1
    omega2 : frequency of drive 2 (typically golden ratio times omega1)
    phi1 : frequency shift of drive 1
    phi2 : frequency shift of drive 2
    times : samples times
    max_drive_stength : max drive strength given as a fraction of the Rabi frequency (in Hz)
    ramp_time : (default None) time scale over which to linearly ramp omega1 and omega2

    Returns
    -------
    hx, hy, hz : the x, y, and z components of the magnetic field each as numpy arrays of length len(times)
    """

    if ramp_time is not None:
        omega1 = create_omega_ramp(omega1, times, ramp_time)
        omega2 = create_omega_ramp(omega2, times, ramp_time)

    hx = eta * np.sin( (omega1*max_drive_strength) * times + phi1) * max_drive_strength 
    hy = eta * np.sin( (omega2*max_drive_strength) * times + phi2) * max_drive_strength 
    hz = eta * (m - np.cos( (omega1*max_drive_strength) * times + phi1) 
                -np.cos( (omega2*max_drive_strength) * times + phi2)) * max_drive_strength 

    return hx, hy, hz

def h1(m, eta, omega1, phi1, times, max_drive_strength, ramp_time=None):
    """ The time dependent magnetic field corresponding to drive 1 in the Hamiltonian 
    in [PRX 7, 041008 (2017)] Eq.(28)

    Parameters
    ----------
    -- see target_h --

    Returns
    -------
    hx, hy, hz : the x, y, and z components of the magnetic field each as numpy arrays of length len(times)
    """

    if ramp_time is not None:
        omega1 = create_omega_ramp(omega1, times, ramp_time)

    hx = eta * np.sin( (omega1*max_drive_strength) * times + phi1) * max_drive_strength 
    hy = 0 * times
    hz = eta * (m/2 - np.cos( (omega1*max_drive_strength) * times + phi1)) * max_drive_strength 

    return hx, hy, hz

def h2(m, eta, omega2, phi2, times, max_drive_strength, ramp_time=None):
    """ The time dependent magnetic field corresponding to drive 2 in the Hamiltonian 
    in [PRX 7, 041008 (2017)] Eq.(28)

    Parameters
    ----------
    -- see target_h --

    Returns
    -------
    hx, hy, hz : the x, y, and z components of the magnetic field each as numpy arrays of length len(times)
    """

    if ramp_time is not None:
        omega2 = create_omega_ramp(omega2, times, ramp_time)

    hx = 0 * times
    hy = eta * np.sin( (omega2*max_drive_strength) * times + phi2) * max_drive_strength 
    hz = eta * (m/2 -np.cos( (omega2*max_drive_strength) * times + phi2)) * max_drive_strength 

    return hx, hy, hz

def h1_dot(m, eta, omega1, phi1, times, max_drive_strength, ramp_time=None):
    """ Time derivate of the Hamiltonian due to drive 1
    in [PRX 7, 041008 (2017)] Eq.(28)

    Parameters
    ----------
    -- see target_h --

    Returns
    -------
    hx, hy, hz : the x, y, and z components of the magnetic field each as numpy arrays of length len(times)
    """

    if ramp_time is not None:
        omega1 = create_omega_ramp(omega1, times, ramp_time)

    hx = eta * (omega1*max_drive_strength) * np.cos( (omega1*max_drive_strength) * times + phi1) * max_drive_strength 
    hy = 0 * times
    hz = eta * (omega1*max_drive_strength) * np.sin( (omega1*max_drive_strength) * times + phi1) * max_drive_strength 

    return hx, hy, hz

def h2_dot(m, eta, omega2, phi2, times, max_drive_strength, ramp_time=None):
    """ Time derivate of the Hamiltonian due to drive 2
    in [PRX 7, 041008 (2017)] Eq.(28)

    Parameters
    ----------
    -- see target_h --

    Returns
    -------
    hx, hy, hz : the x, y, and z components of the magnetic field each as numpy arrays of length len(times)
    """

    if ramp_time is not None:
        omega2 = create_omega_ramp(omega2, times, ramp_time)

    hx = 0 * times
    hy = eta * (omega2*max_drive_strength) * np.cos( (omega2*max_drive_strength) * times + phi2) * max_drive_strength 
    hz = eta * (omega2*max_drive_strength) * np.sin( (omega2*max_drive_strength) * times + phi2) * max_drive_strength 

    return hx, hy, hz

def target_h_t(m, eta, omega1, omega2, phi1, phi2, t, max_drive_strength, ramp_time=None):
    """ Return the Hamiltonian at a given time t
    """
    # not set up for ramping!
    return eta*np.array([ np.sin(omega1*t+phi1), np.sin(omega2*t+phi2),
                          m-np.cos(omega1*t+phi1)-np.cos(omega2*t+phi2) ]) * max_drive_strength

def get_drive_from_hs(hx, hy, hz, dt, max_drive_strength, strength=0.75):
    """ Converts a time dependent effective Hamiltonian into the corresponding qubit drive

    Parameters
    ----------
    hx, hy, hz : the x, y and z components of the effective Hamiltonian
    dt : time step for pulse samples
    max_drive_stength : max drive strength given as a fraction of the Rabi frequency (in Hz)
    strength : (deprecated) drive strength fraction 

    Returns
    -------
    drive : the corresponding drive pulse sequench for the qubit as numpy array
    """

    drive = (hx - 1j*hy) * np.exp(2j*np.cumsum(hz*dt))  # cumsum used since hz is piecewise constant!
    return drive

def get_experimental_hs(hs, dt):
    """ (deprecated) takes the desired target magnetic field and returns the one sent to the device 
    Replaced by get_drive_from_hs
    """
    phases = np.exp(2j*np.cumsum(hs[2]*dt))
    hMinus = (hs[0]-1j*hs[1])*phases
    dx = np.real(hMinus)
    dy =-np.imag(hMinus)
    actualHs = np.array([ dx, dy, np.zeros_like(dx)])
    return actualHs

def instantaneous_eigenstate(hs):
    """ Returns the instantaneous eigenstate psi0 of the Hamiltonian with parameters hs = array([hx,hy,hz])
    """
    psi = np.array([1. + 0*1j,0.])
    magnitude = np.linalg.norm(hs)

    psi[1] = (magnitude - hs[2])/(hs[0] - 1j*hs[1])
    psi = -psi / np.linalg.norm(psi)

    return psi

def psi0_rotation(psi0):
    """ Finds the rotation angles for a u3 rotation to get the initial state psi0
    """
    theta = 2*np.arccos(np.real(psi0[0]))
    sgn = np.sign(np.sin(theta/2))
    lmbda =  -sgn*np.log(psi0[1]/np.abs(psi0[1])).imag
    phi = 0

    return theta, phi, lmbda

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


#### exact simulation ##########################

def exph(h, dt): # hdt = h*dt, which is dimensionless
    """
    Returns the matrix exponential exp(-1j h dt)
    """
    hdt = h*dt
    mag = np.linalg.norm(hdt, axis=0)

    return (np.tensordot(np.cos(mag),np.eye(2),axes = 0)
           -np.tensordot(np.transpose(1j*np.sin(mag)/mag*hdt), paulis, axes=1))


def get_expectation_values(hs, dt, initialState, t1=0, t2=0):
    """ 
    Returns X, Y and Z expectation value under evolution with hs.
    Has optional arguments t1 and t2 to simulate a qubit with 
    finite T1 and T2 times.
    """
    tlist = dt*np.arange(len(hs[0]))
    psi0 = qt.Qobj(initialState)
    Heff  = [[qt.sigmax(), hs[0]], [qt.sigmay(), hs[1]], [qt.sigmaz(), hs[2]]]
    if t1==0 and t2==0:
        c_ops = []
    else:
        Gamma1 = 1/t1 # energy relaxation rate
        Gamma2 = 1/t2 # transverse relaxation rate
        GammaPhi = Gamma2-Gamma1/2 # pure dephasing part of Gamma2
        c_ops = [ np.sqrt(GammaPhi)*qt.sigmaz(), np.sqrt(Gamma1)*qt.sigmam() ]
    result = qt.mesolve(Heff , psi0, tlist, c_ops, [qt.sigmax(), qt.sigmay(), qt.sigmaz()])
    return np.array(result.expect)


#def get_expectation_values(hs, dt, initialState):
#    """ calculates expectation values under ideal evolution """
#    # calculate unitaries     unitaryList = exph(hs, dt)
#    fullUnitaryList = np.array(list(itertools.accumulate(unitaryList, func=lambda x, y: y.dot(x))))
#
#    # set the state to be considered
#    psi0 = initialState
#    
#    # calculate expectation values
#    expect = []
#    for op in [ paulis[0], paulis[1], paulis[2] ]:
#        expect.append([ np.conj(psi0) @ np.conj(np.transpose(u)) @ op @ u @ psi0 for u in fullUnitaryList ])
#
#    return np.array(expect)


#### save to file #####################

def corrected_results(results, drive_lengths, hz, dt):
    """ correct the results for the rotating frame!
    See [arXiv:2012.XXXXX] Eq.(14)

    Parameters
    ----------
    results : the time-resolved x, y and z pauli expectation values as 2D numpy array
    drive_lengths : the corresponding simulation times
    hz : the z component of the Hamiltonian for each time
    dt : the step size

    Returns
    -------
    corrected_results : the rotated expectation values as dict with keys 'x', 'y' and 'z'
    rotation_z : the rotation around z-axis that was used, expressed as a complex phase.
    """
    rotation_z_full = np.exp(2j*np.cumsum(hz*dt))
    rotation_z = []
    for drive_length in drive_lengths:
        drive_samples = get_closest_multiple_of_16(drive_length * us /dt)   # The truncating parameter in units of dt
        if drive_samples == 0:
            rotation_z += [1.+0.*1j]
        else:
            rotation_z += [rotation_z_full[drive_samples-1]]

    corrected_results = {   'x': np.real(rotation_z)*results['x'] - np.imag(rotation_z)*results['y'],
                            'y': np.imag(rotation_z)*results['x'] + np.real(rotation_z)*results['y'],
                            'z': results['z']}

    return corrected_results, rotation_z

def pure_results(results):
    """ results projected onto pure states on the bloch sphere!
    The expectation values are represented as a vector (<x>, <y>, <z>), which we normalise.
    
    Parameters
    ----------
    results : the time-resolved x, y and z pauli expectation values as 2D numpy array

    Returns
    -------
    project_results : the normalised expectation values as dict with keys 'x', 'y' and 'z'

    """
    res_x = np.array(results['x'])
    res_y = np.array(results['y'])
    res_z = np.array(results['z'])

    res = np.vstack((res_x,res_y,res_z))

    norm = np.linalg.norm(res, axis=0)

    projected_results = {   'x': res_x/norm,
                            'y': res_y/norm,
                            'z': res_z/norm}

    return projected_results
