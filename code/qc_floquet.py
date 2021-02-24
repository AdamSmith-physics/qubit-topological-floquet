"""
    If you use any part of this code please cite [arXiv:2012.01459]

    This file contains the following functions:
    
    -- miscellaneous functions --
    save_obj
    load_obj
    create_directory
    saveDataToFile
    get_closest_multiple_of_16

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
