"""
Script used to plot inset of Fig.2 of [arXiv:2012.01459]
"""

import pickle
import os
import numpy as np

from qutip import Bloch as Bloch

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


nice_blue = 0.9*np.array([94., 138., 210.]) / 256
nice_green = 1.3*np.array([82., 112., 63.]) / 256
white = np.array([1., 1., 1.])

path = 'data/data_2020-10-27'
master_filename = 'target_h_t20_m3_omega0-125_90pc'

# load experimental data for plotting
real_data = load_obj_local(path + '/'+ master_filename + '_real')

correct_results_real = real_data['corrected results']
pure_results_real = pure_results(correct_results_real)

# plot the path on Bloch sphere!
u = np.linspace(0, np.pi, 15)
v = np.linspace(0, 2 * np.pi, 15)

bloch_x = np.outer(np.sin(u), np.sin(v))
bloch_y = np.outer(np.sin(u), np.cos(v))
bloch_z = np.outer(np.cos(u), np.ones_like(v))


fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_wireframe(bloch_x, bloch_y, bloch_z, color='k',alpha=0.2)
ax.scatter(pure_results_real['x'], pure_results_real['y'], pure_results_real['z'], marker='.', c=nice_blue)
ax.text(-.2, 0, 1.25, r'$|0\rangle$')
ax.text(-.2, 0, -1.65, r'$|1\rangle$')
ax.set_box_aspect([1,1,1])
plt.axis('off')

myMacRatio = 1680/1280  # this is to get the figure to render properly on my scaled mac screen.
singleColumnWidth = myMacRatio * (3. + 3/8)

fig = plt.gcf()
fig.set_size_inches(singleColumnWidth/1.5, singleColumnWidth/1.5/1.6)
plt.tight_layout(pad=0.1)
plt.show()
