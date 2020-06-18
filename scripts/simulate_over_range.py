'''
Here we run a number of simulations over a range of 
input parameter values

Author: Tim Bardouille <tim.bardouille@dal.ca>
Adapted from:
		Mainak Jas <mainak.jas@telecom-paristech.fr>
       	Sam Neymotin <samnemo@gmail.com>

'''

startingParameterFile = 'AlphaAndBeta_testScript.param'
paramOfInterest = 'input_prox_A_weight_L2Pyr_ampa'
paramMin = 0.1e-5
paramMax = 9.4e-5
numberOfSteps = 10
parameterScale = 'linear'

plotOK = True

###############################################################################
# Imports and setup

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

import hnn_core
from hnn_core import simulate_dipole, read_params, read_dipole, Network

# Directories and Neuron
hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')

# Folder for storing dipole txt files
outDir = '/Users/tbardouille/Documents/Work/Projects/Spectral Events/HNN/Data'


###############################################################################
# Functions
def adjust_param_and_simulate_dipole_mp(thisParams, paramOfInterest, paramValue, outDir):

	thisParams.update({paramOfInterest: paramValue})

	print(thisParams)

	net = Network(thisParams)
	dpl = simulate_dipole(net)

	out_fname = op.join(outDir, "{0:.5e}.txt".format(paramValue))

	return dpl

def adjust_param_and_simulate_dipole_mp(paramValue):

	params.update({paramOfInterest: paramValue})

	net = Network(params)
	dpl = simulate_dipole(net)

	out_fname = op.join(outDir, "{0:.5e}.txt".format(paramValue))

	dpl[0].write(out_fname)

	return dpl

def read_dipoles_over_range(parameterValues, outDir, plotOK):
	
	# Read simulations
	dpls = []
	for p in parameterValues:
		fname = op.join(outDir, "{0:.5e}.txt".format(p))
		dpls.append(read_dipole(fname))

	if plotOK:
		# Plot the results for all parameter values
		dipAmp = []
		for dpl in dpls:
			dipAmp.append(dpl.dpl['agg'])
		dipAmp = np.asarray(dipAmp).T
		plt.plot(dpls[0].t, dipAmp)
		plt.xlim(50, 700)
		plt.ylim(-500,500)
		plt.xlabel('Time [s]')
		plt.ylabel('Dipole Strength [nAm]')
		plt.legend(parameterValues)
		plt.show()

	return dpls

def get_smoothed_ranges(parameterValues, dpls):

	ranges = []
	for dpl in dpls:
		# Note: skip the 0th sample here due to its offset
		dipAmp_sorted = np.sort(dpl.dpl['agg'][1::])
		ranges.append(np.mean(dipAmp_sorted[-11:-1]) - np.mean(dipAmp_sorted[0:10]))

	if plotOK:
		plt.plot(parameterValues, ranges)
		plt.xlabel('Parameter Value')
		plt.ylabel('Dipole Value Range [nAm]')
		plt.show()

	return ranges

###############################################################################
# Main Program

# Read the default parameter file 
params_fname = op.join(hnn_core_root, 'param', startingParameterFile)
params = read_params(params_fname)

# Setup parameter range
if parameterScale == 'linear':
	parameterValues = np.linspace(paramMin, paramMax, numberOfSteps, endpoint=True)
if parameterScale == 'log':
	parameterValues = np.logspace(paramMin, paramMax, numberOfSteps, endpoint=True)


'''
# Run simulations - Loop over parameter range and generate dipole simulations
dpls = []
for p in parameterValues:
	dpls.append(adjust_param_and_simulate_dipole(params, paramOfInterest, p, outDir))
'''

# Run simulations
# Set up the parallel task pool for dipole simulations
count = int(np.round(mp.cpu_count()*1/3))
print(count)
pool = mp.Pool(processes=count)
# Run the jobs
pool.map(adjust_param_and_simulate_dipole_mp, parameterValues)


# Read simulated dipole timecourses for all parameter values
dpls = read_dipoles_over_range(parameterValues, outDir, plotOK)

# Read smoother dipole amplitude ranges for parameters
ranges = get_smoothed_ranges(parameterValues[0:-1], dpls[0:-1])



