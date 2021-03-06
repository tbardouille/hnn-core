'''
Here we run a number of simulations over a range of 
input parameter values

Author: Tim Bardouille <tim.bardouille@dal.ca>
Adapted from:
		Mainak Jas <mainak.jas@telecom-paristech.fr>
       	Sam Neymotin <samnemo@gmail.com>

'''

###############################################################################
# Imports and setup

import os.path as op
import os, shutil

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

import hnn_core
from hnn_core import simulate_dipole, read_params, read_dipole, Network

# Directories and Neuron
hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')


###############################################################################
# User-defined variables
startingParameterFile = 'AlphaAndBeta_testScript.param'
paramsOfInterest = ['input_prox_A_weight_L2Pyr_ampa', 
	'input_prox_A_weight_L5Pyr_ampa',
	'input_dist_A_weight_L2Pyr_ampa',
	'input_dist_A_weight_L5Pyr_ampa']
paramMin = 0.1e-5
paramMax = 9.4e-5
numberOfSteps = 10
parameterScale = 'linear'

dplDirName = 'pyr_ampa'

plotOK = True

# Folder for storing dipole txt files
outDir = '/Users/tbardouille/Documents/Work/Projects/Spectral Events/HNN/Data'

###############################################################################
# Functions
def adjust_param_and_simulate_dipole(thisParams, paramsOfInterest, paramValue, outDir):

	for paramOfInterest in paramsOfInterest:
		thisParams.update({paramOfInterest: paramValue})

	print(thisParams)

	net = Network(thisParams)
	dpl = simulate_dipole(net)

	return dpl

def adjust_param_and_simulate_dipole_mp(paramValue):

	for paramOfInterest in paramsOfInterest:
		params.update({paramOfInterest: paramValue})

	net = Network(params)
	dpl = simulate_dipole(net)

	out_fname = op.join(dplPath, "{0:.5e}.txt".format(paramValue))

	dpl[0].write(out_fname)

	return dpl

def read_dipoles_over_range(parameterValues, plotOK):
	
	# Read simulations
	dpls = []
	for p in parameterValues:
		fname = op.join(dplPath, "{0:.5e}.txt".format(p))
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
		plt.xlabel('Time [ms]')
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

#Function returns maximum correlation coefficient, temporal shift to give max corr and number of local maxima
def get_outputs(parameterValues, dpls):
    
    correlations = []
    temporal_shifts = [] 
    num_maximas = []

    for dpl in dpls:
        #Start by finding the number of local maxima in the signal 
        num_maxima = len(ss.find_peaks(new_vals)[0])
    
        #Now if I have to systematically shift the new data I can do this:
        new_vals = dpl.dpl['agg'][1::]
        sub_corrs = []

        for x in range(0,710):
            #This takes the 40th sample until the end of the array - 40 is chosen becuase there are 40 samples of data/ ms and I want to shift it 1 ms at a time
            new_vals = new_vals[40:]
            #When you do this you have to zero pad the end of the array with 40 zeros so it is the same length as the one you are comparing it to   
            new_vals  = np.pad(new_vals, (0,40), 'constant')
        
            #Calculate the correlation coefficient matrix between values 
            corr_coef = np.corrcoef(init_vals, new_vals)
        
            #Pull out the single corrcoef value of interest 
            corr_coef = corr_coef[0,1]
            sub_corrs.append(corr_coef)
        
        #find the maximum correlation coefficient
        max_corr = max(sub_corrs)
        #find the shift that corresponds to the maximum correlation
        shift = np.where(sub_corrs==max(sub_corrs))[0][0]  
    
        #append the max correlation and associated shift to lists
        correlations.append(max_corr)
        temporal_shifts.append(shift)
        num_maximas.append(num_maxima)
    
    if plotOK:
        #Plot the correlations versus the input vals (delays in start times)
        plt.plot(parameterValues, correlations)
        plt.xlabel('Start times (ms)')
        plt.ylabel('Max Correlation Coefficient')

        #Plot the temporal shift versus the input vals (delays in start times)
        plt.plot(parameterValues, temporal_shifts)
        plt.xlabel('Delay (ms)')
        plt.ylabel('Temporal Shift (ms)')

        #Plot the number of local maxima versus the input vals (delays in start times)
        plt.plot(parameterValues, num_maximas)
        plt.xlabel('Delay (ms)')
        plt.ylabel('Number of Local Maxima')
    
    return correlations, temporal_shifts, num_maximas
    

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
# Run simulations consecutively
# Loop over parameter range and generate dipole simulations
dpls = []
for p in parameterValues:
	dpls.append(adjust_param_and_simulate_dipole(params, paramsOfInterest, p, outDir))
'''

# Run simulations in parallel
# Make the folder for the dpl files
dplPath = op.join(outDir, dplDirName)
if not os.path.exists(dplPath):
    os.makedirs(dplPath)
else:
	# Remove all the subdirectories first
    shutil.rmtree(dplPath)           
    os.makedirs(dplPath)
# Set up the parallel task pool for dipole simulations
count = int(np.round(mp.cpu_count()*1/2))
print(count)
pool = mp.Pool(processes=count)
# Run the jobs
pool.map(adjust_param_and_simulate_dipole_mp, parameterValues)

# Read simulated dipole timecourses for all parameter values
dpls = read_dipoles_over_range(parameterValues, plotOK)

# Read smoother dipole amplitude ranges for parameters
ranges = get_smoothed_ranges(parameterValues[0:-1], dpls[0:-1])

#Calculate outputs for parameter
correlations, temporal_shifts, num_maximas = get_outputs(parameterValues[0:-1], dpls[0:-1])


