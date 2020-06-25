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

import pandas as pd
import support_functions.spectralevents_functions as tse
import scipy.signal as ss

###############################################################################
# User-defined variables
startingParameterFile = 'AlphaAndBeta_testScript_60secondSingleTrial.param'
paramsOfInterest = ['input_prox_A_weight_L2Pyr_ampa', 
	'input_prox_A_weight_L5Pyr_ampa',
	'input_dist_A_weight_L2Pyr_ampa',
	'input_dist_A_weight_L5Pyr_ampa']
paramMin = 0.1e-5
paramMax = 9.4e-5
numberOfSteps = 10
parameterScale = 'linear'

dplDirName = 'pyr_ampa_60seconds'

plotOK = False

# Folder for storing dipole txt files
outDir = './' #'/Users/tbardouille/Documents/Work/Projects/Spectral Events/HNN/Data'

# Downsample factor prior to TFR analysis
downsamp = 20   # Go from 40,000Hz to 2,000Hz

# Frequency range defining events of interest
betaMin = 15
betaMax = 30

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

###############################################################################
# Main Program

dplPath = op.join(outDir, dplDirName)

# Read the default parameter file 
params_fname = op.join(hnn_core_root, 'param', startingParameterFile)
params = read_params(params_fname)

# Setup parameter range
if parameterScale == 'linear':
	parameterValues = np.linspace(paramMin, paramMax, numberOfSteps, endpoint=True)
if parameterScale == 'log':
	parameterValues = np.logspace(paramMin, paramMax, numberOfSteps, endpoint=True)

# Make the folder for the dpl files
if not os.path.exists(dplPath):
    os.makedirs(dplPath)
else:
	# Remove all the subdirectories first
    shutil.rmtree(dplPath)           
    os.makedirs(dplPath)

# Run simulations in parallel
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

# Get and example dipole timecourse and times for further analysis
#   Note dropping first two samples due to large offset
dplData = dpls[3].dpl['agg'][2::]
dplTimes = dpls[3].t[2::]

# Downsample the data for speed (there doesn't seem to be content above 2000 Hz sample rate anyway)
dplData_resamp = dplData[::downsamp]
dplTimes_resamp = dplTimes[::downsamp]

# Make the TFR for transient spectral event analysis 
fmin = 1
fmax = 60
fstep =1
fVec = np.arange(fmin, fmax+1, fstep)
width = 10
Fs = 1/params['dt']*1000
newFs = int(np.round(Fs/downsamp))
TFR, tVec, fVec = tse.spectralevents_ts2tfr(np.expand_dims(dplData_resamp,1), fVec, newFs, width)

# Plot the TFR
if plotOK:
    plt.pcolor(tVec, fVec, np.squeeze(TFR), cmap='jet')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Signal Power [nAm^2 ?]', rotation=270)
    plt.show()

# Find transient spectral events based on TFR
findMethod = 1
thrFOM = 6
classLabels = [1]
neighbourhood_size = (4,160)
threshold = 0.00
spectralEvents = tse.spectralevents_find (findMethod, thrFOM, tVec,
            fVec, TFR, classLabels, neighbourhood_size, threshold, newFs)
df = pd.DataFrame(spectralEvents)

# Get beta range events only
df1 = df[df['Peak Frequency']>=betaMin]
df2 = df1[df1['Peak Frequency']<=betaMax]
betaEvents = df2[df2['Outlier Event']]

# Find time for each event of trough closest to peak time
epochLength = 2000
epochTime = epochLength/newFs
epochTimes = np.arange(epochLength)/newFs-(epochTime/2)

betaTimes = betaEvents['Peak Time'].tolist()
troughTimes = []
for bt in betaTimes:
    peakIndex = int(bt*newFs)
    print(peakIndex)
    firstIndex = int(peakIndex-epochLength/2)
    lastIndex = int(peakIndex+epochLength/2)
    if firstIndex >= 0:
        if lastIndex < dplData_resamp.shape[0]:
            thisData = dplData_resamp[firstIndex:lastIndex]
            localMinIndex = ss.argrelextrema(-thisData, np.greater)[0]-epochLength/2
            localMinClosestToZero = min(localMinIndex, key=lambda x:abs(x))
            troughTimes.append(peakIndex + localMinClosestToZero)

# Extract time course for each event aligned to trough closest to peak time
epochs = []
for tt in troughTimes:
    firstIndex = int(tt-epochLength/2)
    lastIndex = int(tt+epochLength/2)
    if firstIndex >= 0:
        if lastIndex < dplData_resamp.shape[0]:
            thisData = dplData_resamp[firstIndex:lastIndex]
            epochs.append(thisData)
epochs = np.asarray(epochs)

# Burst Triggered Average
betaBurstAvg = np.mean(epochs, 0)
if plotOK:
    plt.plot(epochTimes, betaBurstAvg)
    plt.show()

