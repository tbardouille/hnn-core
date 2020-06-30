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

    return dpls

def get_spectral_events(dplData_resamp, params):
    # Make the TFR for transient spectral event analysis 
    fmin = 1
    fmax = 60
    fstep =1
    fVec = np.arange(fmin, fmax+1, fstep)
    width = 10
    Fs = 1/params['dt']*1000
    newFs = int(np.round(Fs/downsamp))
    TFR, tVec, fVec = tse.spectralevents_ts2tfr(np.expand_dims(dplData_resamp,1), fVec, newFs, width)
    
    # Get the power spectrm from the time-averaged TFR
    TFR_PSD = np.mean(np.squeeze(TFR), axis=1)   

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

    return TFR, TFR_PSD, fVec, betaEvents, newFs

def burst_triggered_average(dplData_resamp, newFs, narrowBandPass, betaEvents):
    # Narrow band-pass filter the time course (2Hz width around the peak in the PSD)
    b,a = ss.butter(4, narrowBandPass, 'bandpass', fs=newFs)
    dplData_BP = ss.filtfilt(b,a,dplData_resamp,padlen=100)

    # Find time for each event of trough closest to peak time using BP data
    epochLength = 2000
    epochTime = epochLength/newFs
    epochTimes = np.arange(epochLength)/newFs-(epochTime/2)

    betaTimes = betaEvents['Peak Time'].tolist()
    waves = []
    troughTimes = []
    # Loop over beta event times
    for bt in betaTimes:
        # Find peak index and epoch edge samples
        peakIndex = int(bt*newFs)
        firstIndex = int(peakIndex-epochLength/2)
        lastIndex = int(peakIndex+epochLength/2)
        # If the epoch fits within the data array
        if firstIndex >= 0:
            if lastIndex < dplData_BP.shape[0]:
                thisData = dplData_BP[firstIndex:lastIndex]
                waves.append(thisData)
                localMinIndex = ss.argrelextrema(-thisData, np.greater)[0]-epochLength/2
                localMinClosestToZero = min(localMinIndex, key=lambda x:abs(x))
                troughTimes.append(peakIndex + localMinClosestToZero)
    waves = np.asarray(waves)

    # Extract unfiltered time course for each event aligned to trough closest to peak time
    epochs = []
    for tt in troughTimes:
        firstIndex = int(tt-epochLength/2)
        lastIndex = int(tt+epochLength/2)
        if firstIndex >= 0:
            if lastIndex < dplData_resamp.shape[0]:
                thisData = dplData_resamp[firstIndex:lastIndex]
                epochs.append(thisData)
    epochs = np.asarray(epochs)

    betaBurstAvg = np.mean(epochs, 0)
    
    return epochTimes, betaBurstAvg, epochs

def get_smoothed_ranges(parameterValues, dpls):

    ranges = []
    for dpl in dpls:
    # Note: skip the 0th sample here due to its offset
        dipAmp_sorted = np.sort(dpl.dpl['agg'][1::])
        ranges.append(np.mean(dipAmp_sorted[-11:-1]) - np.mean(dipAmp_sorted[0:10]))

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

'''
# Make the folder for the dpl files
if not os.path.exists(dplPath):
    os.makedirs(dplPath)
#else:
#	# Remove all the subdirectories first
#    shutil.rmtree(dplPath)           
#    os.makedirs(dplPath)

# Run simulations in parallel
# Set up the parallel task pool for dipole simulations
count = int(np.round(mp.cpu_count()*1/2))
print(count)
pool = mp.Pool(processes=count)
# Run the jobs
pool.map(adjust_param_and_simulate_dipole_mp, parameterValues)
'''

# Read simulated dipole timecourses for all parameter values
dpls = read_dipoles_over_range(parameterValues, plotOK)

# Read smoother dipole amplitude ranges for parameters
ranges = get_smoothed_ranges(parameterValues[0:-1], dpls[0:-1])
if plotOK:
    plt.plot(parameterValues[0:-1], ranges)
    plt.xlabel('Parameter Value')
    plt.ylabel('Dipole Value Range [nAm]')
    plt.show()

# Get and example dipole timecourse and times for further analysis
#   Note dropping first two samples due to large offset
dplData = dpls[3].dpl['agg'][2::]
dplTimes = dpls[3].t[2::]

# Downsample the data for speed (there doesn't seem to be content above 2000 Hz sample rate anyway)
dplData_resamp = dplData[::downsamp]
dplTimes_resamp = dplTimes[::downsamp]

# Run spectral events to get TFR, PSD and all beta bursts
TFR, TFR_PSD, fVec, betaEvents, newFs = get_spectral_events(dplData_resamp, params)
if plotOK:
    plt.plot(fVec, TFR_PSD) 
    plt.show()

# Make burst-triggered average
epochTimes, betaBurstAvg, epochs = burst_triggered_average(dplData_resamp, newFs, (19,21), betaEvents)
if plotOK:
    plt.plot(epochTimes, betaBurstAvg)
    plt.show()

# Beta band-pass filter the epoch time courses 
b,a = ss.butter(4, (betaMin, betaMax), 'bandpass', fs=newFs)
epochs_BP = ss.filtfilt(b,a,epochs,axis=1,padlen=100)

# Take the average of band-pass filtered signals
betaBurstAvg_BP = np.mean(epochs_BP, axis=0)
if plotOK:
    plt.plot(epochTimes, betaBurstAvg)
    plt.plot(epochTimes, betaBurstAvg_BP)
    plt.legend(['Raw', 'Beta Band Filtered'])
    plt.show()


