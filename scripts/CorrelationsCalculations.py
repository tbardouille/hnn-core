import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

#Read in the initial/ default file that you will be comparing other data to
file_i = '/Users/lindseypower/Desktop/dpl1.txt'
data_i = pd.read_csv(file_i, sep="\t", header=None)
init_vals = np.asarray(data_i[1].to_list())

#Create a list of the values that you are manipulating - for example if you are changing the delay and you have simulations with delays of 0, 5, 10 and 15 ms make an array like [0,5,10,15]
input_vals = [50,100,150] 

#Initialize a list to store all of the correlations and shifts of different simulations the default
correlations = []
temporal_shifts = [] 
num_maximas = []

#Here you can put the filenames for the dipole files of all the simulations you want to compare
filenames = ['dpl2', 'dpl3','dpl4']

for name in filenames:
    #Read in new file that is going to be compared to the default
    file_n = '/Users/lindseypower/Desktop/' + name + '.txt'
    data_n = pd.read_csv(file_n, sep="\t", header=None)

    #Pull out the columns of interest from dataframe
    new_vals = np.asarray(data_n[1].to_list())
    
    #Start by finding the number of local maxima in the signal 
    num_maxima = len(ss.find_peaks(new_vals)[0])
    
    #Now if I have to systematically shift the new data I can do this:
    new_vals_cp = new_vals
    sub_corrs = []

    for x in range(0,710):
        #This takes the 40th sample until the end of the array - 40 is chosen becuase there are 40 samples of data/ ms and I want to shift it 1 ms at a time
        new_vals_cp = new_vals_cp[40:]
        #When you do this you have to zero pad the end of the array with 40 zeros so it is the same length as the one you are comparing it to   
        new_vals_cp = np.pad(new_vals_cp, (0,40), 'constant')
        
        #Calculate the correlation coefficient matrix between values 
        corr_coef = np.corrcoef(init_vals, new_vals_cp)
        
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

print(correlations)
print(temporal_shifts)
print(num_maximas)

#Plot the correlations versus the input vals (delays in start times)
plt.plot(input_vals, correlations)
plt.xlabel('Start times (ms)')
plt.ylabel('Max Correlation Coefficient')

#Plot the temporal shift versus the input vals (delays in start times)
plt.plot(input_vals, temporal_shifts)
plt.xlabel('Delay (ms)')
plt.ylabel('Temporal Shift (ms)')

#Plot the number of local maxima versus the input vals (delays in start times)
plt.plot(input_vals, num_maximas)
plt.xlabel('Delay (ms)')
plt.ylabel('Number of Local Maxima')





