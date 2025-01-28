## Plots the time evolition of triplet lifetime. PartWise analysis.
## ----------------------------------------- setting-up libraries ----------------------------------------
# from time import perf_counter
# t0 = perf_counter()
import sys
import numpy as np
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import pandas as pd
# import pickle
# from scipy.signal import find_peaks
# from scipy.optimize import curve_fit
from os import path
import os
# import argparse

## ----------------------------------- matplotlib RC etc -----------------------------------------
rc('figure', autolayout=True, figsize=[10, 6], dpi=125, titlesize=20 )
rc('font', family='monospace')
rc('axes', titlesize=20, titleweight='heavy', labelsize=16, labelweight='bold')
rc(('xtick', 'ytick'), labelsize = 18)
rc('legend', fontsize=14)
rc('lines', linewidth=2.5)
rc('mathtext', default = 'regular')
rc('xtick.minor', visible = False, size = 6)
rc('ytick.minor', visible = True, size = 8)
# rc('axes.formatter', limits=[-1, 1])
np.set_printoptions(formatter={'float': lambda x: f"{x:10.4g}"})

## ----------------------------------------- directory -----------------------------------------

output_dir = '/home/sarthak/my_projects/argset/'

## ----------------------------------------- function definitions -----------------------------------------
def save_plot(fig:matplotlib.figure.Figure, file_name: str):
    fig.savefig(path.join(output_dir, f'{file_name}.pdf'))

## ----------------------------------------- data -----------------------------------------
tau3_file = pd.read_csv(path.join(output_dir, 'results','tau3_run00156_trunc.csv'))

## ----------------------------------------- Program -----------------------------------------
fig, ax = plt.subplots()

ax.scatter(tau3_file['index'], tau3_file['ch0']/1000, s=128, alpha=0.5, label='Ch0', color='C0')
ax.scatter(tau3_file['index'], tau3_file['ch1']/1000, s=128, alpha=0.5, label='Ch1', color='C1')
ax.scatter(tau3_file['index'], tau3_file['ch2']/1000, s=128, alpha=0.5, label='Ch2', color='C2')

# ax.errorbar(tau3_file['index'], tau3_file['ch0']/1000, \
#             yerr = tau3_file['ch0_sigma']/1000, fmt='o', elinewidth=1.0, \
#             linestyle='none', ecolor='C0', label='Ch0') #'#ADFF2F'
# ax.errorbar(tau3_file['index'], tau3_file['ch1']/1000, \
#             yerr = tau3_file['ch1_sigma']/1000, fmt='o', elinewidth=1.0, \
#             linestyle='none', ecolor='C1', label='Ch1') #'#ADFF2F'
# ax.errorbar(tau3_file['index'], tau3_file['ch2']/1000, \
#             yerr = tau3_file['ch2_sigma']/1000, fmt='o', elinewidth=1.0, \
#             linestyle='none', ecolor='C2', label='Ch2') #'#ADFF2F'

# plt.locator_params(axis='x', nbins=5)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlabel('Run part')
ax.set_xticklabels(['$1^{st}$','$2^{nd}$','$3^{rd}$','$4^{th}$','$5^{th}$'])
ax.set_ylabel('Triplet lifetime [$\\mu$s]')
ax.set_ylim(2.9, 3.6)
ax.grid()
ax.legend()
fig.savefig(path.join(output_dir, 'plots','tau3_run00156.png'))