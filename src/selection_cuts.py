## ----------------------------------------- computing resources -----------------------------------------
### srun --mem=16G -A bejger-grp -p dgx --pty bash
## ----------------------------------------- setting-up libraries -----------------------------------------
from time import perf_counter
t0 = perf_counter()

import sys
import numpy as np
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import pandas as pd
import pickle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from os import path
import os

rc('figure', autolayout=True, figsize=[16, 9], dpi=125, titlesize=20 )
rc('font', family='monospace')
rc('axes', titlesize=20, titleweight='heavy', labelsize=16, labelweight='bold')
rc(('xtick', 'ytick'), labelsize = 18)
rc('legend', fontsize=14)

## ----------------------------------------- loading data -----------------------------------------
output_dir = '/home/sarthak/my_projects/argset/output'
data_dir = '/work/chuck/sarthak/argset/event_catalogues'

# filename = 'event_catalogue_run00126.pkl'
filename = 'event_catalogue_run00126_part.pkl'
file_basename = filename.split(sep='_run')[-1].split(sep='.')[0]
output_subdir = path.join(output_dir, f'{file_basename}_output')
if not path.isdir(output_subdir):
        os.mkdir(output_subdir)
event_catalogue_file = path.join(data_dir, filename)
event_catalogue = pd.read_pickle(event_catalogue_file)
wfs = event_catalogue['wf']
del event_catalogue

## ----------------------------------------- ARMA -----------------------------------------
from pyreco.manager.manager import Manager
filename = '/work/sarthak/argset/data/run00126.mid.lz4'
outfile = path.join(output_subdir, 'tempJupyR00126_from_script')
confile = 'argset.ini'
tmin,tmax = 0, 4000
cmdline_args = f'--config {confile} -o {outfile} -i {filename}'
m = Manager( midas=True, cmdline_args=cmdline_args)
from pyreco.reco.filtering import WFFilter
mfilter = WFFilter(m.config)

## ----------------------------------------- function definitions -----------------------------------------
def save_plot(fig:matplotlib.figure.Figure, file_name: str):
    fig.savefig(path.join(output_subdir, f'{file_name}.pdf'))
    pickle.dump(fig, open(path.join(output_subdir, f'{file_name}.pkl'),  'wb') )

def pickle_dict(dict: dict, filename:str):
    with open(path.join(output_subdir, f'{filename}.pkl'), 'wb') as file_handle:
        pickle.dump(dict, file_handle, pickle.HIGHEST_PROTOCOL)

def create_flt_wfs(wfs):
    flt_dict = {0: [], 
                1: [],
                2: []}
    for og_wf in wfs:
        flt_wf = perform_arma(og_wf)
        for ch_id in range(3):
            flt_dict[ch_id].append(flt_wf[ch_id])
    return flt_dict

def pulse_difference(event_x, use_flt_wf:bool):
    if not use_flt_wf:
        # window_range = np.arange(350, 500)
        window_range = np.arange(350, 4000)
        peaks0 =find_peaks(wfs[event_x][0][window_range])
        peaks1 =find_peaks(wfs[event_x][1][window_range])
        peaks2 =find_peaks(wfs[event_x][2][window_range])
        mp0 = np.argmax(wfs[event_x][0][window_range][peaks0[0]])
        mp1 = np.argmax(wfs[event_x][1][window_range][peaks1[0]])
        mp2 = np.argmax(wfs[event_x][2][window_range][peaks2[0]])
        sample_mp0 = wfs[event_x][0][window_range][peaks0[0]][mp0]
        sample_mp1 = window_range[peaks1[0]][mp1]
        sample_mp2 = window_range[peaks2[0]][mp2]

    if use_flt_wf:
        # window_range = np.arange(350, 500)
        window_range = np.arange(350, 4000)
        peaks0 =find_peaks(flt_dict[0][event_x][window_range]) # TODO: better code
        peaks1 =find_peaks(flt_dict[1][event_x][window_range]) # same
        peaks2 =find_peaks(flt_dict[2][event_x][window_range]) # same
        mp0 = np.argmax(flt_dict[0][event_x][window_range][peaks0[0]])
        mp1 = np.argmax(flt_dict[1][event_x][window_range][peaks1[0]])
        mp2 = np.argmax(flt_dict[2][event_x][window_range][peaks2[0]])
        sample_mp0 = flt_dict[0][event_x][window_range][peaks0[0]][mp0]
        sample_mp1 = window_range[peaks1[0]][mp1]
        sample_mp2 = window_range[peaks2[0]][mp2]

    return abs(sample_mp1 - sample_mp2)

def calculate_com(event_x):
    event_com = np.divide(np.sum(np.multiply(wfs[event_x], np.arange(wfs[event_x].shape[1])), axis=1), 
                      np.sum(wfs[event_x], axis=1)
                )
    return event_com

def red_chisq(f_obs, f_exp, fittedparameters):
    chisqr = np.sum((f_obs - f_exp)**2 / f_exp)
    ndf = f_obs.shape[0]
    return chisqr/(ndf -fittedparameters.shape[0])

def fit_com_peak(ch_x: int, ax_1: matplotlib.axes.Axes, hist_features: dict):
    
    def f_gauss(x, f_mean, f_sigma, f_k):
        return f_k*(1/(f_sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-f_mean)/f_sigma)**2)
    
    hist_content, hist_edges, histObjects = hist_features[ch_x]
    x_range = np.arange(239, 268) #np.arange(235, 270)
    # x_range = np.arange(220, 281)

    p0_input = [1270, 208, 1e6]

    fitted_parameters, pcov = curve_fit(f_gauss, 
                                hist_edges[x_range], hist_content[x_range], \
                                p0 = p0_input,

                                )

    red_chisqr_value = red_chisq(hist_content[x_range], \
        f_gauss(hist_edges[x_range], *fitted_parameters), fitted_parameters
        )

    ax_1[ch_x].plot(hist_edges[x_range], f_gauss(hist_edges[x_range], *fitted_parameters), label='fit', color='red')
    text_in_box = AnchoredText(f"statistics = {np.sum(hist_content[x_range])} \nreduced chisqr = {red_chisqr_value:.2f}", \
                                            loc='upper left')
    ax_1[ch_x].add_artist(text_in_box)
    ax_1[ch_x].legend()


    print(f'red_chisqr_value: {red_chisqr_value}')
    print(f'fitted_parameters for {ch_x}: {fitted_parameters}')
    print('\n')
    return fitted_parameters[0], fitted_parameters[1]

def histogram_wf_sum_ch(ch_id):
    hist_plot_range = (-2.5e6, 0.5e7)
    fig_2, ax_2 = plt.subplots( 1, 1, figsize=(10, 8), sharex=True, sharey = False)
    bin_content_0, bin_edges, _PlotsObjects = ax_2.hist(wf_sum_dict[ch_id], bins=10000, range=hist_plot_range, label = f'{ch_id}: full wf sum')
    # np.save(path.join(output_subdir, 'wf_sum_0_content.npy'), bin_content_0)
    # np.save(path.join(output_subdir, 'wf_sum_0_edges.npy'), bin_edges)
    save_plot(fig_2, f'hist_full_wf_sum_{ch_id}')
    plt.close(fig_2)

def apply_cuts(wfs, pretrigger_sum_UpperThreshold=4000, sigma_multiplier=2.0, 
               pulse_difference_threshold=40):
    # com_threshold = 300 # not in use
    ch_id = 1
    com_below_xsigma = com_mean_arr - sigma_multiplier*com_std_arr # was 2 # explore and investigate thresholds
    com_above_xsigma = com_mean_arr + sigma_multiplier*com_std_arr
    for event_x in range(wfs.shape[0]):
        if pretrigger_sum[0][event_x] <= pretrigger_sum_UpperThreshold and pretrigger_sum[0][event_x] >= -6000:
            wf_sum_post_cut_dict[1].append(wf_sum_dict[0][event_x]) # sum is always taken from channel 0; should we change it?
            # if (np.abs(com_dict[0][event_x] - com_dict[1][event_x]) <= com_threshold) and (np.abs(com_dict[2][event_x] - com_dict[1][event_x]) <= com_threshold): # 3rd cut: concurrence of COM
            if (com_dict[ch_id][event_x] <= com_above_xsigma)[ch_id] and (com_dict[ch_id][event_x] >= com_below_xsigma[ch_id]): # 3rd cut: distance from mean COM
            # if True:
                wf_sum_post_cut_dict[2].append(wf_sum_dict[0][event_x])
                if pulse_difference(event_x, use_flt_wf=True) <= pulse_difference_threshold: # 2nd cut: simultaneity of pulses
                    wf_sum_post_cut_dict[3].append(wf_sum_dict[0][event_x])
                    event_PassList.append(event_x) # these events should be pickled or passed for further processing
                    # wf_sum_post_cut_ls.append(np.sum(wfs[event_x][ch_id]))
                    com_post_cut_dict[0].append(com_dict[0][event_x])
                    com_post_cut_dict[1].append(com_dict[1][event_x])
                    com_post_cut_dict[2].append(com_dict[2][event_x])
                else:
                    event_FailList_3rdCut.append(event_x)
    np.save(path.join(output_subdir, 'event_PassList.npy'), np.array(event_PassList))
    np.save(path.join(output_subdir, 'event_FailList_3rdCut.npy'), np.array(event_FailList_3rdCut))
    # return com_post_cut_dict

def hist_pulse_difference():
    pulse_difference_ls = []
    for event_id in range(wfs.shape[0]):
        pulse_difference_ls.append( pulse_difference(event_id, use_flt_wf=True) )
    fig, ax = plt.subplots(1, figsize = (10, 8))
    fig.suptitle('Histogram of Pulse maxima difference in channel 1 and 2')
    ax.hist(pulse_difference_ls, bins=np.arange(0, 2500, 5),
                color=f'C0', label='channels 1 & 2');
    ax.axvline(x = 40, linestyle='--', color='red')
    ax.set_yscale('log')
    ax.set_xlabel('pulse maxima difference in bin units')
    save_plot(fig, 'hist_pulse_difference')

def perform_arma(og_wf):
    flt = np.reshape(mfilter.numba_fast_filter(og_wf), newshape=og_wf.shape)
    # mas = m.algos.running_mean(flt, gate=60)
    # return flt - mas
    return flt

def histogram_wf_sum():

    flt_wf_sum_dict = {
        0: [],
        1: [],
        2: []
    }
    for event_id in range(wfs.shape[0]):
        flt_wf_sum = np.sum(perform_arma(wfs[event_id]), axis=1)
        flt_wf_sum_dict[1].append( flt_wf_sum[1] )
        flt_wf_sum_dict[2].append( flt_wf_sum[2] )
        flt_wf_sum_dict[0].append( flt_wf_sum[0] )

    sum_hist_plot_range = (-2.5e6, 0.5e7)
    # flt_hist_plot_range = (-100, 275)
    fig_4, ax_4 = plt.subplots( 3, 2, figsize=(18, 16), sharex=False, sharey = False)
    for ch_id in range(3):
        ax_4[ch_id][1].hist(flt_wf_sum_dict[ch_id], bins=10000, range=sum_hist_plot_range, color=f'C{ch_id}', label = f'filtered wf sum {ch_id}')
        ax_4[ch_id][1].legend()
        ax_4[ch_id][1].grid()
        ax_4[ch_id][0].hist(wf_sum_dict[ch_id], bins=10000, range=sum_hist_plot_range, color=f'C{ch_id}', label = f'full wf sum {ch_id}')
        ax_4[ch_id][0].legend()
        ax_4[ch_id][0].grid()
    save_plot(fig_4, 'hist_flt_wf')
    plt.close(fig_4)

def stack_flt_wf(flt_dict:dict):
    stacked_flt_wf_dict= {
    0:np.zeros_like(wfs[0][0]),
    1:np.zeros_like(wfs[0][0]),
    2:np.zeros_like(wfs[0][0]),
    }
    # stacked_flt_wf_dict[ch_id] = np.sum(np.array(flt_dict[ch_id]), axis=0)
    for ch_id in range(3):
        for event_id in event_PassList: #TODO: loop can be skipped using pandas.Series
            stacked_flt_wf_dict[ch_id] += flt_dict[ch_id][event_id]
    pickle_dict(stacked_flt_wf_dict, 'stacked_flt_wf_dict')

    return stacked_flt_wf_dict

def stack_wf(wfs:pd.core.series.Series):
    stacked_wf_dict = {
    0:np.zeros_like(wfs[0][0]),
    1:np.zeros_like(wfs[0][0]),
    2:np.zeros_like(wfs[0][0])
    }
    for ch_id in range(3):
        for event_id in range(wfs.shape[0]):
            stacked_wf_dict[ch_id] += wfs[event_id][ch_id]
    pickle_dict(stacked_wf_dict, 'stacked_wf_dict')
# def fit_stackedwf():

## ----------------------------------------- program -----------------------------------------
ch_id = 0

pretrigger_sum = {
    0: [],
    1: [],
    2: []
}
wf_sum_dict = {0: [],
            1: [],
            2: []
            } 
com_dict = {0: [],
            1: [],
            2: []
            }

for event_x in range(wfs.shape[0]):
    pretrigger_sum[1].append( np.sum(wfs[event_x][1][:350]) )
    pretrigger_sum[2].append( np.sum(wfs[event_x][2][:350]) )
    pretrigger_sum[0].append( np.sum(wfs[event_x][0][:350]) )
    wf_sum = np.sum(wfs[event_x], axis=1)
    for ch_x in range(3):
        wf_sum_dict[ch_x].append( wf_sum[ch_x] )
        
    com_arr = calculate_com(event_x)
    com_dict[0].append(com_arr[0])
    com_dict[1].append(com_arr[1])
    com_dict[2].append(com_arr[2])
del com_arr

# histogram_wf_sum() # not in use

flt_dict = create_flt_wfs(wfs) # pass it to pulse_difference

# hist_pulse_difference()

## ----------------------------------------- Histograms -----------------------------------------

fig_0, ax_0 = plt.subplots( 3, 1, figsize=(10, 8), sharex=True, sharey = False)
for ch_x in range(3):
    ax_0[ch_x].hist(pretrigger_sum[ch_x], bins = np.arange(-6000, 200_000, 1000), 
                            color=f'C{ch_x}', label = f'pretrigger sum in channel {ch_x}')
    ax_0[ch_x].set_yscale('log')
    ax_0[ch_x].legend()
    ax_0[ch_x].grid()
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    fig_0.suptitle('hist of pretrigger sum')
save_plot(fig_0, 'hist_pretrigger_sum')
plt.close(fig_0)

hist_features = {
    0: [],
    1: [],
    2: [],
}
com_mean_arr = np.zeros([3,])
com_std_arr = np.zeros([3,])

fig_1, ax_1 = plt.subplots( 3, 1, figsize=(10, 8), sharex=True, sharey = True)
for ch_x in range(3):
    hist_features[ch_x] = ( ax_1[ch_x].hist(com_dict[ch_x], bins=np.arange(-5_000, 5_000, 25), 
                            color=f'C{ch_x}', label = f'{ch_x}')
                            )
    ax_1[ch_x].legend()
    ax_1[ch_x].grid()
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    fig_1.suptitle('hist of Center Of Mass')
    com_mean_arr[ch_x], com_std_arr[ch_x] = fit_com_peak(ch_x, ax_1, hist_features)
save_plot(fig_1, 'hist_COM')
plt.close(fig_1)
del hist_features

## ----------------------------------------- cuts -----------------------------------------
# com_threshold = 300 # not in use
ch_id = 1
com_below_xsigma = com_mean_arr - 1.75*com_std_arr # was 2 # explore and investigate thresholds
com_above_xsigma = com_mean_arr + 1.75*com_std_arr

wf_sum_post_cut_dict = {
    1: [],
    2: [],
    3: []
}
com_post_cut_dict = {0: [],
            1: [],
            2: []
            }
event_PassList= []
event_FailList_3rdCut= []
# wf_sum_post_cut_ls = []

apply_cuts(wfs, pulse_difference_threshold=40)

hist_plot_range = (0e6, 5e6)
fig_2, ax_2 = plt.subplots( 5, 1, figsize=(19, 15), sharex=True, sharey = False)
bin_content_0, bin_edges, _PlotsObjects = ax_2[0].hist(wf_sum_dict[0], bins=500, range = hist_plot_range, label = 'No cut [Channel 0]')
bin_content_1, bin_edges, _PlotsObjects = ax_2[1].hist(wf_sum_post_cut_dict[1], bins=bin_edges, range = hist_plot_range, label = '1st cut [Channel 0]')
bin_content_2, bin_edges, _PlotsObjects  = ax_2[2].hist(wf_sum_post_cut_dict[2], bins=bin_edges, label = '2nd cut [Channel 0]')
bin_content_3, bin_edges, _PlotsObjects  = ax_2[3].hist(wf_sum_post_cut_dict[3], bins=bin_edges, label = '3rd cut [Channel 0]')
plt.subplots_adjust(wspace=0.025, hspace=0.025)
fig_2.suptitle('successive cuts')
ratio_1_0 = np.divide(bin_content_1, bin_content_0, out=np.zeros_like(bin_content_1), where=bin_content_0!=0)
ax_2[4].plot(bin_edges[:-1], 1.-ratio_1_0, alpha=0.5, color='#f86f6c', label = '1st cut')
ratio_2_1 = np.divide(bin_content_2, bin_content_1, out=np.zeros_like(bin_content_2), where=bin_content_1!=0)
ax_2[4].plot(bin_edges[:-1], 1.-ratio_2_1, alpha=0.5, color='#447acd', label = '2nd cut')
ratio_3_2 = np.divide(bin_content_3, bin_content_2, out=np.zeros_like(bin_content_3), where=bin_content_2!=0)
ax_2[4].plot(bin_edges[:-1], 1.-ratio_3_2, alpha=0.5, color='#69ebd2', label = '3rd cut')
ratio_3_0 = np.divide(bin_content_3, bin_content_0, out=np.zeros_like(bin_content_3), where=bin_content_0!=0)
ax_2[4].plot(bin_edges[:-1], 1.-ratio_3_0, alpha=0.5, color='#e59acf', label = 'overall')
ax_2[4].axhline(y=0.2, linestyle='--')
for subplot_x in range(4):
    ax_2[subplot_x].legend()
    ax_2[subplot_x].grid()
    ax_2[subplot_x].set_ylabel('counts')
ax_2[4].legend()
ax_2[4].set_ylabel('cut efficiency')
ax_2[4].set_xlabel('Full WF sum')
save_plot(fig_2, 'successive_cuts')
plt.close(fig_2)
del bin_content_1, bin_content_2, bin_content_3, bin_edges, _PlotsObjects

fig_3, ax_3 = plt.subplots( 3, 1, figsize=(10, 8), sharex=True, sharey = True)
for ch_id in range(3):
    ax_3[ch_id].hist(com_post_cut_dict[ch_id], bins=np.arange(-5_000, 5_000, 25), 
                        color=f'C{ch_id}', label = f'{ch_id}')    
    ax_3[ch_id].legend()
    ax_3[ch_id].grid()
plt.subplots_adjust(wspace=0.025, hspace=0.025)
fig_3.suptitle('hist of Center Of Mass post cuts')
save_plot(fig_3, 'hist_COM_post_cut')
plt.close(fig_3)
## ----------------------------------------- Time Constant -----------------------------------------
# stack_flt_wf(flt_dict)
# stack_wf(wfs)
## ------------------------------ Fit Gauss to integrated Charge Distribution ----------------------

print(f'Execution time: {perf_counter() - t0}')