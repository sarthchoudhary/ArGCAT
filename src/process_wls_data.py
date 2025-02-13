## ----------------------------------------- computing resources -----------------------------------------
### srun --mem=16G -A bejger-grp -p dgx --pty bash
## -------------------------------------------------------------------------------------------------------
## This code is not yet complete nonetheless the code works as expected. Some refactoring is needed.
## -------------------------------------------------------------------------------------------------------
## ----------------------------------------- setting-up libraries ----------------------------------------
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
import argparse
from pyreco.manager.manager import Manager

## ----------------------------------- matplotlib RC etc -----------------------------------------
rc('figure', autolayout=True, figsize=[16, 9], dpi=125, titlesize=20 )
rc('font', family='monospace')
rc('axes', titlesize=20, titleweight='heavy', labelsize=16, labelweight='bold')
rc(('xtick', 'ytick'), labelsize = 18)
rc('legend', fontsize=14)
rc('lines', linewidth=2.5)
rc('mathtext', default = 'regular')
rc('xtick.minor', visible = True, size = 6)
rc('ytick.minor', visible = True, size = 8)
# rc('axes.formatter', limits=[-1, 1])
np.set_printoptions(formatter={'float': lambda x: f"{x:10.4g}"})

## ----------------------------------------- directories -----------------------------------------
# output_dir = '/home/sarthak/my_projects/argset/output'
output_dir = '/work/chuck/sarthak/argset/output_folder/analysis/' #TODO: move to YAML file.
data_dir = '/work/chuck/sarthak/argset/event_catalogues'

## ----------------------------------------- Arguments -----------------------------------------
def argument_collector() ->argparse.Namespace:
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--filename", help='name of the event catalogue file')
    argParser.add_argument("-st", "--sample_type", 
                           choices = ['PEN', 'TPB'],
                           help= "whether the Sample is TPB or PEN")
    args = argParser.parse_args()
    return args

args = argument_collector()

filename = args.filename
file_basename = filename.split(sep='_run')[-1].split(sep='.')[0]
output_subdir = path.join(output_dir, f'{file_basename}_output')
if not path.isdir(output_subdir):
        os.mkdir(output_subdir)
event_catalogue_path = path.join(data_dir, filename)
event_catalogue = pd.read_pickle(event_catalogue_path)
wfs = event_catalogue['wf']
del event_catalogue
# ## remove first 5 hours of data for runs which started taking data while still warm
# # truncate_till = 13000
# # wfs= wfs.truncate(before=truncate_till) #TODO: comment out if not needed
# # print(f'Caution: discarding initial 13000 events!')

## ----------------------------------------- ARMA -----------------------------------------
from pyreco.manager.manager import Manager
raw_filename = '/work/sarthak/argset/data/run00126.mid.lz4'
outfile = path.join(output_subdir, 'tempJupyR00126_from_script')
confile = 'argset.ini'
tmin,tmax = 0, 4000
cmdline_args = f'--config {confile} -o {outfile} -i {raw_filename}'
m = Manager( midas=True, cmdline_args=cmdline_args)
from pyreco.reco.filtering import WFFilter
mfilter = WFFilter(m.config)

## ----------------------------------------- function definitions -----------------------------------------
def f_gauss(x, f_mean, f_sigma, f_k):
    return f_k*(1/(f_sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-f_mean)/f_sigma)**2)

def save_plot(fig:matplotlib.figure.Figure, file_name: str):
    # fig.savefig(path.join(output_subdir, f'{file_name}.pdf'))
    fig.savefig(path.join(output_subdir, f'{file_name}.png'))
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
        peaks0 =find_peaks(wfs.iloc[event_x][0][window_range])
        peaks1 =find_peaks(wfs.iloc[event_x][1][window_range])
        peaks2 =find_peaks(wfs.iloc[event_x][2][window_range])
        mp0 = np.argmax(wfs.iloc[event_x][0][window_range][peaks0[0]])
        mp1 = np.argmax(wfs.iloc[event_x][1][window_range][peaks1[0]])
        mp2 = np.argmax(wfs.iloc[event_x][2][window_range][peaks2[0]])
        sample_mp0 = wfs.iloc[event_x][0][window_range][peaks0[0]][mp0]
        sample_mp1 = window_range[peaks1[0]][mp1]
        sample_mp2 = window_range[peaks2[0]][mp2]

    if use_flt_wf:
        # window_range = np.arange(350, 500)
        window_range = np.arange(350, 4000)
        peaks0 =find_peaks(flt_dict[0][event_x][window_range]) # TODO: find_peaks is probably unnecessary if only the maxima is needed
        peaks1 =find_peaks(flt_dict[1][event_x][window_range]) # same
        peaks2 =find_peaks(flt_dict[2][event_x][window_range]) # same
        mp0 = np.argmax(flt_dict[0][event_x][window_range][peaks0[0]])
        mp1 = np.argmax(flt_dict[1][event_x][window_range][peaks1[0]])
        mp2 = np.argmax(flt_dict[2][event_x][window_range][peaks2[0]])
        sample_mp0 = flt_dict[0][event_x][window_range][peaks0[0]][mp0]
        sample_mp1 = window_range[peaks1[0]][mp1]
        sample_mp2 = window_range[peaks2[0]][mp2]

    return abs(sample_mp1 - sample_mp2)
    # return sample_mp1 - sample_mp2

def calculate_com(wfs, event_x):
    event_com = np.divide(np.sum(np.multiply(wfs.iloc[event_x], np.arange(wfs.iloc[event_x].shape[1])), axis=1), 
                      np.sum(wfs.iloc[event_x], axis=1)
                )
    return event_com

def red_chisq(f_obs, f_exp, fittedparameters):
    chisqr = np.sum((f_obs - f_exp)**2 / f_exp)
    ndf = f_obs.shape[0]
    return chisqr/(ndf -fittedparameters.shape[0])

def fit_com_peak(ch_x: int, ax_1: matplotlib.axes.Axes, hist_features: dict):
    
    def f_gauss(x, f_mean, f_sigma, f_k):
        return f_k*(1/(f_sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-f_mean)/f_sigma)**2)
    
    hist_content, hist_edges, _histObjects = hist_features[ch_x]
    com_peak = np.argmax(hist_content)
    # x_range = np.arange(com_peak - 40, com_peak + 40) # combined run00156 -- not good ##TODO: this should be provided from a config file
    # x_range = [np.arange(com_peak - 30, com_peak + 30), np.arange(com_peak - 40, com_peak + 40), np.arange(com_peak - 40, com_peak + 40)][ch_x] # variable width of fit ranges for combined run00156
    x_range = [np.arange(com_peak - 7, com_peak + 16), np.arange(com_peak - 21, com_peak + 21), np.arange(com_peak - 20, com_peak + 15)][ch_x] # variable width of fit ranges for combined run00156. bins = np.arange(-500,2500,20)
    # x_range = np.arange(com_peak - 25, com_peak + 30) # 00159_truncated
    # x_range = np.arange(com_peak - 15, com_peak + 17) # 00126_truncated
    # x_range = np.arange(com_peak -7, com_peak + 9) # 00126_truncated didn't work
    # x_range = np.arange(com_peak -16, com_peak + 20) # 00156_truncated
    # x_range = np.arange(com_peak -18, com_peak +20) # dynamic works for 152
    # x_range = np.arange(239, 268) #run00126_part
    # x_range = np.arange(235, 270)
    # x_range = np.arange(220, 281)

    ## p0_input = [1270, 208, 1e6]
    p0_input = {0:[1000, 212, 1e6], 1:[1270, 208, 1e6], 2:[1270, 208, 1e6]}

    fitted_parameters, _pcov = curve_fit(f_gauss, 
                                hist_edges[x_range], hist_content[x_range], \
                                p0 = p0_input[ch_x],
                                )

    red_chisqr_value = red_chisq(hist_content[x_range], \
        f_gauss(hist_edges[x_range], *fitted_parameters), fitted_parameters
        )

    ax_1[ch_x].plot(hist_edges[x_range], f_gauss(hist_edges[x_range], *fitted_parameters), label='fit', color=f'C{ch_x+3}')
    text_in_box = AnchoredText(f"statistics = {int(np.sum(hist_content[x_range]))}" '\n$\\chi^{2}/\\nu$' f"= {red_chisqr_value:.2f}", \
                                            prop=dict(size=14), loc='upper left')
    ax_1[ch_x].add_artist(text_in_box)
    ax_1[ch_x].legend()
    ax_1[ch_x].set_xlim(200, 2100)

    print(f'red_chisqr_value: {red_chisqr_value}')
    print(f'fitted_parameters for {ch_x}: {fitted_parameters}')
    print('\n')
    return fitted_parameters[0], fitted_parameters[1]

def histogram_wf_sum_ch(wf_sum_dict, ch_id):
    hist_plot_range = (-2.5e6, 0.5e7)
    fig_2, ax_2 = plt.subplots( 1, 1, figsize=(10, 8), sharex=True, sharey = False)
    bin_content, bin_edges, _PlotsObjects = ax_2.hist(wf_sum_dict[ch_id], bins=10000, range=hist_plot_range, label = f'{ch_id}: full wf sum')
    # np.save(path.join(output_subdir, 'wf_sum_0_content.npy'), bin_content_0)
    # np.save(path.join(output_subdir, 'wf_sum_0_edges.npy'), bin_edges)
    # save_plot(fig_2, f'hist_full_wf_sum_{ch_id}')
    plt.close(fig_2)
    return bin_content, bin_edges

def apply_cuts(wfs, pretrigger_sum_UpperThreshold=4000, sigma_multiplier=2.0, 
               pulse_difference_threshold=40):
            # pulse_difference_threshold=20):
    print('Applying cuts...')
    # com_threshold = 300 # not in use
    ch_id = 1
    com_below_xsigma = com_mean_arr - sigma_multiplier*com_std_arr # sigma_multiplier was 2 # explore and investigate thresholds
    com_above_xsigma = com_mean_arr + sigma_multiplier*com_std_arr
    for event_x in range(wfs.shape[0]): #event_x is NOT pandas index.
        if pretrigger_sum[0][event_x] <= pretrigger_sum_UpperThreshold and pretrigger_sum[0][event_x] >= -6000:
            wf_sum_post_cut_dict[1].append(wf_sum_dict[0][event_x]) # sum is always taken from channel 0; should we change it?
            # if (np.abs(com_dict[0][event_x] - com_dict[1][event_x]) <= com_threshold) and (np.abs(com_dict[2][event_x] - com_dict[1][event_x]) <= com_threshold): # 3rd cut: concurrence of COM
            if (com_dict[ch_id][event_x] <= com_above_xsigma)[ch_id] and (com_dict[ch_id][event_x] >= com_below_xsigma[ch_id]): # 3rd cut: distance from mean COM
            # if True:
                wf_sum_post_cut_dict[2].append(wf_sum_dict[0][event_x])
                pulse_difference_ls.append( pulse_difference(event_x, use_flt_wf=True) )
                if pulse_difference(event_x, use_flt_wf=True) <= pulse_difference_threshold: # 2nd cut: simultaneity of pulses
                    wf_sum_post_cut_dict[3].append(wf_sum_dict[0][event_x])
                    event_PassList.append(event_x) # these events should be pickled or passed for further processing
                    # wf_sum_post_cut_ls.append(np.sum(wfs.iloc[event_x][ch_id]))
                    com_post_cut_dict[0].append(com_dict[0][event_x])
                    com_post_cut_dict[1].append(com_dict[1][event_x])
                    com_post_cut_dict[2].append(com_dict[2][event_x])
                else:
                    event_FailList_3rdCut.append(event_x)
    print(f'Number of events passing all cuts: \n {len(event_PassList)}')
    np.save(path.join(output_subdir, 'event_PassList.npy'), np.array(event_PassList))
    np.save(path.join(output_subdir, 'event_FailList_3rdCut.npy'), np.array(event_FailList_3rdCut))
    # return com_post_cut_dict

def hist_pulse_difference():
    print('Pulse Difference Histogram')
    # pulse_difference_ls = []
    # for event_id in range(wfs.shape[0]):
    #     # pulse_difference_ls.append( pulse_difference(event_id, use_flt_wf=True) )
    #     pulse_difference_ls.append( pulse_difference(event_id, use_flt_wf=False) )
    fig, ax = plt.subplots(1, figsize = (10, 8))
    # fig.suptitle('Histogram of Pulse maxima difference in channel 1 and 2')
    ax.hist(pulse_difference_ls, 
                bins=np.arange(0, 2500, 5),
                # bins=np.arange(-500, 500, 2),
                # bins=500,
                color=f'cornflowerblue', label='Ch 1 & 2 pulse \ntime difference');
    # ax.axvline(x = 40, linestyle=':', linewidth=1, color='red')
    ax.axvline(x = 40, linestyle='--', linewidth=2.0, color='orangered', label='Threshold')
    # ax.axvline(x = -20, linestyle='--', linewidth=1, color='indianred')
    ax.set_yscale('log')
    # ax.set_xlabel('pulse maxima difference in bin units')
    ax.set_ylabel('counts', fontsize=22)
    # ax.set_xlim(-50, 50)
    ax.set_xlabel('Pulse time difference [1 bin = 4ns]', fontsize=22)
    ax.legend()
    # save_plot(fig, 'hist_pulse_difference')
    save_plot(fig, 'hist_pulse_difference_PEN_updated')

def perform_arma(og_wf):
    flt = np.reshape(mfilter.numba_fast_filter(og_wf), newshape=og_wf.shape)
    # mas = m.algos.running_mean(flt, gate=60) # Discussed with Marcin we AR filtered not ARMA filtered
    # return flt - mas
    return flt

def histogram_wf_sum():
    '''comparison of wf sum histogram for filtered and unfiltered waveforms.'''
    flt_wf_sum_dict = {
        0: [],
        1: [],
        2: []
    }
    for event_id in range(wfs.shape[0]):
        flt_wf_sum = np.sum(perform_arma(output_subdir, wfs.iloc[event_id]), axis=1)
        flt_wf_sum_dict[1].append( flt_wf_sum[1] )
        flt_wf_sum_dict[2].append( flt_wf_sum[2] )
        flt_wf_sum_dict[0].append( flt_wf_sum[0] )

    hist_wf_sum_range = {0:(-2.5e6, 0.5e7), 1: (-1e6, 1e6), 2:(-1e6, 1e6)}
    # flt_hist_plot_range = (-100, 275)
    fig_4, ax_4 = plt.subplots( 3, 2, figsize=(18, 16), sharex=False, sharey = False)
    for ch_id in range(3):
        ax_4[ch_id][1].hist(flt_wf_sum_dict[ch_id], bins=10000, range=hist_wf_sum_range[ch_id], color=f'C{ch_id}', label = f'filtered wf sum {ch_id}')
        ax_4[ch_id][1].legend()
        ax_4[ch_id][1].grid()
        ax_4[ch_id][0].hist(wf_sum_dict[ch_id], bins=10000, range=hist_wf_sum_range[ch_id], color=f'C{ch_id}', label = f'full wf sum {ch_id}')
        ax_4[ch_id][0].legend()
        ax_4[ch_id][0].grid()
    save_plot(fig_4, 'hist_flt_wf')
    plt.close(fig_4)

def histogram_wf_sum_before_and_after_cuts(part_event_PassList: list, part_id:str='') -> dict:
    '''The part_id argument is optional. Only useful when this function is called inside partwise analysis loop.'''
    hist_wf_sum = {0:0, 1:0, 2: 0}
    hist_wf_sum_postcut = {0:0, 1:0, 2: 0}
    hist_wf_sum_range = {0:(-0.1E6, 1.0E6), 1: (-2.5E4, 1.0E5), 2:(-2.5E4, 1.30E5)} #combined run00156
    # hist_wf_sum_range = {0:(-2.5E6, 0.5E7), 1: (-1E6, 1E6), 2:(-1E6, 1E6)} #run00126
    # hist_wf_sum_range = {0:(-0.25E6, 2.75E6), 1: (-0.25E5, 0.25E6), 2:(-0.25E5, 0.5E6)} #run00152 #TODO: dynamic or from config
    # hist_wf_sum_range = {0:(-6E3, 1.25E6), 1: (-6E3, 1E5), 2:(-6E3, 1.5E5)} #run00159_truncated
    fig_5, ax_5 = plt.subplots( 3, 2, 
                                # figsize=(18, 16), dpi=150,
                                figsize=(18, 19.8), dpi=150,
                                # figsize=(8.3, 9.3375), dpi=250,
                               sharex=False, sharey=False)
    plt.subplots_adjust(hspace=1.0)
    for ch_id in range(3):
        wf_sum_ch = pd.Series(wf_sum_dict[ch_id])
        # hist_wf_sum[ch_id] = ax_5[ch_id][0].hist(wf_sum_ch, bins=10000,
        # hist_wf_sum[ch_id] = ax_5[ch_id][0].hist(wf_sum_ch, bins=5000,
        hist_wf_sum[ch_id] = ax_5[ch_id][0].hist(wf_sum_ch, bins=250,
                        range=hist_wf_sum_range[ch_id], color=f'C{ch_id}', label = f'{ch_id}')
        ax_5[ch_id][0].set_title(f'Pre cuts {part_id}', fontsize=22)
        ax_5[ch_id][0].set_xlabel('Full waveform sum [4ns$\cdot$ADC units]', fontsize=22)
        # ax_5[ch_id][0].tick_params(axis='x', rotation=30)
        ax_5[ch_id][0].ticklabel_format(style='scientific', axis='x', scilimits=[-1, 2]) # scilimits may be specific to run00156
        # ax_5[ch_id][0].tick_params(which='major', length=9)
        ax_5[ch_id][0].tick_params(axis='both', which='major', labelsize=20)
        ax_5[ch_id][0].set_ylabel('counts', fontsize=22)
        ax_5[ch_id][0].set_ylim(0, 640) #Thesis
        ax_5[ch_id][0].legend(fontsize=24) #Thesis
        ax_5[ch_id][0].grid()
        # hist_wf_sum_postcut[ch_id] = ax_5[ch_id][1].hist(wf_sum_ch[part_event_PassList], bins=10000, #run00126
        # hist_wf_sum_postcut[ch_id] = ax_5[ch_id][1].hist(wf_sum_ch[part_event_PassList], bins=5000, #run00126
        hist_wf_sum_postcut[ch_id] = ax_5[ch_id][1].hist(wf_sum_ch[part_event_PassList], bins=250, #run00126
        # hist_wf_sum_postcut[ch_id] = ax_5[ch_id][1].hist(wf_sum_ch[event_PassList], bins=10000, #run00126 #TODO: event_PassList --> part_event_PassList
        # hist_wf_sum_postcut[ch_id] = ax_5[ch_id][1].hist(wf_sum_ch[event_PassList], bins=5000, #run00152
        # hist_wf_sum_postcut[ch_id] = ax_5[ch_id][1].hist(wf_sum_ch[event_PassList], bins=100, #run00159_truncated
                        range=hist_wf_sum_range[ch_id], color=f'C{ch_id}', label = f'{ch_id}')
        ax_5[ch_id][1].set_title(f'Post cuts {part_id}', fontsize=22)
        ax_5[ch_id][1].set_xlabel('Full waveform sum [4ns$\cdot$ADC units]', fontsize=22)
        ax_5[ch_id][1].ticklabel_format(style='scientific', axis='x', scilimits=[-1, 2])
        # ax_5[ch_id][1].tick_params(axis='both', which='major', length=9)
        ax_5[ch_id][1].tick_params(axis='both', which='major', labelsize=20)
        ax_5[ch_id][1].set_ylabel('counts', fontsize=22)
        ax_5[ch_id][1].set_ylim(0, 640) #Thesis
        ax_5[ch_id][1].legend(fontsize=24) #Thesis
        ax_5[ch_id][1].grid()
        # ax_5[ch_id][1].tick_params(axis='x', rotation=30)
    save_plot(fig_5, f'histogram_wf_sum_before_and_after_cuts{part_id}')
    pickle_dict( hist_wf_sum_postcut, f'hist_wf_sum_postcuts {part_id}')
    # return hist_wf_sum_postcut
    return hist_wf_sum_postcut, ax_5, fig_5

def fit_charge_distribution(hist_wf_sum_postcut: dict, ax_5, fig_5, part_id:str = ''):
    fit_param_dict = {0:0, 1:0, 2:0}
    fit_std_dict = {0:0, 1:0, 2:0}
    red_chisqr_dict = {0:0, 1:0, 2:0}
    peak_loc = {0:0, 1:0, 2:0}
    x_range = {0: 0,1: 0,2: 0}
    # distance_to_peak = {0:[15, 12], 1:[25, 20], 2:[25, 20]} # 00159_truncated
    # distance_to_peak = {0:[1000, 1000], 1:[1000, 1200], 2:[700, 800]}
    # distance_to_peak = {0:[1000, 1000],  1:[545, 645], 2:[700, 800]} # 1:[645, 745], 2:[700, 800]} # run00126_truncated
    # distance_to_peak = {0:[700, 800],  1:[545, 645], 2:[700, 800]}
    # distance_to_peak = {0:[1200, 900],  1:[700, 1100], 2:[1100, 1000]}# run00156_truncated_new
    distance_to_peak = {0:[37, 35],  1:[33, 36], 2:[49, 40]}# run00156_truncated_new. bins=250

    for ch_id in range(3): # dynamic
        peak_loc[ch_id] = np.argmax(hist_wf_sum_postcut[ch_id][0])
        # x_range[ch_id] = np.arange(peak_loc[ch_id]-600, peak_loc[ch_id] + 1000)
        # x_range[ch_id] = np.arange(peak_loc[ch_id]-300, peak_loc[ch_id] + 500)
        # x_range[ch_id] = np.arange(peak_loc[ch_id]-625, peak_loc[ch_id] + 625)
        # x_range[ch_id] = np.arange(peak_loc[ch_id] - distance_to_peak[ch_id][0], 
        #                            peak_loc[ch_id] + distance_to_peak[ch_id][1]) # run00126_truncated, run00156_truncated_new
        x_range[ch_id] = np.arange(peak_loc[ch_id] - distance_to_peak[ch_id][0], 
                                   peak_loc[ch_id] + distance_to_peak[ch_id][1]) # run00126_truncated, run00156_truncated_new, bins=250
        # x_range[ch_id] = np.arange(97, 168)
    # x_range = {0: np.arange(2912, 3560), 1: np.arange(2800, 3534), 2: np.arange(2003, 2803)}  # PEN run00152
    # x_range = {0: np.arange(5332, 7999), 1: np.arange(5549, 6149), # TPB run00126
    #            2: np.arange(5900, 7400)} # (6169, 7250) (6219, 7199) (6149, 6899)
    # x_range = {0: np.arange(2912, 3560), 1: np.arange(2800, 3534), 2: np.arange(2003, 2803)} #PEN run00156_truncated
    # p0_input = {0: [2.5E6, 1E6, 100], 1: [0.18E6, 0.12E6, 100], 2: [0.32E6, 0.18E6, 100]}
    p0_input = {0: [0.5E6, 0.12E6, 1E8], 1: [0.35E5, 0.12E5, 1E7], 2: [0.6E5, 0.17E5, 1E7]} # bins=250

    # fig_7, ax_7 = plt.subplots(3, 1, figsize=(10, 8)) #(18, 16))
    for ch_id in range(3):
        hist_content, hist_edges, _histObjects = hist_wf_sum_postcut[ch_id]
        del _histObjects
        ch_range = x_range[ch_id]
        fitted_parameters, pcov = curve_fit(f_gauss,
                                    hist_edges[ch_range], hist_content[ch_range], \
                                    p0 = p0_input[ch_id],
                                    )

        red_chisqr_value = red_chisq(hist_content[ch_range], \
            f_gauss(hist_edges[ch_range], *fitted_parameters), fitted_parameters
        )
        fit_param_dict[ch_id] = fitted_parameters
        fit_std_dict[ch_id] = np.sqrt(np.diag(pcov))
        red_chisqr_dict[ch_id] = red_chisqr_value
        # ax_7[ch_id].hist(hist_edges[:-1], hist_edges, weights=hist_content, histtype='stepfilled',
        #                  color=f'C{ch_id}', label =f'{ch_id}')
        # ax_5[ch_id][1].hist(hist_edges[:-1], hist_edges, weights=hist_content, histtype='stepfilled',
        #                  color=f'C{ch_id}', label =f'{ch_id}')
#         ax_7[ch_id].plot(hist_edges[ch_range], f_gauss(hist_edges[ch_range], *fitted_parameters),
#                          label='fit', color=f'C{ch_id+3}')
        ax_5[ch_id][1].plot(hist_edges[ch_range], f_gauss(hist_edges[ch_range], *fitted_parameters),
                         label='fit', color=f'C{ch_id+3}')
        # text_in_box = AnchoredText(f"statistics = {np.sum(hist_content[ch_range])} \nreduced chisqr = {red_chisqr_value:.2f}", \
        text_in_box = AnchoredText(f"statistics = {int(np.sum(hist_content[ch_range]))}" '\n$\\chi^{2}/\\nu$' f"= {red_chisqr_value:.2f}", \
                                                    prop=dict(size=24), loc='upper left') #TODO: adjust
        # ax_7[ch_id].add_artist(text_in_box)
        ax_5[ch_id][1].add_artist(text_in_box)
        # ax_7[ch_id].tick_params(axis='x', rotation=30)
        # ax_5[ch_id][1].tick_params(axis='x', rotation=30)
        # ax_7[ch_id].ticklabel_format(style='scientific', axis='x', scilimits=[-1, 2])
        # ax_5[ch_id][1].ticklabel_format(style='scientific', axis='x', scilimits=[-1, 2])
        # if ch_id == 2:
        #     ax_5[ch_id][1].set_xlabel('integrated charge [4ns$\cdot$ADC units]')
        #     ax_7[ch_id].set_xlabel('integrated charge [4ns$\cdot$ADC units]')
        # ax_5[ch_id][1].set_ylabel('counts')
        # ax_5[ch_id][1].grid()
        # ax_5[ch_id][1].set_title('Fit to distribution of full wf sum')
#         ax_7[ch_id].set_ylabel('counts')
#         ax_7[ch_id].grid()
#         # ax_7[ch_id].set_title('Fit to distribution of full wf sum')
#         ax_7[ch_id].legend()

    peak_asymmetry = (fit_param_dict[1][0] - fit_param_dict[2][0])/(fit_param_dict[1][0] + fit_param_dict[2][0])
    # ax_7[0].add_artist(AnchoredText(f"Asymmetry = {peak_asymmetry:.3f}", loc = 'center left'))  # commented out for paper
    asymmetry_box = AnchoredText(f"Asymmetry = {peak_asymmetry:.3f}", prop=dict(size=24), loc = 'center left', bbox_to_anchor=(0., 0.65), bbox_transform=ax_5[0][1].transAxes)
    asymmetry_box.patch.set_alpha(0.5)
    ax_5[0][1].add_artist(asymmetry_box)
    # fig_7.suptitle(f'Fit to distribution of full wf sum{part_id}') # commented out for paper
    save_plot(fig_5, f'charge_distribution_fit{part_id}')
    # save_plot(fig_7, f'charge_distribution_fit{part_id}')
    print(f'\n fit to charge distributions post cuts {part_id}:')
    # print('\n')
    for ch_id in range(3):
        print(f'red. chisqr for {ch_id}: {red_chisqr_dict[ch_id]}')
        print(f'fitted parameters for {ch_id}: {fit_param_dict[ch_id]}')
        print(f'std deviation for the fitted parameters for {ch_id}: {fit_std_dict[ch_id]}')
    print(f'Fitted charge peak Asymmetry: {peak_asymmetry}')
    return fit_param_dict

def stack_flt_wf(flt_dict:dict):
    stacked_flt_wf_dict= {
    0:np.zeros_like(wfs.iloc[0][0]),
    1:np.zeros_like(wfs.iloc[0][0]),
    2:np.zeros_like(wfs.iloc[0][0]),
    }
    for ch_id in range(3):
        # stacked_flt_wf_dict[ch_id] = pd.Series(flt_dict[ch_id])[event_PassList]  #TODO: loop can be skipped using pandas.Series
        for event_id in event_PassList:
            stacked_flt_wf_dict[ch_id] += flt_dict[ch_id][event_id]
    # pickle_dict(stacked_flt_wf_dict, 'stacked_flt_wf_dict')
    return stacked_flt_wf_dict

def create_part_stack_flt_wf(flt_dict:dict, term_points: list):
    part_stacked_flt_wf_dict= {
    0:np.zeros_like(wfs.iloc[0][0]),
    1:np.zeros_like(wfs.iloc[0][0]),
    2:np.zeros_like(wfs.iloc[0][0]),
    }
    for ch_id in range(3):
        for event_id in event_PassList[term_points[0] : term_points[1]]:
            part_stacked_flt_wf_dict[ch_id] += flt_dict[ch_id][event_id]
    return part_stacked_flt_wf_dict

def stack_wf(wfs:pd.core.series.Series):
    stacked_wf_dict = {
    0:np.zeros_like(wfs.iloc[0][0]),
    1:np.zeros_like(wfs.iloc[0][0]),
    2:np.zeros_like(wfs.iloc[0][0])
    }
    for ch_id in range(3):
        for event_id in range(wfs.shape[0]):
            stacked_wf_dict[ch_id] += wfs.iloc[event_id][ch_id]
    # pickle_dict(stacked_wf_dict, 'stacked_wf_dict')
    return stacked_wf_dict

def calculate_time_constant():
    print('\n calculating time constant..')

    def f1_func(x: np.ndarray, a0: float, a2: float, a3: float, a4: float) -> np.ndarray:
    #     return (a0/a2)*np.exp(-(x)/a2) + (a3/a4)*np.exp(-(x)/a4)
       return (a0)*np.exp(-(x)/a2) + (a3)*np.exp(-(x)/a4)
    # def f0_func(x, a0, a1, a2, a3, a4):
    #     # a1 = 424
    #     return (a0/a2)*np.exp(-(x-a1)/a2) + (a3/a4)*np.exp(-(x-a1)/a4)
    def fit_stack(given_func: 'function', wf_ch_data: np.ndarray, fit_range: tuple, 
                  f1_range: np.ndarray):
        (fit_begin, fit_end) = fit_range
        f_xrange = 4*f1_range
        # bounds_input = ([1.0E3, 0E0, 1.0E03, 0E0], [1.0E9, 1.0E6, 1.0E9, 1.0E4]) # f1_func
        # p0_input = [10000, 3000, 5000, 100] # f1_func
        bounds_input = ([1.0E0, 0.0, 0.0, 1.0E03, 0.0], [1.0E10, 4.0E3, 1.0E6, 1.0E9, 1.0E4]) # f0_func
        # p0_input = [1.0E7, 2000, 3000, 1.0E5, 100] # f0_func
        p0_input = [1E7, 2000, 3000, 1E6, 10] # 156 truncated new
        fitted_parameters, pcov = curve_fit(given_func,
                                f_xrange[fit_begin:fit_end], wf_ch_data[f1_range][fit_begin:fit_end], \
                                p0 = p0_input,
                                bounds = bounds_input,
                                )
        return fitted_parameters, pcov

    def fit_all_channels(data_name:str, wf_data: np.ndarray):
        # (fit_begin, fit_end) = (50, 750) #f1_func LIDINE presentation
        (fit_begin, fit_end) = (25, 750) #f0_func
        # (fit_begin, fit_end) = (75, 750) #f0_func run00156_truncated_new
        fit_param_dict= {0:0, 1:0, 2:0}
        fit_std_dict = {0:0, 1:0, 2:0}
        red_chisqr_dict = {0:0, 1:0, 2:0}
        statbox_ls = [] # _0 for unshifted waveform
        fitresult_ls = []

        fig_8, ax_8 = plt.subplots(figsize=(10,6), sharex=True)
        for ch_id in range(3):
            wf_ch_data = wf_data[ch_id]
            wf_peak_ch = np.argmax(wf_ch_data)
            f1_range = np.arange(wf_ch_data.shape[0])
            f_xrange = 4*f1_range # sampling rate 1 sample = 4 ns?
            ax_8.plot(4*f1_range, wf_ch_data, linestyle=':', label = f'{ch_id}')

            ## def f0_func(x, a0, a2, a3, a4): # fit function originally proposed by Marcin
            ##     a1 = wf_peak_ch# debug
            ##     return (a0/a2)*np.exp(-(x-a1)/a2) + (a3/a4)*np.exp(-(x-a1)/a4)
            def f0_func(x: np.ndarray, a0: float, a1: float, a2: float, a3: float, a4: float) -> np.ndarray:
               return (a0)*np.exp(-(x-a1)/a2) + (a3)*np.exp(-(x-a1)/a4)
            print(f'Fitting ch_id:{ch_id}')#temp debug
            fit_param_dict[ch_id], pcov = fit_stack(f0_func, wf_ch_data, 
                                                (fit_begin+wf_peak_ch, fit_end+wf_peak_ch), f1_range)
            fit_std_dict[ch_id] = np.sqrt(np.diag(pcov))
            ax_8.plot(f_xrange[fit_begin+wf_peak_ch:fit_end+wf_peak_ch], 
                         f0_func(f_xrange[fit_begin+wf_peak_ch:fit_end+wf_peak_ch], 
                              *fit_param_dict[ch_id]), linewidth=2.5, color=f'C{ch_id+3}', label=f'fit {ch_id}')
            red_chisqr_dict[ch_id] = red_chisq(wf_ch_data[f1_range][fit_begin+wf_peak_ch:fit_end+wf_peak_ch], 
                                    f0_func(f_xrange[fit_begin+wf_peak_ch:fit_end+wf_peak_ch], 
                                    *fit_param_dict[ch_id]), fit_param_dict[ch_id])
            ## statbox_ls.append(f'{ch_id}: red. chisqr = {red_chisqr_dict[ch_id]:.2f}')
            statbox_ls.append(f'{ch_id}: ' '$\\chi^{2}/\\nu$' f' = {red_chisqr_dict[ch_id]:.2f}')
            # fitresult_ls.append(f'{ch_id}: $\\tau_{3}$ = {fit_param_dict[ch_id][2]:.1f} | $\\tau_{1}$ = {fit_param_dict[ch_id][4]:.1f}')
            fitresult_ls.append(f'{ch_id}: $\\tau_{3}$ = {fit_param_dict[ch_id][2]:.1f} ns')
            ## ax_8.axvline(f_xrange[wf_peak_ch], color='gray', linestyle=':')

        text_in_box = AnchoredText('\n'.join(statbox_ls), prop=dict(size=14), loc='lower left')
        ax_8.set_xlim(0, 6000) # just for the paper
        ax_8.add_artist(text_in_box)
        fit_result_in_box = AnchoredText('\n'.join(fitresult_ls), prop=dict(size=14), loc='lower center')
        ax_8.add_artist(fit_result_in_box)
        ax_8.legend(loc = 'lower right')
        ## ax_8.set_title(f'stacked {data_name}') # =filtered wf
        ax_8.set_yscale('log')
        ax_8.grid()
        ax_8.set_xlabel('time [ns]')
        ax_8.set_ylabel('amplitude [ADC units]')
        if 'Filtered wf' in data_name:
            save_plot(fig_8, 'stacked_flt_wf')
        elif 'Stacked wf' in data_name:
            save_plot(fig_8, 'stacked_wf') 
        elif 'wf part' in data_name:
            save_plot(fig_8, f"wf_{data_name.split(sep = ' ')[-1]}")
        print(f'\n Results using {data_name}:')

        for ch_id in range(3):
            print(f'red. chisqr {ch_id}:{red_chisqr_dict[ch_id]}')
            print(f'fit param {ch_id}:{fit_param_dict[ch_id]}')
            print(f'std deviation for the fitted parameters for {ch_id}: {fit_std_dict[ch_id]}')
        return fit_param_dict, red_chisqr_dict
    stacked_flt_wf_dict = stack_flt_wf(flt_dict)
    ## stacked_flt_wf_dict = pickle.load(open('../output/00126_part_output/stacked_flt_wf_dict.pkl', 'rb')) #debug
    ## stacked_wf_dict = stack_wf(wfs)
    ## stacked_wf_dict = pickle.load(open('../output/00126_part_output/stacked_wf_dict.pkl', 'rb')) #debug
    ## fit_param_dict, red_chisqr_dict = fit_all_channels('Stacked wf', stacked_wf_dict)
    # fit_all_channels('Stacked Filtered wf', stacked_flt_wf_dict)
    fit_param_dict, red_chisqr_dict = fit_all_channels('Stacked Filtered wf', stacked_flt_wf_dict) ## optional: disable if only the PartWise analysis is needed

    ## --- PartWise --- ##
    ## --- create parts: instead of one stack we get n stacks. ----
    # number_of_parts = 5
    # break_indices = np.linspace(0, len(event_PassList), number_of_parts+1)
    # for part_id in range(number_of_parts):
    #     # print('start end event_PassList:', break_indices[part_id], break_indices[part_id + 1])     # debug
    #     part_stacked_flt_wf = create_part_stack_flt_wf(flt_dict, [int(break_indices[part_id]), int(break_indices[part_id + 1])])
    #     part_fit_param_dict, part_red_chisqr_dict = fit_all_channels(f'Stacked Flt wf part_{part_id}', part_stacked_flt_wf)
    
    return fit_param_dict, red_chisqr_dict #TODO: write to csv.

## ----------------------------------------- program -----------------------------------------

print(f'\n Analysis started...')
print(f'\n processing {filename}...')
print(f'\n total number of events = {wfs.shape[0]}...')
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
    pretrigger_sum[1].append( np.sum(wfs.iloc[event_x][1][:350]) )
    pretrigger_sum[2].append( np.sum(wfs.iloc[event_x][2][:350]) )
    pretrigger_sum[0].append( np.sum(wfs.iloc[event_x][0][:350]) )
    wf_sum = np.sum(wfs.iloc[event_x], axis=1)
    for ch_x in range(3):
        wf_sum_dict[ch_x].append( wf_sum[ch_x] )
        
    com_arr = calculate_com(wfs, event_x)
    com_dict[0].append(com_arr[0])
    com_dict[1].append(com_arr[1])
    com_dict[2].append(com_arr[2])
del com_arr

flt_dict = create_flt_wfs(wfs) # pass it to pulse_difference

## ----------------------------------------- Histograms -----------------------------------------

fig_0, ax_0 = plt.subplots( 3, 1, figsize=(10, 8), sharex=True, sharey = True)#False)
for ch_x in range(3):
    ax_0[ch_x].hist(pretrigger_sum[ch_x], bins = np.arange(-6000, 200_000, 1000), 
                            color=f'C{ch_x}', label = f'{ch_x}')
    ax_0[ch_x].set_yscale('log')
    ax_0[ch_x].set_ylabel('Counts')
    ax_0[ch_x].legend()
    ax_0[ch_x].grid()
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    # fig_0.suptitle('Histogram of pretrigger sum')
ax_0[ch_x].set_xlabel('Pre-trigger sum [4ns$\cdot$ADC units]')
# save_plot(fig_0, 'hist_pretrigger_sum')
save_plot(fig_0, 'hist_pretrigger_sum_PEN_updated')
plt.close(fig_0)

hist_features = {
    0: [],
    1: [],
    2: [],
}
com_mean_arr = np.zeros([3,])
com_std_arr = np.zeros([3,])

fig_1, ax_1 = plt.subplots( 3, 1, figsize=(10, 8), sharex=True, sharey=True)

print(f'\n fitting COM distribution peak..')
for ch_x in range(3):
    hist_features[ch_x] = ( ax_1[ch_x].hist(com_dict[ch_x], 
                            # bins=np.arange(-5_000, 5_000, 25),
                            # bins=np.arange(-500, 2500, 10),  # run00126_truncated
                            bins=np.arange(-500, 2500, 20),  # run00156_truncated_new
                            color=f'C{ch_x}', label = f'{ch_x}')
                            )
    ax_1[ch_x].set_ylabel('counts', fontsize=22)
    ax_1[ch_x].legend()
    ax_1[ch_x].grid()
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    # fig_1.suptitle('Histogram of Center Of Mass')
    com_mean_arr[ch_x], com_std_arr[ch_x] = fit_com_peak(ch_x, ax_1, hist_features)
ax_1[2].set_xlabel('Centre Of Mass [4ns$\cdot$ADC units]', fontsize=22)
# save_plot(fig_1, 'hist_COM')
save_plot(fig_1, 'hist_COM_PEN_updated')
plt.close(fig_1)
del hist_features

## ----------------------------------------- cuts -----------------------------------------
# com_threshold = 300 # not in use
ch_id = 1
# com_below_xsigma = com_mean_arr - 1.75*com_std_arr # was 2 # explore and investigate thresholds
# com_above_xsigma = com_mean_arr + 1.75*com_std_arr

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

## pulse difference histogram
pulse_difference_ls = []

apply_cuts(wfs)

# hist_pulse_difference() # unComment if pulse difference histogram is desired.

# hist_plot_range = (0e6, 5e6) #run00126
hist_plot_range = (0.0, 1.0e6) #combined run00156 
# hist_plot_range = (0e6, 2.5e6) #run00152, run00126_truncated
# hist_plot_range = (0e6, 1.0e6) #run00156_truncated, 159_truncated

# fig_2, ax_2 = plt.subplots( 5, 1, figsize=(19, 15), sharex=True, sharey = False)
# fig_2, ax_2 = plt.subplots( 5, 1, figsize=(19, 20), dpi=75, sharex=True, sharey = False)
# fig_2, ax_2 = plt.subplots( 5, 1, figsize=(8.3, 13.2), dpi=150, sharex=True, sharey = False) # A4 size: 8.3, 11.7
fig_2, ax_2 = plt.subplots( 5, 1, figsize=(8.3, 11.7), dpi=150, sharex=True, sharey = False) # A4 size: 8.3, 11.7
fig_2.subplots_adjust(hspace=0)
bin_content_0, bin_edges, _PlotsObjects = ax_2[0].hist(wf_sum_dict[0], bins=500, range = hist_plot_range, label = 'Pre cuts')
bin_content_1, bin_edges, _PlotsObjects = ax_2[1].hist(wf_sum_post_cut_dict[1], bins=bin_edges, range = hist_plot_range, label = 'Post 1$^{st}$ cut')
bin_content_2, bin_edges, _PlotsObjects  = ax_2[2].hist(wf_sum_post_cut_dict[2], bins=bin_edges, label = 'Post 2$^{nd}$ cut')
bin_content_3, bin_edges, _PlotsObjects  = ax_2[3].hist(wf_sum_post_cut_dict[3], bins=bin_edges, label = 'Post 3$^{rd}$ cut')
# plt.subplots_adjust(wspace=0.025, hspace=0.025)
# plt.subplots_adjust(wspace=0.025, hspace=0.0025)
# fig_2.subplots_adjust(hspace=0)
# fig_2.suptitle('successive cuts')
ratio_1_0 = np.divide(bin_content_1, bin_content_0, out=np.zeros_like(bin_content_1), where=bin_content_0!=0)
ax_2[4].plot(bin_edges[:-1], 1.-ratio_1_0, alpha=0.5, color='#f86f6c', label = '1$^{st}$ cut')
ratio_2_1 = np.divide(bin_content_2, bin_content_1, out=np.zeros_like(bin_content_2), where=bin_content_1!=0)
ax_2[4].plot(bin_edges[:-1], 1.-ratio_2_1, alpha=0.5, color='#447acd', label = '2$^{nd}$ cut')
ratio_3_2 = np.divide(bin_content_3, bin_content_2, out=np.zeros_like(bin_content_3), where=bin_content_2!=0)
ax_2[4].plot(bin_edges[:-1], 1.-ratio_3_2, alpha=0.5, color='#69ebd2', label = '3$^{rd}$ cut')
ratio_3_0 = np.divide(bin_content_3, bin_content_0, out=np.zeros_like(bin_content_3), where=bin_content_0!=0)
ax_2[4].plot(bin_edges[:-1], 1.-ratio_3_0, alpha=0.5, color='#e59acf', label = 'Overall')
ax_2[4].set_ylim(0.0, 1.0)
# ax_2[4].axhline(y=0.2, linestyle='--')
ax_2[4].axhline(y=0.35, linestyle=':', color='gray')
for subplot_x in range(4):
    ax_2[subplot_x].legend()
    # ax_2[subplot_x].legend(fontsize=22) #Thesis
    ax_2[subplot_x].grid()
    ax_2[subplot_x].set_ylabel('Counts')
    # ax_2[subplot_x].set_ylabel('Counts', fontsize=22)
    ax_2[subplot_x].set_ylim(0, 300) #TODO: dynamic
    ax_2[subplot_x].set_xlim(0E6, 1E6) #paper
    # ax_2[subplot_x].set_ylim(0, 2000) #combined run00156
# ax_2[4].legend()
ax_2[4].legend(fontsize=11) #Thesis
ax_2[4].set_ylabel('Cut efficiency')
# ax_2[4].set_ylabel('Cut efficiency', fontsize=22)
ax_2[4].set_xlabel('Full waveform sum [4ns$\cdot$ADC units]')
# ax_2[4].set_xlabel('Full waveform sum [4ns$\cdot$ADC units]', fontsize=22)
# save_plot(fig_2, 'successive_cuts')
save_plot(fig_2, 'successive_cuts_PEN_updated')
plt.close(fig_2)
del bin_content_1, bin_content_2, bin_content_3, bin_edges, _PlotsObjects

fig_3, ax_3 = plt.subplots( 3, 1, figsize=(10, 8), sharex=True, sharey = True)
for ch_id in range(3):
    # ax_3[ch_id].hist(com_post_cut_dict[ch_id], bins=np.arange(-5_000, 5_000, 25), 
    ax_3[ch_id].hist(com_post_cut_dict[ch_id], bins=np.arange(-500, 2500, 10), # run00126_truncated_new #paper
                        color=f'C{ch_id}', label = f'{ch_id}')    
    ax_3[ch_id].legend()
    # ax_3[ch_id].legend(fontsize=22) #Thesis
    ax_3[ch_id].grid()
plt.subplots_adjust(wspace=0.025, hspace=0.025)
fig_3.suptitle('Histogram of Center Of Mass post cuts')
save_plot(fig_3, 'hist_COM_post_cut')
plt.close(fig_3)
## ----------------------------------------- Time Constant -----------------------------------------

fit_param_dict, red_chisqr_dict = calculate_time_constant()
# calculate_time_constant() ## optional: useful when disabling fit to entire run

# ## ------------------------------ Fit Gauss to integrated Charge Distribution ----------------------
# hist_wf_sum_postcut = histogram_wf_sum_before_and_after_cuts(event_PassList)
# ## hist_wf_sum_postcut = pickle.load(open('../output/00126_part_output/hist_wf_sum_postcut.pkl', 'rb')) #debug
# fit_charge_distribution(hist_wf_sum_postcut)

hist_wf_sum_postcut, ax_5, fig_5 = histogram_wf_sum_before_and_after_cuts(event_PassList)
fit_charge_distribution(hist_wf_sum_postcut, ax_5, fig_5)
# ## ------------------------------ PartWise: Fit Gauss to integrated Charge Distribution ----------------------
# number_of_parts = 6
# break_indices = np.linspace(0, len(event_PassList), number_of_parts+1)
# for part_id in range(number_of_parts):
#     part_hist_wf_sum_postcut = histogram_wf_sum_before_and_after_cuts(event_PassList[int(break_indices[part_id]) : int(break_indices[part_id + 1])], 
#                                                                         part_id = f'_part_{part_id}')
#     fit_charge_distribution(part_hist_wf_sum_postcut, part_id = f'_part_{part_id}')

print(f'Execution time: {perf_counter() - t0}')