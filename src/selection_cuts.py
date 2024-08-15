from time import perf_counter
t0 = perf_counter()

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

event_catalogue_file = f'/work/chuck/sarthak/argset/event_catalogues/event_catalogue_run00126_part.pkl'

event_catalogue = pd.read_pickle(event_catalogue_file)

wfs = event_catalogue['wf']

## ----------------------------------------- function definitions -----------------------------------------
def pulse_difference(event_x):
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

def fit_com_peak(ch_x):
    
    def f_gauss(x, f_mean, f_sigma, f_k):
        return f_k*(1/(f_sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-f_mean)/f_sigma)**2)
    
    hist_content, hist_edges, histObjects = hist_features[ch_x]
    x_range = np.arange(235, 270)
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
    ax_1[ch_x].legend()


    print(f'red_chisqr_value: {red_chisqr_value}')
    print(f'fitted_parameters for {ch_x}: {fitted_parameters}')
    print('\n')
    return fitted_parameters[0], fitted_parameters[1]
## ----------------------------------------- program -----------------------------------------
ch_id = 0

pretrigger_sum = {
    0: [],
    1: [],
    2: []
}
wf_sum_ls = []
com_dict = {0: [],
            1: [],
            2: []
            }

for event_x in range(wfs.shape[0]):
    # wf_sum_ls.append(np.sum(wfs[event_x][ch_id])) # channel 0 sum
    pretrigger_sum[1].append( np.sum(wfs[event_x][1][:350]) )
    pretrigger_sum[2].append( np.sum(wfs[event_x][2][:350]) )
    pretrigger_sum[0].append( np.sum(wfs[event_x][0][:350]) )
    wf_sum_ls.append( np.sum(wfs[event_x], axis=1) )
    com_arr = calculate_com(event_x)
    com_dict[0].append(com_arr[0])
    com_dict[1].append(com_arr[1])
    com_dict[2].append(com_arr[2])

del com_arr

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

    com_mean_arr[ch_x], com_std_arr[ch_x] = fit_com_peak(ch_x)

fig_1.savefig('hist_COM.pdf')
plt.close(fig_1)

del hist_features

com_threshold = 300

ch_id = 1
com_below_xsigma = com_mean_arr - 2*com_std_arr
com_above_xsigma = com_mean_arr + 2*com_std_arr

wf_sum_post_cut_dict = {
    1: [],
    2: [],
    3: []
}

# wf_sum_post_1_cut_ls = []
# wf_sum_post_2_cut_ls = []
# wf_sum_post_3_cut_ls = []
com_post_cut_dict = {0: [],
            1: [],
            2: []
            }
event_list_post_cut = []
wf_sum_post_cut_ls = []

for event_x in range(wfs.shape[0]):
    if pretrigger_sum[0][event_x] <= 4000: # 1st cut: pretrigger sum
        wf_sum_post_cut_dict[1].append(wf_sum_ls[event_x][0]) # sum is always taken from channel 0; should we change it?
        if pulse_difference(event_x) <= 40: # 2nd cut: simultaneity of pulses
        # if True:
            wf_sum_post_cut_dict[2].append(wf_sum_ls[event_x][0])
            # if (np.abs(com_dict[0][event_x] - com_dict[1][event_x]) <= com_threshold) and (np.abs(com_dict[2][event_x] - com_dict[1][event_x]) <= com_threshold): # 3rd cut: concurrence of COM
            if (com_dict[ch_id][event_x] <= com_above_xsigma)[ch_id] and (com_dict[ch_id][event_x] >= com_below_xsigma[ch_id]): # 3rd cut: distance from mean COM
                wf_sum_post_cut_dict[3].append(wf_sum_ls[event_x][0])
                event_list_post_cut.append(event_x)
                wf_sum_post_cut_ls.append(np.sum(wfs[event_x][ch_id]))
                com_post_cut_dict[0].append(com_dict[0][event_x])
                com_post_cut_dict[1].append(com_dict[1][event_x])
                com_post_cut_dict[2].append(com_dict[2][event_x])

hist_plot_range = (0e6, 7.5e6)

fig_2, ax_2 = plt.subplots( 4, 1, figsize=(10, 8), sharex=True, sharey = False)

bin_content_1, bin_edges, _PlotsObjects = ax_2[0].hist(wf_sum_post_cut_dict[1], bins=500, range = hist_plot_range, label = '1st cut [Channel 0]')
bin_content_2, bin_edges, _PlotsObjects  = ax_2[1].hist(wf_sum_post_cut_dict[2], bins=bin_edges, label = '2nd cut [Channel 0]')
bin_content_3, bin_edges, _PlotsObjects  = ax_2[2].hist(wf_sum_post_cut_dict[3], bins=bin_edges, label = '3rd cut [Channel 0]')

plt.subplots_adjust(wspace=0.025, hspace=0.025)
fig_2.suptitle('successive cuts')
for subplot_x in range(3):
    ax_2[subplot_x].legend()
    ax_2[subplot_x].grid()
eff_2_1 = np.divide(bin_content_2, bin_content_1, out=np.zeros_like(bin_content_2), where=bin_content_1!=0)
ax_2[3].plot(bin_edges[:-1], eff_2_1, alpha=0.5, color='red', label = 'cut_2/cut_1')
eff_3_2 = np.divide(bin_content_3, bin_content_2, out=np.zeros_like(bin_content_3), where=bin_content_2!=0)
ax_2[3].plot(bin_edges[:-1], eff_3_2, alpha=0.5, color='magenta', label = 'cut_3/cut_2')

ax_2[3].legend()
ax_2[0].set_ylabel('counts')
ax_2[1].set_ylabel('counts')
ax_2[2].set_ylabel('counts')
ax_2[3].set_ylabel('ratio')
ax_2[3].set_xlabel('wf sum bin')
fig_2.savefig('successive_cuts.pdf')
plt.close(fig_2)

fig_3, ax_3 = plt.subplots( 3, 1, figsize=(10, 8), sharex=True, sharey = True)
for ch_id in range(3):
    ax_3[ch_id].hist(com_post_cut_dict[ch_id], bins=np.arange(-5_000, 5_000, 25), 
                        color=f'C{ch_id}', label = f'{ch_id}')    
    ax_3[ch_id].legend()
    ax_3[ch_id].grid()
plt.subplots_adjust(wspace=0.025, hspace=0.025)
fig_3.suptitle('hist of Center Of Mass post cuts')
fig_3.savefig('hist_COM_post_cut.pdf')
plt.close(fig_3)

print(f'Execution time: {perf_counter() - t0}')