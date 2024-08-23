from time import perf_counter
t0 = perf_counter()

import numpy as np
# from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import pandas as pd
import pickle
from os import path
import os

from matplotlib import colors

rc('figure', autolayout=True, figsize=[16, 9], dpi=125, titlesize=20 )
rc('font', family='monospace')
rc('axes', titlesize=20, titleweight='heavy', labelsize=16, labelweight='bold')
rc(('xtick', 'ytick'), labelsize = 18)
rc('legend', fontsize=14)

np.set_printoptions(formatter={'float': lambda x: f"{x:10.4g}"})

## ----------------------------------------- loading data -----------------------------------------
output_dir = '/home/sarthak/my_projects/argset/output'
data_dir = '/work/chuck/sarthak/argset/event_catalogues'

## ----------------------------------------- function definitions ---------------------------------
def save_plot(fig:matplotlib.figure.Figure, file_name: str):
    fig.savefig(path.join(output_dir, f'{file_name}.pdf'))
    pickle.dump(fig, open(path.join(output_dir, f'{file_name}.pkl'),  'wb') )

## ----------------------------------------- 2D Histogram -----------------------------------------
subrun_sum_dict = {0: [],
    1: [],
    2: []
    } 
subrun_name_list = ['event_catalogue_run00152_00.pkl',  'event_catalogue_run00152_01.pkl']
for subrun in subrun_name_list:
    subrun_path = path.join(data_dir, subrun)
    subrun_df = pd.read_pickle(subrun_path)
    subrun_wfs = subrun_df['wf']
    del subrun_df
    for event_x in range(subrun_wfs.shape[0]):
        wf_sum = np.sum(subrun_wfs.iloc[event_x], axis=1)
        for ch_id in range(3):
            subrun_sum_dict[ch_id].append(wf_sum[ch_id])
            ## x = histogram_wf_sum_ch(subrun_wfs, ch_id)
sum_cutoff = {0: 2.5E6, 1: 250E3, 2: 250E3}
# fig_9, ax_9 = plt.subplots(3,1, figsize=(18, 16))
# for ch_id in range(1, 2) :
for ch_id in range(3) :
    fig_9, ax_9 = plt.subplots(1,1, figsize=(18, 16))
    print(f'Number of events {ch_id}: {len(subrun_sum_dict[ch_id])}')
    y = subrun_sum_dict[ch_id]
    cutoff = sum_cutoff[ch_id]
    y = [subrun_sum for subrun_sum in y if subrun_sum < cutoff ]
    # y = y[1000:len(y)-1000] #debug
    x = np.arange(len(y))
    hist_content, hist_xedges, hist_yedges, histObjects = \
        ax_9.hist2d(x, y, bins=[100, 50], \
                    # label = f'{ch_id}', norm=colors.LogNorm(), cmap = 'jet')
                    label = f'{ch_id}', cmap = 'jet')
        # ax_9[ch_id].hist2d(x, y, bins=1000, label = f'{ch_id}', cmap = 'jet')
    # ax_9[ch_id].legend()
    # ax_9.legend()
    # ax_9.set_title(f'channel: {ch_id}')
    ax_9.set_xlabel('EventID')
    ax_9.set_ylabel('Full wf sum')
    fig_9.suptitle(f'EventID vs Event wf sum {ch_id}')
    fig_9.colorbar(histObjects, ax=ax_9)
    save_plot(fig_9, f'2d_eventID_sum_{ch_id}')

print(f'Execution time: {perf_counter() - t0}')