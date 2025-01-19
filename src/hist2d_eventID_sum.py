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
import sys
from matplotlib import colors
from glob import glob

rc('figure', autolayout=True, figsize=[16, 9], dpi=125, titlesize=20 )
rc('font', family='monospace')
rc('axes', titlesize=20, titleweight='heavy', labelsize=16, labelweight='bold')
rc(('xtick', 'ytick'), labelsize = 18)
rc('legend', fontsize=14)

np.set_printoptions(formatter={'float': lambda x: f"{x:10.4g}"})

## ----------------------------------------- function definitions ---------------------------------


def hist2d_eventID_sum(run_name: str) -> None:
    def save_plot(fig:matplotlib.figure.Figure, file_name: str):
        fig.savefig(path.join(output_subdir, f'{file_name}.pdf'))
        pickle.dump(fig, open(path.join(output_subdir, f'{file_name}.pkl'),  'wb') )

    output_dir = '/work/chuck/sarthak/argset/output_folder/analysis/'
    data_dir = '/work/chuck/sarthak/argset/event_catalogues'
    # run_name_pattern = f'*run{run_name}*' # outdated
    run_name_pattern = f'*{run_name}*'
    subrun_path_list = glob(path.join(data_dir, run_name_pattern))
    subrun_sum_dict = {0: [],
    1: [],
    2: []
    } 

    run_name = f'{run_name}_output'
    output_subdir = path.join(output_dir, run_name)
    if not path.isdir(output_subdir):
            os.mkdir(output_subdir)
    if not subrun_path_list:
        print(f'No event catalogues found matching given pattern. Exiting...')
        sys.exit()
    for subrun_path in subrun_path_list:
    # for subrun in subrun_name_list:
        # subrun_path = path.join(data_dir, subrun)
        subrun_df = pd.read_pickle(subrun_path)
        subrun_wfs = subrun_df['wf']
        del subrun_df
        print(f'file: {subrun_path}')
        print(f'Number of Events: {subrun_wfs.shape[0]}')
        for event_x in range(subrun_wfs.shape[0]):
            wf_sum = np.sum(subrun_wfs.iloc[event_x], axis=1)
            for ch_id in range(3):
                subrun_sum_dict[ch_id].append(wf_sum[ch_id])

    # sum_cutoff = {0: 2.5E6, 1: 250E3, 2: 250E3} # 152
    # sum_cutoff = {0: 1.0E7, 1: 0.750E6, 2: 1E6} # 126
    # sum_cutoff = {0: 5.0E6, 1: 1.0E6, 2: 1.0E6} # 154
    sum_cutoff = {0: 1.5E6, 1: 2.0E5, 2: 2.0E5} # 156
    # sum_cutoff = {0: 2.0E6, 1: 0.250E6, 2: 0.250E6} # 159
    # sum_cutoff = {0: 1.25E6, 1: 0.150E6, 2: 0.55E6} # 162
    # sum_cutoff = {0: 1.25E6, 1: 0.150E6, 2: 0.55E6} # 162_truncated
    # sum_cutoff = {0: 2.5E6, 1: 2.5E5, 2: 2.5E5} # 156_truncated
    # sum_cutoff = {0: 5.10E6, 1: 0.50E6, 2: 0.750E6} # 126_truncated
    # sum_cutoff = {0: 3.3E6, 1: 5.0E5, 2: 5.0E5} # 159_truncated
    for ch_id in range(3) :
        fig_9, ax_9 = plt.subplots(1,1, figsize=(10, 8))
        print(f'Number of events {ch_id}: {len(subrun_sum_dict[ch_id])}')
        y = subrun_sum_dict[ch_id]
        cutoff = sum_cutoff[ch_id]
        y = [subrun_sum for subrun_sum in y if subrun_sum < cutoff and subrun_sum >= 0]
        x = np.arange(len(y))
        hist_content, hist_xedges, hist_yedges, histObjects = \
            ax_9.hist2d(x, y, bins=[100, 100], \
                        label = f'ch {ch_id}', norm=colors.LogNorm(), cmap = 'jet')
                        # label = f'{ch_id}', cmap = 'jet')
            # ax_9[ch_id].hist2d(x, y, bins=1000, label = f'{ch_id}', cmap = 'jet')
        # ax_9[ch_id].legend()
        # ax_9.legend()
        # ax_9.set_title(f'channel: {ch_id}')
        ax_9.set_xlabel('EventID')
        # ax_9.set_ylabel('Full wf sum')
        ax_9.set_ylabel('integrated charge [4ns$\cdot$ADC units]')
        # ax_9.set_xticklabels(ax_9.get_xticklabels(), rotation=45, ha='right')
        ax_9.ticklabel_format(style='scientific', axis='both', scilimits=[-1, 2])
        # fig_9.suptitle(f'EventID vs Event wf sum in Channel {ch_id}') # commented out for paper
        fig_9.colorbar(histObjects, ax=ax_9)
        save_plot(fig_9, f'2d_eventID_sum_zoomed_in_{ch_id}')
        # save_plot(fig_9, f'2d_eventID_sum_{ch_id}')
## ----------------------------------------- program -----------------------------------------

# run_name_list = ['run00162', 'run00159', 'run00156', 'run00154', 'run00126']
# run_name_list = ['run00162']
# run_name_list = ['run00162_truncated']
# run_name_list = ['run00126_truncated']
# run_name_list = ['run00159_truncated']
# run_name_list = ['run00126_part']
# run_name_list = ['run00108']
# run_name_list = ['run00110']
# run_name_list = ['run00124']
# run_name_list = ['run00132']
# run_name_list = ['combinedrun00152']
run_name_list = ['combinedrun00156']
for run_name in run_name_list:
    hist2d_eventID_sum(run_name=run_name)

print(f'Execution time: {perf_counter() - t0}')