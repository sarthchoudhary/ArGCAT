import pandas as pd
import pickle
from os import path
import os
from glob import glob

run_name = '00162'
data_dir = '/work/chuck/sarthak/argset/event_catalogues'
# run_name_pattern = f'event_catalogue_run{run_name}*'
# subrun_path_list = glob(path.join(data_dir, run_name_pattern))

output_path = path.join(data_dir, f'event_catalogue_run{run_name}_truncated.pkl')

# prints subrun sizes
# for subrun_path in subrun_path_list:    
#     subrun_df = pd.read_pickle(subrun_path)
#     subrun_wfs = subrun_df['wf']
#     print(f"{subrun_path.split(sep='/')[-1], subrun_wfs.shape[0]}")

# for subrun_name in ['event_catalogue_run00156_00.pkl', 'event_catalogue_run00156_01.pkl']:
#     subrun_path = path.join(data_dir, subrun_name)
#     subrun_df = pd.read_pickle(subrun_path)
#     subrun_wfs = subrun_df['wf']
#     print(f"{subrun_path.split(sep='/')[-1], subrun_wfs.shape[0]}")


## join 156 00 and 01; only need between  25000 and 60000 
# file_126 = path.join(data_dir, 'event_catalogue_run00126.pkl')
# combined_series = pd.read_pickle(file_126)['wf']
file_00 =  path.join(data_dir, 'event_catalogue_run00162_00.pkl')
file_00 = pd.read_pickle(file_00)['wf']
file_01 =  path.join(data_dir, 'event_catalogue_run00162_01.pkl')
file_01 = pd.read_pickle(file_01)['wf']
file_02 =  path.join(data_dir, 'event_catalogue_run00162_02.pkl')
file_02 = pd.read_pickle(file_02)['wf']
file_03 =  path.join(data_dir, 'event_catalogue_run00162_03.pkl')
file_03 = pd.read_pickle(file_03)['wf']
file_04 =  path.join(data_dir, 'event_catalogue_run00162_04.pkl')
file_04 = pd.read_pickle(file_04)['wf']

## combine series
combined_series = pd.concat([file_00, file_01, file_02, file_03, file_04], ignore_index=True)
print(f'combined series shape: {combined_series.shape}') #debug
del file_00, file_01, file_02, file_03, file_04
### truncate
combined_series = combined_series.truncate(before=45_000, after=160_000)
# combined_series = combined_series.truncate(after=100_000) # run00126
combined_series = combined_series.reset_index(drop=True)
combined_series = pd.DataFrame(combined_series)
print(f'truncated series shape: {combined_series.shape}') #debug
print('writing to disk')
combined_series.to_pickle(output_path)