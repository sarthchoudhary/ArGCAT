# ArGSet Calibration and Analysis Tools

## About ArGSet
Argon Gas Setup for measurement of wavelength shifting materials at cryogenic temperature.
[Learn about ArGSet here](https://gitlab.camk.edu.pl/mkuzniak/cezamat/-/wikis/ArgSet)

## Repository structure:
Legacy directory is for old code. It is kept only for archival purpose.

## Requirements
To satify all requirements do these:
1. rename jar_*.yml to jar.yml
2. edit jar.yml to change prefix a/c to your miniconda installation.
3. `conda env create -f jar.yml`
4. This will not install peakdetect package which needs to be installed via pip. Also, peakdetect is outdated package, its source code needs to tweaked slightly to make use of newer versions of scipy FFT. Pyreco has separate installation steps. See [pyreco repository](https://gitlab.camk.edu.pl/mkuzniak/pyreco).

## Calibration: (main)

### Computing resources:
wf search script consumes a lot of computing resources, it should only be executed on an interactive node or submitted as a slurm job.
_Warning:_ The code probably won't be able to complete execution on a login node. This is your only warning!

### Calibration Data Processing:
There are several tools available for calibration data processing:
- create_event_catalogue.py
    Expects midas data files as input. Writes a pickle file containing event catalogues for each channel.
- calculate_pulse_param.py
    Does all processing from finding clean waveforms to fitting with SiPM pulse model. Writes results to pickle files.
- notebooks/histogram_pulse_param.ipynb
    - Channelwise concatenation of events from several runs
    - Performs histogramming on fit parameters, applies cut, and fits gaussian to the resultant histogram.
    - Makes Fingerplots from concatenated fit catalogues. The instructions for generating finger plots will be provided in a separate document.
    - Also, calculates the SPE charge and SNR from the data. 

    - ### Special: self consistency of ARMA filter based approach
        - We want to see if the pulses found using calculate_pulse_param.py with a specific set of ARMA parameters would yield back the same set of ARMA parameters. 
        1. keep the argset.ini at default values of ARMA filter.
        1. run the calculate_pulse_param.py script
        1. process the fit_catalogue with histogram_pulse_param.ipynb. Fine tune the fitter and find the optimum value of pulse parameters.
        1. Use these parameter in argset.ini (sigma and tau should be multiplied by 4 ns)
        1. run the calculate_pulse_param.py script again
        1. process the fit_catalogue with histogram_pulse_param.ipynb. Fine tune the fitter and find the optimum value of pulse parameters.
        1. verify that parameters found in previous step are similar to parameters in step 3.
- notebooks/histogram_pulse_param.ipynb contains the code for making fingerplot from filtered waveform.

### Catalogue:
The eventwise data products are catalogued in form of pandas DataFrames. I call these catalogues. The catalogue for all channels are packaged together as a python dict and written to disk as pickle file.
There are three different types of catalogues generated by these codes: 
- event catalogue: it contains waveforms in a DataFrame. _Future:_ I will explore the option to save these as numpy compressed array (.npz) files instead.
- clean catalogue: contains list of events deemed to be clean along with estimated peak location. The precise definition of clean event is defined within the script, but it means an event with only one prominent pulse.
- fit catalogue: contains list of clean event along with SiPM pulse model parameters. The event for which model parameters could not be estimated contain a none object in fit_param column instead. 

### Pickle files:
Pickle is a data serialization format for python objects. Pickle files are neither secure nor memory efficient. Pickle files should never be used for sharing data. _Future:_ I will explore alternatives to pickle files.

## WLS Data: (process_wls_data)
This branch is meant to do provides tools for processing of wavelength shifter data. The following codes are available:
- src/process_wls_data: Does the actual processing: applies event selection cuts, etc. 
- util/hist2d_eventID_sum.py: time evolution of collected charge for all subruns
- util/combine_wfs.py: combines subruns into a single DataFrame and pickles it.
- util/truncate_wfs.py: combines subruns and saves truncated data into a single DataFrame. Saves the pickle.