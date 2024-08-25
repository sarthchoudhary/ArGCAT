#!/bin/bash
##SBATCH --job-name=No_Name    # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sarthak@camk.edu.pl   # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --time=6:00:00               # Time limit hrs:min:sec (you may not want this)
#SBATCH --output=%N-%j.out   # Standard output and error log
#SBATCH --mem=35G                    # Allocated 250 megabytes of memory for the job.
#SBATCH -A bejger-grp
#SBATCH --partition=bigmem
###SBATCH --partition=dgx
##SBATCH --gres=gpu:turing:1                # specify gpu

# python /home/sarthak/my_projects/argset/src/create_event_catalogue.py
# python /home/sarthak/my_projects/argset/src/calculate_wf_param.py
# python /home/sarthak/my_projects/argset/src/cut_eff_comparison.py
python /home/sarthak/my_projects/argset/src/selection_cuts.py
# python /home/sarthak/my_projects/argset/src/truncate_wfs.py
# python /home/sarthak/my_projects/argset/src/hist2d_eventID_sum.py