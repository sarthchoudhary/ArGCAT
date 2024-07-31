#!/bin/bash
##SBATCH --job-name=test_large_midas_files    # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sarthak@camk.edu.pl   # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --time=2-12:00:00               # Time limit hrs:min:sec (you may not want this)
#SBATCH --output=%N-%j.out   # Standard output and error log
#SBATCH --mem=16G                    # Allocated 250 megabytes of memory for the job.
###SBATCH -A bejger-grp
#SBATCH --partition=bigmem
###SBATCH --partition=dgx
##SBATCH --gres=gpu:turing:1                # specify gpu


python /home/sarthak/my_projects/argset/src/create_event_catalogue.py
# python /home/sarthak/my_projects/argset/src/calculate_wf_param.py