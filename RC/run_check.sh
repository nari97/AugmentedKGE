#!/bin/bash

# sbatch run_check.sh /home/crrvcs/OpenKE/

#SBATCH -t 1:0:0

#SBATCH -A StaMp -p tier3 -n 1 -c 2

# Job memory requirements in MB
#SBATCH --mem=1024

#SBATCH --output=./LogsTest/Check_%A_%a.out
#SBATCH --error=./LogsTest/Check_%A_%a.err

folder=$1

# Loop and submit all the jobs
echo " * Submitting job array..."

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

/home/crrvcs/ActivePython-3.7/bin/python3 -u ./Code/check_models.py ${folder}

echo " Done with job array"