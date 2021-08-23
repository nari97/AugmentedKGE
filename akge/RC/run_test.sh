#!/bin/bash

# sbatch run_test.sh /home/crrvcs/OpenKE/ transe 0 valid|test
#SBATCH -t 12:0:0

#SBATCH -A StaMp -p tier3 -n 1 -c 2

# Job memory requirements in MB
#SBATCH --mem=25000

#SBATCH --output=./LogsTest/Test_%A_%a.out
#SBATCH --error=./LogsTest/Test_%A_%a.err

folder=$1
model=$2
dataset=$3
type=$4

# Loop and submit all the jobs
echo " * Submitting job array..."

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

/home/crrvcs/ActivePython-3.7/bin/python3 -u ./Code/test.py ${folder} ${model} ${dataset} ${type}

echo " Done with job array"