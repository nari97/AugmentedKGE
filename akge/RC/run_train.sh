#!/bin/bash

# sbatch --array=0-14 run_train.sh /home/crrvcs/OpenKE/ transe 0

#SBATCH -t 24:0:0

#SBATCH -A StaMp -p tier3 -n 1 -c 2

# Job memory requirements in MB
#SBATCH --mem=20000

#SBATCH --output=./LogsTrain/Train_%A_%a.out
#SBATCH --error=./LogsTrain/Train_%A_%a.err

folder=$1
modelName=$2
dataset=$3

# Loop and submit all the jobs
echo " * Submitting job array..."

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

/home/crrvcs/ActivePython-3.7/bin/python3 -u ./Code/train.py ${folder} ${modelName} ${dataset} ${SLURM_ARRAY_TASK_ID}

echo " Done with job array"