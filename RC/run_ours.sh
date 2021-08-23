#!/bin/bash

# sbatch --array=1-5,9-13,17-21,25-29,33-37,41-45,49-53,57-61 run_ours.sh /home/crrvcs/OpenKE/ 1

#SBATCH -t 120:0:0

#SBATCH -A StaMp -p tier3 -n 1 -c 4

# Job memory requirements in MB
#SBATCH --mem=61440

#SBATCH --output=./LogsTransE/Ours_%A_%a.out
#SBATCH --error=./LogsTransE/Ours_%A_%a.err

folder=$1
norm=$2

# Loop and submit all the jobs
echo " * Submitting job array..."

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

python3 -u ./TransE/TransE3.py ${folder} ${SLURM_ARRAY_TASK_ID} ${norm}

echo " Done with job array"