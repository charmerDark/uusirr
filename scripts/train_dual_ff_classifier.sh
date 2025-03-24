#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-LongJobs
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb


export STUDENT_ID=$(whoami)
source /home/${STUDENT_ID}/miniconda3/bin/activate uusirr

python /home/${STUDENT_ID}/uusirr/dual_image_ff_classifier_training.py 