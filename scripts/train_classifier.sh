#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export STUDENT_ID=$(whoami)
source /home/${STUDENT_ID}/miniconda3/bin/activate uusirr

python /home/s2751455/uusirr/PWC_Classifer_training.py