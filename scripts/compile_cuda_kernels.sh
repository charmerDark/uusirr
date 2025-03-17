#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard

#to 
export STUDENT_ID=$(whoami)
source /home/${STUDENT_ID}/miniconda3/bin/activate uusirr

export CUDA_HOME=/opt/cuda-12.2.0

python /home/s2751455/uusirr/models/correlation_package/setup.py install