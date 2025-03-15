#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1

export STUDENT_ID=$(whoami)
source /home/${STUDENT_ID}/miniconda3/bin/activate uusirr

python inference_frames.py --model IRR_PWC --checkpoint /home/s2751455/uusirr/saved_check_point/pwcnet/IRR-PWC_sintel/checkpoint_best.ckpt  --image1path  MPI-Sintel-complete/training/final/alley_1/frame_0001.png --image2path MPI-Sintel-complete/training/final/alley_1/frame_0002.png --outputpath test2