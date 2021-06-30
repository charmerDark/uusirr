#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME="experiments"

# datasets
SINTEL_HOME=MPI-Sintel-complete/

# model and checkpoint
MODEL=new_model
EVAL_LOSS=MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel
SIZE_OF_BATCH=4

# save path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL-$TIME"

# training configuration
python main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[258, 302, 346, 368, 374, 379, 401, 423, 445, 467]" \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=1.5e-05 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--total_epochs=489 \
--training_augmentation=RandomAffineFlowOccSintel \
--training_augmentation_crop="[384,768]" \
--training_dataset=SintelTrainingCombTrain \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$SINTEL_HOME \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=SintelTrainingCombValid  \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$SINTEL_HOME \
--validation_key=epe \
--validation_loss=$EVAL_LOSS