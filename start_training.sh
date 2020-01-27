#!/bin/bash

#export CUDA_VISIBLE_DEVICES=9

BASE_CONFIGS="AnimalPose/configs/single_cat_pose.yaml AnimalPose/configs/train_unet.yaml"

name="AnimalPose_UNet"
kwargs="-t "

command="edflow -n "$name" -b "$BASE_CONFIGS" "$kwargs
echo $command
$command
