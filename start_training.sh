#!/bin/bash

#export CUDA_VISIBLE_DEVICES=9

BASE_CONFIGS="AnimalPose/configs/single_cat_pose.yaml AnimalPose/configs/VUnet.yaml AnimalPose/configs/train.yaml"

name="AnimalPose_VUnet"
kwargs="-t "

command="edflow -n "$name" -b "$BASE_CONFIGS" "$kwargs
echo $command
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
$command
