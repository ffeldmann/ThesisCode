#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=9

BASE_CONFIGS="FlowFrameGen/configs/Human36MFramesFlowMeta.yaml FlowFrameGen/configs/VUnet.yaml FlowFrameGen/configs/train.yaml"
TARGET_FRAME_STEP=25
WARPING="none"

name="VUnet_s_"$TARGET_FRAME_STEP"_w_"$WARPING"_wsin_if_wosin_t"
kwargs="-t --target_frame_step "$TARGET_FRAME_STEP" --warping "$WARPING
# command="python AnimalPose/scripts/load_config.py -n "$name" -b "$BASE_CONFIGS" "$kwargs
# echo $command
# $command

#python AnimalPose/scripts/load_config.py -b $BASE_CONFIGS $kwargs
command="edflow -n "$name" -b "$BASE_CONFIGS" "$kwargs
echo $command
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
which python
$command
