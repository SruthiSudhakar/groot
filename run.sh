#!/bin/bash
set -e

cd /proj/vondrick3/sruthi/Appaji/Isaac-GR00T

# conda activate does not work in a non-interactive script unless conda is
# initialized first. Without this the activate silently fails and the script
# runs in whatever env is already active (wrong robocasa -> ImportError on
# TASK_SET_REGISTRY).
source /proj/vondrick3/sruthi/miniconda3/etc/profile.d/conda.sh
conda activate gr00t

CKPT=/proj/vondrick3/sruthi/Appaji/released_checkpoints_groot/gr00t_n1-5/multitask_learning/checkpoint-120000

# Map each task to its render_camera value
declare -A TASK_CAMERA
TASK_CAMERA[CloseBlenderLid]="robot0_agentview_right"
TASK_CAMERA[CloseFridge]="robot0_agentview_right"
TASK_CAMERA[CloseToasterOvenDoor]="robot0_agentview_right"
TASK_CAMERA[CoffeeSetupMug]="robot0_eye_in_hand"
TASK_CAMERA[NavigateKitchen]="robot0_agentview_right"
TASK_CAMERA[OpenCabinet]="robot0_agentview_right"
TASK_CAMERA[OpenDrawer]="robot0_agentview_right"
TASK_CAMERA[OpenStandMixerHead]="robot0_agentview_right"
TASK_CAMERA[PickPlaceCounterToCabinet]="robot0_eye_in_hand"
TASK_CAMERA[PickPlaceCounterToStove]="robot0_eye_in_hand"
TASK_CAMERA[PickPlaceDrawerToCounter]="robot0_agentview_right"
TASK_CAMERA[PickPlaceSinkToCounter]="robot0_eye_in_hand"
TASK_CAMERA[PickPlaceToasterToCounter]="robot0_eye_in_hand"
TASK_CAMERA[SlideDishwasherRack]="robot0_eye_in_hand"
TASK_CAMERA[TurnOffStove]="robot0_eye_in_hand"
TASK_CAMERA[TurnOnElectricKettle]="robot0_eye_in_hand"
TASK_CAMERA[TurnOnMicrowave]="robot0_agentview_right"
TASK_CAMERA[TurnOnSinkFaucet]="robot0_eye_in_hand"

TASK=$1
for i in {0..4}; do
    RENDER_CAMERA=${TASK_CAMERA[$TASK]}
    CUDA_VISIBLE_DEVICES=$2 MUJOCO_GL=egl python scripts/run_eval.py \
        --model_path "$CKPT" \
        --data_config panda_omron \
        --embodiment_tag new_embodiment \
        --task_set "$TASK" \
        --split pretrain \
        --port $(($3 + $2)) \
        --n_envs 14 \
        --n_episodes 50 \
        --render_camera "$RENDER_CAMERA"
done