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

for TASK in \
    NavigateKitchen \
    OpenCabinet \
    OpenDrawer \
    OpenStandMixerHead \
    PickPlaceCounterToCabinet \
    PickPlaceCounterToStove \
    PickPlaceDrawerToCounter \
    PickPlaceSinkToCounter \
    PickPlaceToasterToCounter \
    SlideDishwasherRack \
    TurnOffStove \
    TurnOnElectricKettle \
    TurnOnMicrowave \
    TurnOnSinkFaucet \
    CloseFridge \
    CloseToasterOvenDoor \
    CoffeeSetupMug
do
    CUDA_VISIBLE_DEVICES=$1 MUJOCO_GL=egl python scripts/run_eval.py \
        --model_path "$CKPT" \
        --data_config panda_omron \
        --embodiment_tag new_embodiment \
        --task_set "$TASK" \
        --split pretrain \
        --port $((5555 + $1)) \
        --n_envs 1 \
        --n_episodes 50
done
