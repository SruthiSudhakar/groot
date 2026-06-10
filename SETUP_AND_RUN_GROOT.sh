#!/usr/bin/env bash
# Setup + inference for GR00T N1.5 on the RoboCasa benchmark.
# SEPARATE from the diffusion_policy repo / robocasa_dp env.
# The `gr00t` conda env was already built by this procedure on 2026-06-08 and works.
# This file documents exactly what was done (incl. the dependency-conflict fixes).

# ===========================================================================
# PATHS
# ===========================================================================
GROOT_REPO=/proj/vondrick3/sruthi/Appaji/Isaac-GR00T
ROBOCASA_SRC=/proj/vondrick3/sruthi/Appaji/robocasa_new        # robocasa 1.0.1 (shared editable src; requires numpy==2.2.5)
ROBOSUITE_SRC=/proj/vondrick3/sruthi/Appaji/robosuite_new      # robosuite 1.5.2
CKPT=/proj/vondrick3/sruthi/Appaji/released_checkpoints_groot/gr00t_n1-5/multitask_learning/checkpoint-120000

# /home is quota-limited -> redirect ALL caches/tmp to /proj or installs fail with
# "OSError: [Errno 122] Disk quota exceeded".
export PIP_CACHE_DIR=/proj/vondrick3/sruthi/.cache/pip
export TMPDIR=/proj/vondrick3/sruthi/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" /proj/vondrick3/sruthi/.cache/huggingface

PIP=/proj/vondrick3/sruthi/miniconda3/envs/gr00t/bin/pip   # call pip by abs path (activation not needed)

# ===========================================================================
# 1. Env (Python 3.10)
# ===========================================================================
conda create -n gr00t python=3.10 -y

# ===========================================================================
# 2. GR00T with the [base] extra ([base] is what pulls torch 2.5.1 — plain
#    `pip install -e .` does NOT). Then flash-attn, compiled against that torch.
# ===========================================================================
cd "$GROOT_REPO"
$PIP install -e ".[base]"
$PIP install ninja
MAX_JOBS=4 $PIP install --no-build-isolation flash-attn==2.7.1.post4

# ===========================================================================
# 3. RoboCasa sim deps (run_eval.py imports robocasa/robosuite; NOT gr00t deps).
#    WARNING: robocasa pulls lerobot+torchcodec, which CLOBBER gr00t's pins
#    (torch->2.7.1, numpy->2.2.5 ok, tianshou->0.4.10, protobuf->3.19.6, etc).
#    Step 4 puts the gr00t-critical pins back.
# ===========================================================================
$PIP install -e "$ROBOSUITE_SRC"
$PIP install -e "$ROBOCASA_SRC"

# ===========================================================================
# 4. RECONCILE versions (the part that makes both stacks coexist):
#    - torch 2.5.1 (flash-attn was built for it; lerobot/torchcodec bumped it to 2.7.1)
#    - numpy 2.2.5 (robocasa __init__ HARD-asserts ==2.2.5; matches working robocasa_dp)
#    - protobuf 3.20.3 (transformers needs `builder`, added in 3.20; robocasa pulled 3.19.6)
#    - tianshou 0.5.1 (gr00t pin; robocasa wants 0.4.10 but eval uses 0.5.1)
#    - drop opencv-python 4.8 (numpy-1 only); keep opencv-python-headless 4.11
#    - uninstall tensorflow stack: gr00t never imports TF, but `transformers`
#      auto-imports it if present, and its ml_dtypes is numpy-1 -> breaks under numpy 2.
# ===========================================================================
$PIP install torch==2.5.1 torchvision==0.20.1     # restores cu124 nvidia libs + triton 3.1.0
$PIP install --no-deps tianshou==0.5.1
$PIP install numpy==2.2.5 protobuf==3.20.3
$PIP uninstall -y opencv-python
$PIP install --force-reinstall --no-deps opencv-python-headless==4.11.0.86   # repair shared cv2/ dir
$PIP uninstall -y tensorflow tensorflow-estimator tensorflow-io-gcs-filesystem keras

# ===========================================================================
# 5. Persist runtime env vars on the conda env (so caches stay off /home,
#    MuJoCo uses EGL headless, albumentations stops phoning home).
# ===========================================================================
conda env config vars set -n gr00t \
  HF_HOME=/proj/vondrick3/sruthi/.cache/huggingface \
  XDG_CACHE_HOME=/proj/vondrick3/sruthi/.cache \
  PIP_CACHE_DIR=/proj/vondrick3/sruthi/.cache/pip \
  TMPDIR=/proj/vondrick3/sruthi/tmp \
  MUJOCO_GL=egl \
  NO_ALBUMENTATIONS_UPDATE=1

# ===========================================================================
# 6. RUN INFERENCE.  server+client run in one process (threads) by default.
#    --task_set accepts a registry key (atomic_seen, pretrain300, ...) OR, thanks
#    to a local patch in scripts/run_eval.py, an individual task name (TurnOnMicrowave).
#    Videos + per-task stats.json -> $CKPT/evals/<split>/<TaskName>/
# ===========================================================================
cd /proj/vondrick3/sruthi/Appaji/Isaac-GR00T
conda activate gr00t
CKPT=/proj/vondrick3/sruthi/Appaji/released_checkpoints_groot/gr00t_n1-5/multitask_learning/checkpoint-120000
CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py \
  --model_path "$CKPT" \
  --data_config panda_omron \
  --embodiment_tag new_embodiment \
  --task_set atomic_seen \
  --split pretrain \
  --n_envs 14 \
  --n_episodes 50

task-set options: 
- TurnOnMicrowave (or any other task in the task set)
- atomic_seen (eval all atomic tasks)
- pretrain300
# ===========================================================================
# 7. Aggregate
# ===========================================================================
python gr00t/eval/get_eval_stats.py --dir "$CKPT"
