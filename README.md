# Robot Critics That Sweat the Small Stuff

A fork of NVIDIA Isaac GR00T adapted for running **GR00T N1.5** on the [RoboCasa](https://robocasa.ai) benchmark, with an added VLM-ranking evaluation mode that uses the simulator as a lookahead model to steer action-chunk selection.

## Contents

- [Installation](#installation)
- [Download the checkpoint](#download-the-checkpoint)
- [Environment variables](#environment-variables)
- [Basic evaluation](#basic-evaluation)
- [Evaluation with VLM ranking](#evaluation-with-vlm-ranking)
- [Aggregating results](#aggregating-results)

## Installation

You will need three source repos on disk: **this repo**, [RoboCasa 1.0.1](https://github.com/robocasa/robocasa), and [Robosuite 1.5.2](https://github.com/ARISE-Initiative/robosuite/tree/v1.5.2).

```bash
git clone https://github.com/robocasa-benchmark/Isaac-GR00T
cd Isaac-GR00T
```

Set path variables (adjust to your machine):

```bash
export GROOT_REPO=$(pwd)
export ROBOCASA_SRC=/path/to/robocasa-1.0.1
export ROBOSUITE_SRC=/path/to/robosuite-1.5.2
```

Create the conda environment:

```bash
conda create -n gr00t python=3.10 -y
conda activate gr00t
```

Install GR00T (the `[base]` extra is required — plain `pip install -e .` does **not** pull `torch==2.5.1`), then flash-attn:

```bash
cd "$GROOT_REPO"
pip install -e ".[base]"
pip install ninja
MAX_JOBS=4 pip install --no-build-isolation flash-attn==2.7.1.post4
```

Install RoboCasa and Robosuite (needed by the eval scripts). These pull in `lerobot`/`torchcodec` which clobber several of GR00T's pinned versions — we repin them in the next step:

```bash
pip install -e "$ROBOSUITE_SRC"
pip install -e "$ROBOCASA_SRC"
```

Reconcile versions so both stacks coexist:

```bash
pip install torch==2.5.1 torchvision==0.20.1        # flash-attn was built against 2.5.1
pip install --no-deps tianshou==0.5.1               # gr00t pin (robocasa wants 0.4.10)
pip install numpy==2.2.5 protobuf==3.20.3           # robocasa __init__ hard-asserts numpy==2.2.5
pip uninstall -y opencv-python                      # numpy-1 only; conflicts with numpy 2
pip install --force-reinstall --no-deps opencv-python-headless==4.11.0.86
pip uninstall -y tensorflow tensorflow-estimator tensorflow-io-gcs-filesystem keras
```

> The final `tensorflow` uninstall is important: GR00T never imports TF, but `transformers` auto-imports it if it's present, and its `ml_dtypes` is numpy-1 only, which breaks under numpy 2.

## Environment variables

Persist runtime env vars on the conda env (adjust cache paths to a filesystem with quota headroom):

```bash
conda env config vars set -n gr00t \
  HF_HOME=/your/cache/huggingface \
  XDG_CACHE_HOME=/your/cache \
  PIP_CACHE_DIR=/your/cache/pip \
  TMPDIR=/your/tmp \
  MUJOCO_GL=egl \
  NO_ALBUMENTATIONS_UPDATE=1
```

Reactivate the env after setting these:

```bash
conda deactivate && conda activate gr00t
```

## Download the checkpoint

Download the RoboCasa GR00T N1.5 checkpoint from HuggingFace:

- [`robocasa/robocasa365_checkpoints`](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/multitask_learning/checkpoint-120000)

Then export its path:

```bash
export CKPT=/path/to/checkpoint-120000
```

## Basic evaluation

Run the standard (no-VLM) GR00T eval on RoboCasa. `--task_set` accepts either a task-set registry key (`atomic_seen`, `pretrain300`, ...) or an individual task name (`TurnOnMicrowave`).

```bash
cd "$GROOT_REPO"
conda activate gr00t

CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py \
  --model_path "$CKPT" \
  --data_config panda_omron \
  --embodiment_tag new_embodiment \
  --task_set atomic_seen \
  --split pretrain \
  --n_envs 14 \
  --n_episodes 50
```

Videos and per-task `stats.json` are written to `$CKPT/evals/<split>/<TaskName>/`.

Valid `--task_set` options include:

- Any individual task (e.g. `TurnOnMicrowave`, `PickPlaceDrawerToCounter`)
- `atomic_seen` — all atomic tasks
- `pretrain300`

## Evaluation with VLM ranking

VLM ranking uses the simulator as a lookahead model: each decision cycle oversamples action chunks, rolls each candidate forward in the sim into a short mp4, ranks them with a Qwen2.5-VL critic, and executes only the winning chunk. See `scripts/run_eval_vlm_ranking.py` for details.

This runs as **two cooperating processes** in separate terminals (the VLM ranker needs a newer `transformers` than the `gr00t` env ships, and MuJoCo/EGL rendering can't share a CUDA context with the GR00T policy).

### Prerequisites

You will need:

- A separate conda env (e.g. `vlmoverlay`) with the Qwen2.5-VL ranker installed
- A VLM ranker checkpoint (`VLM_CKPT`)
- The ranker serving script (`VLM_RANK_SERVER`) — a job-inbox/`ranking.json` poll-based server
- A shared directory (`RANK_DIR`) for the inbox/done/error queues

Set paths:

```bash
export VLM_RANK_SERVER=/path/to/rank_serve_robocasa.py
export VLM_CKPT=/path/to/vlm_checkpoint
export RANK_DIR=/path/to/shared/rank/dir
export TASK=PickPlaceDrawerToCounter
export SPLIT=pretrain
```

### Terminal 1 — VLM ranker (in the `vlmoverlay` env)

```bash
conda activate vlmoverlay
CUDA_VISIBLE_DEVICES=0 python "$VLM_RANK_SERVER" \
    --inbox_dir "$RANK_DIR/inbox" \
    --done_dir  "$RANK_DIR/done" \
    --error_dir "$RANK_DIR/error" \
    --checkpoint "$VLM_CKPT" \
    --task_name "$TASK" \
    --gpu_ids 0 --batch_size 10 --max_pixels 960x540
```

### Terminal 2 — GR00T eval with VLM ranking (in the `gr00t` env, different GPU)

```bash
conda activate gr00t
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python scripts/run_eval_vlm_ranking.py \
    --model_path "$CKPT" \
    --data_config panda_omron \
    --embodiment_tag new_embodiment \
    --task "$TASK" \
    --split "$SPLIT" \
    --n_episodes 50 \
    --num-samples 5 \
    --oversample 50 \
    --no-launch-rank-server \
    --rank-dir "$RANK_DIR" \
    --vlm-checkpoint "$VLM_CKPT" \
    --port 5555 \
    --render_camera robot0_agentview_right
```

Output goes to `$CKPT/evals/$SPLIT/<TASK>_VLM_<stamp>_seed<seed>/{media,candidates,eval_log.json}`.

### Terminal 3 (optional) — apples-to-apples baseline (no VLM, same code path)

Runs the same script with `--num-samples 1 --oversample 1`, i.e. no candidate ranking:

```bash
conda activate gr00t
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl python scripts/run_eval_vlm_ranking.py \
    --model_path "$CKPT" \
    --data_config panda_omron \
    --embodiment_tag new_embodiment \
    --task "$TASK" \
    --split "$SPLIT" \
    --n_episodes 50 \
    --num-samples 1 \
    --oversample 1
```

### Single-terminal auto-launch alternative

Instead of the three-terminal setup, you can let `run_eval_vlm_ranking.py` spawn the ranker for you by pointing it at the ranker script and the python interpreter for the env the ranker runs under:

```bash
export VLM_RANK_SERVER_SCRIPT=/path/to/rank_serve_robocasa.py
export VLM_RANK_SERVER_PYTHON=/path/to/vlmoverlay/env/bin/python

MUJOCO_GL=egl python scripts/run_eval_vlm_ranking.py \
    --model_path "$CKPT" \
    --data_config panda_omron \
    --embodiment_tag new_embodiment \
    --task "$TASK" \
    --split "$SPLIT" \
    --n_episodes 50 \
    --num-samples 5 \
    --oversample 50 \
    --rank-dir "$RANK_DIR" \
    --vlm-checkpoint "$VLM_CKPT" \
    --rank-server-gpus 1        # CUDA_VISIBLE_DEVICES for the spawned ranker
```

(You can pass `--rank-server-script` / `--rank-server-python` on the command line instead of exporting the env vars.)

### Useful flags

- `--num-samples 1` — no-VLM baseline
- `--picking-strategy {best,worst,random}` — how to pick from ranked candidates
- `--n-action-steps` — actions executed and ranked per cycle
- `--max-infer-batch` — controls GR00T GPU memory usage
- `--no-launch-rank-server` — attach to an already-running ranker (as in the two-terminal setup)

### Supported tasks

The VLM-ranking pipeline has been tested on:

```
PickPlaceCounterToCabinet   PickPlaceCounterToStove   PickPlaceDrawerToCounter
PickPlaceSinkToCounter      PickPlaceToasterToCounter SlideDishwasherRack
TurnOffStove                TurnOnElectricKettle      TurnOnMicrowave
TurnOnSinkFaucet
```

## Aggregating results

After a run, aggregate per-task stats into a summary:

```bash
python gr00t/eval/get_eval_stats.py --dir "$CKPT"
```
