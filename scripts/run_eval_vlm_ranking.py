# SPDX-License-Identifier: Apache-2.0
"""
VLM-ranked GR00T rollout in RoboCasa, using the SIMULATOR as the lookahead model.

This is the GR00T analogue of
    robocasa_diffusion_policy/run_diffusion_policy_robocasa_vlm_ranking.py
It adds VLM steering to the plain GR00T eval (scripts/run_eval.py) using *exactly* the same
recipe the diffusion-policy script uses -- the only difference is where the action chunks come
from (a GR00T inference server instead of an in-process diffusion policy):

    1. Capture one observation; oversample action chunks; prune to --num-samples diverse candidates.
    2. SAVE the full sim state s_t.
    3. For each candidate, restore s_t, roll the chunk forward in the sim, render the executed
       sub-steps into a short mp4 (a real video of s_t -> s_{t+1}).
    4. Restore s_t.
    5. Rank the N candidate mp4s with the local Qwen2.5-VL ranker (trl/.../rank_serve_robocasa.py).
    6. Restore s_t and execute ONLY the winning chunk for real, recording the rollout video.
    7. Loop until success / done / max_steps.

Runs --n_episodes sequential rollouts on a single env (reset between each, scene seed
--test_start_seed + i, matching the plain eval), and reports an aggregate success rate. The
per-decision lookahead is the only "parallelism" (over --num-samples candidates).

ARCHITECTURE (two cooperating processes, like run_eval.py):
  * The GR00T policy runs in a SEPARATE spawn subprocess (RobotInferenceServer over ZMQ). This
    keeps THIS (client) process -- which builds the MuJoCo/EGL env and renders many candidate
    videos -- free of a torch CUDA context, avoiding the documented "OpenGL error in Mujoco"
    when EGL and CUDA share a process (see the comment at the bottom of run_eval.py).
  * The VLM ranker needs a newer transformers than the gr00t env ships, so it ALSO runs out of
    process: rank_serve_robocasa.py under the vlmoverlay env, driven via a local job inbox +
    poll-for-ranking.json protocol. By default we auto-launch it; pass --no-launch-rank-server
    to attach to a server you started yourself in another terminal.

Example (two terminals, mirroring the diffusion-policy workflow):

  # Terminal 1: VLM ranker (in the vlmoverlay env: newer transformers than gr00t)
  RANK_DIR=/path/to/rc_rank/PickPlaceCounterToDrawer
  VLM_CKPT=/path/to/vlm_checkpoint
  VLM_RANK_SERVER=/path/to/rank_serve_robocasa.py
  CUDA_VISIBLE_DEVICES=1 python "$VLM_RANK_SERVER" \
      --inbox_dir $RANK_DIR/inbox --done_dir $RANK_DIR/done --error_dir $RANK_DIR/error \
      --checkpoint "$VLM_CKPT" --task_name PickPlaceCounterToDrawer \
      --gpu_ids 0 --batch_size 10 --max_pixels 960x540

  # Terminal 2: GR00T eval with VLM ranking (gr00t env)
  CKPT=/path/to/gr00t/checkpoint-120000
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python scripts/run_eval_vlm_ranking.py \
      --model_path "$CKPT" --data_config panda_omron --embodiment_tag new_embodiment \
      --task PickPlaceCounterToDrawer --split pretrain \
      --no-launch-rank-server --rank-dir $RANK_DIR --vlm-checkpoint $VLM_CKPT

Or auto-launch the ranker in-process (single terminal) by pointing at the ranker script and
the python interpreter for the env it runs under:

  MUJOCO_GL=egl python scripts/run_eval_vlm_ranking.py \
      --model_path "$CKPT" --data_config panda_omron --embodiment_tag new_embodiment \
      --task PickPlaceCounterToDrawer --split pretrain \
      --rank-dir $RANK_DIR --vlm-checkpoint $VLM_CKPT \
      --rank-server-script "$VLM_RANK_SERVER" \
      --rank-server-python /path/to/vlmoverlay/env/bin/python \
      --rank-server-gpus 1
"""
import sys
# line-buffer stdout/stderr so logs interleave correctly with subprocess output
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import argparse
import json
import os
import pathlib
import subprocess
import time
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# VLM rank server (ported verbatim from the diffusion-policy script -- the
# inbox/ranking.json protocol is policy-agnostic). The ranker script + its
# python interpreter live in a separate env, so both are supplied via CLI
# (or the VLM_RANK_SERVER_SCRIPT / VLM_RANK_SERVER_PYTHON env vars).
# ---------------------------------------------------------------------------
def launch_rank_server(server_python, server_script, checkpoint, task_name, gpus, max_pixels,
                       batch_size, inbox, done, error, log_path):
    """Spawn the ranker script (`server_script`) under `server_python`."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    n_gpu = max(len([g for g in gpus.split(",") if g.strip()]), 1)
    cmd = [server_python, server_script,
           "--inbox_dir", str(inbox), "--done_dir", str(done), "--error_dir", str(error),
           "--checkpoint", checkpoint, "--task_name", task_name,
           "--gpu_ids", ",".join(str(i) for i in range(n_gpu)),
           "--batch_size", str(batch_size), "--max_pixels", max_pixels]
    logf = open(log_path, "w")
    print(f"Launching rank server (CUDA_VISIBLE_DEVICES={gpus}): {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return proc, logf


def wait_for_server_ready(proc, log_path, timeout):
    """Block until the rank server logs its 'ready. watching' line (model loaded)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"rank server exited early (code {proc.returncode}); see {log_path}")
        try:
            if "ready. watching" in pathlib.Path(log_path).read_text():
                return
        except FileNotFoundError:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"rank server not ready within {timeout}s; see {log_path}")


def submit_rank_job(inbox, name, output_subdir, video_paths, task_name):
    """Write an atomic (tmp -> rename) job json into the rank server's inbox."""
    job = {"name": name, "output_subdir": str(output_subdir),
           "video_paths": [str(v) for v in video_paths], "task_name": task_name}
    inbox = pathlib.Path(inbox)
    tmp = inbox / f"{name}.json.tmp"
    with open(tmp, "w") as f:
        json.dump(job, f)
    os.replace(tmp, inbox / f"{name}.json")


def wait_for_ranking(output_subdir, poll, timeout, proc=None, error_dir=None, job_name=None):
    """Block until ranking.json appears in output_subdir; return the parsed dict.

    Also fail fast (instead of polling for the full timeout) if the rank server moves this job
    into error_dir, or if an auto-launched server process dies.
    """
    rp = pathlib.Path(output_subdir) / "ranking.json"
    err_job = (pathlib.Path(error_dir) / f"{job_name}.json") if (error_dir and job_name) else None
    deadline = time.time() + timeout
    while time.time() < deadline:
        if rp.exists():
            try:
                with open(rp) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass  # half-written; retry
        if err_job is not None and err_job.exists():
            raise RuntimeError(f"rank server failed job {job_name} (moved to {err_job}); "
                               f"see the rank server log")
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"rank server died (code {proc.returncode}) while waiting for {rp}")
        time.sleep(poll)
    raise TimeoutError(f"ranking timeout after {timeout}s waiting for {rp}")


def select_diverse_indices(actions_exec, k, terminal_weight=4.0, n_terminal=2,
                           joint_weights=None, seed_idx=0):
    """Greedy farthest-point sampling over terminal-weighted flattened L2 (ported verbatim)."""
    M, T, D = actions_exec.shape
    assert 1 <= k <= M, f"k={k} must be in [1, M={M}]"
    if joint_weights is None:
        joint_weights = np.ones(D, dtype=actions_exec.dtype)
    w_t = np.ones(T, dtype=actions_exec.dtype)
    w_t[-n_terminal:] = terminal_weight
    w = (np.sqrt(w_t)[:, None] * np.sqrt(joint_weights.astype(actions_exec.dtype))[None, :])
    flat = (actions_exec * w[None, :, :]).reshape(M, -1)
    selected = [seed_idx]
    min_d = np.linalg.norm(flat - flat[seed_idx], axis=1)
    min_d[seed_idx] = -np.inf
    for _ in range(k - 1):
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        d_new = np.linalg.norm(flat - flat[nxt], axis=1)
        min_d = np.minimum(min_d, d_new)
        min_d[nxt] = -np.inf
    return selected


# ---------------------------------------------------------------------------
# GR00T policy inference server (runs in a SEPARATE spawn process; copied from
# run_eval.py so this script is self-contained).
# ---------------------------------------------------------------------------
def run_server(data_config, model_path, embodiment_tag, port, seed=None):
    import numpy as _np
    import torch
    from gr00t.eval.robot import RobotInferenceServer
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        _np.random.seed(seed)

    dc = DATA_CONFIG_MAP[data_config]
    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=dc.modality_config(),
        modality_transform=dc.transform(),
        embodiment_tag=embodiment_tag,
        denoising_steps=4,
    )
    RobotInferenceServer(policy, port=port).run()


# ---------------------------------------------------------------------------
# Single-env rollout client with sim-state lookahead.
# ---------------------------------------------------------------------------
def build_obs_batch(latest_obs, batch):
    """Tile a single env observation to (batch, T=1, ...), the format the server expects.

    The plain GR00T eval uses single-frame obs (MultiStepConfig delta_indices default to [0]),
    so each modality gets a leading time axis of length 1. video.* stays uint8, state.* float32,
    and annotation is broadcast to a (batch,) array of strings -- matching what the vectorized
    eval sends with n_envs == batch.
    """
    out = {}
    for k, v in latest_obs.items():
        if k.startswith("video"):
            frame = np.asarray(v)[None]                       # (1, H, W, C)
            out[k] = np.broadcast_to(frame[None], (batch,) + frame.shape).copy()
        elif k.startswith("state"):
            s = np.asarray(v, dtype=np.float32)[None]         # (1, D)
            out[k] = np.broadcast_to(s[None], (batch,) + s.shape).copy()
        elif k.startswith("annotation"):
            out[k] = np.array([v] * batch)
    return out


def oversample_chunks(client, latest_obs, n, max_bs):
    """Query the server for `n` candidate action chunks for the same obs.

    GR00T's flow-matching head draws fresh noise per forward pass, so tiling the obs to a batch
    yields diverse candidates (the analogue of the diffusion policy oversampling in one batched
    forward). We split into sub-batches of <= max_bs so large --oversample never OOMs the model.
    Returns a dict {action.<key>: np.ndarray (n, Hc, dim)}.
    """
    collected = None
    got = 0
    while got < n:
        b = min(max_bs, n - got)
        act = client.get_action(build_obs_batch(latest_obs, b))
        act = {k: np.asarray(v) for k, v in act.items() if k.startswith("action")}
        if collected is None:
            collected = {k: [] for k in act}
        for k in collected:
            collected[k].append(act[k])
        got += b
    return {k: np.concatenate(v, axis=0) for k, v in collected.items()}


def parse_args():
    p = argparse.ArgumentParser(
        description="VLM-ranked GR00T rollout in RoboCasa (simulator lookahead).")
    # --- policy / task (shared names with run_eval.py) ---
    p.add_argument("--model_path", required=True, help="GR00T checkpoint directory.")
    p.add_argument("--data_config", default="panda_omron")
    p.add_argument("--embodiment_tag", default="new_embodiment")
    p.add_argument("-t", "--task", required=True, help="A single RoboCasa task/env name.")
    p.add_argument("--split", required=True, choices=["pretrain", "target"])
    p.add_argument("--n_episodes", type=int, default=50,
                   help="Sequential rollouts (single env, reset between each).")
    p.add_argument("--test_start_seed", type=int, default=100000,
                   help="Base scene seed; rollout i uses test_start_seed + i.")
    p.add_argument("--n-action-steps", type=int, default=32,
                   help="Actions executed (and rendered) per cycle, capped to the chunk length.")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Step budget per rollout (default: 1.5x task horizon).")
    p.add_argument("--seed", type=int, default=None,
                   help="Per-run seed for the policy noise + output dir name (not the scenes).")
    p.add_argument("--port", type=int, default=5555, help="GR00T inference server port.")
    p.add_argument("--video_dir", default=None, help="Output base dir (default: <model_path>).")
    # --- rendering ---
    p.add_argument("--render_camera", default="robot0_agentview_right")
    p.add_argument("--render_size", default="848x480", help="WxH for saved videos (even dims).")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--crf", type=int, default=22)
    # --- candidate sampling / picking (ported from the diffusion script) ---
    p.add_argument("--num-samples", type=int, default=5,
                   help="Candidate chunks ranked by the VLM each cycle (1 = no-VLM baseline).")
    p.add_argument("--oversample", type=int, default=20,
                   help="Sample this many chunks per cycle, prune to --num-samples via FPS. "
                        "0 disables oversampling (== num-samples).")
    p.add_argument("--max-infer-batch", type=int, default=10,
                   help="Max candidates per server forward pass (guards GR00T GPU memory).")
    p.add_argument("--picking-strategy", default="best", choices=["best", "worst", "random"])
    # --- VLM ranker (out-of-process via rank_serve_robocasa.py) ---
    p.add_argument("--vlm-checkpoint", default=None,
                   help="Qwen2.5-VL ranker checkpoint. Required unless --no-launch-rank-server.")
    p.add_argument("--vlm-task-name", default=None, help="Ranker prompt task (default: --task).")
    p.add_argument("--vlm-max-pixels", default="960x540")
    p.add_argument("--vlm-batch-size", type=int, default=4)
    # --- rank server lifecycle ---
    p.add_argument("--rank-server-python",
                   default=os.environ.get("VLM_RANK_SERVER_PYTHON"),
                   help="Python interpreter to launch the rank server with "
                        "(env: VLM_RANK_SERVER_PYTHON). Required unless --no-launch-rank-server.")
    p.add_argument("--rank-server-script",
                   default=os.environ.get("VLM_RANK_SERVER_SCRIPT"),
                   help="Path to the ranker script, e.g. rank_serve_robocasa.py "
                        "(env: VLM_RANK_SERVER_SCRIPT). Required unless --no-launch-rank-server.")
    p.add_argument("--rank-server-gpus", default="0",
                   help="CUDA_VISIBLE_DEVICES for an auto-launched rank server.")
    p.add_argument("--no-launch-rank-server", action="store_true",
                   help="Attach to a rank server already watching --rank-dir/inbox.")
    p.add_argument("--rank-dir", default=None, help="Base dir for inbox/done/error.")
    p.add_argument("--rank-poll-sec", type=float, default=2.0)
    p.add_argument("--rank-timeout-sec", type=float, default=1800.0)
    p.add_argument("--rank-ready-timeout-sec", type=float, default=900.0)
    return p.parse_args()


def main():
    import gymnasium as gym
    import robocasa  # noqa: F401  (registers robocasa/* gym envs)
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from gr00t.eval.simulation import SimulationInferenceClient
    from gr00t.eval.wrappers.video_recording_wrapper import VideoRecorder

    os.environ.setdefault("MUJOCO_GL", "egl")
    args = parse_args()

    if args.num_samples < 1:
        sys.exit("--num-samples must be >= 1")
    oversample = args.oversample if args.oversample != 0 else args.num_samples
    if oversample < args.num_samples:
        sys.exit(f"--oversample ({oversample}) must be >= --num-samples ({args.num_samples})")
    try:
        render_w, render_h = (int(x) for x in args.render_size.lower().split("x"))
    except ValueError:
        sys.exit(f"--render_size must be WxH, got: {args.render_size}")
    if render_w % 2 or render_h % 2:
        sys.exit(f"--render_size dims must be even for h264, got: {args.render_size}")

    vlm_task_name = args.vlm_task_name or args.task
    use_vlm = args.num_samples >= 2  # --num-samples 1 is a no-VLM baseline
    if use_vlm and not args.no_launch_rank_server:
        if not args.vlm_checkpoint:
            sys.exit("--vlm-checkpoint is required unless --no-launch-rank-server is set.")
        if not os.path.exists(args.vlm_checkpoint):
            sys.exit(f"VLM checkpoint not found: {args.vlm_checkpoint}")
        if not args.rank_server_python:
            sys.exit("--rank-server-python (or env VLM_RANK_SERVER_PYTHON) is required "
                     "unless --no-launch-rank-server is set.")
        if not args.rank_server_script:
            sys.exit("--rank-server-script (or env VLM_RANK_SERVER_SCRIPT) is required "
                     "unless --no-launch-rank-server is set.")
        if not os.path.exists(args.rank_server_python):
            sys.exit(f"--rank-server-python not found: {args.rank_server_python}")
        if not os.path.exists(args.rank_server_script):
            sys.exit(f"--rank-server-script not found: {args.rank_server_script}")

    seed = args.seed if args.seed is not None else (
        int(np.random.SeedSequence().generate_state(1)[0]) & 0x7FFFFFFF)
    np.random.seed(seed)
    print(f"Per-run seed (policy noise + dir name): {seed}")

    # ---- 1. Launch the GR00T policy server in a separate spawn process ----
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    server_proc = ctx.Process(
        target=run_server,
        args=(args.data_config, args.model_path, args.embodiment_tag, args.port, seed),
        daemon=True,
    )
    server_proc.start()
    client = SimulationInferenceClient(host="localhost", port=args.port)
    print("Waiting for GR00T inference server ...")
    t0 = time.time()
    while not client.ping():
        if not server_proc.is_alive():
            sys.exit("GR00T server process died during startup.")
        if time.time() - t0 > args.rank_ready_timeout_sec:
            sys.exit("GR00T server did not become ready in time.")
        time.sleep(2.0)
    print("GR00T server ready.")

    # ---- 2. Output dir (mirrors run_eval.py's layout) ----
    base = args.video_dir or args.model_path
    run_stamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    run_dir = pathlib.Path(base) / "evals" / args.split / \
        f"{args.task}_VLM_{run_stamp}_seed{seed}"
    media_dir = run_dir / "media"
    cand_dir = run_dir / "candidates"
    media_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ---- 3. Rank server (out-of-process) ----
    rank_base = pathlib.Path(args.rank_dir) if args.rank_dir else (run_dir / "rank")
    rank_inbox, rank_done, rank_error = rank_base / "inbox", rank_base / "done", rank_base / "error"
    for d in (rank_inbox, rank_done, rank_error):
        d.mkdir(parents=True, exist_ok=True)
    rank_proc = None
    if not use_vlm:
        print("--num-samples=1: no-VLM baseline (no rank server, no candidate rollouts).")
    elif args.no_launch_rank_server:
        print(f"Using external rank server watching {rank_inbox}")
    else:
        rank_log = run_dir / "rank_server.log"
        rank_proc, _ = launch_rank_server(
            args.rank_server_python, args.rank_server_script, args.vlm_checkpoint, vlm_task_name,
            args.rank_server_gpus, args.vlm_max_pixels, args.vlm_batch_size,
            rank_inbox, rank_done, rank_error, rank_log)
        print(f"Waiting for rank server to load the model (log: {rank_log}) ...")
        wait_for_server_ready(rank_proc, rank_log, args.rank_ready_timeout_sec)
        print("Rank server ready.")

    # ---- 4. Build a single env (unwrapped RoboCasaGymEnv: no vector/multistep wrappers) ----
    env = gym.make(
        f"robocasa/{args.task}", split=args.split, enable_render=True,
        render_camera=args.render_camera, render_width=render_w, render_height=render_h,
    ).unwrapped
    rs_env = env.env  # underlying robosuite MujocoEnv: holds .sim and the episode counters

    try:
        control_freq = float(rs_env.control_freq)
    except Exception:
        control_freq = 20.0
    steps_per_render = max(round(control_freq / args.fps), 1)
    horizon = get_task_horizon(args.task)
    max_steps = int(args.max_steps) if args.max_steps is not None else int(horizon * 1.5)
    print(f"control_freq={control_freq}Hz, steps_per_render={steps_per_render}, "
          f"max_steps={max_steps}")

    action_keys = None  # fixed (sorted) order, bound on the first server response

    def save_state():
        return {"sim": rs_env.sim.get_state().flatten(), "timestep": rs_env.timestep,
                "cur_time": rs_env.cur_time, "done": rs_env.done}

    def restore_state(s):
        # Restoring only MuJoCo state is NOT enough: robosuite's step() increments
        # timestep/cur_time and raises if self.done is True, so reset those too.
        rs_env.sim.set_state_from_flattened(s["sim"])
        rs_env.sim.forward()
        rs_env.timestep = s["timestep"]
        rs_env.cur_time = s["cur_time"]
        rs_env.done = s["done"]

    def step_chunk(chunk, n_exec, recorder=None, base_step=0):
        """Execute n_exec actions of `chunk` (dict key->(Hc,dim)); optionally record frames.

        Returns (n_stepped, reached_success, reached_done, succ_step, last_obs).
        """
        reached_success = reached_done = False
        succ_step = None
        last_obs = None
        n = 0
        for t in range(n_exec):
            act = {k: chunk[k][t] for k in action_keys}
            obs, reward, done, _, info = env.step(act)
            last_obs = obs
            n += 1
            if recorder is not None and (base_step + n) % steps_per_render == 0:
                recorder.write_frame(env.render())
            if float(reward) > 0 or bool(info.get("success", False)):
                reached_success = True
                succ_step = n
            if bool(done):
                reached_done = True
            if reached_success or reached_done:
                break
        return n, reached_success, reached_done, succ_step, last_obs

    def render_candidate(chunk, n_exec, vpath):
        """Roll one candidate chunk forward from the current sim state and write its lookahead mp4.

        Unlike step_chunk (the real-rollout executor), this ALWAYS writes an initial and a final
        frame -- so the file always exists and the ranker's last-frame comparison has a valid frame
        even when the chunk terminates after only a step or two (e.g. near the task horizon, where a
        cadence-only recorder could otherwise write zero frames and PyAV would create no file at all).
        It only stops on `done` (robosuite forbids stepping a terminated env); success does NOT stop
        it (success is sparse, done is horizon-based), so the VLM sees the full candidate motion.
        """
        rec = VideoRecorder.create_h264(fps=args.fps, codec="h264",
                                        input_pix_fmt="rgb24", crf=args.crf)
        rec.start(vpath)
        rec.write_frame(env.render())  # initial frame -> guarantees a non-empty file
        for t in range(n_exec):
            act = {k: chunk[k][t] for k in action_keys}
            _, _, done, _, _ = env.step(act)
            if (t + 1) % steps_per_render == 0:
                rec.write_frame(env.render())
            if bool(done):
                break
        rec.write_frame(env.render())  # final frame -> the decisive last state is always captured
        rec.stop()

    def run_one_cycle(scene_seed, r_cand_dir, cycle, latest_obs, rollout_rec, base_step):
        """observe -> oversample -> render candidates -> VLM rank -> execute winner."""
        nonlocal action_keys
        # Oversample candidate chunks from the server.
        act = oversample_chunks(client, latest_obs, oversample, args.max_infer_batch)
        if action_keys is None:
            action_keys = sorted(act.keys())
        Hc = act[action_keys[0]].shape[1]
        n_exec = min(args.n_action_steps, Hc)
        actions_all = np.concatenate([act[k] for k in action_keys], axis=-1)  # (M, Hc, D)

        # Prune to num_samples diverse candidates over the executed window.
        keep = list(range(oversample))
        if oversample > args.num_samples:
            keep = select_diverse_indices(actions_all[:, :n_exec], k=args.num_samples,
                                          terminal_weight=4.0, n_terminal=2, seed_idx=0)
        cands = [{k: act[k][i] for k in action_keys} for i in keep]  # each value (Hc, dim)

        s_t = save_state()
        cycle_subdir = r_cand_dir / f"cycle{cycle:06d}"
        cycle_subdir.mkdir(parents=True, exist_ok=True)

        if len(cands) < 2:
            # no-VLM baseline: nothing to rank, just execute the single chunk.
            winner_idx, votes = 0, None
        else:
            # Render each candidate (named <i>.mp4 so winner_idx maps back to the candidate).
            cand_paths = []
            for i, chunk in enumerate(cands):
                restore_state(s_t)
                vpath = str(cycle_subdir / f"{i}.mp4")
                render_candidate(chunk, n_exec, vpath)
                cand_paths.append(vpath)
            restore_state(s_t)
            if not np.allclose(rs_env.sim.get_state().flatten(), s_t["sim"]):
                print(f"  cycle {cycle}: WARNING sim state not exactly restored")

            # Include the per-run seed so concurrent clients sharing one rank inbox (e.g. one
            # multitask ranker serving several tasks) never collide on a job name -- scene seeds
            # (test_start_seed + i) are identical across runs, so run_stamp + scene_seed alone is
            # not unique if two clients launch in the same second.
            job_name = f"{run_stamp}_run{seed}_seed{scene_seed}_{cycle:06d}"
            submit_rank_job(rank_inbox, job_name, cycle_subdir, cand_paths, vlm_task_name)
            result = wait_for_ranking(cycle_subdir, args.rank_poll_sec, args.rank_timeout_sec,
                                      proc=rank_proc, error_dir=rank_error, job_name=job_name)
            votes = result["votes"]
            if args.picking_strategy == "best":
                winner_idx = int(result["winner_idx"])
            elif args.picking_strategy == "worst":
                winner_idx = int(np.argmin(votes))
            else:
                winner_idx = int(np.random.randint(len(cand_paths)))
            with open(cycle_subdir / "ranking.json", "w") as f:
                json.dump({**result, "picking_strategy": args.picking_strategy,
                           "chosen_idx": winner_idx}, f, indent=2)

        # Restore s_t and execute ONLY the winning chunk for real.
        restore_state(s_t)
        n, succ, done, succ_step, last_obs = step_chunk(
            cands[winner_idx], n_exec, recorder=rollout_rec, base_step=base_step)
        return winner_idx, votes, keep, n, succ, done, succ_step, last_obs

    # ---- 5. Rollout loop (single env, reset between rollouts) ----
    episode_records = []
    interrupted = False
    try:
        for rollout_idx in range(args.n_episodes):
            scene_seed = args.test_start_seed + rollout_idx
            latest_obs, _ = env.reset(seed=scene_seed)

            r_cand_dir = cand_dir / f"seed{scene_seed}"
            rollout_path = str(media_dir / f"seed{scene_seed}.mp4")
            rollout_rec = VideoRecorder.create_h264(fps=args.fps, codec="h264",
                                                    input_pix_fmt="rgb24", crf=args.crf)
            rollout_rec.start(rollout_path)
            print(f"[rollout {rollout_idx + 1}/{args.n_episodes}] scene seed {scene_seed}")

            cycle = total_steps = 0
            success = False
            success_step = None
            stop_reason = "max_steps"
            try:
                while total_steps < max_steps:
                    (winner_idx, votes, keep, n, succ, done, succ_step,
                     latest_obs) = run_one_cycle(scene_seed, r_cand_dir, cycle,
                                                 latest_obs, rollout_rec, total_steps)
                    total_steps += n
                    if succ and not success:
                        success = True
                        success_step = total_steps - n + succ_step
                    print(f"  cycle {cycle}: votes={votes} -> {args.picking_strategy} "
                          f"winner_idx={winner_idx} (steps {total_steps}/{max_steps})")
                    cycle += 1
                    if success:
                        stop_reason = "success"
                        break
                    if done:
                        stop_reason = "env_done"
                        break
            except KeyboardInterrupt:
                stop_reason = "keyboard_interrupt"
                interrupted = True
            except Exception as e:
                import traceback
                stop_reason = f"exception:{type(e).__name__}"
                print(f"  rollout {rollout_idx} crashed: {e!r}")
                traceback.print_exc()
            finally:
                rollout_rec.stop()

            episode_records.append({
                "episode": rollout_idx, "seed": scene_seed, "success": bool(success),
                "reward": 1.0 if success else 0.0, "success_step": success_step,
                "success_time_sec": (round(success_step / control_freq, 2)
                                     if success_step is not None else None),
                "total_steps": total_steps, "cycles": cycle, "stop_reason": stop_reason,
                "video_path": rollout_path,
            })
            print(f"[rollout {rollout_idx + 1}/{args.n_episodes}] seed {scene_seed}: "
                  f"success={success} (step {success_step}), steps={total_steps}, "
                  f"stop_reason={stop_reason}")
            if interrupted:
                print("Interrupted; stopping remaining rollouts.")
                break
    finally:
        if rank_proc is not None and rank_proc.poll() is None:
            print("Shutting down rank server.")
            rank_proc.terminate()
            try:
                rank_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                rank_proc.kill()
        try:
            env.close()
        except Exception as e:
            print(f"env close failed: {e!r}")
        server_proc.terminate()
        server_proc.join(timeout=10)

    # ---- 6. Finalize (mirror run_eval.py's stats.json + eval_log.json) ----
    n_done = len(episode_records)
    n_succ = sum(r["success"] for r in episode_records)
    success_rate = (n_succ / n_done) if n_done else 0.0
    with open(run_dir / "stats.json", "w") as f:
        json.dump({"num_episodes": n_done, "success_rate": success_rate}, f, indent=4)

    eval_log = {
        "eval_args": {
            "checkpoint": args.model_path, "command": " ".join(sys.argv),
            "max_steps": max_steps, "n_action_steps": args.n_action_steps,
            "num_rollouts": args.n_episodes, "render_camera": args.render_camera,
            "render_size": args.render_size, "seed": seed, "split": args.split,
            "task": args.task, "test_start_seed": args.test_start_seed,
            "vlm_checkpoint": args.vlm_checkpoint, "vlm_task_name": vlm_task_name,
            "picking_strategy": args.picking_strategy, "num_samples": args.num_samples,
            "oversample": oversample, "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        f"success_rate/{args.task}": success_rate, "test/mean_score": success_rate,
    }
    for r in episode_records:
        eval_log[f"test/sim_max_reward_{r['seed']}"] = float(r["reward"])
        eval_log[f"test/sim_video_{r['seed']}"] = r["video_path"]
        eval_log[f"test/success_step_{r['seed']}"] = r["success_step"]
        eval_log[f"test/success_time_sec_{r['seed']}"] = r["success_time_sec"]
    eval_log["rollouts"] = episode_records
    with open(run_dir / "eval_log.json", "w") as f:
        json.dump(eval_log, f, indent=4)

    print(f"\nDone. success_rate={success_rate:.3f} ({n_succ}/{n_done} rollouts).")
    print(f"Log: {run_dir / 'eval_log.json'}")


if __name__ == "__main__":
    main()
