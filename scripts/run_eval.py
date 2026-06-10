# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import json
import numpy as np
import threading
import time
from datetime import datetime
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon

from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


def run_server(data_config, model_path, embodiment_tag, port, seed=None):
    # Seed the policy's sampling noise so rollouts are reproducible run-to-run
    # (env scenes are already pinned via per-episode env seeds on the client).
    if seed is not None:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    # Create a policy
    data_config = DATA_CONFIG_MAP[data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=embodiment_tag,
        denoising_steps=4,
    )

    # Start the server
    server = RobotInferenceServer(policy, port=port)
    server.run()


def run_client(
    host,
    port,
    task_set_list,
    video_dir,
    split,
    n_episodes,
    n_envs,
    n_action_steps,
    model_path=None,
    seed=0,
    test_start_seed=100000,
    render_camera=None,
    render_width=None,
    render_height=None,
    max_steps=None,
):
    # Create a simulation client
    simulation_client = SimulationInferenceClient(host=host, port=port)

    print("Available modality configs:")
    modality_config = simulation_client.get_modality_config()
    print(modality_config.keys())
    
    all_env_names = []
    for task_set in task_set_list:
        # Accept either a task-set registry key (expands to many tasks) or an
        # individual task name (e.g. TurnOnMicrowave), mirroring the diffusion
        # policy wrapper so single tasks can be eval'd directly.
        if task_set in TASK_SET_REGISTRY:
            all_env_names += TASK_SET_REGISTRY[task_set]
        else:
            all_env_names.append(task_set)
    # turn into unique list
    all_env_names = set(all_env_names)

    for env_name in all_env_names:
        # Stamp each run with a datetime + seed so repeated runs land in their
        # own dir (matching the diffusion policy layout) instead of overwriting
        # or being skipped: <task>_<n_envs>_<n_episodes>_<datetime>_seed<seed>.
        run_stamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        run_dir_name = f"{env_name}_{n_envs}_{n_episodes}_{run_stamp}_seed{seed}"
        this_video_dir = os.path.join(video_dir, "evals", split, run_dir_name)
        os.makedirs(this_video_dir, exist_ok=True)

        horizon = get_task_horizon(env_name)
        # Step budget per rollout: default is 1.5x the task horizon to match the
        # diffusion policy eval (robocasa_diffusion_policy/eval_robocasa.py).
        episode_max_steps = int(max_steps) if max_steps is not None else int(horizon * 1.5)
        # Create simulation configuration
        config = SimulationConfig(
            env_name=f"robocasa/{env_name}",
            split=split,
            n_episodes=n_episodes,
            n_envs=n_envs,
            video=VideoConfig(
                video_dir=this_video_dir,
                render_camera=render_camera,
                render_width=render_width,
                render_height=render_height,
            ),
            multistep=MultiStepConfig(
                n_action_steps=n_action_steps, max_episode_steps=episode_max_steps,
            ),
        )

        # Run the simulation. Episodes are pinned to env seeds
        # test_start_seed + i, so initial conditions are identical across runs.
        print(f"Running simulation for {env_name}...")
        try:
            _, episode_successes, episode_records = simulation_client.run_simulation(
                config, test_start_seed=test_start_seed
            )
        except Exception as e:
            import traceback

            print(f"Exception while running {env_name}! Skipping this task.")
            traceback.print_exc()
            continue

        # Print results
        success_rate = float(np.mean(episode_successes))

        print(f"Results for {env_name}:")
        print(f"Success rate: {success_rate:.2f}")

        stats_path = os.path.join(this_video_dir, "stats.json")
        with open(stats_path, "w") as f:
            stats = {
                "num_episodes": len(episode_successes),
                "success_rate": success_rate,
            }
            json.dump(stats, f, indent=4)
        print(f"saved stats to {stats_path}")

        # Write a per-episode summary (eval_log.json) mirroring the diffusion
        # policy format: eval args, overall success rate, and per-seed reward /
        # video path / success step / success time.
        eval_log = {
            "eval_args": {
                "checkpoint": model_path,
                "command": " ".join(sys.argv),
                "max_steps": episode_max_steps,
                "n_action_steps": n_action_steps,
                "num_envs": n_envs,
                "num_rollouts": n_episodes,
                "render_camera": render_camera,
                "render_size": (
                    f"{render_width}x{render_height}"
                    if render_width and render_height
                    else None
                ),
                "seed": seed,
                "split": split,
                "task": env_name,
                "test_start_seed": test_start_seed,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
            f"success_rate/{env_name}": success_rate,
            "test/mean_score": success_rate,
        }
        for rec in episode_records:
            eval_log[f"test/sim_max_reward_{rec['seed']}"] = float(rec["reward"])
        for rec in episode_records:
            eval_log[f"test/sim_video_{rec['seed']}"] = rec["video_path"]
        for rec in episode_records:
            eval_log[f"test/success_step_{rec['seed']}"] = rec["success_step"]
        for rec in episode_records:
            eval_log[f"test/success_time_sec_{rec['seed']}"] = rec["success_time_sec"]
        eval_log_path = os.path.join(this_video_dir, "eval_log.json")
        with open(eval_log_path, "w") as f:
            json.dump(eval_log, f, indent=4)
        print(f"saved per-episode summary to {eval_log_path}")

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="<PATH_TO_YOUR_MODEL>",  # change this to your model path
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="new_embodiment",  # change this to your embodiment tag
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="The name of the data config to use.",
        default="panda_omron",  # change this to your embodiment tag
    )
    parser.add_argument(
        "--task_set",
        type=str,
        nargs='+',
        help="Name of the task soup(s)",
        required=True,
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Split to evaluate on. Can be either pretrain or target.",
        choices=["pretrain", "target"],
        required=True,
    )
    parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="localhost"
    )
    parser.add_argument("--video_dir", type=str, help="Directory to save videos.", default=None)
    parser.add_argument("--n_episodes", type=int, help="Number of episodes to run.", default=50)
    parser.add_argument("--n_envs", type=int, help="Number of parallel environments.", default=5)
    parser.add_argument(
        "--n_action_steps",
        type=int,
        help="Number of action steps per environment step.",
        default=16,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Step budget per rollout (default: 1.5x the task horizon, matching "
        "the diffusion policy eval). Lower it for a tighter budget / faster evals.",
    )
    parser.add_argument(
        "--render_camera",
        type=str,
        default=None,
        help="Camera for the saved videos. By default videos are the 256x256 policy "
        "observation image (robot0_agentview_left). Setting this renders a fresh frame "
        "straight from the sim using the named camera (e.g. robot0_agentview_center, "
        "robot0_frontview, robot0_eye_in_hand) -- any camera in the scene, not just the "
        "policy-input ones. Policy obs is unchanged.",
    )
    parser.add_argument(
        "--render_size",
        type=str,
        default="848x480",
        help="Resolution for saved videos as WxH (e.g. 848x480). Triggers a fresh "
        "higher-res render from the sim instead of the 256x256 obs image (policy obs "
        "unchanged, slightly slower). H264 needs even W and H. Default 848x480 matches "
        "the diffusion policy eval; pass an explicit WxH to override, or 256x256 to "
        "fall back to the policy-obs resolution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Per-run seed for the policy's sampling noise + the output dir name. "
        "Does NOT change env scenes (those use --test_start_seed). Default: drawn "
        "from entropy so repeated runs land in their own dir.",
    )
    parser.add_argument(
        "--test_start_seed",
        type=int,
        default=100000,
        help="Base env seed; episode i uses seed test_start_seed + i, giving "
        "identical initial conditions across runs.",
    )
    # server mode
    parser.add_argument("--server", action="store_true", help="Run the server.")
    # client mode
    parser.add_argument("--client", action="store_true", help="Run the client")
    args = parser.parse_args()

    # Resolve the per-run seed up front so the server (noise) and client (dir
    # name / eval_args) agree on it.
    if args.seed is None:
        args.seed = int(np.random.SeedSequence().generate_state(1)[0]) & 0x7FFFFFFF
    print(f"Per-run seed (policy noise + dir name): {args.seed}")

    # Parse --render_size WxH into (width, height) for the saved videos.
    render_width = render_height = None
    if args.render_size is not None:
        try:
            render_width, render_height = (int(x) for x in args.render_size.lower().split("x"))
        except ValueError:
            sys.exit(f"--render_size must be WxH (e.g. 848x480), got: {args.render_size}")
        if render_width % 2 or render_height % 2:
            sys.exit(f"--render_size dims must be even for h264, got: {args.render_size}")

    if args.server:
        run_server(
            data_config=args.data_config,
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
            port=args.port,
            seed=args.seed,
        )
    elif args.client:
        run_client(
            host=args.host,
            port=args.port,
            task_set_list=args.task_set,
            video_dir=args.video_dir or args.model_path,
            split=args.split,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            n_action_steps=args.n_action_steps,
            model_path=args.model_path,
            seed=args.seed,
            test_start_seed=args.test_start_seed,
            render_camera=args.render_camera,
            render_width=render_width,
            render_height=render_height,
            max_steps=args.max_steps,
        )
    else:
        # Run the policy server in a SEPARATE process (not a thread). The client
        # process builds the MuJoCo/EGL vector env -- including the render-enabled
        # probe env gymnasium's AsyncVectorEnv unconditionally creates in the main
        # process (async_vector_env.py: `dummy_env = env_fns[0]()`). An EGL/OpenGL
        # context living in the same process as the policy's torch CUDA context is
        # the documented "OpenGL error in Mujoco" failure (robocasa_diffusion_policy
        # works around it with a non-rendering dummy_env_fn) and was causing worker
        # aborts ("terminate called without an active exception") and hangs. Keeping
        # the server in its own process leaves the GL-building client process
        # CUDA-free, which is the ordinary robocasa-rendering case.
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        server_proc = ctx.Process(
            target=run_server,
            args=(args.data_config, args.model_path, args.embodiment_tag, args.port, args.seed),
            daemon=True,
        )
        server_proc.start()
        time.sleep(1)  # give server time to start (ZMQ REQ also blocks until it binds)
        try:
            run_client(
                host=args.host,
                port=args.port,
                task_set_list=args.task_set,
                video_dir=args.video_dir or args.model_path,
                split=args.split,
                n_episodes=args.n_episodes,
                n_envs=args.n_envs,
                n_action_steps=args.n_action_steps,
                model_path=args.model_path,
                seed=args.seed,
                test_start_seed=args.test_start_seed,
                render_camera=args.render_camera,
                render_width=render_width,
                render_height=render_height,
                max_steps=args.max_steps,
            )
        finally:
            server_proc.terminate()
            server_proc.join(timeout=10)