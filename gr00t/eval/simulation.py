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
import math
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from tqdm import tqdm

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401

from gr00t.data.dataset import ModalityConfig
from gr00t.eval.service import BaseInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from gr00t.model.policy import BasePolicy

# from gymnasium.envs.registration import registry

# print("Available environments:")
# for env_spec in registry.values():
#     print(env_spec.id)


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: Optional[str] = None
    steps_per_render: int = 2
    fps: int = 10
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    # Saved-video camera. When render_camera and/or render_width/render_height are set,
    # the env renders a fresh frame from the sim (any scene camera, arbitrary res)
    # instead of the cached 256x256 policy observation image. Policy obs is unchanged.
    render_camera: Optional[str] = None
    render_width: Optional[int] = None
    render_height: Optional[int] = None


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    split: str = "test"
    n_episodes: int = 2
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


class SimulationInferenceClient(BaseInferenceClient, BasePolicy):
    """Client for running simulations and communicating with the inference server."""

    def __init__(self, host: str = "localhost", port: int = 5555):
        """Initialize the simulation client with server connection details."""
        super().__init__(host=host, port=port)
        self.env = None

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from the inference server based on observations."""
        # NOTE(YL)!
        # hot fix to change the video.ego_view_bg_crop_pad_res256_freq20 to video.ego_view
        if "video.ego_view_bg_crop_pad_res256_freq20" in observations:
            observations["video.ego_view"] = observations.pop(
                "video.ego_view_bg_crop_pad_res256_freq20"
            )
        return self.call_endpoint("get_action", observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """Get modality configuration from the inference server."""
        return self.call_endpoint("get_modality_config", requires_input=False)

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        # Create environment functions for each parallel environment
        env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        # Create vector environment (sync for single env, async for multiple)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
                # Don't auto-reset finished envs (no throwaway RoboCasa scene
                # rebuilds while a chunk drains). See _no_autoreset_worker.
                worker=_no_autoreset_worker,
            )

    def run_simulation(
        self, config: SimulationConfig, test_start_seed: int = 100000
    ) -> Tuple[str, List[bool], List[Dict[str, Any]]]:
        """Run the simulation for the specified number of episodes.

        Episodes are pinned to deterministic env seeds ``test_start_seed + i``
        (so initial conditions are identical across runs) and executed in
        chunks of ``n_envs``. Each episode's video is named ``media/seed{seed}.mp4``.

        Returns the env name, the per-episode success flags (ordered by seed),
        and a list of per-episode record dicts (seed / success / reward /
        success_step / success_time_sec / video_path), suitable for writing a
        per-episode eval summary.
        """
        start_time = time.time()
        n_episodes = config.n_episodes
        n_envs = config.n_envs
        print(
            f"Running {n_episodes} episodes for {config.env_name} with {n_envs} environments"
        )
        # Set up the environment
        self.env = self.setup_environment(config)

        # Per-vector-step wall-clock timeout (seconds). A normal step takes a few
        # seconds; only a wedged worker exceeds this. Override via env var.
        self.step_timeout_sec = float(os.environ.get("EVAL_STEP_TIMEOUT_SEC", "180"))

        # Control frequency (Hz) -> convert the first-success base step into
        # wall-clock seconds. Robocasa defaults to 20 Hz; fall back if unreadable.
        try:
            control_freq = float(self.env.get_attr("control_freq")[0])
        except Exception:
            control_freq = 20.0

        # Deterministic per-episode env seeds and the dir videos land in.
        seeds = [test_start_seed + i for i in range(n_episodes)]
        media_dir = (
            os.path.join(config.video.video_dir, "media")
            if config.video.video_dir is not None
            else None
        )

        episode_records: List[Dict[str, Any]] = []
        n_chunks = math.ceil(n_episodes / n_envs)
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_episodes, start + n_envs)
            k = end - start  # number of *active* (non-padding) envs this chunk
            chunk_seeds = seeds[start:end]
            # AsyncVectorEnv needs exactly n_envs seeds; pad with None (unseeded,
            # no video) for a short final chunk.
            reset_seeds = list(chunk_seeds) + [None] * (n_envs - k)

            # Seeded reset: pins each env's scene and names its video by seed.
            obs, _ = self.env.reset(seed=reset_seeds)

            active_done = [False] * n_envs
            max_reward = [0.0] * n_envs
            succ_step: List[Optional[int]] = [None] * n_envs

            # Step until every active env has finished its (seeded) episode.
            # Finished envs are NOT auto-reset (see _no_autoreset_worker); they
            # just hold their terminal state while the rest of the chunk drains.
            # The bar tracks base env steps for the episode (each vector step
            # advances n_action_steps), so 0 -> max_episode_steps.
            n_action_steps = config.multistep.n_action_steps
            max_steps = config.multistep.max_episode_steps
            pbar = tqdm(
                total=max_steps,
                desc=f"{config.env_name} chunk {chunk_idx + 1}/{n_chunks}",
                unit=" step",
                leave=False,  # bar disappears after the chunk, keeping logs clean
                mininterval=5.0,  # throttle refreshes for non-tty run.sh logs
            )
            timed_out = False
            while not all(active_done[:k]):
                actions = self._get_actions_from_server(obs)
                try:
                    obs, rewards, terminations, truncations, env_infos = self._step(actions)
                except multiprocessing.TimeoutError:
                    # A worker wedged. Don't hang the whole eval: leave the
                    # unfinished envs as not-done (they get recorded as 0 reward /
                    # no success below, i.e. failures) and rebuild the env so the
                    # episode count per task stays fixed at n_episodes.
                    n_unfinished = sum(1 for d in active_done[:k] if not d)
                    print(
                        f"  Step exceeded {self.step_timeout_sec}s on chunk "
                        f"{chunk_idx + 1}; marking {n_unfinished} unfinished "
                        f"episode(s) as failures and rebuilding the env."
                    )
                    timed_out = True
                    break
                pbar.update(n_action_steps)
                pbar.set_postfix_str(f"{max(max_steps - pbar.n, 0)} steps left")
                final_info = env_infos.get("final_info") if isinstance(env_infos, dict) else None
                for env_idx in range(k):
                    if active_done[env_idx]:
                        continue
                    max_reward[env_idx] = max(max_reward[env_idx], float(rewards[env_idx]))
                    if terminations[env_idx] or truncations[env_idx]:
                        active_done[env_idx] = True
                        # Pull the terminal episode's success step out of final_info.
                        # The wrapper emits -1 (sentinel) when the episode never
                        # succeeded; map that back to None here.
                        fi = final_info[env_idx] if final_info is not None else None
                        if isinstance(fi, dict):
                            s = fi.get("success_step")
                            succ_step[env_idx] = int(s) if (s is not None and s >= 0) else None
            pbar.close()

            if timed_out:
                # Force-kill the wedged workers and start a fresh vector env for
                # the remaining chunks.
                self.env.close(terminate=True)
                self.env = self.setup_environment(config)

            for env_idx in range(k):
                seed = chunk_seeds[env_idx]
                step = succ_step[env_idx]
                success = bool(max_reward[env_idx] > 0)
                episode_records.append(
                    {
                        "episode": start + env_idx,
                        "seed": int(seed),
                        "success": success,
                        "reward": float(max_reward[env_idx]),
                        "success_step": (int(step) if step is not None else None),
                        "success_time_sec": (
                            round(step / control_freq, 2) if step is not None else None
                        ),
                        "video_path": (
                            os.path.join(media_dir, f"seed{seed}.mp4")
                            if media_dir is not None
                            else None
                        ),
                    }
                )
            cumulative_sr = float(np.mean([r["success"] for r in episode_records]))
            print(
                f"Chunk {chunk_idx + 1}/{n_chunks} done "
                f"({len(episode_records)}/{n_episodes} episodes); "
                f"cumulative success rate: {cumulative_sr:.3f}"
            )

        # Clean up
        self.env.close()
        self.env = None
        episode_successes = [r["success"] for r in episode_records]
        print(
            f"Collecting {n_episodes} episodes took {time.time() - start_time:.2f} seconds"
        )
        return config.env_name, episode_successes, episode_records

    def _step(self, actions):
        """Step the vector env, applying a wall-clock timeout for async envs.

        Stock ``env.step()`` blocks forever if a worker wedges; ``step_wait``
        with a timeout raises ``multiprocessing.TimeoutError`` instead, so the
        caller can recover rather than hang.
        """
        if isinstance(self.env, gym.vector.AsyncVectorEnv):
            self.env.step_async(actions)
            return self.env.step_wait(timeout=self.step_timeout_sec)
        return self.env.step(actions)

    def _get_actions_from_server(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Process observations and get actions from the inference server."""
        # Get actions from the server
        action_dict = self.get_action(observations)
        # Extract actions from the response
        if "actions" in action_dict:
            actions = action_dict["actions"]
        else:
            actions = action_dict
        # Add batch dimension to actions
        return actions


def _no_autoreset_worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    """AsyncVectorEnv worker that does NOT auto-reset an env when it finishes.

    This is a copy of gymnasium 0.29's default ``_worker`` with one change: a
    terminated/truncated env is left in its terminal state instead of being
    reset. RoboCasa's ``reset()`` rebuilds the entire kitchen scene, so the
    default behaviour makes every early-finishing env regenerate a random
    throwaway scene (and occasionally wedge a worker) while it waits for the
    slowest env in the chunk. We still surface the terminal episode's info as
    ``final_info``/``final_observation``, exactly like the default worker, so the
    run loop is unchanged. Re-stepping a finished env is a no-op because
    MultiStepWrapper short-circuits once it is done.
    """
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation, info = env.reset(**data)
                pipe.send(((observation, info), True))
            elif command == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    # Surface the terminal info the same way gymnasium does on
                    # autoreset, but WITHOUT calling env.reset().
                    final_info = dict(info)
                    # The per-episode summary fields are terminal-only and live
                    # in final_info. Pop them off the top-level info so they are
                    # not batched by the vector env's _add_info across envs.
                    # (With gymnasium's default autoreset these never reached
                    # top-level info.) NOTE: this only fires on terminal steps;
                    # the crash-prevention for mid-episode steps comes from the
                    # MultiStepWrapper emitting an int sentinel (-1) for an
                    # unsolved success_step rather than None, keeping the
                    # _add_info array dtype consistent across envs.
                    for key in ("success_step", "episode_base_step", "episode_success"):
                        info.pop(key, None)
                    info["final_observation"] = observation
                    info["final_info"] = final_info
                pipe.send(((observation, reward, terminated, truncated, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with `_call`. "
                        f"Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                pipe.send(
                    (
                        (data[0] == env.observation_space, data[1] == env.action_space),
                        True,
                    )
                )
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must be one of "
                    "{`reset`, `step`, `seed`, `close`, `_call`, `_setattr`, "
                    "`_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    """Create a single environment with appropriate wrappers."""
    # Create base environment
    env = gym.make(
        config.env_name,
        split=config.split,
        enable_render=True,
        render_camera=config.video.render_camera,
        render_width=config.video.render_width,
        render_height=config.video.render_height,
    )
    # Add video recording wrapper if needed (only for the first environment)
    if config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
        )
    # Add multi-step wrapper
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env


def run_evaluation(
    env_name: str,
    host: str = "localhost",
    port: int = 5555,
    video_dir: Optional[str] = None,
    n_episodes: int = 2,
    n_envs: int = 1,
    n_action_steps: int = 2,
    max_episode_steps: int = 100,
) -> Tuple[str, List[bool]]:
    """
    Simple entry point to run a simulation evaluation.
    Args:
        env_name: Name of the environment to run
        host: Hostname of the inference server
        port: Port of the inference server
        video_dir: Directory to save videos (None for no videos)
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        n_action_steps: Number of action steps per environment step
        max_episode_steps: Maximum number of steps per episode
    Returns:
        Tuple of environment name and list of episode success flags
    """
    # Create configuration
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=n_episodes,
        n_envs=n_envs,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps, max_episode_steps=max_episode_steps
        ),
    )
    # Create client and run simulation
    client = SimulationInferenceClient(host=host, port=port)
    results = client.run_simulation(config)
    # Print results
    print(f"Results for {env_name}:")
    print(f"Success rate: {np.mean(results[1]):.2f}")
    return results


if __name__ == "__main__":
    # Example usage
    run_evaluation(
        env_name="robocasa_gr1_arms_only_fourier_hands/TwoArmPnPCarPartBrakepedal_GR1ArmsOnlyFourierHands_Env",
        host="localhost",
        port=5555,
        video_dir="./videos",
    )
