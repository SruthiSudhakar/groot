"""
Recover the per-episode language instruction (and the underlying object/
fixture choices) for a finished eval run, posthoc, from its eval_log.json.

The scene is a deterministic function of the per-episode env seed
(test_start_seed + i), which is recorded in eval_log.json. We just rebuild the
robocasa env and reset it with each seed -- no policy/server/rendering needed.

python scripts/recover_lang.py \
    /proj/vondrick3/sruthi/Appaji/released_checkpoints_groot/gr00t_n1-5/multitask_learning/checkpoint-120000/evals/pretrain/CloseFridge_14_50_2026.06.11_10.04.39_seed1609363440/eval_log.json \
    --json /proj/vondrick3/sruthi/Appaji/released_checkpoints_groot/gr00t_n1-5/multitask_learning/checkpoint-120000/evals/pretrain/CloseFridge_14_50_2026.06.11_10.04.39_seed1609363440/recovered_lang.json
"""
import argparse
import json
import os

import gymnasium as gym
import robocasa  # noqa: F401  (registers robocasa/<Task> envs)
import robosuite  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_log", help="path to eval_log.json from a run")
    ap.add_argument(
        "--json",
        default=None,
        help="path to dump full per-seed meta (default: sibling 'recovered_lang.json' "
        "next to the eval_log.json)",
    )
    args = ap.parse_args()

    # Always dump the full per-seed meta. Default to a sibling file of the
    # eval_log.json so the recovered info lives alongside the run it came from.
    out_path = args.json or os.path.join(
        os.path.dirname(os.path.abspath(args.eval_log)), "recovered_lang.json"
    )

    with open(args.eval_log) as f:
        log = json.load(f)
    ea = log["eval_args"]
    task = ea["task"]
    split = ea["split"]
    n = ea["num_rollouts"]
    base = ea["test_start_seed"]
    seeds = [base + i for i in range(n)]

    # Map each seed to its full saved video path (recorded in the log as
    # "test/sim_video_<seed>"). The video dir is the key we want in the output;
    # fall back to a synthetic media/seed<seed>.mp4 path if the entry is missing.
    video_for_seed = {}
    for s in seeds:
        video_for_seed[s] = log.get(f"test/sim_video_{s}")

    # Build the env exactly like the eval (simulation.py:_create_single_env),
    # minus rendering -- the scene/lang only depend on the reset seed.
    env = gym.make(f"robocasa/{task}", split=split, enable_render=False)

    out = {}
    print(f"{task} ({split}), seeds {seeds[0]}..{seeds[-1]}\n")
    for s in seeds:
        env.reset(seed=s)
        # unwrap to the underlying robocasa kitchen env that owns get_ep_meta()
        meta = env.unwrapped.get_ep_meta()
        lang = meta.get("lang", "")
        objs = [c.get("name") for c in meta.get("object_cfgs", [])]
        video_path = video_for_seed.get(s)
        # Key the output on the full saved video dir; stash the seed in the value.
        out[video_path] = {
            "seed": s,
            "lang": lang,
            "object_cfgs": meta.get("object_cfgs", []),
            "fixture_refs": meta.get("fixture_refs", {}),
            "layout_id": meta.get("layout_id"),
            "style_id": meta.get("style_id"),
        }
        print(f"seed {s}: {lang!s}   objects={objs}")

    env.close()
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
