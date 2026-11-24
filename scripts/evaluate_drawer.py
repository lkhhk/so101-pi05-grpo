"""Evaluate drawer checkpoints on the same independently seeded layouts."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from drawer_env import DrawerTask, HOLD_STEPS, TARGET_M
from eval_drawer import TASK
import evaluate_so101_mujoco_lora as eval_lib
from pi05_rl_runtime import load_ira_lora, make_processors
from train_so101_mujoco_sft import write_json


def run_episode(env, model, pre, post, mapping, scene_seed):
    env.reset(seed=scene_seed)
    policy_seed = scene_seed + 2_000_000
    torch.manual_seed(policy_seed)
    initial = env.observation()
    maximum = float(env.data.qpos[env.slide])
    times = []
    contact_steps = 0
    while not env.success and env.steps < 900:
        start = perf_counter()
        commands = eval_lib.predict(model, pre, post,
                                    mapping.observation(env.data.qpos[env.position]),
                                    env.images())
        times.append(perf_counter() - start)
        for cmd in commands:
            env.step(np.clip(mapping.radians(cmd), env.low, env.high))
            maximum = max(maximum, float(env.data.qpos[env.slide]))
            contact_steps += int(env.contacts()[1])
            if env.success or env.steps == 900:
                break
    return {"scene_seed": scene_seed, "policy_seed": policy_seed, "success": bool(env.success),
            "steps": env.steps, "simulated_seconds": env.steps/30,
            "initial_state": initial.tolist(), "initial_open_mm": float(initial[6])*1000,
            "max_open_mm": maximum*1000, "final_open_mm": float(env.data.qpos[env.slide])*1000,
            "handle_contact_steps": contact_steps, "inference_calls": len(times)}


def main(args):
    torch.set_num_threads(1)
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    torch.cuda.set_per_process_memory_fraction(9600 / total_mib, device=0)
    eval_lib.TASK = TASK
    seeds = [int(row) for row in args.seeds.read_text().splitlines()]
    assert len(seeds) == len(set(seeds)) == 30
    args.output.mkdir(exist_ok=True, parents=True)
    policy, provenance = load_ira_lora(args.base, args.adapter, device="cuda",
                                      offload_text_tables=True)
    if (provenance["base_sha256"], provenance["adapter_sha256"]) != (eval_lib.BASE_SHA, args.sha):
        raise ValueError("Unexpected checkpoint weights")
    policy.requires_grad_(False)
    pre, post = make_processors(policy, args.adapter, args.tokenizer, "cuda")
    mapping = eval_lib.JointMapping(json.loads(args.calibration.read_bytes()))
    report = {"label": args.label, "checkpoint": provenance, "task": TASK, "status": "running",
              "control_hz": 30, "execute_steps": 50, "denoising_steps": policy.config.num_inference_steps,
              "max_steps": 900, "target_mm": TARGET_M*1000, "hold_steps": HOLD_STEPS,
              "episodes": []}
    print("EVALUATOR_READY", args.label, flush=True)
    env = DrawerTask(args.assets)
    try:
        for seed in seeds:
            record = run_episode(env, policy, pre, post, mapping, seed)
            report["episodes"].append(record)
            write_json(args.output/"report.json", report)
            print("EPISODE", json.dumps(record), flush=True)
    finally:
        env.close()
    report["successes"] = sum(x["success"] for x in report["episodes"])
    report["status"] = "complete"
    write_json(args.output/"report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "base", "adapter", "tokenizer", "calibration", "seeds", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--sha", required=True)
    main(p.parse_args())
