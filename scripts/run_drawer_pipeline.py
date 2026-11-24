"""Generate drawer demonstrations, fine-tune PI05, then run online B8 GRPO."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent
ROOT = Path(os.getenv("SO101_RUN_DIR", PROJECT / "runs/drawer")).resolve()
BASE = Path(os.getenv("SO101_BASE", PROJECT / "models/ira-base")).resolve()
TOKENIZER = Path(os.getenv("SO101_TOKENIZER", PROJECT / "models/paligemma-tokenizer")).resolve()
START = Path(os.getenv("SO101_START_ADAPTER", PROJECT / "models/blockbox-002520")).resolve()
SOURCE_SHA = os.getenv("SO101_START_ADAPTER_SHA", "c4b65f3134cd065f45dc7c7e40f2d9adb3291116e3cc0eec49f1aa60a49a8fd6")
SFT_STEPS = 40000
SFT_INTERVAL = 5000
ROUNDS = 30
ROUND_UPDATES = 40
GROUPS = 90
RL_SEED_BASE = 180000
WORKER = None


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path, data):
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(data, indent=2) + "\n")
    staging.replace(path)


def status(phase, **details):
    publish(ROOT / "status.json", {"phase": phase, **details})


def stop(signum, _frame):
    if WORKER is not None and WORKER.poll() is None:
        WORKER.terminate()
    raise SystemExit(128 + signum)


def wait_gpu():
    while True:
        data = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        mib = int(data.splitlines()[0])
        if mib >= 14000:
            return
        print(f"Waiting for GPU memory ({mib} MiB available)", flush=True)
        time.sleep(30)


def worker(script, argv, log, gpu=True):
    global WORKER
    if gpu:
        wait_gpu()
    with log.open("a") as stream:
        WORKER = subprocess.Popen([sys.executable, "-u", str(SCRIPTS / script), *map(str, argv)],
                                  cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
        try:
            if WORKER.wait():
                raise RuntimeError(f"{script} failed; see {log}")
        finally:
            WORKER = None


def get_dataset():
    data = ROOT / "data"
    with (ROOT / "data.lock").open("a") as lock:
        status("generating_demonstrations", target=200)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (data / "manifest.json").is_file():
            worker("collect_drawer.py", ["--output", data], ROOT / "collect.log", gpu=False)
    manifest = read(data / "manifest.json")
    train, validation = manifest["train"], manifest["validation"]
    rows = {row["episode"]: row for row in manifest["episodes"]}
    if (len(manifest["episodes"]) != 200 or len(train) != 190 or len(validation) != 10
            or set(train) & set(validation) or set(train + validation) != set(rows)
            or len({row["seed"] for row in rows.values()}) != 200
            or not all(row["success"] and row["handle_contact"] and row["drawer_end_mm"] >= 60
                       for row in rows.values())):
        raise ValueError("Invalid 190/10 successful drawer split")
    for row in rows.values():
        folder = data / row["directory"]
        if not (folder / "motion.npz").is_file():
            raise ValueError(f"Incomplete motion file: {folder}")
        for camera in manifest["cameras"]:
            if sum(1 for _ in folder.glob(f"{camera}-*.png")) != row["rgb_frames"]:
                raise ValueError(f"Incomplete {camera} stream: {folder}")
    print("Verified 200 episodes, disjoint 190/10 split", flush=True)
    return data


def fine_tune(data):
    out = ROOT / "sft"
    existing = sorted(p.parent for p in (out / "checkpoints").glob("*/training.json"))
    if not existing and out.exists():
        out.rename(out.with_name(f"sft-interrupted-{time.time_ns()}"))
    if existing:
        for staging in (out / "checkpoints").glob("*.staging"):
            staging.rename(staging.with_name(staging.name + f".interrupted-{time.time_ns()}"))
    schedule = range(SFT_INTERVAL, SFT_STEPS+1, SFT_INTERVAL)
    if not existing or existing[-1].name != f"{SFT_STEPS:06d}":
        status("sft", target_minibatches=SFT_STEPS, target_optimizer_updates=SFT_STEPS//4)
        args = ["--base", BASE, "--adapter", START, "--adapter-sha256", SOURCE_SHA,
                "--tokenizer", TOKENIZER, "--data", data, "--output", out, "--steps",
                SFT_STEPS, "--save-freq", SFT_INTERVAL, "--batch-size", 2, "--accumulate", 4,
                "--lr", "1e-5", "--num-workers", 4, "--train-episodes", 190,
                "--validation-episodes", 10, "--offload-text-tables"]
        if existing:
            args.extend(["--resume", existing[-1]])
        worker("train_drawer_sft.py", args, ROOT / "sft.log")
    scores = []
    manifest_hash = digest(data / "manifest.json")
    for step in schedule:
        folder = out / "checkpoints" / f"{step:06d}"
        row = read(folder / "training.json")
        if (row["step"] != step or row["optimizer_updates"] != step // 4
                or row["config"]["manifest_sha256"] != manifest_hash
                or row["config"]["starting_adapter_sha256"] != SOURCE_SHA
                or row["validation_frames"] != 120 or not math.isfinite(row["validation_loss"])
                or not (folder / "training_state.pt").stat().st_size):
            raise ValueError(f"Invalid validation checkpoint: {folder}")
        scores.append({"step": step, "validation_loss": row["validation_loss"],
                       "path": str(folder), "adapter_sha256": digest(folder / "adapter_model.safetensors")})
    best = min(scores, key=lambda row: (row["validation_loss"], row["step"]))
    publish(ROOT / "sft-selection.json", {"selected": best, "scores": scores})
    print(f"Selected SFT step {best['step']} with validation loss {best['validation_loss']}", flush=True)
    return best


def prune_payloads(folder, rounds_root):
    folder = folder.resolve()
    if folder.parent != rounds_root.resolve() or not (folder / "complete.json").is_file():
        raise ValueError("Unsafe rollout cleanup")
    for manifest in (folder / "accepted").glob("group-*/group.json"):
        if manifest.resolve().parent.parent != folder / "accepted":
            raise ValueError(f"Unexpected group location: {manifest}")
        for entry in read(manifest)["records"]:
            name = entry["file"]
            if not re.fullmatch(r"chunk-[0-9]+\.pt\.gz", name):
                raise ValueError(f"Unexpected record: {name}")
            (manifest.parent / name).unlink(missing_ok=True)


def optimize(data, best):
    checkpoint, sha = Path(best["path"]), best["adapter_sha256"]
    rounds_root = ROOT / "grpo"
    for index in range(1, ROUNDS + 1):
        begin = (index - 1) * ROUND_UPDATES
        finish = begin + ROUND_UPDATES
        folder = rounds_root / f"round-{index:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "checkpoints" / f"{finish:06d}"
        status("grpo", round=index, rounds=ROUNDS, start_update=begin, round_end_update=finish,
               grpo_target=ROUNDS * ROUND_UPDATES, source_checkpoint=str(checkpoint), sft=best)
        if not target.is_dir():
            worker("train_drawer_round.py",
                   ["--base", BASE, "--adapter", checkpoint, "--adapter-sha256", sha,
                    "--tokenizer", TOKENIZER, "--output", folder, "--groups", GROUPS,
                    "--steps", ROUND_UPDATES, "--save-every", ROUND_UPDATES, "--lr", "1e-6",
                    "--start-step", begin, "--scene-offset", RL_SEED_BASE + (index-1)*1000,
                    "--validation-manifest", data / "manifest.json"],
                   folder / "run.log")
        meta = read(target / "rl_training.json")
        if (meta["updates"] != finish or meta["test_only"]
                or meta["run_config"]["provenance"]["adapter_sha256"] != sha
                or meta["run_config"]["task"] != "Open the drawer."
                or not (target / "optimizer.pt").stat().st_size):
            raise ValueError(f"Unexpected GRPO checkpoint: {target}")
        next_sha = digest(target / "adapter_model.safetensors")
        publish(folder / "complete.json", {"round": index, "step": finish,
                "source_sha256": sha, "adapter_sha256": next_sha})
        checkpoint, sha = target, next_sha
        print(f"GRPO round {index}/{ROUNDS}: checkpoint {finish:06d}", flush=True)
        if index > 1:
            prune_payloads(rounds_root / f"round-{index-1:03d}", rounds_root)
    status("complete", sft=best, rounds=ROUNDS, grpo_updates=ROUNDS*ROUND_UPDATES,
           checkpoint=str(checkpoint), adapter_sha256=sha)


if __name__ == "__main__":
    if digest(START / "adapter_model.safetensors") != SOURCE_SHA:
        raise ValueError("Source 002520 adapter hash mismatch")
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        data = get_dataset()
        optimize(data, fine_tune(data))
