# SO-101 π0.5 LoRA and online GRPO

Two MuJoCo experiments on a single RTX 4090. The arm uses a front view, a wrist view and six joint-state values.

| Task | Checkpoint | Same-seed ODE results |
|---|---|---|
| Open a drawer | GRPO 000800 | 66/90 vs 51/90 for SFT 040000 |
| Put a block in a box | GRPO 002400 | 61/100 vs 45/100 for SFT 030000 |

These results describe the MuJoCo scenes and evaluation seeds in this repository. The drawer comparison has 6 GRPO-only successes, 1 SFT-only success and an exact paired p-value of 0.125.

- [Download the blockbox 002400 LoRA and processors](https://github.com/lkhhk/so101-pi05-grpo/releases/tag/so101-grpo-sim-20261124).
- [Drawer per-episode results](results/drawer/) and [blockbox per-episode results](results/blockbox/).
- [Recorded run configurations and checkpoint selection](configs/).
- [Drawer demonstration split and metadata](results/drawer-demonstration-manifest.json).

## Requirements

Use Linux with EGL and CUDA for training. The experiments used Python 3.11, CUDA PyTorch 2.7.1, LeRobot 0.4.4 source and an OpenPI-compatible Transformers 4.53.3 environment. `requirements-sim.txt` records the remaining observed versions. Use matching LeRobot/PI05 and tokenizer assets. A clean install of the full historical environment has not been verified.

Download the exact [MuJoCo Menagerie SO-101 description](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/robotstudio_so101) at commit `370c43965323c0944b433b202aac5058b5f907a9`. Put that model directory at `menagerie/`, with `so101.xml` and `assets/` inside it. Keep the upstream license with these assets.

The base is [Project-IRA V7 Full V2 checkpoint 008000](https://huggingface.co/Project-IRA/TPSoSe2026_Pi05_LeRobot_SO101_Finetuning_V7_Full_V2), under `outputs_V8/train/pi05_6gpu_fsdp_V2/checkpoints/008000/pretrained_model`. The verified `model.safetensors` SHA-256 is `7f8056e0cebbd8767a93c0f51bf5e20751baad455cf4b9965c07efca2df0fdef`. Supply the matching local PaliGemma tokenizer separately.

## Drawer training

Run from the repository root, using an activated compatible Python environment.

```bash
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SO101_BASE=/path/to/ira-008000/pretrained_model
export SO101_TOKENIZER=/path/to/paligemma-tokenizer
export SO101_START_ADAPTER=/path/to/blockbox-002520
export SO101_RUN_DIR="$PWD/runs/drawer"
python -u scripts/run_drawer_pipeline.py
```

The historical drawer experiment starts at blockbox `002520`. An alternate compatible adapter can be supplied by setting `SO101_START_ADAPTER_SHA` to its SHA-256. That changes the experiment's initialization. The published `002400` LoRA can be used this way.

1. Generate 200 successful scripted trajectories, with 190 training and 10 validation episodes. Actions and states are at 30 Hz, while dual-camera RGB observations are saved at 10 Hz.
2. Run 40,000 SFT mini-batches with batch size 2 and gradient accumulation 4, for 10,000 AdamW updates at learning rate 1e-5. Validate on 120 fixed windows and save every 5,000 mini-batches.
3. Select the lowest-loss SFT checkpoint. Run 30 online B8 GRPO rounds at learning rate 1e-6, with 90 mixed-outcome groups per round, 8 episodes per group, 4 groups per update and 40 updates per round.
4. Each round keeps its checkpoint and group metadata. The pipeline deletes a completed round's compressed rollout tensors after the following round commits.

The reward is 1 after contact with the handle and an opening of at least 60 mm held for 10 control steps. Simulation mapping in `official_block_follower.json` stores only the parameters needed for these experiments.

## Blockbox training and evaluation

`scripts/run_blockbox_online.py` is the 30-round blockbox continuation launcher. Use the same environment variables and give it the historical GRPO `001200` checkpoint with optimizer state. To reproduce the training run without intermediate ODE evaluation, create `skip-ode` within the directory designated by `SO101_RUN_DIR` before launching. The older B8 collection and training entrypoints are also included. Each entrypoint provides its arguments with `--help` where applicable.

`scripts/evaluate_so101_mujoco_lora.py` evaluates blockbox adapters. `scripts/evaluate_drawer.py` evaluates drawer adapters using a 30-scene seed file. Both verify the base and adapter SHA-256 and load checkpoint-specific processors. The reports specify the seed lists, 30 Hz control, 10-step denoising and execution of 50 steps per prediction.

This release includes source code, experimental configurations, demonstration metadata, per-episode evaluation reports and blockbox `002400` inference weights. Historical source checkpoints, raw demonstration RGB frames, optimizer states and full base weights must be obtained separately or regenerated where applicable.

## Attribution

The LoRA adapts Project-IRA's [V7 Full V2 checkpoint 008000](https://huggingface.co/Project-IRA/TPSoSe2026_Pi05_LeRobot_SO101_Finetuning_V7_Full_V2) by Cleo Baten, Bela Keppler and Jonas Sapper. The adapted checkpoint package follows [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

The pinned single-step Flow-SDE sampler comes from [RLinf](https://github.com/RLinf/RLinf) and retains its Apache-2.0 license. Its source and version manifest are in [vendor/rlinf-reference](vendor/rlinf-reference). MuJoCo Menagerie, LeRobot and OpenPI assets retain their upstream licenses.
