# DPSA

This repository contains the core implementation of **DPSA: An Efficient
Transferable Adversarial Attack via DINOv3-Guided Path Sampling**.

DPSA uses DINOv3 as the default surrogate, samples neighborhood-aware operator
paths, and fuses path gradients with Gradient Distribution Synthesis (GDS).

The repository is code-only. Datasets, pretrained model weights, generated
adversarial examples, experiment logs, cached files, and plotting outputs are
not included.

## Repository Structure

```text
.
├── main.py               # DPSA generation and transferability evaluation
├── transferattack/       # attack/model utilities adapted from TransferAttack
├── run.sh                # example DPSA run script
├── pretrained_models/    # placeholder for optional local checkpoints
└── requirements.txt      # Python dependency list
```

## Third-Party Code

The `transferattack/` directory is based on the open-source
[Trustworthy-AI-Group/TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack)
project. Please follow the license and citation requirements of that upstream
project when using this code.

## Default Setting

| Item | Value |
|---|---:|
| Surrogate | DINOv3-Small |
| Loss | Cross entropy |
| Epsilon | 16/255 |
| Steps | 15 |
| Operator paths | 30 |
| Neighborhood paths | 15 |
| Chain length | 3 |
| Local probes | 1 |
| GDS diversity | 0.25 |

## Environment

Install a CUDA-enabled PyTorch build that matches your CUDA version, then
install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data Layout

By default, scripts expect datasets under `./data`:

```text
data/
└── ImageNet/
    ├── images/
    └── labels.csv
```

Other datasets use the same root layout:

```text
data/
└── CUB/
    ├── images/
    └── labels.csv
```

Use `--input_dir` to provide another dataset root.

`labels.csv` should contain `filename,label` for untargeted attacks. Targeted
attacks may also use `targeted_label`.

## Pretrained Models

ImageNet models from `torchvision`, `timm`, or HuggingFace can be loaded through
their normal pretrained interfaces. Optional local checkpoints are expected
under:

```text
pretrained_models/
├── dinov3-vits16-5ep-head.pth
├── dinov2-vits14-1k-head.pth
└── mae-base-1k-head.pth
```

Dataset-specific checkpoints can be placed under:

```text
pretrained_models/
└── <DATASET>/
    └── Pretrain/
        └── <MODEL_NAME>/
            └── best_model.pth
```

Use `--pretrained_models_root` to provide another checkpoint root.

## Run DPSA

Generate adversarial examples:

```bash
python main.py \
  --dataset ImageNet \
  --model dinov3 \
  --attack dpsa \
  --num_attacks 100 \
  --input_dir ./data \
  --output_dir ./results_dpsa \
  --pretrained_models_root ./pretrained_models \
  --GPU_ID 0
```

Evaluate transferability:

```bash
python main.py \
  --dataset ImageNet \
  --model dinov3 \
  --attack dpsa \
  --num_attacks 100 \
  --input_dir ./data \
  --output_dir ./results_dpsa \
  --pretrained_models_root ./pretrained_models \
  --GPU_ID 0 \
  -e
```

`dpsa` is the default attack method, so `--attack dpsa` can be omitted.

The helper script wraps generation and evaluation:

```bash
DATASET=ImageNet MODEL=dinov3 GPU_ID=0 ./run.sh
```

## Outputs

Generated outputs are written to `./results_dpsa` or the directory specified by
`--output_dir`. These outputs are ignored by Git and should not be committed.

## Notes

- This cleaned repository keeps only the DPSA core implementation and example
  run script.
- Plotting scripts, analysis scripts, generated figures, cached files,
  pretrained weights, and generated adversarial examples were removed.
- Before public release, add the final paper citation if available.

