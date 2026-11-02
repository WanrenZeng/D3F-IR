# (ECCV2026) D3F-IR: Dual-Domain Deterministic Flow Matching for Visible-to-Infrared Translation

Official implementation of D3F-IR: Dual-Domain Deterministic Flow Matching for Visible-to-Infrared Translation. 

**Paper:** [ECCV 2026](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/6631.pdf)

## Project layout

```text
D3F-IR/
├── train.py
├── train.sh
├── requirements.txt
├── data/                       # split manifests and dataset assets
├── ckpt/
│   ├── DINOv3/                 # default DINOv3 model and processor config
│   └── <dataset>/              # ckpt.pt training checkpoints
├── model/
│   ├── config.py               # ModelConfig
│   ├── encoder.py              # DINO encoder
│   ├── dit.py                  # DiT
│   ├── pixel_predictor.py      # Pixel Predictor
│   ├── model.py                # three-component composition
│   └── modules/                # reusable neural-network blocks
├── dataset/
│   └── data.py
└── d3f_ir/
    ├── checkpoint.py
    ├── common.py
    ├── config.py
    ├── data.py
    ├── engine.py
    ├── evaluation_metrics.py
    ├── sampling.py
    ├── validation.py
    └── visualization.py
```

## Install

Use a Python environment with a PyTorch build compatible with your CUDA driver,
then install the remaining dependencies:

```bash
cd D3F-IR
python -m pip install -r requirements.txt
```

## Data

Train/test split manifests from the source project are included under `data/`.
The dataset images are not copied; place them under the matching dataset folder
using paths in the JSON manifests (for example `data/KAIST_clean/lwir/` and
`data/KAIST_clean/visible/`).

The default dataset root is `data/<dataset-name>`:

- FLIR-align: `annotations_train.json`, `annotations_validation.json`, and
  their referenced images.
- KAIST_clean / M3FD_Detection: `train_set.json`, `test_set.json`, and the
  referenced images. The loader discovers these filenames automatically.

## Weights

The best fine-tuned checkpoints are hosted on the Hugging Face Hub as one
repository per dataset: `wanren02/D3F-IR-FLIR-align`, `wanren02/D3F-IR-KAIST_clean`,
and `wanren02/D3F-IR-M3FD_Detection`. Download each `ckpt.pt` into
`ckpt/<dataset-name>/` so it lands at `ckpt/<dataset-name>/ckpt.pt` (the path
inference and evaluation read by default):

```bash
hf download wanren02/D3F-IR-FLIR-align     ckpt.pt --local-dir ckpt/FLIR-align
hf download wanren02/D3F-IR-KAIST_clean    ckpt.pt --local-dir ckpt/KAIST_clean
hf download wanren02/D3F-IR-M3FD_Detection ckpt.pt --local-dir ckpt/M3FD_Detection
```

- **DINOv3 — training only.** To train, download `facebook/dinov3-vits16-pretrain-lvd1689m`
  and place its files under `ckpt/DINOv3/` so that `ckpt/DINOv3/model.safetensors`
  exists (alongside `config.json` and `preprocessor_config.json`).
  Inference/evaluation do **not** need it: the encoder architecture is defined in
  code and its fine-tuned weights ship inside `ckpt.pt`.
- A pretraining checkpoint can optionally be placed at `ckpt/pretrain/best.pt`.
- Training writes `best.pt` / `latest.pt` under `ckpt/<dataset-name>/`.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 \
./train.sh FLIR-align 0
```

The launcher reads data and weights from the project-local `data/` and `ckpt/`
directories by default. Override them with `D3F_IR_DATA_DIR` and
`D3F_IR_DINO_CKPT` if needed. Set `D3F_IR_PRETRAIN_CKPT` to load an initial
model checkpoint, or `D3F_IR_RESUME_CKPT` to continue a saved run. Existing
default checkpoint files in `ckpt/` are detected automatically.
Training weights default to `ckpt/<dataset-name>/`, while visualizations default
to `outputs/visualizations/`.
The model always uses visible-light features as the flow source and fixed
velocity prediction with flow matching. Training parameters can be overridden
with environment variables such as `EPOCHS`, `LR`, `BATCH_SIZE`, `NUM_WORKERS`,
and `ALIGN_WEIGHT`.

## Inference

`infer.py` loads the best checkpoint for the selected dataset from
`ckpt/<dataset-name>/ckpt.pt` — i.e. the `D3F-IR-<dataset-name>` file you
downloaded above. Provide one RGB image or a directory of images; the directory
structure is preserved in the output folder:

```bash
python infer.py --dataset FLIR-align --input path/to/rgb_images
```

Use `--weights`, `--output_dir`, `--steps`, `--device`, or `--precision` to
override the defaults. Supported datasets are `FLIR-align`, `KAIST_clean`, and
`M3FD_Detection`.

`ModelConfig` in `model/config.py` is the single architecture configuration
for the DINO encoder, DiT, and Pixel Predictor. For direct Python invocation,
`train.py --help` lists the full training CLI.
