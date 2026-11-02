# D3F-IR

Standalone training project for visible-light-to-infrared image translation. The training
implementation, model, dataset reader, and FID evaluator are all inside this
directory; it does not import Python modules from the parent repository.

## Project layout

```text
D3F-IR/
├── train.py
├── train.sh
├── requirements.txt
├── data/                       # split manifests and dataset assets
├── ckpt/
│   ├── DINOv3/                 # default DINOv3 model and processor config
│   └── <dataset>/              # best.pt / latest.pt training checkpoints
├── model/
│   ├── config.py               # ModelConfig
│   ├── encoder.py              # DINO encoder
│   ├── dit.py                  # DiT
│   ├── pixel_predictor.py      # Pixel Predictor (former decoder)
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

## Data and checkpoints

Train/test split manifests from the source project are included under `data/`.
The dataset images are not copied; place them under the matching dataset folder
using paths in the JSON manifests (for example `data/KAIST_clean/lwir/` and
`data/KAIST_clean/visible/`).

The default dataset root is `data/<dataset-name>`:

- FLIR-align: `annotations_train.json`, `annotations_validation.json`, and
  their referenced images.
- KAIST_clean / M3FD_Detection: `train_set.json`, `test_set.json`, and the
  referenced images. The loader discovers these filenames automatically.

DINOv3 weights are included at `ckpt/DINOv3/model.safetensors` with the required
Hugging Face configuration files. Fine-tuning checkpoints default to
`ckpt/<dataset-name>/best.pt` and `latest.pt`; a pretraining checkpoint can be
placed at `ckpt/pretrain/best.pt`.

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

The selected dataset checkpoint is loaded automatically from
`ckpt/<dataset-name>/ckpt.pt`. Provide one RGB image or a directory of images;
directory structure is preserved in the output folder:

The bundled defaults are selected from `red-v-DAB-final_finetune2/FLIR-align/best.pt`,
`red-v-DAB-final_finetune2_fid/KAIST_clean/best copy.pt`, and
`red-v-DAB-final_finetune2_fid/M3FD_Detection/best.pt`, respectively.

```bash
python infer.py --dataset FLIR-align --input path/to/rgb_images
```

Use `--weights`, `--output_dir`, `--steps`, `--device`, or `--precision` to
override the defaults. Supported datasets are `FLIR-align`, `KAIST_clean`, and
`M3FD_Detection`.

`ModelConfig` in `model/config.py` is the single architecture configuration
for the DINO encoder, DiT, and Pixel Predictor. For direct Python invocation,
`train.py --help` lists the full training CLI.
