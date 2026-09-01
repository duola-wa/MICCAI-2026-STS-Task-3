# MICCAI STSR 2026 Task 3


## 1. Installation

Ubuntu and a CUDA-capable GPU are recommended.

```bash
git clone https://github.com/duola-wa/MICCAI-2026-STS-Task-3.git
cd MICCAI-2026-STS-Task-3

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install a CUDA-enabled version of `torch` and `torchvision` compatible with your server, then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

## 2. Prepare the data

This repository requires CBCT volumes and external nnU-Net tooth segmentation results. 

Recommended directory layout:

```text
MICCAI-2026-STS-Task-3/
├── MICCAI-Chllenge-STS26-Task3/
│   ├── Train-Labeled/
│   │   ├── Train-Labeled.csv
│   │   └── <case_id>/<case_id>.nii.gz
│   ├── Validation/
│   │   └── <case_id>/<case_id>.nii.gz
│   └── prediction/
│       └── <case_id>.nii.gz
├── scripts/
├── train.py
└── ...
```

If the data is stored elsewhere, set the paths manually:

```bash
export MMDENTAL_DATA_ROOT=/path/to/MICCAI-Chllenge-STS26-Task3
export MMDENTAL_SEGMENTATION_DIR=/path/to/prediction
```


## 3. Run the pipeline

Check the environment and input data:

```bash
bash scripts/run_ubuntu.sh check
```

Prepare the 3D ROI cache:

```bash
NUM_WORKERS=2 bash scripts/run_ubuntu.sh prepare-roi3d
```

Optionally run a short training test first:

```bash
EXPERIMENT_NAME=dental_roi_3d_smoke \
EPOCHS=2 \
NUM_WORKERS=2 \
bash scripts/run_ubuntu.sh train-fold 0
```

Train all five folds:

```bash
bash scripts/run_ubuntu.sh train-all
```

Fit thresholds and run inference:

```bash
bash scripts/run_ubuntu.sh thresholds
bash scripts/run_ubuntu.sh predict
```

Alternatively, run the complete workflow with one command:

```bash
bash scripts/run_ubuntu.sh all
```

