#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${MMDENTAL_DATA_ROOT:-${PROJECT_ROOT}/MICCAI-Chllenge-STS26-Task3}"
CACHE_DIR="${MMDENTAL_CACHE_DIR:-${PROJECT_ROOT}/cache/views_s12_224}"
RUNS_DIR="${MMDENTAL_RUNS_DIR:-${PROJECT_ROOT}/runs}"
SEGMENTATION_DIR="${MMDENTAL_SEGMENTATION_DIR:-${DATA_ROOT}/prediction}"
MODEL_TYPE="${MODEL_TYPE:-dental_roi_3d}"
if [[ -z "${EXPERIMENT_NAME:-}" ]]; then
  if [[ "${MODEL_TYPE}" == "dental_roi_3d" ]]; then
    EXPERIMENT_NAME="dental_roi_3d_v1"
  else
    EXPERIMENT_NAME="seg_fdi_v1"
  fi
fi
EXPERIMENT_DIR="${RUNS_DIR}/${EXPERIMENT_NAME}"
PREDICTIONS_DIR="${MMDENTAL_PREDICTIONS_DIR:-${PROJECT_ROOT}/predictions/${EXPERIMENT_NAME}}"
PYTHON="${PYTHON:-python}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_FOLDS="${NUM_FOLDS:-5}"
EPOCHS="${EPOCHS:-40}"
AMP="${AMP:-1}"
DEVICE="${DEVICE:-cuda}"
IMAGENET_PRETRAINED="${IMAGENET_PRETRAINED:-0}"
USE_SSL="${USE_SSL:-0}"
RESUME="${RESUME:-0}"
SSL_CHECKPOINT="${SSL_CHECKPOINT:-${RUNS_DIR}/ssl/backbone.pt}"
USE_SEGMENTATION="${USE_SEGMENTATION:-1}"
SEGMENTATION_DROPOUT="${SEGMENTATION_DROPOUT:-0.35}"
INIT_FROM_BASELINE="${INIT_FROM_BASELINE:-0}"
BASELINE_DIR="${BASELINE_DIR:-${RUNS_DIR}/labeled50}"
USE_TOOTH_PAIR_HEAD="${USE_TOOTH_PAIR_HEAD:-0}"
FREEZE_GLOBAL_MODEL="${FREEZE_GLOBAL_MODEL:-0}"
if [[ "${MODEL_TYPE}" == "dental_roi_3d" ]]; then
  BACKBONE_LR="${BACKBONE_LR:-1e-4}"
  GLOBAL_LR="${GLOBAL_LR:-1e-4}"
  FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-0}"
else
  BACKBONE_LR="${BACKBONE_LR:-1e-5}"
  GLOBAL_LR="${GLOBAL_LR:-3e-5}"
  FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-3}"
fi
TOOTH_LR="${TOOTH_LR:-3e-4}"
ROI3D_DEPTH="${ROI3D_DEPTH:-96}"
ROI3D_HEIGHT="${ROI3D_HEIGHT:-160}"
ROI3D_WIDTH="${ROI3D_WIDTH:-160}"
ROI3D_GLOBAL_MARGIN_MM="${ROI3D_GLOBAL_MARGIN_MM:-15}"
ROI3D_TOOTH_MARGIN_MM="${ROI3D_TOOTH_MARGIN_MM:-10}"
ROI3D_BASE_CHANNELS="${ROI3D_BASE_CHANNELS:-24}"

export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"

amp_args=()
if [[ "${AMP}" == "1" ]]; then
  amp_args+=(--amp)
fi

imagenet_args=()
if [[ "${IMAGENET_PRETRAINED}" == "1" ]]; then
  imagenet_args+=(--imagenet-pretrained)
fi

segmentation_check_args=()
if [[ "${MODEL_TYPE}" == "dental_roi_3d" ]]; then
  segmentation_check_args+=(
    --require-segmentations
    --required-segmentation-splits Train-Labeled Validation
  )
elif [[ "${USE_SEGMENTATION}" == "1" ]]; then
  segmentation_check_args+=(--require-segmentations)
fi

checkpoint_paths() {
  local fold
  for ((fold = 0; fold < NUM_FOLDS; fold++)); do
    printf '%s\n' "${EXPERIMENT_DIR}/fold_${fold}/best.pt"
  done
}

require_checkpoints() {
  local path
  while IFS= read -r path; do
    if [[ ! -f "${path}" ]]; then
      echo "Missing fold checkpoint: ${path}" >&2
      exit 1
    fi
  done < <(checkpoint_paths)
}

run_check() {
  local cuda_args=()
  if [[ "${REQUIRE_CUDA:-1}" == "1" ]]; then
    cuda_args+=(--require-cuda)
  fi
  "${PYTHON}" check_environment.py \
    --data-root "${DATA_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    --segmentation-dir "${SEGMENTATION_DIR}" \
    "${segmentation_check_args[@]}" \
    "${cuda_args[@]}" "$@"
}

run_prepare() {
  "${PYTHON}" prepare_data.py \
    --data-root "${DATA_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    "$@"
}

run_prepare_teeth() {
  "${PYTHON}" prepare_teeth.py \
    --data-root "${DATA_ROOT}" \
    --segmentation-dir "${SEGMENTATION_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    "$@"
}

run_prepare_roi3d() {
  "${PYTHON}" prepare_roi3d.py \
    --data-root "${DATA_ROOT}" \
    --segmentation-dir "${SEGMENTATION_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --output-shape "${ROI3D_DEPTH}" "${ROI3D_HEIGHT}" "${ROI3D_WIDTH}" \
    --global-margin-mm "${ROI3D_GLOBAL_MARGIN_MM}" \
    --tooth-margin-mm "${ROI3D_TOOTH_MARGIN_MM}" \
    --workers "${NUM_WORKERS}" \
    "$@"
}

run_smoke() {
  "${PYTHON}" tests/smoke_test.py \
    --data-root "${DATA_ROOT}" \
    --work-dir "${PROJECT_ROOT}/work/smoke" \
    --segmentation-dir "${SEGMENTATION_DIR}" \
    --device "${DEVICE}" \
    "${amp_args[@]}" "$@"
}

run_ssl() {
  "${PYTHON}" pretrain_ssl.py \
    --data-root "${DATA_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    --output "${SSL_CHECKPOINT}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    "${imagenet_args[@]}" "${amp_args[@]}" "$@"
}

run_train_fold() {
  local fold="$1"
  shift
  local ssl_encoder_args=()
  local resume_args=()
  local train_imagenet_args=("${imagenet_args[@]}")
  local segmentation_args=()
  local init_args=()
  local pair_args=()
  local model_args=(--model-type "${MODEL_TYPE}")
  if ! [[ "${fold}" =~ ^[0-9]+$ ]] || ((fold < 0 || fold >= NUM_FOLDS)); then
    echo "Fold must be an integer in [0, $((NUM_FOLDS - 1))]" >&2
    exit 2
  fi
  if [[ "${MODEL_TYPE}" == "dental_roi_3d" && ( "${USE_SSL}" == "1" || "${IMAGENET_PRETRAINED}" == "1" || "${INIT_FROM_BASELINE}" == "1" ) ]]; then
    echo "dental_roi_3d is a true-3-D network and cannot load the old 2-D SSL/ImageNet/baseline weights." >&2
    echo "Set USE_SSL=0 IMAGENET_PRETRAINED=0 INIT_FROM_BASELINE=0." >&2
    exit 2
  fi
  if [[ "${USE_SSL}" == "1" ]]; then
    if [[ ! -f "${SSL_CHECKPOINT}" ]]; then
      echo "Missing SSL checkpoint: ${SSL_CHECKPOINT}" >&2
      echo "Run: bash scripts/run_ubuntu.sh ssl" >&2
      exit 1
    fi
    ssl_encoder_args+=(--encoder-checkpoint "${SSL_CHECKPOINT}")
    train_imagenet_args=()
  fi
  if [[ "${MODEL_TYPE}" == "dental_roi_3d" ]]; then
    model_args+=(--roi3d-base-channels "${ROI3D_BASE_CHANNELS}")
    model_args+=(--segmentation-dropout "${SEGMENTATION_DROPOUT}")
  elif [[ "${USE_SEGMENTATION}" == "1" ]]; then
    segmentation_args+=(--use-segmentation --segmentation-dropout "${SEGMENTATION_DROPOUT}")
  fi
  if [[ "${INIT_FROM_BASELINE}" == "1" ]]; then
    if [[ ! -f "${BASELINE_DIR}/fold_${fold}/best.pt" ]]; then
      echo "Missing baseline checkpoint: ${BASELINE_DIR}/fold_${fold}/best.pt" >&2
      exit 1
    fi
    init_args+=(--init-model-checkpoint "${BASELINE_DIR}/fold_${fold}/best.pt")
    train_imagenet_args=()
    ssl_encoder_args=()
  fi
  if [[ "${USE_TOOTH_PAIR_HEAD}" == "1" ]]; then
    pair_args+=(--use-tooth-pair-head)
  fi
  if [[ "${FREEZE_GLOBAL_MODEL}" == "1" ]]; then
    if [[ "${INIT_FROM_BASELINE}" != "1" ]]; then
      echo "FREEZE_GLOBAL_MODEL=1 requires INIT_FROM_BASELINE=1" >&2
      exit 2
    fi
    pair_args+=(--freeze-global-model)
  fi
  if [[ "${RESUME}" == "1" && -f "${EXPERIMENT_DIR}/fold_${fold}/last.pt" ]]; then
    resume_args+=(--resume "${EXPERIMENT_DIR}/fold_${fold}/last.pt")
  fi
  "${PYTHON}" train.py \
    --data-root "${DATA_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    --output-dir "${EXPERIMENT_DIR}" \
    --fold "${fold}" \
    --num-folds "${NUM_FOLDS}" \
    --epochs "${EPOCHS}" \
    --batch-size 1 \
    --grad-accumulation 4 \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --lr "${TOOTH_LR}" \
    --backbone-lr "${BACKBONE_LR}" \
    --global-lr "${GLOBAL_LR}" \
    --freeze-backbone-epochs "${FREEZE_BACKBONE_EPOCHS}" \
    "${model_args[@]}" \
    "${train_imagenet_args[@]}" "${ssl_encoder_args[@]}" \
    "${segmentation_args[@]}" "${init_args[@]}" "${pair_args[@]}" \
    "${resume_args[@]}" "${amp_args[@]}" "$@"
}

run_train_all() {
  local fold
  for ((fold = 0; fold < NUM_FOLDS; fold++)); do
    echo "===== Training fold ${fold}/${NUM_FOLDS} ====="
    run_train_fold "${fold}" "$@"
  done
}

run_thresholds() {
  require_checkpoints
  local checkpoints=()
  mapfile -t checkpoints < <(checkpoint_paths)
  "${PYTHON}" fit_thresholds.py \
    --checkpoints "${checkpoints[@]}" \
    --schema-dir "${EXPERIMENT_DIR}/schema" \
    --data-root "${DATA_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    --output "${EXPERIMENT_DIR}/thresholds.json" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    "${amp_args[@]}" "$@"
}

run_audit() {
  "${PYTHON}" audit_predictions.py \
    --data-root "${DATA_ROOT}" \
    --predictions-jsonl "${PREDICTIONS_DIR}/predictions.jsonl" \
    --official-json "${PREDICTIONS_DIR}/predictions.json" \
    --submission-zip "${PREDICTIONS_DIR}/submission.zip" \
    "$@"
}

run_predict() {
  require_checkpoints
  if [[ ! -f "${EXPERIMENT_DIR}/thresholds.json" ]]; then
    echo "Missing thresholds: ${EXPERIMENT_DIR}/thresholds.json" >&2
    echo "Run: bash scripts/run_ubuntu.sh thresholds" >&2
    exit 1
  fi
  local checkpoints=()
  mapfile -t checkpoints < <(checkpoint_paths)
  "${PYTHON}" predict.py \
    --checkpoints "${checkpoints[@]}" \
    --schema-dir "${EXPERIMENT_DIR}/schema" \
    --data-root "${DATA_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    --output-dir "${PREDICTIONS_DIR}" \
    --thresholds-json "${EXPERIMENT_DIR}/thresholds.json" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --report-mode template \
    "${amp_args[@]}" "$@"
  run_audit
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_ubuntu.sh COMMAND [extra Python arguments]

Commands:
  check              Validate Python, CUDA, paths, split counts, and free space
  inspect            Audit the dataset records
  prepare            Cache all 300 CBCT cases
  prepare-teeth      Map masks in RAM and cache compact per-tooth CBCT views
  prepare-roi3d      Cache true-3-D dental-arch ROIs and FDI tooth boxes
  smoke              Run the small real-data forward/backward smoke test
  ssl                Optional SimSiam pretraining on 250 training CBCTs
  train-fold N       Train one supervised fold (N=0..4)
  train-all          Train all five folds sequentially
  thresholds         Fit entity thresholds from five-fold OOF predictions
  predict            Ensemble the folds and predict Validation
  audit              Validate official ZIP and report collapse indicators
  all                check, prepare, optional ssl, train-all, thresholds, predict

Environment overrides:
  PYTHON, NUM_WORKERS, NUM_FOLDS, EPOCHS, AMP, DEVICE, RESUME, MODEL_TYPE,
  IMAGENET_PRETRAINED, USE_SSL, SSL_CHECKPOINT, USE_SEGMENTATION,
  SEGMENTATION_DROPOUT, INIT_FROM_BASELINE, BASELINE_DIR, EXPERIMENT_NAME,
  USE_TOOTH_PAIR_HEAD, FREEZE_GLOBAL_MODEL, BACKBONE_LR, GLOBAL_LR,
  TOOTH_LR, FREEZE_BACKBONE_EPOCHS,
  MMDENTAL_DATA_ROOT, MMDENTAL_SEGMENTATION_DIR, MMDENTAL_CACHE_DIR,
  MMDENTAL_RUNS_DIR, MMDENTAL_PREDICTIONS_DIR
  ROI3D_DEPTH, ROI3D_HEIGHT, ROI3D_WIDTH, ROI3D_GLOBAL_MARGIN_MM,
  ROI3D_TOOTH_MARGIN_MM, ROI3D_BASE_CHANNELS
EOF
}

command="${1:-}"
if [[ -n "${command}" ]]; then
  shift
fi
case "${command}" in
  check) run_check "$@" ;;
  inspect) "${PYTHON}" inspect_dataset.py --data-root "${DATA_ROOT}" "$@" ;;
  prepare) run_prepare "$@" ;;
  prepare-teeth) run_prepare_teeth "$@" ;;
  prepare-roi3d) run_prepare_roi3d "$@" ;;
  smoke) run_smoke "$@" ;;
  ssl) run_ssl "$@" ;;
  train-fold)
    if (($# == 0)); then
      usage
      exit 2
    fi
    run_train_fold "$@"
    ;;
  train-all) run_train_all "$@" ;;
  thresholds) run_thresholds "$@" ;;
  predict) run_predict "$@" ;;
  audit) run_audit "$@" ;;
  all)
    run_check
    if [[ "${MODEL_TYPE}" == "dental_roi_3d" ]]; then
      run_prepare_roi3d
    else
      run_prepare
    fi
    if [[ "${MODEL_TYPE}" != "dental_roi_3d" && "${USE_SEGMENTATION}" == "1" ]]; then
      run_prepare_teeth
    fi
    if [[ "${USE_SSL}" == "1" ]]; then
      run_ssl
    fi
    run_train_all
    run_thresholds
    run_predict
    ;;
  *) usage; exit 2 ;;
esac
