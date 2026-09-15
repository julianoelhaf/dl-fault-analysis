#!/bin/bash -l
#
# TEMPLATE SLURM SCRIPT.
#
# This demonstrates the full parameter grid used in the paper (5 window
# lengths x 4 targets x up to 9 models per target = 180 runs), but the
# original per-job wall-clock budget and job-array structure used to submit
# the paper's actual runs could not be recovered from project history. The
# --time value below is not sized for running the full grid in one job;
# split the loops below across a job array (e.g. one job per window length,
# or one per target) sized to your cluster's per-run walltime.
#
#SBATCH --job-name=<your_job_name>
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=0:18:00
#SBATCH --output=./hpc/hpc_logs/%x/%x-%j-on-%N.out

# -----------------------------------------
# Setup & Logging
# -----------------------------------------
mkdir -p ./hpc/hpc_logs/$SLURM_JOB_NAME
echo "Running on $(hostname) | Job ID: $SLURM_JOB_ID | Start: $(date)"
echo "TMPDIR: ${TMPDIR:-UNSET!}"

module load python/3.12-conda
conda activate py_dl

# -----------------------------------------
# Configuration
# -----------------------------------------
TOPOLOGY="hv_double_line_90kv" 
TOPOLOGY_CONFIG="hv_double_line_90kv" 

echo "Using topology config: $TOPOLOGY_CONFIG"

WINDOW_LENGTHS=(
  "0p050"
  "0p040"
  "0p030"
  "0p020"
  "0p010"
)

# Iterate over these targets:
FAULT_TARGETS=(
  "y_fault_line"
  "y_fault_location" # Regression target (continuous)
  "y_fault_present"
  "y_fault_class"
)

# Models per task type
CLASSIFIER_MODELS=(
  "cnn_classifier"
  "cnn_lstm_classifier"
  "dilated_cnn_classifier"
  "gru_classifier"
  "inceptiontime_classifier"
  "lstm_classifier"
  "rnn_classifier"
  "tcn_classifier"
  "tft_classifier"
)

REGRESSOR_MODELS=(
  "cnn_regressor"
  "cnn_lstm_regressor"
  "dilated_cnn_regressor"
  "gru_regressor"
  "inceptiontime_regressor"
  "lstm_regressor"
  "rnn_regressor"
  "tcn_regressor"
  "tft_regressor"
)

REPOSITORY_NAME="dl_fault_analysis"
DATASET_DIR="<your_dataset_directory>"  
PROJECT_DIR="<your_project_directory>"

JOB_TMP_DIR="$TMPDIR/$SLURM_JOB_ID"
WINDOWS_TMP_PATH="$JOB_TMP_DIR/windows_tmp"

export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
# If your cluster requires an HTTP(S) proxy for outbound access (e.g. to
# reach the Weights & Biases API), set it here:
# export https_proxy="<your_cluster_proxy>"


# -----------------------------------------
# Plan Summary
# -----------------------------------------
echo "-----------------------------------------"
echo "Execution Plan Summary"
echo "-----------------------------------------"
for WINDOW_LENGTH in "${WINDOW_LENGTHS[@]}"; do
  for FAULT_TARGET in "${FAULT_TARGETS[@]}"; do
    case "$FAULT_TARGET" in
      y_fault_location) MODEL_LIST=("${REGRESSOR_MODELS[@]}") ;;  # regression
      *)              MODEL_LIST=("${CLASSIFIER_MODELS[@]}") ;; # classification
    esac
    for MODEL_NAME in "${MODEL_LIST[@]}"; do
      echo "→ Topology: ${TOPOLOGY} | Window: ${WINDOW_LENGTH}s | Target: ${FAULT_TARGET} | Model: ${MODEL_NAME}"
    done
  done
done
echo "-----------------------------------------"
echo "Total combinations planned: $(for W in "${WINDOW_LENGTHS[@]}"; do for T in "${FAULT_TARGETS[@]}"; do case "$T" in y_fault_location) M=("${REGRESSOR_MODELS[@]}");; *) M=("${CLASSIFIER_MODELS[@]}");; esac; for _ in "${M[@]}"; do echo x; done; done; done | wc -l)"
echo "-----------------------------------------"


# -----------------------------------------
# File Preparation
# -----------------------------------------
if [ -z "$TMPDIR" ]; then
  echo "Error: TMPDIR is not set!"
  exit 1
fi

mkdir -p "$WINDOWS_TMP_PATH"
echo "Copying window files to: $WINDOWS_TMP_PATH"
START_TIME=$(date +%s)

SRC_DIR="$DATASET_DIR/windows_tmp"
EXTENSIONS=(raw parquet json)

for WINDOW_LENGTH in "${WINDOW_LENGTHS[@]}"; do
  for EXT in "${EXTENSIONS[@]}"; do
    FILE_PATTERN="*${TOPOLOGY}_W${WINDOW_LENGTH}_*.${EXT}"
    echo "Looking for: $FILE_PATTERN"

    if find "$SRC_DIR" -maxdepth 1 -type f -name "$FILE_PATTERN" -print -quit | grep -q .; then
      find "$SRC_DIR" -maxdepth 1 -type f -name "$FILE_PATTERN" \
        -exec cp -t "$WINDOWS_TMP_PATH" {} +
      echo "Copied ${EXT} files for window length: $WINDOW_LENGTH"
    else
      echo "WARNING: No ${EXT} files found for window length: $WINDOW_LENGTH"
    fi
  done
done

COPY_DURATION=$(( $(date +%s) - START_TIME ))
echo "Data copy completed in ${COPY_DURATION} seconds."




# -----------------------------------------
# Run Experiments
# -----------------------------------------
cd "$PROJECT_DIR" || exit 1

for W in "${WINDOW_LENGTHS[@]}"; do
  W_FLOAT="${W/p/.}"   # e.g., 0p020 -> 0.020

  echo "=================================================="
  echo "Window length: ${W} s (Hydra: ${W_FLOAT})"
  echo "=================================================="

  for T in "${FAULT_TARGETS[@]}"; do
    echo "Target: ${T}"

    if [[ "$T" == "y_fault_location" ]]; then
      MODEL_LIST=("${REGRESSOR_MODELS[@]}")
    else
      MODEL_LIST=("${CLASSIFIER_MODELS[@]}")
    fi

    for M in "${MODEL_LIST[@]}"; do
      echo "=================================================="
      echo "  → Model: ${M}"
      echo "=================================================="
      python src/dl_fault_analysis/scripts/run_dl_experiment.py \
        dataset="$TOPOLOGY_CONFIG" \
        model.model_name="$M" \
        window_extraction.window_length="$W_FLOAT" \
        training.target_label="$T" \
        tracking.project="dl_comparison_${TOPOLOGY}" \
        window_extraction.windows_local_dir="$WINDOWS_TMP_PATH"
    done
    echo
  done
done



echo "All jobs completed at $(date)"

echo "Cleaning up temporary directory: $JOB_TMP_DIR"
rm -rf "$JOB_TMP_DIR"
echo "Temporary directory cleaned up."
