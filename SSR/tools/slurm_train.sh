#!/bin/bash -l
#
#SBATCH --job-name=SSR_train               # your job name
#SBATCH --nodes=1                          # 1 node
#SBATCH --ntasks-per-node=1                # one srun task per node
#SBATCH --gres=gpu:rtx3080:8               # 8 GPUs on that node
#SBATCH --cpus-per-task=32                  # CPUs for data loading, etc.
#SBATCH --time=24:00:00                    # hh:mm:ss walltime
#SBATCH --partition=rtx3080,work                 # GPU partition
# (no --output/--error here—handled by srun instead)

# Paths (adjust to your environment)
export SIF_IMAGE="${WORK}/conda_ubuntu24_mmdet3d_plugin_latest.sif"
export WORKSPACE="${HOME}/wm_workspace"
export PROJECT="SSR"
export CODE_DIR="${WORKSPACE}/${PROJECT}"
export DATA_DIR="${WORK}/data"
export CONTAINER_WS="/workspace/${PROJECT}"

# Where to stage outputs/logs
LOCAL_ARTEFACTS_DIR="${HOME}/${PROJECT}_outputs"
NODE_LOCAL_ARTEFACTS_DIR="${TMPDIR}/${PROJECT}_outputs/${SLURM_JOB_ID}"
SLURM_LOGS_DIR="${TMPDIR}/logs"

mkdir -p "${SLURM_LOGS_DIR}" "${NODE_LOCAL_ARTEFACTS_DIR}" "${LOCAL_ARTEFACTS_DIR}"

# Go to your code root
cd "${CODE_DIR}"

# Bind code, data—and also the node‐local scratch—into the container
BIND_LIST="${CODE_DIR}:${CONTAINER_WS},${DATA_DIR}:${CONTAINER_WS}/data,${TMPDIR}:${TMPDIR}"

export WANDB_API_KEY="$(< "${HOME}/.wandb_key")"
export http_proxy="http://proxy.nhr.fau.de:80"
export https_proxy="http://proxy.nhr.fau.de:80"

# Launch training, capturing stdout/err into node-local files
# srun \
#   --output="${SLURM_LOGS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
#   --error ="${SLURM_LOGS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" \
  apptainer exec --nv \
    --bind "${BIND_LIST}" \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env http_proxy="${http_proxy}" \
    --env https_proxy="${https_proxy}" \
    "${SIF_IMAGE}" \
    bash -lc '
      python -m torch.distributed.run \
        --nproc_per_node=${SLURM_GPUS_ON_NODE} \
        --master_port=$((20000 + SLURM_JOB_ID % 10000)) \
        tools/train.py projects/configs/'"${PROJECT}"'/'"${PROJECT}"'_e2e.py \
        --tune_from '"${CODE_DIR}"'/ssr.pth \
        --launcher pytorch \
        --deterministic \
        --cfg-options dist_params.backend=gloo \
        --work-dir '"${CODE_DIR}"'/work_dirs
    '

# Stage back your model outputs
cp -r "${NODE_LOCAL_ARTEFACTS_DIR}" "${LOCAL_ARTEFACTS_DIR}/"

# And copy the captured logs into the same directory
cp "${SLURM_LOGS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"  "${LOCAL_ARTEFACTS_DIR}/${SLURM_JOB_ID}/"
cp "${SLURM_LOGS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"  "${LOCAL_ARTEFACTS_DIR}/${SLURM_JOB_ID}/"
