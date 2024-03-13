#!/bin/bash -l
#SBATCH --job-name="nanotron-llama2-pretrain"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --gpus-per-node=4          # number of gpus per node
#SBATCH --partition=nvgpu
#SBATCH --time=00:10:00            # total run time limit (HH:MM:SS)


# Setting the environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

# Extra debugging flags, slow down training
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Distributed training variables
NNODES=${SLURM_NNODES}
GPUS_PER_NODE=4
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)


# Paths
BASE_PATH="/iopsstor/scratch/cscs/ctianche/playground/nanotron/llama_clariden" # FIXME
cd ${BASE_PATH}
SRC_PATH=${BASE_PATH}/../run_train.py
CACHE_PATH=${BASE_PATH}/caches
mkdir -p ${CACHE_PATH}

LOG_PATH=${BASE_PATH}/logs
mkdir -p ${LOG_PATH}

VENV_PATH=/iopsstor/scratch/cscs/ctianche/playground/pretrain/venv

SAVE_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}
TOKENIZER_PATH=${BASE_PATH}/tokenizer/tokenizer.model

# Set training command
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank \${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "

CMD="\
       ${LAUNCHER} \
       ${SRC_PATH} \
       --config-file ${BASE_PATH}/config_llama_7B.yaml
       "

srun -u --mpi=pmi2 --output=$PWD/logs/%x_${NAME}_%j.log --container-writable --environment=nanotron bash -c "
export NODE_RANK=\${SLURM_NODEID}
source ${VENV_PATH}/bin/activate
export TOKENIZERS_PARALLELISM=false
echo ${CMD}
${CMD}
"