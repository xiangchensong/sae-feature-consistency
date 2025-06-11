#!/bin/bash
#SBATCH --job-name=py-160m
#SBATCH --output=/home/%u/logs/sae-feature-consistency/pythia-160m-ds16k/%A_%a.out
#SBATCH --partition=array
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12-00:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-7%8

source /data/user_data/${USER}/mechanistic-interpretability/sae-feature-consistency/.venv/bin/activate
cd /data/user_data/${USER}/mechanistic-interpretability/sae-feature-consistency/examples

# Define architectures
ARCHITECTURES=("top_k" "batch_top_k" "jump_relu" "matryoshka_batch_top_k" "standard" "standard_new" "p_anneal" "gated")
DICTIONARY_WIDTHS=16384
LEARNING_RATES=3e-4
LAYERS=8

ARCH_IDX=${SLURM_ARRAY_TASK_ID}
ARCHITECTURE=${ARCHITECTURES[$ARCH_IDX]}

echo "Running on host: $(hostname)"
echo "Allocated GPU ID: ${SLURM_JOB_GPUS}"
echo "Running with architecture: ${ARCHITECTURE}, dictionary width: ${DICTIONARY_WIDTHS}, learning rate: ${LEARNING_RATES}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}, Architecture index: ${ARCH_IDX}"

export NCCL_P2P_DISABLE=1 
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python run_single.py \
    --save_checkpoints \
    --save_dir output/pythia-160m-ds16k/${ARCHITECTURE}/resid_post_layer_8 \
    --model_name EleutherAI/pythia-160m-deduped \
    --dataset_name monology/pile-uncopyrighted \
    --layers ${LAYERS} \
    --architectures ${ARCHITECTURE} \
    --random_seeds 42 43 44 \
    --dictionary_widths ${DICTIONARY_WIDTHS} \
    --learning_rates ${LEARNING_RATES} \
    --device cuda:0 \
    --use_wandb \
    --wandb_name pythia-160m-ds16k-${ARCHITECTURE}