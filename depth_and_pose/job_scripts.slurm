#!/bin/bash
#SBATCH -J w2_18_8_1e-3_200_val
#SBATCH -N 1
#SBATCH -o /scratch/meda/large_files/slurm_out_new/w2_18_8_1e-3_200_val-%j.out
#SBATCH -e /scratch/meda/large_files/slurm_out_new/w2_18_8_1e-3_200_val-%j.err
#SBATCH -t 200:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1
 

module purge
module load nvidia/latest


. /home/meda/miniconda3/etc/profile.d/conda.sh
conda activate with_intrinsics

export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python /home/meda/Thesis/Thesis_Project/sc_depth_pl/train.py --config /home/meda/Thesis/Thesis_Project/sc_depth_pl/configs/ego4d.txt --dataset_dir /scratch/meda/large_files/dataset_ego4d --val_mode photo

conda deactivate