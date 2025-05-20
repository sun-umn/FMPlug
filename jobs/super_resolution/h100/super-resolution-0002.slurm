#!/bin/bash -l
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem=100g
#SBATCH --tmp=20g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p msigpu
#SBATCH --gres=gpu:h100:1
cd /users/5/dever120//FMPlug
export PATH=/users/5/dever120/.conda/envs/materialmind/bin:$PATH
echo $PATH
nvidia-smi
eval "$(conda shell.bash hook)"
conda activate materialmind
python -m cli run-super-resolution-task --config_name="super_resolution/config_0002"