#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32g
#SBATCH --tmp=32g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wan01530@umn.edu
#SBATCH -p apollo_agate
#SBATCH --gres=gpu:a100:1
export PATH=/users/9/wan01530/miniconda3/envs/py311/bin:$PATH
echo $PATH
nvidia-smi
eval "$(conda shell.bash hook)"
conda activate py311

python -m cli_ada run-FMPlug-task --config_name="regression/lbfgs"
python -m cli_ada_dgp_reg_hard_align_dip run-FMPlug-task --config_name="gauss_reg_dip/gaussian_blur_config_siren"
python -m cli_ada_dgp_reg_hard_align_dip run-FMPlug-task --config_name="gauss_reg_dip/gaussian_blur_config_dip"