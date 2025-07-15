#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32g
#SBATCH --tmp=32g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wan01530@umn.edu
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
export PATH=/users/9/wan01530/miniconda3/envs/py311/bin:$PATH
echo $PATH
nvidia-smi
eval "$(conda shell.bash hook)"
conda activate py311
# python -m cli run-super-resolution-task --config_name="super_resolution/config_ada_1"
# python -m cli run-super-resolution-task --config_name="super_resolution/config_ada_2"
# python -m cli run-super-resolution-task --config_name="super_resolution/config_ada_3"
# python -m cli run-super-resolution-task --config_name="super_resolution/config_ada_4"

python -m cli_ada_es_dip run-FMPlug-task --config_name="adam_l1_es_dip/superresolution_config"
python -m cli_ada_lora run-FMPlug-task --config_name="adam_l1_es_lora/superresolution_config"
python -m cli_ada_dgp run-FMPlug-task --config_name="adam_l1_es_dgp/superresolution_config"
# python -m cli_ada_dgp run-FMPlug-task --config_name="adam_l1_es_dgp/gaussian_blur_config"
# python -m cli_ada_dgp run-FMPlug-task --config_name="adam_l1_es_dgp/motion_blur_config"

# python -m cli_ada_es_1 run-FMPlug-task --config_name="adam_l1_es_1/inpainting_config"
# python -m cli_ada_es_1 run-FMPlug-task --config_name="adam_l1_es_1/superresolution_config"
# python -m cli_ada_es_1 run-FMPlug-task --config_name="adam_l1_es_1/gaussian_blur_config"
# python -m cli_ada_es_1 run-FMPlug-task --config_name="adam_l1_es_1/motion_blur_config"

# python -m cli_ada run-FMPlug-task --config_name="lbfgs/gaussian_blur_config"
# python -m cli_ada run-FMPlug-task --config_name="lbfgs/inpainting_config"
# python -m cli_ada run-FMPlug-task --config_name="lbfgs/motion_blur_config"
# python -m cli_ada run-FMPlug-task --config_name="lbfgs/superresolution_config"

# python -m cli_ada run-FMPlug-task --config_name="test_with_lpips/gaussian_blur_config"
# python -m cli_ada run-FMPlug-task --config_name="test_with_lpips/inpainting_config"
# python -m cli_ada run-FMPlug-task --config_name="test_with_lpips/motion_blur_config"
# python -m cli_ada run-FMPlug-task --config_name="test_with_lpips/superresolution_config"

# python -m cli_ada run-FMPlug-task --config_name="turbulence_config"