#!/bin/bash

#SBATCH --account=lcnrtx
#SBATCH --gpus=1
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=60G
#SBATCH --time=4-01:15:00
#SBATCH --output=out_log/%j.out
#SBATCH --job-name=ESRGAN
#SBATCH --mail-type=END,FAIL


source /autofs/cluster/HyperfineSR/sonia/BasicSR-master/env/bin/activate
cd /autofs/cluster/HyperfineSR/sonia/BasicSR-master/
python -u basicsr/train.py "$@"


