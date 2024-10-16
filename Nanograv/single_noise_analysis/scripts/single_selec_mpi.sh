#!/bin/bash
#
#SBATCH --job-name=rn_8T_1ksamp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=3000
#SBATCH --output=./single_pulsar/log/single_J1012_rn_8T_1ksp.log

ml purge

ml -q load conda
conda activate nanograv
mpirun -np 8 python noise_selection.py

#mpirun echo "sup" 
