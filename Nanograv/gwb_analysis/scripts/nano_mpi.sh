#!/bin/bash
#
#SBATCH --job-name=curn_32T_1k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=5000
#SBATCH --output=./log/08oct_curn_32T1k.log

ml purge

ml -q load conda
conda activate nanograv
mpirun -np 32 python nano_gss_multi.py

