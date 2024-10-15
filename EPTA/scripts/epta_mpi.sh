#!/bin/bash

#SBATCH --job-name=curn_new_16T_20s
#SBATCH --nodes=1               
#SBATCH --ntasks=16    
#SBATCH --cpus-per-task=1          
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1400
#SBATCH --output=./log/new_curn_16T_20s.log
ml purge

ml -q load conda
conda activate epta

mpirun -np 16  python epta_selection.py --datadir /fred/oz103/ezahraoui/EPTA-DR2/DR2new/ --noisedir /fred/oz103/ezahraoui/EPTA-DR2/noisefiles/DR2new/ --orf_bins /fred/oz103/ezahraoui/EPTA-DR2/scripts_gwb/orf_bins.txt #--orf curn

#mpirun echo "sup" 
####sacct --user ezahraou --format="JobName%-20,avevmsize,State,ExitCode"