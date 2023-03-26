#!/bin/bash

# Example of running python script in a batch mode

#SBATCH -J network_coco
#SBATCH -p normal                          # use normal when ready for production run 
#SBATCH -c 17                            # one CPU core - up to 62
#SBATCH -t 48:00:00
#SBATCH -o slurm_2.out
#SBATCH --mem=100GB                       # total memory - up to 3TB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larayang@stanford.edu

python3 coco_network.py actual long weighted

