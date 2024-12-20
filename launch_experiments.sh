#!/bin/bash
#SBATCH --job-name=experimentation-resselnet         # create a short name for your job

#SBATCH --nodes=1                # node count
#SBATCH --nodelist=gpu08
#SBATCH -o logs/experiments.%j.out    # Specify stdout output file (%j expands to jobId)
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus=a40               # gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)               

hostname
# Load any necessary modules
# Loading modules in the script ensures a consistent environment.
source /etc/profile
source venv/bin/activate

# Launch the executable 
python -u experiment.py

