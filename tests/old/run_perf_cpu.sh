#!/bin/sh
#SBATCH --job-name perf_cpu
#SBATCH --chdir /home/changfenggroup/nrui/works/codes/gw_space/wf4ti/tests
#SBATCH --output /home/changfenggroup/nrui/works/codes/gw_space/wf4ti/tests/perf_cpu.out
#SBATCH --partition changfeng 
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
export OMP_NUM_THREADS=1

python perf_cpu.py