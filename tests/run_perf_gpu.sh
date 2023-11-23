#!/bin/sh
#SBATCH --job-name perf_gpu
#SBATCH --chdir /home/changfenggroup/nrui/works/codes/gw_space/wf4ti/tests
#SBATCH --output /home/changfenggroup/nrui/works/codes/gw_space/wf4ti/tests/perf_gpu.out
#SBATCH --partition changfeng 
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:v100:1

python perf_gpu_ti.py
python perf_gpu_ripple.py
python perf_gpu_bbhx.py
