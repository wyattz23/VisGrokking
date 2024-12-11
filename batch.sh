#!/bin/bash

#SBATCH --job-name=xiang
#SBATCH --partition=beam
#SBATCH --quotatype=reserved
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1


export password=AzoRA5Bi6W86xZDWCZIZnJRw211okqgTpCnwC0pd22BtY8zE4EKCAZteSGIN
export http_proxy=http://zhangxiang:$password@10.1.20.50:23128/
export https_proxy=http://zhangxiang:$password@10.1.20.50:23128/


srun python3 run.py --mode train 