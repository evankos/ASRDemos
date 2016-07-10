#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 8:00:00
module load gcc/4.8.4
module load cuda/7.5.18
module load cudnn
module load mxnet
module load python
THEANO_FLAGS=mode='FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1' srun -u python mlp.py