#!/bin/bash -l
#BSUB -P stf011
#BSUB -J test
#BSUB -o logs.o%J
#BSUB -W 02:00
#BSUB -nnodes 1
#BSUB -alloc_flags "smt4 nvme"
#BSUB -q batch

NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
module load ibm-wml-ce gcc/7.4.0
#HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=$conda_dir HOROVOD_CUDA_HOME=$conda_dir pip install --install-option="--prefix=$(pwd)" --force horovod==0.16.4
export PYTHONPATH=$(pwd)/lib/python3.6/site-packages:$PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL='3'

#X_train=360proj_36KF_train.h5
#X_test=360proj_36KF_test.h5
xtrain=540proj_1acc_train.h5
xtest=540proj_1acc_test.h5
ytrain=2880proj_20acc_train.h5
ytest=2880proj_20acc_test.h5

datadir=/gpfs/alpine/scratch/junqi/stf011/lbpm/TomoGAN/TomoGAN/dataset/shuffled/grayscale
exe=../../main-gan.py
expName=ryan540hvd

jsrun -n$NODES -a1 -c42 -r1 cp $datadir/$xtrain $datadir/$ytrain $datadir/$xtest $datadir/$ytest /mnt/bb/$USER 

jsrun -n$NODES -a6 -g6 -c42 -r1 --bind=proportional-packed:7 --launch_distribution=packed python $exe -expName $expName -xtrain /mnt/bb/$USER/$xtrain -ytrain /mnt/bb/$USER/$ytrain -xtest /mnt/bb/$USER/$xtest -ytest /mnt/bb/$USER/$ytest -depth 3
