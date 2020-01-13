#!/bin/bash

#BSUB -P gen011
#BSUB -W 1:00
#BSUB -nnodes 1

#BSUB -J save
#BSUB -o save.o%J
#BSUB -e save.e%J
##BSUB -alloc_flags NVME

### modules ###
module load gcc/6.4.0
module load fftw/3.3.8

### python ###
PYTHON="/ccs/home/l2v/nv_rapids_0.9_gcc_6.4.0/anaconda3"
export PYTHONIOENCODING="utf8"
export LD_LIBRARY_PATH=${PYTHON}/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4 
export NUMEXPR_MAX_THREADS=4

#Declare your project in the variable
projid=tomopy_run

HOME="/gpfs/alpine/scratch/l2v/gen011"
LOG=${HOME}/$projid/save.log
EXEC="${PYTHON}/bin/python -u ./to_npy.py"
FILE1='sort_rec_half_A.volume'
FILE2='sort_rec_half_B.volume'

cd ${HOME}/$projid

echo $NUMEXPR_MAX_THREADS

jsrun -n 1 -a 1 -c 1 ${EXEC} ${FILE1} > ${LOG}

jsrun -n 1 -a 1 -c 1 ${EXEC} ${FILE2} >> ${LOG}

#cp my_output_file /ccs/proj/abc123/Output.123
