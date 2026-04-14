#!/bin/bash
# MPI-parallel 3D H1 PhC sweep.  Adjust NPROC to your machine.
source /home/zlin/miniforge/etc/profile.d/conda.sh
conda activate meep
NPROC=${NPROC:-16}
cd "$(dirname "$0")"
mpirun -n $NPROC python meep_h1_3d.py "$@"
