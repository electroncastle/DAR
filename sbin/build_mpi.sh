#!/bin/bash

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

DAR_PATH=$(dirname $SCRIPT_DIR)
echo $DAR_PATH

# DAR_PATH=/home/jiri/Lake/DAR/mpi_bin/
./configure --with-cuda=${CUDA_HOME} --exec-prefix=$DAR_PATH  --prefix=$DAR_PATH


