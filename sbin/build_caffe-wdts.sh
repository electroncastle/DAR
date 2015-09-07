#!/bin/bash

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

DAR_PATH=$(dirname $SCRIPT_DIR)
echo $DAR_PATH

cmake .. -DUSE_MPI=ON -DBUILD_matlab=ON -DCMAKE_INSTALL_PREFIX:PATH=$DAR_PATH
make -j 8
make -j 8 pycaffe
make install



