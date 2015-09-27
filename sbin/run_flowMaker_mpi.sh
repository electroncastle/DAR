#!/bin/bash
 
mpirun -host 192.168.0.111 -np 2 ${DAR_ROOT}/src/flowMaker/build-release/flowMaker : \
       -host 192.168.0.108 -np 1 ${DAR_ROOT}/src/flowMaker/build-release-cuda65/flowMaker

