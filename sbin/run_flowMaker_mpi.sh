#!/bin/bash
 
${DAR_ROOT}/bin/mpirun -host 192.168.0.111 -np 2 ${DAR_ROOT}/flowMaker : \
		      -host 192.168.103   -np 1 ${DAR_ROOT}/flowMaker-mbp

