#!/bin/bash

mkdir -p log

GPU_ID=0,1

if [ "${2}" != "" ]; then
	GPU_ID=${2}
	echo "Setting GPU: ${GPU_ID}"
fi

if [ "${1}" == "n" ]; then
echo "Green field training"
GLOG_logbufsecs=1 GOOGLE_LOG_DIR=log \
    stdbuf -oL ${DAR_ROOT}/bin/caffe -gpu ${GPU_ID} train --solver=solver.prototxt
else
echo "Resuming from iteration ${1}"
#-gpu 0,1
GLOG_logbufsecs=1  GOOGLE_LOG_DIR=log \
    stdbuf -oL ${DAR_ROOT}/bin/caffe -gpu ${GPU_ID} train \
    --solver=solver.prototxt \
    --weights=snapshot_iter_${1}.caffemodel
fi 
