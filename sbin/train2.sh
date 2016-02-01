#!/bin/bash

mkdir -p log

if [ "${1}" == "" ]; then
	echo "Green field training"
	GLOG_logbufsecs=1 GOOGLE_LOG_DIR=log \
	    stdbuf -oL ${DAR_ROOT}/bin/caffe -gpu 0,1 train --solver=solver.prototxt
else
	echo "Resuming from iteration ${1}"
	GLOG_logbufsecs=1  GOOGLE_LOG_DIR=log \
	    stdbuf -oL ${DAR_ROOT}/bin/caffe -gpu 0,1 train \
	    --solver=solver.prototxt \
	    --weights=snapshot_iter_${1}.caffemodel
fi 
