#!/usr/bin/env sh

	    
GLOG_logbufsecs=1 GOOGLE_LOG_DIR=log \
    ${DAR_ROOT}/bin/caffe -gpu 0,1 train \
    --solver=../models-proto/two-streams-nvidia/vgg_16_flow_solver.prototxt \
    --weights=../models-bin/vgg_16_action_flow_pretrain.caffemodel

#train --solver=../models/two-streams/vgg_16_flow_solver.prototxt --weights=vgg_16_action_flow_pretrain.caffemodel
