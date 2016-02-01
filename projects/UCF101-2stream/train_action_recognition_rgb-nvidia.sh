#!/usr/bin/env sh

GLOG_logbufsecs=1 GOOGLE_LOG_DIR=log \
    ${DAR_ROOT}/bin/caffe -gpu 0,1 train \
    --solver=../models-proto/two-streams-nvidia/vgg_16_rgb_solver.prototxt \
    --weights=../models-bin/vgg_16_action_rgb_pretrain.caffemodel
#    --weights=cuhk_action_spatial_vgg_16_split1.caffemodel

