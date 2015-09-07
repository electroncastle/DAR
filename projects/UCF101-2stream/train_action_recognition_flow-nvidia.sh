#!/usr/bin/env sh

GOOGLE_LOG_DIR=log \
    ../build-caffe-nvidia/bin/caffe --gpus 0,1 train \
    --solver=../models/two-streams-nvidia/vgg_16_flow_solver.prototxt \
    --weights=vgg_16_action_flow_pretrain.caffemodel
#   --weights=cuhk_action_temporal_vgg_16_split1.caffemodel
 

#train --solver=../models/two-streams/vgg_16_flow_solver.prototxt --weights=vgg_16_action_flow_pretrain.caffemodel
