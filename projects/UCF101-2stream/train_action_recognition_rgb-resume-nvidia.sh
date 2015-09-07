#!/usr/bin/env sh

GOOGLE_LOG_DIR=log \
    ../build-caffe-nvidia/bin/caffe --gpus 0,1 train \
    --solver=../models/two-streams-nvidia/vgg_16_rgb_solver.prototxt \
    --snapshot=two-streams_vgg_16_split1_rgb-nvidia_iter_10000.solverstate
   # --weights=vgg_16_action_rgb_pretrain.caffemodel
#    --weights=cuhk_action_spatial_vgg_16_split1.caffemodel

