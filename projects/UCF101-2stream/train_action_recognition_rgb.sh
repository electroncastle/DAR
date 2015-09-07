#!/usr/bin/env sh

GOOGLE_LOG_DIR=log \
    /home/jiri/attic/mpi/bin/mpirun -np 1 \
    ../caffe-mpi/bin/caffe train \
    --solver=../models/two-streams/vgg_16_rgb_solver.prototxt \
    --weights=cuhk_action_spatial_vgg_16_split1.caffemodel
#    --weights=vgg_16_action_rgb_pretrain.caffemodel

