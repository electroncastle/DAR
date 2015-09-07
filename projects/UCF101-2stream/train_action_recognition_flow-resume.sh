#!/usr/bin/env sh

GOOGLE_LOG_DIR=log \
    /home/jiri/attic/mpi/bin/mpirun -np 1 \
    ../caffe-mpi/bin/caffe train \
    --solver=../models/two-streams/vgg_16_flow_solver.prototxt \
    --snapshot=two-streams_16_split1_flow-2_iter_11000.solverstate
    #--weights=cuhk_action_temporal_vgg_16_split1.caffemodel
    #--weights=vgg_16_action_flow_pretrain.caffemodel

