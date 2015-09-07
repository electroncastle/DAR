#!/usr/bin/env sh

.caffe/build/tools/caffe train \
    --solver=models/two_streams/solver.prototxt

