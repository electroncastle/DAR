#!/bin/bash

proto="-2"
proto=""
proto_name="VGG_CNN_M_2048_OFR"
proto_name="OFR_GOOGLE_25"
proto_name="OFR_DeepMotion"
proto_name="OFR_FlowNet"
proto_name="OFR_VGG16_6"
proto_name="OFR_VGG16_6_fc"
proto_name="OFR_VGG16_6_fc14"
proto_name="OFR_VGG16_6_fc_d"

if [ "${1}" == "" ]; then
echo "Green field training"
GOOGLE_LOG_DIR=log \
    ${DAR_ROOT}/bin/caffe -gpu 0,1 train \
    --solver=models-proto/VGG_CNN_M_2048_OFR/${proto_name}_solver${proto}.prototxt
else
echo "Resuming from iteration ${1}"
GOOGLE_LOG_DIR=log \
    ${DAR_ROOT}/bin/caffe -gpu 0,1 train \
    --solver=models-proto/VGG_CNN_M_2048_OFR/${proto_name}_solver${proto}.prototxt \
    --weights=${proto_name}_iter_${1}.caffemodel
fi 
