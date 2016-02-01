#!/usr/bin/python

from __future__ import division
import sys
import os
import numpy as np

if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set, exiting'
    sys.exit()

dar_root = os.environ['DAR_ROOT']
app_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

#dar_root  ='/home/jiri/Lake/DAR/src/caffe-fcn/'
dar_root  ='/home/jiri/Lake/DAR/src/caffe/'
sys.path.insert(0, dar_root + '/python')
import caffe


# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

# base net -- the learned coarser model
base_weights = 'FCN16/fcn-16s-pascalcontext.caffemodel'
#base_weights = '/home/jiri/Lake/DAR/projects/optical_flow_regression/models-proto/VGG_CNN_M_2048_OFR/VGG_ILSVRC_16_layers.caffemodel'

# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')

if len(sys.argv) < 2:
    # do net surgery to set the deconvolution weights for bilinear interpolation
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    interp_surgery(solver.net, interp_layers)
else:
    base_weights = 'snapshot_iter_'+sys.argv[1]+'.caffemodel'

# copy base weights for fine-tuning
solver.net.copy_from(base_weights)

# solve straight through -- a better approach is to define a solving loop to
# 1. take SGD steps
# 2. score the model by the test net `solver.test_nets[0]`
# 3. repeat until satisfied
solver.step(80000)