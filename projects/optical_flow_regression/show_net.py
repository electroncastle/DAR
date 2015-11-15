#!/usr/bin/python
from gst.video import video_convert_frame
from matplotlib.pyplot import plot

__author__ = "Jiri Fajtl"
__copyright__ = "Copyright 2007, The DAR Project"
__credits__ = [""]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Jiri Fajtl"
__email__ = "ok1zjf@gmail.com"
__status__ = "Prototype"

import os
import io
import sys
# import random
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import PIL.Image
import time
import lmdb
import cv2
import utils

if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set, exiting'
    sys.exit()

dar_root = os.environ['DAR_ROOT']
app_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

sys.path.insert(0, dar_root + '/python')
import caffe


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    if len(data.shape) > 2 and data.shape[2] == 1:
        data = data.transpose(2,0,1)
        data = data[0]

    plt.imshow(data)

    # plt.xlim([0, n])
    # plt.ylim([0, n])
    # plt.xlabel("Iterations")
    # plt.ylabel("Loss")
    # plt.legend(['Training loss', 'Validation loss'])


def go(protocol, network):

    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(0)

    net = caffe.Net(protocol, network, caffe.TEST)

    # the parameters are a list of [weights, biases]
    filters = net.params['conv1'][0].data
    plt.figure(1)
    vis_square(filters.transpose(0, 2, 3, 1))

    filters = net.params['conv2'][0].data
    plt.figure(2)
    vis_square(filters[:64].reshape(64**2, 5, 5))


    return



if __name__ == "__main__":

    # if len(sys.argv) == 1:
    #     print "You need to specify what type of detection you want to run"
    #     print "Options:"
    #     print "\tt\t temporal domain"
    #     print "\ts\t spatial domain"
    #     print "Additionally you can specify second paramters"
    #     print "\tv\t test run - doesn't write out results to output files"
    #
    #     sys.exit(0)
    #
    #
    network_iter = sys.argv[1]
    #network_iter = "61000"

    go('deploy.prototxt', 'snapshot_iter_'+network_iter+'.caffemodel')

    plt.show(block=True)


