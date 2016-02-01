#!/usr/bin/python

#import numpy as np
import sys
import os

if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set, exiting'
    sys.exit()

dar_root = os.environ['DAR_ROOT']
app_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

#dar_root  ='/home/jiri/Lake/DAR/src/caffe-fcn/'
#dar_root  ='/home/jiri/Lake/DAR/src/caffe/'
sys.path.insert(0, dar_root + '/python')
import caffe


def copy_params(src_net, dst_net, blob_name):

    print "Copying weights and biases for: ",blob_name
    # Vector of biases, one for each filter
    # e.g. shape (64,)
    biases = src_net.params[blob_name][1].data

    # Filters [batch, channels, height, width]
    # e.g. shape (64, 3, 3, 3)
    weights = src_net.params[blob_name][0].data

    print "Biases shape: ",biases.shape
    print "Weights shape: ",weights.shape

    # Set new weights
    dst_net.params[blob_name][0].data[...] = weights

    # Set new biases
    # dst_net.params['conv1'][1].data[...] = np.append(biases.copy(), biases.copy())
    dst_net.params[blob_name][1].data[...] = biases.copy()


def go(lstm_net_iter, lstm_seq_length):

    # Load the net, list its data and params, and filter an example image.
    caffe.set_mode_gpu()

    src_net_name = 'snapshot_iter_1-flow-stream.caffemodel'
    lstm_net_name = 'snapshot_iter_'+str(lstm_net_iter)+'-lstm-'+str(lstm_seq_length)+'.caffemodel'

    print "Loading source network (VGG16) ",src_net_name
    print "Loading LSTM network ",lstm_net_name

    flow_net = caffe.Net('deploy-'+str(lstm_seq_length)+'.prototxt',
                            src_net_name,
                    caffe.TEST)

    lstm_net = caffe.Net('train_val_fast-lstm-'+str(lstm_seq_length)+'.prototxt',
                            lstm_net_name,
                    caffe.TEST)

    #print new_net.params['upscore'][0].data[range(2), range(2), :, :]

    print "Flow network"
    print("blobs {}\nparams {}".format(flow_net.blobs.keys(), flow_net.params.keys()))

    print "LSTM network"
    print("blobs {}\nparams {}".format(lstm_net.blobs.keys(), lstm_net.params.keys()))

    print "LSTM network shapes"
    print "shape fc7-reshape:  ",lstm_net.blobs['fc7-reshape'].data.shape
    print "shape lstm1:  ",lstm_net.blobs['lstm1'].data.shape
    print "shape predict:  ",lstm_net.blobs['predict'].data.shape


    blob_names = ['lstm1', 'predict']
    for blob_name in blob_names:
        copy_params(lstm_net, flow_net, blob_name)

    new_net_name = 'snapshot_iter_'+str(lstm_net_iter)+'-'+str(lstm_seq_length)+'.caffemodel'
    print "Saving new network to: ", new_net_name
    flow_net.save(new_net_name)

    return


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage:"
        print "build_net.py <iteration_number> <lstm_sequence_lenght"
        sys.exit()

    go(sys.argv[1], sys.argv[2])

