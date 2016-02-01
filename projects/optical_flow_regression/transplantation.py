#!/usr/bin/python

__author__ = "Jiri Fajtl"
__email__ = "ok1zjf@gmail.com"
__status__ = "Research"
__license__ = "LGPL"
__date__ = "20/10/2015"
__version__ = "1.0.0"



from matplotlib.pyplot import show
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2

#matplotlib inline
import Image

if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set, exiting'
    sys.exit()

dar_root = os.environ['DAR_ROOT']
app_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

#dar_root  ='/home/jiri/Lake/DAR/src/caffe-fcn/'
dar_root  ='/home/jiri/Lake/DAR/src/caffe/'
sys.path.insert(0, dar_root + '/python')
import caffe

# configure plotting
plt.rcParams['figure.figsize'] = (15, 15)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def upsample_filt(size):
    factor = (size + 1.0) // 2
    if size % 2 == 1:
        center = factor - 1.0
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1.0 - abs(og[0] - center) / factor) * \
           (1.0 - abs(og[1] - center) / factor)



# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def config_deconf_layer(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        filt /= np.linalg.norm(filt)
        net.params[l][0].data[range(m), range(k), :, :] = filt

        print "------------------------------------"
        print l
        print filt

    return


def expand_data_channels(src_net, dst_net, multiplier):

    # Vector of biases, one for each filter
    # e.g. shape (64,)
    biases = src_net.params['conv1_1'][1].data

    # Filters [batch, channels, height, width]
    # e.g. shape (64, 3, 3, 3)
    weights = src_net.params['conv1_1'][0].data

    # Duplicate RGB channels twice as we are gonna feed the CNN
    # with two images
    new_weights = None
    zero = np.zeros((3,3,3))

    if 1:
        for f in range(0, weights.shape[0]):
            channel_weights = np.append(weights[f], weights[f], axis=0)        
            if new_weights == None:
                new_weights = np.array([channel_weights])
            else:
                new_weights = np.append(new_weights, [channel_weights], axis=0)
    else:
        for f in range(0, weights.shape[0]):
            channel_weights = np.append(weights[f  ], zero, axis=0)
            if new_weights == None:
                new_weights = np.array([channel_weights])
            else:
                new_weights = np.append(new_weights, [channel_weights], axis=0)


        for f in range(0, weights.shape[0]):
            channel_weights = np.append(zero, weights[f], axis=0)
            new_weights = np.append(new_weights, [channel_weights], axis=0)

    print "New weights: ", new_weights.shape
    print "Target shape: ", dst_net.params['conv1'][0].data.shape

    # Set new weights
    dst_net.params['conv1'][0].data[...] = new_weights

    # Set new biases
    # dst_net.params['conv1'][1].data[...] = np.append(biases.copy(), biases.copy())
    dst_net.params['conv1'][1].data[...] = biases.copy()

    return


def expand_data_channels_alex(src_net, dst_net, multiplier, new_size):

    # Vector of biases, one for each filter
    # e.g. shape (64,)
    biases = src_net.params['conv1'][1].data

    # Filters [batch, channels, height, width]
    # e.g. shape (64, 3, 3, 3)
    weights = src_net.params['conv1'][0].data

    bw_dst = False
    if  dst_net.params['conv1'][0].data.shape[1] == 2:
        # There are only two input channels it must be monochrome type
        bw_dst = True
        pass

    # Duplicate RGB channels twice as we are gonna feed the CNN
    # with two images
    new_weights = None
    zero = np.zeros((3,11,11))

    if 1:
        for f in range(0, weights.shape[0]):
            new_filter = cv2.resize(weights[f].transpose(1,2,0), (new_size, new_size)).transpose(2,0,1)
            if bw_dst:
                channel_weights = new_filter.mean(axis=0)
#                channel_weights /= channel_weights.sum()
                channel_weights = channel_weights[np.newaxis, :, :]
                channel_weights = np.append(channel_weights, channel_weights, axis=0)
            else:
                channel_weights = np.append(new_filter, new_filter, axis=0)
    #            channel_weights = new_filter.copy()


#            channel_weights = np.append(weights[f], weights[f], axis=0)
            if new_weights == None:
                new_weights = np.array([channel_weights])
            else:
                new_weights = np.append(new_weights, [channel_weights], axis=0)
    else:
        for f in range(0, weights.shape[0]):
            channel_weights = np.append(weights[f  ], zero, axis=0)
            if new_weights == None:
                new_weights = np.array([channel_weights])
            else:
                new_weights = np.append(new_weights, [channel_weights], axis=0)


    	for f in range(0, weights.shape[0]):
        	channel_weights = np.append(zero, weights[f], axis=0)
        	new_weights = np.append(new_weights, [channel_weights], axis=0)

    print "new wieghts: ",new_weights.shape
    print "destination: ",dst_net.params['conv1'][0].data.shape

    # Set new weights
    dst_net.params['conv1'][0].data[...] = new_weights

    # Set new biases
    # dst_net.params['conv1'][1].data[...] = np.append(biases.copy(), biases.copy())
    dst_net.params['conv1'][1].data[...] = biases.copy()

    return


def fc_to_conv(src_net, dst_net):

    params = ['fc6', 'fc7', 'fc8']
    # fc_params = {name: (weights, biases)}
    fc_params = {pr: (src_net.params[pr][0].data, src_net.params[pr][1].data) for pr in params}

    print "Source network:"
    for fc in params:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)


#     params_full_conv_ = ['fc6-conv', 'fc7-conv', 'fc8-conv']
    params_full_conv_ = ['fc6-conv', 'fc7-conv']
    params_full_conv = []
    # check params
    for pr in params_full_conv_:
        if pr in dst_net.params:
            print "Converting to FC: => ",pr
            params_full_conv.append(pr)
        else:
            print "Layer ",pr,' does not exist in target networ. Skipping...',

    # conv_params = {name: (weights, biases)}
    conv_params ={}
     #{pr: (dst_net.params[pr][0].data, dst_net.params[pr][1].data) }
    for pr in params_full_conv:
        conv_params[pr] = (dst_net.params[pr][0].data, dst_net.params[pr][1].data)


    print "Destination network:"
    for conv in params_full_conv:
        if conv in dst_net.params:
            print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

    # The convolution weights are arranged in output x input x height x width dimensions.
    # To map the inner product weights to convolution filters, we could roll the flat inner product
    # vectors into channel x height x width filter matrices, but actually these are identical in memory
    # (as row major arrays) so we can assign them directly.
    #
    # The biases are identical to those of the inner product.

    # Transplant only the fc6 and fc7
    for pr, pr_conv in zip(params, params_full_conv):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]

    return


def copy_layers(new, src_net, layers):

    for l in layers:
        net.params[l][0].data[...] = src_net.params[l][0].data.copy()
        net.params[l][1].data[...] = src_net.params[l][1].data.copy()

    return


# helper show filter outputs
def show_filters(net):

    plt.figure()
    c1 = net.blobs['conv1'].data
    filt_min, filt_max = net.blobs['conv1'].data.min(), net.blobs['conv1'].data.max()
    for r in range(8):
        for c in range(8):
            plt.subplot(8,8,r*8+c+1)
            #plt.title("filter #{} output".format(i))
            plt.imshow(net.blobs['conv1'].data[0, r*8+c], vmin=filt_min, vmax=filt_max)

            #plt.tight_layout()
            plt.axis('off')


def test_net(net):

    width = 224
    height = 224
    meanImg = np.array([[np.transpose([104.0 for i in range(width)])]*height,
                       [np.transpose([117.0 for i in range(width)])]*height,
                       [np.transpose([123.0 for i in range(width)])]*height], dtype='float32')


    # load image and prepare as a single input batch for Caffe
    #im = np.array(Image.open('/home/jiri/Lake/DAR/src/caffe/examples/images/cat_gray.jpg'))
    im = np.array(Image.open('/home/jiri/Lake/DAR/src/caffe/examples/images/cat.jpg'))
    plt.title("original image")
    plt.imshow(im)
    plt.axis('off')

    gray = False
    multichannel = True
    channels = 6
    if gray:
        im_input = im[np.newaxis, np.newaxis, :, :]
    else:
        im = cv2.resize(im, (224, 224))
        im = im[:,:,::-1].transpose((2,0,1))
        im = np.subtract(im, meanImg)

        if multichannel:
            batch = []
            for c in range(0, channels/3):
                batch.append(im[0,:,:])
                batch.append(im[1,:,:])
                batch.append(im[2,:,:])

            batch = np.asarray(batch)
            im_input = batch[np.newaxis, :, :]
        else:
            im_input = im[np.newaxis, :, :]


    # net.blobs['data'].reshape(*im_input.shape)
    # net.blobs['data'].data[...] = im_input

    new_net.blobs['data'].reshape(*im_input.shape)
    new_net.blobs['data'].data[...] = im_input

    net.forward()


def go(src_net):

    # Load the net, list its data and params, and filter an example image.
    caffe.set_device(1)
    caffe.set_mode_gpu()
    net = None

    # VGG16
    net = caffe.Net('/home/jiri/Lake/DAR/projects/optical_flow_regression/models-proto/VGG_CNN_M_2048_OFR/VGG_ILSVRC_16_layers_deploy.prototxt',
                     '/home/jiri/Lake/DAR/projects/optical_flow_regression/models-proto/VGG_CNN_M_2048_OFR/VGG_ILSVRC_16_layers.caffemodel',
                     caffe.TEST)

    # Alex Next
    # net = caffe.Net('/home/jiri/Lake/DAR/src/caffe/models/bvlc_reference_caffenet/deploy.prototxt',
    #                 '/home/jiri/Lake/DAR/src/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet_new.caffemodel',
    #                 caffe.TEST)

#    net.save('/home/jiri/Lake/DAR/src/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet_new.caffemodel')

    #net = caffe.Net('/home/jiri/Lake/DAR/projects/UCF101-2stream/models-proto/two-streams-nvidia/vgg_16_flow_deploy.prototxt',
    #'/home/jiri/Lake/DAR/projects/UCF101-2stream/models-bin/cuhk_action_temporal_vgg_16_split1.caffemodel',
    #                caffe.TEST)

    # Load new protocol with the target format.
    # The modified layers must have different names not to conflict with the existing
    # ones in the pretrained caffemodel
    # new_net = caffe.Net('OFR_VGG16_6_deploy.prototxt',
    #             '/home/jiri/Lake/DAR/projects/optical_flow_regression/OFR_VGG16_6_iter_0.caffemodel',
    #             caffe.TEST)




    if src_net=="":
        new_net = caffe.Net('deploy.prototxt', caffe.TEST)
    else:
        new_net = caffe.Net('deploy.prototxt',
#                            src_net,
                        '/home/jiri/Lake/DAR/projects/optical_flow_regression/models-proto/VGG_CNN_M_2048_OFR/VGG_ILSVRC_16_layers.caffemodel',
    #                      'snapshot_iter_13000.caffemodel',
                        #'/home/jiri/Lake/DAR/projects/optical_flow_regression/OFR_VGG16_6_fc_iter_0.caffemodel',
                    caffe.TEST)


    #print new_net.params['upscore'][0].data[range(2), range(2), :, :]

    if net!=None:
        print "Source network"
        print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

    print "Target network"
    print("blobs {}\nparams {}".format(new_net.blobs.keys(), new_net.params.keys()))


    # The new model will have twice the number of channels as the original
    #expand_data_channels_alex(net, new_net, 2, 7)
    expand_data_channels(net, new_net, 2)
    #copy_layers(net, new_net, ['conv1_2'])

    if 1:
        # For deconvolution
        #fc_to_conv(net, new_net)

        # Preload bilinear interpolation filter to all layers with 'up' in name
        interp_layers = [k for k in new_net.params.keys() if 'up' in k]
        #interp_layers =['upsample-final-28']
        print interp_layers
        config_deconf_layer(new_net, interp_layers)

    # Save new model
    new_net.save('snapshot_iter_0.caffemodel')
    #new_net.save('vgg16fc.caffemodel')
    print "Done"

    # test_net(net)
    # show_filters(new_net)


    # Block execution
    # show(block=True)


if __name__ == "__main__":
    cwd = os.getcwd()
#     os.chdir("/home/jiri/Lake/DAR/projects/optical_flow_regression/OFR_VGG16_6_fc_x8")
#     os.chdir("/home/jiri/Lake/DAR/projects/optical_flow_regression/FCN32")
#     os.chdir("/home/jiri/Lake/DAR/projects/optical_flow_regression/OFR_VGG16_6_fc14")
# #     os.chdir("/home/jiri/Lake/DAR/projects/optical_flow_regression/OFR_VGG16_6_hist")
#     os.chdir("/home/jiri/Lake/DAR/projects/optical_flow_regression/OFR_VGG16_6_hist-1")
#     os.chdir("/home/jiri/Lake/DAR/projects/optical_flow_regression/OFR_VGG16_6_hist-1")

    src_net=""
    if len(sys.argv) > 1:
        src_net=sys.argv[1]

    go(src_net)


    os.chdir(cwd)


