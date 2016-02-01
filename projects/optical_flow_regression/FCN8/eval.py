import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set, exiting'
    sys.exit()

dar_root = os.environ['DAR_ROOT']
app_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
dar_root = '/home/jiri/Lake/DAR/src/caffe-fcn/'
dar_root = '/home/jiri/Lake/DAR/src/caffe/'
sys.path.insert(0, dar_root + '/python')
import caffe


if 1:
    gpuid = 0
    caffe.set_mode_gpu()
    caffe.set_device(gpuid)
else:
    caffe.set_mode_cpu()

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
img_path = '/home/jiri/Lake/DAR/share/datasets/VOC2010/VOCdevkit/VOC2010/JPEGImages/2007_000129.jpg'
#img_path='/home/jiri/Lake/DAR/share/datasets/THUMOS2015/thumos15_validation-rgbflow/thumos15_video_validation_0000122/image_0659.jpg'


net = None

def segment(img_path):
    global net

    im = Image.open(img_path)

#    im = cv2.resize(np.array(im), (480,480))

    in_ = np.array(im, dtype=np.float32)

    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    result = net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)


    imout = result['score'][0,15]
    dir, filename = os.path.split(img_path)
    plt.imsave(filename+'-dtc.jpg', imout)
    ims = np.asarray_chkfinite(im)
    plt.imsave(filename, ims)
    # cv2.normalize(imout, outrgb, )
    #cv2.imshow('out', imout)
    # cv2.imwrite(img_path+'dtc.jpg', imout)
    #cv2.waitKey(0)

    #return
    print out

    plt.subplot(1, 2, 1)
    #plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
    plt.imshow(im)
    plt.subplot(1, 2, 2)

    #plt.imshow( result['score'][0,2]) # bicycle
    plt.imshow( result['score'][0,15]) # person
    #plt.imshow( result['score'][0,37]) # ground



os.chdir('/home/jiri/Lake/DAR/projects/optical_flow_regression/FCN8/')

# load net
net = caffe.Net(
#                'deploy-orig.prototxt',
                'deploy.prototxt',
                #'deploy.prototxt',
                'fcn-8s-pascalcontext.caffemodel',
#                'fcn-32s-pascalcontext.caffemodel',
#                'snapshot_iter_2000.caffemodel',
#                'snapshot_iter_10001.caffemodel',
                caffe.TEST)

img_list = []
if 1:
    root = '/home/jiri/Lake/DAR/share/datasets/THUMOS2015/thumos15_validation-rgbflow/thumos15_video_validation_0000122/'

    img_list = ['image_0659.jpg',
                'image_0660.jpg',
                'image_0661.jpg',
                'image_0662.jpg',
                'image_0663.jpg',
                'image_0664.jpg'];

    img_list = []
    for i in range(659, 700):
        img = 'image_{:0>4d}.jpg'.format(i)
        img_path = os.path.join(root, img)
        img_list.append(img_path)
else:
    img_list.append(img_path)

img = '/home/jiri/Lake/DAR/share/datasets/UCF-101//UCF101-rgbflow//BreastStroke/v_BreastStroke_g05_c04/image_0079.jpg'
segment(img)
plt.waitforbuttonpress()
sys.exit()

print 'Start ',len(img_list), '  images'
for img in img_list:
    segment(img)
    # plt.waitforbuttonpress()

print 'Stop'

#plt.show(block=True)