#!/usr/bin/python


import os
import io
import sys
import random
from decorator import append
from matplotlib.pyplot import cla
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
from multiprocessing import Pool
from skimage.util.shape import view_as_blocks
import glob
import gc

if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set, exiting'
    sys.exit()

dar_root = os.environ['DAR_ROOT']
app_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

sys.path.insert(0, dar_root + '/python')
import caffe

import cv2




def load_flow_img(filename, rows, cols):

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    of_y = (img.shape[0]-rows) / 2.0
    of_x = (img.shape[1]-cols) / 2.0
    img_dst = img[of_y:of_y + rows, of_x:of_x + cols]

    return img_dst-128


caffe.set_device(1)
caffe.set_mode_gpu()

net_path = '/home/jiri/Lake/DAR/projects/DAR/temporal_net/'
net_path = ''
net = caffe.Net(net_path+'deploy.prototxt', net_path+'snapshot_iter_1.caffemodel', caffe.TEST)

flow_path = dar_root+'/share/datasets/UCF-101//ucf101_flow_img_tvl1_gpu/'
val_list_filename = dar_root+'/share/datasets/UCF-101/train-1-rnd.txt'
val_list_filename = dar_root+'/share/datasets/UCF-101/val-1-rnd.txt'
video_files = np.loadtxt(val_list_filename, str, delimiter=' ')

video_labels = [ f[2] for f in video_files]
video_dirs = [ flow_path+f[0] for f in video_files]


T = 25
N = 6
channels = 20
rows=224
cols=224

stride = 5
sampling = 5
min_frames = N*stride+sampling*T

video_id = 9

count =0
tp = 0
while True:

    # Load a list of video frames for a given video
    video_id+=1
    video_dir = video_dirs[video_id]
    label = video_labels[video_id]

    frames_tmpl = '%s/flow_x_????.jpg' %(video_dir)
    flow_x_frames = glob.glob(frames_tmpl)
    flow_x_frames.sort()

    frames_tmpl = '%s/flow_y_????.jpg' %(video_dir)
    flow_y_frames = glob.glob(frames_tmpl)
    flow_y_frames.sort()

    if len(flow_x_frames) < min_frames:
        continue

    print "Loading video ",video_dir,"  label=",label,"  frames=",len(flow_x_frames)


    # Load a video frames to a batch with the TxNxD organization
    # T=timesteps, N=num of sequences D=data blob

    # Channel organization is x,y,x,y,x,y....
    batch = np.zeros((N,T,channels,rows, cols), dtype=np.float32)

    fid=0
    for n in range(N):
        for t in range(T):
            p = fid + t*sampling
            for c in range(channels/2):
                img_x = load_flow_img(flow_x_frames[p+c], rows, cols)
                img_y = load_flow_img(flow_y_frames[p+c], rows, cols)

                batch[n,t,c*2] = img_x
                batch[n,t,c*2+1] = img_y

        # Skip some frame to start next sequence
        fid += stride


    seq_len = N*T
    cont = np.ones((seq_len,), dtype=np.float32)
    cont = cont[:, np.newaxis]
    cont[::T] = 0

    labels = np.array([label]*seq_len, dtype=np.float32)

    # Flatten from a row major representation
    batch = batch.reshape((N, T, channels)+img_x.shape).transpose(1,0,2,3,4).reshape((seq_len, channels)+img_x.shape)
    cont = cont.reshape((N, T, 1)).transpose(1,0,2).reshape((seq_len, 1))
    labels = labels.reshape(N, T, 1).transpose(1,0,2).reshape((seq_len, 1))

    # net.blobs['data'].reshape(25,20,224,224)
    # net.blobs['data'].data[...] = batch[0]

    # Load data to CNN and do forward pass
    net.blobs['data'].reshape(seq_len, channels, rows, cols)
    net.blobs['data'].data[...] = batch

    net.blobs['cont_label'].reshape(seq_len,1)
    net.blobs['cont_label'].data[...] = cont

    net.blobs['target_label'].reshape(seq_len,1)
    net.blobs['target_label'].data[...] = labels

    # run net and take argmax for prediction
    result = net.forward()


    # Process the result
    # ac = result['prob']
    # print ac.argmax(axis=1)

    ac = result['accuracy']
    out = net.blobs['softmax'].data[0][0].argmax(axis=0)
    # out = net.blobs['softmax'].data.sum(axis=0).sum(axis=0).argmax()
    # out = net.blobs['softmax'].data.argmax(axis=2)
    # print out

    ps = out == int(labels[0,0])

    if ps:
        tp+=1
    count += 1

    acc = 100.0*tp/count

    print "Accuracy: ",ac,"  label=",out, "  ground truth=",labels[0], "  Total accuracy=", acc
