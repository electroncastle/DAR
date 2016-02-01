#!/usr/bin/python

__author__ = "Jiri Fajtl"
__email__ = "ok1zjf@gmail.com"
__status__ = "Research"
__license__ = "LGPL"
__date__ = "20/10/2015"
__version__ = "1.0.0"

import os
import sys
import argparse
from cv2 import trace
from atk import focus_tracker_notify
from cherrypy.lib.sessions import save
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re, fileinput, math
import glob
import time
from scipy import stats

import cv2

import h5py

from videoprocessor import VideoProcessor

# if 'DAR_ROOT' not in os.environ:
#     print 'FATAL ERROR. DAR_ROOT not set, exiting'
#     sys.exit()

dar_root = os.environ['DAR_ROOT']
app_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
# dar_root = '/home/jiri/Lake/DAR/src/caffe-fcn/'
# dar_root = '/home/jiri/Lake/DAR/src/caffe/'
#dar_root = '/home/jiri/Lake/DAR/src/caffe-recurrent/bin/'
#os.environ['DAR_ROOT'] = dar_root

sys.path.insert(0, dar_root + '/python')
import caffe

#dar_root = os.environ['DAR_ROOT']
dar_root = '/home/jiri/Lake/DAR/'
sys.path.insert(0, dar_root + '/projects/optical_flow_regression')
import utils

import videoprocessor

class ActionDetectorConfiguration:
    def __init__(self):
        self.type_name = "temporal"
        self.net_path = dar_root+'/projects/DAR/spatial_lstm_net-full/'
        self.detections_filename = '/spatial-s8.txt'
        self.flow=True
        self.timesteps=25
        self.streams=6
        self.stride=12
        self.padded=True
        self.sampling=5
        self.channels=20
        self.npy = False


class ActionDetector:

    def __init__(self):
        self.net = None

        return

    def network_init(self, path, iter, lstm_seq_length, gpuid):

        caffe.set_device(int(gpuid))
        caffe.set_mode_gpu()
        # caffe.set_mode_cpu()

        self.net = caffe.Net(
            path+'deploy-'+str(lstm_seq_length)+'.prototxt',
            path+'snapshot_iter_'+str(iter)+'-'+str(lstm_seq_length)+'.caffemodel',
            caffe.TEST)

        return




class ActionDetectorLSTM(ActionDetector):

    def __init__(self):
        ActionDetector.__init__(self)
        self.acummulator_lstm = None
        self.acummulator_frame = None
        self.cont = None
        self.stride = -1    # stride of LSTM sequences
        self.sampling = -1  # video frame sampling (default 5)

        self.seq_video_lstm = None
        self.seq_video_frame = None
        self.frame_activations = None


    def reset(self):
        self.cont = None
        self.acummulator_lstm = None
        self.acummulator_frame = None
        self.seq_video_lstm = None
        self.seq_video_frame = None
        self.frame_activations = None
        self.video_dir = ''


    def detect(self, data):

        batch = data.batch_data
        cont = data.continuation_markers
        labels = data.labels
        self.stride = data.stride
        self.sampling = data.sampling
        self.video_dir = data.data_root_dir

        seq_len = batch.shape[0]
        channels = batch.shape[1]
        rows = batch.shape[2]
        cols = batch.shape[3]

        # Load data to CNN and do forward pass
        self.net.blobs['data'].reshape(seq_len, channels, rows, cols)
        self.net.blobs['data'].data[...] = batch

        self.net.blobs['cont_label'].reshape(seq_len,1)
        self.net.blobs['cont_label'].data[...] = cont

        self.net.blobs['target_label'].reshape(seq_len,1)
        self.net.blobs['target_label'].data[...] = labels

        # run net and take argmax for prediction
        result = self.net.forward()

        cont = self.net.blobs['cont_label-reshape'].data
        softmax = self.net.blobs['softmax'].data

        # Remove the last dummy class
        softmax = softmax[:,:,:101]
        #print softmax.argmax(axis=2)

        softmax_frame = None
        if self.net.blobs.has_key('softmax-frame'):
            softmax_frame = self.net.blobs['softmax-frame'].data.reshape((softmax.shape))

        #softmax = (softmax +softmax_frame)/2.0
        #softmax = softmax*cont[:,:,np.newaxis]

        # Add the detections to accumulator
        if self.acummulator_lstm is None:
            self.acummulator_lstm = softmax.copy()
        else:
            self.acummulator_lstm = np.hstack([self.acummulator_lstm, softmax])

        # This is necessary to record which timestamps are padded
        # The padded timesteps have continuity set to zero
        # Note also the first sequence timestep has continuity set to zero
        if self.cont is None:
            self.cont = cont.copy()
        else:
            self.cont = np.hstack([self.cont, cont])


        if softmax_frame is not None:
            if self.acummulator_frame is None:
                self.acummulator_frame = softmax_frame.copy()
            else:
                self.acummulator_frame = np.hstack([self.acummulator_frame, softmax_frame])


        # Process the result
        # ac = result['prob']
        # print ac.argmax(axis=1)

        # ac = result['accuracy']
        # out = net.blobs['softmax'].data[0][0].argmax(axis=0)

        return softmax



    def classify(self):

        self.seq_video_frame = None

        if self.acummulator_lstm is None:
            return -1, -1, -1

        # Put more weight to activations at the later timesteps since
        # the LSTM becomes more confident at the end of the sequence

        # Create a linear ramp window
        T = self.acummulator_lstm.shape[0]
        self.fusion_weights = np.linspace(0.2, 1.0,  num=T)[:, np.newaxis]
        self.fusion_weights /= self.fusion_weights.sum()

        # Create a box window
        self.fusion_weights = np.zeros((T, 1), dtype=np.float32)
        self.fusion_weights[T/2:] = 1

        fw = np.repeat(self.fusion_weights,  self.acummulator_lstm.shape[1], axis=1)[:, :, np.newaxis]

        #scaled = np.multiply(self.acummulator_lstm, fw)

        # Fuse all activations from all augmented detections
        class_id_lstm = self.acummulator_lstm.sum(axis=1).sum(axis=0).argmax(axis=0)

        class_id_frame = -1
        if self.acummulator_frame is not None:
            class_id_frame = self.acummulator_frame.sum(axis=1).sum(axis=0).argmax(axis=0)


        # Fuse all sequences with the per-frame predictions
        # Take care of the paddings
        seq_num = self.acummulator_lstm.shape[1]

        seq_video = None
        for i in range(seq_num):

            fid = i*self.stride
            seq_lstm = self.acummulator_lstm[:, i]
            if self.acummulator_frame is not None:
                seq_frame = self.acummulator_frame[:, i]
            seq_cont = self.cont[:, i]

            # Trim the padded timesteps
            end = np.nonzero(seq_cont[1:]==0)
            if (len(end[0]) > 0):
                end = end [0][0]+1
                seq_lstm = seq_lstm[:end]
                if self.acummulator_frame is not None:
                    seq_frame = seq_frame[:end]

            # Fuse overlapping sequences
            if i == 0:
                self.seq_video_lstm = seq_lstm.copy()
                if self.acummulator_frame is not None:
                    self.seq_video_frame = seq_frame.copy()

            else:
                #if self.acummulator_frame is not None:
                if self.seq_video_lstm is not None:
                    fragment = len(self.seq_video_lstm) - fid
                    #print fid, '  ', len(self.seq_video_frame),' ',len(seq_frame)

                    if fragment > 0:
                        # Sequences overlap
                        self.seq_video_lstm = np.vstack( [self.seq_video_lstm, seq_lstm[fragment:]])
    #                    self.seq_video_frame[fid:] = (self.seq_video_frame[fid:] + seq_frame[:fragment])/2.0

                        if self.acummulator_frame is not None:
                            self.seq_video_frame = np.vstack( [self.seq_video_frame, seq_frame[fragment:]])
                    else:
                        # The sequence is shorter than the stride.
                        # The segments don't overlap
                        # This happends at the end of file so ignore that for now
                        #print "Eof reached"
                        pass
                else:
                    ## Disregard per-frame order here and use only the laset timestep from the LSTM predictions
                    print "TODO"
                    sys.exit()
                    self.seq_video_lstm = np.vstack( [self.seq_video_lstm, seq_lstm[-1]])


        if self.acummulator_frame is not None:
            self.frame_activations = (self.seq_video_lstm+self.seq_video_frame)/2.0
        else:
            self.frame_activations = self.seq_video_lstm

        class_id = self.frame_activations.sum(axis=0).argmax()
       # print "=>",class_id

        #class_id = self.seq_video_frame.sum(axis=0).argmax()

        #class_id = self.acummulator_frame.sum(axis=1).sum(axis=0).argmax(axis=0)
        # print fused.argmax(axis=1)
        # print class_id

        return class_id, class_id_lstm, class_id_frame


    def save_class_by_frame(self, filename, save_lstm=False, save_frame=False):

        if filename == "":
            "NOT SAVING RESULT !!"
            return

        if self.frame_activations is None:
            return

        fout = open(filename, 'wt')

        for i in range(self.frame_activations.shape[0]):
            record = str(i*self.sampling)+' '+' '.join(['%.10f' % num for num in self.frame_activations[i]])

            if save_lstm:
                record += ' '+' '.join(['%.10f' % num for num in self.seq_video_lstm[i]])

            if save_frame and self.seq_video_frame is not None:
                record += ' '+' '.join(['%.10f' % num for num in self.seq_video_frame[i]])

            fout.write(record+'\n')
            fout.flush()

        fout.close()


#-----------------------------------------------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------------------------------------------
def test_flow():

    vp = VideoProcessor()
    vp.set_default_dataset(vp.DATASET_UCF101)
    video_list = vp.parse_video_list(vp.validation_split1_file)

    acd = ActionDetectorLSTM()

    net_path = '/home/jiri/Lake/DAR/projects/DAR/temporal_lstm_net-full/'
# net_path = ''
# net = caffe.Net(net_path+'deploy.prototxt', net_path+'snapshot_iter_1.caffemodel', caffe.TEST)
#         path = '/home/jiri/Lake/DAR/projects/DAR/ALSTM_VGG16-1-2/'

    acd.network_init(net_path, iter=1, gpuid=1)

    tp=0
    c=0
    stride = 12
    video_id = 0
    fout = open('class_result.txt','a')
    for video_dir, frames_num, video_label in video_list:

        vp.load_video(video_dir, video_label)
        vp.reset_lstm_batch(flow=True, timesteps=25, streams=6, stride=stride, padded=True, sampling=5, channels=20)


        acd.reset(stride=stride, sampling=5)
        while True:
            batch, cont, labels = vp.get_next_lstm_batch()
            if batch is None:
                break

            #print "Detecting ",batch.shape
            result = acd.detect(batch, cont, labels)

            #print "--------------------------------------"
            #print result.argmax(axis=2)

            # out = acd.net.blobs['softmax'].data[0][0].argmax(axis=0)
            # print out

        class_id = acd.classify()
        acd.save_class_by_frame('frames.txt')


        status = 'FAILED'
        if class_id > -1:
            if class_id == int(video_label):
                tp += 1
                status = '      '
            c+=1
            acc = 100.0*tp/c

        rec = str(video_id)+' '+str(round(acc,2))+' '+status+' '+str(video_label)+"=>"+str(class_id)+' '+video_dir
        video_id +=1

        print rec
        fout.write(rec+'\n')
        fout.flush()

    return


def test_rgb():

    vp = VideoProcessor()
    vp.set_default_dataset(vp.DATASET_UCF101)
    video_list = vp.parse_video_list(vp.validation_split1_file)

    acd = ActionDetectorLSTM()

    #net_path = '/home/jiri/Lake/DAR/projects/DAR/temporal_lstm_net-full/'
    net_path = '/home/jiri/Lake/DAR/projects/DAR/spatial_lstm_net-full/'

# net_path = ''
# net = caffe.Net(net_path+'deploy.prototxt', net_path+'snapshot_iter_1.caffemodel', caffe.TEST)
#         path = '/home/jiri/Lake/DAR/projects/DAR/ALSTM_VGG16-1-2/'

    acd.network_init(net_path, iter=1, gpuid=1)

    tp=0
    tp_lstm=0
    tp_frame=0
    c=0
    stride = 9      # LSTM sequence stride
    sampling = 5    # Video frames sampling - every X frame is selected as a timestamp
    video_id = 0
    fout = open('class_result.txt','a')
    for video_dir, frames_num, video_label in video_list:

        vp.load_video(video_dir, video_label)

        # The spatial network has 18 timesteps

#        vp.reset_lstm_flow_batch(timesteps=25, streams=6, stride=stride, padded=True, sampling=5)
        vp.reset_lstm_batch(flow=False, timesteps=18, streams=6, stride=stride, padded=False, sampling=sampling, channels=3)


        acd.reset(stride=stride, sampling=5)
        while True:
            batch, cont, labels = vp.get_next_lstm_batch()
            if batch is None:
                break

            #print "Detecting ",batch.shape
            result = acd.detect(batch, cont, labels)

            #print "--------------------------------------"
            #print result.argmax(axis=2)

            # out = acd.net.blobs['softmax'].data[0][0].argmax(axis=0)
            # print out

        class_id, class_id_lstm, class_id_frame = acd.classify()
        acd.save_class_by_frame('frames.txt')


        status = 'FAILED'
        if class_id > -1:
            if class_id == int(video_label):
                tp += 1
                status = '      '

            if class_id_lstm == int(video_label): tp_lstm+=1
            if class_id_frame == int(video_label): tp_frame+=1


            c+=1
            acc = 100.0*tp/c

        rec = str(video_id)+' '+str(round(acc,2))+'\t'+str(round(100.0*tp_lstm/c, 2))+'\t'+str(round(100.0*tp_frame/c, 2))+'\t'+status+' '+str(video_label)+"=>("+str(class_id)+' '+str(class_id_lstm)+' '+str(class_id_frame)+') '+video_dir
        video_id +=1

        print rec
        fout.write(rec+'\n')
        fout.flush()

    return


if __name__ == "__main__":
    test_flow()
    test_rgb()