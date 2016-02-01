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
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import re, fileinput, math
import glob
import time
from pprint import pprint
import cv2


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
import actiondetector as ac


class VideoDetectorConfiguration(ac.ActionDetectorConfiguration):

    def __init__(self):
        ac.ActionDetectorConfiguration.__init__(self)
        self.type_name = "temporal"
        self.net_path = dar_root+'/projects/DAR/spatial_lstm_net-full/'
        self.net_iter = 1
        self.detections_filename = '/spatial-s8.txt'
        self.gpuid = 0
        self.dataset ='ucf'

    def __str__(self):
        # return str(self.__class__) + ": " + str(self.__dict__)
        s=''
        for k, v in self.__dict__.iteritems():
            s += '%s: %s\n' % (k, str(v))
        return s



class DetectionRecorder():

    def __init__(self, log_filename, append=True):
        self.tp=0
        self.tp_lstm=0
        self.tp_frame=0
        self.c=0  # Number of successfully processed videos
        self.total=0 # Number of total processed videos
        if append:
            flag = 'a'
        else:
            flag = 'w'
        self.fout = open(log_filename, flag)


    def record_start(self):
        self.total+=1
        self.start = time.time()


    def record_finish(self, label, video_file, class_result):
        self.acc = 0
        self.acc_lstm = 0
        self.acc_frame = 0
        self.end = time.time()
        self.took = (self.end - self.start)

        self.label = int(label)
        self.video_file = video_file
        self.class_result = class_result

        class_id, class_id_lstm, class_id_frame = class_result



        if self.label > -1:
            self.status = 'FAILED'
            if class_id > -1:
                if class_id == self.label:
                    self.tp += 1
                    self.status = 'OK    '

                if class_id_lstm == self.label: self.tp_lstm+=1
                if class_id_frame == self.label: self.tp_frame+=1

                self.c+=1
        else:
            self.status = 'NA    '


        if self.c>0:
            self.acc = 100.0*self.tp/self.c
            self.acc_lstm = 100.0*self.tp_lstm/self.c
            self.acc_frame = 100.0*self.tp_frame/self.c

        rec = str(self.total)+' '+\
              ("%6.2f" % self.acc)+' '+("%6.2f" % self.acc_lstm)+' '+("%6.2f" % self.acc_frame)+' '+\
              self.status+' '+\
              ("%3d" % int(label))+' '+("%3d" % class_id)+' '+("%3d" % class_id_lstm)+' '+("%3d" % class_id_frame)+' '+\
              self.video_file+' '+\
              ("%6.2f" % (self.took))

        print rec
        self.fout.write(rec+'\n')
        self.fout.flush()


    def print_stats(self):
        rec = str(self.total)+' '+\
              ("%6.2f%" % self.acc)+' '+("%6.2f%" % self.acc_lstm)+' '+("%6.2f%" % self.acc_frame)+' '+\
              self.status+' '+\
              ("GT=%3d" % int(self.label))+' '+("%3d" % self.class_id)+' '+("%3d" % self.class_id_lstm)+' '+("%3d" % self.class_id_frame)+' '+\
              ("%6.2f sec" % (self.took))

        print rec



class DetectionAnalyzer:

    def __init__(self, name, fusion_weight, labels=[]):
        self.name = name
        self.fusion_weight = fusion_weight
        self.tp_spatial = 0.0
        self.tp_temporal = 0.0
        self.tp_fused=  0.0
        self.counter = 0.0
        self.labels = labels

        self.class_spatial_accuracy = 0
        self.class_temporal_accuracy = 0
        self.class_fused_accuracy = 0

        self.det_matrix_spatial = np.zeros((101, 101), dtype=np.float32)
        self.det_matrix_temporal = np.zeros((101, 101), dtype=np.float32)
        self.det_matrix_fused = np.zeros((101, 101), dtype=np.float32)

        self.det_matrix_fused_tp = np.zeros((101, 1), dtype=np.float32)
        self.det_matrix_fused_num = np.zeros((101, 1), dtype=np.float32)


    def add(self, temporal, spatial, video_label):
        self.counter += 1

        # Trim the spatial and temporal detection to the same number of frames
        max_frames = min(spatial.shape[0], temporal.shape[0])
        detections_spatial = spatial[:max_frames, :]
        detections_temporal = temporal[:max_frames, :]

        # Calculate average activation across all frames in the video
        detections_spatial = detections_spatial.sum(axis=0)/detections_spatial.shape[0]
        detections_temporal = detections_temporal.sum(axis=0)/detections_temporal.shape[0]

        self.det_matrix_spatial[int(video_label)] += detections_spatial
        self.det_matrix_spatial[int(video_label)] /= 2.0

        self.det_matrix_temporal[int(video_label)] += detections_temporal
        self.det_matrix_temporal[int(video_label)] /= 2.0

        if len(detections_spatial) == 101:
            class_spatial = detections_spatial.argmax()
            if class_spatial == video_label: self.tp_spatial+=1.0

        if len(detections_temporal) == 101:
            class_temporal = detections_temporal.argmax()
            if class_temporal == video_label: self.tp_temporal+=1.0


        self.det_matrix_fused_num[int(video_label)] += 1
        if len(detections_temporal) == 101 and len(detections_temporal) == len(detections_spatial):
            # Calculated weighted average between the spatial and temporal detections
            detections_fused = (detections_spatial*(1.0-self.fusion_weight ) + detections_temporal*self.fusion_weight)
            self.det_matrix_fused[int(video_label)] += detections_fused
            self.det_matrix_fused[int(video_label)] /= 2.0
            class_fused = detections_fused.argmax()

            if class_fused == video_label:
                self.tp_fused+=1
                self.det_matrix_fused_tp[int(video_label)] += 1
        else:
            print "Number of activation in temporal and spatial streams differ !!"


        self.class_spatial_accuracy = 100.0*self.tp_spatial/self.counter
        self.class_temporal_accuracy = 100.0*self.tp_temporal/self.counter
        self.class_fused_accuracy = 100.0*self.tp_fused/self.counter


    def print_result(self):
        print "Detections: ",self.name
        print "spatial / temporal / fused"
        print self.class_spatial_accuracy, ' / ', self.class_temporal_accuracy, ' / ',self.class_fused_accuracy


    def __str__(self):
        return ("%10s" % self.name)+'   '+("%6.2f" % self.class_spatial_accuracy)+'   '+ ("%6.2f" % self.class_temporal_accuracy)+'   '+ ("%6.2f" % self.class_fused_accuracy)



    def rotate(shelf, src, angle):
        len = max(src.shape[1], src.shape[0])
        pt = (len/2., len/2.)
        r = cv2.getRotationMatrix2D(pt, angle, 1.0)
        dst = cv2.warpAffine(src, r, (len, len))
        return dst


    def create_scale_bar(self, width, height):
        # Create scale bar
        gradient_pane = np.zeros((height, width, 3), dtype=np.uint8)
        gradient_pane[:] = 255

        bar_height = 10
        gradient = np.repeat(np.linspace(0, 255, width-100)[np.newaxis,:], bar_height, axis=0)
        imout = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        gradient_clr = cv2.applyColorMap(imout, cv2.COLORMAP_JET)

        xd = (width - gradient.shape[1]) / 2
        yd = 10
        gradient_pane[yd:gradient.shape[0]+yd, xd:gradient.shape[1]+xd, :] = gradient_clr

        font = cv2.FONT_HERSHEY_SIMPLEX
        #font = cv2.FONT_HERSHEY_PLAIN
        step = gradient.shape[1] / 10.0
        for i in range(0, 11):
            x = int(xd+i*step)
            cv2.line(gradient_pane, (x, yd+5), (x, yd+10), (0,0,0), 1)
            cv2.putText(gradient_pane, str(i*10), (x-5, yd+30), font, 0.5, (0, 0, 0), 1)


        xd = int((width - 200) / 2)
        cv2.putText(gradient_pane, "Confidence in %", (xd, 80), font, 0.6, (0, 0, 0), 1)


        return gradient_pane



    def detection_matrix_to_image(self, name, matrix, labels, pixel_size=10):

        create_scale_bar_height = 100

        mat_img_spatial = (matrix*255)
        mat_img_spatial = mat_img_spatial.astype(np.uint8)

        # Scale up the confusion matrix
        pixel_size_w = pixel_size # * matrix.shape[0]/matrix.shape[1]
        pixel_size_h = min(pixel_size * matrix.shape[1]/matrix.shape[0], 30)
        row_height = pixel_size_h
        mat_img_spatial = cv2.resize(mat_img_spatial, (matrix.shape[1]*pixel_size_w, matrix.shape[0]*pixel_size_h),  interpolation=cv2.INTER_NEAREST)


        # Get max width of the labels bar on left
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_font_scale = 0.032*min(row_height, 12)
        max_label_width = 0
        for label in labels:
            extend = cv2.getTextSize(str(label), font, label_font_scale, 1)
            if extend[0][0] > max_label_width:
                max_label_width = extend[0][0]

        #imout = cv2.normalize(mat_img_spatial, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        imout = mat_img_spatial
        imout_clr = cv2.applyColorMap(imout, cv2.COLORMAP_JET)


        # Create list of labels on y axis
        labels_img = np.zeros((imout_clr.shape[0]+create_scale_bar_height, max_label_width+5, 3), dtype=np.uint8)
        labels_img[:] = 255

        # Create an empty white bar on right side
        border_right = np.zeros((imout_clr.shape[0]+create_scale_bar_height, 30, 3), dtype=np.uint8)
        border_right[:] = 255

        # Create list of labels on x top axis
        labels_img_h = np.zeros((60, 30+imout_clr.shape[1]+labels_img.shape[1]+border_right.shape[1], 3), dtype=np.uint8)
        labels_img_h[:] = 255


        # Create labels on X axis
        line_ctrl = (20, 20, 40)
        y = 10
        label_id = 0
        for i in range(matrix.shape[1]):

            if i == 22:
                if matrix.shape[0] <= 22:
                    print matrix[0,22]
                else:
                    print "F ",matrix[22,22]

            yd = labels_img_h.shape[0]-10
            xo = labels_img.shape[1]+30-8
            if label_id % 2 == 0:
                yd -= 10

            x=xo+y-2
            cv2.line(labels_img_h, (x, labels_img_h.shape[0]-5), (x, labels_img_h.shape[0]), line_ctrl, 1)

            if label_id > 9:
                xo -= 2

            cv2.putText(labels_img_h, str(label_id), (xo+y-1, yd), font, 0.32, (0, 0, 0), 1)

            # Vertical lines
            cv2.line(imout_clr, (y, 0), (y, imout_clr.shape[0]), line_ctrl, 1)

            y+=10
            label_id +=1


        # Create labels on Y axis
        line_ctrl = (20, 20, 40)
        y = row_height
        label_id = 0
        for i in range(matrix.shape[0]):
            cv2.putText(labels_img, str(labels[label_id]), (5, y-1), font, label_font_scale, (0, 0, 0), 1)
            cv2.line(labels_img, (labels_img.shape[1]-5, y), (labels_img.shape[1], y), line_ctrl, 1)

            # Horizontal lines
            cv2.line(imout_clr, (0, y), (imout_clr.shape[1], y), line_ctrl, 1)

            label_id +=1
            y+=row_height


        # Composite the whole image
        xd = int((labels_img_h.shape[1] - 100) / 2)
        cv2.putText(labels_img_h, "Predictions", (xd, 18), font, 0.6, (0, 0, 0), 1)

        txtimg = np.zeros((30, labels_img.shape[0], 3), dtype=np.uint8)
        txtimg[:] = 255
        xd = int((txtimg.shape[1]) / 2)
        cv2.putText(txtimg, "True Labels", (xd, 25), font, 0.6, (0,0,0), 1)
        txtimg = txtimg.transpose(1,0,2)         #self.rotate(txtimg, -45)
        txtimg = np.flipud(txtimg)
#        imout_clr[20:20+txtimg.shape[0], y:y+txtimg.shape[1]] = txtimg

        gradient_pane = self.create_scale_bar(imout_clr.shape[1], create_scale_bar_height)
        imout_clr = np.vstack([imout_clr, gradient_pane])

        imout_clr = np.hstack([txtimg, labels_img, imout_clr, border_right])
        imout_clr = np.vstack([labels_img_h, imout_clr])


        cv2.imshow(name, imout_clr)
        cv2.imwrite(name+'.png', imout_clr)

        return mat_img_spatial


    def get_high_var_detections(self, matrix):

        # ind = (matrix*(np.identity(matrix.shape[0])*-1+1)) > 0.15
        ind = (matrix*(np.identity(matrix.shape[0]))) > 0.4
        #ind = ~ind
        ind_nonzero = np.nonzero(ind)[0]

        labels = []
        highvar = None
        for i in range(matrix.shape[0]):
            if i not in ind_nonzero:
                labels.append(self.labels[i])
                if highvar is None:
                    highvar=matrix[i].copy()
                else:
                    highvar = np.vstack([highvar, matrix[i]])

        return highvar, labels


    def normalize(self, matrix):
        sum_temp = np.sum(matrix, axis=1)
        for i in range(matrix.shape[0]):
            if sum_temp[i] > 0:
                matrix[i] = matrix[i]/sum_temp[i]
        return matrix


    def plot_detection_matrix(self, pixel_size=550, name_suffix=''):

        self.det_matrix_spatial = self.normalize(self.det_matrix_spatial)
        self.det_matrix_temporal = self.normalize(self.det_matrix_temporal)

        highvar_temp, labels = self.get_high_var_detections(self.det_matrix_temporal)
        labels_text = [ (l[0]+' '+l[1]) for l in labels]
        mat_img_temporal_highvar = self.detection_matrix_to_image(self.name+'-temporal-var-'+name_suffix, highvar_temp, labels_text, pixel_size=10)

        highvar_spatial, labels = self.get_high_var_detections(self.det_matrix_spatial)
        labels_text = [ (l[0]+' '+l[1]) for l in labels]
        mat_img_spatial_highvar = self.detection_matrix_to_image(self.name+'-spatial-var-'+name_suffix, highvar_spatial, labels_text, pixel_size=10)

        highvar_spatial, labels = self.get_high_var_detections(self.det_matrix_fused)
        labels_text = [ (l[0]+' '+l[1]) for l in labels]
        mat_img_fused_highvar = self.detection_matrix_to_image(self.name+'-fused-var-'+name_suffix, highvar_spatial, labels_text, pixel_size=10)

        labels = np.linspace(0, 100,  num=101)
        labels = labels.astype(np.uint8)
        mat_img_temporal = self.detection_matrix_to_image(self.name+'-temporal-'+name_suffix, self.det_matrix_temporal, labels, pixel_size)
        mat_img_spatial = self.detection_matrix_to_image(self.name+'-spatial-'+name_suffix, self.det_matrix_spatial, labels, pixel_size)
        mat_img_fused = self.detection_matrix_to_image(self.name+'-fused-'+name_suffix, self.det_matrix_fused, labels, pixel_size)


        return [mat_img_spatial, mat_img_temporal]



    def plot_confusion_matrix(self, classes=None):


        if classes is None:
            norm_conf = self.det_matrix_fused
            labels = self.labels
        else:

            norm_conf = np.zeros((classes.shape[0], self.det_matrix_fused.shape[1]), dtype=np.float32)
            labels = []
            for i in range(classes.shape[0]):
                norm_conf[i] = self.det_matrix_fused[int(classes[i,0])]
                labels.append(classes[i,0:2])
                #Print the highest 3 activations
                max_args = np.argsort(-norm_conf[i])[:4]
                print classes[i]
                print max_args
                for m in max_args:
                    ac = norm_conf[i,m]
                    lb = self.labels[m]
                    print m,lb,ac



        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                        interpolation='nearest', vmin=0, vmax=1)

        width = norm_conf.shape[1]
        height = norm_conf.shape[0]

        # for x in xrange(width):
        #     for y in xrange(height):
        #         ax.annotate(str(conf_arr[x][y]), xy=(y, x),
        #                     horizontalalignment='center',
        #                     verticalalignment='center')

        cb = fig.colorbar(res, fraction=0.046, pad=0.01, shrink=0.5, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1], orientation='horizontal')

        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.tick_params(axis='both', which='major', labelsize=9)

        labels_text = [ (l[0]+' '+l[1]) for l in self.labels]
        plt.xticks(np.array(range(width))-0.0, labels_text, rotation='vertical', verticalalignment='bottom', ha='center')
        plt.subplots_adjust(bottom=0.2)

        labels_text = [ (l[1]+' '+l[0]) for l in labels]
        plt.yticks(np.array(range(height)), labels_text, rotation='horizontal', verticalalignment='center', ha='right')

        plt.ylabel('True labels')
        plt.xlabel('Predictions')

        plt.savefig('confusion_matrix.png', format='png')
        plt.show(block=False)
        return



    def plot_recall(self, pixel_size=550, name_suffix=''):

        x = np.linspace(0, 100, 101)
        y = self.det_matrix_fused_tp / self.det_matrix_fused_num
        y=np.nan_to_num(y)

        ind = []
        for i in range(y.shape[0]):
            ind.append([self.labels[i][0], self.labels[i][1], y[i][0]])

        #labels_text = [ (l[1]+' '+l[0]) for l in self.labels]
        ind.sort(key=lambda x: x[2], reverse=True)
        ind = np.array(ind)
        y=ind[:,2].astype(np.float32)

        labels_text= [ (l[1]+' '+l[0]) for l in ind]

        # y = np.sort(labels_text, axis=0, order=1)
        # y = np.flipud(y)


        # plt.bar(x, y, 0.25)
        # plt.xticks(x )
        #plt.xlabel(['a', 'b', 'c', 'd', 'e'])

        fig, ax = plt.subplots()
        ax.bar(x, y, 0.65)#, align='center')
#        ax.set_xticks(x)
#        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax.tick_params(axis='x', which='major', labelsize=9)
#        fig.autofmt_xdate()

        plt.xticks(x+0.4, labels_text, rotation='vertical', verticalalignment='top', ha='center')
#        plt.xticks(x+0.3)
        plt.subplots_adjust(bottom=0.2)
#        plt.xlabel(x, labels_text, rotation='vertical', verticalalignment='top', ha='left')
        #plt.xticks(ind + width, ind)

        #plt.xlabel(labels_text)
        plt.ylabel('Recall')
        plt.xlim([0, 101])
        plt.ylim([0.3, 1])

        #plt.show()

        return ind


def format_date(x, pos=None):
    #thisind = np.clip(int(x + 0.5), 0, N - 1)
    #r.date[thisind].strftime('%Y-%m-%d')
    label = 'aafdf'
    return label

#-----------------------------------------------------------------------------------------------------------------
# Detect actions
#-----------------------------------------------------------------------------------------------------------------
def detect(config):
    vp = VideoProcessor()
    vp.set_default_dataset_by_name(config.dataset)
    video_list = vp.parse_video_list(vp.validation_split1_file)

    log_file = config.dataset+'_'+config.type_name+'_result.txt'
    dr = DetectionRecorder(log_file, append=True)


    acd = ac.ActionDetectorLSTM()
    acd.network_init(config.net_path, iter=int(config.net_iter), lstm_seq_length=config.timesteps, gpuid=config.gpuid)

    print "Detector configuration:"
    print config

    print "Detecting actions - "+config.type_name
    print "Processing videos: ",len(video_list)
    print "batch / frame / total_frames"

    for video_dir, video_label in video_list:

        #if video_dir != 'JavelinThrow/v_JavelinThrow_g02_c01': continue

        vp.load_video(video_dir, video_label)
        vp.reset_lstm_batch(config)
        dr.record_start()

        class_result_filename = os.path.join(vp.get_data_root(), config.detections_filename)
        if not config.overwrite and os.path.isfile(class_result_filename):
            print "Already processed: ",video_dir
            continue

        acd.reset()

        batch_id=0
        while True:

            sys.stdout.write("\r%5d /%5d /%5d   " % (batch_id, vp.video_id, len(vp.frames_rgb)) )
            sys.stdout.flush()
            batch_id += 1

            batch = vp.get_next_lstm_batch()
            if batch is None:
                break

            result = acd.detect(batch)

        # Process the collected detections
        class_result = acd.classify()
        acd.save_class_by_frame(class_result_filename, save_lstm=True, save_frame=True)

        dr.record_finish(video_label, video_dir, class_result)

        #break

    print "Done"

    return



def detect_lstm(config):
    vp = VideoProcessor()
    vp.set_default_dataset_by_name(config.dataset)
    video_list = vp.parse_video_list(vp.validation_split1_file)

    log_file = config.dataset+'_'+config.type_name+'_result.txt'
    dr = DetectionRecorder(log_file, append=True)


    acd = ac.ActionDetectorLSTM()
    acd.network_init(config.net_path, iter=int(config.net_iter), lstm_seq_length=config.timesteps, gpuid=config.gpuid)

    print "Detector configuration:"
    print config

    print "Detecting actions - "+config.type_name
    print "Processing videos: ",len(video_list)
    print "batch / frame / total_frames"

    for video_dir, video_label in video_list:

        #if video_dir != 'JavelinThrow/v_JavelinThrow_g02_c01': continue

        vp.load_video(video_dir, video_label)
        vp.reset_lstm_batch(config)
        dr.record_start()

        class_result_filename = os.path.join(vp.get_data_root(), config.detections_filename)
        if not config.overwrite and os.path.isfile(class_result_filename):
            print "Already processed: ",video_dir
            continue

        acd.reset()

        batch_id=0
        while True:

            sys.stdout.write("\r%5d /%5d /%5d   " % (batch_id, vp.video_id, len(vp.frames_rgb)) )
            sys.stdout.flush()
            batch_id += 1

            batch = vp.get_next_lstm_batch()
            if batch is None:
                break

            result = acd.detect(batch)

        # Process the collected detections
        class_result = acd.classify()
        acd.save_class_by_frame(class_result_filename, save_lstm=True, save_frame=True)

        dr.record_finish(video_label, video_dir, class_result)

        #break

    print "Done"

    return

#-----------------------------------------------------------------------------------------------------------------
# Analyse detections
#-----------------------------------------------------------------------------------------------------------------


def fuse_detection(config):
    print "Configuration"
    print config

    vp = VideoProcessor()
    vp.set_default_dataset_by_name(config.dataset)
    video_list = vp.parse_video_list(vp.validation_split1_file)

    fusion_weight = 0.65 # temporal percentage
    da_comb = DetectionAnalyzer('LSTM_Frame', fusion_weight, labels=vp.labels)
    da_frame = DetectionAnalyzer('Frame', fusion_weight, labels=vp.labels)
    da_lstm = DetectionAnalyzer('LSTM', fusion_weight, labels=vp.labels)

    counter = 0
    for video_dir, video_label in video_list:
        video_label = int(video_label)
        counter += 1.0
        sys.stdout.write("\r%d / %d " % (counter, len(video_list)) )
        sys.stdout.flush()

        vp.load_video(video_dir, video_label)
        try:
            detections_temporal = np.loadtxt(vp.video_frames_flow_dir+'/temporal-s8-'+str(config.timesteps)+'.txt', str, delimiter=' ')
            detections_spatial = np.loadtxt(vp.video_frames_rgb_dir+'/spatial-s8-'+str(config.timesteps)+'.txt', str, delimiter=' ')
        except IOError as e:
            print
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print e.filename
            break
        except:
            print
            print "Unexpected error:", sys.exc_info()[0]
            raise

        #fuse(detections_temporal, detections_spatial, 0.5)

        # Split the fused, lstm and per-frame detections
        detections_temporal_lstm = detections_temporal[:, 102:203].astype(np.float32)
        detections_spatial_lstm = detections_spatial[:, 102:203].astype(np.float32)
        da_lstm.add(detections_temporal_lstm, detections_spatial_lstm, video_label)

        detections_temporal_frame = detections_temporal[:, 203:304].astype(np.float32)
        detections_spatial_frame = detections_spatial[:, 203:304].astype(np.float32)
        da_frame.add(detections_temporal_frame, detections_spatial_frame, video_label)

        detections_temporal = detections_temporal[:, 1:102].astype(np.float32)
        detections_spatial = detections_spatial[:, 1:102].astype(np.float32)
        da_comb.add(detections_temporal, detections_spatial, video_label)

        #if counter > 50: break

    # print results
    # da_comb.print_result()
    # da_frame.print_result()
    # da_lstm.print_result()
    print
    print ("%10s" % "Det. type")+'   '+("%6s" % "rgb")+'   '+ ("%6s" % "flow")+'   '+ ("%6s" % "fused")
    print da_comb
    print da_frame
    print da_lstm

    da_comb.plot_detection_matrix(10, name_suffix=str(config.timesteps))
    # da_lstm.plot_detection_matrix(10, name_suffix=str(config.timesteps))
    # da_frame.plot_detection_matrix(10, name_suffix=str(config.timesteps))

    ind = da_comb.plot_recall(10, name_suffix=str(config.timesteps))
    print ind
    da_comb.plot_confusion_matrix()
    da_comb.plot_confusion_matrix(ind[-10:])
    #da_frame.plot_recall(10, name_suffix=str(config.timesteps))

    cv2.waitKey(0)


    return


def print_header():
    header = "batch / frm / tot f   vid   comb   lstm   frm     status    GT comb lstm frm video_name                              proc time (sec)"
    input('')
    print header



def run(args):

    vdc = VideoDetectorConfiguration()
    vdc.dataset = args.dataset
    vdc.gpuid = args.gpu
    vdc.overwrite = args.overwrite

    if args.print_header:
        print_header()
        sys.exit(0)

    if args.temporal:
        vdc.type_name = 'temporal'
        vdc.net_path = dar_root+'/projects/DAR/temporal_lstm_net-full/'
        vdc.net_iter = 4000  # T=25
        vdc.net_iter = 5000  # T=18
        vdc.net_iter = 3000  # T=10
        vdc.detections_filename = 'temporal-s8-10.txt'
        vdc.flow = True
        vdc.timesteps=10  # The used network must be trained for this sequence length !
        vdc.streams=15
        vdc.stride=5
        vdc.padded=True
        vdc.sampling=5
        vdc.channels=20
        detect(vdc)


    if args.spatial:
        vdc.type_name = 'spatial'
        vdc.net_path = dar_root+'/projects/DAR/spatial_lstm_net-full/'
        vdc.net_iter = 7000  # T=25
        # vdc.net_iter = 7000  # T=18
        #vdc.net_iter = 8000  # T=10
        vdc.detections_filename = 'spatial-s8-25.txt'
        vdc.flow = False
        vdc.timesteps=25  # The used network must be trained for this sequence length !
        vdc.streams=7
        vdc.stride=12
        vdc.padded=True
        vdc.sampling=5
        vdc.channels=3
        detect(vdc)


    if args.fuse:
        vdc.timesteps=10
        fuse_detection(vdc)

    if args.lstm:
        # Detects actions from concatenated temporal and spatial features
        vdc.type_name = 'lstm'
        vdc.net_path = dar_root+'/projects/DAR/lstm_net-3/'
        #vdc.net_iter = 17077  # T=25
        vdc.net_iter = 17077  # T=18
        #vdc.net_iter = 8000  # T=10
        vdc.detections_filename = 'lstm-s8-18.txt'
        vdc.flow = False
        vdc.timesteps=18  # The used network must be trained for this sequence length !
        vdc.streams=7
        vdc.stride=9
        vdc.padded=True
        vdc.sampling=1
        vdc.channels=1
        vdc.npy = True
        detect_lstm(vdc)


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="parse some things.")
    p.add_argument("cmd", help=argparse.SUPPRESS, nargs="*")
    p.add_argument("-t","--temporal",  action="store_true", help="detect actions in temporal domain")
    p.add_argument("-s","--spatial",  action="store_true", help="detect actions in spatial domain")
    p.add_argument("-g","--gpu", type=int, help="select GPU device", default=0)
    p.add_argument("-d","--dataset", help="ucf or thumos", default="ucf")
    p.add_argument("-o","--overwrite", action="store_true", help="Overwrite already completed detections", default=False)
    p.add_argument("-f","--fuse",  action="store_true", help="Joins spatial and temporal detections")
    p.add_argument("-p","--print-header",  action="store_true", help="Print only a header for log detector log messages")
    p.add_argument("-l","--lstm",  action="store_true", help="Runs early LSTM fusion detector", default=False)


    args = p.parse_args()
    print args

    if not (args.temporal or args.spatial or args.fuse or args.print_header or args.lstm):
        p.print_help()
        sys.exit(2)

    run(args)
