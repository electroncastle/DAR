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

if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set, exiting'
    sys.exit()

dar_root = os.environ['DAR_ROOT']
app_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

sys.path.insert(0, dar_root + '/python')
import caffe

import cv2


class VideoActionDetector(object):

    def __init__(self):
        self.name = "VideoActionDetector"
        self.net = None
        self.stride = 10
        self.gpuid = 0
        self.threadLock = threading.Lock()
        self.transformer = None

        self.flow_channels = 20  # 10 x + 10 y optical flow frames
        self.flow_batch_size = 50

        self.rgb_channels = 3
        self.rgb_batch_size = 50

        # Target image shape for detection
        self.width = 224
        self.height = 224

        self.classes_spatial = []
        self.classes_temporal = []

        self.flow_transformer = None
        self.rgb_transformer = None

        self.rgb_net = None
        self.flow_net = None

        self.labels = []

    def update_path(self):
        # Common
        self.video_path = os.path.join(self.datasets_root, self.dataset_name, self.dataset_video, self.video_name)+'.'+self.video_ext
        self.rgbflow_path = os.path.join(self.datasets_root, self.dataset_name, self.dataset_rgbflow, self.class_name, self.video_name)

        try:
            self.labels = np.loadtxt(self.labels_file, str, delimiter=' ')
        except:
            print 'Cannot load labels: ',self.labels_file

        return

    def set_video_name(self, video_name):
        self.video_name = video_name
        self.update_path()


    def set_path(self):
        self.flow_proto = app_dir+'/models-proto/two-streams-nvidia/vgg_16_flow_deploy.prototxt'
        self.flow_model = app_dir+'/models-bin/two-streams_16_split1_flow-nvidia_iter_26000.caffemodel'
        self.flow_model = app_dir+'/models-bin/cuhk_action_temporal_vgg_16_split1.caffemodel'

        self.rgb_proto = app_dir+'/models-proto/two-streams-nvidia/vgg_16_rgb_deploy.prototxt'
        self.rgb_model = app_dir+'/models-bin/two-streams_vgg_16_split1_rgb-nvidia_iter_10000.caffemodel'
        # self.rgb_model = app_dir+'/models-bin/cuhk_action_spatial_vgg_16_split1.caffemodel'


        self.datasets_root = os.path.join(dar_root, 'share/datasets/')

        if 0:
            # UCF 101
            self.dataset_name = 'UCF-101'
            self.dataset_video = 'UCF101'
            self.dataset_rgbflow = 'UCF101-rgbflow/'

            self.video_name = 'v_FloorGymnastics_g16_c04'
            self.video_name = 'v_Shotput_g07_c01'

            self.video_ext = 'avi'
            self.class_name = 'Shotput'

            self.labels_file = os.path.join(self.datasets_root, self.dataset_name, 'labels-new.txt')
        else:
            # THUMOS validation
            self.dataset_name = 'THUMOS2015'
            self.dataset_video = 'thumos15_validation'
            self.dataset_rgbflow = 'thumos15_validation-rgbflow'

            self.video_name = 'thumos15_video_validation_0000006'
            self.video_ext = 'mp4'
            self.class_name = ''

            self.labels_file = os.path.join(self.datasets_root, 'UCF-101', 'labels-new.txt')

        self.update_path()

        # If the rgbflow dir has subdirectories with the classnames
        # e.g. 'UCF101-rgbflow/FloorGymnastics/v_FloorGymnastics_g16_c04'
        #self.has_class_dirs = False


    def configureTransformer(self, meanImg, dataShape, rgb):

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        transformer = caffe.io.Transformer({'data': dataShape})
        transformer.set_transpose('data', (2,0,1))
        # transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel

        if meanImg != []:
            transformer.set_mean('data', meanImg) # mean pixel

        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]

        if rgb:
            transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

        return transformer


    def loadStack(self, imageFiles, color):
        #self.threadLock.acquire()

        channels = len(imageFiles)
        # Load X,Y optical filed frames
        batch = [] #np.empty([channels,img.shape[0], img.shape[1]])
        for b in range(0, channels):
            imgFile = imageFiles[b]
            try:
                img = caffe.io.load_image(imgFile, color)
            except:
                break

            # img = transformer.preprocess('data', img)
            channelsEx = img.shape[2]
            for c in range(0, channelsEx):
                batch.append(img[:,:,c])

        #self.threadLock.release()
        return np.asarray(batch)


    def get_flow_sample(self, path, frame_id, channels):

        # Get filenames for the OF frame stack starting with frame_id
        imgBatch = []
        for stack_frame in range(1, channels/2+1):
            xFlowFile = os.path.join(path, 'flow_x_{:0>4d}'.format(frame_id+stack_frame)+'.jpg')
            yFlowFile = os.path.join(path, 'flow_y_{:0>4d}'.format(frame_id+stack_frame)+'.jpg')
            imgBatch.append(xFlowFile)
            imgBatch.append(yFlowFile)

        img_stack = self.loadStack(imgBatch, False)

        return img_stack


    def get_rgb_sample(self, path, frame_id, channels):

        # Get filenames for the OF frame stack starting with frame_id
        imgBatch = []
        for stack_frame in range(1, (channels/3)+1):
            # print 'image_{:0>4d}'.format(frame_id+stack_frame)+'.jpg'
            imageFile = os.path.join(path, 'image_{:0>4d}'.format(frame_id+stack_frame)+'.jpg')
            imgBatch.append(imageFile)

        img_stack = self.loadStack(imgBatch, True)

        return img_stack


    # @profile
    def transformStack(self, transformer, flowStack, crop, mirror, flow, mean_image):

        h = flowStack[0].shape[0]
        w = flowStack[0].shape[1]
        size = len(flowStack)

        # mirror = False
        # crop = 4

        i = 0
        flowStackTrans = np.empty((size, self.height, self.width), dtype='float32')
        for fs in flowStack:

            f = fs.copy()
            if (mirror):
                if (flow):
                    if i % 2 == 0:
                        # For OF x
                        f = 1-np.fliplr(f)
                    else:
                        # For OF y
                        f = np.fliplr(f)
                else:
                    f = np.fliplr(f)

            if (crop == 0):
                ft = f[0:self.height, 0:self.width]
            if (crop == 1):
                ft = f[0:self.height, -self.width::]
            if (crop == 2):
                ft = f[-self.height::, 0:self.width]
            if (crop == 3):
                ft = f[-self.height::, -self.width::]
            if (crop == 4):
                t = (h-self.height)/2
                l = (w-self.width)/2
                ft = f[t:t+self.height, l:l+self.width]

            # flowStackTrans.append(ft)
            flowStackTrans[i]= ft
            i += 1

        # flowStackTrans = np.asarray(flowStackTrans)

        if 1:
            batch = np.transpose(flowStackTrans, (1, 2, 0))
            data = transformer.preprocess('data', batch)
        else:
            batch = flowStackTrans*255.0
            data = np.subtract(batch, mean_image)
            # data1 = np.add(batch, mean_image)

        return data


    def detect_flow(self, samples):

        self.flow_net.blobs['data'].reshape(self.flow_batch_size, self.flow_channels, self.width, self.height)
        self.flow_net.blobs['data'].data[...] = samples

        #print "Runnig forward pass"
        out = self.flow_net.forward()
        classes = out['prob']

        return classes


    def detect_rgb(self, samples):

        self.rgb_net.blobs['data'].reshape(self.rgb_batch_size, self.rgb_channels, self.width, self.height)
        self.rgb_net.blobs['data'].data[...] = samples

        #print "Runnig forward pass"
        out = self.rgb_net.forward()
        classes = out['prob']

        return classes


    def showResult(self, classes):

        if  len(classes.shape) > 1:
            classResultAvg = np.sum(classes, 0)/classes.shape[0]
        else:
            classResultAvg = classes

        result = classResultAvg.argmax()
        # print("Predicted class is #{}.".format(result))

        if len(self.labels) > result:
            print self.labels[result], ' ', classResultAvg[result]
        else:
            print 'Unknown label ', classResultAvg[result]


    def save_result(self, classes, filename):

        fout = open(filename, 'wt')
        for i in range(classes.shape[0]):

            record = str(i*self.stride)+' '+' '.join(['%.6f' % num for num in classes[i]])
            fout.write(record+'\n')
            fout.flush()

        fout.close()


# def recordResult(file_rgb_result, result, test_video, labels):
#
#     classResultAvg = np.sum(result.transpose(),1)/result.shape[1]
#     classId = classResultAvg.argmax()
#
#     print labels[classId]
#
#     # trueClass, founfClass, videoname, feature_vector
#     videoDir = test_video[1].split('/')
#     videoDir = videoDir[len(videoDir)-1]
#     record = '{} {:d} {} '.format(test_video[0], classId, videoDir)
#
#     # Add the feature vector
#     record += ' '.join(['%.6f' % num for num in classResultAvg])
#     # Read it this way
#     #np.fromstring(VIstring, sep=',')
#
#     if (file_rgb_result != None):
#         file_rgb_result.write(record+'\n')
#         file_rgb_result.flush()
#
#     return     classId

    def get_flow_mean_image(self):
        meanValue = 128.0
        mean_image = np.array([[np.transpose([meanValue for i in range(self.width)])]*self.height] * self.flow_channels, dtype='float32')
        return mean_image


    def get_rgb_mean_image(self):
        rgb_mean=[104.0, 117.0, 123.0]
        mean_image = np.array([[np.transpose([rgb_mean[0] for i in range(self.width)])]*self.height,
                   [np.transpose([rgb_mean[1] for i in range(self.width)])]*self.height,
                   [np.transpose([rgb_mean[2] for i in range(self.width)])]*self.height], dtype='float32')
        return mean_image


    #@profile
    def detect_temporal(self, device_id):
        out_filename = os.path.join(self.rgbflow_path)+'/temporal.txt'
        if os.path.isfile(out_filename):
            print 'Already exists: ',out_filename
            return

        caffe.set_mode_gpu()
        caffe.set_device(device_id)

        # One sample is 10 (x,y) frames (20) each augmented in 10 ways

        # Number of (x,y) optical flow frames in each sample
        # Each samples get augmented in 10 transformations
        # 20 x 10 = 200

        augmented_size = 10 # We generate 10 augmented samples
        batch_num = self.flow_batch_size / augmented_size

        mean_image = self.get_flow_mean_image()

        if self.flow_net == None:
            self.flow_net = caffe.Net(self.flow_proto, self.flow_model, caffe.TEST)

        data_shape = self.flow_net.blobs['data'].data.shape
        self.flow_transformer = self.configureTransformer(mean_image, data_shape, False)

        self.classes_temporal = []
        frame_id = 0
        while True:

            samples = []
            for s in range(0, batch_num):
                # print 'FLOW: Loading sample'
                sample = self.get_flow_sample(self.rgbflow_path, frame_id, self.flow_channels)
                if len(sample) != self.flow_channels:
                    break

                # Augment sample
                start = time.time()
                # print 'FLOW: Transforming sample'
                for m in [True, False]:
                    for c in range(0,5):
                        samples.append(self.transformStack(self.flow_transformer, sample, c, m, True, mean_image))

                frame_id += self.stride
                end = time.time()
                # print "FLOW loading & transformation took: ", (end - start)


            if len(samples) != self.flow_batch_size:
                break


            # print 'FLOW: detecting'
            classes = self.detect_flow(samples)

            # Average results from augmented samples
            for s in range(0, batch_num):
                id = s*augmented_size
                classAvg = np.sum(classes[id:id+augmented_size, :], 0)/augmented_size
                # self.showResult(classes[id:id+augmented_size, :] )
                # self.showResult(classAvg)

                if self.classes_temporal == []:
                    self.classes_temporal = classAvg.copy()
                else:
                    self.classes_temporal = np.vstack([self.classes_temporal, classAvg])


            print 'FLOW frame: ',frame_id,'  [',(frame_id*1.0/30.0), ']  ',
            self.showResult(classes)


            # if frame_id > 1000:
            #     break

            # threading._sleep(0.5)

        self.showResult(self.classes_temporal)


        [path, filename] = os.path.split(self.rgbflow_path)
        self.save_result(self.classes_temporal, out_filename)


    #@profile
    def detect_spatial(self, device_id):

        out_filename = os.path.join(self.rgbflow_path)+'/spatial.txt'
        if os.path.isfile(out_filename):
            print 'Already exists: ',out_filename
            return

        caffe.set_mode_gpu()
        caffe.set_device(device_id)

        # One sample is 10 (x,y) frames (20) each augmented in 10 ways

        # Number of (x,y) optical flow frames in each sample
        # Each samples get augmented in 10 transformations
        # 20 x 10 = 200

        augmented_size = 10 # We generate 10 augmented samples
        batch_num = self.rgb_batch_size / augmented_size

        mean_image = self.get_rgb_mean_image()

        if self.rgb_net == None:
            self.rgb_net = caffe.Net(self.rgb_proto, self.rgb_model, caffe.TEST)

        data_shape = self.rgb_net.blobs['data'].data.shape
        self.rgb_transformer = self.configureTransformer(mean_image, data_shape, True)

        self.classes_spatial = []
        frame_id = 0
        while True:

            # start = time.time()

            samples = []
            for s in range(0, batch_num):

                # print 'RGB: loading sample'
                sample = self.get_rgb_sample(self.rgbflow_path, frame_id, self.rgb_channels)
                if len(sample) != self.rgb_channels:
                    break

                # Augment sample
                # print 'RGB: Transforming sample'
                for m in [True, False]:
                    for c in range(0,5):
                        samples.append(self.transformStack(self.rgb_transformer, sample, c, m, False, mean_image))

                frame_id += self.stride


            if len(samples) != self.rgb_batch_size:
                break

            # end = time.time()
            # print "RGB loading & transformation took: ", (end - start)

            # print 'RGB: detecting'
            classes = self.detect_rgb(samples)


            # Average results from augmented samples
            for s in range(0, batch_num):
                id = s*augmented_size
                classAvg = np.sum(classes[id:id+augmented_size, :].transpose(),1)/augmented_size

                if self.classes_spatial == []:
                    self.classes_spatial = classAvg.copy()
                else:
                    self.classes_spatial = np.vstack([self.classes_spatial, classAvg])


            print 'RGB frame: ',frame_id,'  ',
            self.showResult(classAvg)
            pass

            # if frame_id > 1000:
            #     break

            # threading._sleep(0.5)

        self.showResult(self.classes_spatial)


        [path, filename] = os.path.split(self.rgbflow_path)
        self.save_result(self.classes_spatial, out_filename)

        return


    def th_test(self):

        while True:
            start = time.time()
            for i in range(1000000):
                x = 1256456
                x = np.sqrt(x)
            end = time.time()
            print "OO = ", (end - start)

    def th_test2(self):

        while True:
            start = time.time()
            for i in range(1000000):
                x = 1256456
                x = np.sqrt(x)
            end = time.time()
            print "XX = ", (end - start)



    def anotate(self):
        # Do two pass detection, first flow then RGB

        # self.detect_temporal()
        # self.detect_spatial()
        # return

        # flow_thread = threading.Thread(target=self.detect_temporal)
        # #flow_thread.daemon = True
        # flow_thread.start()
        #
        # rgb_thread = threading.Thread(target=self.detect_spatial)
        # #rgb_thread.daemon = True
        # rgb_thread.start()

        # thread = threading.Thread(target=self.th_test)
        #rgb_thread.daemon = True
        # thread.start()

        # thread2 = threading.Thread(target=self.th_test2)
        #rgb_thread.daemon = True
        # thread2.start()


        # thread2.join()

        # rgb_thread.join()
        #flow_thread.join()

        self.detect_temporal()
        self.detect_spatial()

this = None

def distributor(t):
    if t==0:
        this.detect_temporal()
    else:
        this.detect_spatial()


class myThread (threading.Thread):

    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        while True:
            start = time.time()
            for i in range(1000000):
                x = 1256456
                x = np.sqrt(x)
            end = time.time()
            print self.name,"  ", (end - start)



def nothing(x):
    pass

def testSlider():
    # Create a black image, a window
    img = np.zeros((300,512,3), np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R','image',0,255,nothing)
    cv2.createTrackbar('G','image',0,255,nothing)
    cv2.createTrackbar('B','image',0,255,nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image',0,1,nothing)

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R','image')
        g = cv2.getTrackbarPos('G','image')
        b = cv2.getTrackbarPos('B','image')
        s = cv2.getTrackbarPos(switch,'image')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b,g,r]

    cv2.destroyAllWindows()


if __name__ == "__main__":

    # thread1 = myThread(1, "XX", 1)
    # thread2 = myThread(2, "OO", 2)
    #
    # # Start new Threads
    # thread1.start()
    # thread2.start()
    #
    # thread2.join()

    #go(0, True)	# Temporal
    #go(1, False) # Spatial

    video = VideoActionDetector()
    video.set_path()
    # video.detect_temporal(0)
    # sys.exit(0)

    # Load a list of videos to process
    val_list_filename = dar_root+'/share/datasets/THUMOS2015/thumos15_validation/annotated.txt'
    val_list = np.loadtxt(val_list_filename, str, delimiter=' ')

    for i in range(len(val_list)):
        file = val_list[i]

        [file, ext] = os.path.splitext(file)
        # if file == "thumos15_video_validation_0001630":
        #     continue

        video.set_video_name(file)

        print 'Processing: type=temporal video=', file

        start = time.time()
        if sys.argv[1] == 't':
            video.detect_temporal(0)

        if sys.argv[1] == 's':
            video.detect_spatial(1)

        end = time.time()
        print "TOOK: ", (end - start)

