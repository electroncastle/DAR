import os
import io
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import time

if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set, exiting'
    sys.exit()

dar_root = os.environ['DAR_ROOT']
app_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

sys.path.insert(0, dar_root + '/python')
import caffe

import cv2

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
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

    plt.imshow(data)


def getRGBImages(videoImagesPath, samples):

    allImages = []
    for filename in os.listdir(videoImagesPath):
        if filename.startswith('image_'):
            allImages.append(os.path.join(videoImagesPath, filename))
    allImages.sort()

    rgbSamples = []
    sample_step = (len(allImages) ) / samples
    for img in range(0, samples):
        i  = img*sample_step
        rgbSamples.append([allImages[i]])

#    rgbSamples = random.sample(allImages, samples)

    return rgbSamples


def getRGBFrames(videoImagesPath, samples):

    allImages = []
    for filename in os.listdir(videoImagesPath):
        if filename.startswith('image_'):
            allImages.append(os.path.join(videoImagesPath, filename))
    allImages.sort()

    rgbSamples = []
    sample_step = (len(allImages) ) / samples
    for img in range(0, samples):
        i  = img*sample_step
        rgbSamples.append([allImages[i]])

#    rgbSamples = random.sample(allImages, samples)

    return rgbSamples


def getFlowImages(videoImagesPath, stack_size, samples):

    allImages = []
    for filename in os.listdir(videoImagesPath):
        if filename.startswith('flow_x_'):
            allImages.append(os.path.join(videoImagesPath, filename))

    allImages.sort()

    flowxyImages = []
    sample_step = (len(allImages) - (stack_size+1)) / samples

    for img in range(0, samples):
        i  = img*sample_step
        file = allImages[i]
        path, file = os.path.split(file)
        file, ext = os.path.splitext(file)
        fileToks = file.split('_')
        id = int(fileToks[2])

        imgBatch = []
        for stack_frame in range(0, stack_size):
            xFlowFile = os.path.join(path, fileToks[0]+'_x_{:0>4d}'.format(id+stack_frame)+'.jpg')
            yFlowFile = os.path.join(path, fileToks[0]+'_y_{:0>4d}'.format(id+stack_frame)+'.jpg')
            imgBatch.append(xFlowFile)
            imgBatch.append(yFlowFile)

        flowxyImages.append(imgBatch)

    return flowxyImages

def loadStack(imageFiles):

    channels = len(imageFiles)
    # Load X,Y optical filed frames
    batch = [] #np.empty([channels,img.shape[0], img.shape[1]])
    for b in range(0, channels):
        imgFile = imageFiles[b]
        img = caffe.io.load_image(imgFile, False)
        # img = transformer.preprocess('data', img)
        channelsEx = img.shape[2]
        for c in range(0, channelsEx):
            batch.append(img[:,:,c])

        # if isFlow:
        #     img = img.reshape(img.shape[0], img.shape[1])
        #     batch.append(img)
        # else:
        #     batch.append(img[:,:,0])
        #     batch.append(img[:,:,1])
        #     batch.append(img[:,:,2])

        # batch[:,:,b] = img

    return np.asarray(batch)

def initTransformer(meanImg, dataShape):
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': dataShape})
    transformer.set_transpose('data', (2,0,1))
    # transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    if meanImg != []:
        transformer.set_mean('data', meanImg) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    # transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    return transformer

def transformStack(flowStack, transformer, crop, mirror):

    h = flowStack[0].shape[0]
    w = flowStack[0].shape[1]

    # mirror = False
    # crop = 4
    cw=224
    ch=224

    flowStackTrans = []
    for f in flowStack:
        if (crop == 0):
            ft = f[0:ch, 0:cw]
        if (crop == 1):
            ft = f[0:ch, -cw::]
        if (crop == 2):
            ft = f[-ch::, 0:cw]
        if (crop == 3):
            ft = f[-ch::, -cw::]
        if (crop == 4):
            t = (h-ch)/2
            l = (w-cw)/2
            ft = f[t:t+ch, l:l+cw]

        if (mirror):
            ft = np.fliplr(ft)

        flowStackTrans.append(ft)

    flowStackTrans = np.asarray(flowStackTrans)

    batch = np.transpose(flowStackTrans, (1, 2, 0))
    data = transformer.preprocess('data', batch)
    return data


def showResult(classes):

    batchSize = len(classes)
    #result = map(lambda x: classes[x].argmax(), range(0, batchSize))
    # print result
    classResultAvg = np.sum(classes.transpose(),1)/classes.shape[1]
    # print classResultAvg

    result = classResultAvg.argmax()
    # print("Predicted class is #{}.".format(result))

    labels_filename = dar_root+'/share/datasets/UCF-101/labels-new.txt'
    try:
        labels = np.loadtxt(labels_filename, str, delimiter=' ')
    except:
        print 'Cannot load: '.labels_filename
        return

    print labels[result], ' ', classResultAvg[result]


def classSpatial(net, video_path):

    samples = 25 # Number of frame across the video to sample

    # Number of x,y optical flow frames in each sample
    # Each samples get augumented in 10 transformations
    # 20 x 10 = 200
    stack_size = 1
    batch_size = 50
    batch_num = batch_size / 10 # We generate 10 augumented samples from single

    images = getRGBImages(video_path, samples)

    # classify augumented data
    width = 224
    height = 224
    meanValue = 128
    channels = stack_size*3

    meanImg = []
    meanImg = np.array([[np.transpose([104 for i in range(width)])]*height,
                       [np.transpose([117 for i in range(width)])]*height,
                       [np.transpose([123 for i in range(width)])]*height])

    allClasses = []
    for b in range(0, samples/batch_num):
        imgs = []
        for s in range(0, batch_num):
            sample = images[s*b]
            flowStack = loadStack(sample)

            # Perform 10 data augumentation
            # Five spatial positions, left-top, right-top, left-bottom, right-bottom, center
            # For all spatial position perform mirroring
            for m in [True, False]:
                for c in range(0,5):
                    imgs.append(transformStack(flowStack, meanImg, net.blobs['data'].data.shape, c, m))

        # Run forward pass with batch 50
        net.blobs['data'].reshape(batch_size, channels, width, height)
        net.blobs['data'].data[...] = imgs

        #print "Runnig forward pass"
        out = net.forward()
        classes = out['prob']
        if allClasses == []:
            allClasses = classes
        else:
            allClasses = np.append(allClasses, classes, axis=0)

        print b,' ',
        showResult(classes)

    return allClasses


def classTemporal(net, video_path):

    samples = 25 # Number of frame across the video to sample

    # Number of x,y optical flow frames in each sample
    # Each samples get augmented in 10 transformations
    # 20 x 10 = 200
    stack_size = 10
    batch_size = 50
    batch_num = batch_size / 10 # We generate 10 augmented samples from single


    images = getFlowImages(video_path, stack_size, samples)

    # classify augumented data
    width = 224
    height = 224
    meanValue = 128
    channels = stack_size*2
    meanImg = np.array([[np.transpose([meanValue for i in range(width)])]*height] * channels)

    transformer = initTransformer(meanImg, net.blobs['data'].data.shape)

    start = time.time()

    allClasses = []
    for b in range(0, samples/batch_num):
        imgs = []
        for s in range(0, batch_num):

            sample = images[s*b]
            flowStack = loadStack(sample)

            # Perform 10 data augmentation
            # Five spatial positions, left-top, right-top, left-bottom, right-bottom, center
            # For all spatial position perform mirroring
            for m in [True, False]:
                for c in range(0,5):
                    imgs.append(transformStack(flowStack, transformer, c, m))


        # Run forward pass with batch 50
        net.blobs['data'].reshape(batch_size, channels, width, height)
        net.blobs['data'].data[...] = imgs

        end = time.time()
        # print "Imgs loading & transformation took: ", (end - start)

        start = time.time()

        #print "Runnig forward pass"
        out = net.forward()
        classes = out['prob']
        if allClasses == []:
            allClasses = classes
        else:
            allClasses = np.append(allClasses, classes, axis=0)

        print b,' ',
        showResult(classes)
        end = time.time()
        # print end - start


    return allClasses

# Some data analysis
    d = [(k, v.data.shape) for k, v in net.blobs.items()]
    print d

    p = [(k, v[0].data.shape) for k, v in net.params.items()]
    print p

    # the parameters are a list of [weights, biases]
    filters = net.params['conv1_1'][0].data
    #vis_square(filters.transpose(0, 2, 3, 1))


    feat = net.blobs['conv1_1'].data[0, :36]
    vis_square(feat, padval=1)


def loadTestVideos(filename):

    dset = open(filename, 'rt')
    print filename

    videos = []
    for line in dset:
        toks = line.split(' ')
        videos.append([toks[2].strip(), toks[0].strip()])

    return videos


def recordResult(file_rgb_result, result, test_video, labels):

    classResultAvg = np.sum(result.transpose(),1)/result.shape[1]
    classId = classResultAvg.argmax()

    print labels[classId]

    # trueClass, founfClass, videoname, feature_vector
    videoDir = test_video[1].split('/')
    videoDir = videoDir[len(videoDir)-1]
    record = '{} {:d} {} '.format(test_video[0], classId, videoDir)

    # Add the feature vector
    record += ' '.join(['%.6f' % num for num in classResultAvg])
    # Read it this way
    #np.fromstring(VIstring, sep=',')

    if (file_rgb_result != None):
        file_rgb_result.write(record+'\n')
        file_rgb_result.flush()

    return     classId


def go(gpuid, flow):

    flow_proto = app_dir+'/models-proto/two-streams-nvidia/vgg_16_flow_deploy.prototxt'
    flow_model = app_dir+'/models-bin/two-streams_16_split1_flow-nvidia_iter_26000.caffemodel'
    #flow_model = '/home/jiri/Lake/HAR/UCF101-2stream/cuhk_action_temporal_vgg_16_split1.caffemodel'

    rgb_proto = app_dir+'/models-proto/two-streams-nvidia/vgg_16_rgb_deploy.prototxt'
    rgb_model = app_dir+'/models-bin/two-streams_vgg_16_split1_rgb-nvidia_iter_10000.caffemodel'
    #rgb_model = '/home/jiri/Lake/HAR/UCF101-2stream/cuhk_action_spatial_vgg_16_split1.caffemodel'


    video_path = dar_root+'/share/datasets/UCF-101/UCF101-rgbflow/FloorGymnastics/v_FloorGymnastics_g16_c04'
    #video_path = '/home/jiri/Lake/HAR/datasets/UCF-101/UCF101-rgbflow/Diving/v_Diving_g22_c06'
    #video_path = '/home/jiri/Lake/HAR/datasets/UCF-101/UCF101-rgbflow/IceDancing/v_IceDancing_g03_c03'

    test_data_filename = dar_root+'/share/datasets/UCF-101/val-1-rnd.txt'
    test_result_flow_filename = dar_root+'/share/datasets/UCF-101/val-1-rnd-result-flow.txt'
    test_result_rgb_filename = dar_root+'/share/datasets/UCF-101/val-1-rnd-result-rgb.txt'

    imagenet_labels_filename = dar_root+'/share/datasets/UCF-101/labels-new.txt'

    try:
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter=' ')
    except:
        return

    test_videos = loadTestVideos(test_data_filename)

    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(gpuid)

    if flow:
        outFilename = test_result_flow_filename
        net = caffe.Net(flow_proto, flow_model, caffe.TEST)
    else:
        outFilename = test_result_rgb_filename
        net = caffe.Net(rgb_proto, rgb_model, caffe.TEST)

    file_out = None
#    file_out = open(outFilename, 'r+t')
#    file_out.seek(0, io.SEEK_SET)


    correct = 0
    for v in range(0, len(test_videos)):

        test_video = test_videos[v]
        # test_video[1] = video_path

        videoDir = test_video[1].split('/')
        videoDir = videoDir[len(videoDir)-1].strip()

        # Check whether we alredy tested the video
        if (file_out != None):
            line = file_out.readline().strip()
            if (len(line) > 0):
                line_toks = line.split(' ')
                if line_toks[2].strip() == videoDir:
                    if int(line_toks[0]) == int(line_toks[1]):
                        correct += 1
                    correctP = correct / (v + 1.0)
                    print v, "  Already done: ", (round(correctP, 3) * 100), '% ', test_video[0], ' ', videoDir
                    continue

        print "====================="
        print v, "  Classifying: ", test_video[0], ' ', videoDir
        if flow:
            result = classTemporal(net, test_video[1])
        else:
            result = classSpatial(net, test_video[1])

        classId = recordResult(file_out, result, test_video, labels)

        if int(test_video[0]) == classId:
            correct += 1

        correctP = correct/(v+1.0)

        print (round(correctP,3)*100),'%  Full result: '
        showResult(result)


    return


class Video(object):

    def __init__(self):
        self.name = "Video"
        self.net = None
        self.stride = 25
        self.gpuid = 0
        self.transformer = None

        # Target image shape for detection
        self.width = 224
        self.height = 224



    def set_path(self):
        self.flow_proto = app_dir+'/models-proto/two-streams-nvidia/vgg_16_flow_deploy.prototxt'
        self.flow_model = app_dir+'/models-bin/two-streams_16_split1_flow-nvidia_iter_26000.caffemodel'
        #flow_model = '/home/jiri/Lake/HAR/UCF101-2stream/cuhk_action_temporal_vgg_16_split1.caffemodel'

        self.rgb_proto = app_dir+'/models-proto/two-streams-nvidia/vgg_16_rgb_deploy.prototxt'
        self.rgb_model = app_dir+'/models-bin/two-streams_vgg_16_split1_rgb-nvidia_iter_10000.caffemodel'
        #rgb_model = '/home/jiri/Lake/HAR/UCF101-2stream/cuhk_action_spatial_vgg_16_split1.caffemodel'


        self.datasets_root = os.path.join(dar_root, 'share/datasets/')

        if 1:
            # UCF 101
            self.dataset_name = 'UCF-101'
            self.dataset_video = 'UCF101'
            self.dataset_rgbflow = 'UCF101-rgbflow/'

            self.video_name = 'v_FloorGymnastics_g16_c04'
            self.video_name = 'v_JavelinThrow_g02_c01'

            self.video_ext = 'avi'
            self.class_name = 'JavelinThrow'

            self.labels_file = os.path.join(self.datasets_root, self.dataset_name, 'labels-new.txt')
        else:
            # THUMOS validation
            self.dataset_name = 'THUMOS2015'
            self.dataset_video = 'thumos15_validation'
            self.dataset_rgbflow = 'thumos15_validation-rgbflow'

            self.video_name = 'thumos15_video_validation_0000006'
            self.video_ext = 'mp4'
            self.class_name = ''

            self.labels_file = os.path.join(self.datasets_root, self.dataset_name, 'labels-new.txt')

        # Common
        self.video_path = os.path.join(self.datasets_root, self.dataset_name, self.dataset_video, self.video_name)+'.'+self.video_ext
        self.rgbflow_path = os.path.join(self.datasets_root, self.dataset_name, self.dataset_rgbflow, self.class_name, self.video_name)


        # If the rgbflow dir has subdirectories with the classnames
        # e.g. 'UCF101-rgbflow/FloorGymnastics/v_FloorGymnastics_g16_c04'
        #self.has_class_dirs = False


    def configureTransformer(self, meanImg, dataShape):
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': dataShape})
        self.transformer.set_transpose('data', (2,0,1))
        # transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
        if meanImg != []:
            self.transformer.set_mean('data', meanImg) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        # transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB



    def loadStack(self, imageFiles):

        channels = len(imageFiles)
        # Load X,Y optical filed frames
        batch = [] #np.empty([channels,img.shape[0], img.shape[1]])
        for b in range(0, channels):
            imgFile = imageFiles[b]
            try:
                img = caffe.io.load_image(imgFile, False)
            except:
                break

            # img = transformer.preprocess('data', img)
            channelsEx = img.shape[2]
            for c in range(0, channelsEx):
                batch.append(img[:,:,c])

            # if isFlow:
            #     img = img.reshape(img.shape[0], img.shape[1])
            #     batch.append(img)
            # else:
            #     batch.append(img[:,:,0])
            #     batch.append(img[:,:,1])
            #     batch.append(img[:,:,2])

            # batch[:,:,b] = img

        return np.asarray(batch)


    def get_flow_sample(self, path, frame_id, channels):

        # Get filenames for the OF frame stack starting with frame_id
        imgBatch = []
        for stack_frame in range(1, channels, 2):
            xFlowFile = os.path.join(path, 'flow_x_{:0>4d}'.format(frame_id+stack_frame)+'.jpg')
            yFlowFile = os.path.join(path, 'flow_y_{:0>4d}'.format(frame_id+stack_frame)+'.jpg')
            imgBatch.append(xFlowFile)
            imgBatch.append(yFlowFile)

        img_stack = self.loadStack(imgBatch)

        return img_stack


    def transformStack(self, flowStack, crop, mirror, flow):

        h = flowStack[0].shape[0]
        w = flowStack[0].shape[1]

        # mirror = False
        # crop = 4

        i = 0
        flowStackTrans = []
        for f in flowStack:
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

            if (mirror):
                if (flow):
                    if i % 2 == 0:
                        # For OF x
                        ft = 1-np.fliplr(ft)
                    else:
                        # For OF y
                        ft = np.fliplr(ft)
                else:
                    ft = np.fliplr(ft)

            flowStackTrans.append(ft)
            i += 1

        flowStackTrans = np.asarray(flowStackTrans)

        batch = np.transpose(flowStackTrans, (1, 2, 0))
        data = self.transformer.preprocess('data', batch)
        return data


    def detect(self, batch_size, channels, samples):

        self.net.blobs['data'].reshape(batch_size, channels, self.width, self.height)
        self.net.blobs['data'].data[...] = samples

        #print "Runnig forward pass"
        out = self.net.forward()
        classes = out['prob']

        return classes


    def showResult(self, classes):

        batchSize = len(classes)
        #result = map(lambda x: classes[x].argmax(), range(0, batchSize))
        # print result
        classResultAvg = np.sum(classes.transpose(),1)/classes.shape[1]
        # print classResultAvg

        result = classResultAvg.argmax()
        # print("Predicted class is #{}.".format(result))

        try:
            labels = np.loadtxt(self.labels_file, str, delimiter=' ')
        except:
            print 'Cannot load: '.labels_filename
            return

        print labels[result], ' ', classResultAvg[result]


    def detect_temporal(self):

        step = 10 # Every 'step' frame will be sampled

        # One sample is 10 (x,y) frames (20) each augmented in 10 ways

        # Number of (x,y) optical flow frames in each sample
        # Each samples get augmented in 10 transformations
        # 20 x 10 = 200

        channels = 20  # 10 x + 10 y optical flow frames
        batch_size = 50

        augmented_size = 10 # We generate 10 augmented samples
        batch_num = batch_size / augmented_size

        meanValue = 128

        mean_image = np.array([[np.transpose([meanValue for i in range(self.width)])]*self.height] * channels)

        self.net = caffe.Net(self.flow_proto, self.flow_model, caffe.TEST)
        data_shape = self.net.blobs['data'].data.shape
        self.configureTransformer(mean_image, data_shape)

        frame_classes = []
        frame_id = 0
        while True:

            samples = []
            for s in range(0, batch_num):

                sample = self.get_flow_sample(self.rgbflow_path, frame_id, channels)
                if len(sample) != channels:
                    break

                # Augment sample
                for m in [True, False]:
                    for c in range(0,5):
                        samples.append(self.transformStack(sample, c, m, True))

                frame_id += step


            if len(samples) != batch_size:
                break

            classes = self.detect(batch_size, channels, samples)
            print 'Frame: ',frame_id,'  ',
            showResult(classes)

            if frame_classes == []:
                frame_classes = classes
            else:
                frame_classes = np.append(frame_classes, classes, axis=0)

        showResult(frame_classes)


    def detect_spatial(self):
        # outFilename = test_result_rgb_filename
        self.net = caffe.Net(self.rgb_proto, self.rgb_model, caffe.TEST)

        return


    def anotate(self):

        # Do two pass detection, first flow then RGB
        caffe.set_mode_gpu()
        caffe.set_device(self.gpuid)

        self.detect_temporal()
        # self.detect_spatial()


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

    #go(0, True)	# Temporal
    #go(1, False) # Spatial
    video = Video()
    video.set_path()
    video.anotate()


