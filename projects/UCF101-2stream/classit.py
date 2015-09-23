
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
import time

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

# @profile
def loadStack(imageFiles, color):

    channels = len(imageFiles)
    # Load X,Y optical filed frames
    batch = [] #np.empty([channels,img.shape[0], img.shape[1]])
    for b in range(0, channels):
        imgFile = imageFiles[b]
        img = caffe.io.load_image(imgFile, color)
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

def initTransformer(meanImg, dataShape, rgb):
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

# @profile
def transformStack(flowStack, transformer, crop, mirror, flow):

    h = flowStack[0].shape[0]
    w = flowStack[0].shape[1]
    size = len(flowStack)

    # mirror = False
    # crop = 4
    cw=224
    ch=224

    flowStackTrans = np.empty((size,ch,cw), dtype='float32')
    i = 0
    for sf in flowStack:
        # image = (f*255.0).astype(np.uint8)
        # #image.tofile('img-'+str(i)+'.jpg')
        # mpimg.imsave('img-o-'+str(i)+'.jpg', image)

        f=sf.copy()
        if (mirror):
            if (flow):
                if i % 2 == 0:
                    # For OF x
                    f = 1.0-np.fliplr(f)
                     # f = np.fliplr(f)
                else:
                    # For OF y
                    # f = 1.0-np.fliplr(f)
                    f = np.fliplr(f)
            else:
                f = np.fliplr(f)

        if (crop == 0):
            ft = f[0:ch, 0:cw]
        if (crop == 1):
            ft = f[0:ch, -cw::]
        if (crop == 3):
            ft = f[-ch::, 0:cw]
        if (crop == 4):
            ft = f[-ch::, -cw::]
        if (crop == 2):
            t = (h-ch)/2
            l = (w-cw)/2
            ft = f[t:t+ch, l:l+cw]

        # This works well for RGB with mine VGG16 model
        # if (mirror):
        #     if (flow):
        #         if i % 2 == 0:
        #             # For OF x
        #             ft = 1.0-np.fliplr(ft)
        #         else:
        #             # For OF y
        #             ft = np.fliplr(ft)
        #     else:
        #         ft = np.fliplr(ft)

        flowStackTrans[i]= ft
        # flowStackTrans.append(ft)

        # image = (ft*255.0).astype(np.uint8)
        # #image.tofile('img-'+str(i)+'.jpg')
        # mpimg.imsave('img-'+str(i)+'.jpg', image)

        i+=1

    #flowStackTrans = np.asarray(flowStackTrans)

    # batch = np.transpose(flowStackTrans, (1, 2, 0))
    # data = transformer.preprocess('data', batch)

    batch = np.transpose(flowStackTrans, (1, 2, 0))
    data = transformer.preprocess('data', batch)

    return data

labels = []


def poolMax(classes):

    max_classes = np.array([0.0]*classes.shape[1])
    for c in classes:
        id = c.argmax()
        val = c[id]
        max_classes[id] += val

    max_classes /=  classes.shape[0]
    id = max_classes.argmax()
    val = max_classes[id]

    result = np.array([0.0]*classes.shape[1])
    result[id] = val

    return result

def showResult(classes):
    global labels

    if 1:
        batchSize = len(classes)
        #result = map(lambda x: classes[x].argmax(), range(0, batchSize))
        # print result
        classResultAvg = np.sum(classes.transpose(),1)/classes.shape[0]
        # print classResultAvg

        result = classResultAvg.argmax()
        # print("Predicted class is #{}.".format(result))
    else:
        classResultAvg = []
        for i in range(0, len(classes), 50):
            nc = poolMax(classes[i:i+50,:])
            # if classResultAvg == []:
            #     classResultAvg = nc
            # else:
            #     classResultAvg = np.vstack([classResultAvg, nc])
            if classResultAvg == []:
                classResultAvg = nc.copy()
            else:
                classResultAvg = classResultAvg+nc

        classResultAvg /= (len(classes)/50)
        result = classResultAvg.argmax()


    if labels == []:
        labels_filename = '/home/jiri/Lake/DAR/share/datasets/UCF-101/labels-new.txt'
        try:
            labels = np.loadtxt(labels_filename, str, delimiter=' ')
        except:
            print "Couldn't load labels from ", labels_filename
            return

    print labels[result], ' ', classResultAvg[result]


transformer = None

# @profile
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
    channels = stack_size*3

    meanImg = []
    meanImg = np.array([[np.transpose([104.0 for i in range(width)])]*height,
                       [np.transpose([117.0 for i in range(width)])]*height,
                       [np.transpose([123.0 for i in range(width)])]*height], dtype='float32')

    global transformer
    if transformer is None:
        transformer = initTransformer(meanImg, net.blobs['data'].data.shape, True)

    start = time.time()

    allClasses = []
    for b in range(0, samples/batch_num):
        imgs = []
        for s in range(0, batch_num):
            sample = images[s*b]
            flowStack = loadStack(sample, True)

            # Perform 10 data augumentation
            # Five spatial positions, left-top, right-top, left-bottom, right-bottom, center
            # For all spatial position perform mirroring
            for m in [True, False]:
                for c in range(0,5):
                    # net.blobs['data'].data.shape
                    ts=transformStack(flowStack, transformer,  c, m, False)
                    imgs.append(ts)

        # Run forward pass with batch 50
        net.blobs['data'].reshape(batch_size, channels, width, height)
        net.blobs['data'].data[...] = imgs

        #print "Runnig forward pass"
        out = net.forward()
        classes = out['prob']
        if allClasses == []:
            allClasses = np.copy(classes)
        else:
            allClasses = np.append(allClasses, classes, axis=0)

        print b,' ',
        showResult(classes)

    end = time.time()
    print "TOOK: ", (end - start)

    return allClasses

# @profile
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
    meanValue = 128.0
    channels = stack_size*2
    meanImg = np.array([[np.transpose([meanValue for i in range(width)])]*height] * channels, dtype='float32')

    global transformer
    if transformer is None:
        transformer = initTransformer(meanImg, net.blobs['data'].data.shape, False)

    start = time.time()

    allClasses = []
    for b in range(0, samples/batch_num):
        imgs = []
        for s in range(0, batch_num):

            sample = images[s*b]
            flowStack = loadStack(sample, False)

            # Perform 10 data augmentation
            # Five spatial positions, left-top, right-top, left-bottom, right-bottom, center
            # For all spatial position perform mirroring
            for m in [False,True]:
                for c in range(0,5):
                    ts=transformStack(flowStack, transformer, c, m, True)
                    imgs.append(ts)

        # Run forward pass with batch 50
        net.blobs['data'].reshape(batch_size, channels, width, height)
        net.blobs['data'].data[...] = imgs

        # end = time.time()
        # print "Imgs loading & transformation took: ", (end - start)

        # start = time.time()

        #print "Runnig forward pass"
        out = net.forward()
        classes = out['prob']
        if allClasses == []:
            allClasses = np.copy(classes)
        else:
            allClasses = np.append(allClasses, classes, axis=0)

        print b,' ',
        showResult(classes)

    end = time.time()
    print "TOOK: ", (end - start)

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

    if 1:
        classResultAvg = np.sum(result.transpose(),1)/result.shape[0]
        classId = classResultAvg.argmax()
    else:
        classResultAvg = np.array([0.0]*result.shape[1])

        for c in result:
            id = c.argmax()
            val = c[id]
            classResultAvg[id] += val

        classResultAvg = classResultAvg / result.shape[0]
        classId = classResultAvg.argmax()

    # print labels[classId]

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

    return classId


def go(gpuid, flow, no_out):

    flow_proto = app_dir+'/models-proto/two-streams-nvidia/vgg_16_flow_deploy.prototxt'
    flow_model = app_dir+'/models-bin/two-streams_16_split1_flow-nvidia_iter_26000.caffemodel'
    flow_model = app_dir+'/models-bin/cuhk_action_temporal_vgg_16_split1.caffemodel'

    rgb_proto = app_dir+'/models-proto/two-streams-nvidia/vgg_16_rgb_deploy.prototxt'
    rgb_model = app_dir+'/models-bin/two-streams_vgg_16_split1_rgb-nvidia_iter_10000.caffemodel'
    #rgb_model = app_dir+'/models-bin/cuhk_action_spatial_vgg_16_split1.caffemodel'


    video_path = dar_root+'/share/datasets/UCF-101/UCF101-rgbflow/FloorGymnastics/v_FloorGymnastics_g16_c04'
    video_path = dar_root+'/share/datasets/UCF-101/UCF101-rgbflow/Shotput/v_Shotput_g07_c01'
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
        print 'TEMPORAL'
    else:
        outFilename = test_result_rgb_filename
        net = caffe.Net(rgb_proto, rgb_model, caffe.TEST)
        print 'SPATIAL'


    file_out = None
    if not no_out:
        file_out = open(outFilename, 'r+t')
        file_out.seek(0, io.SEEK_SET)


    correct = 0
    test_video_len = len(test_videos)
    for v in range(0, test_video_len):

        test_video = test_videos[v]
        #test_video[1] = video_path

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
        print v, "/",test_video_len,"  Classifying: ", test_video[0], ' ', videoDir
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



if __name__ == "__main__":

    run_type = sys.argv[1]

    dry_run = False
    if len(sys.argv) > 2:
        dry_run = sys.argv[2] == 'v'

    if run_type == 't':
        go(0, True, dry_run)	# Temporal

    if run_type == 's':
        go(1, False, dry_run) # Spatial

