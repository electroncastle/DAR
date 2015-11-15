#!/usr/bin/python
from gst.video import video_convert_frame

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


def get_rgb_mean_image(width, height):
    #rgb_mean=[104.0, 117.0, 123.0]
    #rgb_mean = [ 104.00698793,  116.66876762,  122.67891434]
    #rgb_mean=[127.0, 127.0, 127.0]

    # BGR
    rgb_mean=[62.548199072190982, 74.875980697732018, 82.795824585354055]

    # RGB
    #rgb_mean=[82.795824585354055, 74.875980697732018, 62.548199072190982]
    mean_image = np.array([[np.transpose([rgb_mean[0] for i in range(width)])]*height,
                [np.transpose([rgb_mean[1] for i in range(width)])]*height,
                [np.transpose([rgb_mean[2] for i in range(width)])]*height,

                [np.transpose([rgb_mean[0] for i in range(width)])]*height,
                [np.transpose([rgb_mean[1] for i in range(width)])]*height,
                [np.transpose([rgb_mean[2] for i in range(width)])]*height], dtype='float32')

    return mean_image

# 17.4769976166

iter = 0


def get_video_image_ids(path):

    path, image = os.path.split(path)
    path, video = os.path.split(path)

    image, ext = os.path.splitext(image)

    img_id = image.split('_')[1]
    vid_id = video.split('_')[-1]

    return vid_id, img_id


def get_lmdb_key(path):
    vid_id, img_id = get_video_image_ids(path)

    key = '{:0>5d}{:0>5d}'.format(int(vid_id), int(img_id))
    return key


wnd_x=10
wnd_y=10

def show_of(data, wnd_name):
    global wnd_x
    global wnd_y

    if data == None:
        return

    if data.shape[0] == 3:
        # RGB image
        image = data.astype(np.uint8)
        image = image.transpose(1,2,0)
        filename = 'flow-'+wnd_name+'.jpg'
        cv2.imwrite(filename, image)
        cv2.imshow(wnd_name+" Flow RGB", image)
        cv2.moveWindow(wnd_name+" Flow RGB", wnd_x, wnd_y)
        wnd_x=10
        wnd_y+=300
        return

    # data *= 100.0
    #data += 127.0

    image = data.astype(np.uint8)

    filename = wnd_name+'.jpg'
    cv2.imwrite(filename, image[0])

    ang, mag = utils.of2polar(data[0], data[1])
    img = utils.of2img(ang,mag)
    cv2.imwrite(filename, img)
    cv2.imshow(wnd_name, img)
    cv2.moveWindow(wnd_name, wnd_x, wnd_y)

    filename = wnd_name+'-x.jpg'
    cv2.imwrite(filename, image[0])
    cv2.imshow(wnd_name+"-X", image[0])
    wnd_x+=300
    cv2.moveWindow(wnd_name+"-X", wnd_x, wnd_y)

    filename = wnd_name+'-y.jpg'
    cv2.imwrite(filename, image[1])
    cv2.imshow(wnd_name+"-Y", image[1])
    wnd_x+=300
    cv2.moveWindow(wnd_name+"-Y", wnd_x, wnd_y)

    wnd_x=10
    wnd_y+=300

    # cv2.waitKey(0)


def show_img(data, suffix):
    global wnd_x
    global wnd_y

    if data == None:
        return

    image = data.astype(np.uint8)

    if image.shape[0] == 6:
        img = image.transpose(1,2,0)
        img1 = img[:,:,:3]
        img2 = img[:,:,3:]
    else:
        img1 = image[0]
        img2 = image[1]

    filename = 'img-1-'+suffix+'.jpg'
    cv2.imwrite(filename, img1)
    cv2.imshow("IMG 1", img1)
    cv2.moveWindow("IMG 1", wnd_x, wnd_y)

    filename = 'img-2-'+suffix+'.jpg'
    cv2.imwrite(filename, img2)
    cv2.imshow("IMG 2", img2)
    wnd_x+=300
    cv2.moveWindow("IMG 2",  wnd_x, wnd_y)

    wnd_x=10
    wnd_y+=300


# @profile
def classTemporal(net, img1_file, img2_file, xflow_file, yflow_file, iter):

    grayscale = True

    key = get_lmdb_key(img1_file)

    dataset = 'train'

    key = "0000000047"
    #key = "0000000550"
    #key = "0000000280"
    #key = "0000000380"
    #key = "0000000320"

    print key

    # classify augumented data
    width = 224
    height = 224
    mean_image = get_rgb_mean_image(width, height)

    # global transformer
    # if transformer is None:
    #     transformer = initTransformer(meanImg, net.blobs['data'].data.shape, False)

    start = time.time()

    # print sample
    #flowStack = loadStack(sample, False)

    batch = [] #np.empty([channels,img.shape[0], img.shape[1]])
    if grayscale:
        if net.blobs['data'].data.shape[1] == 6:
            # 2xRGB
            data_img = utils.get_data_lmdb("../"+dataset+"-of-imgs-rgb-mpi", key)
        else:
            # 2xgrayscale
            data_img = utils.get_data_lmdb("../"+dataset+"-of-imgs-mpi", key)

        batch_croppped = data_img - 70.0 # 85
        batch_size = 1
        channels = data_img.shape[0]
        width = data_img.shape[2]
        height = data_img.shape[1]
    else:
        color = True
        # Loads imgs in RGB format
        # Caffe needs BGR !!
        img1 = caffe.io.load_image(img1_file, color)*255.0
    #    img1 = img1.transpose((1,0,2))

        # im = PIL.Image.open(os.path.join(img1_file)).convert('RGB')
        # image = np.array(im, dtype=np.float64)

        img2 = caffe.io.load_image(img2_file, color)*255.0
    #    img2 = img2.transpose((1,0,2))

        for c in range(0, img1.shape[2],  1):
            batch.append(img1[:,:,2-c])
        for c in range(0, img2.shape[2],  1):
            batch.append(img2[:,:,2-c])



        batch = np.asarray(batch)

        h = batch[0].shape[0]
        w = batch[0].shape[1]
        t = (h-height)/2
        l = (w-width)/2
        batch_croppped = batch[:,t:t+height, l:l+width]


        batch_croppped = batch_croppped.astype(float)
        batch_croppped = np.subtract(batch_croppped, mean_image)

        # Run forward pass with batch 50
        batch_size = 1
        channels = 6

    net.blobs['data'].reshape(batch_size, channels, width, height)
    net.blobs['data'].data[...] = batch_croppped

        # end = time.time()
        # print "Imgs loading & transformation took: ", (end - start)

        # start = time.time()

        #print "Runnig forward pass"
    out = net.forward()
    out_layers = out.keys()
    out_layer = out_layers[0]
    print 'Reading from layer: ',out_layer
    #classes = out['prob']
#    of_vec = out['fc7'][0]
#    of_vec = out['conv7'][0]
#    of_vec = out['fc11'][0]
#    of_vec = out['fc8-1'][0]
    #of_vec = out['fc9-conv'][0]
    of_vec = out[out_layer][0]

    #of_vec = net.blobs['fc8-conv'].data[0]

    print of_vec.shape
    print "out min=",of_vec.min(),'  max=',of_vec.max()

    if 0:
       layers = of_vec[:64]
       mag = of_vec[-1]
       print layers.shape 
       print of_vec.min(),' ',of_vec.max()
       print mag.min(),' ',mag.max()
       
       mag *= 452.3
       angle, mag  = utils.layer2polar(layers, mag)
       utils.of2jpg('flow.jpg', angle, mag)
       x_flow, y_flow = utils.polar2of(angle, mag)
       pil = PIL.Image.fromarray(np.uint8(x_flow))
       pil.save('x_flow'+str(iter)+'.jpg', format='JPEG', quality=90)
       pil = PIL.Image.fromarray(np.uint8(y_flow))
       pil.save('y_flow'+str(iter)+'.jpg', format='JPEG', quality=90)    
       sys.exit()   
       




    scale = 1.0 #427.0/127
    flow_mean = 127.0
    of_vec_src = of_vec.flatten()

    if 1:
        x_flow = (of_vec[0]+flow_mean)*scale
        y_flow = (of_vec[1]+flow_mean)*scale
    else:
        #of_vec = np.transpose(of_vec, (1,2,0))
        of_vec = of_vec.flatten()

        # vec1 = of_vec[:256,:,:].flatten()
        # vec2 = of_vec[256:,:,:].flatten()
        # of_vec = np.append(vec1, vec2)

        print "out min=",of_vec.min(),'  max=',of_vec.max()

        of_vec = (of_vec+flow_mean)*scale
        #of_vec = (of_vec)+flow_mean

        # x_flow = of_vec[0:1024].reshape(32,32)
        # y_flow = of_vec[1024:].reshape(32,32)
        x_flow = of_vec[0:50176].reshape(224, 224)
        y_flow = of_vec[50176:].reshape(224, 224)


    show_img(data_img, key+'-'+iter)
    data_img = utils.get_data_lmdb("../"+dataset+"-of-labels-rgb-mpi", key)
    show_of(data_img, 'GT-'+key+'-'+iter)
#    show_of((of_vec+196.0), 'CNN') # 217.0
    show_of((of_vec+192.0), 'CNN-'+key+'-'+iter) # 217.0
    #217.0)*1, 'CNN')
    cv2.waitKey(0)
    sys.exit()


    pil = PIL.Image.fromarray(np.uint8(x_flow))
    pil.save('x_flow'+str(iter)+'.jpg', format='JPEG', quality=90)

    pil = PIL.Image.fromarray(np.uint8(y_flow))
    pil.save('y_flow'+str(iter)+'.jpg', format='JPEG', quality=90)

    x_flow7 = cv2.resize(np.uint8(x_flow), (244,244))
    y_flow7 = cv2.resize(np.uint8(y_flow), (244,244))
#     cv2.imshow("X FLOW", x_flow7)
#     cv2.imshow("Y_FLOW", y_flow7)


   # sys.exit(0)

#    cv2.waitKey(0)


    # xflow_gt = np.load(xflow_file)
    # yflow_gt = np.load(yflow_file)
    # vec_gt = np.append(xflow_gt.flatten(), yflow_gt.flatten())

    lmdb_name = '../train-of-labels-lmdb-full-mpi-large'
    #key = '0000102200'
    lmdb_env = lmdb.open(lmdb_name, map_size=int(1e12))
    lmdb_txn = lmdb_env.begin()  # equivalent to mdb_txn_begin()
    lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()
    # lmdb_cursor.get('{:0>10s}'.format('_6')) #  get the data associated with the 'key' 1, change the value to get other images

    lmdb_cursor.first()
    lmdb_cursor.get(key)
    value = lmdb_cursor.value()

    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)

    data = np.array(datum.float_data).astype(float).reshape(datum.channels, datum.height, datum.width)
    data_img = (data+1.0) * 127.0
#    data_img = (data+ 127.0)
    x_flow = data_img[0]
    y_flow = data_img[1]
    x_flow7 = cv2.resize(np.uint8(x_flow), (244,244))
    y_flow7 = cv2.resize(np.uint8(y_flow), (244,244))
#     cv2.imshow("X GT", x_flow7)
#     cv2.imshow("Y GT", y_flow7)


    d = data.flatten()
    y = (of_vec_src.flatten()-0.5)*2.0

    d = (data.flatten()+1.0)*0.5
    y = of_vec_src.flatten()
    print "GT L2"
    #data_one = (data+1.0)*0.5
    data_one = d
    diff = y - data_one
    t = np.dot(diff, diff)/2.0
    print t
    dist = np.sqrt(t)
    print dist

    print "AWG"
    #data_one = (data.flatten()+1.0)*0.5
    data_one = d
#    data_noise = ((data_one-0.5) + np.random.rand(len(of_vec_src))*0.23)+0.5
    data_noise = (data_one) + (np.random.rand(len(of_vec_src))-0.5)*0.2
    diff = data_noise.flatten() - data_one.flatten()
    t = np.dot(diff, diff)/2.0
    print t
    dist = np.sqrt(t)
    print dist

    print "Random"
    diff = (np.random.rand(len(d)).flatten()*2.0 - 1) - data_one.flatten()
    t = np.dot(diff, diff)/2.0
    print t
    dist = np.sqrt(t)
    print dist

    print "Mean"
    diff = np.array([0.58]*len(d)) - data_one.flatten()
    t = np.dot(diff, diff)/2.0
    print t
    dist = np.sqrt(t)
    print dist

    cv2.waitKey(0)
    return

def loadTestVideos(filename, path):

    dset = open(filename, 'rt')
    print filename

    videos = []
    for line in dset:
        toks = line.split(' ')
        videos.append([toks[2].strip(), path+toks[0].strip()])

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

    flow_proto = app_dir+'/models-proto/VGG_CNN_M_2048_OFR/VGG_CNN_M_2048_OFR_deploy-3.prototxt'
    flow_model = app_dir+'/models-bin/two-streams_16_split1_flow-nvidia_iter_26000.caffemodel'
    flow_model = app_dir+'/VGG_CNN_M_2048_OFR_iter_144.caffemodel'

    flow_proto = app_dir+'/models-proto/VGG_CNN_M_2048_OFR/OFR_GOOGLE_25_deploy.prototxt'
    flow_model = app_dir+'/VGG_CNN_M_2048_OFR_iter_704.caffemodel'

    flow_proto = app_dir+'/models-proto/VGG_CNN_M_2048_OFR/OFR_DeepMotion_deploy.prototxt'
    flow_model = app_dir+'/OFR_DeepMotion_iter_7000.caffemodel'

    # flow_proto = app_dir+'/models-proto/VGG_CNN_M_2048_OFR/VGG_CNN_M_2048_OFR_deploy.prototxt'
    # flow_model = app_dir+'/VGG_CNN_M_2048_OFR_iter_1000.caffemodel'

    flow_proto = app_dir+'/models-proto/VGG_CNN_M_2048_OFR/OFR_FlowNet_deploy.prototxt'
    flow_model = app_dir+'/OFR_FlowNet_iter_14000.caffemodel'


    flow_proto = app_dir+'/models-proto/VGG_CNN_M_2048_OFR/OFR_VGG16_6_deploy.prototxt'
    flow_model = app_dir+'/OFR_VGG16_6_iter_13000.caffemodel'

    flow_proto = app_dir+'/models-proto/VGG_CNN_M_2048_OFR/OFR_VGG16_6_fc14_deploy.prototxt'
    flow_model = app_dir+'/OFR_VGG16_6_fc_iter_18000.caffemodel'
    flow_model = app_dir+'/OFR_VGG16_6_fc14_iter_25000.caffemodel'

    flow_proto = app_dir+'/models-proto/VGG_CNN_M_2048_OFR/OFR_VGG16_6_fc_d_deploy.prototxt'
    proto_name = 'OFR_VGG16_6_fc_d_iter_3000'
    flow_model = app_dir+'/'+proto_name+'.caffemodel'

    global iter
    iter = 0000

    if len(sys.argv) > 1:
        iter = sys.argv[1]

    flow_proto = 'deploy.prototxt'
    flow_model = 'snapshot_iter_'+str(iter)+'.caffemodel'

#-----------------------------------------------------------------------------------------------------
    dataset_dir = dar_root+'/share/datasets/'
    rgbflow_path = dataset_dir+'/UCF-101/UCF101-rgbflow/'
    #rgbflow_path = dataset_dir+'/UCF-101/ucf101_flow_img_tvl1_gpu/'

    video_path = rgbflow_path+'/FloorGymnastics/v_FloorGymnastics_g16_c04'
    video_path = rgbflow_path+'/Shotput/v_Shotput_g05_c07'
    #video_path = '/home/jiri/Lake/HAR/datasets/UCF-101/UCF101-rgbflow/Diving/v_Diving_g22_c06'
    #video_path = '/home/jiri/Lake/HAR/datasets/UCF-101/UCF101-rgbflow/IceDancing/v_IceDancing_g03_c03'

    test_data_filename = dataset_dir+'/UCF-101/val-1-rnd.txt'

    test_result_flow_filename = dataset_dir+'/UCF-101/val-1-rnd-result-flow.txt'
    test_result_rgb_filename = dataset_dir+'/UCF-101/val-1-rnd-result-rgb.txt'

    imagenet_labels_filename = dataset_dir+'/UCF-101/labels-new.txt'


    # try:
    #     labels = np.loadtxt(imagenet_labels_filename, str, delimiter=' ')
    # except:
    #     return
    #
    # test_videos = loadTestVideos(test_data_filename, rgbflow_path)


    img1 = '/home/jiri/Lake/DAR/share/datasets/THUMOS2015/thumos15_validation-rgbflow/thumos15_video_validation_0001340/image_0266.jpg'
    img2 = '/home/jiri/Lake/DAR/share/datasets/THUMOS2015/thumos15_validation-rgbflow/thumos15_video_validation_0001340/image_0267.jpg'

    scene_name = 'temple_2_0000001'
    img1_name = 'image_002200'

    if len(sys.argv) > 2:
        img1_name = sys.argv[2]

    img_num = int(img1_name.split('_')[1])
    img2_num = img_num+100

    img1 = '/home/jiri/Lake/DAR/share/datasets/MPI_Sintel/MPI_Sintel-rgb/clean/'+scene_name+'/'+img1_name+'.jpg'
    img2 = '/home/jiri/Lake/DAR/share/datasets/MPI_Sintel/MPI_Sintel-rgb/clean/'+scene_name+'/image_{:0>6d}'.format(img2_num)+'.jpg'

    xflow_file = '/home/jiri/Lake/DAR/share/datasets/MPI_Sintel/MPI_Sintel-rgbflow/'+scene_name+'/flow_x_0021.npy'
    yflow_file = '/home/jiri/Lake/DAR/share/datasets/MPI_Sintel/MPI_Sintel-rgbflow/'+scene_name+'/flow_y_0021.npy'

    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(gpuid)

    outFilename = test_result_flow_filename
    net = caffe.Net(flow_proto, flow_model, caffe.TEST)
    print 'TEMPORAL'

    classTemporal(net, img1, img2, xflow_file, yflow_file, str(iter))
    sys.exit(0)

    file_out = None
    if not no_out:
        if not os.path.isfile(outFilename):
            file_out = open(outFilename, 'wt')
            file_out.close()
        file_out = open(outFilename, 'r+t')
        file_out.seek(0, io.SEEK_SET)


    correct = 0
    test_video_len = len(test_videos)
    for v in range(0, test_video_len):

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
    #run_type = sys.argv[1]

    dry_run = False
    if len(sys.argv) > 2:
        dry_run = sys.argv[2] == 'v'

    go(1, True, dry_run)	# Temporal


