#!/usr/bin/python
from gst.video import video_convert_frame

__author__ = "Jiri Fajtl"
__email__ = "ok1zjf@gmail.com"
__status__ = "Research"
__license__ = "LGPL"
__date__ = "20/10/2015"
__version__ = "1.0.0"



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
import gc
import epe

if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set, exiting'
    sys.exit()

dar_root = os.environ['DAR_ROOT']
app_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

sys.path.insert(0, dar_root + '/python')
import caffe


sys.path.insert(0, dar_root+'/src/ofEval/build-debug/')
import ofEval_module as em

NO_LABEL = -1
XY_LABEL = 0
XYM_LABEL = 1
RGB_LABEL= 2
XXYY_LABEL = 3

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



wnd_x=10
wnd_y=10


def xym2xy(img, mag_scale=1.0):
    mag = img[2]/mag_scale
    x_flow = img[0]*mag/127.0
    y_flow = img[1]*mag/127.0
    return x_flow, y_flow


def show_of(data, wnd_name, show, label_type, gt=False, mepe=-1):
    global wnd_x
    global wnd_y

    gt_scale = 28

    if data == None:
        return

    if label_type == XXYY_LABEL:
        # LRBT

        data = data.astype(np.float32)
        xf = -1*data[0]
        xf += data[1]
        yf = -1*data[2]
        yf += data[3]
        image = np.array([xf, yf])

        # If it's ground truth scale down for fair comparison
        if gt and gt_scale>0:
            xf = cv2.resize(xf, (gt_scale, gt_scale))
            xf = cv2.resize(xf, (224,224))
            yf = cv2.resize(yf, (gt_scale, gt_scale))
            yf = cv2.resize(yf, (224,224))
            image = np.array([xf, yf])
            clr_flow = em.flowToColor(image[0], image[1], 10)
        else:
            clr_flow = em.flowToColor(image[0], image[1], 10)

        cv2.imshow(wnd_name+" Flow", clr_flow)
        cv2.moveWindow(wnd_name+" Flow", wnd_x, wnd_y)
        wnd_x+=250
        filename = 'flow-'+wnd_name+'.jpg'
        cv2.imwrite(filename, clr_flow)

        image += 127
        image = image.astype(np.uint8)
        cv2.imshow(wnd_name+" Flow X", image[0])
        cv2.moveWindow(wnd_name+" Flow X", wnd_x, wnd_y)
        wnd_x+=250

        cv2.imshow(wnd_name+" Flow Y", image[1])
        cv2.moveWindow(wnd_name+" Flow Y", wnd_x, wnd_y)
        wnd_x=10
        wnd_y+=300
        return

    if data.shape[0] == 3:

        if label_type == XYM_LABEL:
            #Only for XYM format !!!!
            image = data.astype(np.float32)
            image[0], image[1] = xym2xy(image, 1.0)

            clr_flow = em.flowToColor(image[0], image[1], 0)
            cv2.imshow(wnd_name+" Flow", clr_flow)
            cv2.moveWindow(wnd_name+" Flow", wnd_x, wnd_y)
            wnd_x+=250
            filename = 'flow-'+wnd_name+'.jpg'
            cv2.imwrite(filename, clr_flow)


            image += 127
            image = image.astype(np.uint8)
            cv2.imshow(wnd_name+" Flow X", image[0])
            cv2.moveWindow(wnd_name+" Flow X", wnd_x, wnd_y)
            wnd_x+=250

            cv2.imshow(wnd_name+" Flow Y", image[1])
            cv2.moveWindow(wnd_name+" Flow Y", wnd_x, wnd_y)
            wnd_x=10
            wnd_y+=300
            return


        if label_type == RGB_LABEL:
            # RGB image
            image = data.astype(np.uint8)
            image = image.transpose(1,2,0)
            filename = 'flow-'+wnd_name+'.jpg'
            cv2.imwrite(filename, image)
            if show:
               if gt and gt_scale>0:
                    image = cv2.resize(image, (gt_scale, gt_scale))
                    image = cv2.resize(image, (224, 224))
               cv2.imshow(wnd_name+" Flow RGB", image)
               cv2.moveWindow(wnd_name+" Flow RGB", wnd_x, wnd_y)
               wnd_x=10
               wnd_y+=300
            return

    if label_type == XY_LABEL:
    # if data.shape[0] == 2:
        data = data*1.0/0.5
        image = data.astype(np.float32)

        if gt and gt_scale>0:
            xf = cv2.resize(image[0], (gt_scale, gt_scale))
            xf = cv2.resize(xf, (224,224))
            yf = cv2.resize(image[1], (gt_scale, gt_scale))
            yf = cv2.resize(yf, (224,224))
            image = np.array([xf,yf])
            clr_flow = em.flowToColor(image[0], image[1], 15)
        else:
            clr_flow = em.flowToColor(image[0], image[1], -5)
#            clr_flow = em.flowToColor(image[0], image[1], 10)
            #clr_flow = cv2.resize(clr_flow, (gt_scale, gt_scale))


        if mepe > -1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(clr_flow, "EPE: {:.2f}".format(mepe), (int(clr_flow.shape[1]-100), 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow(wnd_name+" Flow", clr_flow)
        cv2.moveWindow(wnd_name+" Flow", wnd_x, wnd_y)
        filename = 'flow-'+wnd_name+'.jpg'
        cv2.imwrite(filename, clr_flow)
        wnd_x+=250

        image += 127
        image = image.astype(np.uint8)
        cv2.imshow(wnd_name+" Flow X", image[0])
        cv2.moveWindow(wnd_name+" Flow X", wnd_x, wnd_y)
        wnd_x+=250

        cv2.imshow(wnd_name+" Flow Y", image[1])
        cv2.moveWindow(wnd_name+" Flow Y", wnd_x, wnd_y)
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
    if show:
       cv2.imshow(wnd_name, img)
       cv2.moveWindow(wnd_name, wnd_x, wnd_y)

    filename = wnd_name+'-x.jpg'
    cv2.imwrite(filename, image[0])
    if show:
       cv2.imshow(wnd_name+"-X", image[0])
       wnd_x+=300
       cv2.moveWindow(wnd_name+"-X", wnd_x, wnd_y)

    filename = wnd_name+'-y.jpg'
    cv2.imwrite(filename, image[1])
    if show:
       cv2.imshow(wnd_name+"-Y", image[1])
       wnd_x+=300
       cv2.moveWindow(wnd_name+"-Y", wnd_x, wnd_y)
       wnd_x=10
       wnd_y+=300

    # cv2.waitKey(0)


def show_img(data, suffix, show):
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
    if show:
       cv2.imshow("IMG 1", img1)
       cv2.moveWindow("IMG 1", wnd_x, wnd_y)

    filename = 'img-2-'+suffix+'.jpg'
    cv2.imwrite(filename, img2)
    if show:
       cv2.imshow("IMG 2", img2)
       wnd_x+=300
       cv2.moveWindow("IMG 2",  wnd_x, wnd_y)
       wnd_x=10
       wnd_y+=300


# @profile

def loadNetwork(iter, gpuid):

    flow_proto = 'deploy.prototxt'
    flow_model = 'snapshot_iter_'+str(iter)+'.caffemodel'

    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(gpuid)

    net = caffe.Net(flow_proto, flow_model, caffe.TEST)

    return net


def forward(net, batch, mean_image):

    start = time.time()

    batch = np.subtract(batch, mean_image)

    # Run forward pass
    batch_size = 1
    height = batch[0].shape[0]
    width = batch[0].shape[1]
    channels = batch.shape[0]

    net.blobs['data'].reshape(batch_size, channels, width, height)
    net.blobs['data'].data[...] = batch

        # end = time.time()
        # print "Imgs loading & transformation took: ", (end - start)

        # start = time.time()

        #print "Runnig forward pass"
    out = net.forward()
    out_layers = out.keys()
    out_layer = out_layers[0]
    #print 'Reading from layer: ',out_layer
    #classes = out['prob']
#    of_vec = out['fc7'][0]
#    of_vec = out['conv7'][0]
#    of_vec = out['fc11'][0]
#    of_vec = out['fc8-1'][0]
    #of_vec = out['fc9-conv'][0]
    of_vec = out[out_layer][0]

    return of_vec


def batch_from_images(img1_filename, img2_filename, color=True):

    img1 = caffe.io.load_image(img1_filename, color)*255.0
#    img1 = img1.transpose((1,0,2))

    # im = PIL.Image.open(os.path.join(img1_file)).convert('RGB')
    # image = np.array(im, dtype=np.float64)

    img2 = caffe.io.load_image(img2_filename, color)*255.0

    batch = [] #np.empty([channels,img.shape[0], img.shape[1]])

    for c in range(0, img1.shape[2],  1):
        batch.append(img1[:,:,2-c])

    for c in range(0, img2.shape[2],  1):
        batch.append(img2[:,:,2-c])

    batch = np.asarray(batch).astype(float)

    return batch


def crop(batch, width, height):
    # Crop around the center by given width, height
    h = batch[0].shape[0]
    w = batch[0].shape[1]

    if width>-1 and height >-1:
        t = (h-height)/2
        l = (w-width)/2
        batch_croppped = batch[:,t:t+height, l:l+width]
        # batch_croppped = batch_croppped.astype(float)

    return batch_croppped


def calc_epe(gt_flow_img, flow_img):
    if gt_flow_img is None:
        return -1.0

    # if flow_img.shape[0] != 14:
    #     return -1.0

    print "gt_flow_img", gt_flow_img.shape
    print "flow_img", flow_img.shape


    data = gt_flow_img
#    data = data*10+127
    # data = data*1.0/0.5
    image = data.astype(np.float32)

#    flow_img = flow_img+127

    # Change the quality of the GT by down/up sampling it to the same
    # resolution
    gt_scale = 14
    xf = image[0]
    yf = image[1]
    # xf = cv2.resize(image[0], (gt_scale, gt_scale))
    # yf = cv2.resize(image[1], (gt_scale, gt_scale))
    # xf = cv2.resize(xf, (224,224))
    # yf = cv2.resize(yf, (224,224))
    image = np.array([xf,yf])

    # flow_img = flow_img*127+127
    # image = image*127+127

    epe_val, mepe_val = epe.calc_epe(image[1], image[0], flow_img[0], flow_img[1])
    print "MEPE ", mepe_val

    return mepe_val


def show_result(src_imgs, flow_img, gt_flow_img, flow_mean, name, label_type):

    flow_img = flow_img[::-1, :, :]

    show = True

    # Show source images, ground truth, and the calculated optical flow image
    show_img(src_imgs, name, show)
    show_of(gt_flow_img, 'GT-'+name, show, label_type, True)

    # for i in range(1,300):
    #     scale = 3.0+float(i) / 100
    #     print "S ",scale
    #     n_img = flow_img/scale
    #
    #     mepe = calc_epe(gt_flow_img, (n_img+flow_mean))

    scale=2.32
#    scale=7.32
    scale=8.0
    flow_mean = 127.0
    #flow_img *= scale
    #flow_img /= 4.79
    #mepe = calc_epe(gt_flow_img, (flow_img/5.23+flow_mean))

    mepe = -1
    mepe = calc_epe(gt_flow_img*10+127.0, flow_img*scale+flow_mean)

    if label_type == XY_LABEL:
        flow_img /= 6
        v = cv2.resize((flow_img[0]), (224, 224))
        u = cv2.resize((flow_img[1]), (224, 224))
        flow_img = np.array([u, v])

    if label_type == RGB_LABEL:
        flow_img = flow_img[::-1, :, :]
        flow_img += 217.0
        flow_img *= 0.9

#    show_of((of_vec+196.0), 'CNN') # 217.0
    show_of(flow_img, 'CNN-'+name, show, label_type, mepe=mepe) # 217.0
    #217.0)*1, 'CNN')


    return


def get_keys(lmdb_label_name):

    print "Rading lmdb: ",lmdb_label_name

    lmdb_env = lmdb.open(lmdb_label_name, map_size=int(1e12))
    lmdb_txn = lmdb_env.begin()  # equivalent to mdb_txn_begin()
    lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()
    # lmdb_cursor.get('{:0>10s}'.format('_6')) #  get the data associated with the 'key' 1, change the value to get other images

    print lmdb_env.info()

    keys = []
    count = 0
    gc.disable()
    for key, value in lmdb_cursor:
        count += 1
        #print count,'  ',(key)
        # data = get_data(value)
        keys.append(key)
    gc.enable()

    return keys


def liveof(iter, label_type):
    global wnd_x
    global wnd_y

    if label_type == RGB_LABEL:
        lmdb_labels = "-of-labels-rgb-mpi_clean_final_90_10"
        img_mean = [70]
        flow_mean = 186.0
    elif label_type == XY_LABEL:
        lmdb_labels = "-of-labels-xy-mpi_clean_final_90_10"
        img_mean = [94]
        flow_mean = 127.0
    elif label_type == XYM_LABEL:
        lmdb_labels = "-of-labels-xym-mpi_clean_final_90_10"
        img_mean = [94]
        flow_mean = 2.0
    elif label_type == XXYY_LABEL:
        lmdb_labels = "-of-labels-xxyy-mpi_clean_final_90_10"
        img_mean = [94]
        flow_mean = 2.24156086712

    crop_size = [224, 224]
    net = loadNetwork(iter, 0)

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    prev = None
    while rval:
        # cv2.imshow("preview", frame)
        rval, frame = vc.read()

        if prev != None:
            batch = [] #np.empty([channels,img.shape[0], img.shape[1]])

            for c in range(0, prev.shape[2],  1):
                batch.append(prev[:,:,2-c])

            for c in range(0, frame.shape[2],  1):
                batch.append(frame[:,:,2-c])

            batch = np.asarray(batch).astype(float)

            batch = crop(batch, crop_size[0], crop_size[1])
            cv2.imshow("preview", batch[0:3].transpose(1,2,0).astype(np.uint8))

            flow_img = forward(net, batch, img_mean)

            wnd_x=10
            wnd_y=10
            show_of((flow_img+flow_mean), 'CNN', True, label_type) # 217.0

        prev = frame
        key = cv2.waitKey(20)
        if key == 1048603: # exit on ESC
            break
    cv2.destroyWindow("preview")
    return


def evaluateImages(iter, label_type, img1="", img2="", flo="", xflow="", yflow=""):
    net = loadNetwork(iter, 1)

    lmdb_images = "-of-imgs-m-rgb-mpi_clean_final_90_10"

    crop_size = [224, 224]

    if label_type == RGB_LABEL:
        lmdb_labels = "-of-labels-rgb-mpi_clean_final_90_10"
        img_mean = [70]
        flow_mean = 186.0
        img_mean = [97]
        #flow_mean = 100.0
    elif label_type == XY_LABEL:
        lmdb_labels = "-of-labels-m-xy-mpi_clean_final_90_10"
        img_mean = [94]
        img_mean = [194]
        flow_mean = 0.0
    elif label_type == XYM_LABEL:
        lmdb_labels = "-of-labels-xym-mpi_clean_final_90_10"
        img_mean = [94]
        flow_mean = 2.0
    elif label_type == XXYY_LABEL:
        lmdb_labels = "-of-labels-xxyy-mpi_clean_final_90_10"
        img_mean = [94]
        flow_mean = 2.24156086712

    batch = batch_from_images(img1, img2)

    batch = crop(batch, crop_size[0], crop_size[1])
    flow_img = forward(net, batch, img_mean)
    flow_img = net.blobs['conv10-1'].data[0]


    # Load ground truth
    gt_flow_img = None
    if flo != '':
        gt_flow_img=utils.read_flo_fast(flo)
        gt_flow_img = np.asarray([gt_flow_img[2], gt_flow_img[3]])
        gt_flow_img = crop(gt_flow_img, crop_size[0], crop_size[1])
        ind1 =  gt_flow_img > 1e8
        gt_flow_img[ind1] = 0

    if xflow != '' and yflow != '':
        gtv = cv2.imread(yflow, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        gtu = cv2.imread(xflow, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        gtv = 10*(gtv-127)/127.0
        gtu = 10*(gtu-127)/127.0
        gt_flow_img = np.array([gtu, gtv], dtype=np.float32)


    # gt_flow_img = utils.get_data_lmdb("../"+dataset+lmdb_labels, key)

    global wnd_x, wnd_y
    wnd_x=10
    wnd_y=10


    if img1[0] == '/':
        p, f = os.path.split(img1)
        p, d = os.path.split(p)
        f, e = os.path.splitext(f)
        name = d+'_'+f+'-'+str(iter)
    else:
        name = img1+'-'+str(iter)

    flow_mean = 10
    show_result(batch, flow_img/8.0, gt_flow_img, flow_mean, name, label_type)

    ch=cv2.waitKey(0)

    return


wait_time=0 # Wait for keypress
def evaluate(iter, label_type):
    global wait_time

    rgb_of = True
    rgb_of = False

    net = loadNetwork(iter, 1)

    clr_sgn = 'rgb'
    bw = False
    if net.blobs['data'].data.shape[1] == 2:
        bw = True
        clr_sgn = 'bw'


    lmdb_images = "-of-imgs-m-"+clr_sgn+"-mpi_clean_final_90_10"

    crop_size = [224, 224]

    if label_type == RGB_LABEL:
        lmdb_labels = "-of-labels-rgb-mpi_clean_final_90_10"
        img_mean = [70]
        flow_mean = 186.0
    elif label_type == XY_LABEL:
        lmdb_labels = "-of-labels-m-xy-mpi_clean_final_90_10"
        img_mean = [75]
        flow_mean = 0.0
    elif label_type == XYM_LABEL:
        lmdb_labels = "-of-labels-xym-mpi_clean_final_90_10"
        img_mean = [94]
        flow_mean = 2.0
    elif label_type == XXYY_LABEL:
        lmdb_labels = "-of-labels-xxyy-mpi_clean_final_90_10"
        img_mean = [94]
        flow_mean = 2.24156086712


    datasets = ['train', 'val']
    datasets = ['val', 'train']
    tests = ["0000000085", "0000000086", "0000000050", "0000000031",
             "0000000712", "0000000713", "0000000677", "0000000658",
             "0000009047", "0000005666", "0000000047", "0000000280", "0000000380", "0000000320", "0000000020", "0000000030", "0000000040",
             "0000000187", "0000000127", "0000000110", "0000000157", "0000000147"]

    key = "0000000047" # val=mountains  train market
    # key = "0000000550"	# val=plain=komori  train=arm
    key = "0000000280"  # val=komori train=dragon left
    key = "0000000380" # val=running
    key = "0000000320" # val=head

    #key = "0000000020" # val=fingers  train=jump on wall
    #key = "0000000030" # val=market2  train=shoulder
    #key = "0000000040" # val=komiri2  train=shoulder
    #key = "0000000187" # val=punch          train=drop
    #key = "0000000127" # val=bamboo         train=dragon head
    #key = "0000000110" # val=dragon leg     train=jump
    #key = "0000000157" # val=arm? C         train=mountain
    #key = "0000000147" # val=market3        train=body   N

    for dataset in datasets:
        #tests = get_keys("../"+dataset+lmdb_images)

        # Interleave with the mirrored images
        # testsm = []
        # for key in tests:
        #     k = int(key)
        #     testsm.append(key)
        #     if dataset=='train':
        #         k+=5619
        #     else:
        #         k+=627
        #     key = '{:0>10d}'.format(k)
        #     testsm.append(key)
        #
        # tests = testsm

        for key in tests:
            print "dataset=",dataset,"  Key=",key
            batch = data_img = utils.get_data_lmdb("../"+dataset+lmdb_images, key)
            batch = crop(batch, crop_size[0], crop_size[1])
            flow_img = forward(net, batch, img_mean)

            # Load ground truth
            gt_flow_img = utils.get_data_lmdb("../"+dataset+lmdb_labels, key)

            global wnd_x, wnd_y
            wnd_x=10
            wnd_y=10

            name = key+'-'+str(iter)
            #flow_img = net.blobs['conv10-1'].data[0]
            #flow_img = net.blobs['fc8-conv'].data[0]
            show_result(batch, flow_img, gt_flow_img, flow_mean, name, label_type)

            ch=cv2.waitKey(wait_time) & 0x00EFFFFF
            if ch == ord('q'): # 'q'
                sys.exit()
            if ch == ord('r'):
                wait_time = 100

            cv2.destroyAllWindows()

    return




def label_type_from_string(type):
    if type == 'xy':
        return XY_LABEL
    elif type == 'xym':
        return XYM_LABEL
    elif type == 'rgb':
        return RGB_LABEL
    elif type == 'xxyy':
        return XXYY_LABEL


import argparse


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Evaluates Optical Flow CNN model')
    parser.add_argument('iteration', metavar='N', type=int, #nargs='+',
                       help='Training iteration', default=-1)

    parser.add_argument("-l", "--live", help="Live view from webcam",
                    action="store_true")

    parser.add_argument("-t", "--type", help="OF type", default="", nargs='?')
    parser.add_argument("-i1", "--img1", help="First image", default="", nargs='?')
    parser.add_argument("-i2", "--img2", help="Second image", default="", nargs='?')
    parser.add_argument("-f", "--flow", help="flo file with ground truth flow", default="", nargs='?')
    parser.add_argument("-x", "--x_flow", help="image with x flow field", default="", nargs='?')
    parser.add_argument("-y", "--y_flow", help="image with y flow field", default="", nargs='?')

    args = parser.parse_args()

    if args.iteration <0:
        parser.print_help()
        sys.exit(2)

    label_type=XY_LABEL
    label_type=XYM_LABEL
#    label_type=RGB_LABEL
    label_type = XXYY_LABEL

    if args.type != '':
        label_type = label_type_from_string(args.type)

    if args.live:
        liveof(args.iteration, label_type)
        sys.exit()

    if args.img1 != "":
        evaluateImages(args.iteration, label_type, args.img1, args.img2, args.flow, args.x_flow, args.y_flow)
        sys.exit()

    evaluate(args.iteration, label_type)




