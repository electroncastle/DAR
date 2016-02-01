#!/usr/bin/python

__author__ = "Jiri Fajtl"
__email__ = "ok1zjf@gmail.com"
__status__ = "Research"
__license__ = "LGPL"
__date__ = "20/10/2015"
__version__ = "1.0.0"



import io
import os
import sys
import random
import cv2
import gc
import errno

import numpy as np
import PIL.Image
from cStringIO import StringIO
import struct

import lmdb
import cv2

import re, fileinput, math
import matplotlib.pyplot as plt
import utils


if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set'
    sys.exit()

caffe_root = os.environ['DAR_ROOT']
sys.path.insert(0, caffe_root + '/src/caffe/python')
import caffe
import caffe.io
from caffe.proto import caffe_pb2

sys.path.insert(0, '/home/jiri/Lake/DAR/src/ofEval/build-debug/')
import ofEval_module as em



NO_LABEL = -1
XY_LABEL = 0
XYM_LABEL = 1
RGB_LABEL= 2
XXYY_LABEL = 3

max_flow = -9999999999
min_flow = 9999999999
mean_flow = 0.0
mean_img = 0
count_flow = 0

s1=0.0
s2=0.0
max_mag = 0.0

def reset():
    global max_flow
    global min_flow
    global mean_flow
    global mean_img
    global count_flow
    global s1
    global s2
    global max_mag

    max_flow = -9999999999
    min_flow = 9999999999
    mean_flow = 0.0
    mean_img = 0
    count_flow = 0

    s1=0.0
    s2=0.0
    max_mag = 0.0


def img_mean(img):
    if len(img.shape) > 2:
        m = []
        for i in range(0,img.shape[0]):
            m.append(float(np.sum(img[i])) / (len(img[i].flatten())))

        return np.asarray(m)

    else:
        return float(np.sum(img)) / (len(img.flatten()))


def get_img(img_file, crop, scale, color = False, mirror=False):
    global max_flow
    global min_flow
    global mean_img

    cv_flags = cv2.IMREAD_GRAYSCALE
    if color:
        cv_flags = cv2.IMREAD_COLOR


    img1 =  cv2.imread(img_file[0], cv_flags)
#    img1 = img1.transpose((2, 0, 1))

    img2 =  cv2.imread(img_file[1], cv_flags)
#    img2 = img1.transpose((2, 0, 1))

    imgs1 = utils.crop_scale(img1, crop_offset, crop, scale, 0, mirror)
    imgs2 = utils.crop_scale(img2, crop_offset, crop, scale, 0, mirror)

    if color:
        imgs1 = imgs1.transpose(0, 3, 1, 2)
        imgs2 = imgs2.transpose(0, 3, 1, 2)
    else:
        imgs1=imgs1[:, np.newaxis, :, :]
        imgs2=imgs2[:, np.newaxis, :, :]

    datums = []

    if  len(imgs1)>0 and len(imgs1[0].shape) == 3 and imgs1[0].shape[0] == 3 and type(mean_img) == int:
        mean_img = [0.0, 0.0, 0.0]

    for i in range(0, len(imgs1)):

        mean_img = mean_img + (img_mean(imgs1[i]) + img_mean(imgs2[i])) / 2.0
        of_vec = np.append(imgs1[i], imgs2[i], axis=0)

        #datums.append(caffe.io.array_to_datum(of_vec.astype(float)))
        datums.append(caffe.io.array_to_datum(of_vec.astype(np.uint8)))

    return datums


def get_xym_image(x_flow, y_flow, mag_scale):
    global max_mag

    # Get magnitude
    cpx = x_flow+(y_flow*1j)
    mag = np.abs(cpx)
    max_mag = max(max_mag, np.max(mag))

    # Scale the flow fields to -127..127
    zero_id = mag == 0.0
    mag[zero_id] = 1.0 # The zero cells will be set to zero later on so this value is not important
    x_flow = 127.0*x_flow/mag
    y_flow = 127.0*y_flow/mag

    x_flow[zero_id] = 0
    y_flow[zero_id] = 0

    # Scale the mag
    mag = mag*mag_scale

    # Trim to range 0..255
    max_id = mag > 255.0
    mag[max_id] = 255.0

    overflow = np.count_nonzero(max_id)
    if overflow > 0:
        print "Overflow: ",overflow

    # cv2.imshow('mag', mag.astype(np.uint8))
    # cv2.waitKey(0)

    xym_flow = np.array([x_flow, y_flow, mag])
    xym_flow = xym_flow.transpose(1, 2, 0)

    return xym_flow



def get_label(img_file, crop, scale, alpha, beta, label_type, mirror):

    global max_flow
    global min_flow
    global mean_flow
    global s1
    global s2
    global max_mag

    if label_type == XY_LABEL:
        # 127.0/427.0
        alpha = 0.57
        beta = 0.0

        # width, height, x_flow, y_flow = utils.read_flo(img_file[2])
        width, height, x_flow, y_flow = utils.read_flo_fast(img_file[2])

        # clr_flow = em.flowToColor(x_flow.copy(), y_flow.copy(), 0)
        # cv2.imshow("flow", clr_flow)
        # cv2.waitKey(0)

        # x_flow = x_flow*alpha+beta
        # y_flow = y_flow*alpha+beta

         # Set invalid flow pixels to zero = no motion
        invalid = x_flow > 1e8
        x_flow[invalid] = 0.0
        invalid = y_flow > 1e8
        y_flow[invalid] = 0.0


        flows = utils.crop_scale_flow(x_flow, y_flow, crop_offset, crop, scale, mirror)

        max_flow = max(max_flow, flows.max())
        min_flow = min(min_flow, flows.min())


        # Translate and scale x/y flows
        if alpha != 1.0 or beta != 0.0:
            flows = flows*alpha+beta

        # Trim the x/y flows to given range
        trim_range = 127.0
        #trim_range = 0.0
        if trim_range > 0:
            min_id = flows < -trim_range
            max_id = flows > trim_range
            flows[min_id] = -trim_range
            flows[max_id] = trim_range

        flows = flows.astype(np.float)

    if label_type == XXYY_LABEL:
        # 127.0/427.0
        alpha = 1.0
        beta = 0.0

        # width, height, x_flow, y_flow = utils.read_flo(img_file[2])
        width, height, x_flow, y_flow = utils.read_flo_fast(img_file[2])

        # clr_flow = em.flowToColor(x_flow.copy(), y_flow.copy())
        # cv2.imshow("flow", clr_flow)
        # cv2.waitKey(0)

        # x_flow = x_flow*alpha+beta
        # y_flow = y_flow*alpha+beta

         # Set invalid flow pixels to zero = no motion
        invalid = x_flow > 1e8
        x_flow[invalid] = 0.0
        invalid = y_flow > 1e8
        y_flow[invalid] = 0.0


        flows = utils.crop_scale_flow(x_flow, y_flow, crop_offset, crop, scale)

        # make xx yy channels
        xxyy_flows=[]
        for flow in flows:
            xflow=flow[0]
            lxid = xflow < 0

            lx = xflow.copy()
            rx = xflow.copy()

            lx[~lxid] = 0
            rx[lxid] = 0

            yflow = flow[1]
            byid = yflow < 0

            by = yflow.copy()
            ty = yflow.copy()

            by[~byid] = 0
            ty[byid] = 0

            xxyy_flows.append([-1*lx, rx, -1*by, ty])

        flows = np.asarray(xxyy_flows)
        # flows = flows.transpose(0, 3, 1, 2)


        max_flow = max(max_flow, flows.max())
        min_flow = min(min_flow, flows.min())


        # Translate and scale x/y flows
        if alpha != 1.0 or beta != 0.0:
            flows = flows*alpha+beta

        # Trim the x/y flows to given range
        trim_range = 255.0
        #trim_range = 0.0
        if trim_range > 0:
            max_id = flows > trim_range
            flows[max_id] = trim_range

        flows = flows.astype(np.float)


    elif label_type == XYM_LABEL:

        # Create unity directional vector and magnitude
        # directional vector -127..+127
        # Magnitude 0..255

        mag_scale = 1.0 #0.55

        # width, height, x_flow, y_flow = utils.read_flo(img_file[2])
        width, height, x_flow, y_flow = utils.read_flo_fast(img_file[2])

        # clr_flow = em.flowToColor(x_flow.copy(), y_flow.copy())
        # cv2.imshow("flow", clr_flow)
        # cv2.waitKey(0)

        # Set invalid flow pixels to zero = no motion
        invalid = x_flow > 1e8
        x_flow[invalid] = 0.0
        invalid = y_flow > 1e8
        y_flow[invalid] = 0.0

        if 0:
            xym_flow = get_xym_image(x_flow, y_flow, mag_scale)
            flows = utils.crop_scale(xym_flow, crop_offset, crop, scale)
            flows = flows.transpose(0, 3, 1, 2)
        else:
            xy_flows = utils.crop_scale_flow(x_flow, y_flow, crop_offset, crop, scale)
            xym_flows=[]
            for flow in xy_flows:
                xym_flow = get_xym_image(flow[0], flow[1], mag_scale)
                xym_flows.append(xym_flow)

            flows = np.asarray(xym_flows)
            flows = flows.transpose(0, 3, 1, 2)

        flows = flows.astype(np.float)

    elif label_type == RGB_LABEL:

        dirs = img_file[2].split(os.sep)
        dirs[-3] = 'flow_viz'
        filename, ext = os.path.splitext(dirs[-1])
        dirs[-1] = filename+'.png'
        path = '/'.join(dirs)

        img1 = cv2.imread(path, True)
        flows = utils.crop_scale(img1, crop_offset, crop, scale)
        flows = flows.transpose(0, 3, 1, 2)


    #  Create Datum from all labels
    datums = []
    for flow in flows:

        # Collects some stats
        # cpx = flow[0]+(flow[1]*1j)
        # mag = np.abs(cpx)
        # max_mag = max(max_mag, np.max(mag))

        mean_flow += img_mean(flow)

        d = flow.flatten()
        s1 += np.sum(d)
        s2 += np.dot(d, d)

        # clr_flow = em.flowToColor(flow[0].copy(), flow[1].copy())
        # cv2.imshow("flow", clr_flow)
        # cv2.waitKey(0)

        datums.append(caffe.io.array_to_datum(flow))
        # datums.append(caffe.io.array_to_datum(flow.astype(np.uint8)))

    return datums


def write_db(lmdb_name, img_list, crop, scale, label_type, rgb_imgs = False):

        write_to_db = True


        print "Writing lmdb: ",lmdb_name
        count = 0
        dup_count = 0
        # Size of buffer: 1000 elements to reduce memory consumption
        block_elements = 1000.0
        blocks = int(math.ceil(len(img_list)/block_elements))
        # for idx in range(blocks):

        in_db_label = lmdb.open(lmdb_name, map_size=int(1e12))

        # block_first = int(block_elements)*idx
        # block_last =  int(block_elements)*(idx+1)

        in_idx=0
        mirror = False
        with in_db_label.begin(write=True) as in_txn:
            for img_file in img_list:

                if label_type != NO_LABEL:
                    # 127.0/427.0
                    alpha = 0.57
                    beta = 0.0
                    datums = get_label(img_file, crop, scale, alpha, beta, label_type, mirror)
                else:
                    datums = get_img(img_file, crop, scale, color=rgb_imgs, mirror=mirror)

                for datum in datums:
                    key = '{:0>10d}'.format(in_idx)
                    dat_str = datum.SerializeToString()

                    #print key
                    if write_to_db and not in_txn.put(key, dat_str, dupdata=False, overwrite=False):
                        dup_count+=1
                        # print
                        # print dup_count," Already exist:  ",key

                    count += 1
                    if 1: # (count % 1000) == 0 or count < 2000:
                        string_ = str(in_idx + 1) + ' / ' + str(len(img_list*len(datums)))
                        sys.stdout.write("\r%s" % string_)
                        sys.stdout.flush()

                    #print in_txn.stat(in_db_label)

                    if in_idx % 1000 == 0:
                        in_db_label.sync()

                    in_idx+=1

            # Generate mirrored images and flows
            mirror = True
            for img_file in img_list:

                if label_type != NO_LABEL:
                    # 127.0/427.0
                    alpha = 0.57
                    beta = 0.0
                    datums = get_label(img_file, crop, scale, alpha, beta, label_type, mirror)
                else:
                    datums = get_img(img_file, crop, scale, rgb_imgs, mirror)

                for datum in datums:
                    key = '{:0>10d}'.format(in_idx)
                    dat_str = datum.SerializeToString()

                    #print key
                    if write_to_db and not in_txn.put(key, dat_str, dupdata=False, overwrite=False):
                        dup_count+=1
                        # print
                        # print dup_count," Already exist:  ",key

                    count += 1
                    if 1: # (count % 1000) == 0 or count < 2000:
                        string_ = str(in_idx + 1) + ' / ' + str(len(img_list*len(datums)))
                        sys.stdout.write("\r%s" % string_)
                        sys.stdout.flush()

                    #print in_txn.stat(in_db_label)

                    if in_idx % 1000 == 0:
                        in_db_label.sync()

                    in_idx+=1

        in_db_label.close()

        print
        print "Records num: ",count
        print "Record duplicates: ", dup_count
        print "Written: ",(count-dup_count)

        print "RGB image:"
        print "\tMean img: ",(mean_img/in_idx)

        print "Flow:"
        print "\tMax/min flow u/v: ",max_flow, ' ', min_flow
        print "\tMax flow mag: ",max_mag

        n = (mean_flow/in_idx)
        print "\tFlow mean: ",n

        pixs = scale[0]*scale[1]
        stdv = np.sqrt(s2/(count * pixs) - (n*n))
        print "\tFlow stdv: ",stdv

        return


def getLabelSuffix(label_type):

    if label_type == XY_LABEL:
        return "xy"
    if label_type == XYM_LABEL:
        return "xym"
    if label_type == RGB_LABEL:
        return "rgb"
    if label_type == XXYY_LABEL:
        return "xxyy"


def run(prefix, suffix, label_type):

    file_list = prefix+'_of_src'+suffix+'.txt'

    print "Loading ",file_list
    img_list = utils.loadDataList(file_list)

    reset()
    db_name = prefix+'-of-labels-m-'+getLabelSuffix(label_type)+'-mpi'+suffix
    #write_db(db_name, img_list, crop, scale, label_type)

    reset()
    # For grayscale input images
    # write_db(prefix+'-of-imgs-gray-mpi'+suffix, img_list, crop, scale, False, False)

    # RGB input images
    bw_images = True
    if bw_images:
        db_name = prefix+'-of-imgs-m-bw-mpi'+suffix
    else:
        db_name = prefix+'-of-imgs-m-rgb-mpi'+suffix
    write_db(db_name, img_list, crop, scale, NO_LABEL, rgb_imgs=not bw_images)


    return


path = caffe_root+'/share/datasets/MPI_Sintel/MPI_Sintel-flow'

size_src = [1024, 436]
crop = [size_src[1], size_src[1]]
#scale = [227, 227]
scale = [224, 224]

crop_offset = utils.get_crop_offsets(size_src, crop)
crop_offset = crop_offset[:3]




if __name__ == '__main__':

    # run('train', '')
    # run('val', '')
    label_type = XYM_LABEL
    label_type = XXYY_LABEL
    label_type = XY_LABEL

    run('val', '_clean_final_90_10', label_type)
    run('train', '_clean_final_90_10', label_type)
