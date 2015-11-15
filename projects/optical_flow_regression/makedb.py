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

max_flow = -9999999999
min_flow = 9999999999
mean_flow = 0.0
mean_img = 0
count_flow = 0

def reset():
    global max_flow
    global min_flow
    global mean_flow
    global mean_img
    global count_flow

    max_flow = -9999999999
    min_flow = 9999999999
    mean_flow = 0.0
    mean_img = 0
    count_flow = 0


def img_mean(img):
    return float(np.sum(img)) / (len(img.flatten()))


def get_img(img_file, crop, scale, color = False):
    global max_flow
    global min_flow
    global mean_img

    img1 =  cv2.imread(img_file[0], color)
#    img1 = img1.transpose((2, 0, 1))

    img2 =  cv2.imread(img_file[1], color)
#    img2 = img1.transpose((2, 0, 1))

    imgs1 = utils.crop_scale(img1, crop_offset, crop, scale)
    imgs2 = utils.crop_scale(img2, crop_offset, crop, scale)

    if color:
        imgs1 = imgs1.transpose(0, 3, 1, 2)
        imgs2 = imgs2.transpose(0, 3, 1, 2)
    else:
        imgs1=imgs1[:, np.newaxis, :, :]
        imgs2=imgs2[:, np.newaxis, :, :]

    datums = []
    for i in range(0, len(imgs1)):

        mean_img = mean_img + (img_mean(imgs1[i]) + img_mean(imgs2[i])) / 2.0
        of_vec = np.append(imgs1[i], imgs2[i], axis=0)

        #datums.append(caffe.io.array_to_datum(of_vec.astype(float)))
        datums.append(caffe.io.array_to_datum(of_vec.astype(np.uint8)))

    return datums


def get_label(img_file, crop, scale, alpha, beta):
    global max_flow
    global min_flow
    global mean_flow

    use_flo = False


    if use_flo:
        width, height, x_flow, y_flow = utils.read_flo(img_file[2])

        # x_flow = x_flow*alpha+beta
        # y_flow = y_flow*alpha+beta

         # set invalid flow pixels to zero
        invalid = x_flow > 1e9
        x_flow[invalid] = 0.0
        invalid = y_flow > 1e9
        y_flow[invalid] = 0.0


        flows = utils.crop_scale_flow(x_flow, y_flow, crop_offset, crop, scale)


        max_flow = max(max_flow, flows.max())
        max_flow = max(max_flow, flows.max())

        min_flow = min(min_flow, flows.min())
        min_flow = min(min_flow, flows.min())

        scale_range = 127.0
        min_id = flows < -scale_range
        max_id = flows > scale_range

        flows[min_id] = -127.0
        flows[max_id] = 127.0

        #flows = flows * (1.0/scale_range)
    else:

        dirs = img_file[2].split(os.sep)
        dirs[-3] = 'flow_viz'
        filename, ext = os.path.splitext(dirs[-1])
        dirs[-1] = filename+'.png'
        path = '/'.join(dirs)

        img1 =  cv2.imread(path, True)
        flows = utils.crop_scale(img1, crop_offset, crop, scale)
        flows = flows.transpose(0, 3, 1, 2)


    datums = []
    for flow in flows:
        mean_flow += img_mean(flow)
        datums.append(caffe.io.array_to_datum(flow.astype(np.uint8)))

    return datums


def write_db(lmdb_name, img_list, crop, scale, labels, rgb_imgs = False):

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
        with in_db_label.begin(write=True) as in_txn:
            for img_file in img_list:

                if labels:
                    datums = get_label(img_file, crop, scale,  127.0/427.0,  0)
                else:
                    datums = get_img(img_file, crop, scale, rgb_imgs)


                for datum in datums:
                    key = '{:0>10d}'.format(in_idx)
                    dat_str = datum.SerializeToString()

                    #print key
                    if not in_txn.put(key, dat_str, dupdata=False, overwrite=False):
                        dup_count+=1
                        print
                        print dup_count," Already exist:  ",key

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
        print "Max/min flow: ",max_flow, ' ', min_flow

        print "Mean img: ",(mean_img/in_idx)
        print "Mean flow: ",(mean_flow/in_idx)

        return


def run(prefix, suffix):

    file_list = prefix+'_of_src'+suffix+'.txt'

    print "Loading ",file_list
    img_list = utils.loadDataList(file_list)

    reset()
    # write_db(prefix+'-of-imgs-rgb-mpi'+suffix, img_list, crop, scale, False)
    write_db(prefix+'-of-imgs-mpi'+suffix, img_list, crop, scale, False, False)

    reset()
    write_db(prefix+'-of-labels-rgb-mpi'+suffix, img_list, crop, scale, True)

    return


path = caffe_root+'/share/datasets/MPI_Sintel/MPI_Sintel-flow'

size_src = [1024, 436]
crop = [size_src[1], size_src[1]]
scale = [224, 224]

crop_offset = utils.get_crop_offsets(size_src, crop)
crop_offset = crop_offset[:3]



if __name__ == '__main__':

    # run('train', '')
    # run('val', '')

    run('train', '_clean_final')
    run('val', '_clean_final')
