#!/usr/bin/python

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

if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set'
    sys.exit()

caffe_root = os.environ['DAR_ROOT']
sys.path.insert(0, caffe_root + '/src/caffe/python')
import caffe
import caffe.io
from caffe.proto import caffe_pb2

import utils

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise



# def show_of(data):
#
#     if data == None:
#         return
#
#     if data.shape[0] == 3:
#         # RGB image
#         image = data.astype(np.uint8)
#         image = image.transpose(1,2,0)
#         filename = 'flow.jpg'
#         cv2.imwrite(filename, image)
#         cv2.imshow("Flow RGB", image)
#         cv2.moveWindow("Flow RGB", 10, 10)
#         return
#
# #    data = (data)+127
#
#     image = data.astype(np.uint8)
#
#     ang, mag = utils.of2polar(data[0], data[1])
#     img = utils.of2img(ang, mag)
#
#     filename = 'flow-x.jpg'
#     cv2.imwrite(filename, img)
#     cv2.imshow("OF X", img)
#     cv2.moveWindow("OF X", 10, 10)
#
#     filename = 'flow-y.jpg'
#     cv2.imwrite(filename, image[1])
#     cv2.imshow("OF Y", image[1])
#     cv2.moveWindow("OF Y", 300, 10)
#
#     # cv2.waitKey(0)

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


def show_img(data):
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

    filename = 'img-1.jpg'
    cv2.imwrite(filename, img1)
    cv2.imshow("IMG 1", img1)
    cv2.moveWindow("IMG 1", wnd_x, wnd_y)

    filename = 'img-2.jpg'
    cv2.imwrite(filename, img2)
    cv2.imshow("IMG 2", img2)
    wnd_x+=300
    cv2.moveWindow("IMG 2",  wnd_x, wnd_y)

    wnd_x=10
    wnd_y+=300


def get_data(value):

    if len(value) == 0:
        return None

    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
#        image = np.zeros((datum.channels, datum.height, datum.width))

    # flat_x = np.fromstring(datum.data, dtype=np.uint8)
    # x = flat_x.reshape(datum.channels, datum.height, datum.width)
    if (datum.data == ''):
        data = np.array(datum.float_data).astype(float).reshape(datum.channels, datum.height, datum.width)
    else:
        data = np.fromstring(datum.data, dtype=np.uint8).reshape(datum.channels, datum.height, datum.width)

    #print data.shape

    return data

def get_image(lmdb_label_name, id, show_stats):

    print "Rading lmdb: ",lmdb_label_name
    lmdb_env = lmdb.open(lmdb_label_name, map_size=int(1e12))
    lmdb_txn = lmdb_env.begin()  # equivalent to mdb_txn_begin()
    lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()
    # lmdb_cursor.get('{:0>10s}'.format('_6')) #  get the data associated with the 'key' 1, change the value to get other images

    print lmdb_env.info()

    if show_stats:
        mean_val = 0.0
        count = 0
        for key, value in lmdb_cursor:
            count += 1
            #print count,'  ',(key)
            data = get_data(value)

    print "------------"
    #lmdb_cursor.first()
    lmdb_cursor.get(id) # '{:0>10s}'.format('_6')

    value = lmdb_cursor.value()
#       key = lmdb_cursor.key()

    data = get_data(value)
    print "min/max  ",data.min(), '  ',data.max()

    lmdb_cursor.first()
    lmdb_env.close()

    return data


if __name__ == "__main__":

    labels_db = ''
    img_db = ''
    key = ''
    show_stats = False

    i=1
    while i<len(sys.argv):

        p = sys.argv[i]

        if p == '-o':
            i+=1
            if len(sys.argv) > i:
                labels_db = sys.argv[i]
        elif(p=='-s'):
            show_stats=True
        elif(img_db == ''):
            img_db = p
        else:
            key = p

        i+=1

    if key == '':
        print sys.argv[0],'  <lmdb_name> [<key>]'
        sys.exit(0)


    img_data = get_image(img_db, key, show_stats)
    of_data = get_image(labels_db, key, show_stats)

    show_img(img_data)
    show_of(of_data, 'GT')

    cv2.waitKey(0)

