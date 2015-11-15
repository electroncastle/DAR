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
from scipy.special.orthogonal import he_roots

if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set'
    sys.exit()

caffe_root = os.environ['DAR_ROOT']
sys.path.insert(0, caffe_root + '/src/caffe/python')
import caffe
import caffe.io
from caffe.proto import caffe_pb2


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_flo(filename):
    file = open(filename, 'rb')
    signature = struct.unpack('f', file.read(4))
    width = struct.unpack('i', file.read(4))[0]
    height = struct.unpack('i', file.read(4))[0]

    x_flow = np.array([[0]*width]*height, dtype=float)
    y_flow = np.array([[0]*width]*height, dtype=float)

    for r in range(0,height):
        for c in range(0,width):
            x_flow[r,c] = struct.unpack('f', file.read(4))[0]
            y_flow[r,c] = struct.unpack('f', file.read(4))[0]

    file.close()

#    sys.exit(0)

    return [width, height, x_flow, y_flow]


def crop_scale(img, crop_offsets, crop_size, scale_size, scale_magnitude=0):

    width = img.shape[1]
    height = img.shape[0]

    if scale_magnitude > 0:
        x_flow_scale = float(scale_size[0]) / width
        y_flow_scale = float(scale_size[1]) / height

    imgs = []
    for crop in crop_offsets:

        of_y = crop[0]
        of_x = crop[1]
        img_dst = img[of_y:of_y + crop_size[1], of_x:of_x+crop_size[0]]
        img_dst = cv2.resize(img_dst, (scale_size[0], scale_size[1]))

        if scale_magnitude == 1:
            img_dst *= x_flow_scale
        if scale_magnitude == 2:
            img_dst *= y_flow_scale

        imgs.append(img_dst)

    return np.asarray(imgs)


def crop_scale_flow(xflow, yflow, crop_offsets, crop_size, scale_size):

    width = xflow.shape[1]
    height = xflow.shape[0]

    x_flow_scale = float(scale_size[0]) / crop_size[0]
    y_flow_scale = float(scale_size[1]) / crop_size[1]

    flows = []

    crop_id=0
    for crop in crop_offsets:

        of_y = crop[0]
        of_x = crop[1]
        x_flow_c = xflow[of_y:of_y + crop_size[1], of_x:of_x+crop_size[0]]
        y_flow_c = yflow[of_y:of_y + crop_size[1], of_x:of_x+crop_size[0]]


        x_flow_small = cv2.resize(x_flow_c, (scale_size[0], scale_size[1])) * x_flow_scale
        y_flow_small = cv2.resize(y_flow_c, (scale_size[0], scale_size[1])) * y_flow_scale

        flows.append([x_flow_small, y_flow_small])

    return np.asarray(flows)


def flo2jpg(filename, x_flow_filename, y_flow_filename, crop_offsets, crop_width, crop_height, of_width, of_height):

    [width, height, xflow, yflow]= read_flo(filename)

    crop_id=0
    for crop in crop_offsets:

        of_y = crop[0]
        of_x = crop[1]
        x_flow_c = xflow[of_y:of_y + crop_height, of_x:of_x+crop_width]
        y_flow_c = yflow[of_y:of_y + crop_height, of_x:of_x+crop_width]

        x_flow_small = cv2.resize(x_flow_c, (of_width, of_height))
        y_flow_small = cv2.resize(y_flow_c, (of_width, of_height))


        # Images have mean=0
        np.save(x_flow_filename+'{:0>2d}'.format(crop_id), x_flow_small)
        np.save(y_flow_filename+'{:0>2d}'.format(crop_id), y_flow_small)

        if crop_height != of_height:
            pil = PIL.Image.fromarray(np.uint8(x_flow_c+127.0))
            pil.save(x_flow_filename+'{:0>2d}'.format(crop_id)+'.jpg', format='JPEG', quality=90)

            pil = PIL.Image.fromarray(np.uint8(y_flow_c+127.0))
            pil.save(y_flow_filename+'{:0>2d}'.format(crop_id)+'.jpg', format='JPEG', quality=90)

        crop_id+=1

    return

# position
#  0  2  4   11
#  5  6  7   12
#  8  9  10  13

# 12 * 2(mirros) * 2(two image sets) = 48
def crop_img(src_img_filename, dst_img_filename, crop_width, crop_height, crop_offset):

    if os.path.isfile(dst_img_filename):
        return

    if not os.path.isfile(src_img_filename):
        return

    crop_id = 0
    for crop in crop_offset:

        # copy and resize the image frame
        im = PIL.Image.open(src_img_filename)
        image = np.array(im)

        t = crop[0]
        l = crop[1]
        image = image[t:t+crop_height, l:l+crop_width]

        out_file = dst_img_filename+'{:0>2d}.jpg'.format(crop_id)
        pil = PIL.Image.fromarray(np.uint8(image))
        pil.save(out_file, format='JPEG', quality=90)
        crop_id+=1




def get_crop_offsets(size, crop):

    datum_width = size[0]
    datum_height = size[1]
    crop_width = crop[0]
    crop_height = crop[1]

    height_off = (datum_height - crop_height)/2.0
    width_off = (datum_width - crop_width)/2.0

    height_off2 = (datum_height - crop_height)/4.0
    width_off2 = (datum_width - crop_width)/4.0

    offsets=[]
    offsets.append([0, 0])   #upper left left
    offsets.append([0, 2 * width_off])   #upper right right
    offsets.append([height_off, width_off])   #center

    offsets.append([2 * height_off, 0])   #lower left
    offsets.append([2 * height_off, 2 *width_off])   #lower right

    #-------
    offsets.append([0, width_off])   #upper center
    offsets.append([2 * height_off, width_off])   #lower center
    offsets.append([height_off, 0])   # left center
    offsets.append([height_off, 2*width_off])   #right center

    #-------
    offsets.append([0, width_off2])   #upper left
    offsets.append([0, 3*width_off2])   #upper right

    offsets.append([2*height_off, width_off2])   # left left
    offsets.append([2*height_off, 3*width_off2])   #lower right

    return offsets


def layer2polar(layers, mag):
    h, w = mag.shape
    bins = layers.shape[0]
    bin_width = 2.0*np.pi/bins
    angle = np.zeros((h, w))

    d = 0.0
    for i in range(0, bins):
        d += bin_width
#         print layers[i]
        layer_dir = layers[i]*d
        angle += layer_dir
#         print angle


    return angle, mag


def polar2of(angle, mag):

    ofx = np.multiply(mag, np.cos(angle))
    ofy = np.multiply(mag, np.sin(angle))

    return ofx, ofy


def of2img(ang, mag):
    hsv = np.zeros([mag.shape[0],mag.shape[1],3], dtype=np.uint8)
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb



def of2polar(x,y):
    cpx = x+(y*(1j))
    mag = np.abs(cpx)
    angle_src = np.angle(cpx) # +/- PI

    # Convert to range 0..2pi
    os = angle_src < 0.0
    angle_src[os]=2.0*np.pi+angle_src[os]

    return angle_src, mag



def of2jpg(filename, ang, mag):
    img  = of2img(ang, mag)
    cv2.imwrite(filename, img)


def get_data_lmdb(lmdb_name , key):
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

    if datum.data == '':
        data = np.array(datum.float_data).astype(float).reshape(datum.channels, datum.height, datum.width)
    else:
        data = np.fromstring(datum.data, dtype=np.uint8).reshape(datum.channels, datum.height, datum.width)

    return data


def polar2Layers(angle_src, mag, bins, min_mag):

    h, w = mag.shape
    bin_width = 2.0*np.pi/bins
    layers = np.zeros((bins, h, w))

    noise_of = mag < min_mag # ??

    angle = angle_src.copy()
    e=0.0
    for i in range(0, bins):

        e += bin_width
        # print e
        dir_p = angle < e
        # print dir_p
        layers[i][dir_p] = 1
        layers[i][noise_of] = 0

        # flag the already recorded angles
        angle[dir_p] = 99


    return layers, mag


def showPolar(angle_src, mag):
    plt.subplot(121)
    # plt.imshow(mag, extent=[-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi])
    plt.imshow(mag,  extent=[0, 224, 224, 0])
    plt.title('Magnitude of exp(x)')

    plt.subplot(122)
    plt.imshow(angle_src, extent=[-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi])
    #plt.imshow(angle_src)
    plt.title('Phase (angle) of exp(x)')
    # plt.show()


def loadDataList(filename):
    data = []
    for line in fileinput.input(filename):
        data.append(line.strip().split(' '))

    return data





if __name__ == "__main__":
	print "Main"


