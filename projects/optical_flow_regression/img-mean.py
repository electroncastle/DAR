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

# caffe_root = os.environ['DAR_ROOT']
# sys.path.insert(0, caffe_root + '/src/caffe/python')
# import caffe
# import caffe.io
# from caffe.proto import caffe_pb2


imgs = []
keys = []
dc = {}

max_images = 0
image_sum = None
img_count = 0

def get_img_key(filename, flow):

    path_toks = os.path.split(filename)
    video_name = path_toks[0]
    video_name_toks = video_name.split('_')
    video_id_str = video_name_toks[-1]
    video_id = int(video_id_str)

    base_filename = os.path.basename(filename)
    file, ex = os.path.splitext(base_filename)
    toks = file.split('_')
    frame_id = int(toks[-1])

    if frame_id > 99999:
        print "Number of images too high. max=99999 "
        sys.exit(0)

    if video_id > 99999:
        print "Number of videos too high. max=99999 "
        sys.exit(0)

    if (flow):
        fd = 0 # X flow
        if toks[1] == 'y':
            fd = 1
        # Max 99 999 videos
        # Max 99 999 images
        # Img type: 1=x flow    0=y flow
        key = '{:0>5d}{:0>5d}'.format(video_id, frame_id)
    else:
        # Max 99 999 videos
        # Max 99 999 images
        key = '{:0>5d}{:0>5d}'.format(video_id, frame_id)

    return frame_id, key



def load_img_list(img_list_file):
    global dc

    gc.disable()
    count = 0
    for line in fileinput.input(img_list_file):
        img1_file, img2_file, ofx_file, ofy_file = line.strip().split(' ')
        #print "[",count,"] ", img1_file, img2_file, ofx_file, ofy_file

        if img1_file not in dc:
            #id, key = get_img_key(img1_file, False)
            imgs.append(img1_file)
            #keys.append(key)
            dc[img1_file] = 1

        if img2_file not in dc:
            #id, key = get_img_key(img2_file, False)
            imgs.append(img2_file)
            #keys.append(key)
            dc[img2_file] = 1

        count += 1
        if max_images > 0 and  count > max_images:
            break

        #string_ = str(block_first + in_idx + 1) + ' / ' + str(len(img_list))
        if (count % 1000) == 0:
            sys.stdout.write("\r%d" % count)
            sys.stdout.flush()

    gc.enable()
    fileinput.close()
    print

    return


def get_img_mean(path, imglists):


    for imlist in imglists:
        print "Loading ", imlist
        load_img_list(imlist)

    calc_mean(path)


def calc_mean(path):

    global image_sum
    global img_count

    # self.s1 = 0.0
    # self.s2 = 0.0
    # self.ofmin = 99999999
    # self.ofmax = -9999999999
    #
    # self.image_sum = None
    # self.img_count = 0

    count = 0
    for img in imgs:

        im = PIL.Image.open(os.path.join(path, img)).convert('RGB')
        image = np.array(im, dtype=np.float64)

        if image_sum is None:
            image_sum = image
        else:
            image_sum += image

        img_count += 1

        count += 1
        if (count % 1000) == 0:
            string_ = '['+str(count)+']  '+img
            sys.stdout.write("\r%s" % string_)
            sys.stdout.flush()


    mean_rgb_planes = image_sum / float(img_count)

    mean_rgb = [np.mean(mean_rgb_planes[:,:,0]), np.mean(mean_rgb_planes[:,:,1]),
        np.mean(mean_rgb_planes[:,:,2])]

    print
    print mean_rgb

    # self.mean = mean_vec
    #
    # # self.stdv = np.sqrt(self.img_count*self.s2 - self.s1*self.s1)/self.img_count
    #
    # n = np.mean(self.mean)
    # self.stdv = np.sqrt(self.s2/(self.img_count*datum.width) - (n*n))
    #
    # print "mean=", n
    # print "std=",self.stdv
    # print "min/max = ", self.ofmin,' / ', self.ofmax

    return


if __name__ == "__main__":

    max_images = 1100

    imglists = sys.argv[2:]
    get_img_mean(sys.argv[1], imglists)



