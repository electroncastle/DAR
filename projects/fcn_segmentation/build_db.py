import caffe
import lmdb
from PIL import Image
import re, fileinput, math
import os
import numpy as np


root = '/home/jiri/Lake/DAR/share/datasets/VOC2010/VOCdevkit/VOC2010/JPEGImages/'

def read_img_list(path, file):

    list = []
    id = 0
    for line in fileinput.input(file):
        list.append(os.path.join(path, line.strip())+'.jpg')
        id+=1

    return list

def build_db(inputs):
    in_db = lmdb.open('image-lmdb', map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, in_ in enumerate(inputs):
            # load image:
            # - as np.uint8 {0, ..., 255}
            # - in BGR (switch from RGB)
            # - in Channel x Height x Width order (switch from H x W x C)
            img = Image.open(in_)
            im = np.array(img) # or load whatever ndarray you need
            im = im[:,:,::-1]
            im = im.transpose((2,0,1))
            im_dat = caffe.io.array_to_datum(im)
            key = '{:0>10d}'.format(in_idx)
            print '['+str(in_idx)+' ]'+key,
            print '\r',
            in_txn.put(key, im_dat.SerializeToString())
    in_db.close()
    print


print "Starting..."
list = read_img_list(root, '/home/jiri/Lake/DAR/share/datasets/VOC2010/VOCdevkit/VOC2010/ImageSets/Segmentation/trainval.txt')
build_db(list)


