import io
import os
import sys
import random
import cv2

import numpy as np
import lmdb

import re, fileinput, math
import numpy as np

if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set'
    sys.exit()

caffe_root = os.environ['DAR_ROOT']
sys.path.insert(0, caffe_root + '/python')
import caffe


def build_lmdb(root, filename, name):

    fin = open(filename, 'rt')

    map_size = 10000000000

    env = lmdb.open(name, map_size=map_size)

    with env.begin(write=True) as txn:

        # txn is a Transaction object
        i = 0
        for line in fin:
            print line,
            # Load Images
            img1_file, img2_file, ofx_file, ofy_file = line.split(' ')

            img1 = caffe.io.load_image(os.path.join(root, img1_file), True)
            img2 = caffe.io.load_image(os.path.join(root, img2_file), True)

            batch = []
            channelsEx = img1.shape[2]
            for c in range(0, channelsEx):
                batch.append(img1[:,:,c])

            channelsEx = img2.shape[2]
            for c in range(0, channelsEx):
                batch.append(img2[:,:,c])

            datum = caffe.proto.caffe_pb2.Datum()

            datum.channels = 6
            datum.height = 224
            datum.width = 224

            # data
            npbatch = np.asarray(batch)
            datum.data = npbatch.tostring() #.toString()  #tobytes()  # or .tostring() if numpy < 1.9

            # label
            ox_img = caffe.io.load_image(os.path.join(root, ofx_file.strip()), False)
            oy_img = caffe.io.load_image(os.path.join(root, ofy_file.strip()), False)

            batch = []
            channelsEx = ox_img.shape[2]
            for c in range(0, channelsEx):
                batch.append(ox_img[:,:,c])

            channelsEx = oy_img.shape[2]
            for c in range(0, channelsEx):
                batch.append(oy_img[:,:,c])


            datum.channels = 2
            datum.height = 32
            datum.width = 32
            npbatch = np.asarray(batch)
            datum.label = npbatch.tostring()
            str_id = '{:08}'.format(i)

            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
            i += 1

    return


def build_list(path, train_filename, val_filename):

    maxLen = None
    imgList = []
    count = 0
    for subdir, dirs, files in os.walk(path):
        dirs = subdir.strip().split(os.sep)
        last_subdir = dirs[len(dirs)-1]

        print "Processing: ",subdir
        for file in files:
            filename, ext = os.path.splitext(file)
            toks = filename.split('_')
            if len(toks) < 2 or toks[0] != 'image':
                continue

            img_id = int(toks[1])
            img_id1_str = '{:0>4d}'.format(img_id)
            img_id2_str = '{:0>4d}'.format(img_id+1)

            img1 = file
            img1_path = os.path.join(subdir, img1)
            img2= 'image_'+img_id2_str+ext
            img2_path = os.path.join(subdir, img2)
            ofx = 'flow_x_'+img_id1_str+ext
            ofx_path = os.path.join(subdir, ofx)
            ofy = 'flow_y_'+img_id1_str+ext
            ofy_path = os.path.join(subdir, ofy)

            if os.path.isfile(img2_path) and os.path.isfile(ofx_path) and os.path.isfile(ofy_path):
                record = [os.path.join(last_subdir, img1), os.path.join(last_subdir, img2), os.path.join(last_subdir, ofx), os.path.join(last_subdir, ofy)]
                recordStr = ' '.join(record)
                imgList.append(recordStr)
                # print recordStr

            if maxLen <> None and len(imgList) > maxLen:
                break


    # Split into train and val datasets

    random.shuffle(imgList)
    listLen = len(imgList)
    trainSize = int(listLen*0.5)

    trainFile = open(train_filename, 'wt')
    for i in range(0, trainSize):
        recordStr = imgList[i]
        trainFile.write(recordStr+'\n')
    trainFile.close()

    valFile = open(val_filename, 'wt')
    for i in range(trainSize, listLen):
        recordStr = imgList[i]
        valFile.write(recordStr+'\n')
    valFile.close()

    # for im_f in glob.glob(path)
    #     inputs =[im_f for im_f in glob.glob(args.input_file + '/*.' + args.ext)]

    return imgList


def write_labels(lmdb_label_name, root, Inputs):

    # Size of buffer: 1000 elements to reduce memory consumption
    for idx in range(int(math.ceil(len(Inputs)/1000.0))):
        in_db_label = lmdb.open(lmdb_label_name, map_size=int(1e12))
        with in_db_label.begin(write=True) as in_txn:
            for in_idx, in_ in enumerate(Inputs[(1000*idx):(1000*(idx+1))]):
                im = load_labels(root, in_)
                # im_dat = caffe.io.array_to_datum(im.astype(float).transpose((2, 0, 1)))
                # im_dat = caffe.io.array_to_datum(im.astype(float))
                im_dat = caffe.io.array_to_datum(im.astype(float).reshape(1,1,np.asarray(im.shape).prod()))
                in_txn.put('{:0>10d}'.format(1000*idx + in_idx), im_dat.SerializeToString())

                string_ = str(1000*idx+in_idx+1) + ' / ' + str(len(Inputs))
                sys.stdout.write("\r%s" % string_)
                sys.stdout.flush()
        in_db_label.close()

# @profile
def write_images(lmdb_label_name, root, Inputs):

    # Size of buffer: 1000 elements to reduce memory consumption
    block_elements = 1000.0
    blocks = int(math.ceil(len(Inputs)/block_elements))
    for idx in range(blocks):
        in_db_label = lmdb.open(lmdb_label_name, map_size=int(1e12))

        block_first = int(block_elements)*idx
        block_last =  int(block_elements)*(idx+1)

        with in_db_label.begin(write=True) as in_txn:
            for in_idx, in_ in enumerate(Inputs[block_first:block_last]):
                im = load_images(root, in_)
                # im_dat = caffe.io.array_to_datum(im.astype(float).transpose((2, 0, 1)))
                # im_dat = caffe.io.array_to_datum(im.astype(float))
                imf = im.astype(float)
                im_dat = caffe.io.array_to_datum(imf)
                im_dat_str = im_dat.SerializeToString()
                in_txn.put('{:0>10d}'.format(block_first + in_idx), im_dat_str)

                string_ = str(block_first + in_idx + 1) + ' / ' + str(len(Inputs))
                sys.stdout.write("\r%s" % string_)
                sys.stdout.flush()
        in_db_label.close()



def load_labels(root, line):

    # print line,
    img1_file, img2_file, ofx_file, ofy_file = line.split(' ')

    ox_img = caffe.io.load_image(os.path.join(root, ofx_file.strip()), False)
    oy_img = caffe.io.load_image(os.path.join(root, ofy_file.strip()), False)

    ox_img = cv2.resize(ox_img, (32, 32))
    oy_img = cv2.resize(oy_img, (32, 32))

    batch = []
    batch.append(ox_img[:,:])
    batch.append(oy_img[:,:])

    return np.asarray(batch)

# @profile
def load_images(root, line):

    # print line,
    img1_file, img2_file, ofx_file, ofy_file = line.split(' ')

    img1 = caffe.io.load_image(os.path.join(root, img1_file), True)
    img2 = caffe.io.load_image(os.path.join(root, img2_file), True)

    batch = []
    channelsEx = img1.shape[2]
    for c in range(0, channelsEx):
        batch.append(img1[:,:,c])

    channelsEx = img2.shape[2]
    for c in range(0, channelsEx):
        batch.append(img2[:,:,c])

    return np.asarray(batch)

if __name__ == "__main__":


    path = caffe_root+'/share/datasets/THUMOS2015/thumos15_validation-rgbflow/'
    train_filename = 'train-of.txt'
    val_filename = 'val-of.txt'

    build_img_list = False
    build_img_lmdb = True

    if build_img_list:
        imgList = build_list(path, train_filename, val_filename)

    if build_img_lmdb:
        label_files = []
        for line in fileinput.input(train_filename):
            entries = re.split(' ', line.strip())
            # labels.append([entries[2], entries[3].strip()])
            label_files.append(line.strip())
            # print entries

        # labels = load_labels(path, label_files)

        # build_lmdb(path,  train_filename, 'train_lmdb')

        #write_labels('train-of-labels-lmdb', path, label_files)
        write_images('train-of-images-lmdb', path, label_files)

