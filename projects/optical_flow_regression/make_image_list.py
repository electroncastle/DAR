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

import numpy as np
import PIL.Image
from cStringIO import StringIO

import lmdb

import re, fileinput, math

if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set'
    sys.exit()

caffe_root = os.environ['DAR_ROOT']
sys.path.insert(0, caffe_root + '/python')
import caffe
import caffe.io
from caffe.proto import caffe_pb2



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

        print "[",count+1,"]  Processing: ",subdir
        count += 1

        gc.disable()
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

        gc.enable()


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

    decode = False
    # Size of buffer: 1000 elements to reduce memory consumption
    block_elements = 1000.0
    blocks = int(math.ceil(len(Inputs)/block_elements))
    for idx in range(blocks):
        in_db_label = lmdb.open(lmdb_label_name, map_size=int(1e12))

        block_first = int(block_elements)*idx
        block_last =  int(block_elements)*(idx+1)

        with in_db_label.begin(write=True) as in_txn:
            for in_idx, in_ in enumerate(Inputs[block_first:block_last]):
                # im = load_images(root, in_, decode)
                # # im_dat = caffe.io.array_to_datum(im.astype(float).transpose((2, 0, 1)))
                # # im_dat = caffe.io.array_to_datum(im.astype(float))
                # imf = im.astype(float)
                # im_dat = caffe.io.array_to_datum(imf)

                im_dat = load_to_datum(root, in_, decode)

                im_dat_str = im_dat.SerializeToString()
                in_txn.put('{:0>10d}'.format(block_first + in_idx), im_dat_str)

                string_ = str(block_first + in_idx + 1) + ' / ' + str(len(Inputs))
                sys.stdout.write("\r%s" % string_)
                sys.stdout.flush()
        in_db_label.close()


def path_to_datum(path, label, image_sum, compute_mean, encode):
    """
    Creates a Datum from a path and a label
    May also update image_sum, if computing mean
    Arguments:
    path -- path to the image (filesystem path or URL)
    label -- numeric label for this image's category
    Keyword arguments:
    image_sum -- numpy array that stores a running sum of added images
    """
    # prepend path with image_folder, if appropriate
    # path = os.path.join(image_folder, path)



    image = caffe.io.load_image(path, False)
    # image = utils.image.load_image(path)
    if image is None:
        return None

    # Resize
    # image = utils.image.resize_image(image,
    #         self.height, self.width,
    #         channels    = self.channels,
    #         resize_mode = self.resize_mode,
    #         )

    image = cv2.resize(image, (32, 32))


    if compute_mean and image_sum is not None:
        image_sum += image

    if encode:
        datum = caffe_pb2.Datum()
        if image.ndim == 3:
            datum.channels = image.shape[2]
        else:
            datum.channels = 1
        datum.height = image.shape[0]
        datum.width = image.shape[1]
        datum.label = label
        datum.encoded = True

        s = StringIO()
        b = np.uint8(image)
        pil = PIL.Image.fromarray(b)
        pil.save(s, format='JPEG', quality=90)
        datum.data = s.getvalue()
    else:
        # Transform to caffe's format requirements
        if image.ndim == 3:
            # Transpose to (channels, height, width)
            image = image.transpose((2,0,1))
        elif image.ndim == 2:
            # Add a channels axis
            image = image[np.newaxis,:,:]
        else:
            raise Exception('Image has unrecognized shape: "%s"' % image.shape)
        datum = caffe.io.array_to_datum(image, label)

    return datum


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
def load_images(root, line, decode=True):

    # print line,
    img1_file, img2_file, ofx_file, ofy_file = line.split(' ')

    img1_path = os.path.join(root, img1_file)
    img2_path = os.path.join(root, img2_file)

    batch = []
    if decode:
        img1 = caffe.io.load_image(img1_path, True)
        img2 = caffe.io.load_image(img2_path, True)

        channelsEx = img1.shape[2]
        for c in range(0, channelsEx):
            batch.append(img1[:,:,c])

        channelsEx = img2.shape[2]
        for c in range(0, channelsEx):
            batch.append(img2[:,:,c])
    else:
        img1 = open(img1_path, 'rb')
        buffer = img1.readall()
        batch.append(buffer)
        img1.close()

        img2 = open(img2_path, 'rb')
        buffer = img2.readall()
        batch.append(buffer)
        img2.close()

    return np.asarray(batch)


class DBuilder:

    def __init__(self):
        self.image_sum = None
        self.compute_mean = True
        self.encode = True
        self.new_height = 0
        self.new_width = 0
        self.recompress = False
        self.img_count = 0
        pass


    def of2datum(self, ofx_filename, ofy_filename, crop_height, crop_width):

        image1 = caffe.io.load_image(ofx_filename, False)
        image1 *= 255
        if len(image1.shape) > 2 and  image1.shape[2] == 1:
            image1 = image1[:,:,0]

#        image1 = cv2.resize(image1, (self.new_width, self.new_height))

        t = (image1.shape[0]-crop_height)/2
        l = (image1.shape[1]-crop_width)/2
        image1 = image1[t:t+crop_height, l:l+crop_width]


        if self.compute_mean:
            if self.image_sum is None:
                self.image_sum = image1
            else:
                self.image_sum += image1
            self.img_count += 1


        # Transform to caffe's format requirements
        if image1.ndim == 3:
            # Transpose to (channels, height, width)
            image = image1.transpose((2,0,1))
        elif image1.ndim == 2:
            # Add a channels axis
            image = image1[np.newaxis,:,:]
        else:
            raise Exception('Image has unrecognized shape: "%s"' % image1.shape)
        datum = caffe.io.array_to_datum(image, 0)

        return datum

    def img2datum(self, filename):

        label = 0
        recompress = self.recompress
        image1 = caffe.io.load_image(filename, False)
        image1 *= 255
        if len(image1.shape) > 2 and  image1.shape[2] == 1:
            image1 = image1[:,:,0]


        if (self.new_height!=0 and self.new_height!=0) and \
                (self.new_height != image1.shape[1] or self.new_height != image1.shape[0]):

            image1 = cv2.resize(image1, (self.new_width, self.new_height))
            recompress = True

        if self.compute_mean:
            if self.image_sum is None:
                self.image_sum = image1
            else:
                self.image_sum += image1
            self.img_count += 1

        if self.encode:
            datum = caffe_pb2.Datum()
            if image1.ndim == 3:
                datum.channels = image1.shape[2]
            else:
                datum.channels = 1

            datum.height = image1.shape[0]
            datum.width = image1.shape[1]
            datum.label = label
            datum.encoded = True

            s1 = StringIO()
            if recompress:
                pil = PIL.Image.fromarray(np.uint8(image1))
                pil.save(s1, format='JPEG', quality=90)
            else:
                image1 = open(filename, 'rb')
                buf = bytearray(os.path.getsize(filename))
                image1.readinto(buf)
                s1.write(buf)

            datum.data = s1.getvalue()
        else:
            # Transform to caffe's format requirements
            if image1.ndim == 3:
                # Transpose to (channels, height, width)
                image = image1.transpose((2,0,1))
            elif image1.ndim == 2:
                # Add a channels axis
                image = image1[np.newaxis,:,:]
            else:
                raise Exception('Image has unrecognized shape: "%s"' % image1.shape)
            datum = caffe.io.array_to_datum(image, label)

        return datum


    def write_images(self, lmdb_label_name, root, img_list, key_list, labels):

        # Size of buffer: 1000 elements to reduce memory consumption
        block_elements = 1000.0
        blocks = int(math.ceil(len(img_list)/block_elements))
        for idx in range(blocks):
            in_db_label = lmdb.open(lmdb_label_name, map_size=int(1e12))

            block_first = int(block_elements)*idx
            block_last =  int(block_elements)*(idx+1)

            with in_db_label.begin(write=True) as in_txn:
                for in_idx, img_file in enumerate(img_list[block_first:block_last]):
                    # im = load_images(root, in_, decode)
                    # # im_dat = caffe.io.array_to_datum(im.astype(float).transpose((2, 0, 1)))
                    # # im_dat = caffe.io.array_to_datum(im.astype(float))
                    # imf = im.astype(float)
                    # im_dat = caffe.io.array_to_datum(imf)

                    im_dat = self.img2datum(os.path.join(root, img_file))

                    im_dat_str = im_dat.SerializeToString()

                    #key = '{:0>10d}'.format(block_first + in_idx)
                    key = key_list[block_first+in_idx]
                    in_txn.put(key, im_dat_str)

                    string_ = str(block_first + in_idx + 1) + ' / ' + str(len(img_list))
                    sys.stdout.write("\r%s" % string_)
                    sys.stdout.flush()
            in_db_label.close()


    def get_img_key(self, filename, flow):

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
            key = '{:0>5d}{:0>5d}{:d}'.format(video_id, frame_id, fd)
        else:
            # Max 99 999 videos
            # Max 99 999 images
            key = '{:0>5d}{:0>5d}'.format(video_id, frame_id)

        return frame_id, key


    def build_db(self, name, img_list_file, labels = False):

        print 'Building ',name
        print 'Building image list...'

        imgs_exist = [0]*10000000
        imgs = []
        keys = []

        dc = {}

        gc.disable()
        count = 0
        for line in fileinput.input(img_list_file):

            img1_file, img2_file, ofx_file, ofy_file = line.strip().split(' ')
            #print "[",count,"] ", img1_file, img2_file, ofx_file, ofy_file

            if labels:

                imgs.append(ofx_file)
                id, key = self.get_img_key(ofx_file, labels)
                keys.append(key)

                imgs.append(ofy_file)
                id, key = self.get_img_key(ofy_file, labels)
                keys.append(key)
            else:
                if img1_file not in dc:
                    id, key = self.get_img_key(img1_file, labels)
                    imgs.append(img1_file)
                    keys.append(key)
                    dc[img1_file] = 1

                if img2_file not in dc:
                    id, key = self.get_img_key(img2_file, labels)
                    imgs.append(img2_file)
                    keys.append(key)
                    dc[img2_file] = 1

            count += 1
            # if count > 10000:
            #     break

            #string_ = str(block_first + in_idx + 1) + ' / ' + str(len(img_list))
            if (count % 1000) == 0:
                sys.stdout.write("\r%d" % count)
                sys.stdout.flush()

        gc.enable()
        fileinput.close()
        print

        print len(dc.keys()),'  ',len(imgs)

        print 'Building LMDB...'
        dbb.write_images(name, path, imgs, keys, labels)
        print '\nDone'

        if self.compute_mean:
            mean_img = self.image_sum / float(self.img_count)
            pil = PIL.Image.fromarray(np.uint8(mean_img))
            pil.save(name+'/mean.jpg', format='JPEG', quality=90)


    def get_image(self, lmdb_label_name, id):
        lmdb_env = lmdb.open(lmdb_label_name, map_size=int(1e12))
        lmdb_txn = lmdb_env.begin()  # equivalent to mdb_txn_begin()
        lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()
        # lmdb_cursor.get('{:0>10s}'.format('_6')) #  get the data associated with the 'key' 1, change the value to get other images

        count = 0
        for key, value in lmdb_cursor:
            count += 1
            print count,'  ',(key)


        #lmdb_cursor.first()
        lmdb_cursor.get(id) # '{:0>10s}'.format('_6')

        value = lmdb_cursor.value()
        key = lmdb_cursor.key()

        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        image = np.zeros((datum.channels, datum.height, datum.width))

        jpeg=True
        if jpeg:
            # for jpeg payload
            image = np.fromstring(datum.data, dtype=np.uint8)
            image = image.astype(np.uint8)
            image.tofile('img.jpg')

        lmdb_cursor.first()
        lmdb_env.close()


if __name__ == "__main__":

    path = caffe_root+'/share/datasets/THUMOS2015/thumos15_validation-rgbflow/'
    train_filename = 'train-of.txt'
    val_filename = 'val-of.txt'

    build_img_list = False
    build_img_lmdb = True

    if build_img_list:
        imgList = build_list(path, train_filename, val_filename)


    dbb = DBuilder()

    if len(sys.argv) > 2:
        dbb.get_image(sys.argv[1], sys.argv[2])
        sys.exit(0)

    dbb.new_height = 0
    dbb.new_width = 0
    dbb.image_sum = 0
    dbb.img_count = 0
    dbb.build_db('train-of-images-lmdb', train_filename, False)


    dbb.new_height = 32
    dbb.new_width = 32
    dbb.image_sum = 0
    dbb.img_count = 0
    dbb.build_db('train-of-labels-lmdb', train_filename, True)


