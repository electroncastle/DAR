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




def get_crop_offsets(datum_width, datum_height, crop_width, crop_height):
    height_off = (datum_height - crop_height)/2.0
    width_off = (datum_width - crop_width)/2.0

    height_off2 = (datum_height - crop_height)/4.0
    width_off2 = (datum_width - crop_width)/4.0

    offsets=[]
    offsets.append([0, 0])   #upper left left
    offsets.append([0, 2 * width_off])   #upper right right
    offsets.append([2 * height_off, 0])   #lower left
    offsets.append([2 * height_off, 2 *width_off])   #lower right
    offsets.append([height_off, width_off])   #center

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


def make_rgbflow_from_dir(src_path, image_dirs, flow_dir, dst_path, dst_rgb):

    crop_size = 224
    of_scale = 32

    of_full_res = True

    if of_full_res:
        of_scale = 224

    offsets = get_crop_offsets(1024, 436, crop_size, crop_size)

    video_counter = 0
    for subdir, dirs, files in os.walk(os.path.join(src_path,flow_dir)):
        if len(files) == 0:
            continue

        print '['+str(video_counter)+']  '+subdir

        dirs = subdir.strip().split(os.sep)
        last_subdir = dirs[len(dirs)-1]


        video_dir = last_subdir+'_{:0>7d}/'.format(video_counter)
        flow_dir = dst_path+'/'+video_dir
        mkdir_p(flow_dir)

        file_counter = 0
        for f in files:
            flo, ext = os.path.splitext(f)
            name, fid = flo.split('_')
            fid = int(fid)

            if of_full_res:
                x_flow_file = 'flow_x_full_{:0>4d}'.format(fid)
                y_flow_file = 'flow_y_full_{:0>4d}'.format(fid)
            else:
                x_flow_file = 'flow_x_{:0>4d}'.format(fid)
                y_flow_file = 'flow_y_{:0>4d}'.format(fid)


            # flo2jpg(subdir+'/'+f, video_dir+x_flow_file, video_dir+y_flow_file, 32, 32)
            # First crop to 244 by offsets then scalle to 32x32
            flo2jpg(subdir+'/'+f, flow_dir+x_flow_file, flow_dir+y_flow_file, offsets,
                    crop_size, crop_size, of_scale, of_scale)


            if not of_full_res:
                for img_dir in image_dirs:
                    img_path = os.path.join(src_path, img_dir)
                    dst_img_path = os.path.join(dst_rgb, img_dir, video_dir)
                    mkdir_p(dst_img_path)
                    src_img_filename = img_path+'/'+last_subdir+'/frame_{:0>4}.png'.format(fid)
                    dst_img_filename = os.path.join(dst_img_path, 'image_{:0>4d}'.format(fid))
                    crop_img(src_img_filename, dst_img_filename, crop_size, crop_size, offsets)

                    src_img_filename = img_path+'/'+last_subdir+'/frame_{:0>4}.png'.format(fid+1)
                    dst_img_filename = os.path.join(dst_img_path, 'image_{:0>4d}'.format(fid+1))
                    crop_img(src_img_filename, dst_img_filename, crop_size, crop_size, offsets)


        video_counter += 1


    return



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


def build_image_list(path, train_filename, val_filename, maxLen = None, stride=1):

     #10000
    imgList = []
    count = 0

    for p in path:
        pd = p.strip().split(os.sep)
        parent_subdir = pd[len(pd)-1]
        for subdir, dirs, files in os.walk(p):
            dirs = subdir.strip().split(os.sep)
            last_subdir = dirs[len(dirs)-1]

            print "[",count+1,"]  Processing: ",p,' => ',subdir
            count += 1

            gc.disable()

            files.sort()
            files_num = len(files)
            for file_id in range(0, files_num):
                file = files[file_id]

                filename, ext = os.path.splitext(file)
                toks = filename.split('_')
                frame_name = toks[0]
                if len(toks) < 2 or (frame_name != 'image' and frame_name != 'frame'):
                    continue

                id_len = len(toks[1])
                img_id = int(toks[1])
                img_id1_str = str(img_id).rjust(id_len, '0')
                img_id2_str = str(img_id+stride).rjust(id_len, '0')

                img1 = file
                img1_path = os.path.join(subdir, img1)
                img2 = frame_name+'_'+img_id2_str+ext
                img2_path = os.path.join(subdir, img2)

                ofx = 'flow_x_'+img_id1_str+ext
                ofx_path = os.path.join(subdir, ofx)
                ofy = 'flow_y_'+img_id1_str+ext
                ofy_path = os.path.join(subdir, ofy)

                if os.path.isfile(img2_path): # and os.path.isfile(ofx_path) and os.path.isfile(ofy_path):
                    record = [os.path.join(parent_subdir, last_subdir, img1), os.path.join(parent_subdir, last_subdir, img2),
                              os.path.join(last_subdir, ofx), os.path.join(last_subdir, ofy)]
                    recordStr = ' '.join(record)
                    imgList.append(recordStr)
                # print recordStr


                if maxLen <> None and len(imgList) > maxLen:
                    break

            gc.enable()

            if maxLen <> None and len(imgList) > maxLen:
                break

    # Split into train and val datasets

    random.shuffle(imgList)
    listLen = len(imgList)
    trainSize = int(listLen*0.7)

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

# def layer2polar(layers, mag):
#     h, w = mag.shape
#     bins = layers.shape[0]
#     bin_width = 2.0*np.pi/bins
#     angle = np.zeros((h, w))
#
#     d = 0.0
#     for i in range(0, bins):
#         d += bin_width
#         print layers[i]
#         layer_dir = layers[i]*d
#         angle += layer_dir
#         print angle
#
#
#     return angle, mag
#
#
# def of2polar(x,y):
#     cpx = x+(y*(1j))
#     mag = np.abs(cpx)
#     angle_src = np.angle(cpx) # +/- PI
#
#     # Convert to range 0..2pi
#     os = angle_src < 0.0
#     angle_src[os]=2*np.pi+angle_src[os]
#
#     return angle_src, mag
#
#
# def polar2Layers(angle_src, mag, bins, min_mag):
#
#     h, w = mag.shape
#     bin_width = 2.0*np.pi/bins
#     layers = np.zeros((bins, h, w))
#
#     noise_of = mag < min_mag # ??
#
#     angle = angle_src.copy()
#     e=0.0
#     for i in range(0, bins):
#
#         e += bin_width
#         # print e
#         dir_p = angle < e
#         # print dir_p
#         layers[i][dir_p] = 1
#         layers[i][noise_of] = 0
#
#         # flag the already recorded angles
#         angle[dir_p] = 99
#
#
#     return layers, mag
#
#
# def showPolar(angle_src, mag):
#     plt.subplot(121)
#     # plt.imshow(mag, extent=[-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi])
#     plt.imshow(mag,  extent=[0, 224, 224, 0])
#     plt.title('Magnitude of exp(x)')
#
#     plt.subplot(122)
#     plt.imshow(angle_src, extent=[-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi])
#     #plt.imshow(angle_src)
#     plt.title('Phase (angle) of exp(x)')
#     # plt.show()


class DBuilder:

    def __init__(self):
        self.image_sum = None
        self.compute_mean = False
        self.encode = True
        self.new_height = 0
        self.new_width = 0
        self.recompress = False
        self.img_count = 0
        self.mean = None
        self.max_images = 0

        self.ofmin = 9999999
        self.ofmax = -9999999
        self.mag_max = -9999999

        pass


    #@profile
    def load_of_image(self, filename, key=None):

        image1 = caffe.io.load_image(filename, False)
#        image1 *= 255.0
        if len(image1.shape) > 2 and image1.shape[2] == 1:
            image1 = image1[:,:,0]

        # A Hack!!
        # Crop rectangular central part of the image before scalling to the new width/height
        crop_height = image1.shape[0]
        crop_width = image1.shape[0]
        crop_height = 224
        crop_width = 224
        t = (image1.shape[0]-crop_height)/2
        l = (image1.shape[1]-crop_width)/2
        image1 = image1[t:t+crop_height, l:l+crop_width]

        if key != None:
            path = 'videos/'
            video_id = int(key[:5])
            img_id = int(key[5:])

            video_path = path+'{:0>7d}'.format(video_id)
            mkdir_p(video_path)

            toks = filename.split('_')

            fname = video_path+'/flow_'+toks[-2]+'_{:0>4d}_224.jpg'.format(img_id)
            pil = PIL.Image.fromarray(np.uint8(image1))
            pil.save(fname, format='JPEG', quality=90)

        image1 = cv2.resize(image1, (self.new_width, self.new_height))

        # Transform to caffe's format requirements
        if image1.ndim == 3:
            print "ERROR. OF image has three channels!!"
            sys.exit(0)
            # Transpose to (channels, height, width)
            image = image1.transpose((2,0,1))

        elif image1.ndim == 2:
            # Add a channels axis
            # image = image1[np.newaxis,:,:]
            image1 = image1.flatten()

        return image1

    #@profile
    def of2datum(self, ofx_filename, ofy_filename, key=None):

        path = 'videos/'
        ofx_npy_file, ex = os.path.splitext(ofx_filename)
        path, filename = os.path.split(ofx_npy_file)
        toks = ofx_npy_file.split('_')
        ofx_npy_file = path+'/flow_x_full_'+toks[-1]
        ofx_npy_file += '.npy'
        if os.path.isfile(ofx_npy_file):
            ofx = np.load(ofx_npy_file)
            #ofx = ofx/400.0 + 0.5
            #ofx += 0.5
            # self.ofmin = min(self.ofmin, ofx.min())
            # self.ofmax = max(self.ofmax, ofx.max())
        else:
            ofx = self.load_of_image(ofx_filename, key)

        ofy_npy_file, ex = os.path.splitext(ofy_filename)
        path, filename = os.path.split(ofy_npy_file)
        toks = ofy_npy_file.split('_')
        ofy_npy_file = path+'/flow_y_full_'+toks[-1]
        ofy_npy_file += '.npy'
        if os.path.isfile(ofy_npy_file):
            ofy = np.load(ofy_npy_file)
            # ofy = ofy/f400.0 + 0.5
            #ofy += 0.5
            # self.ofmin = min(self.ofmin, ofy.min())
            # self.ofmax = max(self.ofmax, ofy.max())
        else:
            ofy = self.load_of_image(ofy_filename, key)

        # set invalid flow pixels to zero
        invalid = ofx > 1e9
        ofx[invalid] = 0.0
        invalid = ofy > 1e9
        ofy[invalid] = 0.0

        if 0:
            of_vec = np.append(ofx, ofy)
            of_vec = of_vec[np.newaxis, np.newaxis, :]
        else:
            of_vec = np.array([ofx, ofy])

        # Subtract mean
        #of_vec = (of_vec-0.5)

        # self.ofmin = min(self.ofmin, of_vec.min())
        # self.ofmax = max(self.ofmax, of_vec.max())

        # print 'Min=',of_vec.min(),'   Max=',of_vec.max()

        if self.compute_mean:
            if self.image_sum is None:
                self.image_sum = of_vec.flatten()
            else:
                self.image_sum += of_vec.flatten()
            self.img_count += 1

        if key != None:
            self.label_to_img_raw(path, key, of_vec[0][0], 'src')

        # set invalid flow pixels to zero
        # invalid = of_vec > 1e9
        # of_vec[invalid] = 0.0

        of_hist = True
        if of_hist:
            ofx_new = cv2.resize(ofx, (56, 56))
            ofy_new = cv2.resize(ofy, (56, 56))
        
            theta, mag = utils.of2polar(ofx_new, ofy_new)
#             normalize magnitude
            mag = mag / 452.3
            if self.mag_max < mag.max():
                self.mag_max = mag.max()

            layers, mag_dst = utils.polar2Layers(theta, mag, 64, 0.0)
            of_vec = np.append(layers, [mag_dst],  axis=0)


        datum = caffe.io.array_to_datum(of_vec.astype(float))

        return datum


    def img2datum(self, filename):

        label = 0
        recompress = self.recompress
        image1 = caffe.io.load_image(filename, False)
        image1 *= 255
        if len(image1.shape) > 2 and  image1.shape[2] == 1:
            image1 = image1[:,:,0]

        crop_height = image1.shape[0]
        crop_width = image1.shape[0]
        t = (image1.shape[0]-crop_height)/2
        l = (image1.shape[1]-crop_width)/2
        image1 = image1[t:t+crop_height, l:l+crop_width]

        image1 = cv2.resize(image1, (self.new_width, self.new_height))
        recompress = True

        # if (self.new_height!=0 and self.new_height!=0) and \
        #         (self.new_height != image1.shape[1] or self.new_height != image1.shape[0]):
        #
        #     image1 = cv2.resize(image1, (self.new_width, self.new_height))
        #     recompress = True

        if self.compute_mean:
            if self.image_sum is None:
                self.image_sum = image1/255.0
            else:
                self.image_sum += image1/255.0
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

        count = 0
        dup_count = 0
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

                    #key = '{:0>10d}'.format(block_first + in_idx)
                    key = key_list[block_first+in_idx]

                    if labels:
                        # im_dat = self.of2datum(os.path.join(root, img_file[0]),  os.path.join(root, img_file[1]), key)
                        im_dat = self.of2datum(os.path.join(root, img_file[0]),  os.path.join(root, img_file[1]))
                    else:
                        im_dat = self.img2datum(os.path.join(root, img_file))

                    im_dat_str = im_dat.SerializeToString()

                    #print key
                    if not in_txn.put(key, im_dat_str, dupdata=False, overwrite=False):
                        dup_count+=1
                        print
                        print dup_count," Already exist:  ",key

                    count += 1
                    if (count % 1000) == 0 or count < 2000:
                        string_ = str(block_first + in_idx + 1) + ' / ' + str(len(img_list))
                        sys.stdout.write("\r%s" % string_)
                        sys.stdout.flush()

                    #print in_txn.stat(in_db_label)
            in_db_label.close()

        print
        print "Records num: ",count
        print "Record duplicates: ", dup_count
        print "Written: ",(count-dup_count)
        print "Max mag: ",self.mag_max

        return


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
            key = '{:0>5d}{:0>5d}'.format(video_id, frame_id)
        else:
            # Max 99 999 videos
            # Max 99 999 images
            key = '{:0>5d}{:0>5d}'.format(video_id, frame_id)

        return frame_id, key

    def get_img_list(self, img_list_file):

        imgs = []
        keys = []
        dc = {}

        gc.disable()
        count = 0
        for line in fileinput.input(img_list_file):

            img1_file, img2_file, ofx_file, ofy_file = line.strip().split(' ')
            #print "[",count,"] ", img1_file, img2_file, ofx_file, ofy_file

            if img1_file not in dc:
                id, key = self.get_img_key(img1_file, False)
                imgs.append(img1_file)
                keys.append(key)
                dc[img1_file] = 1

            if img2_file not in dc:
                id, key = self.get_img_key(img2_file, False)
                imgs.append(img2_file)
                keys.append(key)
                dc[img2_file] = 1

            count += 1
            if self.max_images > 0 and  count > self.max_images:
                break

            #string_ = str(block_first + in_idx + 1) + ' / ' + str(len(img_list))
            if (count % 1000) == 0:
                sys.stdout.write("\r%d" % count)
                sys.stdout.flush()

        gc.enable()
        fileinput.close()
        print

        return imgs, keys


    def get_of_list(self, img_list_file):
        print 'Reading optical flow iamge list ',img_list_file
        imgs = []
        keys = []

        dc = {}

        gc.disable()
        count = 0
        for line in fileinput.input(img_list_file):

            img1_file, img2_file, ofx_file, ofy_file = line.strip().split(' ')
            #print "[",count,"] ", img1_file, img2_file, ofx_file, ofy_file

            imgs.append([ofx_file, ofy_file])
            id, key = self.get_img_key(ofx_file, True)
            keys.append(key)

            count += 1
            if self.max_images > 0 and  count > self.max_images:
                break

            #string_ = str(block_first + in_idx + 1) + ' / ' + str(len(img_list))
            if (count % 1000) == 0 or count < 1000:
                sys.stdout.write("\r%d" % count)
                sys.stdout.flush()

        gc.enable()
        fileinput.close()
        print

        return imgs, keys


    def label_to_img(self, path, key, flow_vec, suffix):

        flow_mean = 127.0

        flow_vec_scaled = (flow_vec*255.0)+flow_mean

        self.label_to_img_raw(path, key, flow_vec_scaled, suffix)


    def label_to_img_raw(self, path, key, flow_vec, suffix):

        video_id = int(key[:5])
        img_id = int(key[5:])

        x_flow = flow_vec[:1024].reshape(32,32)
        y_flow = flow_vec[1024:].reshape(32,32)

        path += '{:0>7d}'.format(video_id)
        mkdir_p(path)

        filename = path+'/flow_x_{:0>4d}_'.format(img_id)+suffix+'.jpg'
        pil = PIL.Image.fromarray(np.uint8(x_flow))
        pil.save(filename, format='JPEG', quality=90)

        filename = path+'/flow_y_{:0>4d}_'.format(img_id)+suffix+'.jpg'
        pil = PIL.Image.fromarray(np.uint8(y_flow))
        pil.save(filename, format='JPEG', quality=90)


    def labels_to_imgs(self, lmdb_name, labels):

        self.image_sum = None
        self.img_count = 0

        lmdb_env = lmdb.open(lmdb_name, map_size=int(1e12))
        lmdb_txn = lmdb_env.begin()  # equivalent to mdb_txn_begin()
        lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()

#        lmdb_cursor.first()
        lmdb_cursor.get('{:0>5d}{:0>5d}'.format(999, 1900))

        count = 0
        m = 1.0/255.0
        for key, value in lmdb_cursor:

            if key != '0099901900':
                continue

            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)

            if labels:
                data = np.array(datum.float_data).astype(float).reshape(datum.channels, datum.height, datum.width)

                flow_vec = data[0][0]
                #self.label_to_img('/home/jiri/Lake/DAR/share/datasets/THUMOS2015/thumos15_validation//thumos15_video_validation_', key, flow_vec)
                self.label_to_img('videos_', key, flow_vec, 'test')

            count += 1
            if (count % 1000) == 0:
                string_ = '['+str(count)+']  Key='+key
                sys.stdout.write("\r%s" % string_)
                sys.stdout.flush()

        lmdb_env.close()

        return

   # @profile
    def calc_mean(self, lmdb_name, labels):

        self.s1 = 0.0
        self.s2 = 0.0
        self.ofmin = 99999999
        self.ofmax = -9999999999

        self.image_sum = None
        self.img_count = 0

        lmdb_env = lmdb.open(lmdb_name, map_size=int(1e12))
        lmdb_txn = lmdb_env.begin()  # equivalent to mdb_txn_begin()
        lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()
        # lmdb_cursor.get('{:0>10s}'.format('_6')) #  get the data associated with the 'key' 1, change the value to get other images

        lmdb_cursor.first()
        count = 0
        m = 1.0/255.0
        for key, value in lmdb_cursor:

            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)

            if labels:
                data = np.array(datum.float_data).astype(float).reshape(datum.channels, datum.height, datum.width)
                #data *= m
                if self.image_sum is None:
                    self.image_sum = data
                else:
                    self.image_sum += data

                d = data.flatten()
                self.ofmin = min(self.ofmin, d.min())
                self.ofmax = max(self.ofmax, d.max())

                self.s1 += np.sum(d)
                self.s2 += np.dot(d, d)

                self.img_count += 1

                # n = self.image_sum / float(self.img_count)
                # n = np.mean(n)
                # stdv = np.sqrt(self.s2/(self.img_count*datum.width) - (n*n))
                # print stdv

            else:
                data = np.fromstring(datum.data, dtype=np.uint8)
                # image_jpg = data.astype(np.uint8)
                pil = PIL.Image.open(StringIO(data))
                #image = np.array(pil.getdata()).reshape(pil.size[0], pil.size[1], 3) / 255.0
                image = np.array(pil)
                image = image.astype(float) * m

                if self.image_sum is None:
                    self.image_sum = image
                else:
                    self.image_sum += image

                self.img_count += 1

            count += 1
            if (count % 1000) == 0:
                string_ = '['+str(count)+']  Key='+key
                sys.stdout.write("\r%s" % string_)
                sys.stdout.flush()

        lmdb_env.close()

        mean_vec = self.image_sum / float(self.img_count)
        self.mean = mean_vec

        # self.stdv = np.sqrt(self.img_count*self.s2 - self.s1*self.s1)/self.img_count

        n = np.mean(self.mean)
        self.stdv = np.sqrt(self.s2/(self.img_count*datum.width) - (n*n))

        print "mean=", n
        print "std=",self.stdv
        print "min/max = ", self.ofmin,' / ', self.ofmax

        return mean_vec


    def subtract_mean(self, lmdb_name, labels):

        print "Subtracting mean for ", lmdb_name
        lmdb_env = lmdb.open(lmdb_name, map_size=int(1e12))
        lmdb_txn = lmdb_env.begin(write=True)  # equivalent to mdb_txn_begin()
        lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()
        # lmdb_cursor.get('{:0>10s}'.format('_6')) #  get the data associated with the 'key' 1, change the value to get other images

        lmdb_cursor.first()
        count = 0
        m = 1.0/255.0
        for key, value in lmdb_cursor:

            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)

            if labels:
                data = np.array(datum.float_data).astype(float).reshape(datum.channels, datum.height, datum.width)

                #new_val = (data - self.mean) #/255.0
                # min = -347.364898682
                # max = 284.490142822
                # new_val = (data - min)/ (max-min)

                range = 100.0 #2*self.stdv
                n = np.mean(self.mean)
                data_zero = data - n
                data_zero = data_zero+range
                min = data_zero<0
                max = data_zero>range

                data_zero[min] = 0.0
                data_zero[max] = 1.0

                new_val = data_zero / range

                # DEBUG
                if 0:
                    flow_vec = new_val[0][0]
                    self.label_to_img('videos/', key, flow_vec, 'reco')

                datum = caffe.io.array_to_datum(new_val.astype(float))

            else:
                data = np.fromstring(datum.data, dtype=np.uint8)
                # image_jpg = data.astype(np.uint8)
                pil = PIL.Image.open(StringIO(data))
                #image = np.array(pil.getdata()).reshape(pil.size[0], pil.size[1], 3) / 255.0
                image = np.array(pil)

                new_image = image-self.mean

                pil = PIL.Image.fromarray(np.uint8(new_image))
                s1 = StringIO()
                pil.save(s1, format='JPEG', quality=90)
                datum.data = s1.getvalue()

            datum_str = datum.SerializeToString()
            #lmdb_txn.put(key, datum_str, dupdata=False)
            result = lmdb_txn.replace(key, datum_str)
            if not result:
                print "Error lmdb_txn.replace()"

            count += 1
            if (count % 1000) == 0 or count < 1000:
                string_ = '['+str(count)+']  Key='+key
                sys.stdout.write("\r%s" % string_)
                sys.stdout.flush()


        lmdb_txn.commit()
        lmdb_env.close()
        print

        return


    def load_mean(self, dir, labels):
        if labels:
            print "Loading mean vector:  "+dir+'/mean.npy'
            self.mean = np.load(dir+'/mean.npy')
        else:
            print "Loading image mean :  "+dir+'/mean.npy'
            self.mean = np.load(dir+'/mean.npy')
            # blob=self.array_to_blobproto(self.mean)
            # self.save_binaryproto(dir+'/data_mean.binaryproto', blob)


    def save_mean(self, dir, labels):
        if self.compute_mean:
            if labels:
                print "Saving mean vector:  "+dir+'/mean'
                #mean_vec = 255.0 * self.image_sum / float(self.img_count)
                np.save(dir+'/mean', self.mean)
            else:
                print "Saving mean image:  "+dir+'/mean.jpg'
                mean_img = 255.0 * self.image_sum / float(self.img_count)
                np.save(dir+'/mean', mean_img)

                blob=self.array_to_blobproto(mean_img)
                self.save_binaryproto(dir+'/data_mean.binaryproto', blob)

                pil = PIL.Image.fromarray(np.uint8(mean_img))
                pil.save(dir+'/mean.jpg', format='JPEG', quality=90)


    def blobproto_to_array(blob, return_diff=False):
        """Convert a blob proto to an array. In default, we will just return the data,
        unless return_diff is True, in which case we will return the diff.
        """
        if return_diff:
            return np.array(blob.diff).reshape(
                blob.num, blob.channels, blob.height, blob.width)
        else:
            return np.array(blob.data).reshape(
                blob.num, blob.channels, blob.height, blob.width)


    def array_to_blobproto(self, arr, diff=None):
        """
        Converts a 4-dimensional array to blob proto. If diff is given, also
        convert the diff. You need to make sure that arr and diff have the same
        shape, and this function does not do sanity check.
        """
        # if arr.ndim != 4:
        #     raise ValueError('Incorrect array shape.')
        blob = caffe_pb2.BlobProto()
        blob_array = arr[np.newaxis,:,:,:]
        blob_array = blob_array.transpose(0,3,1,2)
        blob.num, blob.channels, blob.height, blob.width = blob_array.shape
        blob.data.extend(blob_array.astype(float).flat)
        if diff is not None:
            blob.diff.extend(diff.astype(float).flat)

        return blob


    def save_binaryproto(self, filename, blob):
        binaryproto_file = open(filename, 'wb' )
        binaryproto_file.write(blob.SerializeToString())
        binaryproto_file.close()


    def get_mean_RGB(self):

        r = self.mean[:,:,0]
        dim = (r.shape[0]*r.shape[1])
        r = np.sum(r)/dim

        g = self.mean[:,:,1]
        g = np.sum(g)/dim

        b = self.mean[:,:,2]
        b = np.sum(b)/dim

        return [r,g,b]


    def build_db(self, name, img_list_file, labels = False):

        print 'Building ',name
        print 'Building image list...'

        imgs = []
        keys = []

        if labels:
            imgs, keys = self.get_of_list(img_list_file)
        else:
            imgs, keys = self.get_img_list(img_list_file)

        print 'Building LMDB...'
        dbb.write_images(name, path, imgs, keys, labels)
        print '\nDone'

        mean_vec = self.image_sum / float(self.img_count)
        self.mean = mean_vec

        self.save_mean(name, labels)

        print self.ofmax
        print self.ofmin


    def show_label(self, lmdb_label_name, id):


        return


    def get_image(self, lmdb_label_name, id, img=False):
        lmdb_env = lmdb.open(lmdb_label_name, map_size=int(1e12))
        lmdb_txn = lmdb_env.begin()  # equivalent to mdb_txn_begin()
        lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()
        # lmdb_cursor.get('{:0>10s}'.format('_6')) #  get the data associated with the 'key' 1, change the value to get other images

        print lmdb_env.info()

        count = 0
        for key, value in lmdb_cursor:
            count += 1
            print count,'  ',(key)

        print "------------"
        #lmdb_cursor.first()
        lmdb_cursor.get(id) # '{:0>10s}'.format('_6')

        value = lmdb_cursor.value()
 #       key = lmdb_cursor.key()

        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
#        image = np.zeros((datum.channels, datum.height, datum.width))

        # flat_x = np.fromstring(datum.data, dtype=np.uint8)
        # x = flat_x.reshape(datum.channels, datum.height, datum.width)
        data = np.array(datum.float_data).astype(float).reshape(datum.channels, datum.height, datum.width)
        print data

        if img:
            jpeg=True
            if jpeg:
                # for jpeg payload
                image = np.fromstring(datum.data, dtype=np.uint8)
                image = image.astype(np.uint8)
                image.tofile('img.jpg')
                cv2.imshow("Image", image)

        lmdb_cursor.first()
        lmdb_env.close()


if __name__ == "__main__":

    path = caffe_root+'/share/datasets/THUMOS2015/thumos15_validation-rgbflow/'
    path = caffe_root+'/share/datasets/MPI_Sintel/MPI_Sintel-rgbflow/'
    path = caffe_root+'/share/datasets/MPI_Sintel/MPI_Sintel-flow'

    scene = ''
    scene = '-mpi-large'
    #scene = '-s'
    dataset_type = 'train'
    dataset_type = 'val'
    dataset_filename = dataset_type+'-of'+scene+'.txt'
    lmdb_images_name = dataset_type+'-of-images-lmdb-hist'+scene
    lmdb_labels_name = dataset_type+'-of-labels-lmdb-hist'+scene
    #lmdb_labels_name = dataset_type+'-of-labels-lmdb-orig'

    train_filename = 'train-of'+scene+'.txt'
    val_filename = 'val-of'+scene+'.txt'

    build_img_list = False
    #build_img_list = True
    build_img_lmdb = True

    if 0:
        path_mpi = caffe_root+'/share/datasets/MPI_Sintel/'
        path = caffe_root+'/share/datasets/MPI_Sintel/MPI_Sintel-flow/'
        path_rgb = caffe_root+'/share/datasets/MPI_Sintel/MPI_Sintel-rgb'
        # flow2jpg_dir(path_mpi +'/training/flow/', path_mpi +'/training/final/', path)
        make_rgbflow_from_dir(path_mpi +'/training/', ['clean', 'final'], 'flow', path, path_rgb)
        sys.exit(0)

    if build_img_list:
        path = [caffe_root+'/share/datasets/MPI_Sintel/MPI_Sintel-rgb/clean',
            caffe_root+'/share/datasets/MPI_Sintel/MPI_Sintel-rgb/final']
        imgList = build_image_list(path, train_filename, val_filename, None, stride=100)
        sys.exit(0)


    dbb = DBuilder()

    if 0 and len(sys.argv) > 2:
        dbb.get_image(sys.argv[1], sys.argv[2])
        sys.exit(0)

    dbb.max_images = 0
    dbb.compute_mean=True

    #dbb.labels_to_imgs(lmdb_labels_name, True)
    # dbb.load_mean('train-of-images-lmdb', False)
    # dbb.subtract_mean('train-of-images-lmdb', False)
    #mean_rgb = dbb.get_mean_RGB()
    #print mean_rgb


    #dbb.calc_mean('train-of-images-lmdb', False)
    #dbb.save_mean('train-of-images-lmdb', False)



    #dbb.calc_mean(lmdb_labels_name, True)
    #dbb.save_mean('train-of-labels-lmdb', True)

    #sys.exit(0)

    sys.argv.append('o')
    if len(sys.argv) > 1:
        if sys.argv[1] == 'i':
            dbb.compute_mean = True
            dbb.new_height = 224
            dbb.new_width = 224
            dbb.image_sum = None
            dbb.img_count = 0
            dbb.build_db(lmdb_images_name, dataset_filename, False)


        if sys.argv[1] == 'o':
            dbb.compute_mean = True
            dbb.new_height = 32
            dbb.new_width = 32
            dbb.image_sum = None
            dbb.img_count = 0
            dbb.build_db(lmdb_labels_name, dataset_filename, True)

            #dbb.load_mean(lmdb_labels_name, True)
            #dbb.calc_mean(lmdb_labels_name, True)
            #dbb.subtract_mean(lmdb_labels_name, True)
            #dbb.calc_mean(lmdb_labels_name, True)
            #print dbb.calc_mean(lmdb_labels_name, True)





