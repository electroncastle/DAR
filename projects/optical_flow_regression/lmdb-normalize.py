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
        pass



   # @profile

    def reset_stats(self):
        self.s1 = 0.0
        self.s2 = 0.0
        self.ofmin = 99999999
        self.ofmax = -9999999999

        self.image_sum = None
        self.img_count = 0


    def print_stats(self):

        vec_len = len(self.mean_vec)
        n = self.mean
        self.stdv = np.sqrt(self.s2/(self.img_count * vec_len) - (n*n))
        # self.stdv = np.sqrt(self.img_count*self.s2 - self.s1*self.s1)/self.img_count

        print "mean=", n
        print "std=",self.stdv
        print "min/max = ", self.ofmin,' / ', self.ofmax


    def calc_stats(self, lmdb_name, xxx):

        lmdb_env = lmdb.open(lmdb_name, map_size=int(1e12))
        lmdb_txn = lmdb_env.begin()  # equivalent to mdb_txn_begin()
        lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()
        # lmdb_cursor.get('{:0>10s}'.format('_6')) #  get the data associated with the 'key' 1, change the value to get other images

        lmdb_cursor.first()
        count = 0
        for key, value in lmdb_cursor:

            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)

            if 1:
                data = np.array(datum.float_data).astype(float).reshape(datum.channels, datum.height, datum.width)

                if data.shape[0] == 2:
                    xflow = data[0]
                    yflow = data[0]

                    d = data.flatten()
                    if self.image_sum is None:
                        self.image_sum = d
                    else:
                        self.image_sum += d

                    self.ofmin = min(self.ofmin, d.min())
                    self.ofmax = max(self.ofmax, d.max())

                    self.s1 += np.sum(d)
                    self.s2 += np.dot(d, d)

                    self.img_count += 1
                else:
                    print "Record has wrong shape!"

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
            if (count % 1000) == 0 or 1:
                string_ = '['+str(count)+']  Key='+key
                sys.stdout.write("\r%s" % string_)
                sys.stdout.flush()

        lmdb_env.close()
        print
        print "Records: ",count

        self.mean_vec = self.image_sum / float(self.img_count)
        self.mean = np.mean(self.mean_vec)

        return


    def normalize(self, lmdb_name, labels):

        self.outliers_neg = 0
        self.outliers_pos = 0
        print "Normalizing ", lmdb_name

        scale_range = max(np.abs(self.ofmin), np.abs(self.ofmax))
        print "Scale range: ",scale_range
        scale_range = 3.0*self.stdv
        print "3stdv range: ",scale_range

        scale_range=127.0
        print "Fixed range: ",scale_range


        lmdb_env = lmdb.open(lmdb_name, map_size=int(1e12))
        lmdb_txn = lmdb_env.begin(write=True )  # equivalent to mdb_txn_begin()
        lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()
        # lmdb_cursor.get('{:0>10s}'.format('_6')) #  get the data associated with the 'key' 1, change the value to get other images

        lmdb_cursor.first()
        count = 0
        width = -1
        height = -1
        channels = -1
        for key, value in lmdb_cursor:

            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)

            if labels:
                data = np.array(datum.float_data).astype(float).reshape(datum.channels, datum.height, datum.width)
                if width < 0:
                    width = datum.width
                    height = datum.height
                    channels = datum.channels

                # Trim the data to be withing scale_range
                data_zero = data
                min_id = data_zero < -scale_range
                max_id = data_zero > scale_range

                data_zero[min_id] = -1.0
                data_zero[max_id] = 1.0

                omin = np.count_nonzero(min_id)
                omax = np.count_nonzero(max_id)
                #if omin>0 or omax>0: print key, "  outliers [-/+]  ",omin,"/",omax

                self.outliers_neg += omin
                self.outliers_pos += omax

                # Scale data to range -1..1
                data_zero /= scale_range


                new_val = data_zero

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

            if 1:
                result = lmdb_txn.replace(key, datum_str)

                if result == None:
                    print "Error lmdb_txn.replace()    No key exists"

            count += 1
            if (count % 1000) == 0 or count < 1000:
                string_ = '['+str(count)+']  Key='+key
                sys.stdout.write("\r%s" % string_)
                sys.stdout.flush()


        lmdb_txn.commit()
        lmdb_env.close()
        print
        print "Normalization done"
        print "Record: ",count
        print "Total:  outliers -/+", self.outliers_neg," / ", self.outliers_pos
        print "Outliers % ", (100.0*(self.outliers_neg + self.outliers_pos)/(float)(count*(width*height*channels)))

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


if __name__ == "__main__":

    if len(sys.argv)  < 2:
        print "Usage:"
        print sys.argv[0],"  <lmdb_name> <lmdb_name> ....."
        sys.exit(0)

    dbb = DBuilder()
    dblist = sys.argv[1:]

    dbb.reset_stats()
    for db in dblist:
        dbb.calc_stats(db, True)
        dbb.print_stats()
    print

    for db in dblist:
        dbb.normalize(db, True)
    print

    print "New Stats:"
    dbb.reset_stats()
    for db in dblist:
        dbb.calc_stats(db, True)
        dbb.print_stats()





