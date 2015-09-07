import lmdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Make sure that caffe is on the python path:
caffe_root = '/home/jiri/Lake/HAR/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

db_path = '/home/jiri/attic/digits-2.0/digits/digits/jobs/20150729-111817-b2c8/train_db/'

lmdb_env = lmdb.open(db_path)  # equivalent to mdb_env_open()
lmdb_txn = lmdb_env.begin()  # equivalent to mdb_txn_begin()
lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()
# lmdb_cursor.get('{:0>10s}'.format('_6')) #  get the data associated with the 'key' 1, change the value to get other images

for key, value in lmdb_cursor:
        print(key)

lmdb_cursor.first()



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
else:
    # raw bitmap ?
    image = caffe.io.datum_to_array(datum)
    image = np.transpose(image, (1, 2, 0))
    image = image[:, :, (2, 1, 0)]
    image = image.astype(np.uint8)
    mpimg.imsave('out.png', image)
