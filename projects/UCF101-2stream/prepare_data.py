import os
import sys
from __builtin__ import file
import lmdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Make sure that caffe is on the python path:
caffe_root = '/home/jiri/Lake/HAR/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe


def extractImage():

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

    return


def processClassData(filename, outFile, labels, video_imgs_path):
        dset = open(filename, 'rt')
        print filename

        for line in dset:
            # print line
            className, videoFile = os.path.split(line)
            videoFile, ext = os.path.splitext(videoFile)

            # Get video length (num frames)
            frames = 0
            classPath = video_imgs_path+'/'+className+'/'+videoFile;
            for filename in os.listdir(classPath):
                if filename.startswith('image_'):
                    frames += 1

            # print frames

            # Write train record
            label = labels[className]
            record = classPath+' '+str(frames)+' '+str(label)+'\n'
            # print record
            outFile.write(record)
            outFile.flush()

        dset.close()


def ren():
    data_root = "/home/jiri/Lake/HAR/datasets/UCF-101/"

    # Source
    video_imgs_path = data_root +"/UCF101-rgbflow/"

    for subdir, dirs, files in os.walk(video_imgs_path):
        print subdir
        for file in files:
            # print os.path.join(subdir, file)
            if (file.startswith('image__')):
                img, sufix= file.split('__')
                fileNew = 'image_'+sufix
                os.rename(os.path.join(subdir, file), os.path.join(subdir, fileNew))
                # print os.path.join(subdir, file), os.path.join(subdir, fileNew)

            # filepath = subdir + os.sep + file
            #
            # if filepath.endswith(".asm"):
            #     print (filepath)


def shiftClassLabel(input, output):
    trainFileNew = open(output, 'wt')
    trainFile = open(input, 'rt')

    for line in trainFile:
        # print line
        path, length, label = line.split(' ')
        label = int(label)

        record = path+' '+length+' '+str(label-1)
        trainFileNew.write(record+'\n')

    trainFile.close()
    trainFileNew.close()


def reclass():
    data_root = "/home/jiri/Lake/HAR/datasets/UCF-101/"

    # Source
    video_imgs_path = data_root +"/UCF101-rgbflow/"
    train_split = data_root + "/ucfTrainTestlist/"

    # output files
    rgb_img_path = data_root +"/rgb/"
    flow_img_path = data_root +"/flow/"
    labels_file = data_root + "labels.txt"
    labels_file_new = data_root + "labels-new.txt"
    train_file = data_root + "/train-1.txt"
    test_file = data_root + "/val-1.txt"

    train_file_new = data_root + "/train-1-new.txt"
    test_file_new = data_root + "/val-1-new.txt"

    labels_file_new = open(labels_file_new, 'wt')
    labelsFile = open(labels_file, 'rt')

    for line in labelsFile:
        # print line
        label, className = line.split(' ')
        label = int(label)

        record = str(label-1)+' '+(className.strip())
        labels_file_new.write(record+'\n')

    labelsFile.close()
    labels_file_new.close()

    shiftClassLabel(train_file, train_file_new)
    shiftClassLabel(test_file, test_file_new)


def run():

    data_root = "/home/jiri/Lake/HAR/datasets/UCF-101/"

    # Source
    video_imgs_path = data_root +"/UCF101-rgbflow/"
    train_split = data_root + "/ucfTrainTestlist/"

    # output files
    rgb_img_path = data_root +"/rgb/"
    flow_img_path = data_root +"/flow/"
    labels_file = data_root + "labels.txt"
    train_file = data_root + "/train.txt"
    test_file = data_root + "/test.txt"

    labelsFile = open(labels_file, 'rt')
    labels = {}
    for line in labelsFile:
        # print line
        label, className = line.split(' ')
        labels[className.strip()]=label

    labelsFile.close()

    trainFile = open(train_file, 'wt')
    testFile = open(test_file, 'wt')
    for i in range(1,2):

        # Process train set
        filename = train_split+'/trainlist'+'{:0>2d}'.format(i)+'.txt'
        processClassData(filename, trainFile, labels, video_imgs_path)

        filename = train_split+'/testlist'+'{:0>2d}'.format(i)+'.txt'
        processClassData(filename, testFile, labels, video_imgs_path)

    trainFile.close()
    testFile.close()

        #train_file = open('train_split', 'w')



    return


if __name__ == "__main__":
    reclass()
    # ren()
    # run()

