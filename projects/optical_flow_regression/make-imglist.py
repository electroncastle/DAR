import os
import gc
import sys
import random


caffe_root = os.environ['DAR_ROOT']
sys.path.insert(0, caffe_root + '/src/caffe/python')
import caffe
import caffe.io
from caffe.proto import caffe_pb2

def build_image_list(path, of_path, train_filename, val_filename, maxLen = None, stride=1):

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


                of_file = of_path+last_subdir+'/frame_{:0>4d}.flo'.format(img_id)

                if os.path.isfile(img2_path): # and os.path.isfile(ofx_path) and os.path.isfile(ofy_path):
                    record = [img1_path, img2_path, of_file]
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



if __name__ == "__main__":
    img_path = ['/home/jiri/Lake/DAR/share/datasets/MPI_Sintel/training/final/',
                '/home/jiri/Lake/DAR/share/datasets/MPI_Sintel/training/clean/']

    offlow_path = '/home/jiri/Lake/DAR/share/datasets/MPI_Sintel/training/flow/'

    build_image_list(img_path, offlow_path, 'train_of_src_clean_final.txt', 'val_of_src_clean_final.txt')