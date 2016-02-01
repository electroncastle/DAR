#!/usr/bin/python

__author__ = "Jiri Fajtl"
__email__ = "ok1zjf@gmail.com"
__status__ = "Research"
__license__ = "LGPL"
__date__ = "20/10/2015"
__version__ = "1.0.0"

import os
import sys
import numpy as np
import glob
import cv2


# if 'DAR_ROOT' not in os.environ:
#     print 'FATAL ERROR. DAR_ROOT not set, exiting'
#     sys.exit()

dar_root = os.environ['DAR_ROOT']
app_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

# DAR_ROOT path override
# dar_root = '/home/jiri/Lake/DAR/src/caffe-fcn/'
# dar_root = '/home/jiri/Lake/DAR/src/caffe/'
#dar_root = '/home/jiri/Lake/DAR/src/caffe-recurrent/bin/'
os.environ['DAR_ROOT'] = dar_root

sys.path.insert(0, dar_root + '/python')
import caffe

# dar_root = '/home/jiri/Lake/DAR/'
# sys.path.insert(0, dar_root + '/projects/optical_flow_regression')
# import utils


class DataBlob:

    def __init__(self):
        self.batch_data = None
        self.continuation_markers = None
        self.labels = None
        self.stride = -1
        self.sampling = -1
        self.data_root_dir=''


class VideoProcessor:

    def __init__(self):
        self.DATASET_UCF101 = 0
        self.DATASET_THUMOS = 1
        self.DATASET_UNKNOWN = 2

        # Defaults
        self.crop_size = [224, 224]

        self.T=18
        self.N=4
        self.stride = 1
        self.sampling = 1

        self.frames_files=[]
        self.label = 0
        self.video_id=0

        self.set_default_dataset(self.DATASET_UCF101)

        self.rgb_mean_values = np.array([104.0, 117.0, 123.0])
        self.flow_mean_values = 128

        self.rgb_mean_image = self.get_rgb_mean_image(self.crop_size).transpose(1,2,0)
        self.flow_mean_image = self.get_flow_mean_image(self.crop_size)


        # Image cache
        self.image_cache_size = 500
        self.image_cache ={}
        self.image_cache_keys = []
        self.cache_images = True


    def set_default_dataset_by_name(self, dataset):

        if dataset == 'ucf':
            self.set_default_dataset(self.DATASET_UCF101)

        if dataset == 'thumos':
            self.set_default_dataset(self.DATASET_THUMOS)


    def set_default_dataset(self, dataset):

        self.datasets_root = os.path.join(dar_root, 'share/datasets/')
        self.video_frames_flow_dir = ''
        self.video_frames_rgb_dir = ''


        if dataset == self.DATASET_UCF101:
            self.dataset = dataset
            # UCF 101
            self.dataset_name = 'UCF-101'   # root dir of the whole dataset
            self.dataset_video = 'UCF101'
            self.dataset_rgb = 'UCF101-rgbflow/'
            self.dataset_flow = 'UCF101-flow/'  #'ucf101_flow_img_tvl1_gpu/'
            self.video_name = ''
            self.class_name = ''

            # The below are explicit overrides
            #self.video_name = 'v_FloorGymnastics_g16_c04'
            #self.video_name = 'v_Shotput_g07_c01'

            self.video_ext = 'avi'
            #self.class_name = 'Shotput'
            self.class_name = '' # When reading videos from the val...txt or train.....txt files the class is already
            # part of the path

            # If the files the train/val list have relative path this is the root
            self.data_train_val_root = ''
            self.data_train_split1 = 'train-2-rnd.txt'
            self.data_validation_split1 = 'val-2-rnd.txt'

            self.data_train_split1 = 'train-1-rnd.txt'
            self.data_validation_split1 = 'val-1-rnd.txt'

            self.labels_file = os.path.join(self.datasets_root, self.dataset_name, 'labels-new.txt')

        elif dataset == self.DATASET_THUMOS:
            self.dataset = dataset

            # THUMOS validation
            self.dataset_name = 'THUMOS2015'
            self.dataset_video = 'thumos15_validation'
            self.dataset_rgb = 'thumos15_validation-rgbflow'
            self.dataset_flow = 'thumos15_validation-rgbflow'

            self.video_name = 'thumos15_video_validation_0000006'
            self.video_name = ''
            self.video_ext = 'mp4'
            self.class_name = ''


            # dar_root+'/share/datasets/THUMOS2015/thumos15_validation/annotated.txt'
            self.data_validation_split1 = os.path.join(self.dataset_video, 'annotated.txt')

            # THUMOS uses some files from the UCF101
            self.labels_file = os.path.join(self.datasets_root, 'UCF-101', 'labels-new.txt')

        else:
            self.dataset = self.DATASET_UNKNOWN
            print "ERROR Wrong dataset ID !!"
            sys.exit(0)


        self.update_path()

        # If the rgbflow dir has subdirectories with the classnames
        # e.g. 'UCF101-rgbflow/FloorGymnastics/v_FloorGymnastics_g16_c04'
        #self.has_class_dirs = False


    def update_path(self):
        # Common
        self.video_path = os.path.join(self.datasets_root, self.dataset_name, self.dataset_video, self.video_name)+'.'+self.video_ext
        self.rgb_path = os.path.join(self.datasets_root, self.dataset_name, self.dataset_rgb, self.class_name, self.video_name)
        self.flow_path = os.path.join(self.datasets_root, self.dataset_name, self.dataset_flow, self.class_name, self.video_name)

        self.train_split1_file = os.path.join(self.datasets_root, self.dataset_name, self.data_train_split1)
        self.validation_split1_file = os.path.join(self.datasets_root, self.dataset_name, self.data_validation_split1)

        try:
            self.labels = np.loadtxt(self.labels_file, str, delimiter=' ')
        except:
            print 'Cannot load labels: ',self.labels_file

        return


    def get_rgb_mean_image(self, size):

        rgb_mean_image = np.array([[np.transpose([ self.rgb_mean_values[0] for i in range(size[0])])]*size[0],
                       [np.transpose([ self.rgb_mean_values[1] for i in range(size[0])])]*size[0],
                       [np.transpose([ self.rgb_mean_values[2] for i in range(size[0])])]*size[0]], dtype='float32')

        return rgb_mean_image


    def get_flow_mean_image(self, size):

        flow_mean_image = np.array([self.flow_mean_values]*size[0]*size[1], dtype='float32')

        return flow_mean_image.reshape((size[0], size[1]))


    def load_video(self, video_dir, label):
        self.label = label
        self.video_id=0

        # Split extension if there is any
        video_dir, ext = os.path.splitext(video_dir)


        # If the video dir is absolute path extract the video class and filename
        if video_dir[0] == '/':
            path, video_dir = os.path.split(video_dir)
            path, video_class = os.path.split(path)
            self.video_frames_flow_dir = os.path.join(self.flow_path, video_class, video_dir)
            self.video_frames_rgb_dir = os.path.join(self.rgb_path, video_class, video_dir)
        else:
            self.video_frames_flow_dir = os.path.join(self.flow_path, video_dir)
            self.video_frames_rgb_dir = os.path.join(self.rgb_path, video_dir)



        frames_tmpl = '%s/image_????.jpg' %(self.video_frames_rgb_dir)
        self.frames_rgb = glob.glob(frames_tmpl)
        self.frames_rgb.sort()

        frames_tmpl = '%s/flow_x_????.jpg' %(self.video_frames_flow_dir)
        self.frames_flow_x = glob.glob(frames_tmpl)
        self.frames_flow_x.sort()

        frames_tmpl = '%s/flow_y_????.jpg' %(self.video_frames_flow_dir)
        self.frames_flow_y = glob.glob(frames_tmpl)
        self.frames_flow_y.sort()

        frames_tmpl = '%s/fc6_rgb_????.npy' %(self.video_frames_rgb_dir)
        self.frames_rgb_fc6 = glob.glob(frames_tmpl)
        self.frames_rgb_fc6.sort()

        frames_tmpl = '%s/fc6_????.npy' %(self.video_frames_flow_dir)
        self.frames_flow_fc6= glob.glob(frames_tmpl)
        self.frames_flow_fc6.sort()




    #@profile
    def transform_image(self, img, mean_image, flow, crop_size, crop=4, mirror=False, transformer=None):
        h = img.shape[0]
        w = img.shape[1]

        # mirror = False
        # crop = 4

        cw = crop_size[1]
        ch = crop_size[0]

        i = 0
#        flowStackTrans = np.empty((size, ch, cw), dtype='float32')

        f = img
        if (mirror):
            if (flow):
                if i % 2 == 0:
                    # For OF x
                    f = 1-np.fliplr(f)
                else:
                    # For OF y
                    f = np.fliplr(f)
            else:
                f = np.fliplr(f)

        if (crop == 0):
            ft = f[0:ch, 0:cw]
        if (crop == 1):
            ft = f[0:ch, -cw::]
        if (crop == 2):
            ft = f[-ch::, 0:cw]
        if (crop == 3):
            ft = f[-ch::, -cw::]
        if (crop == 4):
            t = (h-ch)/2
            l = (w-ch)/2
            ft = f[t:t+ch, l:l+cw]

        # images are already read with opencv in range to 0..255
#        ft*=255.0
        data = np.subtract(ft, mean_image)
        return data


    def transform_stack(self, img_stack, mean_image, flow, crop_size, crop=4, mirror=False, transformer=None):

        h = img_stack[0].shape[0]
        w = img_stack[0].shape[1]
        size = len(img_stack)

        # mirror = False
        # crop = 4

        cw = crop_size[1]
        ch = crop_size[0]

        i = 0
        flowStackTrans = np.empty((size, ch, cw), dtype='float32')
        for fs in img_stack:

            f = fs.copy()
            if (mirror):
                if (flow):
                    if i % 2 == 0:
                        # For OF x
                        f = 1-np.fliplr(f)
                    else:
                        # For OF y
                        f = np.fliplr(f)
                else:
                    f = np.fliplr(f)

            if (crop == 0):
                ft = f[0:ch, 0:cw]
            if (crop == 1):
                ft = f[0:ch, -cw::]
            if (crop == 2):
                ft = f[-ch::, 0:cw]
            if (crop == 3):
                ft = f[-ch::, -cw::]
            if (crop == 4):
                t = (h-ch)/2
                l = (w-ch)/2
                ft = f[t:t+ch, l:l+cw]

            # flowStackTrans.append(ft)
            flowStackTrans[i]= ft
            i += 1

        # flowStackTrans = np.asarray(flowStackTrans)

        if transformer is not None:
            batch = np.transpose(flowStackTrans, (1, 2, 0))
            data = transformer.preprocess('data', batch)
        else:
            # no need to multiply by 255 when reading img by opencv
            # batch = flowStackTrans
            batch = flowStackTrans*255.0
            data = np.subtract(batch, mean_image)

        return data


    def get_image_from_cache(self, filename, color=False):

        if self.image_cache.has_key(filename):
            return self.image_cache[filename]

        if color:
            img = cv2.imread(filename, cv2.IMREAD_COLOR).astype(np.float32)
        else:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)


        self.image_cache[filename] = img
        self.image_cache_keys.append(filename)

        if len(self.image_cache_keys)>self.image_cache_size:
            old_image = self.image_cache_keys.pop(0)
            del self.image_cache[old_image]

        return img


    def load_flow_img(self, frame_id):

        if frame_id >= len(self.frames_flow_x):
            return None, None

        img_x = self.get_image_from_cache(self.frames_flow_x[frame_id], color=False)
        img_y = self.get_image_from_cache(self.frames_flow_y[frame_id], color=False)

        # img_x = cv2.imread(self.frames_flow_x[frame_id], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # img_y = cv2.imread(self.frames_flow_y[frame_id], cv2.IMREAD_GRAYSCALE).astype(np.float32)

        return img_x, img_y


    def load_rgb_img(self, frame_id):

        if frame_id >= len(self.frames_rgb):
            return None

        img = self.get_image_from_cache(self.frames_rgb[frame_id], color=True)

        return img


    def load_fc6(self, frame_id):

        if frame_id >= len(self.frames_flow_fc6) or frame_id >= len(self.frames_rgb_fc6):
            return None

        fc6_flow = np.load(self.frames_flow_fc6[frame_id])
        fc6_rgb = np.load(self.frames_rgb_fc6[frame_id])

        # Flow must be first, this is how we trained our LSTM network
        return np.append(fc6_flow, fc6_rgb, axis=1);


    def get_data_root(self):
        if self.is_flow:
            return self.video_frames_flow_dir
        else:
            return self.video_frames_rgb_dir


    '''
        stride          How many frames to skip between LSTM sequences
        channels        For flow batch x AND y flow filed frames in the flow stack
                        For RGB there are three channels
        sampling        How many frames to skip between flow stacks
    '''
    # def reset_lstm_batch(self, flow, timesteps, streams, stride, channels=20, sampling=5, padded=False):
    #     self.video_id = 0
    #     self.T = timesteps
    #     self.N = streams
    #     self.stride = stride
    #     self.channels = channels
    #     self.sampling = sampling
    #     self.flow_padded = padded
    #     self.is_flow = flow

    def reset_lstm_batch(self, config):
        self.video_id = 0
        self.T = config.timesteps
        self.N = config.streams
        self.stride = config.stride
        self.channels = config.channels
        self.sampling = config.sampling
        self.flow_padded = config.padded
        self.is_flow = config.flow
        self.is_npy = config.npy


    #@profile
    def get_next_lstm_batch(self):

        # if len(self.frames_flow_x)-self.video_id < self.T*self.flow_sampling+self.flow_channels/2:
        #     print "Only frames left: ", (len(self.frames_flow_x)-self.video_id)
        #     return None, None, None

        if self.is_npy:
            self.crop_size = np.array([1, 2*4096], dtype=np.float32)
            self.channels = 1

        batch = np.zeros((self.T, self.N, self.channels, self.crop_size[0], self.crop_size[1] ), dtype=np.float32)
        cont = np.zeros((self.T, self.N, 1 ), dtype=np.float32)
        labels = np.zeros((self.T, self.N, 1 ), dtype=np.float32)

        eof = False
        fid=self.video_id

        # Build N streams
        for n in range(self.N):

            # Build a sequence of T timesteps
            for t in range(self.T):
                labels[t,n] = self.label

                if t==0:
                    # Set LSTM continuity to 0. This means a beginning of a new seqeneces
                    cont[t,n] = 0
                else:
                    cont[t,n] = 1

                # Build a data sample
                p = fid + t*self.sampling
                if self.is_flow:
                    #print 'fid=',fid,' n=',n,' t=',t
                    for c in range(self.channels/2):
                        img_x, img_y = self.load_flow_img(p+c)

                        if img_x is None:
                            # No more images left
                            eof = True
                            break
                        else:
                            # if batch is None:
                            #     batch = np.zeros((self.N, self.T, self.flow_channels, img_x.shape[0], img_x.shape[1] ), dtype=np.float32)
                            img_x = self.transform_image(img_x, self.flow_mean_image, flow=True, crop_size=self.crop_size, crop=4, mirror=False)
                            img_y = self.transform_image(img_y, self.flow_mean_image, flow=True, crop_size=self.crop_size, crop=4, mirror=False)
                            # batch[t,n,c*2][...] = img_x
                            # batch[t,n,c*2+1][...] = img_y
                            batch[t,n,c*2]  = img_x
                            batch[t,n,c*2+1] = img_y

                            # np.copyto(batch[t,n,c*2], img_x, casting='no')
                            # np.copyto(batch[t,n,c*2+1], img_y, casting='no')
                else:
                    if self.is_npy:
                        fc6_sample = self.load_fc6(p)
                        if fc6_sample is None:
                            # No more images left
                            eof = True
                            break
                        batch[t,n,0,0] = fc6_sample[7]

                    else:
                        # RGB
                        img = self.load_rgb_img(p)
                        if img is None:
                            eof = True
                            break

                        img = self.transform_image(img, self.rgb_mean_image, flow=True, crop_size=self.crop_size, crop=4, mirror=False)

                        # imgs are loaded with openCV in BGR so we don't have to swap the channels
                        batch[t,n,0] = img[::,::,0]
                        batch[t,n,1] = img[::,::,1]
                        batch[t,n,2] = img[::,::,2]


                if eof:
                    # No more images to fill the sequence
                    break

            # Skip some frames to start next sequence
            fid += self.stride*self.sampling

            if eof:
                if self.flow_padded:
                    # If the number of timesteps is less than 1/2 of the sequence don't bother with padding
                    # and terminated the batch
                    if t < self.T/2 and (n>0 or t < self.T/4):
                       # print "Terminating batch. samples left out",t
                        n=0
                        t=0
                        break

                    #print "Padding sequence t=",t,"  stream=",n
                    batch[t:self.T, n] = np.zeros((self.channels, self.crop_size[0], self.crop_size[1]), dtype=np.float32)
                    #np.copyto(batch[t:self.T, n], np.zeros((self.channels, self.crop_size[0], self.crop_size[1]), dtype=np.float32))
                    cont[t:self.T,n] = 0 # set continuity to zero
                    labels[t:self.T,n] = 101 # a dummy label
                else:
                    # If we are not padding discard the uncompleted sequence
                    # and possibly finish - if the n==0
                    t=0

                break



        self.video_id=fid


        if eof:
            if n == 0 and t==0:
                #print "End of file"
                return None

            if self.flow_padded:
                # Include the last padded sequence. Without padding we would discard the incomplete one
                n+=1

            # Trim the containers to the number of completed streams
            batch = batch[::,:n,::,::,::]
            cont = cont[::, :n]
            labels = labels[::, :n]

        #print "Built streams: ",batch.shape[1]

        streams_num = batch.shape[1]
        seq_len = streams_num*self.T
        # cont = np.ones((seq_len,), dtype=np.float32)
        # cont = cont[:, np.newaxis]
        # cont[::self.T] = 0

#        labels = np.array([self.label]*seq_len, dtype=np.float32)

        # Flatten from a row major representation
        crop_shape = ( self.crop_size[0], self.crop_size[1])
        batch = batch.reshape((seq_len, self.channels)+crop_shape )

        #cont = cont.reshape((streams_num, self.T, 1)).transpose(1,0,2).reshape((seq_len, 1))
        cont = cont.reshape((seq_len, 1))
        # labels = labels.reshape(streams_num, self.T, 1).transpose(1,0,2).reshape((seq_len, 1))
        labels = labels.reshape((seq_len, 1))


        db = DataBlob()
        db.batch_data = batch
        db.continuation_markers = cont
        db.labels = labels
        db.stride = self.stride
        db.sampling = self.sampling
        db.data_root_dir = self.get_data_root()

        return db


#-------------------
    #@profile
    def get_rgb_batch_sampled(self, augment=False, samples_num=25):

        augmentations = 10

        frames = len(self.frames_rgb)
        step = frames / samples_num

        batch_size = samples_num
        if augment:
            batch_size *= augmentations

        print "Batch size: ",batch_size

        channels = 3
        batch = np.zeros((batch_size, channels, self.crop_size[0], self.crop_size[1] ), dtype=np.float32)

        fid = 0
        b=0
        eof=False
        for i in range(samples_num):
            img = self.load_rgb_img(fid)
            if img is None:
                eof = True
                break

            # Create 10x augmentations
            if augment:
                for m in [True, False]:
                    for c in range(0,5):
                        # img_stack, mean_image, flow, crop_size, crop=4, mirror=False, transformer=None):
                        img = self.transform_image(img, self.rgb_mean_image, False, self.crop_size, crop=c, mirror=m)

                        # This assignment is faster than all channels at once
                        batch[b,0]=img[::,::,0]
                        batch[b,1]=img[::,::,1]
                        batch[b,2]=img[::,::,2]
                        b+=1
            else:
                # Only center crop
                img = self.transform_image(img, self.rgb_mean_image, False, self.crop_size, crop=4, mirror=False)
                batch[b,0]=img[::,::,0]
                batch[b,1]=img[::,::,1]
                batch[b,2]=img[::,::,2]
                b+=1

            fid += step

        self.video_id=fid

        if eof:
            print "ERROR Couldn't finish batch!!"

            if b == 0:
                print "End of file"
                return None

            # Trim the batch
            batch = batch[:b, ::, ::, ::]


        return batch



    def reset_rgb_batch(self, batch_size, stride):
        self.video_id = 0
        self.rgb_batch_size = batch_size
        self.rgb_stride = stride
        pass


    #@profile
    def get_next_rgb_batch(self, crop_id=2, mirror=False):

        augmentations = 10
        batch_src_images_num = self.rgb_batch_size / augmentations

        channels = 3
        batch = np.zeros((self.rgb_batch_size, channels, self.crop_size[0], self.crop_size[1] ), dtype=np.float32)

        fid = self.video_id
        b=0
        eof=False
        for i in range(batch_src_images_num):
            img = self.load_rgb_img(fid)
            if img is None:
                eof = True
                break

            # Create 10x augmentations
            for m in [True, False]:
                for c in range(0,5):
                    # img_stack, mean_image, flow, crop_size, crop=4, mirror=False, transformer=None):
                    img = self.transform_image(img, self.rgb_mean_image, False, self.crop_size, crop=c, mirror=m)

                    batch[b,0]=img[::,::,0]
                    batch[b,1]=img[::,::,1]
                    batch[b,2]=img[::,::,2]
                    b+=1

            fid += self.rgb_stride

        self.video_id=fid
        if eof:
            if b == 0:
                print "End of file"
                return None
            batch = batch[:b, ::, ::, ::]


        return batch

    def parse_video_list(self, filename):

        video_list = np.loadtxt(filename, str, delimiter=' ')
        if len(video_list) > 0 and len(video_list[0]) == 3 and self.dataset == self.DATASET_UCF101:
            # UCF101 has frame number and video label following the video directory
            result = [[f[0], f[2]] for f in video_list]
            return result

        result = [[f, -1] for f in video_list]
        return result


def test():

    vp = VideoProcessor()
    vp.set_default_dataset(vp.DATASET_UCF101)

    video_list = vp.parse_video_list(vp.train_split1_file)

    video_dir, frames_num, video_label = video_list[10]

    vp.load_video(video_dir, video_label)



    vp.reset_lstm_flow_batch(25,10,5)

    print "Testing flow lstm batch"
    # while True:
    #     batch = vp.get_next_lstm_flow_batch()
    #     if batch is None:
    #         break


    print "Testing rgb lstm batch"
    # vp.reset_lstm_rgb_batch(25,10,5)
    # while True:
    #     batch = vp.get_next_lstm_rgb_batch()
    #     if batch is None:
    #         break
    #

    print "Testing rgb batch"
    # vp.reset_rgb_batch(batch_size=150, stride=5)
    # while True:
    #     batch = vp.get_next_rgb_batch()
    #     if batch is None:
    #         break

    print "Testing sampled rgb batch"
    batch = vp.get_rgb_batch_sampled(augment=True, samples_num=25)
    batch = vp.get_rgb_batch_sampled(augment=True, samples_num=100)
    batch = vp.get_rgb_batch_sampled(augment=False, samples_num=50)


    print "Finished"






if __name__ == "__main__":
    print "This is a package."
    print "Running test"
    test()
