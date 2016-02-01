#!/usr/bin/python

__author__ = "Jiri Fajtl"
__email__ = "ok1zjf@gmail.com"
__status__ = "Research"
__license__ = "LGPL"
__date__ = "20/10/2015"
__version__ = "1.0.0"


import os
import sys
import math
import numpy as np
import fileinput
import glob
import time
import h5py
import cv2
import lmdb
import gc
import argparse
import random

# sys.path.insert(0, '/home/jiri/Lake/DAR/src/ofEval/build-debug/')
# import ofEval_module as em
#
# sys.path.insert(0, '/home/jiri/Lake/DAR/projects/optical_flow_regression')
# import utils

caffe_root = os.environ['DAR_ROOT']
sys.path.insert(0, caffe_root + '/src/caffe/python')
import caffe
import caffe.io
#from caffe.proto import caffe_pb2

'''
The LSTM batch:
- Should be as long are sequences (T) !!
- includes number of sequences (N) with different labels !!


= data
 Size: TxNxD
    T - number of timesteps
    N - number o sequences
    D - Sequence data item for time unit

= cont_label
 Size:  TxN
 For each sequence (0..N) -> Begining of sequence has item 0, while continuation 1

= target_label
 Size:  TxN
    Label number for each time stamp within each sequence

= input_label
 Size:  TxN
    Label number for each time stamp within each sequence


'''

class sequenceWriter:

    def __init__(self):
        self.N = 10
        self.T = 25
        self.stride = 1
        self.padded = False
        self.sampling = 1

        self.seqs = []
        self.video_id = 0
        self.video_list = 0
        self.last_seq = 0

        # Frame type constants
        self.FLOW_FC6 = 0
        self.RGB_JPG = 1

        self.frame_type = self.RGB_JPG
        self.ignored_videos = 0
        self.trimmed_frames = 0

        return


    def get_frames(self, video):

        class_id = int(video[2])
        if self.frame_type == self.FLOW_FC6:

            video_frames_flow_dir = video[0]
            if video_frames_flow_dir[0] == '/':
                path, video_dir = os.path.split(video_frames_flow_dir)
                path, video_class = os.path.split(path)
                flow_path  =  '/home/jiri/Lake/DAR/share/datasets/UCF-101/UCF101-flow/'
                video_frames_flow_dir = os.path.join(flow_path, video_class, video_dir)

            tmpl = '%s/fc6_????.npy' %(video_frames_flow_dir)
            spatial = glob.glob(tmpl)
            spatial.sort()

            # Check the rgb fc6 feature vectors
            rgb_path  =  '/home/jiri/Lake/DAR/share/datasets/UCF-101/UCF101-rgbflow/'
            video_frames_rgb_dir = os.path.join(rgb_path, video_class, video_dir)
            tmpl = '%s/fc6_rgb_????.npy' %(video_frames_rgb_dir)
            rgb = glob.glob(tmpl)
            rgb.sort()

            min_len = min(len(rgb), len(spatial))
            max_len = max(len(rgb), len(spatial))
            self.trimmed_frames += max_len-min_len

            # if min_len != max_len:
            #     print
            #     if len(rgb) < len(spatial):
            #         print "RGB frames missing:   ", (max_len-min_len), "   ",video_dir,
            #     else:
            #         print "Flow frames missing:   ", (max_len-min_len), "   ",video_dir,

            rgb = rgb[:min_len]
            spatial = spatial[:min_len]


        elif self.frame_type == self.RGB_JPG:
            tmpl = '%s/image_????.jpg' %(video[0])

            spatial = glob.glob(tmpl)
            spatial.sort()

        return [class_id, 0, spatial]


    def ensure_video_frame_cache(self):

        # Load video frames to cache
        while len(self.seqs) < self.N*3:

            if self.video_id >= len(self.video_list):
                break

            video = self.video_list[self.video_id]

            path, filename = os.path.split(video[0])
            data = self.get_frames(video)
            #print '[',self.video_id,'] Loading ',filename,'  (', len(data[2]) ,')  '
            self.video_id += 1

            # Check if the video has at least T number of frames to build one sequence
            if len(data[2]) < self.T:
                print "WARNING: Ignoring video ",filename,"  Not enough frames! frames: ",len(data[2]),"   sequence lenght: ",self.T
                self.ignored_videos += 1
                continue

            self.seqs.append(data)

        return len(self.seqs) > 0


    #@profile
    def get_next_sequence(self):

        if not self.ensure_video_frame_cache():
            return [], [], [], []

        # Increment the sequence counter
        self.last_seq += 1
        if self.last_seq >= len(self.seqs):
            self.last_seq = 0

        # Interleave frames in a batch by selecting next video from the cache
        frames = self.seqs[self.last_seq]

        # Build sequence self.T frames long
        label = frames[0] # label for the currently selected video
        frs = frames[2]
        frs_count = len(frs)

        if frs_count-frames[1] <= self.T:
            frames[1] = frs_count - self.T # leave room for at least one frame for optical flow
            self.seqs.pop(self.last_seq)

        pos = frames[1]
        seq = np.array(frs[pos:pos+self.T])

        # Create a list with image pairs for the optical flow calculation
        # pos += 1
        # seq_flow = np.array(frs[pos:pos+self.T])

        frames[1] += self.stride

        # If there are not enough frames for next sequence remove the video
        # if (frames[1] >= frs_count):
        #     self.seqs.pop(self.last_seq)

        cont = np.ones((self.T), dtype=np.uint8)
        cont[0] = 0 # start of sequence

        labels = np.array([label]*self.T)

        seq_flow = []
        return seq, cont, labels, seq_flow



    def build_list_fast(self, video_list, file_out, with_flow=False):
        self.video_list = video_list

        print "Assembling LSTM sequences for videos:"
        sequences = []
        video_counter=0
        ignored = 0
        for video in self.video_list:
            path, filename = os.path.split(video[0])
            label, tmp, frames = self.get_frames(video)
            #print video_counter,"/",len(self.video_list) ," Processing video ",filename, "  frames: ",len(frames)," ",
            video_counter+=1

            if video_counter % 10 == 0:
                sys.stdout.write("\r%d / %d " % (video_counter, len(self.video_list)) )
                sys.stdout.flush()

            fc = 0
            sec = 0
            while True:


                if self.padded:
                    if fc+self.T/2 >= len(frames):
                        if fc == 0:
                            #print ignored," IGNORED too few frames"
                            ignored +=1
                        else:
                            #print "Finished video. Sequences: ",sec
                            pass
                        break

                else:
                    if fc+self.T >= len(frames):
                        if fc == 0:
                            #print ignored," IGNORED too few frames"
                            ignored +=1
                        else:
                            #print "Finished video. Sequences: ",sec
                            pass

                        break

                c = 0
                seq = []
                frame_id = fc
                for s in range(self.T):
                    l = label
                    if frame_id >= len(frames):
                        if self.padded:
                            frame = '-' # dummy filename, npydatalayer in caffe will generate zero blob instead
                            c = 0
                            l = '101'   # dummy label
                        else:
                            break
                    else:
                        frame = frames[frame_id]

                    rec = frame+' '+str(c)+' '+str(l)+'\n'
                    seq.append(rec)
                    c=1

                    frame_id+=self.sampling

                if len(seq) == self.T:
                    sec += 1
                    sequences.append(seq)

                fc += self.stride*self.sampling

        print
        print "Total sequences: ",len(sequences)
        print "Ignord videos (due to low number of frames) ",ignored

        print "Shuffling..."
        random.shuffle(sequences)

        print "Writing out..."
        fout = open(file_out, 'w')

        p=0
        while p+self.N<len(sequences):

            # Interleave sequences
            for t in range(self.T):
                for n in range(self.N):
                    f = sequences[p+n][t]
                    # print f
                    fout.write(f)

                    if p % 1000 == 0:
                        sys.stdout.write("\r%d / %d " % (p, len(sequences)))
                        sys.stdout.flush()


            p+=self.N

        fout.close()

        print
        print "Done ",(len(sequences)-p)," sequences left out out of ",len(sequences)
        print "Frames trimmed due to misalignment ",self.trimmed_frames

        return


    def build_list(self, video_list, file_out, with_flow=False):

        self.video_list = video_list
        self.seqs = []
        self.video_id = 0
        self.last_seq = 0
        max_seq = 0

        fout = open(file_out, 'w')

        # flow_file_out, ext = os.path.splitext(file_out)
        # flow_file_out += "-flow"+ext
        # fout_flow = open(flow_file_out, 'w')
        count = 0
        while True:

            start = time.time()
            seqs=[]
            seqs_flow = []
            conts=[]
            labels=[]
            for n in range(self.N):
                seq, cont, label, seq_flow = self.get_next_sequence()
                if seq == []:
                    print "Finshed. No more video frames"
                    # Discard the last not completed batch since the the image files list neeeds to roll over
                    # while maintaining batch alignment
                    fout.close()
                    return

                # Zero first X labels in the sequence. The LSTM needs to first learn the seqeunce
                # thus may not be sure about the action at the begining
                #label[:2]=0

                if seqs == []:
                    seqs = seq.copy()
#                    seqs_flow = seq_flow.copy()
                    conts = cont.copy()
                    labels = label.copy()
                else:
                    seqs = np.append(seqs, seq)
#                    seqs_flow = np.append(seqs_flow, seq_flow)
                    conts = np.append(conts, cont)
                    labels = np.append(labels, label)

            # Reshape to row-major format that is
            # reshape(T,N) must results in each column being a sequence of consecutive video frames
            # of the same video

            conts = conts.reshape((self.N, self.T)).transpose(1,0).reshape((self.N*self.T))
            seqs = seqs.reshape((self.N, self.T)).transpose(1,0).reshape((self.N*self.T))
            labels = labels.reshape((self.N, self.T)).transpose(1,0).reshape((self.N*self.T))
#            seqs_flow = seqs_flow.reshape((self.N, self.T)).transpose(1,0).reshape((self.N*self.T))

            # Write out
            for i in range(len(seqs)):

                if with_flow:
                    record = str(seqs[i])+' '+str(seqs_flow[i])+' '+str(conts[i])+' '+str(labels[i])+'\n'
                else:
                    record = str(seqs[i])+' '+str(conts[i])+' '+str(labels[i])+'\n'

                fout.write(record)

                # record = str(seqs_flow[i])+'\n'
                # fout_flow.write(record)

            end = time.time()
            count+=1

            if max_seq>0 and count > max_seq:
                break

        fout.close()
        # fout_flow.close()

        return


    def build_npy_list(self, video_list, file_out):



        pass

    def set_shape(self, shape):

        t,n,s = shape.split(',')
        self.T = int(t)
        self.N = int(n)
        self.stride = int(s)

        print "New shape set: T=",self.T," N=",self.N," stride=",self.stride," padding=",self.padded," sampling=",self.sampling


        return

def load_data_list(filename):
    data = []
    for line in fileinput.input(filename):
        data.append(line.strip().split(' '))

    return data


debug = False


if __name__ == "__main__":
    sw = sequenceWriter()

    p = argparse.ArgumentParser(description="parse some things.")

    p.add_argument("cmd", help=argparse.SUPPRESS, nargs="*")

    p.add_argument("-t","--train", action="store_true", help="some help", default=False)
    p.add_argument("-v","--val", action="store_true", help="some help")
    p.add_argument("-n","--npy", action="store_true", help="some help", default=False)
    p.add_argument("-i","--img", action="store_true", help="some help", default=False)
    p.add_argument("-f","--flow", action="store_true", help="some help", default=False)
    p.add_argument("-l","--lstm", action="store_true", help="some help")
    # p.add_argument("-s","--shape", type=str, help="some help", default=str(sw.T)+","+str(sw.N) )
    p.add_argument("-s","--shape", type=str, help="some help", default="18,100,9" )
    p.add_argument("-p","--pad",  action="store_true", help="some help", default=False )
    p.add_argument("-a","--sampling", type=int, help="some help", default=1 )


    args = p.parse_args()
    print args


    path = '/home/jiri/Lake/DAR/share/datasets/UCF-101/'
    train_file = path+'train-2-rnd.txt'
    val_file =  path+'val-2-rnd.txt'

    train_list = load_data_list(train_file)
    val_list = load_data_list(val_file)

    with_flow=args.flow

    sw.sampling = args.sampling
    sw.padded = args.pad
    sw.set_shape(args.shape)

    if args.img:

        out_filename =''
        if args.train:
            out_filename = 'train'
        elif args.val:
            out_filename = 'val'

        out_filename += '_lstm_'+str(sw.T)+'-'+str(sw.N)
        if args.flow:
            out_filename += '-flow'
        out_filename += '.txt'

        if args.train:
            print "*************************************************************"
            print "***************** Building training image list **************"
            print "*************************************************************"
            sw.build_list_fast(train_list, out_filename, with_flow)

        if args.val:
            print "*************************************************************"
            print "***************** Building validation image list ************"
            print "*************************************************************"
            sw.build_list_fast(val_list, out_filename, with_flow)

    if args.npy:
        sw.frame_type = sw.FLOW_FC6
        if args.train:
            print "*************************************************************"
            print "***************** Building training npy list ****************"
            print "*************************************************************"
            sw.build_list_fast(train_list, 'train_flow_fc6_lstm_'+str(sw.T)+'-'+str(sw.N)+'.txt')
            pass

        if args.val:
            print "*************************************************************"
            print "***************** Building validation npy list **************"
            print "*************************************************************"
            sw.build_list_fast(val_list, 'val_flow_fc6_lstm_'+str(sw.T)+'-'+str(sw.N)+'.txt')
            pass


    print "Ignored videos: ", sw.ignored_videos

