import os
import sys
import io

import numpy as np
import matplotlib.pyplot as plt


if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set, exiting'
    sys.exit()

dar_root = os.environ['DAR_ROOT']
app_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

sys.path.insert(0, dar_root + '/python')
# import caffe
# import cv2

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)




def loadTestVideos(filename):

    dset = open(filename, 'rt')
    print filename

    videos = []
    for line in dset:
        toks = line.split(' ')
        videos.append([toks[2].strip(), toks[0].strip()])

    return videos


#
# Record format:
# [video_name] [starting_time] [ending_time] [class_label] [confidence_score]
#
class VideoActionsLabeler(object):

    def __init__(self):
        self.name = "Action Labeler"
        self.labels_file = ''
        self.labels = []
        self.annotations_gt = {}

    def update_path(self):
        # Common
        self.video_path = os.path.join(self.datasets_root, self.dataset_name, self.dataset_video, self.video_name)+'.'+self.video_ext
        self.rgbflow_path = os.path.join(self.datasets_root, self.dataset_name, self.dataset_rgbflow, self.class_name, self.video_name)

        try:
            self.labels = np.loadtxt(self.labels_file, str, delimiter=' ')
            self.labels = np.vstack([self.labels, ['101', 'BACKGROUND']])
        except:
            print 'Cannot load labels: ',self.labels_file


        try:
            labels_annotation = np.loadtxt(self.labels_file_annotation, str, delimiter=' ')

            self.labels_annotation = ['Background']*len(self.labels)
            self.labels_annotation_ids = []

            for l in labels_annotation:
                id = int(l[0])
                self.labels_annotation_ids.append(id)
                self.labels_annotation[id] = l[1]


        except:
            print 'Cannot load labels: ',self.labels_file_annotation


        # load ground truth
        self.load_ground_truth(self.annotations_path)

        return

    def set_video_name(self, video_name):
        self.video_name = video_name
        self.update_path()


    def set_path(self):

        self.datasets_root = os.path.join(dar_root, 'share/datasets/')

        if 0:
            # UCF 101
            self.dataset_name = 'UCF-101'
            self.dataset_video = 'UCF101'
            self.dataset_rgbflow = 'UCF101-rgbflow/'

            self.video_name = 'v_FloorGymnastics_g16_c04'
            self.video_name = 'v_JavelinThrow_g02_c01'

            self.video_ext = 'avi'
            self.class_name = 'JavelinThrow'

            self.labels_file = os.path.join(self.datasets_root, self.dataset_name, 'labels-new.txt')
        else:
            # THUMOS validation
            self.dataset_name = 'THUMOS2015'
            self.dataset_video = 'thumos15_validation'
            self.dataset_rgbflow = 'thumos15_validation-rgbflow'

            self.video_name = 'thumos15_video_validation_0000006'
            self.video_ext = 'mp4'
            self.class_name = ''

            self.annotations_path = os.path.join(self.datasets_root, self.dataset_name, 'TH15_Temporal_annotations_validation/annotations/')
            self.labels_file = os.path.join(self.datasets_root, 'UCF-101', 'labels-new.txt')

            self.labels_file_annotation = os.path.join(self.datasets_root, self.dataset_name, 'TH15_Temporal_annotations_validation/labels.txt')

        self.update_path()

        # If the rgbflow dir has subdirectories with the classnames
        # e.g. 'UCF101-rgbflow/FloorGymnastics/v_FloorGymnastics_g16_c04'
        #self.has_class_dirs = False


    def showResult(self, classes):

        if  len(classes.shape) > 1:
            classResultAvg = np.sum(classes)/classes.shape[0]
        else:
            classResultAvg = classes

        result = classResultAvg.argmax()
        # print("Predicted class is #{}.".format(result))

        try:
            labels = np.loadtxt(self.labels_file, str, delimiter=' ')
        except:
            print 'Cannot load: ',self.labels_file
            return

        print labels[result], ' ', classResultAvg[result]


    def save_result(self, classes, filename):

        fout = open(filename, 'wt')
        for i in range(classes.shape[0]):

            record = str(i*self.stride)+' '+' '.join(['%.6f' % num for num in classes[i]])
            fout.write(record+'\n')
            fout.flush()

        fout.close()


    def get_class_name(id):

        pass








    def merge_action_list_x(self, file_name, classes_flt):

        classes = classes_flt

        # Build a list of actions with the time window
        action_list = []
        current_action = []
        action_len = 0;
        for i in range(len(classes)):

            cc = classes[i]
            if cc[1] not in self.labels_annotation_ids:
                continue

            # Remove short action and action with low confidence
            if cc[2] < 0.3:
                continue

            if current_action == []:
                # Create new record
                current_action = [file_name, cc[0], cc[0], cc[1], cc[2], self.labels[cc[1]-1][1] ]
                action_len = 1
            else:
                if current_action[3] != cc[1]:

                    # Finish current action
                    current_action[2] = cc[0] # set the END time
                    current_action[4] = current_action[4]/float(action_len)
                    action_list.append(current_action)

                    # Create new action
                    current_action = [file_name, cc[0], cc[0], cc[1], cc[2], self.labels[cc[1]][1]]
                    action_len = 1
                else:
                    current_action[4] += cc[2]
                    action_len += 1


        # Clean up. Set the foreign classes as background
        action_list_clean = []
        for a in action_list:

            # Remove background
            if a[3] not in self.labels_annotation_ids:
                continue

            if 0:
                # Remove short action and action with low confidence
                if a[2]-a[1] < 0.35 or a[4] < 0.3:
                    continue

            action_list_clean.append(a)
            # if 'Shotput' == a[5]:
            #     print a
            print a

        return action_list_clean


    def process(self, temporal_data, spatial_data, video_filename):
        temp_len=len(temporal_data)
        spatial_len=len(spatial_data)
        if (spatial_len != temp_len):
            print "[WARNING] Temporal and spatial streams have different number of frames! spatial=",spatial_len," temporal=",temp_len

        max_frames = min(temp_len, spatial_len)
        print max_frames

        frames = temporal_data[:max_frames, 0].astype(int)

        temp_w = 0.35   # Continuous Best 0.35
        temp_w = 0.5   # Overlapping Best 0.5
        classes = (temporal_data[:max_frames, 1:]*temp_w + spatial_data[:max_frames, 1:]*(1.0-temp_w))
        # classes = temporal_data[:max_frames, 1:]
        print classes.shape

        step = frames[1]-frames[0]
        fps = 30.0

        wnd_size_sec = 0.3  # continuous best: 0.3
        wnd_size_sec = 0.4  # overlapping best: 0.4
        wnd_size = int(wnd_size_sec*fps/step)+1

        classes_flt = []
        for i in range(len(classes)-wnd_size):
            b = classes[i:i+wnd_size, :]
            b = np.sum(b, 0) / b.shape[0]
            class_id = b.argmax()
            class_val = b[class_id]
            class_id += 1 # The Thumos labels are indexed 1..N while our 0..N-1

            # Detection with confidence lower than this will be labeled as
            # background
            if (class_val < 0.3):
                class_id = 102

            # Set all unknown classes to background
            if class_id not in self.labels_annotation_ids:
                class_id = 102

            time = round(i*step/fps, 3)
            classes_flt.append([time, class_id, class_val, self.labels[class_id-1][1] ])

            # if 'Shotput' == self.labels[class_id-1][1]:
            #     print time, class_id, class_val, self.labels[class_id-1][1]


            # Remove noisy detections such as:
            # - remove class B if is positioned as AABAA
        if 1:
            for i in range(len(classes_flt)-2):
                cc = classes_flt[i]
                # print cc

                if classes_flt[i-1][1] == classes_flt[i+1][1]: #and \
                    # classes_flt[i-2][1] == classes_flt[i+2][1] and \
                    # classes_flt[i-1][1] == classes_flt[i-2][1]:

                    if cc[1] != classes_flt[i-1][1]:
                        cc[1] = classes_flt[i-1][1]
                        cc[3] = classes_flt[i-1][3]

                        wnd = np.asarray(classes_flt[i-2:i]+classes_flt[i+1:i+3])
                        # wnd = np.asarray(classes_flt[i-1:i]+classes_flt[i+1:i+2])
                        avg = wnd[:,2].astype(float)
                        avg = sum(avg)/4.0
                        # avg = sum(avg)/2.0

                        cc[2] = avg

        # Close gaps
        if 0:
            gap_size_sec = 1.3  #sec
            gap_size = int(gap_size_sec*fps/step)
            classes_final = []
            last_class_id = -1
            for i in range(len(classes_flt)-gap_size):
                cc = classes_flt[i]
                print cc

                if cc[1] == classes_flt[i+gap_size][1]:
                    for g in range(gap_size):
                        classes_flt[i+g] = cc

        return classes_flt


    def build_action_list(self, file_name, classes):

        # Build a list of actions with the time window
        action_list = []
        for i in range(len(classes)):
            cc = classes[i]

            # Create new record
            current_action = [file_name, cc[0], cc[0], cc[1], cc[2], self.labels[cc[1]-1][1]]
            action_list.append(current_action)

        return action_list


    def merge_action_list(self, action_list):

        # Build a list of actions with the time window
        action_list_new = []
        current_action = []
        action_len = 0;
        for i in range(len(action_list)):

            cc = action_list[i]
            # print cc

            if current_action == []:
                # Create new record
                current_action = cc #[file_name, cc[0], cc[0], cc[1], cc[2], self.labels[cc[1]][1] ]
                action_len = 1
            else:
                if current_action[3] != cc[3]:

                    # Finish current action
                    current_action[2] = cc[2] # set the END time
                    current_action[4] = current_action[4]/float(action_len)
                    action_list_new.append(current_action)

                    # Create new action
                    cc[1] = current_action[2]
                    current_action = cc
                    action_len = 1
                else:
                    current_action[4] += cc[4]
                    action_len += 1

        return action_list_new


    def clean_action_list(self, action_list):
        # Clean up. Set the foreign classes as background
        action_list_clean = []
        for a in action_list:

            # Remove background
            if a[3] not in self.labels_annotation_ids:
                continue

            # Remove short action and action with low confidence
            # if a[2]-a[1] < 0.4 or a[4] < 0.4:
            #     continue

            action_list_clean.append(a)
            # if 'Shotput' == a[5]:
            #     print a
            #print a

        return action_list_clean


    def record(self, actions, fout):
        # [video_name] [starting_time] [ending_time] [class_label] [confidence_score]

        for action in actions:
            action_str = [str(a) for a in action]

            # Remove the last item in the action array, This is not compatible with the THUMOS2015 format
            rec = ' '.join(action_str[:-1])
            fout.write(rec+'\n')
        fout.flush()
        return



    def get_overlap(self,a, b):
        mii = min(a[1], b[1])
        mai = max(a[0], b[0])

        mau = max(a[1], b[1])
        miu = min(a[0], b[0])

        i = max(0, mii-mai) # Intersection
        u = max(0, mau-miu) # Union

        po = i/u    # Overlap in range 0..1

        return [i, u, po]


    def get_actions(self, actions, interval, gt_class_name, gti):

        act_in = []
        for ai in range(len(actions)):
            action = actions[ai]
            # if action[0] != file:
            #     print "ERROR Wrong file referenced ",action[0], "  should be ", file
            #     return

            # Test action
            #print action
            start = action[1]
            end = action[2]
            class_name = action[5]

            if gt_class_name != class_name:
                continue

            i, u, po = self.get_overlap([start, end], interval)
            if i > 0:
                # intersectin_rec = [i, u, po, ai]
                # act_in.append(intersectin_rec)

                # If this action already overlap with some ground truth check
                # whether it has larger overlap with this ground truth
                # If yes update it
                if action[6][2] < po:
                    action[6] = [i, u, po, gti]


        return act_in


    def evaluate(self, actions, file, fout_ap):

        # get GT for the file
        if file not in self.annotations_gt:
            print "ERROR There is no ground truth for ",file
            return

        for a in actions:
            a.append([0.0, 0.0, 0.0, 0])

        gt = self.annotations_gt[file]

        for gti in range(len(gt)):
            g = gt[gti]
            # get detected actions overlapping with this ground truth
            # Modifieds action[6] array
            # This array has has formant [intersection, union, overlap percentage, ground_truth_id]
            self.get_actions(actions, [g[0], g[1]], g[2], gti)

        # Find TP and FP
        tp = 0
        fp = 0
        for a in actions:
            print a
            if a[6][2] > 0.5:
                tp += 1
            else:
                fp += 1
            pass

        fn = len(gt)-tp
        rec = file+"  TP="+str(tp)+"   FP="+str(fp)+"   FN="+str(fn)+"\n"
        print rec

        fout_ap.write(rec)
        fout_ap.flush()

        return [tp, fp, fn]

    def load_ground_truth(self, gt_path):

        self.annotations_gt = {}
        # video_name => [actions]
        for filename in os.listdir(gt_path):
            [file, ext] = os.path.splitext(filename)
            if ext != '.txt':
                continue

            toks = file.split('_')
            if len(toks) < 2:
                continue

            class_name = toks[-1]

            print "Reading GT file: ",filename
            recs = np.loadtxt(gt_path+"/"+filename, str, delimiter=' ')
            for rec in recs:
                video_filename = rec[0]
                if video_filename not in self.annotations_gt:
                    self.annotations_gt[video_filename] = []

                self.annotations_gt[video_filename].append([float(rec[1]), float(rec[2]), class_name])


        return


    def open_append(self, filename):
        if os.path.isfile(filename):
            fout = open(filename, 'w+t')
        else:
            fout = open(filename, 'w')
        return fout


    def annotate(self):

        val_list_filename = dar_root+'/share/datasets/THUMOS2015/thumos15_validation/annotated.txt'
        val_list = np.loadtxt(val_list_filename, str, delimiter=' ')

        fout = self.open_append(dar_root+'/share/datasets/THUMOS2015/detections.txt')
        fout_ap = self.open_append(dar_root+'/share/datasets/THUMOS2015/detections-ap.txt')


        stat = np.array([0.0, 0.0, 0.0])
        count = 0
        for i in range(len(val_list)):
            file = val_list[i]

            [file, ext] = os.path.splitext(file)
            print 'Processing:  ',file
            # if file == "thumos15_video_validation_0001630":
            #      continue

            video_path = os.path.join(self.datasets_root, self.dataset_name, self.dataset_rgbflow, file)

            try:
                # temporal_data = np.loadtxt(video_path+'/temporal.txt', float, delimiter=' ')
                # spatial_data = np.loadtxt(video_path+'/spatial.txt', float, delimiter=' ')
                temporal_data = np.loadtxt(video_path+'/temporal-s5.txt', float, delimiter=' ')
                spatial_data = np.loadtxt(video_path+'/spatial-s5.txt', float, delimiter=' ')
            except:
                print "Cannot read temporal or spatial data. Finishing..."
                break
#            file_out = video_path+'/actions.txt'

            classes = self.process(temporal_data, spatial_data, file)
            # for c in classes:
            #     print c
            action_list = self.build_action_list(file, classes)

            action_list = self.merge_action_list(action_list)
            action_list = self.clean_action_list(action_list)
            # action_list = self.merge_action_list(action_list)
            # action_list = self.clean_action_list(action_list)
            # action_list = self.merge_action_list(action_list)

            self.record(action_list, fout)

            rec = self.evaluate(action_list, file, fout_ap)
            stat += rec
            count += 1
            pass

        fout.close()
        fout_ap.close()

        print stat/count
        #[  1.78881988  28.03726708  14.33540373]
        return



if __name__ == "__main__":

    # thread1 = myThread(1, "XX", 1)
    # thread2 = myThread(2, "OO", 2)
    #
    # # Start new Threads
    # thread1.start()
    # thread2.start()
    #
    # thread2.join()

    #go(0, True)	# Temporal
    #go(1, False) # Spatial

    video = VideoActionsLabeler()
    video.set_path()
    video.annotate()


