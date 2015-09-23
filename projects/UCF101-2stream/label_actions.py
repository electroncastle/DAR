import os
import sys

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

    def update_path(self):
        # Common
        self.video_path = os.path.join(self.datasets_root, self.dataset_name, self.dataset_video, self.video_name)+'.'+self.video_ext
        self.rgbflow_path = os.path.join(self.datasets_root, self.dataset_name, self.dataset_rgbflow, self.class_name, self.video_name)

        try:
            self.labels = np.loadtxt(self.labels_file, str, delimiter=' ')
        except:
            print 'Cannot load labels: ',self.labels_file

        try:
            labels_annotation = np.loadtxt(self.labels_file_annotation, str, delimiter=' ')

            self.labels_annotation = ['Background']*len(self.labels)
            self.labels_annotation_ids = []

            for l in labels_annotation:
                id = int(l[0])-1
                self.labels_annotation_ids.append(id)
                self.labels_annotation[id] = l[1]


        except:
            print 'Cannot load labels: ',self.labels_file_annotation

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

    def process(self, temporal_data, spatial_data, file):
        max_frames = min(len(temporal_data), len(spatial_data))
        print max_frames

        frames = temporal_data[:max_frames, 0].astype(int)

        temp_w = 0.8
        classes = (temporal_data[:max_frames, 1:]*temp_w + spatial_data[:max_frames, 1:]*(1.0-temp_w))
        # classes = temporal_data[:max_frames, 1:]
        print classes.shape

        step = frames[1]-frames[0]
        fps = 30.0
        wnd_size_sec = 0.3
        wnd_size = int(wnd_size_sec*fps/step)+1


        classes_flt = []
        for i in range(len(classes)-wnd_size):
            b = classes[i:i+wnd_size, :]
            b = np.sum(b, 0) / b.shape[0]
            class_id = b.argmax()
            class_val = b[class_id]

            if (class_val < 0.4):
                continue

            time = round(i*step/fps, 3)
            classes_flt.append([time, class_id, class_val, self.labels[class_id][1] ])

            if 'Shotput' == self.labels[class_id][1]:
                print time, class_id, class_val, self.labels[class_id][1]

        # Close gaps
        gap_size_sec = 0.3#sec
        gap_size = int(gap_size_sec*fps/step)
        classes_final = []
        last_class_id = -1
        for i in range(len(classes_flt)-gap_size):
            cc = classes_flt[i]
            print cc

            if cc[1] == classes_flt[i+gap_size][1]:
                for g in range(gap_size):
                    classes_flt[i+g] = cc


        # Build a list of actions with the time window
        action_list = []
        current_action = []

        for i in range(len(classes_flt)):

            cc = classes_flt[i]

            if current_action == []:
                # Create new record
                current_action = [file, cc[0], cc[0], cc[1], 0, self.labels[cc[1]][1] ]
            else:
                if current_action[3] != cc[1]:

                    # Finish current action
                    current_action[2] = cc[0] # set the END time
                    action_list.append(current_action)

                    # Create new action
                    current_action = [file, cc[0], cc[0], cc[1], 0, self.labels[cc[1]][1] ]


        # Clean up. Set the foreign classes as background
        action_list_clean = []
        for a in action_list:
            if a[3] in self.labels_annotation_ids:
                action_list_clean.append(a)
                if 'Shotput' == a[5]:
                    print a


        return

    def record(self, file, classes, file_out):
        # [video_name] [starting_time] [ending_time] [class_label] [confidence_score]

        return


    def evaluate(self):

        return

    def annotate(self):

        val_list_filename = dar_root+'/share/datasets/THUMOS2015/thumos15_validation/annotated.txt'
        val_list = np.loadtxt(val_list_filename, str, delimiter=' ')

        detections_filename = dar_root+'/share/datasets/THUMOS2015/detections.txt'
        file_out = open(detections_filename, 'wt')

        for i in range(len(val_list)):
            file = val_list[i]

            [file, ext] = os.path.splitext(file)
            print 'Processing:  ',file

            video_path = os.path.join(self.datasets_root, self.dataset_name, self.dataset_rgbflow, file)

            temporal_data = np.loadtxt(video_path+'/temporal.txt', float, delimiter=' ')
            spatial_data = np.loadtxt(video_path+'/spatial.txt', float, delimiter=' ')

            classes = self.process(temporal_data, spatial_data, file)
            self.record(file, classes, file_out)

            break

        self.evaluate()

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


