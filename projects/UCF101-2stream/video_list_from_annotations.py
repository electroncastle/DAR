import os
import sys
import numpy as np


if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set, exiting'
    sys.exit()

dar_root = os.environ['DAR_ROOT']
app_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

if __name__ == "__main__":


    annotations_path = '/home/jiri/Lake/DAR/share/datasets/THUMOS2015/TH15_Temporal_annotations_validation/annotations/'

    video_list_full_path = '/home/jiri/Lake/DAR/share/datasets/THUMOS2015/thumos15_validation/annotated.txt'

    videos = {}
    for subdir, dirs, files in os.walk(annotations_path):
        print subdir
        for file in files:
            full_path = os.path.join(subdir, file)
            # print os.path.join(subdir, file)
            records = np.loadtxt(full_path, str, delimiter=' ')
            for rec in records:
                videos[rec[0]] = 1

        break


    fout = open(video_list_full_path, 'wt')
    for key in videos.viewkeys():
        print key
        fout.write(key+'.mp4\n')

    fout.close()



