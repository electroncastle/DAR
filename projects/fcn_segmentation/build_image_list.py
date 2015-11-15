import caffe
import lmdb
from PIL import Image

path = ''

for subdir, dirs, files in os.walk(path):
    dirs = subdir.strip().split(os.sep)
    last_subdir = dirs[len(dirs)-1]