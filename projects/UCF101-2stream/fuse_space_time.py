import os
import io
from pcardext import cp
import sys
import random
from IPython.nbconvert.preprocessors.coalescestreams import coalesce_streams
import numpy as np
import matplotlib.pyplot as plt

class Detection:
  decoderName = ''
  className = ''
  trueClassId = -1
  detectedClassId = -1
  features = np.array([])

  def correctDetection(self):
      return self.trueClassId == self.detectedClassId

  def __str__(self):
      info = 'Failed '
      if self.correctDetection():
        info = 'OK '

      return info+self.decoderName+'  '+self.className+' gt='+str(self.trueClassId)+' det='+str(self.detectedClassId)


def parseLine(line):

    det = Detection()

    rgb_toks = line.split(' ')
    det.trueClassId = int(rgb_toks[0].strip())
    det.detectedClassId = int(rgb_toks[1].strip())
    det.className = rgb_toks[2].strip()
    rgb_features = np.asarray(rgb_toks)
    det.features = rgb_features[3:].astype(np.float)

    return det


def go():

    test_result_flow_filename = '/home/jiri/Lake/DAR/share/datasets/UCF-101/val-1-rnd-result-flow.txt'
    test_result_rgb_filename = '/home/jiri/Lake/DAR/share/datasets/UCF-101/val-1-rnd-result-rgb.txt'

    imagenet_labels_filename = '/home/jiri/Lake/DAR/share/datasets/UCF-101/labels-new.txt'

    try:
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter=' ')
    except:
        return

    # test_videos = loadTestVideos(test_data_filename)

    flow_file = open(test_result_flow_filename, 'rt')
    rgb_file = open(test_result_rgb_filename, 'rt')

    correct = 0
    correctRGB = 0
    correctFlow = 0
    total = 0

    while (True):

        rgb_line = rgb_file.readline().strip()
        flow_line = flow_file.readline().strip()
        if (len(flow_line) == 0 or len(rgb_line) == 0):
            break

        total+=1

        rgbDet = parseLine(rgb_line)
        rgbDet.decoderName = 'RGB'
        if rgbDet.correctDetection():
            correctRGB += 1

        flowDet = parseLine(flow_line)
        flowDet.decoderName = 'Flow'
        if flowDet.correctDetection():
            correctFlow += 1


        featureSum = (rgbDet.features + flowDet.features)
        resultAvg = featureSum/2.0
        classId = resultAvg.argmax()

        # print rgbDet
        # print flowDet
        # print 'Both ',classId

        if (classId == rgbDet.trueClassId):
            correct+=1

    correctP = 100.0*correct/float(total)
    print 'Correct: ',correctP

    correctRGBP = 100.0*correctRGB/float(total)
    print 'Correct RGB: ',correctRGBP

    correctFlowP = 100.0*correctFlow/float(total)
    print 'Correct Flow: ',correctFlowP


    return



if __name__ == "__main__":
    go()

