import os
import io
from pcardext import cp
import sys
import random
from IPython.nbconvert.preprocessors.coalescestreams import coalesce_streams
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Jiri Fajtl"
__email__ = "ok1zjf@gmail.com"
__status__ = "Research"
__license__ = "LGPL"
__date__ = "20/10/2015"
__version__ = "1.0.0"


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


def parseLineList(line_list):

    det = Detection()
    det.trueClassId = int(line_list[0].strip())
    det.detectedClassId = int(line_list[1].strip())
    det.className = line_list[2].strip()
    rgb_features = np.asarray(line_list)
    det.features = rgb_features[3:].astype(np.float)

    return det


def test(rgb_data, flow_data, temp_weight):

    correct = 0
    correctRGB = 0
    correctFlow = 0
    total = 0

    for i in range(len(rgb_data)):

        # if i>643:
        #     break

        rgb_line = rgb_data[i]
        if len(flow_data) <= i:
            break

        flow_line = flow_data[i]

        total += 1

        rgbDet = parseLineList(rgb_line)
        rgbDet.decoderName = 'RGB'
        if rgbDet.correctDetection():
            correctRGB += 1

        flowDet = parseLineList(flow_line)
        flowDet.decoderName = 'Flow'
        if flowDet.correctDetection():
            correctFlow += 1

        # temp_weight = 0.7
        featureSum = rgbDet.features*(1.0-temp_weight) + flowDet.features*temp_weight
        resultAvg = featureSum
        classId = resultAvg.argmax()

        # print rgbDet
        # print flowDet
        # print 'Both ',classId

        if (classId == rgbDet.trueClassId):
            correct+=1

    # print "Weighted avarage temp/spatial: ",temp_weight,"/",(1.0-temp_weight)
    correctP = 100.0*correct/float(total)
    # print 'Combined: ',correctP

    correctRGBP = 100.0*correctRGB/float(total)
    # print 'Correct RGB: ',correctRGBP

    correctFlowP = 100.0*correctFlow/float(total)
    # print 'Correct Flow: ',correctFlowP

    return [correctRGBP, correctFlowP, correctP]


def go():

    test_result_flow_filename = '/home/jiri/Lake/DAR/share/datasets/UCF-101/val-1-rnd-result-flow.txt'
#    test_result_rgb_filename = '/home/jiri/Lake/DAR/share/datasets/UCF-101/val-1-rnd-result-rgb.txt'
    test_result_rgb_filename = '/home/jiri/Lake/DAR/share/datasets/UCF-101/val-1-rnd-result-rgb-789.txt'

    imagenet_labels_filename = '/home/jiri/Lake/DAR/share/datasets/UCF-101/labels-new.txt'

    try:
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter=' ')
    except:
        return

    # test_videos = loadTestVideos(test_data_filename)

    # flow_file = open(test_result_flow_filename, 'rt')
    # rgb_file = open(test_result_rgb_filename, 'rt')

    flow_data = np.loadtxt(test_result_flow_filename, str, delimiter=' ')
    rgb_data = np.loadtxt(test_result_rgb_filename, str, delimiter=' ')

    maximize_weight = False
    weight = 0.4
    weight_step = 0.005
    last_step = 0.00
    last_gain = -1.0

    max_weight = 0.0
    max_gain = -1.0
    dir = 1.0

    random.seed()
    if maximize_weight:

        while True:
            correctRGBP, correctFlowP, gain = test(rgb_data, flow_data, weight)

            if gain > max_gain:
                max_weight = weight
                max_gain = gain

            print "w=",weight, "  gain=", gain, "   max gain=",max_gain, "  max_weight=",max_weight

            weight += weight_step

            if weight > 1.0:
                break
    else:
        weight = 0.715
        weight = 0.713
        weight = 0.6666666
        correctRGBP, correctFlowP, correctP = test(rgb_data, flow_data, weight)
    #
    # print 'UCF101 split 1 validation dataset video num: 3783'
    # print 'Video num: ', total

    # print "Weighted avarage temp/spatial: ",temp_weight,"/",(1.0-temp_weight)
    # correctP = 100.0*correct/float(total)
    print 'Combined: ',correctP

    # correctRGBP = 100.0*correctRGB/float(total)
    print 'Correct RGB: ',correctRGBP

    # correctFlowP = 100.0*correctFlow/float(total)
    print 'Correct Flow: ',correctFlowP


    return



if __name__ == "__main__":
    go()

