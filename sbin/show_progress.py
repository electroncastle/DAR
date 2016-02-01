#!/usr/bin/python

__author__ = "Jiri Fajtl"
__email__ = "ok1zjf@gmail.com"
__status__ = "Research"
__license__ = "LGPL"
__date__ = "20/10/2015"
__version__ = "1.0.0"


import sys
import fileinput
import matplotlib.pyplot as plt
import numpy as np
import time
from select import select
import math

sgn = 'Train net output #0: mse ='
sgn_len = len(sgn)


# Install drawnow
# pip install drawnow.
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np


inc=100

file = None
last_iter = 0
last_iter_base = 0
plots = []
plots_legend = []
clrs = ['b', 'r', 'g', 'y']

def makeFig():
    #plt.scatter(xList,yList)

    # plt.plot(xList,yList)
    # plt.ylim([0, float(sys.argv[1]) ])
    # plt.plot(xListt, yListt, 'r')


    legend = []
    clrs_ = []
    for i in range(len(plots)):
        if len(plots[i][0]) > 0:
            legend.append(plots_legend[i])
            #clrs_.append(clrs[i])


#    plt.ylim([0, int(sys.argv[1]) ])
    lns=None
    ax = plt.gca()
    for i in range(len(plots)):
        p = plots[i]
        clr = clrs[i]
        if i == 2:
            #accuracy
            ax2 = ax.twinx()
            ax2.set_ylabel('Accuracy', color=clr)
            ln = ax2.plot(p[0], p[1], clr)
            if lns is None:
                lns = ln
            else:
                lns += ln
            ax2.set_ylim(ymax=1)
            # ax2.legend('Accuracy %')
        else:
            if len(plots[i][0]) == 0:
                continue
            ln = plt.plot(p[0], p[1], clr)
            if lns is None:
                lns = ln
            else:
                lns += ln

            plt.ylim([0, float(sys.argv[1]) ])
            plt.ylabel("Loss")
            # plt.legend(plots_legend,  loc=0)

        # plt.plot(xListt, yListt, 'r')

    if lns is None:
        return

    plt.grid(True)
    plt.xlabel("Iterations")
    #plt.legend(['Training loss', 'Validation loss'])


    ax.legend(lns, legend)



def plotValue(plot, x, y):

    global plots

    while len(plots) <= plot:
        plots.append([list(), list()])

    plots[plot][0].append(x)
    plots[plot][1].append(y)

    if file == None:
        drawnow(makeFig)
        plt.pause(0.000000000001)

    pass


def parseLine(line):
    global last_iter
    global last_iter_base

    toks = line.split(' ')

    toks_=[]
    for t in toks:
       if t.strip() == '':
            continue
       toks_.append(t)
    toks=toks_

    if len(toks) > 8 and toks[4] == 'Iteration':
        if toks[5][:-1] == '' or toks[5][-1] != ',':
            return
        current_iter = int(toks[5][:-1])

        if current_iter < last_iter:
            print "=== Appending new log data. Offset=",last_iter
            last_iter_base += last_iter

        last_iter =  current_iter

    iter = last_iter_base + last_iter

    #    if len(toks) > 14 and toks[-9] == '#0:' and  toks[-10] == 'output' and toks[-7] == '=':
    if len(toks) > 7 and toks[6] == 'loss' and toks[7] == '=':
            loss = toks[8].strip()
            date = toks[1].strip()
            print date, iter, loss

            plotValue(0, iter, loss)

    elif len(toks) > 8 and toks[6] == 'output' and toks[9] == '=' and toks[4] == 'Test':
        if (toks[8] == "entropy_loss" or toks[8] == "mse" or toks[8] == "loss"):
            loss = float(toks[10].strip())
            date = toks[1].strip()
            print "** Test loss  ", date, iter, loss
            if math.isnan(loss):
                return

            plotValue(1, iter, loss)


        elif (toks[8] == "accuracy"):
            # The test data
            loss = float(toks[10].strip())
            date = toks[1].strip()
            print "** Test Accuracy  ", date, iter, loss
            if math.isnan(loss):
                return

            plotValue(2, iter, loss)

            pass


def run():


    global file
    global plots_legend

    plt.ion() # enable interactivity
    fig=plt.figure() # make a figure

    plots_legend = ['Training loss', 'Validation loss', 'Accuracy']

    #for line in fileinput.input():

    file_list = []
    file_id = -1
    if len(sys.argv) > 2:
        file_list = sys.argv[2:]
        file_id = 0

    while 1:
        if file_id > -1:
            if file == None:
                print "Opening file: ",file_list[file_id]
                file = open(file_list[file_id], 'rt')

            line = file.readline()
            if not line:
                file.close()
                file = None
                file_id+=1
                if file_id == len(file_list):
                    file_id = -1

        else:
            timeout = 0.1
            rlist, _, _ = select([sys.stdin], [], [], timeout)
            if rlist:
                line = sys.stdin.readline()
            else:
                #print "waiting"
                #plt.show()
                drawnow(makeFig)
                plt.pause(0.2)

                continue


        parseLine(line)

        pass


run()
