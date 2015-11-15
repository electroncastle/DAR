#!/usr/bin/python

import sys
import fileinput
import matplotlib.pyplot as plt
import numpy as np
import time
from select import select

sgn = 'Train net output #0: mse ='
sgn_len = len(sgn)


# Install drawnow
# pip install drawnow.
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np

def makeFig():
    #plt.scatter(xList,yList) # I think you meant this
    plt.plot(xList,yList) # I think you meant this
    plt.ylim([0, int(sys.argv[1]) ])

    plt.plot(xListt, yListt, 'r')
#    plt.ylim([0, int(sys.argv[1]) ])

    plt.grid(True)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(['Training loss', 'Validation loss'])


plt.ion() # enable interactivity
fig=plt.figure() # make a figure

xList=list()
yList=list()
x=0

xListt=list()
yListt=list()
xt=0

#for line in fileinput.input():

file = None
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

    #print "data: ",line

    # pos = line.find(sgn)
    # if pos>0:
    toks = line.split(' ')

    if len(toks) > 14 and toks[-9] == '#0:' and  toks[-10] == 'output' and toks[-7] == '=':
        if toks[-12] == 'Train':
            # loss_str = line[pos+sgn_len:].strip()
            # toks = loss_str.split(' ')
            loss = toks[-2]
            date = toks[1]
            print date, loss

            yList.append(loss)
            xList.append(x)
            if file == None:
                drawnow(makeFig)
                plt.pause(0.000000000001)
            x += 20
        else:
            # The test data
            losst = toks[-2]
            date = toks[1]
            print date, losst

            yListt.append(losst)
            xListt.append(x)
            if file == None:
                drawnow(makeFig)
                plt.pause(0.000000000001)
#            xt += 20

            pass

    pass


plt.show(block=True)
sys.exit(0)

for i in np.arange(50):
    y=np.random.random()
    xList.append(i)
    yList.append(y)
    drawnow(makeFig)
    #makeFig()      The drawnow(makeFig) command can be replaced
    #plt.draw()     with makeFig(); plt.draw()
    plt.pause(0.001)



fig = plt.figure()
ax = fig.add_subplot(111)

# some X and Y data
x = np.arange(10000)
y = np.random.randn(10000)

li, = ax.plot(x, y)

# draw and show it
fig.canvas.draw()
plt.show(block=False)

# loop to update the data
while True:
    try:
        y[:-10] = y[10:]
        y[-10:] = np.random.randn(10)

        # set the new data
        li.set_ydata(y)

        fig.canvas.draw()

        time.sleep(0.1)

    except KeyboardInterrupt:
        break