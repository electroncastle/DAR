#!/usr/bin/python


import os
import sys
import numpy as np
import utils
import cv2


import os
import io
import sys
# import random
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import PIL.Image
import time
import lmdb
import cv2
import utils
import gc

if 'DAR_ROOT' not in os.environ:
    print 'FATAL ERROR. DAR_ROOT not set, exiting'
    sys.exit()

dar_root = os.environ['DAR_ROOT']
app_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

sys.path.insert(0, dar_root + '/python')
import caffe

sys.path.insert(0, dar_root+'/src/ofEval/build-debug/')
import ofEval_module as em


def of2polar(x,y):
    cpx = x+(y*(1j))
    mag = np.abs(cpx)
    angle_src = np.angle(cpx) # +/- PI

    # Convert to range 0..2pi
    os = angle_src < 0.0
    angle_src[os]=2.0*np.pi+angle_src[os]

    return angle_src, mag



def calc_epe(gtu, gtv, u, v):

    bord = 3
    smallflow=0.0

    stu=gtu[bord+1:-bord, bord+1:-bord]
    stv=gtv[bord+1:-bord, bord+1:-bord]
    su=u[bord+1:-bord, bord+1:-bord]
    sv=v[bord+1:-bord, bord+1:-bord]

    # ignore a pixel if both u and v are zero
    #ind2=find((stu(:).*stv(:)|sv(:).*su(:))~=0);
    # ind2=find(abs(stu(:))>smallflow | abs(stv(:)>smallflow));

    ind1 = np.abs(stu.flatten()) > smallflow
    ind2 = np.abs(stv.flatten()) > smallflow
    ind2 = ind1 | ind2

    ind3 = np.abs(stu.flatten()) < 1e8
    ind4 = np.abs(stv.flatten()) < 1e8
    ind2 = ind2 & ind3 & ind4

    # #length(ind2)
    # # n = 1.0/np.sqrt(su(ind2)**2 + sv(ind2)**2+1)
    # # un=su(ind2)*n
    # # vn=sv(ind2)*n
    su = su.flatten()
    sv = sv.flatten()
    # n = 1.0/np.sqrt(su[ind2]**2 + sv[ind2]**2+1)
    # un=su[ind2]*n
    # vn=sv[ind2]*n
    print "flow: min/max ", su.min(), ', ',sv.min(),' / ',su.max(), ', ',sv.max()
    #
    # # n = 1.0/np.sqrt(su**2 + sv**2+1)
    # # un=su*n
    # # vn=sv*n
    #
    # # tn=1.0/np.sqrt(stu(ind2)**2+stv(ind2)**2+1)
    # # tun=stu(ind2)*tn
    # # tvn=stv(ind2)*tn
    stu = stu.flatten()
    stv = stv.flatten()
    # tn = 1.0/np.sqrt(stu[ind2]**2 + stv[ind2]**2+1)
    # tun=stu[ind2]*tn
    # tvn=stv[ind2]*tn
    # # tn=1.0/np.sqrt(stu**2+stv**2+1)
    # # tun=stu*tn
    # # tvn=stv*tn
    print "GT: min/max ", stu[ind2].min(), ', ',stv[ind2].min(),' / ',stu[ind2].max(), ', ',stv[ind2].max()

    # ang=np.acos(un*tun + vn*tvn + (n*tn))
    # mang=np.mean(ang)
    #
    # # Calculate output
    # mang=mang*180.0/np.pi
    # stdang = np.std(ang*180.0 / np.pi)

    epe = np.sqrt((stu-su)**2 + (stv-sv)**2)
    epe = epe[ind2]
    mepe = np.mean(epe[:])

    return epe, mepe



def crop(img, width, height):
    # Crop around the center by given width, height
    h = img.shape[0]
    w = img.shape[1]

    if width>-1 and height >-1:
        t = (h-height)/2
        l = (w-width)/2
        batch_croppped = img[t:t+height, l:l+width]
        # batch_croppped = batch_croppped.astype(float)

    return batch_croppped


def test_epe(img1, img2):

    # Read the flow as RGB
    gt_flow_filename = '/home/jiri/Lake/DAR/share/datasets/middlebury/other-gt-flow/RubberWhale/flow10.flo'
    gt_flow_filename = '/home/jiri/Lake/DAR/share/datasets/MPI_Sintel/training/flow/ambush_2/frame_0001.flo'

    # Load ground truth
    id = '0000000031'
    id = '0000000050'
    id = '0000000085'
    id = '0000000086'
    x_filename = 'orig-'+id+'-img-2'
    gt_path = '/home/jiri/Dropbox/Kingston/Final/doc/msc-thesis/Figures/of/sintel/'
    gt_xflow_filename = gt_path+'orig-'+id+'-x-flow.jpg'
    gt_yflow_filename = gt_path+'/orig-'+id+'-y-flow.jpg'

    #------------------------------------------------------------------------------------------------------------
    gt_path = '/home/jiri/Dropbox/Kingston/Final/doc/msc-thesis/Figures/of/thumos/'
    x_filename = 'flow_2272'
    gt_xflow_filename = gt_path+'flow_x_2272.jpg'
    gt_yflow_filename = gt_path+'flow_y_2272.jpg'

    gtv = cv2.imread(gt_yflow_filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    gtu = cv2.imread(gt_xflow_filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    gtv = 10*(gtv-127)/127.0
    gtu = 10*(gtu-127)/127.0

    gtv = crop(gtv, 224, 224)
    gtu = crop(gtu, 224, 224)

    #------------------------------------------------------------------------------------------------------------
    gt_path = '/home/jiri/Dropbox/Kingston/Final/doc/msc-thesis/Figures/of/midlebury/'
    x_filename = 'img-2-Hydrangea_frame10-15161'
    gt_xflow_filename = gt_path+x_filename
    gt_flow_filename = '/home/jiri/Lake/DAR/share/datasets/middlebury/other-gt-flow/Hydrangea/flow10.flo'

    #------------------------------------------------------------------------------------------------------------
    # gt_path = '/home/jiri/Dropbox/Kingston/Final/doc/msc-thesis/Figures/of/midlebury/'
    # x_filename = 'img-2-Urban2_frame10-15161'
    # gt_xflow_filename = gt_path+x_filename
    # gt_flow_filename = '/home/jiri/Lake/DAR/share/datasets/middlebury/other-gt-flow/Urban2/flow10.flo'

    #------------------------------------------------------------------------------------------------------------
    x,h,u,v = utils.read_flo_fast(gt_flow_filename)
    gtu = u.astype(np.float32)
    gtv = v.astype(np.float32)
    gtu = crop(gtu, 224, 224)/12.0
    gtv = crop(gtv, 224, 224)/12.0


    # Load tested sample
    flow_filename = '/home/jiri/Lake/DAR/src/flownet-release/models/flownet/'+x_filename+'.flo' #flownets-pred-0000000.flo'
    x,h,u,v = utils.read_flo_fast(flow_filename)
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    print "u/v ",u.shape, "  ",v.shape
    print "gtu/gtv ",gtu.shape, "  ",gtv.shape

    invalid = u > 1e8
    u[invalid] = 0.0
    invalid = v > 1e8
    v[invalid] = 0.0

    clr_flow = em.flowToColor(u, v, -30)
    gtclr_flow = em.flowToColor(gtu*12, gtv*12, -1)

    gtu = gtu*12+127
    gtv = gtv*12+127
    u = u*1+127
    v = v*1+127
    epe_val, mepe = calc_epe(gtu, gtv, u, v)


    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(clr_flow, "EPE={:.2f}".format(mepe), (int(clr_flow.shape[1]-clr_flow.shape[1]*0.13), int(clr_flow.shape[0]*0.07)),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if mepe > -1:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(clr_flow, "EPE: {:.2f}".format(mepe), (int(clr_flow.shape[1]-100), 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("Flow", clr_flow)
    cv2.imwrite(gt_xflow_filename+'-flownet.jpg', clr_flow)
    #cv2.moveWindow(wnd_name+" Flow", wnd_x, wnd_y)

    cv2.imshow("GT Flow", gtclr_flow)
    cv2.imwrite(gt_xflow_filename+'-GT.jpg', gtclr_flow)

    cv2.waitKey(0)




if __name__ == "__main__":
    test_epe(sys.argv[1], sys.argv[2])
