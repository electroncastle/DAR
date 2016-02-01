#include "ofEval.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/optflow.hpp"
#include <colorcode.h>
#include "flowIO.h"

//#include "numpy/ndarrayobject.h"

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <exception>
#include "np_opencv_converter.hpp"


void initInterface()
{
    boost::python::def("flowToColor", &flowToColor);
}

/**
 * @brief ofEval
 *  Compares color coded optical fow images and outputs EPE (end point pixel error)
 * @param flow
 * @param gt
 * @return
 */
double ofEval(cv::Mat flow, cv::Mat gt)
{
    cv::imshow("Flow", flow);
    cv::moveWindow("Flow", 50,50);
    cv::imshow("GT", gt);
    cv::moveWindow("GT", 550,50);

    cv::Mat uflow, vflow;
    ReadFlowFile(uflow, vflow, "/home/jiri/Lake/DAR/share/datasets/MPI_Sintel/training/flow/market_2/frame_0004.flo");

    cv::Mat clrMotion = motionToColor(vflow, uflow);

    cv::imshow("flow", clrMotion);

    cv::waitKey(0);
    return 0.0;
}


cv::Mat flowToColor(const cv::Mat &uflow, const cv::Mat &vflow, float maxmotion)
{
  cv::Mat clrMotion = motionToColor(uflow, vflow, maxmotion);
//  cv::Mat clrMotion = cv::Mat(244,244, CV_8UC3, cv::Scalar(127,127,127));
  return clrMotion;
}

