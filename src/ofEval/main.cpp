#include <stdio.h>
#include <string>
#include "ofEval.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <exception>
#include "np_opencv_converter.hpp"

#include "flowIO.h"
#include <colorcode.h>


typedef struct {
    double r;       // percent
    double g;       // percent
    double b;       // percent
} rgb;

typedef struct {
    double h;       // angle in degrees
    double s;       // percent
    double v;       // percent
} hsv;

static hsv   rgb2hsv(rgb in);
static rgb   hsv2rgb(hsv in);

hsv rgb2hsv(rgb in)
{
    hsv         out;
    double      min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min  < in.b ? min  : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max  > in.b ? max  : in.b;

    out.v = max;                                // v
    delta = max - min;
    if (delta < 0.00001)
    {
        out.s = 0;
        out.h = 0; // undefined, maybe nan?
        return out;
    }
    if( max > 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
        out.s = (delta / max);                  // s
    } else {
        // if max is 0, then r = g = b = 0
            // s = 0, v is undefined
        out.s = 0.0;
        out.h = NAN;                            // its now undefined
        return out;
    }
    if( in.r >= max )                           // > is bogus, just keeps compilor happy
        out.h = ( in.g - in.b ) / delta;        // between yellow & magenta
    else
    if( in.g >= max )
        out.h = 2.0 + ( in.b - in.r ) / delta;  // between cyan & yellow
    else
        out.h = 4.0 + ( in.r - in.g ) / delta;  // between magenta & cyan

    out.h *= 60.0;                              // degrees

    if( out.h < 0.0 )
        out.h += 360.0;

    return out;
}


rgb hsv2rgb(hsv in)
{
    double      hh, p, q, t, ff;
    long        i;
    rgb         out;

    if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch(i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;
}

cv::Mat toRGB(cv::Mat hsv)
{


}

cv::Mat toHSV(cv::Mat rgb)
{


}


int main(int argc, char **argv)
{

    std::string path = "/home/jiri/Lake/DAR/projects/optical_flow_regression/OFR_rgb_flownet_direct/";
    std::string flow_filename = path+"flow-CNN-0000000320-35000.jpg";
    std::string gt_filename = path+"flow-GT-0000000320-35000.jpg";

    cv::Mat flow = cv::imread(flow_filename, cv::IMREAD_COLOR);
    cv::Mat gt = cv::imread(gt_filename, cv::IMREAD_COLOR);

    //ofEval(flow, gt);


    // OF to HSV test
    cv::Mat uflow, vflow;
  //  ReadFlowFile(uflow, vflow, "/home/jiri/Lake/DAR/share/datasets/MPI_Sintel/training/flow/market_2/frame_0004.flo");
    ReadFlowFile(uflow, vflow, "/home/jiri/Lake/DAR/src/flownet-release/models/flownet/flownets-pred-0000000.flo");

//    vflow = cv::imread("/home/jiri/Dropbox/Kingston/Final/doc/msc-thesis/Figures/of/candidates/flow_x_0113.jpg", cv::IMREAD_GRAYSCALE);
//    uflow = cv::imread("/home/jiri/Dropbox/Kingston/Final/doc/msc-thesis/Figures/of/candidates/flow_y_0113.jpg", cv::IMREAD_GRAYSCALE);
//    uflow.convertTo(uflow, CV_32FC1, 1.0/255.0, -0.5);
//    vflow.convertTo(vflow, CV_32FC1, 1.0/255.0, -0.5);

    // Read two independent images
    cv::Mat clrMotion = motionToColor(uflow, vflow, 0);
    cv::imwrite("/home/jiri/Lake/DAR/src/flownet-release/models/flownet/flow_img.png", clrMotion);
//    cv::imwrite("/home/jiri/Dropbox/Kingston/Final/doc/msc-thesis/Figures/of/sintel/flow_1832.png", clrMotion);


    cv::imshow("rgb", clrMotion);
    cv::moveWindow("rgb", 100,100);

//    ReadFlowFile(uflow, vflow, "/home/jiri/Lake/DAR/share/datasets/middlebury/other-gt-flow/Venus/flow10.flo");
//    clrMotion = motionToColor(uflow, vflow, 0);
//    cv::imshow("GT", clrMotion);

    cv::waitKey(0);
    return 0;


    /*
    cv::Mat mag;
    cv::Mat theta;
    cv::magnitude(uflow, vflow, mag);
    cv::phase(uflow, vflow, theta);


    std::vector<cv::Mat> channels;
    channels.resize(3);
//    Mat tmp1=(channels[2]/255);
//    Mat tmp;
//    pow(tmp1,1.5,tmp);
//    channels[2]=255 *tmp;

    //cv::Mat result;

    //Hue
//    channels[0] = cv::Mat(uflow.rows, uflow.cols, CV_32FC1, cv::Scalar(1.0));
    channels[0] = theta*180/(M_PI_2) ;

    // Saturation
    channels[1] = cv::Mat(uflow.rows, uflow.cols, CV_32FC1, cv::Scalar(255.0));

    // Value
    cv::normalize(mag, channels[2], 0, 255, cv::NORM_MINMAX);

    cv::Mat hsv;
    cv::merge(channels, hsv);
    cv::Mat img;
    cv::cvtColor(hsv, img, CV_HSV2BGR_FULL);
    */


    //calculate angle and magnitude
    cv::Mat magnitude, angle;
    cv::cartToPolar(uflow, vflow, magnitude, angle, true);

    //translate magnitude to range [0;1]
    double mag_max;
    double mag_min;
    cv::minMaxLoc(magnitude, &mag_min, &mag_max);
    //magnitude.convertTo(magnitude, -1, 1.0/mag_max); //-1
    //magnitude.convertTo(magnitude, -1, 10.0/mag_max); //-1
    cv::minMaxLoc(magnitude, &mag_min, &mag_max);

//    cv::minMaxLoc(angle, &mag_min, &mag_max);
//    angle.convertTo(angle, CV_32F, 1.0/mag_max); //-1

    //build hsv image
    cv::Mat _hsv[3], hsv;
    _hsv[0] = angle;// / 360.0;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[1] = cv::Mat(uflow.rows, uflow.cols, CV_32FC1, cv::Scalar(100));
    _hsv[2] = magnitude;
//    _hsv[2] = cv::Mat(uflow.rows, uflow.cols, CV_32FC1, cv::Scalar(0.5));
    cv::merge(_hsv, 3, hsv);

    //convert to BGR and show
    cv::Mat img;//CV_32FC3 matrix
    cv::cvtColor(hsv, img, cv::COLOR_HSV2BGR);
    cv::imshow("hsv->rgb", img);

    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    cv::cvtColor(hsv, img, cv::COLOR_HSV2BGR);
    cv::imshow("rgb", img);
    cv::waitKey(0);

//    cv::cvtColor(img, hsvImg, CV_BGR2HSV);


    return 0;
}


// Wrap a few functions and classes for testing purposes
namespace fs { namespace python {

BOOST_PYTHON_MODULE(ofEval_module)
{
  // Main types export
  fs::python::init_and_export_converters();
  py::scope scope = py::scope();

  // Basic test
  //py::def("test_np_mat", &test_np_mat);
  boost::python::def("flowToColor", &flowToColor);

//  // With arguments
//  py::def("test_with_args", &test_with_args,
//          (py::arg("src"), py::arg("var1")=1, py::arg("var2")=10.0, py::arg("name")="test_name"));

//  // Class
//  py::class_<GenericWrapper>("GenericWrapper")
//      .def(py::init<py::optional<int, float, double, std::string> >(
//          (py::arg("var_int")=1, py::arg("var_float")=1.f, py::arg("var_double")=1.d,
//           py::arg("var_string")=std::string("test"))))
//      .def("process", &GenericWrapper::process)
//      ;
}

} // namespace fs
} // namespace python
