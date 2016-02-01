#ifndef OFEVAL_H
#define OFEVAL_H

#include <string>
#include <opencv2/core.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>


//double ofEval(cv::Mat flow, cv::Mat gt);
cv::Mat flowToColor(const cv::Mat &uflow, const cv::Mat &vflow,  float maxmotion);

#ifdef __cplusplus
}
#endif

#endif
