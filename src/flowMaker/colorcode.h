#include "opencv2/core.hpp"

void computeColor(float fx, float fy, cv::Vec3b &pix);
cv::Mat motionToColor(cv::Mat &flow_x, cv::Mat &flow_y, float maxmotion=0);
