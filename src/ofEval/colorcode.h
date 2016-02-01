#include "opencv2/core.hpp"

void computeColor(float fx, float fy, cv::Vec3b &pix);
cv::Mat motionToColor(const cv::Mat &flow_x, const cv::Mat &flow_y, float maxmotion=0);
