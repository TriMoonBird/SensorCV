#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

#ifndef _IMAGETRASFORM_H_
#define _IMAGETRASFORM_H_

Mat changeColor(Mat& image);
Mat rotateImage(Mat& src, double angle);
Mat matMultiply(Mat& A, Mat& B);

#endif
