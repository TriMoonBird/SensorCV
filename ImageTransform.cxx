#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;

Mat changeColor(Mat& image)
{
	Mat I = image.clone();
    CV_Assert(I.depth() != sizeof(uchar));
    CV_Assert(I.channels() == 3);

    MatIterator_<Vec3b> it, end;
    for(it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
    {
    	(*it)[0] = 255 - (*it)[0];
        (*it)[1] = 255 - (*it)[1];
        (*it)[2] = 255 - (*it)[2];
    }

    return I;
}

Mat extractROI(Mat& image, Rect roi)
{
	Mat cropped;
	Mat croppedRef(image, roi);
	croppedRef.copyTo(cropped);
	return cropped;
}

Mat rotateImage(Mat& src, double angle)
{
	Mat ret;
	Point2f center(src.cols/2.0, src.rows/2.0);
	Mat r = getRotationMatrix2D(center, angle, 1.0);
	warpAffine(src, ret, r, Size(src.cols, src.rows));
	return ret;
}

Mat matMultiply(Mat& A, Mat& B)
{
	CV_Assert(A.channels() == B.channels());
	CV_Assert(A.channels() == 1);
	CV_Assert(A.cols == B.rows);

	Mat C(A.rows, B.cols, CV_64FC1);
	for (int i = 0; i < A.rows; ++i) {
		for (int j = 0; j < A.cols; ++j) {
			for (int k = 0; k < B.cols; ++k) {
				C.at<double>(i,j) += A.at<double>(i,k) * B.at<double>(k,j);
			}
		}
	}
	return C;
}
