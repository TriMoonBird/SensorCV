#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include "ImageTransform.h"
#include "Util.h"

#define MIN_HESSIAN (300)
#define TIME_THRES (2)
#define PICTURE_SIZE (500)

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
  String path_1 = "../Pic/img_1.jpg";
  String path_2 = "../Pic/img_2.jpg";

  Mat rawimg_1 = imread( path_1, CV_LOAD_IMAGE_COLOR );
  Mat rawimg_2 = imread( path_2, CV_LOAD_IMAGE_COLOR );

  if( !rawimg_1.data || !rawimg_2.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  Rect myROI_1(0, 0, rawimg_1.cols, rawimg_1.rows);
  Rect myROI_2(0, 0, rawimg_2.cols, rawimg_2.rows);

  Mat img_1, img_2;
  resize(extractROI(rawimg_1, myROI_1), img_1, Size(PICTURE_SIZE, PICTURE_SIZE));
  resize(extractROI(rawimg_2, myROI_2), img_2, Size(PICTURE_SIZE, PICTURE_SIZE));

  //-- Step 1: Detect the keypoints using SIFT Detector
  int minHessian = MIN_HESSIAN;

  SiftFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  clock_t detectStart = clock();
  detector.detect( img_1, keypoints_1 );
  detector.detect( img_2, keypoints_2 );
  clock_t detectEnd = clock();
  printTimeMs("Detect time: ", clockDiffMs(detectEnd, detectStart));

  //-- Step 2: Calculate descriptors (feature vectors)
  SiftDescriptorExtractor extractor;

  Mat descriptors_1, descriptors_2;

  clock_t extractStart = clock();
  extractor.compute( img_1, keypoints_1, descriptors_1 );
  extractor.compute( img_2, keypoints_2, descriptors_2 );
  clock_t extractEnd = clock();
  printTimeMs("Extract time: ", clockDiffMs(extractEnd, extractStart));

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;

  clock_t matchStart = clock();
  matcher.match( descriptors_1, descriptors_2, matches );
  clock_t matchEnd = clock();
  printTimeMs("Match time: ", clockDiffMs(matchEnd, matchStart));

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_1.rows; i++ )
  {
	double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  cout << "-- Max dist : " << max_dist << endl;
  cout << "-- Min dist : " << min_dist << endl;

  //-- Draw only "good" matches (i.e. whose distance is less than TIME_THRES*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_1.rows; i++ )
  {
	if( matches[i].distance <= max(TIME_THRES*min_dist, 0.02) )
    { good_matches.push_back( matches[i]); }
  }

  //-- Draw only "good" matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //-- Show detected matches
  //imshow( "Good Matches", img_matches );
  imwrite("../Result/matchPicture.jpg", img_matches);

  //for( int i = 0; i < (int)good_matches.size(); i++ )
  //{ printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

  ofstream matchFile ("../Result/matchPoints.txt");
  if (!matchFile.is_open())
  { cout << "File Error" << endl; return -2; }

  for( int i = 0; i < (int)good_matches.size(); i++ )
  {
	int j = good_matches[i].queryIdx;
	int k = good_matches[i].trainIdx;
	matchFile << (int)keypoints_1[j].pt.x << " " << (int)keypoints_1[j].pt.y << " "
			<< (int)keypoints_2[k].pt.x << " " << (int)keypoints_2[k].pt.y << endl;
  }
  matchFile.close();

  waitKey(0);

  return 0;
}
