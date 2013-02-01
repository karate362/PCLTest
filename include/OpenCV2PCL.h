// OpenCV2PCL.h : include necessary functions to get RGB-D images and point cloud from kinect.
// Transform to PCL types and openCV image types
//Kinect SDK and openCV are needed.
//It is for windows base only.

#pragma once
#include<Windows.h>
#include<NuiApi.h>	//KinectSDK利用時にinclude
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h> //basic pcl includes
#include <pcl/point_types.h>


#define KINECT_IMAGE_WIDTH 640
#define KINECT_IMAGE_HEGIHT 480
#define KINECT_DEPTH_WIDTH 320
#define KINECT_DEPTH_HEGIHT 240


int GetImage(cv::Mat &image,HANDLE frameEvent,HANDLE streamHandle);//Get Image, for color image, it should be CV_8UC4, for depth image, it should be CV_16UC1
void retrievePointCloudMap(cv::Mat &depth,cv::Mat &pointCloud_XYZ);//Match to Depth image 
void retrieveRGBCloudMap(cv::Mat &depth,cv::Mat &pointCloud_XYZ);//Match to RGB image 
void Mat2PCL_XYZRGB(cv::Mat &color, cv::Mat &pointCloud_XYZ, pcl::PointCloud<pcl::PointXYZRGBA>& cloud,std::vector<cv::KeyPoint>& keypoints,int range = -1);// Search in +-range square for a minimum depth
void Mat2PCL_XYZRGB_ALL(cv::Mat &color,cv::Mat &pointCloud_XYZ,pcl::PointCloud<pcl::PointXYZRGBA>& cloud);
void Mat2PCL_XYZRGB_MATCH(cv::Mat &color,cv::Mat &color2,cv::Mat &pointCloud_XYZ,cv::Mat &pointCloud_XYZ2,pcl::PointCloud<pcl::PointXYZRGBA>& cloud,pcl::PointCloud<pcl::PointXYZRGBA>& cloud2,std::vector<cv::KeyPoint>& keypoints,std::vector<cv::KeyPoint>& keypoints2,std::vector<cv::DMatch >& matches);
void Mat2PCL_XYZRGB_MATCH_PnP(cv::Mat &pointCloud_XYZ,std::vector<cv::Point3f>& p3ds,std::vector<cv::Point2f>& p2ds,std::vector<cv::KeyPoint>& keypoints,std::vector<cv::KeyPoint>& keypoints2,std::vector<cv::DMatch >& matches);
cv::Point3f * minimum_depth_point(cv::Mat &pointCloud_XYZ,int x,int y,int range);