#include "OpenCV2PCL.h"
#include "RobustMatcher.h"
#include "Geometry_Functions.h"
using namespace std;
using namespace cv;

extern double InvCam[3][3];


void CalibratePoints(std::vector<cv::Point3f>& points){

	for(int i=0;i<points.size();++i){
		Point3f p = points[i];
		p.y = 480.0f - p.y;//for flipped image
		points[i].x = InvCam[0][0]*p.x + InvCam[0][1]*p.y + InvCam[0][2]*p.z;
		points[i].y = InvCam[1][0]*p.x + InvCam[1][1]*p.y + InvCam[1][2]*p.z;
		points[i].z = InvCam[2][0]*p.x + InvCam[2][1]*p.y + InvCam[2][2]*p.z;
	}
}

//wrapper
void Reconstruct3D(RT3D& RT, std::vector<cv::DMatch> & Matches,std::vector<cv::KeyPoint>& img1_keypoints, std::vector<cv::KeyPoint>& img2_keypoints, std::vector<int>& pidxs, std::vector<cv::Point3f>& points){
	
	printf("Calibrating!\n");
	vector<Point3f> points1, points2;
	Point3f p;
	p.z = 1.0;

	for(int i=0;i<img1_keypoints.size();++i){
		p.x = img1_keypoints[i].pt.x;
		p.y = img1_keypoints[i].pt.y;
		points1.push_back(p);
	}

	for(int i=0;i<img2_keypoints.size();++i){
		p.x = img2_keypoints[i].pt.x;
		p.y = img2_keypoints[i].pt.y;
		points2.push_back(p);
	}
	CalibratePoints(points1);
	CalibratePoints(points2);

	printf("Reconstruct!\n");
	Reconstruct3D(RT, Matches, points1, points2, pidxs, points);
}


//Given RT, matches, keypoints, output the 3D points and corresponding index in keypoints 1
//keypoints 1, 2 should be calibrated
void Reconstruct3D(RT3D& RT, std::vector<cv::DMatch> & Matches,std::vector<cv::Point3f>& img1_keypoints, std::vector<cv::Point3f>& img2_keypoints, std::vector<int>& pidxs, std::vector<cv::Point3f>& points){
	float X2[3][3];
	float a[3];
	float b[3];
	float c[3];
	Point3f np;

	pidxs.clear();
	points.clear();

	for(int i=0;i<Matches.size(); ++i){
		printf("loop %d\n",i);
		Point3f x1 = img1_keypoints[Matches[i].queryIdx];
		Point3f x2 = img2_keypoints[Matches[i].trainIdx];
		
		int pidx = Matches[i].queryIdx;
		
		X2[0][0] = 0.0f;
		X2[0][1] = -1.0f * x2.z;
		X2[0][2] = x2.y;
		X2[1][0] = x2.z;
		X2[1][1] = 0.0f;
		X2[1][2] = -1.0f * x2.x;
		X2[2][0] = -1.0f * x2.y;
		X2[2][1] = x2.x;
		X2[2][2] = 0.0f;
		
		c[0] = RT.R[0][0] * x1.x + RT.R[0][1] * x1.y + RT.R[0][2] * x1.z;
		c[1] = RT.R[1][0] * x1.x + RT.R[1][1] * x1.y + RT.R[1][2] * x1.z;
		c[2] = RT.R[2][0] * x1.x + RT.R[2][1] * x1.y + RT.R[2][2] * x1.z;

		a[0] = X2[0][0] * c[0] + X2[0][1] * c[1] + X2[0][2] * c[2]; 
		a[1] = X2[1][0] * c[0] + X2[1][1] * c[1] + X2[1][2] * c[2]; 
		a[2] = X2[2][0] * c[0] + X2[2][1] * c[1] + X2[2][2] * c[2]; 

		b[0] = X2[0][0] * RT.T[0] + X2[0][1] * RT.T[1] + X2[0][2] * RT.T[2]; 
		b[1] = X2[1][0] * RT.T[0] + X2[1][1] * RT.T[1] + X2[1][2] * RT.T[2]; 
		b[2] = X2[2][0] * RT.T[0] + X2[2][1] * RT.T[1] + X2[2][2] * RT.T[2]; 

		float d = -1.0f * (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) / (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]) ;
		//Crush here! why??
		np.x = d * x1.x;
		np.y = d * x1.y;
		np.z = d;
		//printf("3D: %.2f,%.2f,%.2f\n",x1.x,x1.y,x1.z);
		//printf("3D: %.2f,%.2f,%.2f\n",X2[0][1],X2[1][1],X2[1][2]);

		pidxs.push_back(pidx);
		points.push_back(np);
	}

}
