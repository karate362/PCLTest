#pragma once

#include <fstream>
#include <Eigen\src\Core\Matrix.h>

void ComputePnP(const char* folder_name,int start_idx,int total_num,int jump_num);//Compute two-view PnP

void Compute_Geometry(const char* folder_name,int start_idx,int total_num,int jump_num,const int app_idx,int icp_itnum,double icp_threshold = 0.05);//Compute geometry

void FindMatchSeq(const char* folder_name,int ref_idx,int start_idx,int total_num,int jump_num,double epidist,double confidence,bool is_save = false);//Find matching sequence

void ViewGeometry(const char* folder_name,int start_idx,int total_num,int jump_num);//Read computed geometry

void FindFeatureAndDraw(const char* folder_name,int start_idx,int total_num,int jump_num);

void PnPLocalization(const char* folder_name,int start_idx,int total_num,int jump_num);

void GenerateActionTrainData(int t1,int t2,int cidx);//delta is computed by spacing between t1~t2, cidx is the class index

void ActionRecognition(const char* folder_name,int start_idx,int total_num,int jump_num);

template <typename PointT> void PCL2P3d(const pcl::PointCloud<PointT> &cloud,vector<cv::Point3f>& p3ds);
void Kps2P2d(int idx,std::vector<cv::DMatch >& matches,std::vector<cv::KeyPoint>& keypoints,std::vector<cv::Point2f>& p2ds);
void PnPret2Mat4f(cv::Mat& rvec,cv::Mat& tvec,Eigen::Matrix4f& M);


class RT3D{

public:
	RT3D(){
		R[0][0] = 1;
		R[0][1] = 0;
		R[0][2] = 0;
		T[0] = 0;

		R[1][0] = 0;
		R[1][1] = 1;
		R[1][2] = 0;
		T[1] = 0;

		R[2][0] = 0;
		R[2][1] = 0;
		R[2][2] = 1;
		T[2] = 0;
		
	}

	void ReadFromFile(std::ifstream &fin);

	void Transform3D(double &x, double &y, double &z){
		double ix = x;
		double iy = y;
		double iz = z;

		x = R[0][0]*ix + R[0][1]*iy + R[0][2]*iz + T[0];
		y = R[1][0]*ix + R[1][1]*iy + R[1][2]*iz + T[1];
		z = R[2][0]*ix + R[2][1]*iy + R[2][2]*iz + T[2];
	}

	void ReadFromMatrix4f(Eigen::Matrix4f &M);

public:
	double R[3][3];
	double T[3];
};