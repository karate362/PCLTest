#include "OpenCV2PCL.h"
#include <iostream>
#include <fstream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/transformation_estimation.h>
#include "pcl/registration/warp_point_rigid_6d.h" 
#include <pcl/common/transformation_from_correspondences.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <string>
#include "Geometry_Functions.h"
#include "ProjFunc.h"
#include "LDARec.h"

using namespace std;

void ReadMatrix4f(Eigen::Matrix4f &M, ifstream &in);

enum action_idx{
	forward = 0,
	back,
	turnleft,
	turnright,
	up,
	down,
	stop
};

int LDAActionRecognition(RT3D& dRT,LDARec& LREC){
	
	const double tmin = 0.05;//5 cm
	const double amin = 0.08;//5 deg
	
	double input[6];
	input[0] = dRT.T[0];
	input[1] = dRT.T[1];
	input[2] = dRT.T[2];
	R2Euler(dRT.R,input[3],input[4],input[5]);

	if(fabs(input[0])<tmin && fabs(input[1])<tmin && fabs(input[2])<tmin )
		if(fabs(input[3])<amin && fabs(input[4])<amin && fabs(input[5])<amin)
			return 6;//stop

	return LREC.ActionRecognition(input);
}

void ActionRecognition(const char* folder_name,int start_idx,int total_num,int jump_num){
	
	int img_idx = start_idx;
	int img_num = total_num;
	int img_num2 = jump_num;

	int window_size = 5;//sliding window size

	char imgname[64];
	char pcdname[64];

	ifstream in_pose;

	in_pose.open("RT.txt");

 cv::Mat colorImage_1(KINECT_IMAGE_HEGIHT,KINECT_IMAGE_WIDTH,CV_8UC4);

 cv::Mat img1;
  /////////////////////LDA objects//////////////////////////
 vector<string> action_num;
 action_num.push_back("forward");
 action_num.push_back("back");
 action_num.push_back("turn left");
 action_num.push_back("turn right");
 action_num.push_back("up");
 action_num.push_back("down");
 action_num.push_back("stop");

 std::printf("Initializing\n");
 LDARec LRec(6,action_num);


 //////////////////////PCL rigid motion estimation/////////////////////////
	Eigen::Matrix4f TotaltransforMation = Eigen::Matrix4f::Identity(); 
	Eigen::Matrix4f LastTotaltransforMation = Eigen::Matrix4f::Identity(); 
	Eigen::Matrix4f Deltatrans = Eigen::Matrix4f::Identity(); 
	RT3D deltaRT;

	vector<Eigen::Matrix4f> RTs;
	RTs.resize(window_size);

	vector<RT3D> dRTs;
	//dRTs.resize(window_size2);
	vector<int> vote_counts(7,0);//class voting, 0~5 are class, 6 means stop
//////////////////////////////////////////////////////////////////////////////

	int d = 0;
	int input_idx = 0;//push new elements to the dRTs[input_idx]

	std::printf("Start Recognition\n");

	for(int i=0; i<img_num-1; ++i, d+=img_num2){

		 sprintf(imgname,"%s/rgb_%d.jpg",folder_name,img_idx+d);
		 ////////////Reading data/////////////////////
		 img1 = cv::imread(imgname,CV_LOAD_IMAGE_COLOR);

		 //Flip the image
		 cv::flip(img1,colorImage_1,1);

		 ReadMatrix4f(TotaltransforMation,in_pose);
		 
		 dRTs.clear();
		 //Compute delta transformation...
		 /*
		 for(int j=0;j<window_size2;++j){
			 int idx = (input_idx - 2 - j)%window_size;
			 Deltatrans = RTs[idx].inverse()*TotaltransforMation;
			 deltaRT.ReadFromMatrix4f(Deltatrans);
			 dRTs.push_back(deltaRT);
		 }*/
		 
		for(int j=0;j<RTs.size();++j){
			 Deltatrans = RTs[j].inverse()*TotaltransforMation;
			 deltaRT.ReadFromMatrix4f(Deltatrans);
			 dRTs.push_back(deltaRT);
		 }

		 RTs[input_idx] = TotaltransforMation;
		 input_idx = (input_idx+1)%window_size;

		 //Compute the voting

		 for(int j=0;j<vote_counts.size();++j)
			 vote_counts[j] = 0;

		 for(int j=0;j<dRTs.size();++j){
			 int c_idx = LDAActionRecognition(dRTs[j],LRec);
			 vote_counts[c_idx]++;

			 std::printf("%d ",c_idx);
		 }
		  std::printf("\n");

		 //////////Recognition by voting////////////
		 int now_action = stop;
		 int max_action = -1;
		 for(int j=0;j<vote_counts.size();++j){
			 if(vote_counts[j]>max_action){
				max_action = vote_counts[j];
				now_action = j;
			 }
		 }

		 ////Show Result///
		 cv::putText(colorImage_1,action_num[now_action],cv::Point(20,20),cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,200,0), 1, CV_AA);
		 cv::imshow("Rec result",colorImage_1);

		 cv::waitKey(500);
	}

	in_pose.close();

}
