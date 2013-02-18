#include "OpenCV2PCL.h"
#include <pcl/visualization/cloud_viewer.h>
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
#include "RobustMatcher.h"
#include "Geometry_Functions.h"

using namespace std;


double CamIntrinsic[3][3] = {{531.15f, 0.0f, 320.0f}, {0.0f, 531.15f, 240.0f}, {0.0f, 0.0f, 1.0f}};
double InvCam[3][3] = {{531.15f, 0.0f, 320.0f}, {0.0f, 531.15f, 240.0f}, {0.0f, 0.0f, 1.0f}};
double DisCoef[5] = {0.0f,0.0f,0.0f,0.0f,0.0f};

template <typename PointT> void PCL2P3d(const pcl::PointCloud<PointT> &cloud,vector<cv::Point3f>& p3ds){

	int s = cloud.size();
	p3ds.resize(s);
	for(int i=0;i<s;++i){
		p3ds[i].x = cloud[i].x;
		p3ds[i].y = cloud[i].y;
		p3ds[i].z = cloud[i].z;
	}

}

void Kps2P2d(int idx,std::vector<cv::DMatch >& matches,std::vector<cv::KeyPoint>& keypoints,std::vector<cv::Point2f>& p2ds){
	
	int s = matches.size();
	p2ds.resize(s);

	for( int i = 0; i < matches.size(); i++ ){
		//-- Get the keypoints from the good matches
		if(idx == 0)
			p2ds[i] = keypoints[ matches[i].queryIdx ].pt;
		else
			p2ds[i] = keypoints[ matches[i].trainIdx ].pt;

	}
}

void PnPret2Mat4f(cv::Mat& rvec,cv::Mat& tvec,Eigen::Matrix4f& M){
	
	for(int i=0;i<3;++i)
		for(int j=0;j<3;++j)
			M(i,j) = (float)(rvec.at<double>(i,j));
	
	for(int i=0;i<3;++i)
		M(i,3) = (float)(tvec.at<double>(i,0));

	M(3,3) = 1.0f;

}

void ComputePnP(const char* folder_name,int start_idx,int total_num,int jump_num){//Compute two-view PnP
	int img_idx = start_idx;
	int img_num = total_num;
	int img_num2 = jump_num;

	char imgname[64];
	char pcdname[64];
	char imgname2[64];
	char pcdname2[64];


	ofstream out_pose,out_pose2;
	out_pose.open("RT.txt");
	out_pose2.open("invRT.txt");
 cv::Mat colorImage_1(KINECT_IMAGE_HEGIHT,KINECT_IMAGE_WIDTH,CV_8UC4);
 cv::Mat pointCloud_XYZforRGB_1(KINECT_IMAGE_HEGIHT,KINECT_IMAGE_WIDTH,CV_32FC3,cv::Scalar::all(0));

 cv::Mat colorImage_2(KINECT_IMAGE_HEGIHT,KINECT_IMAGE_WIDTH,CV_8UC4);
 cv::Mat pointCloud_XYZforRGB_2(KINECT_IMAGE_HEGIHT,KINECT_IMAGE_WIDTH,CV_32FC3,cv::Scalar::all(0));
 
 
 cv::Mat img1,img2;
  /////////////////////PCL objects//////////////////////////

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_1 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_1f (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_2f (new pcl::PointCloud<pcl::PointXYZRGBA>);

  
	pcl::visualization::PCLVisualizer vislm; 

////////////////////Find cv::Matched points between two RGB images////////////////////////
	ifstream cam_in("CamPara.txt");
	cam_in>>CamIntrinsic[0][0]>>CamIntrinsic[0][1]>>CamIntrinsic[0][2]
	>>CamIntrinsic[1][0]>>CamIntrinsic[1][1]>>CamIntrinsic[1][2]
	>>CamIntrinsic[2][0]>>CamIntrinsic[2][1]>>CamIntrinsic[2][2];

	cam_in.close();

	const cv::Mat Camera_Matrix(3,3,CV_64F,CamIntrinsic);
	const cv::Mat disCoef(1,5,CV_64F,DisCoef);

	cv::Mat img_Matches;

	int numKeyPoints = 400; 
	int numKeyPoints2 = 400; 

	RobustMatcher rMatcher; 
 

	cv::Ptr<cv::FeatureDetector> detector = new cv::FastFeatureDetector(0); 
	cv::Ptr<cv::FeatureDetector> detector2 = new cv::FastFeatureDetector(0); 
	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::OrbDescriptorExtractor(); 
	cv::Ptr<cv::DescriptorMatcher> Matcher= new cv::BFMatcher(cv::NORM_HAMMING);
	rMatcher.setFeatureDetector(detector); 
	rMatcher.setDescriptorExtractor(extractor); 
	rMatcher.setDescriptorMatcher(Matcher); 

	std::vector<cv::KeyPoint> img1_keypoints; 
	cv::Mat img1_descriptors; 
	std::vector<cv::KeyPoint> img2_keypoints; 
	cv::Mat img2_descriptors; 

	std::vector<cv::DMatch > Matches; 

 //////////////////////PCL rigid motion estimation///////////////////////////
	Eigen::Matrix4f TotaltransforMation = Eigen::Matrix4f::Identity(); 
	Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity();

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr Transcloud_2 (new pcl::PointCloud<pcl::PointXYZRGBA>);

//////////////////////////////////////////////////////////////////////////////

	printf("From %d to %d",img_idx,img_idx+(img_num-1)*img_num2);

	//Camera position
	pcl::PointCloud<pcl::PointXYZ> Camera_pose0;
	pcl::PointCloud<pcl::PointXYZ> Camera_pose;

	Camera_pose0.push_back(pcl::PointXYZ(0,0,0));
	Camera_pose0.push_back(pcl::PointXYZ(0.2,0,0));
	Camera_pose0.push_back(pcl::PointXYZ(0,0.2,0));
	Camera_pose0.push_back(pcl::PointXYZ(0,0,0.2));


	/////////////Looping///////////////////////////
	int d = 0;
	for(int i=0; i<img_num-1; ++i, d+=img_num2){

		sprintf(imgname,"%s/rgb_%d.jpg",folder_name,img_idx+d);

    	sprintf(pcdname,"%s/pcd_%d",folder_name,img_idx+d);

		sprintf(imgname2,"%s/rgb_%d.jpg",folder_name,img_idx+d+img_num2);

    	sprintf(pcdname2,"%s/pcd_%d",folder_name,img_idx+d+img_num2);

		printf("comparing %s & %s\n",imgname,imgname2);

		 ////////////Reading data/////////////////////
		 img1 = cv::imread(imgname,CV_LOAD_IMAGE_GRAYSCALE);
		 colorImage_1 = cv::imread(imgname,CV_LOAD_IMAGE_COLOR);
 
		 printf("reading\n");
		 FILE* fin = fopen(pcdname,"rb");
		 fread(pointCloud_XYZforRGB_1.data,1,pointCloud_XYZforRGB_1.step*pointCloud_XYZforRGB_1.rows,fin);
		 fclose(fin);


		 img2 = cv::imread(imgname2,CV_LOAD_IMAGE_GRAYSCALE);
		 colorImage_2 = cv::imread(imgname2,CV_LOAD_IMAGE_COLOR);


		 printf("reading\n");
		 fin = fopen(pcdname2,"rb");
		 fread(pointCloud_XYZforRGB_2.data,1,pointCloud_XYZforRGB_2.step*pointCloud_XYZforRGB_2.rows,fin);
		 fclose(fin);

		 cloud_1->clear();
		 cloud_2->clear();
		 Mat2PCL_XYZRGB_ALL(colorImage_1,pointCloud_XYZforRGB_1,*cloud_1);
		 Mat2PCL_XYZRGB_ALL(colorImage_2,pointCloud_XYZforRGB_2,*cloud_2);

        ///////////////////Finding 2D features//////////////////////////

		img1_keypoints.clear();
		img2_keypoints.clear();

		detector->detect(img1,img1_keypoints); 
		extractor->compute(img1,img1_keypoints,img1_descriptors); 
	
		detector2->detect(img2,img2_keypoints); 
		extractor->compute(img2,img2_keypoints,img2_descriptors); 

		Matches.clear();

		printf("cv::Matching\n");

		rMatcher.match(Matches, img1_keypoints, img2_keypoints,img1_descriptors,img2_descriptors); 

		//printf("drawing\n");
		drawMatches(img1, img1_keypoints, img2, img2_keypoints,Matches, img_Matches, cv::Scalar::all(-1), cv::Scalar::all(-1),vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		cv::imshow("Good matches",img_Matches);
		cv::waitKey(10);
		///////////////////Find corresponding 3D pints////////////////////////////////////////
		//We should draw point cloud from the Matches...  
		//////////////////3D geometry///////////////////////////////////////////////////////

		 ///////////////////////// 
		 //Compute PnP
		 //////////////////////// 
		cv::Mat rvec(1,3,cv::DataType<double>::type);
		cv::Mat tvec(1,3,cv::DataType<double>::type);
		cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);

		vector<cv::Point2f> p2ds;
		vector<cv::Point3f> p3ds;
		 
		Mat2PCL_XYZRGB_MATCH_PnP(pointCloud_XYZforRGB_1,p3ds,p2ds,img1_keypoints,img2_keypoints,Matches);
		printf("3D/2D points:%d,%d\n",p3ds.size(),p2ds.size());

		if(p3ds.size() > 5){
			cv::solvePnPRansac(p3ds,p2ds,Camera_Matrix,disCoef,rvec,tvec);
			cv::Rodrigues(rvec,rotationMatrix);
			PnPret2Mat4f(rotationMatrix,tvec,Ti);
			Ti = Ti.inverse();


		}
		///////////////////////Compose the translation, and transform the point cloud////////////////

		Transcloud_2->clear();


		std::cout<<"\nLocal motion from PnP is \n"<<Ti;
		TotaltransforMation = TotaltransforMation*Ti;
		pcl::transformPointCloud(*cloud_2,*Transcloud_2,TotaltransforMation);

		std::cout<<"\nTotal motion from PnP is \n"<<TotaltransforMation<<endl; 

       			//Add camera coordinate
			pcl::transformPointCloud(Camera_pose0,Camera_pose,TotaltransforMation);
			sprintf(pcdname2,"X%d",img_idx+d+img_num2);
			vislm.addLine(Camera_pose[0],Camera_pose[1],255,0,0,pcdname2);
			sprintf(pcdname2,"Y%d",img_idx+d+img_num2);
			vislm.addLine(Camera_pose[0],Camera_pose[2],0,255,0,pcdname2);
			sprintf(pcdname2,"Z%d",img_idx+d+img_num2);
			vislm.addLine(Camera_pose[0],Camera_pose[3],0,0,255,pcdname2);

			if(i==0){
				vislm.addPointCloud(cloud_1->makeShared(),pcdname);
				//PCLviewer.showCloud(cloud_1,"init");
				out_pose<<Eigen::Matrix4f::Identity()<<endl;
			}

			vislm.addPointCloud(Transcloud_2->makeShared(),pcdname2); 
			out_pose<<TotaltransforMation<<endl; 
			out_pose2<<TotaltransforMation.inverse()<<endl; 
	}
	vislm.resetCamera(); 
	vislm.spin();

	out_pose.close();
	out_pose2.close();
}

