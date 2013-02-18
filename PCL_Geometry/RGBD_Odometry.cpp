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
#include "ProjFunc.h"

using namespace std;

extern double CamIntrinsic[3][3];
extern double InvCam[3][3];
extern double DisCoef[5];

extern class RT3D;


void Compute_Geometry (const char* folder_name,int start_idx,int total_num,int jump_num,const int app_idx,int icp_itnum,double icp_threshold)
{
	int img_idx = start_idx;
	int img_num = total_num;
	int img_num2 = jump_num;

	char imgname[64];
	char pcdname[64];
	char imgname2[64];
	char pcdname2[64];

	ifstream cam_in("CamPara.txt");
	cam_in>>CamIntrinsic[0][0]>>CamIntrinsic[0][1]>>CamIntrinsic[0][2]
	>>CamIntrinsic[1][0]>>CamIntrinsic[1][1]>>CamIntrinsic[1][2]
	>>CamIntrinsic[2][0]>>CamIntrinsic[2][1]>>CamIntrinsic[2][2];

	cam_in.close();

	const cv::Mat Camera_Matrix(3,3,CV_64F,CamIntrinsic);
	const cv::Mat disCoef(1,5,CV_64F,DisCoef);
	cv::Mat Inv_Camera_Matrix(3,3,CV_64F,InvCam);
	Inv_Camera_Matrix = Camera_Matrix.inv();

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
	
	cv::Mat img_Matches;

	int numKeyPoints = 400; 
	int numKeyPoints2 = 400; 

	RobustMatcher rMatcher; 
 

	//cv::Ptr<cv::FeatureDetector> detector = new cv::OrbFeatureDetector(numKeyPoints); 
	//cv::Ptr<cv::FeatureDetector> detector2 = new cv::OrbFeatureDetector(numKeyPoints2); 
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

	vector<cv::Point3f> MatchedFeaPoints;
	vector<int> pidxs;

 //////////////////////PCL rigid motion estimation///////////////////////////
	RT3D RT;
	Eigen::Matrix4f TotaltransforMation = Eigen::Matrix4f::Identity(); 
	Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity();

	pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
	icp.setRANSACOutlierRejectionThreshold(icp_threshold);

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr Transcloud_2 (new pcl::PointCloud<pcl::PointXYZRGBA>);

    pcl::registration::TransformationEstimationLM<pcl::PointXYZRGBA,pcl::PointXYZRGBA> transLM;

	std::vector<int> indices_src,indices_tgt;

//////////////////////////////////////////////////////////////////////////////

	printf("From %d to %d",img_idx,img_idx+(img_num-1)*img_num2);

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
		Eigen::Matrix4f transforMationLM = Eigen::Matrix4f::Identity();

		if(app_idx == 1 || app_idx == 2){
			
			//We should draw point cloud from the Matches...  
			Mat2PCL_XYZRGB_MATCH(colorImage_1,colorImage_2,pointCloud_XYZforRGB_1,pointCloud_XYZforRGB_2,*cloud_1f,*cloud_2f,img1_keypoints,img2_keypoints,Matches);
			//////////////////3D geometry///////////////////////////////////////////////////////

			 ///////////////////////// 
			 //transform LM 
			 //////////////////////// 
		
			indices_src.clear();
			indices_tgt.clear();

			for(int i=0;i<cloud_1f->size();++i){
				indices_src.push_back(i);
				indices_tgt.push_back(i);
			}
			printf("size of match: %d,%d\n",cloud_1f->size(),cloud_2f->size());

			if(cloud_1f->size() >= 10){
				transLM.estimateRigidTransformation(*cloud_2f,indices_tgt,*cloud_1f,indices_src,transforMationLM);

				std::cout<<"\ntransforcv::Motion from lm is \n"<<transforMationLM<<endl; 
			}
			else{//without enough 2D features
				std::cout<<"Too few features! using pure ICP\n"<<endl; 
			}
		}

		Eigen::Matrix4f transforMationPnP = Eigen::Matrix4f::Identity();

		if(app_idx==3){

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
				PnPret2Mat4f(rotationMatrix,tvec,transforMationPnP);	
			}

			transforMationPnP = transforMationPnP.inverse();
		}
		///////////////////////Compose the translation, and transform the point cloud////////////////

		Transcloud_2->clear();

		////////////////////////////ICP///////////////////////////////////////////////////

		if(app_idx == 0)//Pure ICP
			Ti = Eigen::Matrix4f::Identity();
		else//Pure feature or Mixed
			if(app_idx==1 || app_idx==2)
				Ti = transforMationLM;
		else
			if(app_idx == 3)
				Ti = transforMationPnP;


		icp.setInputTarget(cloud_1);

		pcl::transformPointCloud(*cloud_2,*Transcloud_2,Ti);//Initial guess
		
		for(int i=0;i<icp_itnum;++i){
			icp.setInputCloud(Transcloud_2);
		
			icp.align(*Transcloud_2);
			printf("num:%d",Transcloud_2->size());
			//accumulate transformation between each Iteration
			Ti = icp.getFinalTransformation () * Ti;

		}
		std::cout<<"\nTotal motion from ICP is \n"<<Ti;
		TotaltransforMation = TotaltransforMation*Ti;
		pcl::transformPointCloud(*cloud_2,*Transcloud_2,TotaltransforMation);

		std::cout<<"\nTotal motion from lm is \n"<<TotaltransforMation<<endl; 

       
			if(i==0){
				vislm.addPointCloud(cloud_1->makeShared(),pcdname);
				//PCLviewer.showCloud(cloud_1,"init");
				out_pose<<Eigen::Matrix4f::Identity()<<endl;
			}

			vislm.addPointCloud(Transcloud_2->makeShared(),pcdname2); 
			out_pose<<TotaltransforMation<<endl; 
			out_pose2<<TotaltransforMation.inverse()<<endl; 

			//Draw points
			printf("Draw 3D points\n");
			char fstr[32];
			Ti = Ti.inverse();
			RT.ReadFromMatrix4f( Ti );
			Reconstruct3D(RT,Matches,img1_keypoints,img2_keypoints,pidxs,MatchedFeaPoints);
			for(int i=0;i<MatchedFeaPoints.size(); ++i){
				sprintf(fstr,"%d",i);
				cv::Point3f &fp = MatchedFeaPoints[i];
				vislm.addSphere(pcl::PointXYZ(fp.x,fp.y,fp.z),0.05,0,0,255,fstr);
			}

	}
	vislm.resetCamera(); 
	vislm.spin();

	out_pose.close();
	out_pose2.close();
}

void ReadMatrix4f(Eigen::Matrix4f &M, ifstream &in){
	in>>M(0,0)>>M(0,1)>>M(0,2)>>M(0,3)
		>>M(1,0)>>M(1,1)>>M(1,2)>>M(1,3)
		>>M(2,0)>>M(2,1)>>M(2,2)>>M(2,3)
		>>M(3,0)>>M(3,1)>>M(3,2)>>M(3,3);
}

void ViewGeometry(const char* folder_name,int start_idx,int total_num,int jump_num){
	
	int img_idx = start_idx;
	int img_num = total_num;
	int img_num2 = jump_num;

	char imgname[64];
	char pcdname[64];
	char imgname2[64];
	char pcdname2[64];

	ifstream in_pose;

	in_pose.open("RT.txt");

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


 //////////////////////PCL rigid motion estimation///////////////////////////
	Eigen::Matrix4f TotaltransforMation = Eigen::Matrix4f::Identity(); 
	Eigen::Matrix4f LastTotaltransforMation = Eigen::Matrix4f::Identity(); 
	Eigen::Matrix4f Deltatrans = Eigen::Matrix4f::Identity(); 
	ReadMatrix4f(TotaltransforMation,in_pose);
	RT3D deltaRT;
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


		 //Compute delta...
		 
		 double thx,thy,thz;

		 LastTotaltransforMation = TotaltransforMation;
		 ReadMatrix4f(TotaltransforMation,in_pose);
		 Deltatrans = LastTotaltransforMation.inverse()*TotaltransforMation;
		 deltaRT.ReadFromMatrix4f(Deltatrans);
		 R2Euler(deltaRT.R,thx,thy,thz);

		 std::printf("Delta motion (x,y,z,tx,ty,tz): %f,%f,%f,%f,%f,%f\n",deltaRT.T[0],deltaRT.T[1],deltaRT.T[2],thx,thy,thz);
		 
       
			if(i==0){
				vislm.addPointCloud(cloud_1->makeShared(),pcdname);
			}
			else{
				pcl::transformPointCloud(*cloud_2,*Transcloud_2,TotaltransforMation);
				vislm.addPointCloud(Transcloud_2->makeShared(),pcdname2); 
			}
			//Add camera coordinate
			pcl::transformPointCloud(Camera_pose0,Camera_pose,TotaltransforMation);
			sprintf(pcdname2,"X%d",img_idx+d+img_num2);
			vislm.addLine(Camera_pose[0],Camera_pose[1],255,0,0,pcdname2);
			sprintf(pcdname2,"Y%d",img_idx+d+img_num2);
			vislm.addLine(Camera_pose[0],Camera_pose[2],0,255,0,pcdname2);
			sprintf(pcdname2,"Z%d",img_idx+d+img_num2);
			vislm.addLine(Camera_pose[0],Camera_pose[3],0,0,255,pcdname2);
	}

	vislm.addLine(pcl::PointXYZ(0,0,0),pcl::PointXYZ(1,0,0),255,0,0,"Xaxis");
	vislm.addLine(pcl::PointXYZ(0,0,0),pcl::PointXYZ(0,1,0),0,255,0,"Yaxis");
	vislm.addLine(pcl::PointXYZ(0,0,0),pcl::PointXYZ(0,0,1),0,0,255,"Zaxis");
	vislm.resetCamera(); 
	vislm.spin();

	in_pose.close();

}


void GenerateActionTrainData(int t1,int t2,int cidx){
	

	char imgname[64];
	char pcdname[64];

	ifstream in_pose;

	in_pose.open("RT.txt");
  
pcl::visualization::PCLVisualizer vislm; 


 //////////////////////PCL rigid motion estimation///////////////////////////
	Eigen::Matrix4f TotaltransforMation = Eigen::Matrix4f::Identity(); 
	Eigen::Matrix4f LastTotaltransforMation = Eigen::Matrix4f::Identity(); 
	Eigen::Matrix4f Deltatrans = Eigen::Matrix4f::Identity(); 
	ReadMatrix4f(TotaltransforMation,in_pose);
	RT3D deltaRT;
	vector<Eigen::Matrix4f> RTs;
	vector<RT3D> deltaRTs;
//////////////////////////////////////////////////////////////////////////////

	//Camera position
	pcl::PointCloud<pcl::PointXYZ> Camera_pose0;
	pcl::PointCloud<pcl::PointXYZ> Camera_pose;

	Camera_pose0.push_back(pcl::PointXYZ(0,0,0));
	Camera_pose0.push_back(pcl::PointXYZ(0.2,0,0));
	Camera_pose0.push_back(pcl::PointXYZ(0,0.2,0));
	Camera_pose0.push_back(pcl::PointXYZ(0,0,0.2));
	/////////////Looping///////////////////////////
	int d = 0;

	//Read data
	while(1){
		 //Compute delta..
		 ReadMatrix4f(TotaltransforMation,in_pose);

		 if(in_pose.eof())
			 break;
		 else
			 RTs.push_back(TotaltransforMation);

		 deltaRT.ReadFromMatrix4f(Deltatrans);
		
			//Add camera coordinate
			pcl::transformPointCloud(Camera_pose0,Camera_pose,TotaltransforMation);
			sprintf(pcdname,"X%d",d);
			vislm.addLine(Camera_pose[0],Camera_pose[1],255,0,0,pcdname);
			sprintf(pcdname,"Y%d",d);
			vislm.addLine(Camera_pose[0],Camera_pose[2],0,255,0,pcdname);
			sprintf(pcdname,"Z%d",d);
			vislm.addLine(Camera_pose[0],Camera_pose[3],0,0,255,pcdname);
			++d;
	}

	in_pose.close();

	//Process data: generate delta RT
	
	for(int i=0;i<RTs.size();++i){
		for(int t=t1; t<=t2 ; ++t){//From i-t1 ~ i-t2
			if(i-t < 0)
				break;
			Deltatrans = RTs[i-t].inverse()*RTs[i];
			deltaRT.ReadFromMatrix4f(Deltatrans);
			deltaRTs.push_back(deltaRT);
		}
	}

	//Output
	ofstream out_euler;
	out_euler.open("DeltaEuler.txt");

	for(int i=0; i<deltaRTs.size();++i){
		double thx,thy,thz;

		R2Euler(deltaRTs[i].R,thx,thy,thz);

		out_euler<<cidx<<" "<<deltaRTs[i].T[0]<<" "<<deltaRTs[i].T[1]<<" "<<deltaRTs[i].T[2]<<" "<<thx<<" "<<thy<<" "<<thz<<endl;
	}
	out_euler.close();

	vislm.addLine(pcl::PointXYZ(0,0,0),pcl::PointXYZ(1,0,0),255,0,0,"Xaxis");
	vislm.addLine(pcl::PointXYZ(0,0,0),pcl::PointXYZ(0,1,0),0,255,0,"Yaxis");
	vislm.addLine(pcl::PointXYZ(0,0,0),pcl::PointXYZ(0,0,1),0,0,255,"Zaxis");
	vislm.resetCamera(); 
	vislm.spin();
}