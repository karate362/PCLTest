#include "OpenCV2PCL.h"
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <fstream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <string>
#include "RobustMatcher.h"
#include "Geometry_Functions.h"
#include <vector>
using namespace std;

extern double CamIntrinsic[3][3];
extern double DisCoef[5];
void ReadMatrix4f(Eigen::Matrix4f &M, ifstream &in);


void RT3D:: ReadFromFile(ifstream &fin){
		double dummy[4];
		fin>>R[0][0]>>R[0][1]>>R[0][2]>>T[0]
		>>R[1][0]>>R[1][1]>>R[1][2]>>T[1]
		>>R[2][0]>>R[2][1]>>R[2][2]>>T[2]
		>>dummy[0]>>dummy[1]>>dummy[2]>>dummy[3];
	}

void RT3D::ReadFromMatrix4f(Eigen::Matrix4f &M){
		R[0][0] = M(0,0);
		R[0][1] = M(0,1);
		R[0][2] = M(0,2);
		T[0] = M(0,3);

		R[1][0] = M(1,0);
		R[1][1] = M(1,1);
		R[1][2] = M(1,2);
		T[1] = M(1,3);

		R[2][0] = M(2,0);
		R[2][1] = M(2,1);
		R[2][2] = M(2,2);
		T[2] = M(2,3);
	}

struct FeaPoint{
	//position in the image sequence
	int img;
	float u;
	float v;
	//position in 3D coordinate, it can be assigned in the last stage
	int dep;//idx of the depth image
	double x;
	double y;
	double z;

	//if the 3D position is available
	bool is_3D;
	//Count number
	int count;
};

struct MatchUpdateData{
	int qidx;
	int tidx;
	int* qpar;//idx of qidx feature in FPoints, -1 if none
	int* tpar;//idx of tidx feature in FPoints, -1 if none
	int qimg;
	int timg;
	cv::KeyPoint qkp;

};


int UpdateFeapoint(std::vector<FeaPoint>& FPoints,void* data){
//update FPoints for only one matching pair
//return the related FPoint idx

	//get necessary data
	MatchUpdateData* MUdata = (MatchUpdateData*)data;
	int fidx = -1;
	int qidx = MUdata->qidx;
	int tidx = MUdata->tidx;
	int &qpar = *(MUdata->qpar);
	int &tpar = *(MUdata->tpar);
	int img = MUdata->qimg;
	float u = MUdata->qkp.pt.x;
	float v = MUdata->qkp.pt.y;

	int situ = 0;

	if(qpar>=0 && tpar<0)//one is old, one is new
		situ = 0;
	else
	if(qpar<0 && tpar>=0)//one is old, one is new
		situ = 1;
	else
	if(qpar<0 && tpar<0) //both are new feature
		situ = 2;
	else//both are old feature
	{
		if(qpar == tpar)
			situ = 3;
		else
			situ = 4;
	}

	//TODO: 5 situations, o depicts that the parent feature exists

	switch(situ){

	case 0:
	// qidx o, tidx x: increase feature count, tidx.parent = qidx.parent
		tpar = qpar;
		++FPoints[qpar].count;
		fidx = qpar;
		break;

	case 1:
	// qidx x, tidx o: increase feature count, qidx.parent = tidx.parent
		qpar = tpar;
		++FPoints[tpar].count;
		fidx = tpar;
		break;

	case 2:
	// qidx x, tidx x: insert new feature and set their parents
		FeaPoint fp;
		fp.count = 2;
		fp.img = img;
		fp.u = u;
		fp.v = v;
		fp.is_3D = false;
		qpar = (int)FPoints.size();
		tpar = qpar;
		FPoints.push_back(fp);
		fidx = qpar;
		break;

	case 3:
	// qidx o, tidx o and their parents are the same: do nothing
		fidx = qpar;
		break;

	case 4:
	// qidx o, tidx o and their parents are different: do nothing
		fidx = qpar;
		break;
	}

	return fidx;
}


//Find features from the 2D images
void FindMapFeatures(const char* folder_name,int start_idx,int total_num,int jump_num,vector<FeaPoint>& FPoints){
	
	int img_idx = start_idx;
	int img_num = total_num;
	int img_num2 = jump_num;

	char imgname[64];
	char pcdname[64];

	ifstream in_pose;

	in_pose.open("RT.txt");
	
 cv::Mat img1;
 cv::Mat pt3D_1(KINECT_IMAGE_HEGIHT,KINECT_IMAGE_WIDTH,CV_32FC3,cv::Scalar::all(0));

 vector< vector<cv::KeyPoint> > keypoints;
 vector< cv::Mat > descriptors;
 vector< vector<int> > Fparents;
 vector<RT3D> RTs;
 keypoints.resize(total_num);
 descriptors.resize(total_num);
 Fparents.resize(total_num);
 RTs.resize(total_num);
//////////////////////////////////////////////////////////////////////////////

	std::printf("From %d to %d\n",img_idx,img_idx+(img_num-1)*img_num2);

	/////////////Reading images and extract features///////////////////////////
	cv::Ptr<cv::FeatureDetector> detector = new cv::FastFeatureDetector(0); 
	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::OrbDescriptorExtractor(); 
	cv::Ptr<cv::DescriptorMatcher> Matcher= new cv::BFMatcher(cv::NORM_HAMMING2);

	RobustMatcher rMatcher; 
	rMatcher.setFeatureDetector(detector); 
	rMatcher.setDescriptorExtractor(extractor); 
	rMatcher.setDescriptorMatcher(Matcher); 

	int d = 0;
	for(int i=0; i<img_num-1; ++i, d+=img_num2){

		sprintf(imgname,"%s/rgb_%d.jpg",folder_name,img_idx+d);

		 ////////////Reading data and extract/////////////////////
		 img1 = cv::imread(imgname,CV_LOAD_IMAGE_GRAYSCALE);
		 detector->detect(img1,keypoints[i]); 
         extractor->compute(img1,keypoints[i],descriptors[i]);
		 
		 std::printf("%s: %d features\n",imgname,keypoints[i].size());

		 Fparents[i].resize(keypoints[i].size());
		 for(int j=0;j<Fparents[i].size();++j)//Initialize to -1
			 (Fparents[i])[j] = -1;
		 //////////Reading RT//////////////////////////////////////
		 RTs[i].ReadFromFile(in_pose);
	}
		in_pose.close();



	std::vector<cv::DMatch > Matches; 
	MatchUpdateData MUdata;
	/////////////Find matchings and update feature points///////////////////////////	
	for(int j1=0; j1<img_num-2; ++j1){

			//Load query Depth data			
			sprintf(pcdname,"%s/pcd_%d",folder_name,img_idx + img_num2*j1);
			std::printf("reading %s\n",pcdname);

			FILE* fin = std::fopen(pcdname,"rb");
			std::fread(pt3D_1.data,1,pt3D_1.step*pt3D_1.rows,fin);
			std::fclose(fin);

		for(int j2=j1+1; j2<img_num-1; ++j2){

			//Find matches between two images
			
			Matches.clear();
			rMatcher.match(Matches, keypoints[j1],keypoints[j2], descriptors[j1],descriptors[j2]); 
			std::printf("matching %d to %d, %d matches\n",j1,j2,Matches.size());

				for(int k=0;k<Matches.size();++k){//Find useful features
					int &qidx = Matches[k].queryIdx;
					int &tidx = Matches[k].trainIdx;
					MUdata.qidx = qidx;
					MUdata.tidx = tidx;
					MUdata.qpar = &( Fparents[j1][qidx] );
					MUdata.tpar = &( Fparents[j2][tidx] ); 
					MUdata.qimg = j1;
					MUdata.qkp = keypoints[j1][qidx];
					MUdata.timg = j2;
					int fidx = UpdateFeapoint(FPoints,&MUdata);
					//Update the Feature point's 3D position...
					//only update when the query image has depth value
					FeaPoint &fp = FPoints[fidx];
					if(!fp.is_3D){//Find from pt_3D-1
						cv::Point2f& p2d_1 = keypoints[j1][qidx].pt;
						cv::Point3f *p3d_1 = minimum_depth_point(pt3D_1,(int)(p2d_1.x),(int)(p2d_1.y),4);
						if(p3d_1!=0){
							fp.x = (double)p3d_1->x;
							fp.y = (double)p3d_1->y;
							fp.z = (double)p3d_1->z;
							fp.dep = j1;
							fp.is_3D = true;
						}
					}//Find from pt_3D-1
				}//Find useful features
			}
		}

		//Check if there are 3D points and transform to global coordinate
		for(int i=0;i<FPoints.size();++i){
			FeaPoint &fp = FPoints[i];
			if(fp.is_3D)
				RTs[fp.dep].Transform3D(fp.x,fp.y,fp.z);
		}
}



void DrawFeatureMap(const char* folder_name,int start_idx,int total_num,int jump_num,std::vector<FeaPoint>& FPoints){
	
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
 
  /////////////////////PCL objects//////////////////////////

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_1 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr Transcloud_2 (new pcl::PointCloud<pcl::PointXYZRGBA>);



pcl::visualization::PCLVisualizer vislm; 

 //////////////////////PCL rigid motion estimation///////////////////////////
	Eigen::Matrix4f TotaltransforMation = Eigen::Matrix4f::Identity(); 

//////////////////////////////////////////////////////////////////////////////

	std::printf("From %d to %d\n",img_idx,img_idx+(img_num-1)*img_num2);


	/////////////Looping///////////////////////////
	int d = 0;
	for(int i=0; i<img_num-1; ++i, d+=img_num2){

		sprintf(imgname,"%s/rgb_%d.jpg",folder_name,img_idx+d);

    	sprintf(pcdname,"%s/pcd_%d",folder_name,img_idx+d);

		 ////////////Reading data/////////////////////
		 colorImage_1 = cv::imread(imgname,CV_LOAD_IMAGE_COLOR);
 
		 printf("reading %s\n",pcdname);
		 FILE* fin = fopen(pcdname,"rb");
		 fread(pointCloud_XYZforRGB_1.data,1,pointCloud_XYZforRGB_1.step*pointCloud_XYZforRGB_1.rows,fin);
		 fclose(fin);

		 cloud_1->clear();

		 Mat2PCL_XYZRGB_ALL(colorImage_1,pointCloud_XYZforRGB_1,*cloud_1);

		 ReadMatrix4f(TotaltransforMation,in_pose);//Read_RT

		 pcl::transformPointCloud(*cloud_1,*Transcloud_2,TotaltransforMation);

		std::cout<<"\nTotal motion from lm is \n"<<TotaltransforMation<<endl; 

		vislm.addPointCloud(Transcloud_2->makeShared(),pcdname); 

	}

	//Draw Feature points
	char fstr[32];
	for(int i=0;i<FPoints.size();++i){
		FeaPoint &fp = FPoints[i];

		if(fp.count>2 && fp.is_3D){
			sprintf(fstr,"%d",i);
			vislm.addSphere(pcl::PointXYZ(fp.x,fp.y,fp.z),0.05,0,0,255,fstr);//blue points as features
		}

	}


	vislm.addLine(pcl::PointXYZ(0,0,0),pcl::PointXYZ(1,0,0),255,0,0,"Xaxis");
	vislm.addLine(pcl::PointXYZ(0,0,0),pcl::PointXYZ(0,1,0),0,255,0,"Yaxis");
	vislm.addLine(pcl::PointXYZ(0,0,0),pcl::PointXYZ(0,0,1),0,0,255,"Zaxis");
	vislm.resetCamera(); 
	vislm.spin();

	in_pose.close();

}


void FindFeatureAndDraw(const char* folder_name,int start_idx,int total_num,int jump_num){


	vector<FeaPoint> FPoints;

	FindMapFeatures(folder_name, start_idx, total_num, jump_num, FPoints);

	std::printf("%d features\n",FPoints.size());

	for(int i=0;i<FPoints.size();++i){
		FeaPoint &fp = FPoints[i];
		if(fp.is_3D && fp.count >=1 )
			std::printf("fidx: %d at img: %d, (%.2f,%.2f,%.2f)\n",i,fp.img,fp.x,fp.y,fp.z);
	}

	DrawFeatureMap(folder_name, start_idx, total_num, jump_num, FPoints);
}


void FPoint2PnPdata(vector<FeaPoint> &FPoints, vector<cv::Point2f> &p2ds, vector<cv::Point3f> &p3ds){

}

void ReadFeaPointFromFile(ifstream &in,vector<FeaPoint>& FPoints){

	FeaPoint fp;

	FPoints.clear();

	while(1){
		in>>fp.dep>>fp.img>>fp.u>>fp.v;
		fp.is_3D = false;
		if(in.eof())
			break;
		FPoints.push_back(fp);
		std::printf("Reading: %d,%d,%f,%f\n",fp.dep,fp.img,fp.u,fp.v);
	}
}

void PnPLocalization(const char* folder_name,int start_idx,int total_num,int jump_num){
	//Same input format as map drawing function
	//Here we use "dep" as "fidx", "img" as :img_idx"
	//Additional input: PnPmeasures: [fidx][img_idx][u][v]
	//Additional input: PnPfeatures: [fidx][img_idx][u][v]

	//TODO: create a FeaPoint array, here the feature index is saved in "img" attribute
	vector<FeaPoint> PnPmeasures;
	vector<FeaPoint> PnPfeatures;

	//Notice that PnPmeasure should be 0,1,2...
	ifstream in_measure;
	in_measure.open("PnPmeasures.txt");
	ReadFeaPointFromFile(in_measure,PnPmeasures);
	in_measure.close();

	ifstream in_feature;
	in_feature.open("PnPfeatures.txt");
	ReadFeaPointFromFile(in_feature,PnPfeatures);
	in_feature.close();

	int input_img_idx = PnPmeasures[0].img;

	//TODO: Read the corresponding point cloud, and examine all corresponding measurements...

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
 
  /////////////////////PCL objects//////////////////////////

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_1 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr Transcloud_2 (new pcl::PointCloud<pcl::PointXYZRGBA>);

  pcl::PointCloud<pcl::PointXYZ>::Ptr Feature3D (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr TransFeature3D (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::visualization::PCLVisualizer vislm; 

 //////////////////////PCL rigid motion estimation///////////////////////////
	Eigen::Matrix4f TotaltransforMation = Eigen::Matrix4f::Identity(); 

//////////////////////////////////////////////////////////////////////////////

	std::printf("From %d to %d\n",img_idx,img_idx+(img_num-1)*img_num2);


//////////////////PnP matching data//////////////////////////

	RT3D RT3d;
/////////////////////////////////////////////////////////////


	/////////////Looping///////////////////////////
	int d = 0;
	for(int i=0; i<img_num-1; ++i, d+=img_num2){

		sprintf(imgname,"%s/rgb_%d.jpg",folder_name,img_idx+d);

    	sprintf(pcdname,"%s/pcd_%d",folder_name,img_idx+d);

		 ////////////Reading data/////////////////////
		 colorImage_1 = cv::imread(imgname,CV_LOAD_IMAGE_COLOR);
 
		 printf("reading %s\n",pcdname);
		 FILE* fin = fopen(pcdname,"rb");
		 fread(pointCloud_XYZforRGB_1.data,1,pointCloud_XYZforRGB_1.step*pointCloud_XYZforRGB_1.rows,fin);
		 fclose(fin);

		 cloud_1->clear();

		 Mat2PCL_XYZRGB_ALL(colorImage_1,pointCloud_XYZforRGB_1,*cloud_1);

		 ReadMatrix4f(TotaltransforMation,in_pose);

		pcl::transformPointCloud(*cloud_1,*Transcloud_2,TotaltransforMation);

		std::cout<<"\nTotal motion from lm is \n"<<TotaltransforMation<<endl; 

		vislm.addPointCloud(Transcloud_2->makeShared(),pcdname); 

////////////////////////////////////////PnP data process/////////////////////////////////////////////////
		//Examine the FPoint array, find if there are 3D measures in this sequence
		RT3d.ReadFromMatrix4f(TotaltransforMation);

		for(int j=0;j<PnPfeatures.size();++j){
			FeaPoint &fp = PnPfeatures[j];

			if(fp.img != img_idx+d)//same image
				continue;

			FeaPoint &fm = PnPmeasures[fp.dep];//corresponding point in the measurement image

			//update 3D point
			if(fm.is_3D)
				continue;

			cv::Point3f *p3d_1 = minimum_depth_point(pointCloud_XYZforRGB_1,(int)fp.u,(int)fp.v,3);
			if(p3d_1!=0){
					fm.x = (double)p3d_1->x;
					fm.y = (double)p3d_1->y;
					fm.z = (double)p3d_1->z;
					fm.is_3D = true;
					RT3d.Transform3D(fm.x,fm.y,fm.z);
			}//update 3D point

		}
////////////////////////////////////////PnP data process/////////////////////////////////////////////////
	}

	in_pose.close();


	///////////////////////////////////Compute PnP//////////////////////////////////////////////
	Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(); 

	ifstream cam_in("CamPara.txt");
	cam_in>>CamIntrinsic[0][0]>>CamIntrinsic[0][1]>>CamIntrinsic[0][2]
	>>CamIntrinsic[1][0]>>CamIntrinsic[1][1]>>CamIntrinsic[1][2]
	>>CamIntrinsic[2][0]>>CamIntrinsic[2][1]>>CamIntrinsic[2][2];

	cam_in.close();

	const cv::Mat Camera_Matrix(3,3,CV_64F,CamIntrinsic);
	const cv::Mat disCoef(1,5,CV_64F,DisCoef);

	cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);
	cv::Mat rvec(1,3,cv::DataType<double>::type);
	cv::Mat tvec(1,3,cv::DataType<double>::type);


	vector<cv::Point2f> p2ds;
	vector<cv::Point3f> p3ds;
	cv::Point2f p2d;
	cv::Point3f p3d;

	for(int i=0;i<PnPmeasures.size();++i){
		FeaPoint &fm = PnPmeasures[i];
		if(!fm.is_3D)
			continue;
		else
			std::printf("Fea %d: %f,%f,%f\n ",fm.dep,fm.x,fm.y,fm.z);

		p2d.x = (float)fm.u;
		p2d.y = 480.0f - (float)fm.v;
		p3d.x = (float)fm.x;
		p3d.y = (float)fm.y;
		p3d.z = (float)fm.z;
		p2ds.push_back(p2d);
		p3ds.push_back(p3d);
			if(p3ds.size() > 4){
				cv::solvePnPRansac(p3ds,p2ds,Camera_Matrix,disCoef,rvec,tvec);
				cv::Rodrigues(rvec,rotationMatrix);
				PnPret2Mat4f(rotationMatrix,tvec,Ti);
				Ti = Ti.inverse();
				std::cout<<"\nTotal motion from PnP is \n"<<Ti<<endl; 
			}
	}
	///////////////////////////////////Compute PnP//////////////////////////////////////////////



	/////////////////////////////Draw PnP result/////////////////////////////

	sprintf(imgname,"%s/rgb_%d.jpg",folder_name,input_img_idx);
    sprintf(pcdname,"%s/pcd_%d",folder_name,input_img_idx);

	////////////Reading data/////////////////////
    colorImage_1 = cv::imread(imgname,CV_LOAD_IMAGE_COLOR);
 
	printf("reading %s\n",pcdname);
	FILE* fin = fopen(pcdname,"rb");
    fread(pointCloud_XYZforRGB_1.data,1,pointCloud_XYZforRGB_1.step*pointCloud_XYZforRGB_1.rows,fin);
	fclose(fin);

	cloud_1->clear();


	Mat2PCL_XYZRGB_ALL(colorImage_1,pointCloud_XYZforRGB_1,*cloud_1);

	pcl::transformPointCloud(*cloud_1,*Transcloud_2,Ti);

	vislm.addPointCloud(Transcloud_2->makeShared(),pcdname); 
	/////////////////////////////////////////////////////////////////////////


	vislm.addLine(pcl::PointXYZ(0,0,0),pcl::PointXYZ(1,0,0),255,0,0,"Xaxis");
	vislm.addLine(pcl::PointXYZ(0,0,0),pcl::PointXYZ(0,1,0),0,255,0,"Yaxis");
	vislm.addLine(pcl::PointXYZ(0,0,0),pcl::PointXYZ(0,0,1),0,0,255,"Zaxis");
	vislm.resetCamera(); 
	vislm.spin();

	

}
