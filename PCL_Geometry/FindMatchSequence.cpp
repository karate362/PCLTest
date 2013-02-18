#include "OpenCV2PCL.h"
#include "RobustMatcher.h"
#include "Geometry_Functions.h"
using namespace std;

void FindMatchSeq(const char* folder_name,int ref_idx,int start_idx,int total_num,int jump_num,double epidist,double confidence,bool is_save){//Compute two-view PnP
	int img_idx = start_idx;
	int img_num = total_num;
	int img_num2 = jump_num;

	char imgname[64];
	char pcdname[64];
	char imgname2[64];
	char pcdname2[64];
 
 cv::Mat img1,img2;
 cv::Mat pointCloud_XYZforRGB_1(KINECT_IMAGE_HEGIHT,KINECT_IMAGE_WIDTH,CV_32FC3,cv::Scalar::all(0));
////////////////////Find cv::Matched points between two RGB images////////////////////////
	
	cv::Mat img_Matches;

	int numKeyPoints = 400; 
	int numKeyPoints2 = 400; 

	RobustMatcher rMatcher; 
 
	//ORB
	//cv::Ptr<cv::FeatureDetector> detector = new cv::OrbFeatureDetector(numKeyPoints); 
	//cv::Ptr<cv::FeatureDetector> detector2 = new cv::OrbFeatureDetector(numKeyPoints2); 
	//cv::Ptr<cv::DescriptorExtractor> extractor = new cv::OrbDescriptorExtractor(); 
	
	//Fast + ORB
	//cv::Ptr<cv::FeatureDetector> detector = new cv::FastFeatureDetector(0); 
	//cv::Ptr<cv::FeatureDetector> detector2 = new cv::FastFeatureDetector(0); 
	//cv::Ptr<cv::DescriptorExtractor> extractor = new cv::OrbDescriptorExtractor(); 
	
	//SIFT
	cv::Ptr<cv::FeatureDetector> detector = new cv::SiftFeatureDetector(); 
	cv::Ptr<cv::FeatureDetector> detector2 = new cv::SiftFeatureDetector(); 
	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor(); 

	//BRISK
//	cv::Ptr<cv::FeatureDetector> detector = new cv::BRISK(0,3,1.2); 
//	cv::Ptr<cv::FeatureDetector> detector2 = new cv::BRISK(0,3,1.2); 
//	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::BRISK(0,3,1.2); 

	//FREAK
//	cv::Ptr<cv::FeatureDetector> detector = new cv::ORB(); 
//	cv::Ptr<cv::FeatureDetector> detector2 = new cv::ORB(); 
//	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::FREAK(); 

	cv::Ptr<cv::DescriptorMatcher> Matcher= new cv::BFMatcher(cv::NORM_L2);
	//cv::Ptr<cv::DescriptorMatcher> Matcher= new cv::BFMatcher(cv::NORM_HAMMING2);

	
	rMatcher.setFeatureDetector(detector); 
	rMatcher.setDescriptorExtractor(extractor); 
	rMatcher.setDescriptorMatcher(Matcher); 
	rMatcher.setConfidenceLevel(confidence);
	rMatcher.setMinDistanceToEpipolar(epidist);

	std::vector<cv::KeyPoint> img1_keypoints; 
	cv::Mat img1_descriptors; 
	std::vector<cv::KeyPoint> img2_keypoints; 
	cv::Mat img2_descriptors; 

	std::vector<cv::DMatch > Matches; 

//////////////////////////////////////////////////////////////////////////////

	printf("From %d to %d\n",img_idx,img_idx+(img_num-1)*img_num2);



	sprintf(imgname,"%s/rgb_%d.jpg",folder_name,ref_idx);
	img1 = cv::imread(imgname,CV_LOAD_IMAGE_GRAYSCALE);
	detector->detect(img1,img1_keypoints); 
    extractor->compute(img1,img1_keypoints,img1_descriptors); 

	cv::imshow("New Image",img1);

	printf("reading\n");
	sprintf(pcdname,"%s/pcd_%d",folder_name,ref_idx);
	FILE* fin = fopen(pcdname,"rb");
	fread(pointCloud_XYZforRGB_1.data,1,pointCloud_XYZforRGB_1.step*pointCloud_XYZforRGB_1.rows,fin);
	fclose(fin);


	ofstream out_1,out_2;
	out_1.open("feature3D.txt");
	out_2.open("measure2D.txt");

	vector<cv::Point3f> p3d_f1;
	cv::Point3f np;

	for(int i=0;i<img1_keypoints.size();++i){
		cv::Point3f *rp = minimum_depth_point(pointCloud_XYZforRGB_1,(int)(img1_keypoints[i].pt.x),(int)(img1_keypoints[i].pt.y),3);
		if(!rp){
			np.x = -1;
			np.y = -1;
			np.z = -1;
		}
		else{
			np.x = rp->x;
			np.y = rp->y;
			np.z = rp->z;
		}
		out_1<<np.x<<" "<<np.y<<" "<<np.z<<endl;
		p3d_f1.push_back(np);
	}
	out_1.close();

	/////////////Looping///////////////////////////
	int d = 0;
	for(int i=0; i<img_num; ++i, d+=img_num2){

		sprintf(imgname2,"%s/rgb_%d.jpg",folder_name,img_idx+d);

		printf("comparing %s & %s\n",imgname,imgname2);

		 ////////////Reading data/////////////////////


		 img2 = cv::imread(imgname2,CV_LOAD_IMAGE_GRAYSCALE);

        ///////////////////Finding 2D features//////////////////////////

		img2_keypoints.clear();
	
		detector2->detect(img2,img2_keypoints); 
		extractor->compute(img2,img2_keypoints,img2_descriptors); 

		printf("%d features\n",img2_keypoints.size());

		Matches.clear();

		printf("cv::Matching\n");

		rMatcher.match(Matches, img1_keypoints, img2_keypoints,img1_descriptors,img2_descriptors); 
		//Matcher->match(img1_descriptors,img2_descriptors,Matches); 
		printf("drawing\n");
		drawMatches(img1, img1_keypoints, img2, img2_keypoints,Matches, img_Matches, cv::Scalar::all(-1), cv::Scalar::all(-1),vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		
		for(int j=0;j<Matches.size();++j){
			cv::Point2f &pt = img2_keypoints[ Matches[j].trainIdx ].pt;
			if( p3d_f1[Matches[j].queryIdx].z > 0 )
			  out_2<<pt.x<<" "<<pt.y<<" "<<Matches[j].queryIdx<<" "<<i<<endl;
		}
		
		cv::imshow("Good matches",img_Matches);
		
		if(is_save){
			IplImage matsave = img_Matches;
			sprintf(imgname2,"match_%d.jpg",img_idx+d);
			cvSaveImage(imgname2,&matsave);
		}
		//cv::waitKey(0);

	}

	out_2.close();

}
