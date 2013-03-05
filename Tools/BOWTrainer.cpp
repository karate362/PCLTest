//BOWTrainer.cpp : Train the Bag of word model
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/calib3d/calib3d.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/features2d/features2d.hpp"  
#include "opencv2/nonfree/nonfree.hpp"  

#include <iostream>
#include <fstream>
#include <direct.h>
#include <stdio.h>
#include <stdlib.h>
#include "Tools.h"
#include "RobustMatcher.h"
#include "DBSCAN.h"
using namespace cv;
using namespace std;

void BOFTraining(const char* input_name,const char* output_name,int ClusterNum, int kp_type,int dp_type){//Train BOF dictionary, the input specifies all training images

	BOWKMeansTrainer bowK(ClusterNum,cvTermCriteria (CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0.1),3,2);  
	
	cv::Mat img,descriptor;
	cv::Mat dictionary;
	vector<KeyPoint> Keypoints;
    string s;  
     
	cv::Ptr<cv::FeatureDetector> detector = new cv::SiftFeatureDetector();  
	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor(); 
    
	Ptr<DescriptorMatcher> Matcher = DescriptorMatcher::create( "BruteForce" );  
	vector<DMatch> matches;  

	ifstream in(input_name);
	while( getline(in,s) )
    {    
         cout << "Read from file: " << s << endl; 
		 img = cv::imread(s.c_str());
		 detector->detect(img,Keypoints);
		 extractor->compute(img,Keypoints,descriptor);
		 bowK.add(descriptor);
		 cv::drawKeypoints(img,Keypoints,img);
		 cv::imshow("keypt",img);
		 //cv::waitKey(0);
    }
	in.close();

	dictionary = bowK.cluster();

	cv::FileStorage fs(output_name, cv::FileStorage::WRITE);
	fs<<"BOW"<<dictionary;
	fs.release();

	//Draw the classifier example
	in.open(input_name);
	char fstr[64];
	while( getline(in,s) )
    {    
         cout << "Read from file: " << s << endl; 
		 img = cv::imread(s.c_str());
		 detector->detect(img,Keypoints);
		 extractor->compute(img,Keypoints,descriptor);
		 cv::drawKeypoints(img,Keypoints,img);

		 Matcher->match(descriptor,dictionary, matches);  

	    for (vector<DMatch>::iterator iter=matches.begin();iter!=matches.end();iter++)  
		{   
			Point center= Keypoints[iter->queryIdx].pt;  
			sprintf(fstr,"%d",iter->trainIdx);
			putText(img, fstr,  center, FONT_HERSHEY_SIMPLEX, 0.4 ,Scalar :: all(-1));
		}  

		 cv::imshow("keypt",img);
		 cv::waitKey(0);
    }
	in.close();
}

void BOFMatching(const char* input_name,const char* input_name2,int kp_type,int dp_type){//equals localization

	
	cv::Mat img,descriptor;
	cv::Mat dictionary;
	vector<KeyPoint> Keypoints;
    string s;  
     
	cv::Ptr<cv::FeatureDetector> detector = new cv::SiftFeatureDetector();  
	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor(); 
    
	Ptr<DescriptorMatcher> Matcher = DescriptorMatcher::create( "BruteForce" );  
	vector<DMatch> matches;  

	ifstream in(input_name);

	cv::FileStorage fs(input_name2, cv::FileStorage::READ);
	fs["BOW"]>>dictionary;
	fs.release();

	char fstr[64];
	while( getline(in,s) )
    {    
         cout << "Read from file: " << s << endl; 
		 img = cv::imread(s.c_str());
		 detector->detect(img,Keypoints);
		 extractor->compute(img,Keypoints,descriptor);
		 cv::drawKeypoints(img,Keypoints,img);

		 Matcher->match(descriptor,dictionary, matches);  

	    for (vector<DMatch>::iterator iter=matches.begin();iter!=matches.end();iter++)  
		{   
			Point center= Keypoints[iter->queryIdx].pt;  
			sprintf(fstr,"%d",iter->trainIdx);
			putText(img, fstr,  center, FONT_HERSHEY_SIMPLEX, 0.4 ,Scalar :: all(-1));
		}  

		 cv::imshow("keypt",img);
		 cv::waitKey(0);
    }
	in.close();


}

void BOFDiffMatching(const char* folder_name,int start_idx,int total_num,int jump_num,const char* dic_name,int kp_type,int dp_type){

	int img_idx = start_idx;
	int img_num = total_num;
	int img_num2 = jump_num;

	char imgname[64];
	char imgname2[64];
	char fstr[64];

	cv::Mat img1,img2,img_Matches;
	std::vector<cv::KeyPoint> img1_keypoints; 
	cv::Mat img1_descriptors; 
	std::vector<cv::KeyPoint> img2_keypoints; 
	cv::Mat img2_descriptors; 
	cv::Mat dictionary;

	std::vector<cv::KeyPoint> ret_keypoints; 

	cv::FileStorage fs(dic_name, cv::FileStorage::READ);
	fs["BOW"]>>dictionary;
	fs.release();

	std::vector<cv::DMatch > Matches, dicmatch1, dicmatch2; 



	RobustMatcher rMatcher; 
	cv::Ptr<cv::FeatureDetector> detector = new cv::FastFeatureDetector(5); 
	//cv::Ptr<cv::FeatureDetector> detector = new cv::SiftFeatureDetector(); 
	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor(); 
	//cv::Ptr<cv::FeatureDetector> detector = new cv::SurfFeatureDetector(); 
	//cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SurfDescriptorExtractor(); 
	//cv::Ptr<cv::DescriptorExtractor> extractor = new cv::OrbDescriptorExtractor(); 
	cv::Ptr<cv::DescriptorMatcher> Matcher= new cv::BFMatcher(cv::NORM_L2);

	
	//cv::Ptr<cv::DescriptorExtractor> extractor = new cv::OrbDescriptorExtractor(); 
	//cv::Ptr<cv::DescriptorMatcher> Matcher= new cv::BFMatcher(cv::NORM_HAMMING2);

	rMatcher.setFeatureDetector(detector); 
	rMatcher.setDescriptorExtractor(extractor); 
	rMatcher.setDescriptorMatcher(Matcher); 

	//Clustering
	DBSCAN dbscan;

	int d = 0;
	for(int i=0; i<img_num-1; ++i, d+=img_num2){

		sprintf(imgname,"%s/rgb_%d.jpg",folder_name,img_idx+d);

		sprintf(imgname2,"%s/rgb_%d.jpg",folder_name,img_idx+d+img_num2);

		img1 = cv::imread(imgname,CV_LOAD_IMAGE_GRAYSCALE);
		img2 = cv::imread(imgname2,CV_LOAD_IMAGE_GRAYSCALE);
	
		detector->detect(img1,img1_keypoints); 
		extractor->compute(img1,img1_keypoints,img1_descriptors); 
		Matcher->match(img1_descriptors,dictionary, dicmatch1);  
	
		detector->detect(img2,img2_keypoints); 
		extractor->compute(img2,img2_keypoints,img2_descriptors); 
		//Matcher->match(img2_descriptors,dictionary, dicmatch2);  
		
		rMatcher.match(Matches, img1_keypoints, img2_keypoints,img1_descriptors,img2_descriptors); 
		

		//////////////////Draw/////////////////////
		img1 = cv::imread(imgname);
		img2 = cv::imread(imgname2);

		ret_keypoints.clear(); 
		
		for (vector<DMatch>::iterator iter=Matches.begin();iter!=Matches.end();iter++)  //for each matched point in img1
		{   
			ret_keypoints.push_back(img1_keypoints[iter->queryIdx]);
			Point center= img1_keypoints[iter->queryIdx].pt;  

			//Find d in img1_matches
			/*
			sprintf(fstr,"%d",-1);
			for (vector<DMatch>::iterator iter1=dicmatch1.begin();iter1!=dicmatch1.end();iter1++){
				if(iter1->queryIdx == iter->queryIdx){
					sprintf(fstr,"%d",iter1->trainIdx);
					break;
				}
			}
			putText(img1, fstr,  center, FONT_HERSHEY_SIMPLEX, 0.5 ,Scalar :: all(-1));*/
		}  

		cv::drawKeypoints(img1,ret_keypoints,img1);
		//cv::drawKeypoints(img1,img1_keypoints,img1);

		//clustering
		dbscan.compute2D(ret_keypoints,15.0f,3);
		std::vector<int>& cidxs = *(dbscan.GetCidxs());
		std::vector<std::vector<int>>& clusters = *(dbscan.GetClusters());

		printf("%d clusters\n",clusters.size());
		for(int i=0;i<cidxs.size();++i){
			Point center= ret_keypoints[i].pt;
			sprintf(fstr,"%d",cidxs[i]);
			if(cidxs[i] >=0)
			putText(img1, fstr,  center, FONT_HERSHEY_SIMPLEX, 0.5 ,Scalar(255,255,0,255));
		}


		
		cv::imshow("keypt",img1);
		cv::waitKey(10);

		//Save Image
		IplImage matsave = img1;
		sprintf(imgname2,"match_%d.jpg",img_idx+d);
		cvSaveImage(imgname2,&matsave);

	}

}

