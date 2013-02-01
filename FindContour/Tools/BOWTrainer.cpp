//BOWTrainer.cpp : Train the Bag of word model
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/calib3d/calib3d.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/features2d/features2d.hpp"  
#include "opencv2/nonfree/nonfree.hpp"  

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "Tools.h"


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