// Tools.cpp : Defines the entry point for the console application.
//
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "Tools.h"

using namespace cv;
using namespace std;

void printMatrix(cv::Mat M){
cout<<"Type:"<<M.type()<<endl;
 // dont print empty matrices
  if (M.empty()){
    cout << "---" << endl;
    return;
  }
  // loop through columns and rows of the matrix
  for(int i=0; i < M.rows; i++){
      for(int j=0; j < M.cols ; j++){
      cout << M.at<double>(i,j) << ", ";
      }
    cout<<endl;
}
  }

void TestFunction(){//For testing something

	double CamInt[3][3] = {{531.15f, 0.0f, 320.0f}, {0.0f, 531.15f, 240.0f}, {0.0f, 0.0f, 1.0f}};
	double InvCamInt[3][3] = {{531.15f, 0.0f, 320.0f}, {0.0f, 531.15f, 240.0f}, {0.0f, 0.0f, 1.0f}};
	cv::Mat Camera_Matrix(3,3,CV_64F,CamInt);

	cv::Mat InvCam(3,3,CV_64F,InvCamInt); 
	InvCam = Camera_Matrix.inv();
	printMatrix(Camera_Matrix);

	printMatrix(InvCam);

	cout<<InvCamInt[0][0]<<endl;
	//If we have set the memory address... It will only operate on it!
}


/** @function main */
int main( int argc, char** argv )
{
	char folder_name[256];
	int img_idx;
	int img_num,img_num2;
	int Func = 0;
	bool save = 0;

	if(argc <2)
	{
		printf( "First parameter: 0:FindContour,1:Train BOF,2: Test BOF\n");
		return 0;
	}
	else
		Func = atoi(argv[1]);


	switch(Func){

	case 0:
			FindContour( argc, argv+1 );
		break;

	case 1://Train BOW
		BOFTraining(argv[2],argv[3],atoi(argv[4]),0,0);//Train BOF dictionary, the input specifies all training images
		break;

	case 2://Test BOW
		BOFMatching(argv[2],argv[3],0,0);//imgname & dictionary
		break;

	case 3:
		BOFDiffMatching(argv[2],atoi(argv[3]),atoi(argv[4]),atoi(argv[5]),argv[6],0,0);
		break;

	case 100://Test function
		TestFunction();
		break;
	}

  return(0);
}
