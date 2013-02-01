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

		break;
	}

  return(0);
}
