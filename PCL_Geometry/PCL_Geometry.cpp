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
//using namespace cv;
int user_data;



int 
main (int argc, const char ** argv)
{
	char folder_name[256];
	int img_idx;
	int img_num,img_num2;
	int app_idx,icp_itnum;
	double icp_threshold  = 0.05;
	double epidist = 3.0;
	double conf = 0.99;
	int Func = 0;
	bool save = 0;

	if(argc <2)
	{
		printf( "First parameter: 0:Viewer,1:Multi-view RGBD odometry,2: PnP algorithm\n");
		return 0;
	}
	else
		Func = atoi(argv[1]);


	switch(Func){

	case 0:
		if ( argc-2 < 3 ) /* argc should be 2 for correct execution */
		{
			/* We print argv[0] assuming it is the program name */
			printf( "usage: %s %d foldname, start index, total image number, jump number\n", argv[0],Func );
			return 0;
		}
		else 
		{
			sprintf(folder_name,"%s",argv[2]);

			img_idx = atoi(argv[3]);

			img_num = atoi(argv[4]);//total image number

			img_num2 = atoi(argv[5]);//space of image reading
		}

		ViewGeometry(folder_name,img_idx,img_num,img_num2);
		break;

	case 1:
		if ( argc-2 < 7 ) /* argc should be 2 for correct execution */
		{
			/* We print argv[0] assuming it is the program name */
			printf( "usage: %s %d foldname, start index, total image number, jump number, approach index, ICP iteration, ICP threshold\n", argv[0],Func );
			return 0;
		}
		else 
		{
			sprintf(folder_name,"%s",argv[2]);

			img_idx = atoi(argv[3]);

			img_num = atoi(argv[4]);//total image number

			img_num2 = atoi(argv[5]);//space of image reading

			app_idx = atoi(argv[6]);//0: pure ICP, 1: pure feature matching, 2: mixed method

			if(app_idx == 1)
				icp_itnum = 0;
			else
				icp_itnum = atoi(argv[7]);//ICP iteration number
		
			icp_threshold = atof(argv[8]);
		}

		Compute_Geometry(folder_name,img_idx,img_num,img_num2, app_idx, icp_itnum, icp_threshold);

		break;

		case 2:
			if ( argc-2 < 3 ) /* argc should be 2 for correct execution */
			{
				/* We print argv[0] assuming it is the program name */
				printf( "usage: %s %d foldname, start index, total image number, jump number\n", argv[0],Func );
				return 0;
			}
			else 
			{
				sprintf(folder_name,"%s",argv[2]);

				img_idx = atoi(argv[3]);

				img_num = atoi(argv[4]);//total image number

				img_num2 = atoi(argv[5]);//space of image reading
			}

			ComputePnP(folder_name,img_idx,img_num,img_num2);

		break;

		case 3://Generate local map localization data: saved in feature.txt, (x,y,feature_idx,img_idx)
			if ( argc-2 < 4 ) /* argc should be 2 for correct execution */
			{
				/* We print argv[0] assuming it is the program name */
				printf( "usage: %s %d foldname, ref_idx  start index, total image number, jump number\n", argv[0],Func );
				return 0;
			}
			else 
			{
				sprintf(folder_name,"%s",argv[2]);

				img_idx = atoi(argv[4]);

				img_num = atoi(argv[5]);//total image number

				img_num2 = atoi(argv[6]);//space of image reading

				epidist = atof(argv[7]);//total image number

				conf = atof(argv[8]);//space of image reading

				save = atoi(argv[9]);//space of image reading
			}
			FindMatchSeq(folder_name,atoi(argv[3]),img_idx,img_num,img_num2,epidist,conf,save);
		break;


	   case 4:
		if ( argc-2 < 3 ) /* argc should be 2 for correct execution */
		{
			/* We print argv[0] assuming it is the program name */
			printf( "usage: %s %d foldname, start index, total image number, jump number\n", argv[0],Func );
			return 0;
		}
		else 
		{
			sprintf(folder_name,"%s",argv[2]);

			img_idx = atoi(argv[3]);

			img_num = atoi(argv[4]);//total image number

			img_num2 = atoi(argv[5]);//space of image reading
		}

		FindFeatureAndDraw(folder_name,img_idx,img_num,img_num2);
		break;

		case 5:
		if ( argc-2 < 3 ) /* argc should be 2 for correct execution */
		{
			/* We print argv[0] assuming it is the program name */
			printf( "usage: %s %d foldname, start index, total image number, jump number\n", argv[0],Func );
			return 0;
		}
		else 
		{
			sprintf(folder_name,"%s",argv[2]);

			img_idx = atoi(argv[3]);

			img_num = atoi(argv[4]);//total image number

			img_num2 = atoi(argv[5]);//space of image reading
		}

		PnPLocalization(folder_name,img_idx,img_num,img_num2);
		break;


		case 6://Generate Training data
		if ( argc-2 < 3 ) /* argc should be 2 for correct execution */
			{
				/* We print argv[0] assuming it is the program name */
				printf( "usage: %s %d foldname, start index, total image number, jump number\n", argv[0],Func );
				return 0;
			}
			else 
			{

				img_idx = atoi(argv[2]);//class

				img_num = atoi(argv[3]);//t1

				img_num2 = atoi(argv[4]);//t2
			}
			GenerateActionTrainData(img_num,img_num2,img_idx);
		break;

		case 8://Action recognition
		
			if ( argc-2 < 3 ) /* argc should be 2 for correct execution */
			{
				/* We print argv[0] assuming it is the program name */
				printf( "usage: %s %d foldname, start index, total image number, jump number\n", argv[0],Func );
				return 0;
			}
			else 
			{
				sprintf(folder_name,"%s",argv[2]);

				img_idx = atoi(argv[3]);

				img_num = atoi(argv[4]);//total image number

				img_num2 = atoi(argv[5]);//space of image reading
			}
			ActionRecognition(folder_name,img_idx,img_num,img_num2);
		break;

	}


	int i;
	cin>>i;
    return 0;
}
