#include<Windows.h>
#include "OpenCV2PCL.h"
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include <string>
#include <time.h>
#include <direct.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace cv;

HANDLE m_hNextImageFrameEvent;
HANDLE m_hNextDepthFrameEvent;
HANDLE m_pImageStreamHandle;
HANDLE m_pDepthStreamHandle;

int Kinect_init();


int main(int argc, char *argv[])
{
	int numArgs = argc;

	//First parameter: If to save file
	//Second parameter: ??

	if(numArgs < 2){
		printf("insufficient parameters\n");
		exit(0);
	}

	int num = atoi(argv[1]);//If to save image
	
  bool if_save = false;

  if(num == 1)
		if_save = true;
  //////////////////Use time as folder name
  cv::FileStorage	cvfs;
  char foldername[64];
  char imgname[64];

  time_t rawtime;
  struct tm * timeinfo;
  if(if_save){
	  time ( &rawtime );
	  timeinfo = localtime ( &rawtime );
	  strftime(foldername,64,".\\%H%M%S_%y%m%d",timeinfo);
	  printf("%s",foldername);
	  _mkdir(foldername);
	  sprintf(imgname,"%s/PCD.xml",foldername);
	  cvfs.open(imgname,CV_STORAGE_WRITE);
  }
  /////////////////////PCL objects//////////////////////////
  pcl::visualization::CloudViewer PCLviewer("Cloud Viewer");
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
  ////////////////OpenCV objects///////////////////////////
	Mat colorImage(KINECT_IMAGE_HEGIHT,KINECT_IMAGE_WIDTH,CV_8UC4);
	Mat depthImage(KINECT_DEPTH_HEGIHT,KINECT_DEPTH_WIDTH,CV_16UC1);
	Mat pointCloud_XYZforRGB(KINECT_IMAGE_HEGIHT,KINECT_IMAGE_WIDTH,CV_32FC3,cv::Scalar::all(0));

	namedWindow("colorImage", CV_WINDOW_AUTOSIZE);
	namedWindow("depthImage", CV_WINDOW_AUTOSIZE);

	int count = 0;


	Kinect_init();

    while (1) 
    { 
		if(GetImage(colorImage,m_hNextImageFrameEvent,m_pImageStreamHandle)==-1)
			continue;
		if(GetImage(depthImage,m_hNextDepthFrameEvent,m_pDepthStreamHandle)==-1)
			continue;

		retrieveRGBCloudMap(depthImage,pointCloud_XYZforRGB);

		Mat2PCL_XYZRGB_ALL(colorImage,pointCloud_XYZforRGB,*cloud);
		PCLviewer.showCloud(cloud);

        imshow("colorImage", colorImage); 
        imshow("depthImage", depthImage);  


		/////////Save Image//////////////////
		if(if_save){

			sprintf(imgname,"%s/rgb_%d.jpg",foldername,count);
			imwrite(imgname,colorImage);
			sprintf(imgname,"%s/pcd_%d",foldername,count);
			FILE* fout = fopen(imgname,"wb");
			fwrite(pointCloud_XYZforRGB.data,1,pointCloud_XYZforRGB.step*pointCloud_XYZforRGB.rows,fout);
			fclose(fout);
			count++;
		}
		////////////////////////////////////
 
        if(cvWaitKey(100)==27) 
            break; 
    } 
 
    NuiShutdown(); 
    return 0; 
}


//初期化
int Kinect_init(){	
	//Kinectの初期化関数
	NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX | NUI_INITIALIZE_FLAG_USES_SKELETON );

	//各ハンドルの設定
	m_hNextImageFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
	m_pImageStreamHandle   = NULL;
	m_hNextDepthFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
	m_pDepthStreamHandle   = NULL;

	//深度センサストリームの設定
	HRESULT hr;
	hr = NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR , NUI_IMAGE_RESOLUTION_640x480 , 0 , 2 , m_hNextImageFrameEvent , &m_pImageStreamHandle );
	if( FAILED( hr ) ) 
		return -1;//取得失敗
#if KINECT_DEPTH_WIDTH == 320
	//奥行き画像の解像度が320x240の場合
	hr = NuiImageStreamOpen(NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX , NUI_IMAGE_RESOLUTION_320x240 , 0 , 2 , m_hNextDepthFrameEvent , &m_pDepthStreamHandle );
#else if KINECT_DEPTH_WIDTH == 640
	//奥行き画像の解像度が640x480の場合
	hr = NuiImageStreamOpen(NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX , NUI_IMAGE_RESOLUTION_640x480 , 0 , 2 , m_hNextDepthFrameEvent , &m_pDepthStreamHandle );
#endif
	if( FAILED( hr ) ) 
		return -1;//取得失敗

	return 1;
}