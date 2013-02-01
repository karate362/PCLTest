#include "OpenCV2PCL.h"

using namespace cv;

Point3f * minimum_depth_point(cv::Mat &pointCloud_XYZ,int x,int y,int range){
	Point3f *rp = 0;
	
	int i1 = x-range;
	int i2 = x+range;
	int j1 = y-range;
	int j2 = y+range;

	float z_min = FLT_MAX;

	for(int i = i1;i<=i2;++i)
		for(int j = j1;j<=j2;++j){//for
			/////if in the image range/////
			if(i>=0 && i<pointCloud_XYZ.rows && j>=0 && j<pointCloud_XYZ.cols){
				Point3f *point =(Point3f *)( pointCloud_XYZ.data + j * pointCloud_XYZ.step + i * pointCloud_XYZ.channels()*4 );
				if(point->z > 0 && point->z < z_min){
					rp = point;
					z_min = point->z;
				}
			}
		}//for

	return rp;
}

//Keypoints should be matched and all have depth value
void Mat2PCL_XYZRGB_MATCH(cv::Mat &color,cv::Mat &color2,cv::Mat &pointCloud_XYZ,cv::Mat &pointCloud_XYZ2,pcl::PointCloud<pcl::PointXYZRGBA>& cloud,pcl::PointCloud<pcl::PointXYZRGBA>& cloud2,std::vector<cv::KeyPoint>& keypoints,std::vector<cv::KeyPoint>& keypoints2,std::vector<cv::DMatch > &matches){
	
	cv::Point2f p2d_1,p2d_2;
	Point3f *p3d_1,*p3d_2; 
    pcl::PointXYZRGBA p;
	cloud.clear();
	cloud2.clear();
	for( int i = 0; i < matches.size(); i++ ){
		//-- Get the keypoints from the good matches
		p2d_1 = keypoints[ matches[i].queryIdx ].pt;
		p2d_2 = keypoints2[ matches[i].trainIdx ].pt;
		p3d_1 = minimum_depth_point(pointCloud_XYZ,(int)(p2d_1.x),(int)(p2d_1.y),4);
		p3d_2 = minimum_depth_point(pointCloud_XYZ2,(int)(p2d_2.x),(int)(p2d_2.y),4);	

		if(p3d_1!=0 && p3d_2!=0){
			if(p3d_1->z>0 && p3d_2->z>0){
				p.x = p3d_1->x;
				p.y = p3d_1->y;
				p.z = p3d_1->z;
				cloud.push_back(p);
				p.x = p3d_2->x;
				p.y = p3d_2->y;
				p.z = p3d_2->z;
				cloud2.push_back(p);
			}
		}

	}

}

void Mat2PCL_XYZRGB_MATCH_PnP(cv::Mat &pointCloud_XYZ,vector<cv::Point3f>& p3ds,vector<cv::Point2f>& p2ds,std::vector<cv::KeyPoint>& keypoints,std::vector<cv::KeyPoint>& keypoints2,std::vector<cv::DMatch >& matches){
	
	cv::Point2f p2d_1,p2d_2;
	Point3f *p3d_1,*p3d_2; 
	cv::Point3f p;
	p2ds.clear();
	p3ds.clear();

	for( int i = 0; i < matches.size(); i++ ){
		//-- Get the keypoints from the good matches
		p2d_1 = keypoints[ matches[i].queryIdx ].pt;
		p2d_2 = keypoints2[ matches[i].trainIdx ].pt;
		p3d_1 = minimum_depth_point(pointCloud_XYZ,(int)(p2d_1.x),(int)(p2d_1.y),4);

		if(p3d_1!=0){
			if(p3d_1->z>0){
				p.x = p3d_1->x;
				p.y = p3d_1->y;
				p.z = p3d_1->z;
				p3ds.push_back(p);

				//Notice!  because the RGB image if flipped from Kinect!!
				//The coordinate of Kinect is different from openCV
				//p2d_2.x = 640 - p2d_2.x;
				p2d_2.y = 480.0f - p2d_2.y;
				p2ds.push_back(p2d_2);
			}
		}

	}

}



void Mat2PCL_XYZRGB_ALL(cv::Mat &color, cv::Mat &pointCloud_XYZ, pcl::PointCloud<pcl::PointXYZRGBA>& cloud){
	std::vector<cv::KeyPoint> keypoints;

	Point3f *point = (Point3f*)pointCloud_XYZ.data;
	cv::KeyPoint kp;
	for(int y = 0;y < pointCloud_XYZ.rows;y++)
		for(int x = 0;x < pointCloud_XYZ.cols;x++,point++){
			if(point->z > 0){
				kp.pt.x = (float)x;
				kp.pt.y = (float)y;
				keypoints.push_back(kp);
			}
		}
		
	Mat2PCL_XYZRGB(color,pointCloud_XYZ, cloud, keypoints);

}


//Keypoints should be matched and all have depth value
/*
void Mat2PCL_XYZRGB_MATCH(int idx,cv::Mat &color,cv::Mat &pointCloud_XYZ,pcl::PointCloud<pcl::PointXYZRGBA>& cloud,std::vector<cv::KeyPoint>& keypoints,std::vector<cv::DMatch > matches){//Only for cloud 1
	
	cv::Point2f p2d_1;
	Point3f *p3d_1; 
    pcl::PointXYZRGBA p;
	cloud.clear();

	for( int i = 0; i < matches.size(); i++ ){
		//-- Get the keypoints from the good matches
		if(idx == 0)
			p2d_1 = keypoints[ matches[i].queryIdx ].pt;
		else
			p2d_1 = keypoints[ matches[i].trainIdx ].pt;
		p3d_1 = minimum_depth_point(pointCloud_XYZ,(int)(p2d_1.x),(int)(p2d_1.y),4);

		if(p3d_1!=0){
			if(p3d_1->z>0){
				p.x = p3d_1->x;
				p.y = p3d_1->y;
				p.z = p3d_1->z;
				cloud.push_back(p);
			}
		}

	}

}
*/

void Mat2PCL_XYZRGB(cv::Mat &color,cv::Mat &pointCloud_XYZ,pcl::PointCloud<pcl::PointXYZRGBA>& cloud,std::vector<cv::KeyPoint>& keypoints,int range ){
		
	int s = keypoints.size();
	//if(cloud.size() != s)
		//cloud.resize(s);
	cloud.clear();
	pcl::PointXYZRGBA p;
	//cloud is 32F...
	Point3f *point = 0;

	for(int i=0;i<s;++i){
		KeyPoint &kp = keypoints[i];

		if(range<=0)
			point =(Point3f *)( pointCloud_XYZ.data + (int)(kp.pt.y) * pointCloud_XYZ.step + (int)(kp.pt.x) * pointCloud_XYZ.channels()*4 );
		else
			point = minimum_depth_point(pointCloud_XYZ, (int)(kp.pt.x),(int)(kp.pt.y), range);

		if(!point)
			continue;

		p.x = point->x;
		p.y = point->y;
		p.z = point->z;
		cloud.push_back(p);
		
		uchar *cp = &color.data[ (int)(kp.pt.y) * color.step + (int)(kp.pt.x) * color.channels() ];
	    p.b = *(cp);
		p.g = *(cp+1);
		p.r = *(cp+2);
		p.a = *(cp+3);
	}
}


int GetImage(cv::Mat &image,HANDLE frameEvent,HANDLE streamHandle){
	//フレームを入れるクラス
	const NUI_IMAGE_FRAME *pImageFrame = NULL;
	
	//次のRGBフレームが来るまで待機
	WaitForSingleObject(frameEvent,INFINITE);
	HRESULT hr = NuiImageStreamGetNextFrame(streamHandle, 30 , &pImageFrame );
	if( FAILED( hr ) ) 
		return -1;//取得失敗

	//フレームから画像データの取得
	INuiFrameTexture * pTexture = pImageFrame->pFrameTexture;
	NUI_LOCKED_RECT LockedRect;
	pTexture->LockRect( 0, &LockedRect, NULL, 0 );
    
	if( LockedRect.Pitch != 0 ){
		//pBitsに画像データが入っている
		BYTE *pBuffer = (BYTE*) LockedRect.pBits;
		memcpy(image.data,pBuffer,image.step * image.rows);
	}

	hr = NuiImageStreamReleaseFrame( streamHandle, pImageFrame );
	if( FAILED( hr ) ) 
		return -1;//取得失敗

	return 0;
}


//3次元ポイントクラウドのための座標変換
void retrievePointCloudMap(Mat &depth,Mat &pointCloud_XYZ){
    unsigned short* dp = (unsigned short*)depth.data;
	Point3f *point = (Point3f *)pointCloud_XYZ.data;
	for(int y = 0;y < depth.rows;y++){
		for(int x = 0;x < depth.cols;x++, dp++,point++){
#if KINECT_DEPTH_WIDTH == 320
			//奥行き画像の解像度が320x240の場合
			Vector4 RealPoints = NuiTransformDepthImageToSkeleton(x,y,*dp);
#else if KINECT_DEPTH_WIDTH == 640
			//奥行き画像の解像度が640x480の場合
			Vector4 RealPoints = NuiTransformDepthImageToSkeleton(x,y,*dp, NUI_IMAGE_RESOLUTION_640x480);
#endif

			
			point->x = RealPoints.x;
			point->y = RealPoints.y;
			point->z = RealPoints.z;
		}
	}
}

//3次元ポイントクラウドのための座標変換
void retrieveRGBCloudMap(Mat &depth,Mat &pointCloud_XYZ){

	memset(pointCloud_XYZ.data,0x00,pointCloud_XYZ.step * pointCloud_XYZ.rows);

	LONG colorX,colorY;

    unsigned short* dp = (unsigned short*)depth.data;
	Point3f *point = (Point3f *)pointCloud_XYZ.data;

	for(int y = 0;y < depth.rows;y++){
		for(int x = 0;x < depth.cols;x++, dp++){


#if KINECT_DEPTH_WIDTH == 320
			//奥行き画像の解像度が320x240の場合
			Vector4 RealPoints = NuiTransformDepthImageToSkeleton(x,y,*dp);//real position of corresponding depth pixel
			NuiImageGetColorPixelCoordinatesFromDepthPixel(NUI_IMAGE_RESOLUTION_640x480,NULL,x,y,*dp,&colorX,&colorY);//Get corresponding pixel on RGB image
#else if KINECT_DEPTH_WIDTH == 640
			//奥行き画像の解像度が640x480の場合
			Vector4 RealPoints = NuiTransformDepthImageToSkeleton(x,y,*dp, NUI_IMAGE_RESOLUTION_640x480);
			NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(NUI_IMAGE_RESOLUTION_640x480,NUI_IMAGE_RESOLUTION_640x480,NULL,x,y,*dp,&colorX,&colorY);
#endif

				
			if(0 <= colorX && colorX < pointCloud_XYZ.cols && 0 <= colorY && colorY < pointCloud_XYZ.rows){
				point = (Point3f *)( pointCloud_XYZ.data + colorY*pointCloud_XYZ.step + colorX*pointCloud_XYZ.channels()*4 );//because it is 32F, not 8U
				point->x = RealPoints.x;
				point->y = RealPoints.y;
				point->z = RealPoints.z;
			}
		}
	}
}