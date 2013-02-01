/**
	author @kassy708
	OpenGLとKinect SDK v1.0を使ったポイントクラウド
*/

#include<Windows.h>
#include "OpenCV2PCL.h"
#include <glut.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <string>


using namespace cv;
using namespace std;

//PCL object and viewer
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);

pcl::visualization::CloudViewer PCLviewer("Cloud Viewer");

//Color and depth images
Mat image(KINECT_IMAGE_HEGIHT,KINECT_IMAGE_WIDTH,CV_8UC4);
Mat depth(KINECT_DEPTH_HEGIHT,KINECT_DEPTH_WIDTH,CV_16UC1);
//ポイントクラウドの座標
Mat pointCloud_XYZ(KINECT_DEPTH_HEGIHT,KINECT_DEPTH_WIDTH,CV_32FC3,cv::Scalar::all(0));
Mat pointCloud_XYZforRGB(KINECT_IMAGE_HEGIHT,KINECT_IMAGE_WIDTH,CV_32FC3,cv::Scalar::all(0));

void drawPointCloud(Mat &rgbImage,Mat &pointCloud_XYZ,Mat &depthImage);		//ポイントクラウド描画
void drawPointCloud_easy(Mat &rgbImage,Mat &pointCloud_XYZ);
//openGLのための宣言・定義
//---変数宣言---
int FormWidth = 640;
int FormHeight = 480;
int mButton;
float twist, elevation, azimuth;
float cameraDistance = 0,cameraX = 0,cameraY = 0;
int xBegin, yBegin;
//---マクロ定義---
#define glFovy 45		//視角度
#define glZNear 1.0		//near面の距離
#define glZFar 150.0	//far面の距離
void polarview();		//視点変更

//ハンドル
HANDLE m_hNextImageFrameEvent;
HANDLE m_hNextDepthFrameEvent;
HANDLE m_pImageStreamHandle;
HANDLE m_pDepthStreamHandle;




//描画
void display(){
    // clear screen and depth buffer
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    // Reset the coordinate system before modifying
    glLoadIdentity(); 
    glEnable(GL_DEPTH_TEST); //「Zバッファ」を有効
    gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);   //視点の向き設定

	if(GetImage(image,m_hNextImageFrameEvent,m_pImageStreamHandle)==-1)
		return;
	if(GetImage(depth,m_hNextDepthFrameEvent,m_pDepthStreamHandle)==-1)
		return;

    //3次元ポイントクラウドのための座標変換
    //retrievePointCloudMap(depth,pointCloud_XYZ);
	retrieveRGBCloudMap(depth,pointCloud_XYZforRGB);

    //視点の変更
    polarview();

    imshow("image",image);
    imshow("depth",depth);

	//RGBAからBGRAに変換
    cvtColor(image,image,CV_RGBA2BGRA);  

    //ポイントクラウド
    //drawPointCloud(image,pointCloud_XYZ,depth);
    drawPointCloud_easy(image,pointCloud_XYZforRGB);
    

    glFlush();
    glutSwapBuffers();

	//Draw on PCL viewer

	Mat2PCL_XYZRGB_ALL(image,pointCloud_XYZforRGB,*cloud);
	PCLviewer.showCloud(cloud);
}
//初期化
int init(){	
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
// アイドル時のコールバック
void idle(){
    //再描画要求
    glutPostRedisplay();
}
//ウィンドウのサイズ変更
void reshape (int width, int height){
    FormWidth = width;
    FormHeight = height;
    glViewport (0, 0, (GLsizei)width, (GLsizei)height);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    //射影変換行列の指定
    gluPerspective (glFovy, (GLfloat)width / (GLfloat)height,glZNear,glZFar);
    glMatrixMode (GL_MODELVIEW);
}
//マウスの動き
void motion(int x, int y){
    int xDisp, yDisp;
    xDisp = x - xBegin;
    yDisp = y - yBegin;
    switch (mButton) {
    case GLUT_LEFT_BUTTON:
        azimuth += (float) xDisp/2.0;
        elevation -= (float) yDisp/2.0;
        break;
    case GLUT_MIDDLE_BUTTON:
        cameraX -= (float) xDisp/40.0;
        cameraY += (float) yDisp/40.0;
        break;
    case GLUT_RIGHT_BUTTON:
		cameraDistance += xDisp/40.0;
        break;
    }
    xBegin = x;
    yBegin = y;
}
//マウスの操作
void mouse(int button, int state, int x, int y){
    if (state == GLUT_DOWN) {
        switch(button) {
        case GLUT_RIGHT_BUTTON:
        case GLUT_MIDDLE_BUTTON:
        case GLUT_LEFT_BUTTON:
            mButton = button;
            break;
        }
        xBegin = x;
        yBegin = y;
    }
}
//視点変更
void polarview(){
    glTranslatef( cameraX, cameraY, cameraDistance);
    glRotatef( -twist, 0.0, 0.0, 1.0);
    glRotatef( -elevation, 1.0, 0.0, 0.0);
    glRotatef( -azimuth, 0.0, 1.0, 0.0);
}
//メイン
int main(int argc, char *argv[]){

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(FormWidth, FormHeight);
    glutCreateWindow(argv[0]);
    //コールバック
    glutReshapeFunc (reshape);
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    init();
    glutMainLoop();
	
	NuiShutdown();
    return 0;
}


//ポイントクラウド描画
//Need to calibrate the difference between RGB and depth image
void drawPointCloud(Mat &rgbImage,Mat &pointCloud_XYZ,Mat &depthImage){
	static int x,y;
	glPointSize(2);
	glBegin(GL_POINTS);
	uchar *p;
	Point3f *point = (Point3f*)pointCloud_XYZ.data;
	LONG colorX,colorY;

	for(y = 0;y < pointCloud_XYZ.rows;y++){
		for(x = 0;x < pointCloud_XYZ.cols;x++,point++){
			if(point->z == 0)
				continue;

			USHORT pBufferRun = depthImage.at<USHORT>(y,x);

			                        // 真实深度
            int  RealDepth = (pBufferRun & 0xfff8) >> 3;
 
            // RGB和深度对齐，跳过无效点
            if (RealDepth==0)
                   continue;


//Correcting the difference between RGB and D image
#if KINECT_DEPTH_WIDTH == 320
			//奥行き画像の解像度が320x240の場合
			NuiImageGetColorPixelCoordinatesFromDepthPixel(NUI_IMAGE_RESOLUTION_640x480,NULL,x,y,pBufferRun,&colorX,&colorY);
#else if KINECT_DEPTH_WIDTH == 640
			//奥行き画像の解像度が640x480の場合
			NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(NUI_IMAGE_RESOLUTION_640x480,NUI_IMAGE_RESOLUTION_640x480,NULL,x,y,pBufferRun,&colorX,&colorY);
#endif
			//画像内の場合
			if(0 <= colorX && colorX < rgbImage.cols && 0 <= colorY && colorY < rgbImage.rows){
				p = &rgbImage.data[colorY * rgbImage.step + colorX * rgbImage.channels()];
				glColor3ubv(p);
				glVertex3f(point->x,point->y,point->z);
			}			

		}
	}
	glEnd();
}



void drawPointCloud_easy(Mat &rgbImage,Mat &pointCloud_XYZforRGB){
	static int x,y;
	glPointSize(2);
	glBegin(GL_POINTS);
	uchar *p;
	Point3f *point = (Point3f*)pointCloud_XYZforRGB.data;
	LONG colorX,colorY;

	for(y = 0;y < pointCloud_XYZforRGB.rows;y++){
		for(x = 0;x < pointCloud_XYZforRGB.cols;x++,point++){
			if(point->z == 0)
				continue;

			if(0 <= x && x < rgbImage.cols && 0 <= y && y < rgbImage.rows){
				p = &rgbImage.data[y * rgbImage.step + x * rgbImage.channels()];
				glColor3ubv(p);
				glVertex3f(point->x,point->y,point->z);
			}			

		}
	}
	glEnd();
}
