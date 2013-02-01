#include "ProjFunc.h"
#include "math.h"
#include <stdio.h>
#include <tchar.h>


 # define M_PI 3.14159265358979323846 /* pi */
 # define M_PI_2 1.57079632679489661923 /* pi/2 */
 # define M_PI_4 0.78539816339744830962 /* pi/4 */
 # define M_1_PI 0.31830988618379067154 /* 1/pi */
 # define M_2_PI 0.63661977236758134308 /* 2/pi */

void R2Euler(const double R[3][3], double &thx, double &thy, double &thz){
	double cx,sx,cy,sy,cz,sz;
	sy = -1.0 * R[2][0];
	cy = sqrt(R[2][1]*R[2][1] + R[2][2]*R[2][2]);
	thy = atan2(sy,cy);

	if(cy == 0){
		thz = 0;
		thx = M_PI - thz;
		return ;
	}

	thx = atan2(R[2][1],R[2][2]);
	thz = atan2(R[1][0],R[0][0]);

	
}

void Euler2R(double R[3][3], const double &thx, const double &thy, const double &thz){
	double cx = cos(thx),sx = sin(thx);
	double cy = cos(thy),sy = sin(thy);
	double cz = cos(thz),sz = sin(thz);

	 R[0][0] = cy*cz;
	 R[1][0] = cy*sz;
	 R[2][0] = -1.0*sy;
	 R[0][1] = sx*sy*cz - cx*sz;
	 R[1][1] = sx*sy*sz + cx*cz;
	 R[2][1] = sx*cy;
	 R[0][2] = cx*sy*cz + sx*sz;
	 R[1][2] = cx*sy*sz - sx*cz;
	 R[2][2] = cx*cy;
}


 void proj3Dto2DEuler(int j, int i, double *aj, double *bi, double *xij, void *adata){
	 //aj:x,y,z,thx,thy,thz, here the R&T depicts the inverse of global camera pose!
	 //adata = camera Intrinsic parameters,3*3 double
	 
	 double R[3][3];
	 double T[3]; 
	 double X[3];
	 double U[3];
	 double* A = (double*)adata;
	 //Euler angle to Rotation matrix
	 Euler2R(R, aj[3], aj[4], aj[5]);

	 T[0] = aj[0];
	 T[1] = aj[1];
	 T[2] = aj[2];
	 //Projection
	 X[0] = R[0][0]*bi[0] + R[0][1]*bi[1] + R[0][2]*bi[2] + T[0];
	 X[1] = R[1][0]*bi[0] + R[1][1]*bi[1] + R[1][2]*bi[2] + T[1];
	 X[2] = R[2][0]*bi[0] + R[2][1]*bi[1] + R[2][2]*bi[2] + T[2];

	 U[0] = A[0]*X[0] + A[1]*X[1] + A[2]*X[2] + T[0];
	 U[1] = A[3]*X[0] + A[4]*X[1] + A[5]*X[2] + T[1];
	 U[2] = A[6]*X[0] + A[7]*X[1] + A[8]*X[2] + T[2];
 
	 xij[0]=U[0]/U[2];
	 xij[1]=U[1]/U[2];

 }