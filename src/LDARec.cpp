#include "LDARec.h"
#include "IO_Functions.h"

const char settingpath[] = "ACT_REC";

LDARec::LDARec(int class_num,vector<string>& IDs)
{

	cnum = class_num;
	for(int i=0;i<cnum;++i)
		this->StIDs.push_back(IDs[i]);

	char str[100];

	int width,height;

	FILE *fin;

	std::printf("read LDAWcross\n");

	sprintf(str,"%s/LDAWcross",settingpath);
    fin = fopen(str,"r");
	ReadDoubleMatrix(fin,w,width,height);
	fclose(fin);

	this->Ws = cvMat(height,width,CV_64FC1,(void*)(&(w[0])) );
	printf("Ws %d,%d,%f,%f\n",width,height,cvmGet(&Ws, 0, 0),cvmGet(&Ws, 0, 1));

	std::printf("read LDAMeancross\n");

	sprintf(str,"%s/LDAMEANcross",settingpath);
	fin = fopen(str,"r");
	ReadDoubleMatrix(fin,m1,width,height);
	fclose(fin);

	this->mean1d = cvMat(height,width,CV_64FC1,(void*)(&(m1[0])) );
	printf("mean1d %d,%d,%f,%f\n",width,height,cvmGet(&mean1d, 0, 0),cvmGet(&mean1d, 0, 1));

	ydata = new double[Ws.rows];
	y = cvMat(1,Ws.rows,CV_64FC1,ydata);
}


LDARec::~LDARec(void)
{
}



void LDARec::dataNormalize(CvMat* aligndata){

}

int LDARec::ActionRecognition(double* input){//Input: 1*6 vector


	vector <double> dist;
	dist.resize(cnum);

	ydata[0] = input[0];
	ydata[1] = input[1];
	ydata[2] = input[2];
	ydata[3] = input[3];		
	ydata[4] = input[4];
	ydata[5] = input[5];	
	//Compute one-dimensionl result

	//First: only compare once!

	int count = 0;


	for(int i=0;i<cnum;++i){		
		for(int j=i+1;j<cnum;++j){

			double x1d = 0;

			for(int l=0;l<Ws.cols;++l)
				x1d += cvmGet(&y, 0, l) * cvmGet(&Ws, count, l);
		    
			
		    double d1 = fabs(x1d - cvmGet(&mean1d, count, 0));
			double d2 = fabs(x1d - cvmGet(&mean1d, count, 1));//i-th with j-th

			if(d2>d1){//
				dist[i] +=1;
				dist[j] -=1;
			}
			else{
				dist[i] -=1;
				dist[j] +=1;
			}

			++count;

		}



	}

	int midx = 0;
	double mval = dist[0];

	for(int i=0;i<dist.size();++i){
		if(dist[i]>mval)
			midx = i,mval = dist[i];
	}

	return midx;

}