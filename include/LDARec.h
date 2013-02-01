#pragma once

#include<String>
#include<vector>
#include<fstream>
#include"cv.h"
using namespace std;

class LDARec
{
public:
	LDARec(int class_num,vector<string>& IDs);
	~LDARec(void);


public:

	int ActionRecognition(double* input);

private:

	void dataNormalize(CvMat* aligndata);

	int dim1;
	int dim2;
	int cnum;

	CvMat Ws;
	CvMat mean1d;

	CvMat y;

	vector <double> w;
	vector <double> co;
	vector <double> m1;
	vector <double> am;

	double* ydata;//input data

	vector <string> StIDs;
};

