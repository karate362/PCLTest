#include "IO_Functions.h"

using namespace std;

#define MAX_STRING_LENGTH 16000

char ReadBuf[MAX_STRING_LENGTH];


void _declspec(dllexport) StrToDoubleArray(char* raw, vector<double> &data){

	char* it = raw;

	data.clear();

	it = strtok(raw," ");

	while(*it != '\n')
	{
        data.push_back(atof(it));
		//printf("%f ",atof(it));
		it = strtok(NULL," ");

		
	}

}

void _declspec(dllexport) ReadDoubleMatrix(FILE* fin,vector<double>& reading,int& width,int& height){


	vector<double> buf;

	reading.clear();

	int dst = 0;

	height = 0;

	while( fgets(ReadBuf,MAX_STRING_LENGTH,fin) ){

		StrToDoubleArray(ReadBuf,buf);//Where's the problem??
		++height;
		width = buf.size();
		reading.resize(reading.size() + width);
		copy(buf.begin(),buf.end(),reading.begin() + dst);
		dst += width;
	}


}
