#pragma once
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <vector>


void _declspec(dllexport) ReadDoubleMatrix(FILE* fin,std::vector<double>& reading,int& width,int& height);
void _declspec(dllexport) StrToDoubleArray(char* raw, std::vector<double> &data);
