#pragma once

int FindContour( int argc, char** argv );

void BOFTraining(const char* input_name,const char* output_name,int cluster_num,int kp_type,int dp_type);//Train BOF dictionary, the input specifies all training images

void BOFMatching();//equals localization