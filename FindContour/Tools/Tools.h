#pragma once

int FindContour( int argc, char** argv );

void BOFTraining(const char* folder_name,int start_idx,int total_num,int jump_num);//Train BOF dictionary

void BOFMatching();//equals localization