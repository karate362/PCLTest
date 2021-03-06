#pragma once

int FindContour( int argc, char** argv );

//void Find2Dfeature(const char* folder_name,int start_idx,int total_num,int jump_num,int det_type,bool if_save);

void BOFTraining(const char* input_name,const char* output_name,int cluster_num,int kp_type,int dp_type);//Train BOF dictionary, the input specifies all training images

void BOFMatching(const char* input_name,const char* input_name2,int kp_type,int dp_type);//equals localization

void BOFDiffMatching(const char* folder_name,int start_idx,int total_num,int jump_num,const char* dic_name,int kp_type,int dp_type);//equals localization

void DiffMatching(int patchsize1, int patchsize2, const char* folder_name,int start_idx,int total_num,int jump_num,const char* dic_name,int kp_type,int dp_type,bool is_save);//equals localization

void FarMatchUsingKP(const char* folder_name,int start_idx,int total_num,int jump_num,bool is_save,int kp_type,int de_type,const char* ref_name,const char* out_name);

void FarMatchUsingRobustKP(const char* folder_name,int start_idx,int total_num,int jump_num,bool is_save,int kp_type,int de_type,const char* ref_name,const char* out_name);
