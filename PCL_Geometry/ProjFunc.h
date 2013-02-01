#pragma once


void R2Euler(const double R[3][3], double &thx, double &thy, double &thz);

void Euler2R(double R[3][3], const double &thx, const double &thy, const double &thz);

void proj3Dto2DEuler(int j, int i, double *aj, double *bi, double *xij, void *adata);