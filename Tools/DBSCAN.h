#pragma once
#include <vector>
#include "opencv/cv.h"

class DBSCAN
{
public:
	DBSCAN(void);
	~DBSCAN(void);

	void compute2D(const std::vector<cv::KeyPoint>& points,double eps,int minPts);

	//void compute2D(std::vector<cv::Point3f>& points,double eps,int minPts);

	void compute3D(const std::vector<cv::Point3f>& points,double eps,int minPts);

	std::vector<int>* GetCidxs(){
		return &(Cidxs);
	}

	std::vector<std::vector<int>>* GetClusters(){
		return &(clusters);
	}

private:
	
	void expandCluster(const std::vector<cv::Point3f>& D,const int Pi, std::vector<int>& neighbor, const int C, double eps,int minPts);//C is the cluster index
	void regionQuery(const std::vector<cv::Point3f>& D,const cv::Point3f& P, double eps,std::vector<int>& neighbor);//return all points within P's eps-neighborhood

	void ClusterJoint(std::vector<int> &C1, std::vector<int>& C2, std::vector<bool>& inC1);


	std::vector<int> Cidxs;//same size as D, depicts the cluster number...
	std::vector<bool> visited;
	std::vector<std::vector<int>> clusters;
};

