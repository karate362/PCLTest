#include "DBSCAN.h"
using namespace std;
using namespace cv;

DBSCAN::DBSCAN(void)
{
}


DBSCAN::~DBSCAN(void)
{
}

inline double dist(const cv::Point3f& P1,const cv::Point3f& P2){
	double dx = P1.x - P2.x;
	double dy = P1.y - P2.y;
	double dz = P1.z - P2.z;

	return sqrt( dx*dx + dy*dy + dz*dz );
}

void DBSCAN:: compute2D(const std::vector<cv::KeyPoint>& points,double eps,int minPts){

	std::vector <cv::Point3f> D;
	D.resize(points.size());
	for(int i=0;i<D.size();++i){
		D[i].x = points[i].pt.x;
		D[i].y = points[i].pt.y;
		D[i].z = 0.0;
	}
	compute3D(D, eps, minPts);
}

void DBSCAN::compute3D(const std::vector<cv::Point3f>& D,double eps,int minPts){
	Cidxs.resize(D.size());
	visited.resize(D.size());
	for(int i=0;i<Cidxs.size();++i){
		Cidxs[i]=-1;
		visited[i] = false;
	}
	clusters.clear();
	clusters.resize(1);

	vector<int> neighbor;

	int C = 0;//present C number, -1 depicts unvisited, -2 depicts noise

	for(int i=0;i<D.size();++i){
		
		if(visited[i])
			continue;
		
		visited[i] = true;
		regionQuery(D,D[i],eps,neighbor);
		if(neighbor.size() < minPts){
			Cidxs[i] = -2;
		}
		else{
			expandCluster(D,i,neighbor, C++, eps,minPts);
			clusters.resize(clusters.size()+1);
		}

	}

}

void DBSCAN::ClusterJoint(std::vector<int> &C1, std::vector<int>& C2, std::vector<bool>& inC1){
	for(int i=0;i<C2.size();++i){
		if(!inC1[C2[i]]){ //c2[i] is not in C1
			inC1[C2[i]] = true;
			C1.push_back(C2[i]);
		}
	}
}

void DBSCAN::expandCluster(const std::vector<cv::Point3f>& D,const int Pi, std::vector<int>& neighbor, int C, double eps,int minPts){
	
	//Push P into cluster C
	Cidxs[Pi] = C;
	clusters[C].push_back(Pi);
	//printf("%d neighbors\n",neighbor.size());

	//add P to...
	std::vector<bool> inN1(D.size(),false);
	//initialize inN1
	for(int i=0;i<neighbor.size();++i){
		inN1[neighbor[i]] = true;
	}


	std::vector<int> neighbor2;


	for(int i=0;i<neighbor.size();++i){
		int Pi2 = neighbor[i];

		if(! visited[Pi2]){
			visited[Pi2] = true;
			regionQuery(D,D[Pi2], eps,neighbor2);

			if(neighbor2.size() >= minPts)
				ClusterJoint(neighbor,neighbor2,inN1);//joint neighbor & neighbor2!, notice that the new member will be pushed in the back, so the for loop would not be influenced
		}
		
		if(Cidxs[Pi2]<0){//not clustered
			Cidxs[Pi2] = C;
			clusters[C].push_back(Pi2);
		}
	}

}

void DBSCAN::regionQuery(const std::vector<cv::Point3f>& D,const cv::Point3f& P, double eps,std::vector<int>& neighbor){//return all points within P's eps-neighborhood in D
	
	neighbor.clear();
	for(int i=0;i<D.size();++i){
		if( dist(D[i],P)<eps )
			neighbor.push_back(i);
	}
}