#pragma once

#ifndef SLIC_H
#define SLIC_H

#include<stdio.h>
#include<iostream>
#include<math.h>
#include<vector>
#include<string>
#include<float.h>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace std;

#define BLOCK_DIM 16

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define _CPU_AND_GPU_CODE_ __device__	// for CUDA device code
#else
#define _CPU_AND_GPU_CODE_ 
#endif

struct Point{
	int x;
	int y;
	_CPU_AND_GPU_CODE_ Point(){}
	_CPU_AND_GPU_CODE_ Point(int xx, int yy): x(xx), y(yy){} 
};

struct Color
{
	double l;
	double a;
	double b;
	_CPU_AND_GPU_CODE_ Color(){}
	_CPU_AND_GPU_CODE_ Color(double ll, double aa, double bb): l(ll), a(aa), b(bb){}
	_CPU_AND_GPU_CODE_ Color(CvScalar c){
		l = c.val[0];
		a = c.val[1];
		b = c.val[2];
	}
};

struct SuperPoint
{
	// (l,a,b,x,y)
	Color color;
	Point point;
	_CPU_AND_GPU_CODE_ SuperPoint(){}
	_CPU_AND_GPU_CODE_ SuperPoint(Color c, Point p): color(c), point(p){}
	_CPU_AND_GPU_CODE_ inline SuperPoint operator+=(const SuperPoint& p){
		this->color.l += p.color.l;
		this->color.a += p.color.a;
		this->color.b += p.color.b;
		this->point.x += p.point.x;
		this->point.y += p.point.y;
		return *this;
	}
	_CPU_AND_GPU_CODE_ inline SuperPoint operator/=(const int& k){
		this->color.l /= k;
		this->color.a /= k;
		this->color.b /= k;
		this->point.x /= k;
		this->point.y /= k;
		return *this;
	}
};

class gSlic{
private:
	Color* img_host;
	Color* img_dev;
	int* cluster_host;
	int* cluster_dev;
	SuperPoint* centers;
	int* cluster_count;
	SuperPoint* color_acc;

	int it_num;
	int m;
	int h,w,sp_h,sp_w, b_h, b_w;
	int sp_size;

	void init();
public:
	gSlic(Color* image, int height, int width, int it_num, int sp_size, int m);
	~gSlic();
	void kmeans();
	void force_connectivity(vector<vector<int>>& cluster);
	void get_result();
	void draw(vector<vector<int>>& cluster);
};

__global__ void init_centers(int sp_h, int sp_w, int sp_size, Color* img, SuperPoint* centers);
__device__ inline SuperPoint find_center(Color *image, int w, Point center);
__device__ inline double compute_dist(SuperPoint p1, SuperPoint p2, int S, int m);
__global__ void assign_label(int h, int w, int sp_h, int sp_w, int sp_size, int m, 
							Color* img, int* cluster, SuperPoint* centers);
__global__ void clustering(int h, int w, int sp_h, int sp_w, int b_h, int b_w, int b_size,
						 Color* img, int* cluster, int* cluster_count, SuperPoint* color_acc);
__device__ inline int helper(int n);
__global__ void reduce_count(int numClusters, int idx_cluster, int numBlocksLast, int* cluster_count);
__global__ void reduce_color(int numClusters, int idx_cluster, int numBlocksLast, SuperPoint* color_acc);
__global__ void updtate_center(int sp_h, int sp_w, int b_h, int b_w, SuperPoint* centers, int* cluster_count, SuperPoint* color_acc);



#endif