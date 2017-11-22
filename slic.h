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
	_CPU_AND_GPU_CODE_ Color(CvScalar c){ // construction from data type in opencv
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
	// overload operator + and / for convenient computation
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
	Color* img_host;		// image in cpu, h*w
	Color* img_dev;			// image in gpu, h*w
	int* cluster_host;		// cluster label for every pixel in cpu, h*w
	int* cluster_dev;		// cluster label for every pixel in gpu, h*w
	SuperPoint* centers;	// cluster center coordinates, sp_h*sp_w
	int* cluster_count;		// number of pixels in each class
	SuperPoint* color_acc;	// sum of coordinates of pixels in each class

	int it_num;				// iteration number
	int m;					// parameter used to calculate distance
	int h, w, sp_h, sp_w, b_h, b_w;	// the image has h*w pixels, sp_h*sp_w superpixels, and b_h*b_w 16*16 blocks
	int sp_size;			// size of superpixel is sp_size*sp_size

	void init();
public:
	gSlic(Color* image, int height, int width, int it_num, int sp_size, int m);
	~gSlic();
	void kmeans();
	void force_connectivity(vector<vector<int>>& cluster);
	void get_result();
	//void draw(vector<vector<int>>& cluster);
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