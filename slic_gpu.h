#pragma once

#ifndef SLIC_GPU_H
#define SLIC_GPU_H

#include<stdio.h>
#include<iostream>
#include<math.h>
#include<vector>
#include<string>
#include<float.h>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cuda.h>
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
	_CPU_AND_GPU_CODE_ inline SuperPoint operator/(const int& k){
		this->color.l /= k;
		this->color.a /= k;
		this->color.b /= k;
		this->point.x /= k;
		this->point.y /= k;
		return *this;
	}
};




#endif