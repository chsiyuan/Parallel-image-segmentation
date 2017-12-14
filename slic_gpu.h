#pragma once

#ifndef SLIC_GPU_H
#define SLIC_GPU_H

#include<stdio.h>
#include<iostream>
#include<fstream>
#include<math.h>
#include<vector>
#include<string>
#include<set>
#include<float.h>
#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

#define BLOCK_DIM 16

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define _CPU_AND_GPU_CODE_ __device__	// for CUDA device code
#else
#define _CPU_AND_GPU_CODE_ 
#endif

#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if(code != cudaSuccess){
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if(abort) exit(code);
	}
}

struct Point{
	float x;
	float y;
	_CPU_AND_GPU_CODE_ Point(){}
	_CPU_AND_GPU_CODE_ Point(float xx, float yy): x(xx), y(yy){} 
};

struct Color
{
	float l;
	float a;
	float b;
	_CPU_AND_GPU_CODE_ Color(){}
	_CPU_AND_GPU_CODE_ Color(float ll, float aa, float bb): l(ll), a(aa), b(bb){}
	void toLab();
	void toRgb();
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