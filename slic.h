#ifndef SLIC_H
#define SLIC_H

#include "slic_gpu.h"

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
	void force_connectivity(vector< vector<int> >& cluster);
	void get_result();
	//void draw(vector<vector<int>>& cluster);
};

#endif