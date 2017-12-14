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
	float weight;					// parameter used to calculate distance
	int h, w, sp_h, sp_w, b_h, b_w;	// the image has h*w pixels, sp_h*sp_w superpixels, and b_h*b_w 16*16 blocks
	int sp_size;			// size of superpixel is sp_size*sp_size

	float norm_xy_dist;
	float norm_color_dist;

	void read_label();
public:
	gSlic(Color* image, int height, int width, int it_num, int sp_size, float m);
	~gSlic();
	void init();
	void kmeans();
	void copy_result();
	void force_connectivity();
	Color* draw_boundary();
};

#endif