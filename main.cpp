#include "slic.h"

void draw(vector< vector<int> >& cluster, IplImage* image, int w, int h){
	CvScalar color = CvScalar(0,0,255);
	int dir[] = {1,0,-1,0,1};
	for(int i = 1; i < h-1; i++){
		for(int j = 1; j < w-1; j++){
			for(int k = 0; k < 4; k++){
				int ii = i + dir[k];
				int jj = j + dir[k+1];
				if(cluster[i][j] != cluster[ii][jj]){
					cvSet2D(image, i, j, color);
				}
			}
		}
	}

	cvShowImage("Superpixel", image);
}

/*
 * Input: image address, number of iterations, number of superpixels, weight to calculate distance
 */
int main(int argc, char* argv[]){
	IplImage* image = cvLoadImage(argv[1],1);
	IplImage* image_lab = cvCloneImage(image);
	cvCvtColor(image, image_lab, CV_BGR2Lab);

	int it = atoi(argv[2]);
	int num = atoi(argv[3]);
	int m = atoi(argv[4]);
	int h = image->width;
	int w = image->height;
	int sp_size = (int) sqrt(h * w / (double) num);
	Color* img = (Color*) malloc(h * w * sizeof(Color));
	for (int i = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j)
		{
			img[i * w + j] = Color(cvGet2D(image, i, j));
		}
	}

	gSlic* slic = new gSlic(img, h, w, it, sp_size, m);
	slic->kmeans();
	slic->get_result();
	vector< vector<int> > cluster;
	slic->force_connectivity(cluster);
	draw(cluster, image, w, h);

	return 0;
}