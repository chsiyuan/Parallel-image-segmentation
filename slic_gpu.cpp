#include "slic.h"

//===============================
//          GPU CODE
//===============================

// global functions, called from host
__global__ void init_centers(int h, int w, int sp_h, int sp_w, int sp_size, Color* img, SuperPoint* centers);
__global__ void assign_label(int h, int w, int sp_h, int sp_w, int sp_size, float m, Color* img, int* cluster, SuperPoint* centers);
__global__ void clustering(int h, int w, int sp_h, int sp_w, int b_h, int b_w, int b_size,
						 Color* img, int* cluster, int* cluster_count, SuperPoint* color_acc);
__global__ void reduce_count(int numClusters, int numBlocksLast, int* cluster_count);
__global__ void reduce_color(int numClusters, int numBlocksLast, SuperPoint* color_acc);
__global__ void updtate_center(int numClusters, SuperPoint* centers, int* cluster_count, SuperPoint* color_acc);

// device functions called from kernels
__device__ inline SuperPoint find_center(Color *img, int w, Point center);
__device__ inline double compute_dist(SuperPoint p1, SuperPoint p2, int S, float m);

//===============================
//           CPU CODE
//===============================

gSlic::gSlic(Color* image, int height, int width, int it_num, int sp_size, float m){
	this->it_num = it_num;
	this->sp_size = sp_size;
	this->m = m;
	this->h = height;
	this->w = width;
	this->img_host = image;
	init();
}

gSlic::~gSlic(){
	free(img_host);
	free(cluster_host);
	cudaFree(img_dev);
	cudaFree(cluster_dev);
	cudaFree(centers);
	cudaFree(cluster_count);
	cudaFree(color_acc);
}

/*
 * Initialization 
 *
 */
void gSlic::init(){
	cout << "========Initialization=======" << endl;
	cudaMalloc(&img_dev, h * w * sizeof(Color));
	cudaMemcpy(img_dev, img_host, h * w * sizeof(Color), cudaMemcpyHostToDevice);
	cudaMalloc(&cluster_dev, h * w * sizeof(int));

	sp_h = (h + sp_size - 1) / sp_size;
	sp_w = (w + sp_size - 1) / sp_size;
	b_h = (h + BLOCK_DIM - 1) / BLOCK_DIM;
	b_w = (w + BLOCK_DIM - 1) / BLOCK_DIM;

	cudaMalloc(&centers, sp_h * sp_w * sizeof(SuperPoint));
	cudaMalloc(&cluster_count, b_h * b_w * sp_h * sp_w * sizeof(int));
	cudaMemset(cluster_count, 0, b_h * b_w * sp_h * sp_w * sizeof(int));
	cudaMalloc(&color_acc, b_h * b_w * sp_h * sp_w * sizeof(SuperPoint));
	cudaMemset(color_acc, 0, b_h * b_w * sp_h * sp_w * sizeof(SuperPoint));

	int numThreads = BLOCK_DIM * BLOCK_DIM;
	int numBlocks = (sp_w * sp_h + numThreads - 1) / numThreads; 
	// initialize cluster centers
	init_centers<<<numBlocks , numThreads>>>(h, w, sp_h, sp_w, sp_size, img_dev, centers);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

// Calculate the smallest power of to larger than n
inline int helper(int n){
	n--;
	n = n >> 1 | n;
	n = n >> 2 | n;
	n = n >> 4 | n;
	n = n >> 8 | n;
	n = n >> 16 | n;
	return ++n;
}

/*
 * Generate superpixels
 */
void gSlic::kmeans(){
	cout << "========Kmeans=======" << endl;
	// for assign_label()
	int numThreads = BLOCK_DIM * BLOCK_DIM;
	int numBlocksPerCluster = (sp_size*sp_size + numThreads - 1) / numThreads;
	dim3 numBlocks(sp_h*sp_w , numBlocksPerCluster);

	// for clustering()
	int numBlocksClusters = (sp_h*sp_w + numThreads - 1) / numThreads;
	dim3 numBlocks2(b_h*b_w , numBlocksClusters);

	for(int i = 0; i < it_num; i++){

		// Assign labels to every pixel
		// Locally search the nearby 9 cluster centers for the nearest one
		//
		// divide each (square) cluster block into blocks because all
		// the pixels in the same cluster block share the same 9 nearest
		// neighbors
		cout << "---> Assign labels" << endl;
		assign_label<<<numBlocks, numThreads>>>(h, w, sp_h, sp_w, sp_size, m, img_dev, cluster_dev, centers);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// Sum up pixels belonging to the same cluster within one block
		//
		// divide img into 16*16 blocks
		// within each block, one thread responsible for counting one cluster
		// those threads share the image block locally
		cout << "---> clustering" << endl;
		clustering<<<numBlocks2 , numThreads>>>(h, w, sp_h, sp_w, b_h, b_w, numThreads, img_dev, cluster_dev, cluster_count, color_acc);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// Reduce and update center
		// Sum up the count of one cluster over all blocks
		//
		// one parent thread responsible for summing up one column of cluster_count and color_acc
		// it calls child blocks and threads repeatedly
		cout << "---> Reduction" << endl;
		int numBlocksLast = b_h*b_w;
		do{
 			// if there is less than 16*16 elements to reduce, then we don't need to create that many threads
 			int numThreadsPerBlock = min(BLOCK_DIM * BLOCK_DIM , helper(numBlocksLast));
 			int numBlocksCurr = (numBlocksLast + numThreadsPerBlock - 1) / numThreadsPerBlock;
 			dim3 numBlocks3(numBlocksCurr , sp_w * sp_h);
			reduce_count<<<numBlocks , numThreadsPerBlock>>>(sp_h * sp_w, numBlocksLast, cluster_count);
 			reduce_color<<<numBlocks , numThreadsPerBlock>>>(sp_h * sp_w, numBlocksLast, color_acc);
 			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
 			numBlocksLast = numBlocksCurr;
 		}while(numBlocksLast > 1);
 		cout << "---> Update center" << endl;
		updtate_center<<<numBlocksClusters , numThreads>>>(sp_h * sp_w, centers, cluster_count, color_acc);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
	}

}

/*
 * Copy clustering result to CPU
 */
void gSlic::get_result(){
	cout << "========Read result=======" << endl;
	cluster_host = (int*) malloc(h * w * sizeof(int));
	cudaMemcpy(cluster_host, cluster_dev, h * w * sizeof(int), cudaMemcpyDeviceToHost);
}

void gSlic::read_label(){
	cout << "Clusters:..." << endl;
	set<int> cid;
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			int id = cluster_host[i*w+h];
			if(cid.find(id)==cid.end()){
				cid.insert(id);
				cout << id << " ";
			}
		}
	}
	cout << endl;
}

/*
 * Force the superpixels to be connected components
 */
void gSlic::force_connectivity(vector< vector<int> >& cluster){
	cout << "========Enforce connectivity=======" << endl;
	int label = 0, adjlabel = 0;
	int thres = h * w / (sp_h * sp_w) / 4;

	int dir[] = {1,0,-1,0,1};

	cluster.resize(h, vector<int>(w, -1));  // new cluster labels
	for(int i = 0; i < h; i++){
		for(int j = 0; j < w; j++){
			if(cluster[i][j] == -1){  // if this pixel has not been visited yet
				vector<Point> points;
				points.push_back(Point(i,j));

				// record a different label of neighbors for future use
				for(int k = 0; k < 4; k++){  
					int ii = i + dir[k];
					int jj = j + dir[k+1];
					if(ii>=0 && ii<h && jj>=0 && jj<h && cluster[ii][jj]>=0){
						adjlabel = cluster[ii][jj];
					}
				}

				// do bfs to find all the connected pixels with the same label
				int count = 1;
				for(int c = 0; c < count; c++){
					for(int k = 0; k < 4; k++){
						int ii = points[c].x + dir[k];
						int jj = points[c].y + dir[k+1];
						if(ii>=0 && ii<h && jj>=0 && jj<w &&
						 cluster[ii][jj]==-1 && cluster_host[ii*w+jj]==cluster_host[i*w+j]){
							points.push_back(Point(ii,jj));
							cluster[ii][jj] = label;
							count++;
						}
					}
				}

				// if this connected component is too small, assign the pixels with the label of neighbor
				if(count <= thres){
					for(int c = 0; c < points.size(); c++){
						cluster[points[c].x][points[c].y] = adjlabel;
					}
					label--;
				}
				label++;
			}
		}
	}
}


//===============================
//    GPU CODE IMPLEMENTATION
//===============================
/*
 * Initialize centers of clusters. Called by init().
 */
__global__ void init_centers(int h, int w, int sp_h, int sp_w, int sp_size, Color* img, SuperPoint* centers){

	if(threadIdx.x == 0)
		printf("GPU: init centers\n");

	int idx_x = (blockIdx.x * blockDim.x + threadIdx.x) / sp_w;
	int idx_y = (blockIdx.x * blockDim.x + threadIdx.x) % sp_w;
	if(idx_x<sp_h && idx_y<sp_w){
		// initial center
		int height = min(h,(idx_x+1)*sp_size) - idx_x*sp_size;
		int width = min(w,(idx_y+1)*sp_size) - idx_y*sp_size;
		Point center(idx_x*sp_size+ height/2 , idx_y*sp_size+width/2);
		centers[threadIdx.x] = find_center(img, w, center);
	}
}

// Find the pixel with smallest gradient in the 8 adjacent neighbors.
// gradient is respect to gray-scale value
__device__ inline SuperPoint find_center(Color *img, int w, Point center){
	
	// load 4*4 area around the initial center to local
	Color img_local[4*4];
	for (int i = 0; i < 4; i++)
	{
		int ii = center.x-1+i;
		for (int j = 0; j < 4; j++)
		{
			int jj = center.y-1+j;
			img_local[i*4+j] = img[ii*w+jj];
		}
	}

	double min_grad = FLT_MAX;
	Point loc_min = center;
	Color col_min = img_local[1*4+1];
    
    // find the pixel with smallest gradient
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
        	Color v = img_local[i*4+j];
            Color v1 = img_local[(i+1)*4+j];
            Color v2 = img_local[i*4+j+1];
        	double grad = sqrtf((v1.l-v.l)*(v1.l-v.l)+(v2.l-v.l)*(v2.l-v.l));
        	if(grad < min_grad){
        		min_grad = grad;
        		loc_min.x = center.x-1+i;
        		loc_min.y = center.y-1+j;
        		col_min = v;
        	}
        }
    }

    return SuperPoint(col_min, loc_min);
}

/*
 * Assign labels to every pixel
 */
__global__ void assign_label(int h, int w, int sp_h, int sp_w, int sp_size, float m, Color* img, int* cluster, SuperPoint* centers){

	if(threadIdx.x == 0)
		printf("GPU: assign label\n");

	__shared__ SuperPoint neighbors[9];

	//----Indices initialization----
	int blockIdx_x = blockIdx.x / sp_w; // 2D block id
	int blockIdx_y = blockIdx.x % sp_w;
	int x_start = blockIdx_x * sp_size;	// start x,y position in the image
	int y_start = blockIdx_y * sp_size; 
	int x_end, y_end;
	if(blockIdx_x == sp_h-1) x_end = h;
	else x_end = (blockIdx_x+1) * sp_size;

	if(blockIdx_y == sp_w-1) y_end = w;
	else x_end = (blockIdx_y+1) * sp_size;

	int width = y_end - y_start; 		// actual w and h of block
	int height = x_end - x_start;

	int idx_local_1d = blockIdx.y * blockDim.x + threadIdx.x;  // 1D thread id in the current block
	int idx_local_x = idx_local_1d / sp_size;  // 2D thread id in the current block
	int idx_local_y = idx_local_1d % sp_size;
	//------------------------------

	// Load 9 nearby centers
	if(threadIdx.x < 9){
		int idx_temp_x = threadIdx.x / 3 - 1;
		int idx_temp_y = threadIdx.x % 3 - 1;
		int idx_n_x = blockIdx_x + idx_temp_x;
		int idx_n_y = blockIdx_y + idx_temp_y;
		if(idx_n_x>=0 && idx_n_x<sp_h && idx_n_y>=0 && idx_n_y<sp_w)
			neighbors[threadIdx.x] = centers[idx_n_x * sp_w + idx_n_y];
	}
	__syncthreads();

	// Assign label to every pixel
	if(idx_local_x < height && idx_local_y < width){
		Point v = Point(x_start + idx_local_x, y_start + idx_local_y);
		Color color = img[v.x * w + v.y];
		SuperPoint p = SuperPoint(color, v);

		double min_dist = FLT_MAX;
		int label = blockIdx.x;
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				int idx_n_x = blockIdx_x + i;  // 2D block id of neighbors
				int idx_n_y = blockIdx_y + j;
				if(idx_n_x>=0 && idx_n_x<sp_h && idx_n_y>=0 && idx_n_y<sp_w){
					double dist = compute_dist(p, neighbors[(i+1)*3+j+1], sp_size, m);
					if(dist < min_dist){
						min_dist = dist;
						label = idx_n_x * sp_w + idx_n_y;
					}
				}
			}
			
		}
		
		cluster[v.x * w + v.y] = label;
	}
	
}

// Distance between two superpoints is defined as |(l1,a1,b1)-(l2,a2,b2)|| + m/S ||(x1,y1)-(x2,y2)||
// m and S are parameters, see paper
__device__ inline double compute_dist(SuperPoint p1, SuperPoint p2, int S, float m){
	double d_lab = sqrtf( (p1.color.l-p2.color.l) * (p1.color.l-p2.color.l)
					   	+ (p1.color.a-p2.color.a) * (p1.color.a-p2.color.a)
						+ (p1.color.b-p2.color.b) * (p1.color.b-p2.color.b));
	double d_xy = sqrtf(  (p1.point.x-p2.point.x) * (p1.point.x-p2.point.x)
						+ (p1.point.y-p2.point.y) * (p1.point.y-p2.point.y));
	return d_lab + m/S*d_xy;
}


/*
 * Sum up pixels belonging to the same cluster within one block.
 */
__global__ void clustering(int h, int w, int sp_h, int sp_w, int b_h, int b_w, int b_size,
						 Color* img, int* cluster, int* cluster_count, SuperPoint* color_acc){
	// Note: block size is set to be 16*16

	__shared__ Color img_local[BLOCK_DIM * BLOCK_DIM];
	__shared__ int cluster_local[BLOCK_DIM * BLOCK_DIM];

	//----Indices initialization----
	int blockIdx_x = blockIdx.x / b_w;
	int blockIdx_y = blockIdx.x / b_w;
	int x_start = blockIdx_x * b_size;
	int y_start = blockIdx_y * b_size;
	int x_end, y_end;
	if(blockIdx_x == b_h-1) x_end = h;
	else x_end = (blockIdx_x + 1) * b_size;

	if(blockIdx_y == b_h-1) y_end = h;
	else y_end = (blockIdx_y + 1) * b_size;

	int height = x_end - x_start;
	int width = y_end - y_start;
	
	int idx_local_x = threadIdx.x / b_size;
	int idx_local_y = threadIdx.x % b_size;
	int idx_global = (x_start+idx_local_x) * w + idx_local_y;
	int idx_cluster = blockIdx.y * blockDim.x + threadIdx.x; // one thread is responsible for one cluster
	//------------------------------

	// Load img block to local
	// Since b_size is set to be 16*16, each thread can load one pixel value
	if(idx_local_x < height && idx_local_y < width){
		img_local[threadIdx.x] = img[idx_global];
		cluster_local[threadIdx.x] = cluster[idx_global];
	}
	
	__syncthreads();

	// Count pixels in a cluster and accumulate the 5-D coordinates
	if(idx_cluster < sp_h*sp_w){
		int count = 0;
		SuperPoint p(Color(0,0,0),Point(0,0));
		for(int i = 0; i < b_size; i++){
			for(int j = 0; j < b_size; j++){
				if(i < height && j < width){
					int idx = i * b_size + j;
					if(cluster_local[idx] == idx_cluster){
						count++;
						p += SuperPoint(img_local[idx],Point(x_start+i,y_start+j));
					}
				}
			}
		}

		// cluster_count is block_num * cluster_num = (b_h*b_w) * (sp_h*sp_w)
		cluster_count[blockIdx.x * sp_h * sp_w + idx_cluster] = count;
	}
}

/*
 * Sum up count and coordinates over all the blocks.
 */
__global__ void updtate_center(int numClusters, SuperPoint* centers, int* cluster_count, SuperPoint* color_acc){
	
	int idx_cluster = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx_cluster < numClusters){
		// update cluster center
		centers[idx_cluster] = color_acc[idx_cluster] / cluster_count[idx_cluster];
	}
}


// calculate the sum in one block (for count reduction)
__global__ void reduce_count(int numClusters, int numBlocksLast, int* cluster_count){

	__shared__ int cluster_count_local[BLOCK_DIM * BLOCK_DIM];

	int idx_cluster = blockIdx.y;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < numBlocksLast){
		cluster_count_local[threadIdx.x] = cluster_count[idx * numClusters + idx_cluster];
	}
	__syncthreads();

	// reduce like a tree structure
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(threadIdx.x < i && idx + i < numBlocksLast){
			cluster_count_local[threadIdx.x] += cluster_count_local[threadIdx.x + i];
		}
		__syncthreads();
	}

	// save the reduction result to cluster_count[block id, cluster id]
	if(threadIdx.x==0) cluster_count[blockIdx.x * numClusters + idx_cluster] = cluster_count_local[0];
}

// calculate the sum in one block (for coordinates reduction)
__global__ void reduce_color(int numClusters, int numBlocksLast, SuperPoint* color_acc){

	__shared__ SuperPoint color_acc_local[BLOCK_DIM * BLOCK_DIM];

	int idx_cluster = blockIdx.y;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < numBlocksLast){
		color_acc_local[threadIdx.x] = color_acc[idx * numClusters + idx_cluster];
	}
	__syncthreads();

	for(int i=blockDim.x/2; i>0; i>>=1){
		if(threadIdx.x < i && idx + i < numBlocksLast){
			color_acc_local[threadIdx.x] += color_acc_local[threadIdx.x + i];
		}
		__syncthreads();
	}

	if(threadIdx.x==0) color_acc[blockIdx.x * numClusters + idx_cluster] = color_acc_local[0];
}
