#include "slic.h"

void readImg(char* addr, Color* &img, int& h, int& w){
    ifstream raw_file;
    raw_file.open(addr,std::ifstream::binary);
    if(!raw_file){
        cerr << "Unable to open file " << addr << endl;
        exit(1);
    }

    raw_file >> h >> w;
    raw_file.ignore();
    char* R = new char[h*w];
    char* G = new char[h*w];
    char* B = new char[h*w];
    raw_file.read(R,h*w);
    raw_file.read(G,h*w);
    raw_file.read(B,h*w);
    raw_file.close();
    
    img = (Color*) malloc(h * w * sizeof(Color));
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            Color color((float) uint8_t(R[i*w+j]), (float) uint8_t(G[i*w+j]), (float) uint8_t(B[i*w+j]));
            color.toLab();
            img[i * w + j] = color;
        }
    }

    delete[] R;
    delete[] G;
    delete[] B;
}

void writeImg(char* addr, Color* img, int h, int w){
	char* outbuf = new char[h*w*3];
    
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            Color color = img[i * w + j];
            color.toRgb();
            outbuf[i*w+j] = char(color.l);
            outbuf[i*w+j + w*h] = char(color.a);
            outbuf[i*w+j + 2*w*h] = char(color.b);
        }
    }
    
    ofstream res_file(addr);
    res_file << to_string(h) << " " << to_string(w) << endl;
    res_file.write(outbuf, h*w*3);
    res_file.close();

    delete[] outbuf;
}

void Color::toLab(){
    double R, G, B;
    R = l / 255.0;
    G = a / 255.0;
    B = b / 255.0;
    double X, Y, Z;
    
    R = (R > 0.04045) ? pow((R + 0.055) / 1.055, 2.4) : R / 12.92;
    G = (G > 0.04045) ? pow((G + 0.055) / 1.055, 2.4) : G / 12.92;
    B = (B > 0.04045) ? pow((B + 0.055) / 1.055, 2.4) : B / 12.92;
    
    X = (R * 0.4124 + G * 0.3576 + B * 0.1805) / 0.95047;
    Y = (R * 0.2126 + G * 0.7152 + B * 0.0722) / 1.00000;
    Z = (R * 0.0193 + G * 0.1192 + B * 0.9505) / 1.08883;
    
    X = (X > 0.008856) ? pow(X, 1.0/3.0) : (7.787 * X) + 16.0/116.0;
    Y = (Y > 0.008856) ? pow(Y, 1.0/3.0) : (7.787 * Y) + 16.0/116.0;
    Z = (Z > 0.008856) ? pow(Z, 1.0/3.0) : (7.787 * Z) + 16.0/116.0;
    
    l = (116.0 * Y) - 16.0;
    a = 500.0 * (X - Y);
    b = 200.0 * (Y - Z);
}
void Color::toRgb(){
    double X, Y, Z;
    Y = (l + 16.0) / 116.0;
    X = a / 500.0 + Y;
    Z = Y - b / 200.0;
    double R, G, B;
    
    X = 0.95047 * ((X * X * X > 0.008856) ? X * X * X : (X - 16.0/116.0) / 7.787);
    Y = 1.00000 * ((Y * Y * Y > 0.008856) ? Y * Y * Y : (Y - 16.0/116.0) / 7.787);
    Z = 1.08883 * ((Z * Z * Z > 0.008856) ? Z * Z * Z : (Z - 16.0/116.0) / 7.787);
    
    R = X *  3.2406 + Y * -1.5372 + Z * -0.4986;
    G = X * -0.9689 + Y *  1.8758 + Z *  0.0415;
    B = X *  0.0557 + Y * -0.2040 + Z *  1.0570;
    
    R = (R > 0.0031308) ? (1.055 * pow(R, 1.0/2.4) - 0.055) : 12.92 * R;
    G = (G > 0.0031308) ? (1.055 * pow(G, 1.0/2.4) - 0.055) : 12.92 * G;
    B = (B > 0.0031308) ? (1.055 * pow(B, 1.0/2.4) - 0.055) : 12.92 * B;
    
    l = max(min(round(R * 255.0), 255.0),0.0);
    a = max(min(round(G * 255.0), 255.0),0.0);
    b = max(min(round(B * 255.0), 255.0),0.0);
}


/*
 * Input: input image txt file address, number of iterations, number of superpixels, weight to calculate distance, outout image txt file address
 */
int main(int argc, char* argv[]){

	Color* img = nullptr;
	int h, w;
	readImg(argv[1], img, h, w);

	int it = atoi(argv[2]);
	int num = atoi(argv[3]);
	float weight = atof(argv[4]);
	int sp_size = (int) sqrt(h * w / (double) num);

	cout << "======Input=======" << endl;
	cout << "Read image: " << argv[1] << " with size: " << h << "x" << w << endl;
	cout << "Max iteration: " << it << endl;
	cout << "Number of superpixels: "<< num << endl;
	cout << "Size of superpixels: " << sp_size << endl;
	cout << "Weight: " << weight << endl;

    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now(); // start timing
	
    gSlic* slic = new gSlic(img, h, w, it, sp_size, weight);
    slic->init();
	slic->kmeans();
    slic->copy_result();

    high_resolution_clock::time_point end = high_resolution_clock::now();  // end timing
    duration<double> time = duration_cast<duration<double>>(end - start);
    
    cout << "Wall time: " << time.count() << endl;

    slic->force_connectivity();
    img = slic->draw_boundary();

	writeImg(argv[5], img, h, w);

	return 0;
}