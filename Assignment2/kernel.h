#include <cuda_runtime.h>
#include <iostream>

// declear you kernels
__global__ void rgb2gray(unsigned char * d_src, unsigned char * d_dst, int width, int height);

void rgb2gray_cpu(unsigned char * d_src, unsigned char * d_dst, int width, int height);

__global__ void histgram_gpu(int* hist, unsigned char*gray,int width, int height);

__global__ void ContrastEnhancement_gpu(unsigned char*gray,unsigned char*res,int width, int height, int min, int max);

__global__ void Smoothing_gpu (unsigned char*gray,unsigned char*res,int width, int height);

