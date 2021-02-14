#include <iostream>

void rgb2gray_cpu(unsigned char * d_src, unsigned char * d_dst, int width, int height);

void histgram_cpu(int* hist, unsigned char *gray,int width, int height);

void ContrastEnhancement_cpu(unsigned char*gray,unsigned char*res,int width, int height, int min, int max);

void Smoothing_cpu(unsigned char*gray,unsigned char*res,int width, int height);