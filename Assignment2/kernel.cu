#include <cuda_runtime.h>
#include <iostream>
#include "kernel.h"
// implement your kernels
__global__ void rgb2gray(unsigned char * d_src, unsigned char * d_dst, int width, int height)
{
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos_x >= width || pos_y >= height)
        return;

    /*
     * CImg RGB channels are split, not interleaved.
     * (http://cimg.eu/reference/group__cimg__storage.html)
     */
    unsigned char r = d_src[pos_y * width + pos_x];
    unsigned char g = d_src[(height + pos_y ) * width + pos_x];
    unsigned char b = d_src[(height * 2 + pos_y) * width + pos_x];

    unsigned int _gray = (unsigned int)((float)(r + g + b) / 3.0f + 0.5);
    unsigned char gray = _gray > 255 ? 255 : _gray;

    d_dst[pos_y * width + pos_x] = gray;
}


void rgb2gray_cpu(unsigned char * d_src, unsigned char * d_dst, int width, int height){

    for (int i = 0; i < width ; i++){
      for (int j = 0; j < height ; j++){
        unsigned char r = d_src[j * width + i];
        unsigned char g = d_src[(height + j ) * width + i];
        unsigned char b = d_src[(height * 2 + j) * width + i];
        unsigned int _gray = (unsigned int)((float)(r + g + b) / 3.0f + 0.5);
        unsigned char gray = _gray > 255 ? 255 : _gray;
        d_dst[j * width + i] = gray;
      }
    }

}

void histgram_cpu(int* hist, unsigned char*gray,int width, int height){
    int size = width*height;

    for (int i=0;i<size;i++){
        unsigned char gray_val=gray[i];
        hist[gray_val]++;
    }
}

__global__ void histgram_gpu(int* hist, unsigned char*gray,int width, int height){
    int size = width*height;
   
   int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (pos_x >= size/2)
        return;
        hist[gray[pos_x]]++;
        hist[gray[pos_x+1]]++;
        //hist[gray[pos_x+2]]++;
        //hist[gray[pos_x+3]]++;
    
}



__global__ void ContrastEnhancement_gpu(unsigned char*gray,unsigned char*res,int width, int height, int min, int max){
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos_x >= width || pos_y >= height)
        return;
        
            unsigned char gray_val = gray[pos_y * width + pos_x];
            if (gray_val > max){
                res[pos_y * width + pos_x] = 255;
            }
            else if (gray_val < min)
            {
                res[pos_y * width + pos_x] = 0;
            }else
            {
                res[pos_y * width + pos_x] =static_cast<unsigned char>( 255.0 * (gray_val-min)/(max-min));
            }
        }

void ContrastEnhancement_cpu(unsigned char*gray,unsigned char*res,int width, int height, int min, int max){
    for(int i =0;i<width;i++){
        for(int j=0; j<height;j++){
            unsigned char gray_val = gray[j * width + i];
            if (gray_val > max){
                res[j * width + i] = 255;
            }
            else if (gray_val < min)
            {
                res[j * width + i] = 0;
            }else
            {
                res[j * width + i] =static_cast<unsigned char>( 255.0 * (gray_val-min)/(max-min));
            }
        }
    }
}

void Smoothing_cpu(unsigned char*gray,unsigned char*res,int width, int height){

    // case boundary

    res[0] = static_cast<unsigned char>((gray[0] + gray[1] +gray[width] + gray[width+1]) / 4.0)  ; // top left
    res[width-1] = static_cast<unsigned char>( (gray[width-1] + gray[width-2] +  gray[width*2-1] + gray[width*2-2]) / 4.0) ; //top right
    res[width*height-1] = static_cast<unsigned char>( (gray[height*width-1] + gray[height*width-2] + gray[(height-1)*width-1] + gray[(height-1)*width-2])/4.0) ; //bottom right
    res[(height-1)*width] = static_cast<unsigned char>((gray[(height-1)*width] + gray[(height-1)*width + 1] + gray[(height-2)*width] + gray[(height-2)*width+1]) / 4.0); //bottom left

    for (int i = 1;i<width-1;i++){
        res[i] =static_cast<unsigned char>( (gray[i] + gray[i+1] + gray[i-1] + gray[ width+ i-1] + gray[ width+ i] + gray[ width+ i+1]) / 6.0); // top row
        res[(height-1)*width + i] =static_cast<unsigned char>( (gray[(height-1)*width +i] + gray[(height-1)*width +i+1] + gray[(height-1)*width +i-1] + gray[(height-2)*width + i] +gray[(height-2)*width + i-1] + gray[(height-2)*width + i+1]) / 6.0); // bottom row
    }

    for (int i=1;i<height-1;i++){
        res[width*i] = static_cast<unsigned char>((gray[width*i] + gray[width*i+1] + gray[width*(i-1)] + gray[width*(i-1)+1] + gray[width*(i+1)] + gray[width*(i+1)+1]) / 6.0);

        //res[width*i+1 - 1] = static_cast<unsigned char>( (gray[width*i - 1] + gray[width*i - 1 -1] + gray[width*(i-1) - 1] + gray[width*(i-1) - 1 -1] + gray[width*(i+1) - 1] +  gray[width*(i+1) - 1 - 1]) / 6.0);

        res[width*(i+1) - 1] = static_cast<unsigned char>( (gray[width*(i+1) - 1] + gray[width*(i+1) - 2] + gray[width*i - 1] + gray[width*i - 2] + gray[width*(i+2) - 1] +  gray[width*(i+2) - 2]) / 6.0);

    }

    // not boundry
    for(int i = 1;i<width-1;i++){
        for (int j =1; j<height-1; j++){
            unsigned char avg = static_cast<unsigned char>( (  gray[j * width + i] + gray[j * width + i -1 ] + gray[j * width + i +1 ] + gray[(j - 1)* width + i  ] + gray[(j + 1)* width + i  ]  + gray[(j-1) * width + i-1] + gray[(j-1) * width + i+1] + gray[(j+1) * width + i-1] + gray[(j+1) * width + i+1] ) / 9.0 );
            res[j * width + i] = avg;
        }
    }

}



__global__ void Smoothing_gpu(unsigned char*gray,unsigned char*res,int width, int height){

    
    
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pos_x >= width || pos_y >= height)
        return;
        
    // not boundry
    unsigned char avg = static_cast<unsigned char>( (  gray[pos_y * width + pos_x] + gray[pos_y * width + pos_x - 1 ] + gray[pos_y * width + pos_x +1 ] + gray[(pos_y - 1)*                 width + pos_x  ] + gray[(pos_y + 1)* width + pos_x  ]  + gray[(pos_y-1) * width + pos_x - 1] + gray[(pos_y-1) * width + pos_x +1] + gray[(pos_y+1) * width + pos_x -1] +                gray[(pos_y+1) * width + pos_x +1] ) / 9.0 );
    res[pos_y * width + pos_x ] = avg;
    
    if (pos_x < (width -1)){
    
        res[pos_x] =static_cast<unsigned char>( (gray[pos_x] + gray[pos_x+1] + gray[pos_x-1] + gray[ width+ pos_x-1] + gray[ width+ pos_x] + gray[ width+ pos_x+1]) / 6.0); // top row
        res[(height-1)*width + pos_x] =static_cast<unsigned char>( (gray[(height-1)*width +pos_x] + gray[(height-1)*width +pos_x+1] + gray[(height-1)*width +pos_x-1] + gray[(height-2)         *width + pos_x] +gray[(height-2)*width + pos_x-1] + gray[(height-2)*width + pos_x+1]) / 6.0); // bottom row
    }
    
    
    if (pos_y < (height -1)){
        res[width*pos_y] = static_cast<unsigned char>((gray[width*pos_y] + gray[width*pos_y+1] + gray[width*(pos_y-1)] + gray[width*(pos_y-1)+1] + gray[width*(pos_y+1)] + gray[width*          (pos_y+1)+1]) / 6.0);

        //res[width*i+1 - 1] = static_cast<unsigned char>( (gray[width*i - 1] + gray[width*i - 1 -1] + gray[width*(i-1) - 1] + gray[width*(i-1) - 1 -1] + gray[width*(i+1) - 1] +  gray         [width*(i+1) - 1 - 1]) / 6.0);

        res[width*(pos_y +1) - 1] = static_cast<unsigned char>( (gray[width*(pos_y +1) - 1] + gray[width*(pos_y +1) - 2] + gray[width*pos_y - 1] + gray[width*pos_y - 2] + gray[width*             (pos_y+2) - 1] +  gray[width*(pos_y+2) - 2]) / 6.0);

    }
    
    
    // case boundary
    res[0] = static_cast<unsigned char>((gray[0] + gray[1] +gray[width] + gray[width+1]) / 4.0)  ; // top left
    res[width-1] = static_cast<unsigned char>( (gray[width-1] + gray[width-2] +  gray[width*2-1] + gray[width*2-2]) / 4.0) ; //top right
    res[width*height-1] = static_cast<unsigned char>( (gray[height*width-1] + gray[height*width-2] + gray[(height-1)*width-1] + gray[(height-1)*width-2])/4.0) ; //bottom right
    res[(height-1)*width] = static_cast<unsigned char>((gray[(height-1)*width] + gray[(height-1)*width + 1] + gray[(height-2)*width] + gray[(height-2)*width+1]) / 4.0); //bottom left


}
