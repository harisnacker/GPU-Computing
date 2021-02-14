#include "CImg.h"
#include <iostream>
#include "kernelcpu.h"
#include <chrono>
using namespace std;
using namespace cimg_library;


int main() {

    CImg<unsigned char> src("cat2.jpg"); //read image 
    int width = src.width(); 
    int height = src.height();
    unsigned long size = width*height;
    

    //create pointer to image
    unsigned char *h_src = src.data();
    
    CImg<unsigned char> dst(width, height, 1, 1); // create image to store gray
    unsigned char *h_dst = dst.data(); //gray image
    // for contrast enhancemant
    CImg<unsigned char> contrast(width, height, 1, 1);
    unsigned char *h_contrast = contrast.data();

    // for contrast enhancemant
    CImg<unsigned char> smoothing(width, height, 1, 1);
    unsigned char *h_smoothing = smoothing.data();

    int* hist = new int[256]();
    /* measure runtime*/
    auto t1 = std::chrono::high_resolution_clock::now();
    rgb2gray_cpu(h_src,h_dst,width,height);

    histgram_cpu(hist,h_dst,width,height);

    int min,max;
    int temp = 0;
    for (int i = 0;i<256;i++){
        temp += hist[i];
        if(temp > size*0.1){
            min = i;
            temp = 0;
            break;
        }
    }

    for (int i=255;i>=0;i--){
        temp += hist[i];
        if(temp > size*0.1){
            max = i;
            temp = 0;
            break;
        }
    }

    ContrastEnhancement_cpu(h_dst,h_contrast,width,height,min,max);
    
    Smoothing_cpu(h_contrast,h_smoothing,width,height);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration <<"ms"<<std::endl;
    smoothing.save("smoothing.jpg");
    //contrast.save("con.jpg");
    //dst.save("file.jpg"); //save image

}