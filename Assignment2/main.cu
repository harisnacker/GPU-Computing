#include <cuda_runtime.h>
#include "CImg.h"
#include <iostream>
#include "kernel.h"
#include "kernelcpu.h"
#define BLOCKSIZE 32
using namespace std;
using namespace cimg_library;


int compute_diff(unsigned char * res_cpu, unsigned char * res_gpu, unsigned long size){
  int res = 0;
  for(int i = 0;i < size; i++){
    res += res_cpu[i] - res_gpu[i];
  }
  return res;
}


int main()
{
    //load image
    CImg<unsigned char> src("cat2.jpg"); // we use cat2.jpg to grade
    int width = src.width();
    int height = src.height();
    unsigned long size = src.size();

    //create pointer to image
    unsigned char *h_src = src.data();
    
    CImg<unsigned char> dst(width, height, 1, 1);
    unsigned char *h_dst = dst.data();
    
    
    // for contrast enhancemant
    CImg<unsigned char> contrast(width, height, 1, 1);
    unsigned char *h_contrast = contrast.data();
    
    // for contrast enhancemant
    CImg<unsigned char> contrast_gpu(width, height, 1, 1);
    unsigned char *hgpu_contrast = contrast_gpu.data();

    // for smoothing
    CImg<unsigned char> smoothing(width, height, 1, 1);
    unsigned char *h_smoothing = smoothing.data();
    //unsigned char *hgpu_smoothing = smoothing.data();
    
    // for smoothing_gpu
    CImg<unsigned char> smoothing_gpu(width, height, 1, 1);
    unsigned char *hgpu_smoothing = smoothing_gpu.data();
    
    //device variables
    unsigned char *d_src;
    unsigned char *d_dst;
    unsigned char *d_contrast;
    unsigned char *d_smoothing;
    int *d_hist ;
    
    int* hist = new int[256]();
    //int* hist2 = new int[256]();

    cudaEvent_t start; // to record processing time
    cudaEvent_t stop;
    float msecTotal;
    float msecTotal2;


    // rgb2gray CPU kernel 1    
    std::cout << "Start CPU processing" << std::endl;
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // hisgram cpu
    unsigned char *cpu_ref = new unsigned char [width*height];
    rgb2gray_cpu(h_src,cpu_ref,width,height);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    float cpu_time = msecTotal;  
    //std::cout <<"width " << width <<std::endl;
    //std::cout <<"height " << height <<std::endl;
    std::cout <<"CPU processing time: " << cpu_time << " ms" <<std::endl;  
    
    
    //HISTGRAM CPU kernel 2   
    std::cout << "Start CPU2 processing" << std::endl;
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    //unsigned char *cpu_ref = new unsigned char [width*height];
    histgram_cpu(hist,cpu_ref,width,height);     
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    float cpu_time2 = msecTotal;
    std::cout <<"CPU2 processing time: " << cpu_time2 << " ms" <<std::endl;
   
   
   
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
	  //std::cout <<"min value" << min <<std::endl;
	  //std::cout <<"max value" << max <<std::endl;  
      
      
    //Contrast enhancement CPU kernel 3           
    std::cout << "Start CPU3 processing" << std::endl;
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // contrast enhancement
    ContrastEnhancement_cpu(cpu_ref,h_contrast,width,height,min,max);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    float cpu_time3 = msecTotal;
    std::cout <<"CPU processing3 time: " << cpu_time3 << " ms" <<std::endl;
    

    //Smoothing cpu kernel 4   
    std::cout << "Start CPU4 processing" << std::endl;
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // smoothing_cpu function call
    Smoothing_cpu(h_contrast,h_smoothing,width,height);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    float cpu_time4 = msecTotal;
    std::cout <<"CPU4 processing time: " << cpu_time4 << " ms" <<std::endl;
    
 
 
    // GPU Implementation 
    //rgb2gray GPU Kernel 1     
    std::cout << "Start GPU processing" << std::endl;
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 

    cudaMalloc((void**)&d_src, size);
    cudaMalloc((void**)&d_dst, width*height*sizeof(unsigned char));
    
    //copy host to device
    cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice);

    //launch the kernel
    dim3 blkDim (BLOCKSIZE, BLOCKSIZE, 1);
    dim3 grdDim ((width + BLOCKSIZE -1)/BLOCKSIZE, (height + BLOCKSIZE - 1)/BLOCKSIZE, 1);
    rgb2gray<<<grdDim, blkDim>>>(d_src, d_dst, width, height);

    //wait until kernel finishes
    cudaDeviceSynchronize();

    
    //copy back the result to CPU
    cudaMemcpy(h_dst, d_dst, width*height, cudaMemcpyDeviceToHost);
    
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);    
    float gpu_time = msecTotal;

    int res = compute_diff(cpu_ref,h_dst,width*height);
    
    cudaFree(d_src);
    cudaFree(d_dst);

    std::cout << "diff cpu and gpu " << res <<std::endl; // do not change this
    std::cout <<"GPU processing time: " << gpu_time << " ms" <<std::endl; // do not change this
    std::cout << "achieved speedup: " << cpu_time/gpu_time<<std::endl; // do not change this

    // add other three kernels here
    // clock starts -> copy data to gpu -> kernel1 -> kernel2->kernel3->kernel 4 ->copy result to cpu -> clock stops
    


    //Histgram GPU kernel 2	
	  std::cout << "Start GPU2 processing" << std::endl;
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 

    cudaMalloc((void**)&d_hist, 256);
    cudaMalloc((void**)&d_dst, 3*width*height*sizeof(unsigned char));

    cudaMemcpy(d_dst, h_dst, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist, hist, 256, cudaMemcpyHostToDevice);
    

    //launch the kernel
    dim3 blkDim2 (BLOCKSIZE, BLOCKSIZE, 1);
    dim3 grdDim2 ((width + BLOCKSIZE -1)/BLOCKSIZE, (height + BLOCKSIZE - 1)/BLOCKSIZE, 1);
    histgram_gpu<<<grdDim2, blkDim2>>>(d_hist, d_dst, width, height);

    
    //wait until kernel finishes
    cudaDeviceSynchronize();

    
    //copy back the result to CPU
    cudaMemcpy(h_dst, d_dst, width*height, cudaMemcpyDeviceToHost);
    
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal2, start, stop);
    float gpu_time2 = msecTotal2;
    
    res = compute_diff(cpu_ref,h_dst,width*height);

    cudaFree(d_hist);
    cudaFree(d_dst);

    std::cout << "diff cpu and gpu " << res <<std::endl; // do not change this
    std::cout <<"GPU2 processing time: " << gpu_time2 << " ms" <<std::endl; // do not change this
    std::cout << "achieved speedup: " << cpu_time2/gpu_time2<<std::endl; // do not change this
    
    
    
    
    //kernel 3	
	  std::cout << "Start GPU3 processing" << std::endl;
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 

    cudaMalloc((void**)&d_dst, 3*width*height*sizeof(unsigned char));
    cudaMalloc((void**)&d_contrast, width*height*sizeof(unsigned char));

    cudaMemcpy(d_dst, h_dst, size, cudaMemcpyHostToDevice);

    //launch the kernel 3
    dim3 blkDim3 (BLOCKSIZE, BLOCKSIZE, 1);
    dim3 grdDim3 ((width + BLOCKSIZE -1)/BLOCKSIZE, (height + BLOCKSIZE - 1)/BLOCKSIZE, 1);
    ContrastEnhancement_gpu<<<grdDim3, blkDim3>>>(d_dst,d_contrast,width,height,min,max);
    
    
    //wait until kernel finishes
    cudaDeviceSynchronize();

    
    //copy back the result to CPU
    cudaMemcpy(hgpu_contrast, d_contrast, width*height, cudaMemcpyDeviceToHost);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    float gpu_time3 = msecTotal;
      
    res = compute_diff(h_contrast,hgpu_contrast,width*height);

    cudaFree(d_contrast);
    cudaFree(d_dst);

    std::cout << "diff cpu and gpu " << res <<std::endl; // do not change this
    std::cout <<"GPU3 processing time: " << gpu_time3 << " ms" <<std::endl; // do not change this
    std::cout << "achieved speedup: " << cpu_time3/gpu_time3<<std::endl; // do not change this
    



    //kernel 4    
   	std::cout << "Start GPU4 processing" << std::endl;
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 

    cudaMalloc((void**)&d_smoothing, width*height*sizeof(unsigned char));
    cudaMalloc((void**)&d_contrast, width*height*sizeof(unsigned char));

    cudaMemcpy(d_contrast, hgpu_contrast, width*height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_smoothing, hgpu_smoothing, width*height, cudaMemcpyHostToDevice);
    
    //launch the kernel 4
    dim3 blkDim4 (BLOCKSIZE, BLOCKSIZE, 1);
    dim3 grdDim4 ((width + BLOCKSIZE -1)/BLOCKSIZE, (height + BLOCKSIZE - 1)/BLOCKSIZE, 1);
    Smoothing_gpu<<<grdDim4, blkDim4>>>(d_contrast,d_smoothing,width,height);
    
    //wait until kernel finishes
    cudaDeviceSynchronize();

    
    //copy back the result to CPU
    cudaMemcpy(hgpu_smoothing, d_smoothing, width*height, cudaMemcpyDeviceToHost);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    float gpu_time4 = msecTotal;
      
    res = compute_diff(h_smoothing,hgpu_smoothing,width*height);

    cudaFree(d_smoothing);
    cudaFree(d_contrast);

    std::cout << "diff cpu and gpu " << res <<std::endl; // do not change this
    std::cout <<"GPU4 processing time: " << gpu_time4 << " ms" <<std::endl; // do not change this
    std::cout << "achieved speedup: " << cpu_time4/gpu_time4<<std::endl; // do not change this



    //you need to save your final output, we need to measure the correctness of your program
    //read test.cpp to learn how to save a image
    //contrast.save("contrast_cpu.jpg");
    //smoothing.save("smoothing_cpu.jpg");
    
    //contrast_gpu.save("contrast_gpu.jpg");
    
    smoothing_gpu.save("smoothing_gpu.jpg"); 

    return 0;
}