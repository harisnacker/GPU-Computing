# GPU-Computing

# GPU Lab2
## Preparations 
* Login our GPU server(***co25.ics.ele.tue.nl***) with your account
* Download the codes
```
git clone https://gitlab.tue.nl/20177018/gpu_2019 
```
* Add CUDA toolkit to your PATH
```
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64{LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
Type
```
which nvcc
```
You should see 
```
/usr/local/cuda-10.1/bin/nvcc
```
* Try an example
```
cd gpu_2019
cd MatrixMultiplication/NaiveGPU
make
./app
```
Then you should see something like
```
Matrix Size : 512 x 512
GPU Device 0: "GeForce RTX 2070" with compute capability 7.5

Start CPU processing
Naive CPU processing time: 386.702 ms
block Dim  32 x 32
Grid Dim  16 x 16
start GPU processing
CUDA kernel processing time: 1.3209 ms
speed-up : time_CPU/time_GPU 292.758
Total Errors = 0
```

## Introduction
In the lab2 you will be asked to port a C-program consisting of a couple of **image processing filters** to a GPU with the CUDA programming framework. The goal is to achieve the highest speed by using all kinds of GPU optimization techniques you learned from lectures or online materials. The student who acheves highest speed(on our GPU server) will win **a bottle of beer**.
* Note : If you have no access to a Nvidia GPU, you may use our GPU server. You may try OpenCL instead of CUDA if you want, but you will get less support. These [slides](https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/opencl/opencl-03-basics.pdf?__blob=publicationFile) could be a good material for you if you go to OpenCL.
## Learn basic CUDA programming concepts by trying Mat Mul CUDA codes
* Good matrials: **Processional CUDA C programming** and **Cuda C Programming guide**
  * Google them and you could find the PDF version.
  * These materials cover many advanced topics for CUDA programming, the more you read/learn, the more likely you will be able to achieve higher speed-up. Please keep in mind that you are only expected to spend 30 hours on this lab. First try to get your program **work** and then think how to improve.
* Cookbook from previous year: [Matrix Multiplication](https://ecatue.gitlab.io/gpu2018/pages/Cookbook/matrix_multiplication_cuda.html) . **Just read the material. Do not run the old codes, use new codes in this repo**
    * Folder MatrixMultiplication contains several techniques for optimizations.
        * AvoidSharedBankConflict
        * ComputationOptimization
        * CUDAtiling
        * GlobalMemoryCoalescing
        * GPUunroll
        * NaiveGPU
        * Prefetching
    * Run the codes, and modify the matrix size or any other parameters if you want.
    * The Codes were implemented several years ago. Although the optimization techniques are correct, the results you will get could break your expectations. Think about why.
    * Feel free to try any new optimization techniques you learned or your new ideas if you want.
* [CUDA profiler](https://docs.nvidia.com/cuda/profiler-users-guide/index.html) is an important tool for analyzing your CUDA program. With the help of CUDA profiler, you can localize the bottleneck of your Kernels and choose the most suitable optimization techniques. Read this [post](https://gist.github.com/sonots/5abc0bccec2010ac69ff74788b265086) to learn how to use profiler. If you want to use 
Visual Profiler with our GPU server, then you need to install Visual Profiler on your own machine and load the outputs(prof.nvvp) on your own machine because our GPU server does not have GUI.



## Assignment: Accelerate the given C program.
* Go to folder **Assignment2**
* You are given four image processing filters (CPU code)
    * rgb2grayscale
    * Histograms calculation
    * Contrast Enhancement
    * Smoothing
* The CUDA code for rgb2grayscale has been provided. 
    * Feel free to modify it if you can see optimization chances
* Profile each CUDA code independently (nvprof)
    * For each kernel, you should profile it.
* Profile the program
    * First read the C-program(test.cpp) to understand the pipeline.
    * Combine your CUDA kernels together to do the same thing of C-program
    * Input : RGB image. Output : image after Smoothing. 
    * Pipeline: RGB -> grayscale -> Histograms->  Contrast Enhancement - > Smoothing
        


## Grading:
Final grade will based on
* The correctness of your program/kernels (30%)
    * Make sure your program does a correct job, otherwise you will get **0** point.
* You report (20%)
    * Explain your implementations (max 4 pages)
        * The speed-up compared with original C version
            * Speed-up of each kernel and profile CUDA kernels
            * Speed-up of the processing pipeline (see test.cpp)
        * Which optimization techniques you applied
        * Runtime spent on communication(CPU-GPU) and computation
        * The speed-up of your program (see below)
* The speed-up of **the image processing pipeline(not each kernel)** you get compared with other students' results (50%). Note, combining four kernels into a single large one is not allowed.
    * Submit your codes.
        * You only need to submit **kernel.h** **kernel.cu** **main.cu** and **Makefile**. Do not change other files
        * Use **makefile** to organize your project
        * You program should output your smoothing_gpu.jpg
        * Make sure we can compile you CUDA program by  **make cuda** command.
    * We will measure your speed-up again ourself on server.
        * If you did not submit correct makefile project, then we may not be able to grade your codes
