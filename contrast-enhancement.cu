#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define GPU_CUDA_CODE
#include "hist-equ.h"

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{

    //init gpu
    int* d_hist;
    cudaError_t cudaError = cudaMalloc((void**) &d_hist,256 * sizeof(int));
    cudaCheckError();
    dim3 block_dim , grid_dim;
    int img_size = img_in.w  *img_in.h;
    int* cdf;
    cudaError = cudaMalloc((void**) &cdf,256 * sizeof(int));
    int* lut;
    cudaError = cudaMalloc((void**) &lut,256 * sizeof(int));
    
    //image larger than a block
    if(img_size > BLOCK_SIZE){
        block_dim.x = BLOCK_SIZE;
        grid_dim.x = img_size / BLOCK_SIZE;
        grid_dim.x+= img_size % BLOCK_SIZE ? 1 : 0; //image does not fite perfect
    }
    else{//image smaller than a block
        block_dim.x = img_size;
        grid_dim.x = 1;
    }
    // block_dim.x = img_size > BLOCK_SIZE ? BLOCK_SIZE : img_size;
    // grid_dim.x = img_size > BLOCK_SIZE ? img_size / BLOCK_SIZE : 1;
    // grid_dim.x+= img_size % BLOCK_SIZE ?  : 0;
    // int remainingThreads = img_size > BLOCK_SIZE ? img_size % BLOCK_SIZE : 0;





    //init
    PGM_IMG result;
    result.w = img_in.w;
    result.h = img_in.h;
    cudaMallocHost(&(result.img),result.w * result.h * sizeof(unsigned char));
    cudaCheckError();

    //timer
    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    

    cudaMemset(d_hist,0,sizeof(int)*256);
    cudaCheckError();
    histogram<<<grid_dim,block_dim>>>(d_hist, img_in.img, img_in.h * img_in.w, 256);
    
    //----WORKING-CODE------//
    cudaDeviceSynchronize();
    cudaCheckError();
    histogram_prefixsum<<< 1, 256>>>(d_hist,cdf,256,  img_in.h * img_in.w);
    cudaDeviceSynchronize();
    cudaCheckError();
    histogram_calcdf<<< 1 , 256 >>>(cdf,lut,img_in.h * img_in.w);
    cudaCheckError();
    //-----------------------//

    cudaDeviceSynchronize();
    cudaCheckError();
    histogram_equalization<<<grid_dim,block_dim>>>(result.img ,img_in.img ,lut,result.w*result.h , 256);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    //timer
    cudaCheckError();

    printf("Elapsed time in GPU:  %3.12f ms \n", time);
    return result;
}
