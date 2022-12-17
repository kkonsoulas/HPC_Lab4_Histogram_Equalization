#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define GPU_CUDA_CODE
#include "hist-equ.h"

// __device__ __constant__ unsigned char image_chunck[CONSTANT_MEMORY_SIZE];



PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{

    //init gpu
    int* d_hist;
    cudaError_t cudaError = cudaMalloc((void**) &d_hist,256 * sizeof(int));
    int* cdf;
    cudaError = cudaMalloc((void**) &cdf,256 * sizeof(int));
    int* lut;
    cudaError = cudaMalloc((void**) &lut,256 * sizeof(int));
    unsigned char* image_chunck;
    cudaError = cudaMalloc((void**) &image_chunck,CONSTANT_MEMORY_SIZE * sizeof(unsigned char));
    cudaCheckError();
    // __constant__ int *quick_lut;
    // cudaError = cudaMalloc((void**) &quick_lut,256 * sizeof(int));
    int img_size = img_in.w  *img_in.h;
    dim3 block_dim , grid_dim;
    int i;
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


    //Tilling Execution
    // for(i = 0; i <((img_in.w  *img_in.h) / CONSTANT_MEMORY_SIZE) ;i++){
    //     cudaMemcpy(image_chunck,&img_in.img[i*CONSTANT_MEMORY_SIZE],CONSTANT_MEMORY_SIZE*sizeof(unsigned char),cudaMemcpyHostToDevice);
    //     // cudaMemcpyToSymbol(image_chunck,&img_in.img[i*CONSTANT_MEMORY_SIZE],CONSTANT_MEMORY_SIZE*sizeof(unsigned char));
    //     cudaCheckError();
    //     // printf("i: %d\n",i);
    //     cudaDeviceSynchronize();
    //     histogram<<<grid_dim,BLOCK_SIZE>>>(d_hist, image_chunck, CONSTANT_MEMORY_SIZE, 256);
    //     img_size -= CONSTANT_MEMORY_SIZE;
    // }
    // if(img_size > 0){
    //     cudaMemcpy(image_chunck,&img_in.img[i*CONSTANT_MEMORY_SIZE],img_size*sizeof(unsigned char),cudaMemcpyHostToDevice);
    //     // cudaMemcpyToSymbol(image_chunck,&img_in.img[i*CONSTANT_MEMORY_SIZE],img_size*sizeof(unsigned char));
    //     int grid_size = (img_size / BLOCK_SIZE) + 1;
    //     histogram<<<grid_size,1024>>>(d_hist, image_chunck, img_size, 256);
    // }
    // img_size = img_in.w  *img_in.h;
    
    //Non tilling execution
    histogram<<<grid_dim,block_dim>>>(d_hist, img_in.img, img_in.h * img_in.w, 256);

    //----WORKING-CODE------//
    cudaDeviceSynchronize();
    cudaCheckError();
    histogram_prefixsum<<< 1, 256>>>(d_hist,cdf,256,  img_in.h * img_in.w, 0);
    cudaDeviceSynchronize();
    cudaCheckError();
    histogram_calcdf<<< 1 , 256 >>>(cdf,lut,img_in.h * img_in.w);
    cudaCheckError();
    //-----------------------//

    cudaDeviceSynchronize();
    // cudaMemcpy(quick_lut,lut,256*sizeof(int),cudaMemcpyHostToDevice);
    // cudaMemcpy(&minimum,cdf,sizeof(int),cudaMemcpyDeviceToDevice);
    
	// cudaMemcpyToSymbol(quick_lut,lut,256*sizeof(int));
	// cudaMemcpyToSymbol(&minimum,cdf,sizeof(int),0,cudaMemcpyDeviceToDevice);

    cudaCheckError();
    histogram_equalization<<<grid_dim,block_dim>>>(result.img ,img_in.img ,lut,result.w*result.h , 256);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    //timer
    cudaCheckError();
    printf("Elapsed time in GPU:  %3.12f ms \n", time);

    cudaFree(lut);
    cudaFree(cdf);
    cudaFree(image_chunck);
    cudaFree(d_hist);
    return result;
}
