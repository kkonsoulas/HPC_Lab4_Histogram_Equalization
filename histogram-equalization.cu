#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define GPU_CUDA_CODE
#include "hist-equ.h"
#define BIN_SIZE 256



__global__ void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= img_size)
		return;

	int pixel = img_in[i]; //prefetching

	__shared__ int block_hist[256];
	if(threadIdx.x < 256)
		block_hist[threadIdx.x] = 0;
	__syncthreads();


	atomicAdd(&block_hist[pixel],1);

	__syncthreads();

	if(threadIdx.x <256)
	 	atomicAdd( &hist_out[threadIdx.x], block_hist[threadIdx.x]);

	return;
}


//  __global__ void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
//  							int * hist_in, int img_size, int nbr_bin,int * cdf,int * lut)
// {
	
	
//  	 //int *lut; 
//  	 //cudaMalloc(&lut,sizeof(int)*nbr_bin);            
//  	 //if(i == 0)
//  	 //	hist_in[20000] =400;
//  	 //cdf = 0;
//  	int i = threadIdx.x + blockIdx.x * blockDim.x;
//  	if(i >= img_size)
//  		return;

//  	int d=0;
// 	int min;
//  	//__shared__ int lut[BIN_SIZE];
//  	__shared__ int scanning[BIN_SIZE];

//  	/* Construct the LUT by calculating the CDF */

//  	//major unoptimised code
	
//  	/*if(threadIdx.x == 0){
//  		while(min == 0){
//  			min = hist_in[d++];
//  		}
		
//  	}
//  	__syncthreads();
//  	d = img_size - min;
// 	*/

//  	//Parallel scan 
//  	if(i < nbr_bin)
//  		scanning[threadIdx.x] = hist_in[threadIdx.x];
//  	unsigned int stride;
//  	for(stride = 1; stride < blockIdx.x; stride *= 2){
//  		__syncthreads();
 		
//  		if (stride <= threadIdx.x)
//  			scanning[threadIdx.x] += scanning[threadIdx.x - stride];
 		
//  	}

//  	__syncthreads(); 
// 	for(int k=0; k < nbr_bin; k++){
// 		cdf[k] = scanning[k];
// 	}
//  	/*if(stride <= threadIdx.x)
//  		cdf[threadIdx.x] += temp;
//  	}*/
	
	
// 		min = cdf[0];
// 		d = min - img_size;
		
// 		lut[threadIdx.x] = (int)(((float)cdf[threadIdx.x] - min)*255/d + 0.5);
// 		if(lut[threadIdx.x] < 0){
// 			lut[threadIdx.x] = 0;
// 		} 
	
//  	/* Get the result image */
//  	if(lut[img_in[i]] > 255){
//  		img_out[i] = 255;
//  	}
//  	else{
//  		img_out[i] = (unsigned char)lut[img_in[i]];
//  	}
// 	__syncthreads();
// }


// --------------------WORKING CODE---------------------//
__global__ void histogram_prefixsum(int * hist_in,int * cdf,int nbr_bin,int img_size,int TypeofAlgo){
	
	if (TypeofAlgo == 0){
		__shared__ int partialScan[BIN_SIZE];
		int i = threadIdx.x + blockIdx.x * blockDim.x;

		//// Major unoptimised code	
		if(i < nbr_bin)
			partialScan[threadIdx.x] = hist_in[threadIdx.x];
		
		for(int stride = 1; stride < blockDim.x; stride <<= 1){
			__syncthreads();

			if(threadIdx.x >= stride){
				partialScan[threadIdx.x] += partialScan[threadIdx.x - stride];
			}
		}
		__syncthreads();

		cdf[i] = partialScan[i];
	}

	if (TypeofAlgo == 1){
		__shared__ int partialScan[2 * BIN_SIZE];
		int i = threadIdx.x + blockDim.x * blockIdx.x;

		if (i < nbr_bin)
			partialScan[threadIdx.x] = hist_in[threadIdx.x];
		
		__syncthreads();
		//Redution
		for(int stride = 1; stride <= blockDim.x; stride <<= 1){
			__syncthreads();
			int index = (threadIdx.x + 1)*2*stride - 1;

			if (index < blockDim.x)
				partialScan[index] += partialScan[index - stride];
		}

		//Post-Reduction
		for(int stride = BIN_SIZE / 4; stride > 0; stride >>= 1){
			__syncthreads();
			int index = (threadIdx.x + 1)*2*stride - 1;

			if (index + stride < BIN_SIZE)
				partialScan[index + stride] += partialScan[index];
		}

		__syncthreads();
		cdf[i] = partialScan[i];
	}
}

__global__ void histogram_calcdf(int * cdf,int * lut ,int img_size){
	__shared__ int min; 
	__shared__ int d; 
	min = cdf[0];
	d = img_size - min;

	__shared__ int block_lut[BIN_SIZE];
	block_lut[threadIdx.x] = (int)(((float)cdf[threadIdx.x] - min)*255/d + 0.5);
    if(block_lut[threadIdx.x] < 0)
        block_lut[threadIdx.x] = 0;
    
	lut[threadIdx.x] = block_lut[threadIdx.x];
	return ;
}

__global__ void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
							int * lut, int img_size, int nbr_bin)
{	
	// __constant__ int quick_lut[BIN_SIZE];
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= img_size)
		return;
	int reg = lut[img_in[i]];
	if( reg > 255){
		img_out[i] = 255;
	}
	else{
		img_out[i] = (unsigned char)reg;
	}

	__syncthreads();
}
