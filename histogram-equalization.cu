#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define GPU_CUDA_CODE
#include "hist-equ.h"
#define BIN_SIZE 256


__global__ void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockIdx.x*gridDim.x;
	if(i >= img_size)
		return;
	// if(BLOCK_SIZE-BIN_SIZE >= threadIdx.x)
	// if(i < nbr_bin)
		// hist_out[i] = 0;
	__shared__ int block_hist[256];
	if(threadIdx.x < 256)
		block_hist[threadIdx.x] = 0;
	__syncthreads();

	//while(i < img_size){
		atomicAdd(&block_hist[img_in[i]],1);
	//	i += stride;
	//}
	__syncthreads();

	if(threadIdx.x <256)
	 	atomicAdd( &hist_out[threadIdx.x], block_hist[threadIdx.x]);
	// hist_out[threadIdx.x % 256] = 1;
	// hist_out[img_in[i]] ++;
	// hist_out + img_in[x]
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
 
__global__ void histogram_prefixsum(int * hist_in,int * cdf,int nbr_bin,int img_size){
	__shared__ int partialScan[BIN_SIZE];
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	// if(i >= img_size)
	// 	return;
	//// Major unoptimised code	
	if(i < nbr_bin)
		partialScan[threadIdx.x] = hist_in[threadIdx.x];
	
	for(int stride = 1; stride < blockDim.x; stride *= 2){
		__syncthreads();

		if(threadIdx.x >= stride){
			partialScan[threadIdx.x] += partialScan[threadIdx.x - stride];
		}
	}
	__syncthreads();

	// for(int k=0; k<nbr_bin; k++){
		cdf[i] = partialScan[i];
	// }	

}

__global__ void histogram_calcdf(int * cdf,int * lut ,int img_size){
	int min = cdf[0];
	int d = img_size - min;
	lut[threadIdx.x] = (int)(((float)cdf[threadIdx.x] - min)*255/d + 0.5);
    if(lut[threadIdx.x] < 0)
        lut[threadIdx.x] = 0;
    
	return ;
}

__global__ void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
							int * lut, int img_size, int nbr_bin)
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(lut[img_in[i]] > 255){
		img_out[i] = 255;
	}
	else{
		img_out[i] = (unsigned char)lut[img_in[i]];
	}

	__syncthreads();
}
