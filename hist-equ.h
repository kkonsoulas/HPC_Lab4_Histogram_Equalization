#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

#ifdef GPU_CUDA_CODE
#include <cuda.h>


#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}
#define BLOCK_SIZE 1024
#define CONSTANT_MEMORY_SIZE (8388608 )
// #define CONSTANT_MEMORY_GRID (CONSTANT_MEMORY_SIZE / BLOCK_SIZE)
#define CONSTANT_MEMORY_GRID (8192)


void __global__ histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void __global__ histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin);
void __global__ histogram_prefixsum(int * hist_in,int * cdf,int nbr_bin,int img_size/*,int TypeofAlgo*/);
void __global__ histogram_calcdf(int * cdf,int * lut , int img_size);



#else
void  histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void  histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin);
#endif

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    



PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);





//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);

#endif
