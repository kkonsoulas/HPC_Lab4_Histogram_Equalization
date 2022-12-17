OPTFLAGS = -g -Xcompiler -O3

CFLAGS = $(OPTFLAGS)

NVCCFLAGS = $(OPTFLAGS) -G


all: objects
	nvcc hist.o contrast.o main.o -o gHist -L/usr/local/cuda/lib64 -lcudart -lcuda -lm

cpu: 
	gcc -O3 contrast-enhancement.c histogram-equalization.c main.c -o hist 	


objects:
	nvcc $(NVCCFLAGS) -c histogram-equalization.cu -o hist.o
	nvcc $(NVCCFLAGS) -c contrast-enhancement.cu -o contrast.o
	nvcc $(NVCCFLAGS) -c main.cu -o main.o


diff: gHist hist
	./gHist uth.pgm guth.out
	./hist uth.pgm cuth.out
	diff cuth.out guth.out

planet:
	./gHist ../Images/planet_surface.pgm planet_surface.gpu
	./hist ../Images/planet_surface.pgm planet_surface.cpu

clean:
	rm *.o