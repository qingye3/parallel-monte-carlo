#include <ctime>
#include "cuda_runtime.h"
#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <string.h>
#include <thrust\sort.h>


#define N_ATOMS 64
#define L 4.2323167f
#define beta 0.3
#define cellsPerSide 10

#define BLOCK_SIZE 1024


// Fisher Yates shuffle as described on wiki
void FY_Shuffle(float * a, int n){
	int i, randindex;
	float temp;
	srand(time(NULL));
	for (i = n - 1; i > 0; i--){
		randindex = rand() % (i + 1);
		temp = a[n - i];
		a[n - i] = a[randindex];
		a[randindex] = temp;
	}
}


__global__ void init_r(float* r, int N_cube){
	int ix = threadIdx.x + blockDim.x* blockIdx.x;
	int iy = threadIdx.y + blockDim.y* blockIdx.y;
	int iz = threadIdx.z + blockDim.z* blockIdx.z;
	int index = ix + iy*N_cube + iz * N_cube * N_cube;
	r[index] = L / 2.0 * (1.0 - float(2 * ix + 1) / N_cube);
	r[index + N_ATOMS] = L / 2.0 * (1.0 - float(2 * iy + 1) / N_cube);
	r[index + 2 * N_ATOMS] = L / 2.0 * (1.0 - float(2 * iz + 1) / N_cube);
}

void assign(float *r, float *disk,int* index, int* n){
	int i, cellx, celly, cellz;
	int write_index;
	int CPS2 = cellsPerSide*cellsPerSide;
	int CPS3 = CPS2 * cellsPerSide;
	int presum[cellsPerSide*cellsPerSide*cellsPerSide];
	presum[0] = 0;
	for (i = 0; i < CPS3; i++){
		n	[i] = 0;
	}
	for (i = 0; i < N_ATOMS; i++){
		cellx = int((r[i] + L / 2.0f) / L*cellsPerSide);
		celly = int((r[i + N_ATOMS] + L / 2.0f) / L*cellsPerSide);
		cellz = int((r[i + 2 * N_ATOMS] + L / 2.0f) / L*cellsPerSide);
		write_index = cellx + celly*cellsPerSide + cellz*CPS2;
		index[i] = write_index;
		n[write_index] += 1;
		
	}
	for (i = 1; i < CPS3; i++){
		presum[i] += 3*presum[i - 1];
	}
	for (i = 0; i < N_ATOMS; i++){
		disk[presum[index[i]]] = r[i];
		disk[presum[index[i]] + n[index[i]]] = r[i+N_ATOMS];
		disk[presum[index[i]] + 2 * n[index[i]]] = r[i + 2 * N_ATOMS];
	}

}

// assign kernel - takes the positions and assigns them a cell index
//__global__ void assign(float*r, float* disk, int*n, int len){
//	__shared__ float r_sh[N_ATOMS * 3];
//	int tid = threadIdx.x + blockIdx.x*blockDim.x;
//	float xub, xlb, yub, ylb, zub, zlb;
//	int write_index,cellx,celly,cellz;
//	int CPS2 = cellsPerSide*cellsPerSide;
//	int CPS3 = CPS2 * cellsPerSide;
//
//	__shared__ int n_sh[cellsPerSide*cellsPerSide*cellsPerSide];
//
//	if (tid < 3 * N_ATOMS){
//		// storing positions in shared memory for faster access
//		// will have to change this if N_ATOMS > 2048
//		r_sh[tid] = r[tid];
//	}
//	__syncthreads();
//	float * r;
//	if (tid < CPS3){
//		n_sh[tid] = 0;
//		cellz = int(tid / CPS2);
//		celly = int((tid - cellz*CPS2)/cellsPerSide);
//		cellz = int(tid - cellz*CPS2 * celly * cellsPerSide);
//		for (int i = 0; i < N_ATOMS; i++){
//			if (r_sh[i]<N_ATOMS)
//		}
//		xlb = L / tid;
//		xub = xlb + L / cellsPerSide;
//		ylb = L / tid
//	}
//}

// Sub-sweep kernel

// cell redraw boundaries kernel


int main(){
	// test input to check Fisher yates
	float a[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	int len = 8;
	FY_Shuffle(&a[0], len);
	for (int i = 0; i < len; i++){
		printf("%f \t", a[i]);
	}

	// declare variables
	float * r;
	float * d_r;
	unsigned int * d_disk;
	unsigned int * d_n;
	int size = 3 * N_ATOMS * sizeof(float);
	const int sizeint = 2, nmax = 8;
	int nsize = sizeint * cellsPerSide * cellsPerSide * nmax;

	// allocate space on CPU
	r = (float *)malloc(size);

	// allocate space on GPU
	cudaMalloc((void **)d_r, size);
	cudaMalloc((void **)d_disk, nsize * cellsPerSide);
	cudaMalloc((void **)d_n, nsize);

	// initialize positions
	int N_cube = int(cbrt(float(N_ATOMS)));
	int gd = int(ceil(double(N_ATOMS) / N_cube));
	dim3 gridSize(1, 1, 1);
	dim3 blockSize(N_cube, N_cube, N_cube);
	init_r << <gridSize, blockSize >> >(d_r, N_cube);

	// copy positions to GPU
	cudaMemcpy(d_r, r, size, cudaMemcpyHostToDevice);

	assign << <int(ceil((float)N_ATOMS * 3 / BLOCK_SIZE)), BLOCK_SIZE >> >(d_r, d_disk, 3 * N_ATOMS);
	cudaDeviceSynchronize();
	// problem with sorting is that the order of x, y, z  positions may get interchanged!
	// We may store x, y and z as separate arrays but that too does not ensure that we read the x, y ,z of same atom!
	thrust::sort_by_key(d_cellindex, d_cellindex + 3 * N_ATOMS, d_r);

	// sub-sweep kernel

	// cell redraw boundaries kernel

	// memcpy results to CPU

	// Have fun! Grab a drink! 
}

