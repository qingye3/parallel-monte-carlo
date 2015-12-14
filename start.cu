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


#define N_ATOMS 64
#define L 10.0f
#define beta 0.3
#define cellsPerSide 4
#define w 2.5f
#define nmax 10
#define BLOCK_SIZE 1024
#define n_M 10
#define sigma 0.5f
#define dimCB 8
#define MCpasses 1000

const int CPS2 = cellsPerSide*cellsPerSide;
const int CPS3 = CPS2*cellsPerSide;

#include "subsweep.h"
#include "shiftCells.h"


// Fisher Yates shuffle as described on wiki
void FY_Shuffle(int * a, int n){
	int i, randindex;
	int temp;
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
	if (ix < N_cube & iy < N_cube & iz < N_cube){
		int index = ix + iy*N_cube + iz * N_cube * N_cube;
		//if (index == 1001){ printf("Init r kernel success! "); }
		r[index] = L / 2.0 * (1.0 - float(2 * ix + 1) / N_cube);
		r[index + N_ATOMS] = L / 2.0 * (1.0 - float(2 * iy + 1) / N_cube);
		r[index + 2 * N_ATOMS] = L / 2.0 * (1.0 - float(2 * iz + 1) / N_cube);
	}
}

//void assign(float *r, float *disk,int* index, int* n, int* presum){
//	int i, cellx, celly, cellz;
//	int write_index;
//	presum[0] = 0;
//	for (i = 0; i < CPS3; i++){
//		n[i] = 0;
//	}
//	for (i = 0; i < N_ATOMS; i++){
//		cellx = int((r[i] + L / 2.0f) / L*cellsPerSide);
//		celly = int((r[i + N_ATOMS] + L / 2.0f) / L*cellsPerSide);
//		cellz = int((r[i + 2 * N_ATOMS] + L / 2.0f) / L*cellsPerSide);
//		write_index = cellx + celly*cellsPerSide + cellz*CPS2;
//		index[i] = write_index;
//		n[write_index] += 1;
//		
//	}
//	//for (i = 1; i < CPS3; i++){
//	//	presum[i] = n[i] + presum[i - 1];
//	//}
//	//for (i = 0; i < N_ATOMS; i++){
//	//	disk[presum[index[i]]] = r[i];
//	//	disk[presum[index[i]] + n[index[i]]] = r[i+N_ATOMS];
//	//	disk[presum[index[i]] + 2 * n[index[i]]] = r[i + 2 * N_ATOMS];
//	//}
//}

// assign kernel - takes the positions and assigns them a cell index
__global__ void assign(float*r, float* disk, short int*n){
	__shared__ float r_sh[N_ATOMS*3];
	__shared__ short int n_sh[CPS3];
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	float xub, xlb, yub, ylb, zub, zlb, x, y, z;
	int cellx,celly,cellz, i;
	for (i = 0; i < int(ceil(3 * N_ATOMS / (float)BLOCK_SIZE)); i++){
		if (i*BLOCK_SIZE + tid < 3 * N_ATOMS){
			// storing positions in shared memory for faster access
			// will have to change this if N_ATOMS > 2048
			r_sh[i*BLOCK_SIZE + tid] = r[i*BLOCK_SIZE + tid];
		}
	}
	__syncthreads();
	if (tid < CPS3){
		n_sh[tid] = 0;
	}
	__syncthreads(); //check for syncthreads();

	// Approach 1 - with atomic add
	/****************************************/
	/*     cannot avoid race condition      */
	/****************************************/
	//if (tid < N_ATOMS){
	//	cellx = int((r_sh[tid] + L / 2.0f) / L*cellsPerSide);
	//	celly = int((r_sh[tid + N_ATOMS] + L / 2.0f) / L*cellsPerSide);
	//	cellz = int((r_sh[tid + 2 * N_ATOMS] + L / 2.0f) / L*cellsPerSide);
	//	int write_index = cellx + celly * cellsPerSide + cellz * CPS2;
	//	disk[write_index*nmax * 3 + n_sh[write_index]] = r_sh[tid];
	//	disk[write_index*nmax * 3 + n_sh[write_index]+ nmax] = r_sh[tid];
	//	disk[write_index*nmax * 3 + n_sh[write_index] + 2 *nmax] = r_sh[tid];
	//	atomicAdd(&n_sh[write_index], 1);
	//}

	// Approach 2 - no atomic add
	/****************************************/
	/*   No race - but redundancy in checks */
	/****************************************/
	if (tid < CPS3){
		cellz = tid / CPS2;
		celly = (tid - cellz*CPS2) / cellsPerSide;
		cellx = tid - cellz*CPS2 - celly * cellsPerSide;
		xlb = cellx*w - L / 2.0f; xub = xlb + w;
		ylb = celly*w - L / 2.0f; yub = ylb + w;
		zlb = cellz*w - L / 2.0f; zub = zlb + w;
		for (i = 0; i < N_ATOMS; i++){
			x = r_sh[i];  y = r_sh[i + N_ATOMS]; z = r_sh[i + 2 * N_ATOMS];
			if ((x <= xub && x > xlb) && (y <= yub &&  y > ylb) && (z <= zub && z > zlb)){
				disk[tid*nmax * 3 + n_sh[tid]] = x;
				disk[tid*nmax * 3 + n_sh[tid] + nmax] = y;
				disk[tid*nmax * 3 + n_sh[tid] + nmax * 2] = z;
				n_sh[tid] += 1;
			}
		}
	}
	//if (tid == 0) { printf("cell 0 has %i particles\n", n_sh[0] ); }
	if (tid < CPS3){
		n[tid] = n_sh[tid];
	}
}



// Sub-sweep kernel

// cell redraw boundaries kernel
void itoa(int * r, int n){
	r[2] = n % 2;
    r[1] = (n/2) % 2;
    r[0] = (n/4) % 2;
}

void host_print_disk(float * disk, short int * n){
	for (int i = 0; i < CPS3; i++){
		for (int j = 0; j < n[i]; j++){
			printf("Position of atom %i in cell %i: %f\t%f\t%f\n", j, i, disk[nmax * 3 * i + j],
				disk[nmax * 3 * i + j + nmax], disk[nmax * 3 * i + j + 2 * nmax]);
		}
	}
}


int main(){
	// test iput to check Fisher yates
	//float a[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	//int len = 8;
	//FY_Shuffle(&a[0], len);
	//for (int i = 0; i < len; i++){
	//	printf("%f \t", a[i]);
	//}

	// declare variables
	float* d_r;
	float * d_disk;
	short int * d_n;
	float * disk;
	short int * n;
	cudaError_t state;

	int rsize = 3 * N_ATOMS * sizeof(float);
	int nsize = sizeof(short int) * CPS3;
	int disksize = sizeof(float) * 3 * nmax * CPS3;
    int cboard_index[8]= {0,1,2,3,4,5,6,7};
    int i, MC_step;
	int f;
	float d;
    int * off;
	int * d_off;

	
    off = (int *) malloc(sizeof(int) * 3);
	disk = (float *)malloc(disksize);
	n = (short int *)malloc(nsize);

	// allocate space on GPU
	cudaMalloc((void **)&d_r, rsize);
	cudaMalloc((void **)&d_disk, disksize);
	cudaMalloc((void **)&d_n, nsize);
	cudaMalloc((void **)&d_off, sizeof(int) * 3);
	
	// initialize positions
	int N_cube = int(cbrt(float(N_ATOMS)));
	int gd = int(ceil(double(N_ATOMS) / N_cube));
	dim3 gridSize(gd, gd, gd);
	dim3 blockSize(10, 10, 10);
	init_r << <gridSize, blockSize >> >(d_r, N_cube);
	state = cudaDeviceSynchronize();
	if (state != cudaSuccess){
		printf("Init r kernel failed : %s", cudaGetErrorString(state));
	}

	// Check positions generated on GPU
	//cudaMemcpy(r, d_r, rsize, cudaMemcpyDeviceToHost);
	//for (int ii = 0; ii < N_ATOMS * 3; ii++){
	//printf("%f \t", r[ii]);
	//if ((ii + 1) % N_ATOMS == 0)
	//printf("\n");
	//}

	// cell assignment
	assign << <int(ceil(float(CPS3)/BLOCK_SIZE)), BLOCK_SIZE >> >(d_r, d_disk, d_n);
	state = cudaDeviceSynchronize();
	if (state != cudaSuccess){
		printf("Assign kernel failed : %s", cudaGetErrorString(state));
	}
	//cudaMemcpy(disk, d_disk, disksize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(n, d_n, nsize, cudaMemcpyDeviceToHost);
	//host_print_disk(disk, n);
	dim3 blockSize_SS(cellsPerSide/2, cellsPerSide/2, cellsPerSide/2);
	dim3 blockSize_SC(cellsPerSide, cellsPerSide, cellsPerSide);
	for (MC_step = 0; MC_step < MCpasses; MC_step++){
		FY_Shuffle(cboard_index, dimCB);
		for (i = 0; i < dimCB; i++){
			//printf("Running checkerboard %i\n", cboard_index[i]);
			itoa(off, cboard_index[i]);
			cudaMemcpy(d_off, off, sizeof(int) * 3, cudaMemcpyHostToDevice);
			// sub-sweep kernel
			//puts(r);
			subsweep_kernel << <1, blockSize_SS >> >(d_disk, d_n, d_off);
			state = cudaDeviceSynchronize();
			if (state != cudaSuccess){
				printf("Subsweep failed : %s\n", cudaGetErrorString(state));
			}
		}
		f = rand() % 3 - 1;
		d = float(rand()) / RAND_MAX * w - w / 2.0f;
		//f = 0 ; d = w / 4;
		// cell redraw boundaries kernel
		shiftCells << <1, blockSize_SC >> >(d_disk, d_n, f, d);
		state = cudaDeviceSynchronize();
		if (state != cudaSuccess){
			printf("Shiftcells failed : %s\n", cudaGetErrorString(state));
		}
	}
	cudaMemcpy(disk, d_disk, disksize, cudaMemcpyDeviceToHost);
	cudaMemcpy(n, d_n, nsize, cudaMemcpyDeviceToHost);
	host_print_disk(disk,n);

    free(off);
	cudaFree(d_off);
	cudaFree(d_r);
	cudaFree(d_disk);
	cudaFree(d_n);
// Have fun! Grab a drink! 
// changed a line
}

