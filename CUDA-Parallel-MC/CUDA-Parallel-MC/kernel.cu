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
#include "device_atomic_functions.h"

#define N_ATOMS 200
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

__device__ short int get_cellindex(short int cellx, short int celly, short int cellz){
	return cellx + celly * cellsPerSide + cellz * CPS2;
}

__global__ void make_nl(short int* nl){
	short int cellx, celly, cellz;
	short int nb_cellx, nb_celly, nb_cellz;
	short int cell_index, write_index;
	short int i, j, k;
	int p[3] = { 0, -1, 1 };
	cellx = threadIdx.x + blockIdx.x*blockDim.x;
	celly = threadIdx.y + blockIdx.y*blockDim.y;
	cellz = threadIdx.z + blockIdx.z*blockDim.z;
	cell_index = get_cellindex(cellx, celly, cellz);
	for (i = 0; i < 3; i++){
		for (j = 0; j < 3; j++){
			for (k = 0; k < 3; k++){
				nb_cellx = cellx + p[k];
				if (nb_cellx < 0){ nb_cellx = cellsPerSide-1; }
				if (nb_cellx >= cellsPerSide){ nb_cellx = 0; }

				nb_celly = celly + p[j];
				if (nb_celly < 0){ nb_celly = cellsPerSide-1; }
				if (nb_celly >= cellsPerSide){ nb_celly = 0; }

				nb_cellz = cellz + p[i];
				if (nb_cellz < 0){ nb_cellz = cellsPerSide-1; }
				if (nb_cellz >= cellsPerSide){ nb_cellz = 0; }
				write_index = i * 9 + j * 3 + k;
				nl[cell_index * 27 + write_index] = get_cellindex(nb_cellx, nb_celly, nb_cellz);
			}
		}
	}
}

__global__ void init_r(float* r, int N_cube){
	int ix = threadIdx.x + blockDim.x* blockIdx.x;
	int iy = threadIdx.y + blockDim.y* blockIdx.y;
	int iz = threadIdx.z + blockDim.z* blockIdx.z;
	int index = ix + iy*N_cube + iz * N_cube * N_cube;
	if (ix < N_cube && iy < N_cube && iz < N_cube && index<N_ATOMS){
		//if (index == 1001){ printf("Init r kernel success! "); }
		r[index] = L / 2.0 * (1.0 - float(2 * ix + 1) / N_cube);
		r[index + N_ATOMS] = L / 2.0 * (1.0 - float(2 * iy + 1) / N_cube);
		r[index + 2 * N_ATOMS] = L / 2.0 * (1.0 - float(2 * iz + 1) / N_cube);
	}
}

__global__ void assign(float*r, float* disk, short int*n){
	__shared__ float r_sh[N_ATOMS * 3];
	__shared__ short int n_sh[CPS3];
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	float xub, xlb, yub, ylb, zub, zlb, x, y, z;
	int cellx, celly, cellz, i;
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

__device__ void calc_presum(short int*n, short int * presum, short int len){
	//TODO: implement this
	presum[0] = 0;
	for (int i = 1; i < len; i++){
		presum[i] = presum[i - 1] + n[i-1];
	}
}

__device__ void print_ldisk(float* ldisk, int len, int len_line_break){
	for (int i = 0; i < len; i++){
		if (i%len_line_break == 0){
			printf("\n \n");
		}
		printf("%f\t", ldisk[i]);
	}
}

__device__ int random_int(curandState_t * random_state, int range) {
	return ((int)curand_uniform(random_state)) % (range + 1);
}


__device__ void swap(float *a, float *b){
	float temp = *a;
	*a = *b;
	*b = temp;
}


__device__ void random_shuffle(float * ldisk, int atom_counts, int totalAtoms, curandState_t * random_state){
	int j;
	for (int i = atom_counts - 1; i >= 0; i--){
		j = ((int)curand_uniform(random_state)) % (i + 1);
		for (int dim = 0; dim < 3; dim++){
			swap(ldisk + i + dim*totalAtoms, ldisk + j + dim*totalAtoms );
		}
	}
}

__device__ void make_move(float * proposed_move, float * old_pos, curandState_t * random_state)
{
	for (int dim = 0; dim < 3; dim++){
		proposed_move[dim] =  old_pos[dim] + curand_normal(random_state) * sigma;
	}

}

__device__ bool out_of_bound(float * proposed_move, int cell_x, int cell_y, int cell_z){
	float x = cell_x * w - L / 2 + w / 2;
	float y = cell_y * w - L / 2 + w / 2;
	float z = cell_z * w - L / 2 + w / 2;
	//if (threadIdx.x == 1 && threadIdx.y == 0 && threadIdx.z == 0){ print_3f(proposed_move); }
	if (((proposed_move[0] - x) > w / 2) || ((proposed_move[0] - x) < -w / 2)){
		return true;
	}
	if (((proposed_move[1] - y) > w / 2) || ((proposed_move[1] - y) < -w / 2)){
		return true;
	}
	if (((proposed_move[2] - z) > w / 2) || ((proposed_move[2] - z) < -w / 2)){
		return true;
	}
	return false;
}


__global__ void subSweep(float*disk, short int* nl, short int* n){
	//__shared__ int currWriteIndex;
	__shared__ short int presum[27];
	__shared__ float ldisk[27*nmax*3]; // format : all x of all cells followed by y then z and then garbage
	__shared__ short int nlist_s[27];
	__shared__ short int localn_s[27];
	__shared__ float e_old[1024];
	__shared__ float e_new[1024];
	__shared__ int totalAtoms;
	__shared__ float new_pos[3];
	__shared__ bool rejected;
	float old_pos[3], r_old, r_new, del_old, del_new, power6;
	int tx, ty , tz, tindex, write_index, diskRead_index, dim;
	short int cellx, celly, cellz;
	bool accepted;
	tx = threadIdx.x;
	ty = threadIdx.y;
	tz = threadIdx.z;
	tindex = tx + ty*blockDim.x + tz*blockDim.x*blockDim.x;
	if (tindex < 27){
		nlist_s[tindex] = nl[tindex];
		localn_s[tindex] = n[nlist_s[tindex]];
	}
	e_old[tindex] = 0.0f;
	e_new[tindex] = 0.0f;
	if (tindex < 24){
		e_old[1000 + tindex] = 0.0f;
		e_new[1000 + tindex] = 0.0f;
	}
	__syncthreads();

	cellx = 2*blockIdx.x;
	celly = 2*blockIdx.y;
	cellz = 2*blockIdx.z;
	
	if (tx == 0 && ty == 0 && tz == 0){
		calc_presum(localn_s, presum, 27);
		totalAtoms = presum[26] + localn_s[26];
	}
	__syncthreads();

	if (tindex < 27){
		for (int i = 0; i < localn_s[tindex]; i++){
			for (int dim = 0; dim < 3; dim++){
				write_index = presum[tindex] * 3 + i + dim*(totalAtoms);
				diskRead_index = nlist_s[tindex] * nmax * 3 + i + dim*nmax;
				ldisk[write_index] = disk[diskRead_index];
			}
		}
	}
	__syncthreads();
	curandState_t localRandomState;
	int id = tindex*blockIdx.x + blockIdx.y + blockIdx.z;
	curand_init(tindex, id, 0, &localRandomState);
	
	if (tindex == 0){ random_shuffle(ldisk, localn_s[0], totalAtoms, &localRandomState); }
	if (tindex == 0 && cellx + celly + cellz == 0){ print_ldisk(ldisk, totalAtoms * 3, totalAtoms); }
	int atomInCell_id = 0;
	for (int move = 0; move < n_M; move++){
		old_pos[0] = ldisk[atomInCell_id];
		old_pos[1] = ldisk[atomInCell_id + totalAtoms];
		old_pos[2] = ldisk[atomInCell_id + 2*totalAtoms];
		if (tindex == 0){
			make_move(new_pos, old_pos, &localRandomState);
			rejected = out_of_bound(new_pos, cellx, celly, cellz);
		}
		__syncthreads();
		if (!rejected){
			//calculate energy
				//apply PBC
			if (tindex < totalAtoms){
				r_old = 0;
				r_new = 0;
				for (dim = 0; dim < 3; dim++){
					del_old = old_pos[dim] - ldisk[tindex + dim*totalAtoms];
					del_new = new_pos[dim] - ldisk[tindex + dim*totalAtoms];
					if (fabs(del_old) > 2 * w){
						del_old += (2 * (del_old < 0) - 1)*L;
					}
					r_old += del_old * del_old;
					if (fabs(del_new) > 2 * w){
						del_new += (2 * (del_new < 0) - 1)*L;
					}
					r_new += del_new * del_new;
				}
				r_old = sqrtf(r_old);
				r_new = sqrtf(r_new);
				if (r_new != 0){
					power6 = __powf(r_new, -6.0);
					e_new[tindex] = 4.0f * (power6 * power6 - power6);
				}
				if (r_old != 0){
					power6 = __powf(r_old, -6.0);
					e_old[tindex] = 4.0f * (power6 * power6 - power6);
				}
				// reduction
				//Loading input floats to shared memory
				//Take care of the boundary conditions 
				if (tindex < 512){
					e_old[tindex] += e_old[tindex + 512];
					e_new[tindex] += e_new[tindex + 512];
				} __syncthreads();
				if (tindex < 256){
					e_old[tindex] += e_old[tindex + 256];
					e_new[tindex] += e_new[tindex + 256];
				} __syncthreads();
				if (tindex < 128){
					e_old[tindex] += e_old[tindex + 128];
					e_new[tindex] += e_new[tindex + 128];
				} __syncthreads();
				if (tindex < 64){
					e_old[tindex] += e_old[tindex + 64];
					e_new[tindex] += e_new[tindex + 64];
				} __syncthreads();
				if (tindex < 32){
					e_old[tindex] += e_old[tindex + 32];
					e_new[tindex] += e_new[tindex + 32];
					e_old[tindex] += e_old[tindex + 16];
					e_new[tindex] += e_new[tindex + 16];
					e_old[tindex] += e_old[tindex + 8];
					e_new[tindex] += e_new[tindex + 8];
					e_old[tindex] += e_old[tindex + 4];
					e_new[tindex] += e_new[tindex + 4];
					e_old[tindex] += e_old[tindex + 2];
					e_new[tindex] += e_new[tindex + 2];
					e_old[tindex] += e_old[tindex + 1];
					e_new[tindex] += e_new[tindex + 1];
				}
			}
				if (tindex == 0){
					if (e_new[0] - e_old[0] < 0){
						accepted = true;
					}
					else if (__expf(-beta*(e_new[0] - e_old[0])) < curand_uniform(&localRandomState)){
						accepted = true;
					}
					else{
						accepted = false;
					}
					if (accepted){
						ldisk[atomInCell_id] = new_pos[0];
						ldisk[atomInCell_id + totalAtoms] = new_pos[1];
						ldisk[atomInCell_id + 2 * totalAtoms] = new_pos[2];
					}
				}
				__syncthreads();
			}
		atomInCell_id++;
		if (atomInCell_id >= localn_s[0]){ atomInCell_id = 0; }
	}
	if (tindex == 0 && cellx + celly + cellz == 0){ printf("\n"); print_ldisk(ldisk, totalAtoms * 3, totalAtoms); }
}

void print_nl(short int* nl, int len, int len_line_break){
	for (int i = 0; i < len; i++){
		if (i%len_line_break == 0){
			printf("\n");
		}
		printf("%i\t", nl[i]);
	}
}


int main(){
	short int * d_nl;
	short int * nl;
	float * d_disk;
	short int * d_n;
	float * d_r;

	cudaError_t state;
	
	int Ncbrt = int(ceil(cbrt((float)N_ATOMS)));
	
	int nlsize = sizeof(short int) * 27 * CPS3;
	int nsize = sizeof(short int) * CPS3;
	int disksize = sizeof(float) * 3 * nmax * CPS3;
	int rsize = sizeof(float) * 3 * N_ATOMS;

	nl = (short int *)malloc(nlsize);

	cudaMalloc((void **)&d_nl, nlsize);
	cudaMalloc((void **)&d_disk, disksize);
	cudaMalloc((void **)&d_n, nsize);
	cudaMalloc((void **)&d_r, rsize);

	dim3 bs_make_nl(cellsPerSide,cellsPerSide,cellsPerSide);
	dim3 bs_init_r(Ncbrt, Ncbrt, Ncbrt);
	dim3 bs_subSweep(10, 10, 10);
	
	make_nl << <1, bs_make_nl >> >(d_nl);
	init_r << <1, bs_init_r >> >(d_r, Ncbrt);
	state = cudaDeviceSynchronize();
	if (state != cudaSuccess){
		printf("Check init_r : %s", cudaGetErrorString(state));
	}

	assign << <int(ceil(float(CPS3) / BLOCK_SIZE)), BLOCK_SIZE >> >(d_r, d_disk, d_n);
	state = cudaDeviceSynchronize();
	if (state != cudaSuccess){
		printf("Assign kernel failed : %s", cudaGetErrorString(state));
	}

	subSweep << <1, bs_subSweep >> >(d_disk, d_nl, d_n);
	state = cudaDeviceSynchronize();
	if (state != cudaSuccess){
		printf("Controversial kernel failed! : %s", cudaGetErrorString(state));
	}

	//cudaMemcpy(disk, d_nl, nlsize, cudaMemcpyDeviceToHost);

	//print_nl(nl, 27 * CPS3, 27);

	return 0;
}