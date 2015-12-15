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
#define beta 1.0
#define cellsPerSide 4
#define w 2.5f
#define rc 2.5f
#define nmax 20
#define BLOCK_SIZE 1024
#define n_M 20
#define sigma 0.5f
#define dimCB 8
#define MCpasses 100

const int CPS2 = cellsPerSide*cellsPerSide;
const int CPS3 = CPS2*cellsPerSide;

#include "shiftCells.h"


// changed random shuffle to allow curand_uniform to give output greater than 1
// problems: 
// 1) energy going to extremely low values:
//		check if PCB are working
//		check energy calc and reduction
//		check shift cells -- UPDATE 1: commented shift cells and run simulation, same problem!
// 2) d not exactly random -
//
//


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
				if (nb_cellx < 0){ nb_cellx = cellsPerSide - 1; }
				if (nb_cellx >= cellsPerSide){ nb_cellx = 0; }

				nb_celly = celly + p[j];
				if (nb_celly < 0){ nb_celly = cellsPerSide - 1; }
				if (nb_celly >= cellsPerSide){ nb_celly = 0; }

				nb_cellz = cellz + p[i];
				if (nb_cellz < 0){ nb_cellz = cellsPerSide - 1; }
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
		presum[i] = presum[i - 1] + n[i - 1];
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
		j = (int(curand_uniform(random_state) * 200)) % (i + 1);
		for (int dim = 0; dim < 3; dim++){
			swap(ldisk + i + dim*totalAtoms, ldisk + j + dim*totalAtoms);
		}
	}
}

__device__ void make_move(float * proposed_move, float * old_pos, curandState_t * random_state)
{
	for (int dim = 0; dim < 3; dim++){
		proposed_move[dim] = old_pos[dim] + curand_normal(random_state) * sigma;
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

__global__ void subSweep(float*disk, short int* nl, short int* n, int offx, int offy, int offz, float* d_Eblocks){
	__shared__ short int presum[27];
	__shared__ float ldisk[27 * nmax * 3]; // format : all x of all cells followed by y then z and then garbage
	__shared__ short int nlist_s[27];
	__shared__ short int localn_s[27];
	__shared__ float e_old[1024];
	__shared__ float e_new[1024];
	float oldE;
	float newE;
	__shared__ int totalAtoms;
	__shared__ float new_pos[3];
	__shared__ bool rejected;
	__shared__ int checker_id;
	// new variables start
	checker_id = offx + offy * 2 + offz * 4;
	int block_id = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.x;
	// new variables end
	float old_pos[3], r_old, r_new, del_old, del_new, power6;
	int tx, ty, tz, tindex, write_index, diskRead_index, dim, i;
	short int cellx, celly, cellz, cellId;
	int accept_counter = 0;
	bool accepted;
	tx = threadIdx.x;
	ty = threadIdx.y;
	tz = threadIdx.z;
	tindex = tx + ty*blockDim.x + tz*blockDim.x*blockDim.x;

	cellx = 2 * blockIdx.x + offx;
	celly = 2 * blockIdx.y + offy;
	cellz = 2 * blockIdx.z + offz;
	cellId = get_cellindex(cellx, celly, cellz);

	if (tindex < 27){
		nlist_s[tindex] = nl[cellId * 27 + tindex];
		localn_s[tindex] = n[nlist_s[tindex]];
	}
	__syncthreads();

	// Initialize d_Eblocks
	if (tindex == 0) { d_Eblocks[block_id] = 0.0f; }

	// generate presum and total atoms here
	if (tindex == 0) {
		calc_presum(localn_s, presum, 27);
		totalAtoms = presum[26] + localn_s[26];
		//if (totalAtoms > 9) { printf("\nTotal number of atoms in system for cell %i & CB# %i is: %i\n", block_id, checker_id, totalAtoms); }
		/*if (cellx + celly + cellz == 0) {
		printf("\nPresum = ");
		for (i = 0; i < 27; i++){
		printf("%i\t", presum[i]);
		}
		}*/
	}
	__syncthreads();

	//generate local copy of disk
	if (tindex < 27){
		for (int i = 0; i < localn_s[tindex]; i++){
			for (int dim = 0; dim < 3; dim++){
				write_index = presum[tindex] + i + dim*totalAtoms;
				diskRead_index = nlist_s[tindex] * nmax * 3 + i + dim*nmax;
				ldisk[write_index] = disk[diskRead_index];
			}
		}
	}
	__syncthreads();
	//if (tindex == 0 && cellx + celly + cellz == 9) {
	//	printf("\n\nOriginal ldisk before shuffle:\n(number of atoms in current block = %i\n \
				//		number of atoms in current cell = %i\n", totalAtoms, localn_s[0]);
	//	print_ldisk(ldisk, totalAtoms * 3, totalAtoms);
	//}
	curandState_t localRandomState;
	int id = tindex*blockIdx.x + blockIdx.y + blockIdx.z;
	curand_init((unsigned long long)clock(), id, 0, &localRandomState);

	if (tindex == 0){ random_shuffle(ldisk, localn_s[0], totalAtoms, &localRandomState); }
	__syncthreads();
	//if (tindex == 0 && cellx + celly + cellz == 9) {
	//	printf("\n\nOriginal ldisk after shuffle:\n(number of atoms in current block = %i\n \
				//		   			   			number of atoms in current cell = %i\n", totalAtoms, localn_s[0]);
	//	print_ldisk(ldisk, totalAtoms * 3, totalAtoms);
	//}
	//if (tindex == 0 && cellx + celly + cellz == 9){ print_ldisk(ldisk, totalAtoms * 3, totalAtoms); }
	int atomInCell_id = 0;
	for (int move = 0; move < n_M; move++){
		e_old[tindex] = 0.0f;
		e_new[tindex] = 0.0f;

		if (tindex < 24){
			e_old[1000 + tindex] = 0.0f;
			e_new[1000 + tindex] = 0.0f;
		}
		__syncthreads();
		old_pos[0] = ldisk[atomInCell_id];
		old_pos[1] = ldisk[atomInCell_id + totalAtoms];
		old_pos[2] = ldisk[atomInCell_id + 2 * totalAtoms];
		if (tindex == 0){
			oldE = 0.0f;
			newE = 0.0f;
			make_move(new_pos, old_pos, &localRandomState);
			rejected = out_of_bound(new_pos, cellx, celly, cellz);
			//if (rejected && (cellx + celly + cellz == 0)){ printf("move %i rejected : out of bounds! %i \n\n",move, rejected); }
		}
		__syncthreads();
		if (!rejected){
			//calculate energy
			//apply PBC
			if (tindex < totalAtoms && tindex != atomInCell_id){
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
				if (r_new > 0.0f && r_new <= rc){
					power6 = __powf(r_new, -6.0);
					e_new[tindex] = 4.0f * (power6 * power6 - power6);
					/*	if (r_old < 1 && r_old!=0) {
					printf("r old <1 : %f for atom pair: %i and %i\t\n", r_old,tindex,atomInCell_id);
					}
					if (e_new[tindex] < -1) {
					printf("Energy too high: %f\n", e_new[tindex]);
					}*/
					//if (e_new[tindex] < -1.0) {
					//	printf("Energy too high (negative): %f for distance: %f between atoms %i and %i\n", e_new[tindex],r_new, tindex, atomInCell_id);
					//}
					/*if (r_new < 1 || e_new[tindex] > 0) {
					printf("r_new <1 or e > 0: \tr_new = %f; e_new = %f\n", r_new, e_new[tindex]);
					}*/
				}
				if (r_old > 0.0f && r_old <= rc){
					power6 = __powf(r_old, -6.0);
					e_old[tindex] = 4.0f * (power6 * power6 - power6);
					/*if (e_old[tindex] < -1) {
					printf("Energy too high (negative): %f\n", e_old[tindex]);
					}*/
					if (e_old[tindex] > 1000) {
						printf("e > 10: atom %i and %i \tr_old = %f; e_old = %f\n", atomInCell_id, tindex, r_old, e_old[tindex]);
					}
				}
				//if (e_new[tindex] - e_old){
				//	for (int z = 0; z < totalAtoms; z++)
				//		printf("e calculated for atom pair %i and %i is old:%f\t new:%f\n", atomInCell_id, z, e_old[z], e_new[z]);
				//}

			}
			__syncthreads();
			// reduction
			//Loading input floats to shared memory
			//Take care of the boundary conditions 
			/*if (tindex < 512){
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
			}__syncthreads();
			if (tindex == 0) {
			oldE = e_old[0];
			newE = e_new[0];
			}*/
			/*__syncthreads();*/


			// Alternate reduction - sequential:
			if (tindex == 0) {
				oldE = 0.0;
				newE = 0.0;
				for (i = 0; i < totalAtoms; i++) {
					oldE += e_old[i];
					newE += e_new[i];
				}
				/*if (cellId== 15){
				printf("total atoms are: %i\n old E:", totalAtoms);
				for (i = 0; i < totalAtoms; i++) {
				printf("%f\t", e_old[i]);
				}
				printf("\n oldE total %f\t", oldE);
				printf("\n new E:");
				for (i = 0; i < totalAtoms; i++) {
				printf("%f\t", e_new[i]);
				}
				printf("\n newE total %f\t", newE);
				printf("\n\n");
				}*/

			}

			float temp;
			if (tindex == 0){
				if (newE - oldE < 0){
					accepted = true;
					//if (cellx + celly + cellz == 0){ printf("\nmove %i accepted, new energy lower!\n", move); }
				}
				else if (__expf(-beta*(newE - oldE)) > curand_uniform(&localRandomState)){
					accepted = false;
					//if (cellx + celly + cellz == 0){
					//	printf("\nmove %i accepted, even though new energy higher! Old E = %f, New E = %f, prob = %f\n", move, oldE,newE, temp);
					//}
				}
				else{
					accepted = false;
					//if (cellx + celly + cellz == 0){
					//	printf("\nmove %i REJECTED, new energy higher, acceptance criterion NOT met!\n", move);
					//}
				}
				if (accepted){
					ldisk[atomInCell_id] = new_pos[0];
					ldisk[atomInCell_id + totalAtoms] = new_pos[1];
					ldisk[atomInCell_id + 2 * totalAtoms] = new_pos[2];
					accept_counter++;
					//if (cellx + celly + cellz == 0){
					//	printf("\nmove %i accepted, old E = %f, new E = %f\n", move, e_old[5], e_new[5]);
					//}
					temp = newE - oldE;
					if (temp > 0){
						if (block_id == 0) {
							printf("Energy change = %f\t for block %i and CB %i\nOldE = %f; NewE = %f\n\n", temp, block_id, checker_id, oldE, newE);
						}
						if (block_id == 1) {
							printf("Energy change = %f\t for block %i and CB %i\nOldE = %f; NewE = %f\n\n", temp, block_id, checker_id, oldE, newE);
						}
						if (block_id == 2) {
							printf("Energy change = %f\t for block %i and CB %i\nOldE = %f; NewE = %f\n\n", temp, block_id, checker_id, oldE, newE);
						}
						if (block_id == 3) {
							printf("Energy change = %f\t for block %i and CB %i\nOldE = %f; NewE = %f\n\n", temp, block_id, checker_id, oldE, newE);
						}
						if (block_id == 4) {
							printf("Energy change = %f\t for block %i and CB %i\nOldE = %f; NewE = %f\n\n", temp, block_id, checker_id, oldE, newE);
						}
						if (block_id == 5) {
							printf("Energy change = %f\t for block %i and CB %i\nOldE = %f; NewE = %f\n\n", temp, block_id, checker_id, oldE, newE);
						}
						if (block_id == 6) {
							printf("Energy change = %f\t for block %i and CB %i\nOldE = %f; NewE = %f\n\n", temp, block_id, checker_id, oldE, newE);
						}
						if (block_id == 7) {
							printf("Energy change = %f\t for block %i and CB %i\nOldE = %f; NewE = %f\n\n", temp, block_id, checker_id, oldE, newE);
						}
					}
					d_Eblocks[block_id] += temp;
					//if (temp < -25.0f){ printf("Energy change too high:%f particle: %i, in cell: %i\n new_pos: %f\t%f\t%f\t old_pos:%f\t%f\t%f\t\n", 
					//	temp, atomInCell_id, cellId, new_pos[0], new_pos[1], new_pos[2], old_pos[0], old_pos[1], old_pos[2] );
					//}
				}
			}
			__syncthreads();
		} // end of if(!rejected)

		atomInCell_id++;
		if (atomInCell_id >= localn_s[0]){ atomInCell_id = 0; }
	} // end of for loop for moves

	//__syncthreads();
	//if (tindex == 0 && cellx + celly + cellz == 0){ 
	//	printf("\n\n"); 
	//	printf("Accepted %i moves\nNew ldisk is:\n", accept_counter); 
	//	print_ldisk(ldisk, totalAtoms * 3, totalAtoms); 
	//}
	//__syncthreads();
	if (tindex == 0){
		for (int i = 0; i < localn_s[tindex]; i++){
			for (int dim = 0; dim < 3; dim++){
				write_index = i + dim*(totalAtoms);
				diskRead_index = nlist_s[tindex] * nmax * 3 + i + dim*nmax;
				disk[diskRead_index] = ldisk[write_index];
			}
		}
	}
}

__global__ void sumEnergy(float* d_Eblocks, float* d_Etrace, int MCstep){
	float temp;
	for (int i = 0; i < CPS3 / 8; i++){
		temp += d_Eblocks[i];
	}
	d_Etrace[MCstep] = temp;
}

__global__ void shiftatoms(float*disk, short int*n, float* r, float d, int f){
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	float temp = 0.0f;
	__shared__ short int localn_s[CPS3];
	__shared__ short int presum[CPS3];
	if (tx < CPS3){ localn_s[tx] = n[tx]; }
	__syncthreads();
	if (tx == 0){ calc_presum(localn_s, presum, CPS3); }
	__syncthreads();
	if (tx < CPS3){
		for (int i = 0; i < localn_s[tx]; i++){
			for (int dim = 0; dim < 3; dim++){
				r[presum[tx] + i + dim*N_ATOMS] = disk[tx*nmax * 3 + i + dim * nmax];
			}
		}
	}
	if (tx < N_ATOMS){
		temp = r[tx + f*N_ATOMS] - d;
		if (fabs(temp) > L / 2.0f){ temp += (2 * (temp < 0) - 1) * L; }
		r[tx + f*N_ATOMS] = temp;
	}
	__syncthreads();
}

float calc_energy(float * r){
	float energy = 0.0f, del, dist, power6;
	for (int i = 0; i < N_ATOMS; i++){
		for (int j = i + 1; j < N_ATOMS; j++){
			dist = 0.0f;
			for (int dim = 0; dim < 3; dim++){
				del = fabs(r[i + dim*N_ATOMS] - r[j + dim*N_ATOMS]);
				if (del > L / 2){ del -= L; }
				dist += del*del;
			}
			dist = sqrt(dist);
			if (dist <= rc){
				power6 = powf(dist, -6.0f);
				energy += 4.0f* (power6*power6 - power6);
			}
		}
	}
	return energy;
}

void print_nl(short int* nl, int len, int len_line_break){
	for (int i = 0; i < len; i++){
		if (i%len_line_break == 0){
			printf("\n");
		}
		printf("%i\t", nl[i]);
	}
}

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

void itoa(int * r, int n){
	r[2] = n % 2;
	r[1] = (n / 2) % 2;
	r[0] = (n / 4) % 2;
}

void host_print_disk(float * disk, short int * n){
	for (int i = 0; i < CPS3; i++){
		for (int j = 0; j < n[i]; j++){
			printf("Position of atom %i in cell %i: %f\t%f\t%f\n", j, i, disk[nmax * 3 * i + j],
				disk[nmax * 3 * i + j + nmax], disk[nmax * 3 * i + j + 2 * nmax]);
		}
	}
}

void disk_to_r(float* r, float* disk, short int* n){
	int counter = 0;
	for (int i = 0; i < CPS3; i++){
		for (int j = 0; j < n[i]; j++){
			for (int dim = 0; dim < 3; dim++){
				r[N_ATOMS*dim + counter] = disk[i*nmax * 3 + j + dim*nmax];
			}
			counter++;
		}
	}
}

void create_dump(char *filename, float *data, int steps, int nparticles){
	printf("\n Creating %s.txt file", filename);
	FILE *fp;
	int i, j;
	filename = strcat(filename, ".txt");
	fp = fopen(filename, "w+");
	int stride;
	//fprintf(fp,"Student Id, Physics, Chemistry, Maths");
	/*ITEM: TIMESTEP
	0
	ITEM: NUMBER OF ATOMS
	130
	ITEM: BOX BOUNDS ff pp pp
	0 30
	0 30
	-0.5 0.5
	ITEM: ATOMS id type x y z ix iy iz*/
	for (i = 0; i<steps; i++){
		fprintf(fp, "ITEM: TIMESTEP \n%i\nITEM: NUMBER OF ATOMS\n%i\nITEM: BOX BOUNDS\n%f %f\n%f %f\n%f %f\nITEM: ATOMS id type x y z ix iy iz\n",
			i, N_ATOMS, -L / 2.0f, L / 2.0f, -L / 2.0f, L / 2.0f, -L / 2.0f, L / 2.0f);
		stride = i * 3 * N_ATOMS;
		for (j = 0; j<nparticles; j++)
			fprintf(fp, "%i %i %f %f %f 0 0 0\n", j + 1, j + 1, data[stride + j], data[stride + j + N_ATOMS], data[stride + j + 2 * N_ATOMS]);
	}
	fclose(fp);
	printf("\n %sfile created \n", filename);
}

int main(){
	short int * d_nl;
	short int * nl;
	float * d_disk;
	short int * d_n;
	short int *n;
	float * d_r;
	float *r;
	int f;
	float d;
	float *Rtrace;
	int RNG[MCpasses * 3];
	float * energytrace;
	float * d_energyblocks;
	float * energyblocks;
	float * disk;

	int offset[3];
	cudaError_t state;

	srand(time(NULL));
	for (int i = 0; i < MCpasses * 3; i++){
		RNG[i] = rand();
	}

	int Ncbrt = int(ceil(cbrt((float)N_ATOMS)));
	int cboard_index[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
	int i, MC_step;

	int nlsize = sizeof(short int) * 27 * CPS3;
	int nsize = sizeof(short int) * CPS3;
	int disksize = sizeof(float) * 3 * nmax * CPS3;
	int rsize = sizeof(float) * 3 * N_ATOMS;

	nl = (short int *)malloc(nlsize);
	n = (short int *)malloc(nsize);
	r = (float *)malloc(rsize);
	energyblocks = (float *)malloc(sizeof(float)*CPS3 / 8);
	energytrace = (float *)malloc(sizeof(float)*(MCpasses + 1));
	disk = (float *)malloc(disksize);
	Rtrace = (float *)malloc(rsize*MCpasses);

	cudaMalloc((void **)&d_nl, nlsize);
	cudaMalloc((void **)&d_disk, disksize);
	cudaMalloc((void **)&d_n, nsize);
	cudaMalloc((void **)&d_r, rsize);
	cudaMalloc((void **)&d_energyblocks, sizeof(float) * CPS3 / 8);

	dim3 bs_make_nl(cellsPerSide, cellsPerSide, cellsPerSide);
	dim3 bs_init_r(Ncbrt, Ncbrt, Ncbrt);
	dim3 bs_subSweep(10, 10, 10);
	dim3 gs_subSweep(cellsPerSide / 2, cellsPerSide / 2, cellsPerSide / 2);
	dim3 bs_shiftCells(cellsPerSide, cellsPerSide, cellsPerSide);

	make_nl << <1, bs_make_nl >> >(d_nl);
	init_r << <1, bs_init_r >> >(d_r, Ncbrt);
	state = cudaDeviceSynchronize();
	if (state != cudaSuccess){
		printf("Check init_r : %s", cudaGetErrorString(state));
	}

	cudaMemcpy(r, d_r, rsize, cudaMemcpyDeviceToHost);
	energytrace[0] = calc_energy(r);
	printf("energytrace[0] = %f\n", energytrace[0]);
	assign << <int(ceil(float(CPS3) / BLOCK_SIZE)), BLOCK_SIZE >> >(d_r, d_disk, d_n);
	state = cudaDeviceSynchronize();
	if (state != cudaSuccess){
		printf("Assign kernel failed : %s", cudaGetErrorString(state));
	}

	//cudaMemcpy(n, d_n, nsize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(disk, d_disk, disksize, cudaMemcpyDeviceToHost);
	//host_print_disk(disk, n);
	for (MC_step = 0; MC_step < MCpasses; MC_step++){
		energytrace[MC_step + 1] = 0.0f;
		FY_Shuffle(cboard_index, dimCB);
		for (i = 0; i < dimCB; i++){
			itoa(offset, i);
			subSweep << <gs_subSweep, bs_subSweep >> >(d_disk, d_nl, d_n, offset[2], offset[1], offset[0], d_energyblocks);
			state = cudaDeviceSynchronize();
			if (state != cudaSuccess){
				printf("Controversial kernel failed! : %s", cudaGetErrorString(state));
			}
			cudaMemcpy(energyblocks, d_energyblocks, sizeof(float)*CPS3 / 8, cudaMemcpyDeviceToHost);
			//cudaMemcpy(n, d_n, nsize, cudaMemcpyDeviceToHost);

			//printf("Energy block at MC_step = %i:\t",MC_step);
			//printf("\nEnergy change block for MC_step #%i and CB #%i:\n", MC_step+1, i);
			//for (int count = 0; count < CPS3 / 8; count++) {
			//	printf("%f\t", energyblocks[count]);
			//}
			//printf("\n");
			for (int k = 1; k < CPS3 / 8; k++){
				//if (energyblocks[k] < -25.0f){ printf("energy shot up for block %i of checkerboard %i at MC_step %i to - %f\n",k,i,MC_step, energyblocks[k]); }
				energyblocks[0] += energyblocks[k];
			}
			energytrace[MC_step + 1] += energyblocks[0];
		}

		energytrace[MC_step + 1] += energytrace[MC_step];
		f = RNG[MC_step] % 3;
		d = (float)RNG[MC_step + MCpasses] / RAND_MAX * w - w / 2.0f;
		printf("Shifting in direction %i by %fw\n", f, d / w);
		/*shiftCells << <1, bs_shiftCells >> >(d_disk, d_n, f, d);*/
		shiftatoms << <int(ceil((float)N_ATOMS / BLOCK_SIZE)), BLOCK_SIZE >> >(d_disk, d_n, d_r, d, f);
		cudaDeviceSynchronize();
		assign << <int(ceil(float(CPS3) / BLOCK_SIZE)), BLOCK_SIZE >> >(d_r, d_disk, d_n);
		state = cudaDeviceSynchronize();

		if (state != cudaSuccess){
			printf("Shiftcells failed : %s\n", cudaGetErrorString(state));
		}
		printf("Energy at MC step %i is %f\n\n", MC_step + 1, energytrace[MC_step + 1]);
		cudaMemcpy(disk, d_disk, disksize, cudaMemcpyDeviceToHost);
		cudaMemcpy(n, d_n, nsize, cudaMemcpyDeviceToHost);

		/*printf("Number of atoms in cells: \n");
		for (int num = 0; num < CPS3; num++){
		printf("%i: %i\t", num, n[num]);
		}
		printf("\n");
		*/

		disk_to_r(Rtrace + MC_step * 3 * N_ATOMS, disk, n);
	}
	char s[20] = "dumpR";
	create_dump(s, Rtrace, MCpasses, N_ATOMS);
	//cudaMemcpy(nl, d_nl, nlsize, cudaMemcpyDeviceToHost);

	//print_nl(nl, 27 * CPS3, 27);

	return 0;
}