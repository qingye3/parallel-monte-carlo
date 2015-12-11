__device__ int get_cell_index(int cell_x, int cell_y, int cell_z){
    return cell_x + cell_y * cellsPerSide + cell_z * CPS2;;
}

__device__ void cpy_to_Dsh(float * D_sh, float * disk, int cell_index, int atom_counts)
{
    int disk_index = cell_index * 3 * nmax;
    for (int i = 0; i < atom_counts; i++){
        D_sh[i] = disk[disk_index + i];
        D_sh[i + nmax] = disk[disk_index + nmax + i]
        D_sh[i + 2*nmax] = disk[disk_index + nmax * 2 + i]
    }
}


__device__ random_shuffle(float * D_sh, int atom_counts){
    //TODO: shuffle this
}

__global__ void subsweep_kernel(float* r, float * disk, short int *n, short int * offset){
    int cell_x = 2*(blockIdx.x * blockDim.x + threadIdx.x) + off[0];
    int cell_y = 2*(blockIdx.y * blockDim.y + threadIdx.y) + off[1];
    int cell_z = 2*(blockIdx.z * blockDim.z + threadIdx.z) + off[2];

    int cell_index = get_cell_index(cell_x, cell_y, cell_z);
    int atom_counts = n[cell_index]

	__shared__ short float D_sh[CPS3];
    cpy_to_Dsh(D_sh, disk, cell_index, atom_counts);
    random_shuffle(D_sh, atom_counts);

    int i = 0;
    for (int s = 0; s < n_M; n++){
    }
}
