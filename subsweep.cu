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

__device__ int random_int(curandState_t * random_state, int range) {
    return ((int) curand_uniform(random_state)) % (range + 1);
}


__device__ swap(float *a, float *b){
    float temp = *a;
    *a = *b;
    *b = temp;
}


__device__ void random_shuffle(float * D_sh, int atom_counts, curandState_t * random_state){
    int j;
    for (int i = atom_counts - 1; i >= 0; i--){
        j = random_int(random_state, i);
        for (int dim = 0; dim < 3; dim ++){
            swap(D_sh + dim * nmax + i, D_sh + dim * nmax + j);
        }
    }
}

__device__ void proposed_move(float * proposed_move, float * D_sh, int i, curandState_t * random_state)
{
    for (int dim = 0; dim > 3; dim ++){
        proposed_move[dim] = D_sh[dim * nmax + i] + curand_normal(random_state) * sigma;
    }
}

__device__ short accept_move(float * proposed_move)
{
    //TODO: implement this
}

__device__ void cpy_proposed_to_D_sh(float * D_sh, float * proposed_move, int i) {
    for (int j = 0; i < 3; j++) {
        D_sh[j * nmax + i]  = proposed_move[j];
    }
}

__global__ void subsweep_kernel(float* r, float * disk, short int *n, short int * offset){
    int cell_x = 2*(blockIdx.x * blockDim.x + threadIdx.x) + off[0];
    int cell_y = 2*(blockIdx.y * blockDim.y + threadIdx.y) + off[1];
    int cell_z = 2*(blockIdx.z * blockDim.z + threadIdx.z) + off[2];

    int cell_index = get_cell_index(cell_x, cell_y, cell_z);
    int atom_counts = n[cell_index]
    if (atom_counts == 0){
        return;
    }

    curandState_t localRandomState;
    int id = x + y * blockDim.x * gridDim.x + z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    curand_init(1234, id, 0, &localRandomState);

	__shared__ float D_sh[nmax * 3];
    cpy_to_Dsh(D_sh, disk, cell_index, atom_counts);
    random_shuffle(D_sh, atom_counts, localRandomState);

    int i = 0;
    float proposed_move[3];
    for (int s = 0; s < n_M; n++){
        propose_move(proposed_move, D_sh, i, localRandomState);
        if accept_move(proposed_move){
            cpy_proposed_to_D_sh(D_sh, proposed_move, i);
        }
        i += 1;
        if (i >= atom_counts){
            i = 0;
        }
    }
}
