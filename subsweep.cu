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
    	// same seed for all dimensions?
        proposed_move[dim] = D_sh[dim * nmax + i] + curand_normal(random_state) * sigma;
    }
}
// return type boolean not short?
__device__ short out_of_bound(float * proposed_move, int cell_x, int cell_y, int cell_z){
    //my bad : the formula should be -- cell_x*w - L/2 + w/2 
    float x = (cell_x * w - L / 2) / 2;
    float y = (cell_y * w - L / 2) / 2;
    float z = (cell_z * w - L / 2) / 2;
    if (((proposed_move[0] - x) > L / 2 )|| ((proposed_move[0] - x) < - L / 2)){
        return false;
    }
    if (((proposed_move[0] - y) > L / 2 )|| ((proposed_move[0] - y) < - L / 2)){
        return false;
    }
    if (((proposed_move[0] - z) > L / 2 )|| ((proposed_move[0] - z) < - L / 2)){
        return false;
    }
    return true;
}
float calculate_energy_in_cell(float * D_sh, int i, float * position){
    //TODO: implement this
}


float calculate_old_energy(int cell_x, int cell_y, int cell_z, float * D_sh, float * disk, int i, int atom_counts, short int * n){
    //TODO: implement this
}

float calculate_new_energy(float * proposed_move, int cell_x, int cell_y, int cell_z, float * D_sh, float * disk, int i, int atom_counts, short int * n){
    //TODO: implement this
}


// return type boolean, right? not short
__device__ short accept_move(float * proposed_move, int cell_x, int cell_y, int cell_z, float * disk, float * D_sh, int i, curandState_t * randomState, int atom_counts, short int * n)
{
    if (out_of_bound(proposed_move)){
        return false;
    }
    float old_energy = calculate_old_energy(cell_x, cell_y, cell_z, D_sh, disk, i, atom_counts, n);
    float new_energy = calculate_new_energy(proposed_move, cell_x, cell_y, cell_z, D_sh, disk, i, atom_counts, n);
    if (new_energy < old_energy){
        return true;
    }
    if (curand_uniform(random_state) < __expf(-beta * (new_energy - old_energy)){
        return true;
    } else {
        return false;
    }
}

__device__ void cpy_proposed_to_D_sh(float * D_sh, float * proposed_move, int i) {
    for (int j = 0; i < 3; j++) {
        D_sh[j * nmax + i]  = proposed_move[j];
    }
}

__global__ void subsweep_kernel(float * disk, short int *n, short int * offset){
    int cell_x = 2*(blockIdx.x * blockDim.x + threadIdx.x) + off[0];
    int cell_y = 2*(blockIdx.y * blockDim.y + threadIdx.y) + off[1];
    int cell_z = 2*(blockIdx.z * blockDim.z + threadIdx.z) + off[2];

    int cell_index = get_cell_index(cell_x, cell_y, cell_z);
    int atom_counts = n[cell_index]
    if (atom_counts == 0){
        return;
    }

    curandState_t localRandomState;
    // define x and y here
    int id = x + y * blockDim.x * gridDim.x + z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    curand_init(1234, id, 0, &localRandomState);

	__shared__ float D_sh[nmax * 3];
    cpy_to_Dsh(D_sh, disk, cell_index, atom_counts);
    random_shuffle(D_sh, atom_counts, localRandomState);

    int i = 0;
    float proposed_move[3];
    for (int s = 0; s < n_M; n++){
        propose_move(proposed_move, D_sh, i, localRandomState);
        if accept_move(proposed_move, cell_x, cell_y, cell_z, disk, D_sh, i, atom_counts){
            cpy_proposed_to_D_sh(D_sh, proposed_move, i);
        }
        i += 1;
        if (i >= atom_counts){
            i = 0;
        }
    }
}
