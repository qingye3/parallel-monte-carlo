__device__ int get_cell_index(int cell_x, int cell_y, int cell_z){
    return cell_x + cell_y * cellsPerSide + cell_z * CPS2;;
}

__device__ void cpy_to_Dsh(float * D_sh, float * disk, int cell_index, int atom_counts)
{
    int disk_index = cell_index * 3 * nmax;
    for (int i = 0; i < atom_counts; i++){
        D_sh[i] = disk[disk_index + i];
        D_sh[i + nmax] = disk[disk_index + nmax + i];
        D_sh[i + 2*nmax] = disk[disk_index + nmax * 2 + i];
    }
}

__device__ void cpy_D_sh_to_Disk(float * D_sh, float * disk, int cell_index, int atom_counts){
    int disk_index = cell_index * 3 * nmax;
    for (int i = 0; i < atom_counts; i++){
        disk[disk_index + i] = D_sh[i]; 
        disk[disk_index + nmax + i] = D_sh[i + nmax]; 
        disk[disk_index + nmax * 2 + i] = D_sh[i + 2*nmax];
    }
}

__device__ int random_int(curandState_t * random_state, int range) {
    return ((int) curand_uniform(random_state)) % (range + 1);
}


__device__ void swap(float *a, float *b){
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

__device__ void make_move(float * proposed_move, float * D_sh, int i, curandState_t * random_state)
{
    for (int dim = 0; dim > 3; dim ++){
        proposed_move[dim] = D_sh[dim * nmax + i] + curand_normal(random_state) * sigma;
    }
}

__device__ bool out_of_bound(float * proposed_move, int cell_x, int cell_y, int cell_z){
    float x = cell_x * w - L / 2 + w / 2;
    float y = cell_y * w - L / 2 + w / 2;
    float z = cell_z * w - L / 2 + w / 2;
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

__device__ float calculate_pair_energy(float * the_atom, float * other_atom){
    float dist = 0, power6, dd;
    for (int dim = 0; dim < 3; dim ++){
        dd = the_atom[0] - other_atom[0];
        dist += dd * dd;
    }
    dist = sqrtf(dist);
    if (dist > w){
        return 0;
    }
    power6 = __powf(dist,-6);
    return 4.0f*(power6*power6 - power6);
}

__device__ float calculate_energy_in_cell(float * D_sh, float * the_atom, int i, int atom_counts){
    float e = 0;
    float other_atom[3];
    for (int j = 0; j < atom_counts; j++){
        if (j != i){
            for (int dim = 0; dim < 3; dim ++){
                other_atom[dim] = D_sh[dim * nmax + i];
            }
            e += calculate_pair_energy(the_atom, other_atom);
        }
    }
    return e;
}

__device__ void get_neighbors(int * neighbors,int cell_x,int cell_y,int cell_z){
    int helper[3] = {-1, 0, 1};
    for (int i = 0; i < 3; i ++){
        for (int j = 0; j < 3; j ++){
            for (int k = 0; k < 3; k ++){
                if (i == 0 && j == 0 && k == 0){
                    continue;
                }
                int x = cell_x + helper[i];
                int y = cell_y + helper[j];
                int z = cell_z + helper[k];
                x = (x + cellsPerSide) % cellsPerSide;
                y = (y + cellsPerSide) % cellsPerSide;
                z = (z + cellsPerSide) % cellsPerSide;
                neighbors[i*9 + j*3 + k] = get_cell_index(x, y, z);
            }
        }
    }
}

__device__ void apply_PBC(float * other_atom, float * the_atom)
{
    for (int dim = 0; dim < 3; dim++){
        if (other_atom[dim] - the_atom[dim] > 2*w)
        {
            other_atom[dim] -= L;
        }
        if (other_atom[dim] - the_atom[dim] < -2*w)
        {
            other_atom[dim] += L;
        }
    }
}

__device__ float calculate_energy_in_neighbors(float * disk, float * the_atom, short int * n, int cell_x, int cell_y, int cell_z){
    int neighbors[26];
    get_neighbors(neighbors, cell_x, cell_y, cell_z);

    float e = 0;
    float other_atom[3];
    int nb_cell_index;
    for (int j = 0; j < 26; j++){
        nb_cell_index = neighbors[j];
        for (int k = 0; k < n[nb_cell_index]; k++)
        {
            for (int dim = 0; dim < 3; dim ++){
                other_atom[dim] = disk[nb_cell_index*3*nmax + nmax*dim + k];
            }
            apply_PBC(other_atom, the_atom);
            e += calculate_pair_energy(the_atom, other_atom);

        }
    }
    return e;
}


__device__ float calculate_old_energy(int cell_x, int cell_y, int cell_z, float * D_sh, float * disk, int i, int atom_counts, short int * n){
    float atom_positions[3];
    for (int dim = 0; dim < 3; dim ++){
        atom_positions[dim] = D_sh[dim * nmax + i];
    }
    float e = calculate_energy_in_cell(D_sh, atom_positions, i, atom_counts);
    e += calculate_energy_in_neighbors(disk, atom_positions, n, cell_x, cell_y, cell_z);
    return e;
}

__device__ float calculate_new_energy(float * proposed_move, int cell_x, int cell_y, int cell_z, float * D_sh, float * disk, int i, int atom_counts, short int * n){
    float e = calculate_energy_in_cell(D_sh, proposed_move, i, atom_counts);
    e += calculate_energy_in_neighbors(disk, proposed_move, n, cell_x, cell_y, cell_z);
    return e;
}


__device__ bool accept_move(float * proposed_move, int cell_x, int cell_y, int cell_z, float * disk, float * D_sh, int i, curandState_t * randomState, int atom_counts, short int * n)
{
    if (out_of_bound(proposed_move, cell_x, cell_y, cell_z)){
        return false;
    }
    float old_energy = calculate_old_energy(cell_x, cell_y, cell_z, D_sh, disk, i, atom_counts, n);
    float new_energy = calculate_new_energy(proposed_move, cell_x, cell_y, cell_z, D_sh, disk, i, atom_counts, n);
    if (new_energy < old_energy){
        return true;
    }
    if (curand_uniform(randomState) < __expf(-beta * (new_energy - old_energy))){
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
__device__ void print_D_sh(float * D_sh){
	for (int i = 0; i < nmax*3; i++){
		printf("D_sh[%i] = %f\n", i, D_sh[i]);
	}
}


__global__ void subsweep_kernel(float * disk, short int *n, int * offset){
	
	int cell_x = 2 * (blockIdx.x * blockDim.x + threadIdx.x)+offset[0];
	int cell_y = 2 * (blockIdx.y * blockDim.y + threadIdx.y) +offset[1];
	int cell_z = 2 * (blockIdx.z * blockDim.z + threadIdx.z) +offset[2];
	
    int cell_index = get_cell_index(cell_x, cell_y, cell_z);
	if (cell_x < cellsPerSide && cell_y < cellsPerSide && cell_z < cellsPerSide){
		int atom_counts = n[cell_index];
		if (atom_counts == 0){
			return;
		}
		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){ printf("Atom counts = %i", atom_counts); }
		curandState_t localRandomState;
		// define x and y here
		int id = cell_x + cell_y * blockDim.x * gridDim.x + cell_z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
		curand_init(1234, id, 0, &localRandomState);
		if (threadIdx.x == 1 && threadIdx.y == 0 && threadIdx.z == 0){ printf("RNG 1 done! \n"); }
		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){ printf("RNG 0 done! \n"); }
		__shared__ float D_sh[nmax * 3];
		cpy_to_Dsh(D_sh, disk, cell_index, atom_counts);
		__syncthreads();
		random_shuffle(D_sh, atom_counts, &localRandomState);
		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){ printf("shuffle 0 done! \n"); }
		if (threadIdx.x == 1 && threadIdx.y == 0 && threadIdx.z == 0){ printf("shuffle 1 done! \n"); }
		if (threadIdx.x == 1 && threadIdx.y == 0 && threadIdx.z == 0){ print_D_sh(D_sh); }
		int i = 0;
		float proposed_move[3];
		for (int s = 0; s < n_M; n++){
			make_move(proposed_move, D_sh, i, &localRandomState);
			if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){ printf("make move 0 done! \n"); }
			if (threadIdx.x == 1 && threadIdx.y == 0 && threadIdx.z == 0){ printf("make move 1 done! \n"); }

			if (accept_move(proposed_move, cell_x, cell_y, cell_z, disk, D_sh, i, &localRandomState, atom_counts, n)){
				cpy_proposed_to_D_sh(D_sh, proposed_move, i);
			}
			i += 1;
			if (threadIdx.x == 0){ printf("pr make move 0 done! \n"); }
			if (threadIdx.x == 1){ printf("pr make move 1 done! \n"); }
			if (i >= atom_counts){
				i = 0;
			}
		}
		cpy_D_sh_to_Disk(D_sh, disk, cell_index, atom_counts);
	}
}
