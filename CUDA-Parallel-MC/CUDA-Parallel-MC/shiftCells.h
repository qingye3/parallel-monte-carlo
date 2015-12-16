/*
//	Inputs:
//	f = 0, 1, or 2 => [x,y,z] => direction of cell shift
//	Converted to dir = [+/-1/0,+/-1/0,+/-1/0] => [+/-x,+/-y,+/-z]
//	e.g. +x => [+1,0,0]

//	-w/2<d<=w/2 => shift length
*/


__device__ void print_array(float * D_sh, int len){
	for (int i = 0; i < len; i++){
		printf("Array[%i] = %f\n", i, D_sh[i]);
	}
}

__device__ void init_array(float * D_sh, int len){
	for (int i = 0; i < len; i++){
		D_sh[i] = -20.0f;
	}
}

__global__ void shiftCells(float *disk, short int *n, int f, float d) {

	// Declarations/initializations
	int cellId, cellId_nb, nCurr, nNew, nNb, cid[3];
	int i, j, dim;
	float offset, D, s[3];

	__shared__ float shortDisk[CPS3*nmax];
	float D_sh[nmax * 3]; // D_sh stores global coordinates of all atoms in current cell after shift

	cid[0] = threadIdx.x + blockIdx.x*blockDim.x;
	cid[1] = threadIdx.y + blockIdx.y*blockDim.y;
	cid[2] = threadIdx.z + blockIdx.z*blockDim.z;

	// Conversion of f->dir	
	int dir[3] = { 0, 0, 0 };
	if (d <= 0){
		dir[f] = -1;
	}
	else {
		dir[f] = 1;
	}

	offset = cid[f] * w - L / 2.0f;

	cellId = cid[0] + cid[1] * cellsPerSide + cid[2] * CPS2;
	nCurr = n[cellId];

	// Storing local coordinate (only shift direction) in shortDisk
	for (i = 0; i<nCurr; i++) {
		shortDisk[cellId*nmax + i] = disk[cellId*nmax * 3 + f*nmax + i] - offset;
	}
	__syncthreads();

	nNew = 0;
	// Checking current cell's atoms
	for (i = 0; i<nCurr; i++) {
		D = shortDisk[cellId*nmax + i] - d;

		if (D>0 && D <= w) {
			for (dim = 0; dim<3; dim++) {
				if (dim == f)
					D_sh[dim*nmax + nNew] = D + offset; //storing in global co-ordinates
				else
					D_sh[dim*nmax + nNew] = disk[cellId*nmax * 3 + dim*nmax + i];
			}
			nNew++;
		}
	}
	// Neighboring cell in direction of f
	cid[f] += dir[f];

	// Implementing PBCs
	if (cid[f]<0) {
		cid[f] = cellsPerSide - 1;
	}
	else if (cid[f] >= cellsPerSide){
		cid[f] = 0;
	}

	for (dim = 0; dim<3; dim++) {
		s[dim] = w*dir[dim];
	}

	cellId_nb = cid[0] + cid[1] * cellsPerSide + cid[2] * cellsPerSide*cellsPerSide;
	nNb = n[cellId_nb];

	for (i = 0; i<nNb; i++) {
		D = shortDisk[cellId_nb*nmax + i] - d;

		if (!(D>0 && D <= w)){
			for (dim = 0; dim<3; dim++) {
				if (dim == f)
					D_sh[dim*nmax + nNew] = D + offset + s[dim];
				else
					D_sh[dim*nmax + nNew] = disk[cellId_nb*nmax * 3 + dim*nmax + i];
			}
			nNew++;
		}
	}

	__syncthreads();
	n[cellId] = nNew;

	for (i = 0; i<nmax * 3; i++) {
		disk[cellId*nmax * 3 + i] = D_sh[i];
	}

	__syncthreads();
}
