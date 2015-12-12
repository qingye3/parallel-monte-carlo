/* 
//	Inputs:
//	f = 0, 1, 2, 3, 4, or 5 => [+x,-x,+y,-y,+z,-z] => direction of cell shift
//	Converted to dir = [+/-1/0,+/-1/0,+/-1/0] => [+/-x,+/-y,+/-z]

//	0<d<=w/2 => shift length
*/

/*	Stuff remaining:
//	1. Coordinate transformation												!!DONE!!
//	2. Comparing only the coordinate that has changed		!!DONE!!
//	3. f = 0,1,2; d -> (-w/2,w/2]												!!DONE!!
//	4. Periodic Boundary Conditions 										!!DONE!!
//	5. s[] across boundary
*/


__global__ void shiftCells(float *disk, short int *n, int f, float d, float *disk_dbl) {
	
	// Declarations/initializations
	int s[3],cellId,cellId_nb,nCurr,nNew,nNb,cid[3];
	int i,j;
	float offset,D;
	
	__shared__ float shortDisk[CPS3*nmax];
	__shared__ float D_sh[nmax*3]; // D_sh stores global coordinates of all atoms in current cell after shift
	
	cid[0] = threadIdx.x + blockIdx.x*blockDim.x;
	cid[1] = threadIdx.y + blockIdx.y*blockDim.y;
	cid[2] = threadIdx.z + blockIdx.z*blockDim.z;

	// Conversion of f->dir	
	int dir[3] = {0,0,0};
    if (d <= 0){
        dir[f] = -1;
    } else {
        dir[f] = 1;
    }
	
	switch(f) {
		case 0: offset = cid[0]*w-L/2;
						break;
		case 1: offset = cid[1]*w-L/2;
						break;
		case 2: offset = cid[2]*w-L/2;
						break;
	}
	
	cellId = cid[0] + cid[1]*cellsPerSide + cid[2]*cellsPerSide*cellsPerSide;
	nCurr = n[cellId];
	
	// Storing local coordinate (only shift direction) in shortDisk
	for(i=0; i<nCurr; i++) {
		shortDisk[cellId*nmax+i] = disk[cellId*nmax*3 + f*nmax + i]-offset;
	}
	__syncthreads();
/*	// Coordinate transformation to local coordinates
	for(i=0; i<nmax; i++) {
		
	}
*/
	
/*	// Initializing atom coordinates for the new cell (after shift)
	for(i=0; i<nmax; i++) {
		D_sh[i*3+0] = -10;
		D_sh[i*3+1] = -10;
		D_sh[i*3+2] = -10;
	}
	*/
	
	nNew = 0;
	
	// Checking current cell's atoms
	for(i=0; i<nCurr; i++) {
		D = shortDisk[cellId*nmax+i]-d;
		
		if(D>0 && D<=w) {
			for(j=0; j<3; j++) {
				if(j==f)
					D_sh[j*nmax+nNew] = D + offset;
				else
					D_sh[j*nmax+nNew] = disk[cellId*nmax*3 + j*nmax + i];			
			}
			nNew++;
		}
	}
	
	// Neighboring cell in direction of f
	cid[f] += dir[f];
	
	// Implementing PBCs
	if(cid[f]<0) {
		cid[f] = cellsPerSide-1;
    }
	else if(cid[f]>=cellsPerSide){
		cid[f] = 0;
    }
	
	for(i=0;i<3;i++) {
		s[i] = -w*dir[i];
	}
	
	cellId_nb = cid[0] + cid[1]*cellsPerSide + cid[2]*cellsPerSide*cellsPerSide;
	nNb = n[cellId_nb];
	
	for(i=0; i<nNb; i++) {
		D = shortDisk[cellId_nb*nmax+i]-d;
		
		if(!(D>0 && D<=w)){
			for(j=0; j<3; j++) {
				if(j==f)
					D_sh[j*nmax+nNew] = D + offset;
				else
					D_sh[j*nmax+nNew] = disk[cellId_nb*nmax*3 + j*nmax + i] + s[j];
			}
			nNew++;
		}
	}
	
	// Transform back to global coordinates and store in double-buffered
	for(i=0; i<nmax; i++) {
		for(j=0; j<3; j++)
			disk_dbl[cellId*nmax + j*nmax + i] = D_sh[j*nmax + i];
	}
	
    __syncthreads();
	n[cellId] = nNew;

    for(i=0; i<nmax*3; i++) {
			disk[cellId*nmax + i] = D_sh[i];
	}
    __syncthreads();
}
