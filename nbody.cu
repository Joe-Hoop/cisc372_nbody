#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include <cuda_runtime.h>
#include "compute.h"

// represents the objects in the system.  Global variables
vector3 *hVel, *d_hVel;
vector3 *hPos, *d_hPos;
double *mass, *d_mass;
vector3 *values, *d_values;
vector3 **accels, **d_accels;
// initHostMemory: Create storage for numObjects entities in our system
// Parameters: numObjects: number of objects to allocate
// Returns: None
// Side Effects: Allocates memory in the hVel, hPos, and mass global variables
void initHostMemory(int numObjects)
{
	hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
	hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
	mass = (double *)malloc(sizeof(double) * numObjects);
	values = (vector3 *)malloc(sizeof(vector3) * numObjects * numObjects);
	accels = (vector3 **)malloc(sizeof(vector3 *) * numObjects);
}
void initDeviceMemory(int numObjects)
{
	cudaMalloc(&d_hVel, sizeof(vector3) * numObjects);
	cudaMalloc(&d_hPos, sizeof(vector3) * numObjects);
	cudaMalloc(&d_values, sizeof(vector3) * numObjects * numObjects);
	cudaMalloc(&d_accels, sizeof(vector3 *) * numObjects);
	cudaMalloc(&d_mass, sizeof(double) * numObjects);
}

// freeHostMemory: Free storage allocated by a previous call to initHostMemory
// Parameters: None
// Returns: None
// Side Effects: Frees the memory allocated to global variables hVel, hPos, and mass.
void freeHostMemory()
{
	free(hVel);
	free(hPos);
	free(mass);
	free(accels);
	free(values);
}

void freeDeviceMemory()
{
	cudaFree(d_hVel);
	cudaFree(d_hPos);
	cudaFree(d_mass);
	cudaFree(d_accels);
	cudaFree(d_values);
}

// planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
// Parameters: None
// Returns: None
// Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
void planetFill()
{
	int i, j;
	double data[][7] = {SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE};
	for (i = 0; i <= NUMPLANETS; i++)
	{
		for (j = 0; j < 3; j++)
		{
			hPos[i][j] = data[i][j];
			hVel[i][j] = data[i][j + 3];
		}
		mass[i] = data[i][6];
	}
}

// randomFill: FIll the rest of the objects in the system randomly starting at some entry in the list
// Parameters: 	start: The index of the first open entry in our system (after planetFill).
//				count: The number of random objects to put into our system
// Returns: None
// Side Effects: Fills count entries in our system starting at index start (0 based)
void randomFill(int start, int count)
{
	int i, j, c = start;
	for (i = start; i < start + count; i++)
	{
		for (j = 0; j < 3; j++)
		{
			hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
			hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
			mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
		}
	}
}

// printSystem: Prints out the entire system to the supplied file
// Parameters: 	handle: A handle to an open file with write access to prnt the data to
// Returns: 		none
// Side Effects: Modifies the file handle by writing to it.
void printSystem(FILE *handle)
{
	int i, j;
	for (i = 0; i < NUMENTITIES; i++)
	{
		fprintf(handle, "pos=(");
		for (j = 0; j < 3; j++)
		{
			fprintf(handle, "%lf,", hPos[i][j]);
		}
		printf("),v=(");
		for (j = 0; j < 3; j++)
		{
			fprintf(handle, "%lf,", hVel[i][j]);
		}
		fprintf(handle, "),m=%lf\n", mass[i]);
	}
}

int main(int argc, char **argv)
{
#define BLOCK_SIZE 16
	clock_t t0 = clock();
	int t_now;
	// srand(time(NULL));
	srand(1234);
	initHostMemory(NUMENTITIES);
	initDeviceMemory(NUMENTITIES);
	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);
// now we have a system.
#ifdef DEBUG
	printSystem(stdout);
#endif
	printf("In the right block\n");
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((NUMENTITIES + BLOCK_SIZE - 1) / dimBlock.x, (NUMENTITIES + BLOCK_SIZE - 1) / dimBlock.y);

	for (t_now = 0; t_now < DURATION; t_now += INTERVAL)
	{
		cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_values, values, sizeof(vector3) * NUMENTITIES * NUMENTITIES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_accels, accels, sizeof(vector3 *) * NUMENTITIES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

		compute<<<dimGrid, dimBlock>>>(d_values, d_accels, d_hPos, d_hVel, d_mass);
		cudaDeviceSynchronize();
		cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
		cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
		cudaMemcpy(values, d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES, cudaMemcpyDeviceToHost);
		cudaMemcpy(accels, d_accels, sizeof(vector3 *) * NUMENTITIES, cudaMemcpyDeviceToHost);
		cudaMemcpy(mass, d_mass, sizeof(double) * NUMENTITIES, cudaMemcpyDeviceToHost);
	}

	clock_t t1 = clock() - t0;
#ifdef DEBUG
	printSystem(stdout);
#endif
	printf("This took a total time of %f seconds\n", (double)t1 / CLOCKS_PER_SEC);

	freeHostMemory();
	freeDeviceMemory();
}
