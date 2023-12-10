#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "compute.h"

// worked with Joe Hooper

__global__ void computeAccels(double *d_hPos, double *d_mass, vector3 *d_accels)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < NUMENTITIES && j < NUMENTITIES)
	{
		if (i == j)
		{
			FILL_VECTOR(d_accels[i * NUMENTITIES + j], 0, 0, 0);
		}
		else
		{
			vector3 distance;
			for (int k = 0; k < 3; k++)
			{
				distance[k] = d_hPos[i * 3 + k] - d_hPos[j * 3 + k];
			}
			double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
			double magnitude = sqrt(magnitude_sq);
			double accelmag = -1 * GRAV_CONSTANT * d_mass[j] / magnitude_sq;
			FILL_VECTOR(d_accels[i * NUMENTITIES + j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
		}
	}
}

// compute: Updates the positions and locations of the objects in the system based on gravity.
// Parameters: None
// Returns: None
// Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute()
{

	double *d_mass, *d_hPos;
	vector3 *d_accels;

	cudaMalloc((void **)&d_hPos, sizeof(double) * 3 * NUMENTITIES);
	cudaMalloc((void **)&d_mass, sizeof(double) * NUMENTITIES);
	cudaMalloc((void **)&d_accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);

	cudaMemcpy(d_hPos, hPos, sizeof(double) * 3 * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16);
	dim3 gridsize((NUMENTITIES + blockSize.x - 1) / blockSize.x, (NUMENTITIES + blockSize.y - 1) / blockSize.y);
	computeAccels<<<gridsize, blockSize>>>(d_hPos, d_mass, d_accels);
	cudaDeviceSynchronize();

	vector3 *h_accels = (vector3 *)malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	cudaMemcpy(h_accels, d_accels, NUMENTITIES * NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost);

	cudaFree(d_hPos);
	cudaFree(d_mass);
	cudaFree(d_accels);

	for (int i = 0; i < NUMENTITIES; i++)
	{
		vector3 sum = {0, 0, 0};
		for (int j = 0; j < NUMENTITIES; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				sum[k] += h_accels[i * NUMENTITIES + j][k];
			}
		}
		for (int k = 0; k < 3; k++)
		{
			hVel[i][k] += sum[k] * INTERVAL;
			hPos[i][k] += hVel[i][k] * INTERVAL;
		}
	}
	free(h_accels);
}
