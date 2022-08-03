/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include "cuda_runtime.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define PI 3.1415926535897931f

//Define some error checks for cuda kernels
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//Define some typedefs so I don't have to write long long all the time
//typedef long long int i64;
//typedef unsigned long long int u64;
//Currently runs into issues with d_atomicAdd

// generate millions of random samples
static int samples = 30e6;

// Number of elements to sum for pi_v2
static int N = 16;

#ifdef AF_OPENCL
namespace opencl
#else
namespace cuda
#endif
{

	//Variables
	static unsigned int seed; //seed for curand, can be the same for each thread
	static int block_size; //How many threads per block
	static int grid_size; //How many blocks per grid
	static int block_size2; //How many threads per block
	static int grid_size2; //How many blocks per grid
	static curandState* d_states; //pointer to the states

	//variables for v2
	static double C;

	//Helper Functions
	__device__ double d_atomicAdd(double* address, double val);
	__device__ long long int fact(int n);

	//Cuda Kernels
	__global__ void random_init(unsigned int seed, int samples, curandState* states);
	__global__ void est_pi(int* count, unsigned int seed, int samples, curandState* states);

	//Functions
	void pi_init_v1();
	double pi_v1();
	void pi_reset_v1();

	void pi_init_v2();
	double pi_v2();
	void pi_reset_v2();

}

#ifdef AF_OPENCL
namespace detail = opencl;
#else
namespace detail = cuda;
#endif
