/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include "cuda_runtime.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define PI 3.1415926535897931f

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// generate millions of random samples
static int samples = 30e6;
//static int samples = 10000;

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
	static curandState* d_states; //pointer to the states
	static thrust::device_vector<int> count;

	//Functions
	void pi_init();
	double pi_v1();
	void pi_reset();

}

#ifdef AF_OPENCL
namespace detail = opencl;
#else
namespace detail = cuda;
#endif
