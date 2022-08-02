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

#ifdef AF_OPENCL
namespace opencl
#else
namespace cuda
#endif
{

	//Variables
	static unsigned int seed; //seed for curand, can be the same for each thread
	static unsigned int count; //number of values distance < 1 away from center
	static unsigned int block_size; //How many threads per block
	static unsigned int grid_size; //How many blocks per grid

	//CUDA kernels
	__global__ void state_init(unsigned int sseed, unsigned int ssamples, curandState_t* sstates);

	//Functions
	void pi_init();
	double pi();
	void pi_reset();

}

#ifdef AF_OPENCL
namespace detail = opencl;
#else
namespace detail = cuda;
#endif
