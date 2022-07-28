/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common.h>

namespace cuda
{

void pi_init()
{
    /*
      TODO any initialization code you need goes here, e.g. random
      number seeding, cudaMalloc allocations, etc.  Random number
      _generation_ should still go in pi().
    */

    //Set seed, (seconds since 00:00 hours, Jan 1, 1970 UTC)
    srand(time(0));

    //Set count to 0
    int h_count = 0;

    //Set block_size to maximum value of 1024 so we can fit all 30e6 samples in one kernel call
    int block_size = 1024;
    int grid_size = ceil(float(samples)/float(1024)); //roundup(30e6/1024)

    //We want to have x and y values for each sample
    //allocate device memory for x, y, and count
    float *d_x, *d_y, *d_count;
    cudaMalloc(&d_x, samples*sizeof(float));
    cudaMalloc(&d_y, samples*sizeof(float));
    cudaMalloc(&d_count, sizeof(int));

    //copy 0 to d_count
    cudaMemcpy(d_count, h_count, sizeof(int),cudaMemcpyHostToDevice);

}

//Idea 1: do all calculations in the same cuda kernel
__global__ void calculate_pi(int samples, int* count, float* x, float* y){
  //loop through samples (30e6)
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < samples; idx += gridDim.x * blockDim.x) 
	{
    x[idx] = float(rand()) / RAND_MAX;
    y[idx] = float(rand()) / RAND_MAX;
    if (x[ix]*x[idx] + y[idx]*y[idx] < 1) { //length of vector (x,y)
      //We want to avoid anything like a race condition, so we can use atomicAdd
      //this allows us to increment the count correctly.
      //Without atomicAdd, it might be possible for count to be much lower than it actually is
      //because the value at count might be overwritten at the same time by more than one thread
      atomicAdd(&count,1);
    }
  }
}

//Idea 2: calculate x and y in a separate kernel
//Idea 3: just use thrust functions to calculate the random values and get pi

double pi()
{
    /*
      TODO Put your code here.  You can use anything in the CUDA
      Toolkit, including libraries, Thrust, or your own device
      kernels, but do not use ArrayFire functions here.  If you have
      initialization code, see pi_init().
    */


  //Call cuda kernel
  calculate_pi <<<block_size, grid_size>>> (samples, d_count, d_x, d_y);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

  //Copy count back MIGHT NOT BE NECESSARY, MAYBE KEEP COUNT ON CPU
  cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

  //return pi
  return 4.0 * h_count / samples;
}

void pi_reset()
{
    /*
      TODO This function should contain the clean up. You should add
      memory deallocation etc here.
    */

  //Free memory on device
  cudaFree(d_count);
  cudaFree(d_x);
  cudaFree(d_y);
}

}
