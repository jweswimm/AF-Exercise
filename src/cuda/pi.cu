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

 //State Initialization Kernel
__global__ void state_init(unsigned int seed, unsigned int samples, curandState_t* states) {
 /*
 Utility: To initialize "samples" many states. These states
    will be used to generate random numbers in the pi() function
 Inputs: seed, number of samples, array of states(empty)
 Output: array of states(full)
 */
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < samples; idx += gridDim.x * blockDim.x) //thread loop
    {
        curand_init(seed, idx, 0, &states[idx]); //args:(seed for each core, sequence of numbers, offset, state value)
    }
}

void pi_init()
{
    /*
      TODO any initialization code you need goes here, e.g. random
      number seeding, cudaMalloc allocations, etc.  Random number
      _generation_ should still go in pi().
    */

    //Initialize the number of samples less than distance 1 unit away from origin
    count = 0;

    //Create seed--this can be the same for each thread, see https://ianfinlayson.net/class/cpsc425/notes/cuda-random
    seed = time(0);

    //We will want to create a random state for each thread, so use curandState_t* to store them
    //Fill states in pi(), for now just declare
    curandState_t* states;

    //Allocate memory on the device to have a random state for each sample
    cudaMalloc((void**)&states, samples * sizeof(curandState_t));

    //Block size
    block_size = 1024;
    grid_size = ceil(double(samples) / double(1024));
    //Initialize "samples" many states
    state_init <<<grid_size, block_size>>> (seed, samples, states);
     gpuErrchk(cudaPeekAtLastError());
     gpuErrchk(cudaDeviceSynchronize());
     





}


double pi()
{
    /*
      TODO Put your code here.  You can use anything in the CUDA
      Toolkit, including libraries, Thrust, or your own device
      kernels, but do not use ArrayFire functions here.  If you have
      initialization code, see pi_init().
    */


    return 0.0;
}

void pi_reset()
{
    /*
      TODO This function should contain the clean up. You should add
      memory deallocation etc here.
    */

}

}
