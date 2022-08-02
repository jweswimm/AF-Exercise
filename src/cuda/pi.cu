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
//Initialize necessary variables, create seed, and allocate space on the device
void pi_init()
{
    //Create seed--this can be the same for each thread, see https://ianfinlayson.net/class/cpsc425/notes/cuda-random
    seed = time(0);

    //Initialize block size and grid size for the kernel call
    //For now just do 1D
    block_size = 1024;
    grid_size = ceil(double(samples) / double(block_size)); //round up (block_size*grid_size = samples)

    //Allocated space on the device for the number of samples we need
    cudaMalloc((void**)&d_states, block_size * grid_size * sizeof(curandState));

    //initialize count to 0
    count.push_back(0); 

}


__global__ void est_pi(int* count, unsigned int seed, int samples, curandState* states) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < samples; idx += gridDim.x * blockDim.x) {

        curand_init(seed, idx, 0, &states[idx]);

        double x = curand_uniform_double(&states[idx]);
        double y = curand_uniform_double(&states[idx]);

        if (x * x + y * y < 1) { //if within unit circle

            //To avoid race conditions where multiple threads try to
            //write to the same location at the same time, we use
            //the atomicAdd operation.
            atomicAdd((count), 1);

        }

    }
    
}



//Monte Carlo estimation of pi
double pi_v1()
{
    /*
    Utility & Explanation: We use a Monte Carlo method to estimate pi.
    In 2D, we stack a unit circle on a unit square. The area of the
    unit circle is \pi r^2 and the area of the unit square is 4 r^2.
    If we randomly generate x and y coordinates for each point, 
    the ratio of points inside the unit circle to points outside
    the unit circle and inside the unit square will be an excellent
    approximation of pi.
    I.e. (inside circle)/(outside circle) = (\pi r^2) / (4 r^2) = \pi / 4
    <=> 4*(inside circle)/(outside circle) = \pi
    Admittedly, this random number generation only generates positive numbers,
    but because the unit circle and unit square are entirely symmetric across 
    axes and we are calculating distances from the origin,
    it doesn't change the estimation.

    This function returns the estimation of pi.
    */


    
    //Call kernel
    est_pi << <grid_size, block_size >> > (thrust::raw_pointer_cast(count.data()), seed, samples, d_states);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //std::cout << 4*double(count[0])/double(samples) << std::endl;


    return 4*double(count[0])/double(samples);
}

void pi_reset()
{
    /*
      TODO This function should contain the clean up. You should add
      memory deallocation etc here.
    */

    //Free d_states, as it was on the device
    cudaFree(d_states);

    //Don't need to free thrust device vector count
}

}
