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

    //Initialize random states for curand call later
    __global__ void random_init(unsigned int seed, int samples, curandState* states) {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < samples; idx += gridDim.x * blockDim.x) {

            curand_init(seed, idx, 0, &states[idx]);

        }

    }

    //Initialize necessary variables, create seed, and allocate space on the device
    void pi_init_v1()
    {

        //Create seed--this can be the same for each thread, see https://ianfinlayson.net/class/cpsc425/notes/cuda-random
        seed = time(0);

        //Initialize block size and grid size for the kernel call
        //For now just do 1D
        block_size = 1024;
        grid_size = ceil(double(samples) / double(block_size)); //round up (block_size*grid_size = samples)

        //Allocated space on the device for the number of samples we need
        cudaMalloc((void**)&d_states, samples * sizeof(curandState));

        //Call curand_init kernel
        random_init <<< grid_size, block_size >>> (seed, samples, d_states);

    }


    //kernel for Monte Carlo estimation of pi
    __global__ void est_pi(int* count, unsigned int seed, int samples, curandState* states) {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < samples; idx += gridDim.x * blockDim.x) {

            //Note: uniform distribution excludes 0 and includes 1
            //We may want a real distribution instead
            double x = curand_uniform_double(&states[idx]);
            double y = curand_uniform_double(&states[idx]);

            if (x * x + y * y < 1) { //if within unit circle

                //To avoid race conditions where multiple threads try to
                //write to the same location at the same time, we use
                //the atomicAdd operation.
                atomicAdd(count, 1);

            }

        }

    }



    //Monte Carlo estimation of pi function
    double pi_v1()
    {
        /*
        Utility & Explanation: We use a Monte Carlo method to estimate pi.
        In 2D, we overlay a unit circle on a unit square. The area of the
        unit circle is \pi r^2 and the area of the unit square is 4 r^2.
        If we randomly generate x and y coordinates for points,
        the fraction of points inside the unit circle to points outside
        the unit circle and inside the unit square will be an excellent
        approximation of pi, as long as all coordinates are equally likely.
        I.e. (inside circle)/(outside circle) = (\pi r^2) / (4 r^2) = \pi / 4
        <=> 4*(inside circle)/(outside circle) = \pi
        Admittedly, this random number generation only generates positive numbers,
        but because the unit circle and unit square are entirely symmetric across
        axes and we are calculating distances from the origin,
        it doesn't change the estimation.

        This function returns the estimation of pi.
        */

        //Initialize count to 0
        thrust::device_vector<int> count(1, 0);
        //It would be faster (negligibly so, I think) to not use thrust here
        //However, because we have to continuously reset the value of count,
        //as af::timeit() calls the function over and over again, I thought
        //it's better to just use thrust to do this instead of creating
        //a pointer that points to 0, then cudamemcpy'ing the host pointer
        // to the device pointer

        //Call kernel
        est_pi <<< grid_size, block_size >>> (thrust::raw_pointer_cast(count.data()), seed, samples, d_states);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        return 4 * double(count[0]) / double(samples);
    }

    void pi_reset_v1()
    {
        //Free d_states, as it was on the device
        cudaFree(d_states);
        //Don't need to free thrust device vector count
    }


    //---Pi V2---//
    //Cute example of some quick calculation to estimate pi
    //using Chudnovsky algorithm

    //Kernel for the chudnovsky algorithm to calculate pi
    __global__ void chudnovsky(int N, double* sum) {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < N; idx += gridDim.x * blockDim.x) {

            //Calculate constituent parts of the sum
            long long int M = (fact(6 * idx))/(fact(3*idx)*pow(fact(idx),3));
            long long int L = 54514013 * idx + 13591409;
            long long int X = pow(-262537412640768000, idx);

                //Calculate sum and avoid race conditions
                d_atomicAdd(sum, (M * L) / X);

        }

    }

    void pi_init_v2() {
        //Calculate C
        C = 426880 * sqrt(10005);

        //How many elements to sum
        N = 16;

        //Set kernel init variables
        block_size2 = 16;
        grid_size2 = ceil(float(N) / float(block_size));

    }

    double pi_v2() {
        /*
        This algorithm is based on Ramanujan's infinite series calculation of pi
        See https://en.wikipedia.org/wiki/List_of_formulae_involving_%CF%80#Infinite_series
        See https://en.wikipedia.org/wiki/Chudnovsky_algorithm
        See Chudnovsky, David; Chudnovsky, Gregory (1988)

        Bottlenecked by storage space (large computations for factorials)
        Might have to use hexadecimal for huge computations
        See this interesting blog http://www.karrels.org/pi/ where they talk about
        dealing with huge numbers
        See this interesting software http://www.numberworld.org/y-cruncher/
        */


        //Reset sum to 0
        thrust::device_vector<double> sum(1, 0.0);
        //It would be faster (negligibly so, I think) to not use thrust here
        //However, because we have to continuously reset the value of count,
        //as af::timeit() calls the function over and over again, I thought
        //it's better to just use thrust to do this instead of creating
        //a pointer that points to 0, then cudamemcpy'ing the host pointer
        // to the device pointer

        //Call kernel
        chudnovsky <<< grid_size2, block_size2 >>> (N, thrust::raw_pointer_cast(sum.data()));

        return C * (1/sum[0]);

    }

    void pi_reset_v2() {
        //No cleanup required
        //Thrust deallocates any associated storage when it goes out of scope
    }


    //--Helper Functions--//

    //Cute, slow, factorial function
    __device__ long long int fact(int n)
    {
        //Can make this faster by having 3 different functions
        //fact, 3fact, and 6fact
        //They will compute and store factorials, so that when we have to compute
        //the next factorial, it can just take the last factorial and muliply by
        //n, 3n, or 6n
        //Leave it for now, as chudnovsky kernel is already fast and this was 
        //just a fun toy example
        if (n<2)
            return 1;
        return n*fact(n - 1);
    }

    //atomicAdd function for doubles
    __device__ double d_atomicAdd(double* address, double val)
    {
        //For some reason, I wasn't able to use regular atomicAdd with double values
        //So I just defined my own function double atomicAdd function to get around it
        //I'm assuming this is an issue with my own setup, but on the off chance it isn't,
        //it's best to just define it here
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                 __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
    return __longlong_as_double(old);
    }

}
