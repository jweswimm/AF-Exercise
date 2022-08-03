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

    __device__ double d_atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                 __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
    return __longlong_as_double(old);
    }



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
        cudaMalloc((void**)&d_states, block_size * grid_size * sizeof(curandState));

        //Call curand_init kernel
        random_init << <grid_size, block_size >> > (seed, samples, d_states);


    }


    //kernel for Monte Carlo estimation of pi
    __global__ void est_pi(int* count, unsigned int seed, int samples, curandState* states) {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < samples; idx += gridDim.x * blockDim.x) {

//            curand_init(seed, idx, 0, &states[idx]);

            //Note: uniform distribution excludes 0 and includes 1
            //We may want a real distribution instead
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



    //Monte Carlo estimation of pi function
    double pi_v1()
    {
        /*
        Utility & Explanation: We use a Monte Carlo method to estimate pi.
        In 2D, we overlay a unit circle on a unit square. The area of the
        unit circle is \pi r^2 and the area of the unit square is 4 r^2.
        If we randomly generate x and y coordinates for points,
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

        //Initialize count to 0
        thrust::device_vector<int> count(1, 0);

        //Call kernel
        est_pi << <grid_size, block_size >> > (thrust::raw_pointer_cast(count.data()), seed, samples, d_states);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        return 4 * double(count[0]) / double(samples);
    }

    void pi_reset_v1()
    {
        /*
          TODO This function should contain the clean up. You should add
          memory deallocation etc here.
        */

        //Free d_states, as it was on the device
        cudaFree(d_states);


        //Don't need to free thrust device vector count
    }
    __device__ double fact(int n)
    {
        if (n<2)
            return 1;

        return n*fact(n - 1);
    }

    __global__ void chudnovsky(int N, double* m, double* l, double* x, double* sum) {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < N; idx += gridDim.x * blockDim.x) {

//            l[idx] = l[idx - 1] + 163*127*19 * 11 * 9 * 2;
 //           x[idx] = x[idx - 1] * 640320*640320*640320;
  //          m[idx] = m[idx - 1] * (((12*idx - 2) * (12*idx -6) * (12*idx - 10))/(idx*idx*idx));
            double M = (fact(6 * idx))/(fact(3*idx)*pow(fact(idx),3));
            double L = 54514013 * idx + 13591409;
            double X = pow(-262537412640768000, idx);

            //Calculate individual components of the sum
//            components[idx] = ;

                d_atomicAdd((sum), (M * L) / X);


        }

    }

    void pi_init_v2() {
        //Calculate C
        C = 426880 * sqrt(10005);

        N = 4;

        //Allocate space for the sum
        //Each thread will be one value of the sum
        thrust::device_vector<double> dum(N, 0);
        sum_components = dum;
        m = dum;
        m[0] = 1;
        l = dum;
        l[0] = 13591409;
        x = dum;
        x[0] = 1;

        block_size = 1024;
        grid_size = ceil(float(N) / float(block_size));


    }





    double pi_v2() {


        thrust::device_vector<double> sum(1, 0.0);

        chudnovsky << < grid_size, block_size >> > (N, thrust::raw_pointer_cast(m.data()),
            thrust::raw_pointer_cast(l.data()), thrust::raw_pointer_cast(x.data()),
                thrust::raw_pointer_cast(sum.data()));


//        std::cout << C * (1 / sum[0]);
//        std::cout << C * (1/sum[0]) << std::endl;

        return C * (1/sum[0]);
    }

    void pi_reset_v2() {

    }
}
