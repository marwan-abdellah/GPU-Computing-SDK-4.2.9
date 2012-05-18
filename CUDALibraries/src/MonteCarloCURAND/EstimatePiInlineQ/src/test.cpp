/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "../inc/test.h"

#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <cassert>
#include <typeinfo>
#include <shrUtils.h>
#include <cuda_runtime_api.h>

#include "../inc/piestimator.h"

template <typename Real>
bool Test<Real>::operator()()
{
    using std::stringstream;
    using std::endl;
    using std::setw;

    // Get device properties
    struct cudaDeviceProp deviceProperties;
    cudaError_t cudaResult = cudaGetDeviceProperties(&deviceProperties, device);
    if (cudaResult != cudaSuccess)
    {
        std::string msg("Could not get device properties: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Evaluate on GPU
    shrLog("Estimating Pi on GPU (%s)\n\n", deviceProperties.name);
    PiEstimator<Real> estimator(numSims, device, threadBlockSize);
    shrDeltaT(0);
    Real result = estimator();
    elapsedTime = shrDeltaT(0);

    // Tolerance to compare result with expected
    // This is just to check that nothing has gone very wrong with the
    // test, the actual accuracy of the result depends on the number of
    // Monte Carlo trials
    const Real tolerance = static_cast<Real>(0.01);
    
    // Display results
    Real abserror = fabs(result - static_cast<Real>(PI));
    Real relerror = abserror / static_cast<Real>(PI);
    shrLog("Precision:      %s\n", (typeid(Real) == typeid(double)) ? "double" : "single");
    shrLog("Number of sims: %d\n", numSims);
    shrLog("Tolerance:      %e\n", tolerance);
    shrLog("GPU result:     %e\n", result);
    shrLog("Expected:       %e\n", PI);
    shrLog("Absolute error: %e\n", abserror);
    shrLog("Relative error: %e\n\n", relerror);

    // Check result
    if (relerror > tolerance)
    {
        shrLogEx(LOGBOTH | ERRORMSG, 0, "computed result (%e) does not match expected result (%e).\n", result, PI);
        pass = false;
    }
    else
    {
        pass = true;
    }

    // Print results
#ifdef GPU_PROFILING
    shrLogEx(LOGBOTH | MASTER, 0, "MonteCarloEstimatePiInlineP, Performance = %.4f sims/s, Time = %.5f s, NumDevsUsed = %u, Blocksize = %u\n", 
            numSims / elapsedTime, elapsedTime, 1, threadBlockSize);
#endif
        
    return pass;
}

// Explicit template instantiation
template struct Test<float>;
template struct Test<double>;
