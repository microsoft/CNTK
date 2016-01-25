//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include <cassert>
#include <stdio.h>
#include "Profiler.h"
#include "BestGpu.h" // for CPUONLY flag only

#ifndef CPUONLY
#include <cuda_profiler_api.h>
#else
// If compiling without CUDA, defining profiler control functions as no-op stubs
void cudaProfilerStart()
{
}
void cudaProfilerStop()
{
}
#endif

Profiler::Profiler(int numSamples)
    : m_numSamples(numSamples),
      m_isProfilingActive(false)
{
}

Profiler::~Profiler()
{
    if (m_isProfilingActive)
        Stop();
}

void Profiler::Start()
{
    assert(!m_isProfilingActive);
    m_isProfilingActive = true;
    fprintf(stderr, "Starting profiling\n");
    cudaProfilerStart();
}

void Profiler::NextSample()
{
    if (m_isProfilingActive)
    {
        if (--m_numSamples == 0)
            Stop();
    }
    else
    {
        if (m_numSamples > 0)
            Start();
    }
}

void Profiler::Stop()
{
    assert(m_isProfilingActive);
    cudaProfilerStop();
    fprintf(stderr, "Stopping profiling\n");
    m_isProfilingActive = false;
}
