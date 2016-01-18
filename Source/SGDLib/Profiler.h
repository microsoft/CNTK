//
// <copyright file="Profiler.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

class Profiler
{
public:
    // Initializes profiler asking it to take given number of samples (0 to disable) and then stop
    Profiler(int numSamples);
    ~Profiler(); // stops the profiler
    // Notifies transition to the next sample
    void NextSample();

private:
    void Start();
    void Stop();

    int m_numSamples;
    bool m_isProfilingActive;
};
