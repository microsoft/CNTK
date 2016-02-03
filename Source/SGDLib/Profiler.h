//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
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
