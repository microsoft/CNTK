//
// <copyright file="fixtures.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

class RandomSeedFixture
{
    static unsigned long s_counter;
public:
    RandomSeedFixture();
    unsigned long IncrementCounter();
};
