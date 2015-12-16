//
// <copyright file="fixtures.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

// This fixure is used in the unit tests to provide a sequence of consecutive 
// values to seed the random number generators. In order to generate the seed 
// values independent of the execution order of the tests, in each unit test 
// the sequence is restarted from 0.
class RandomSeedFixture
{
    unsigned long counter = 0ul;
public:
    unsigned long IncrementCounter() { return counter++; };
};
