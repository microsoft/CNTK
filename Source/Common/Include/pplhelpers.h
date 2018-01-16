//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// pplhelpers.h -- some helpers for PPL library
//

#pragma once

#ifndef __unix__
#include <ppl.h>
#endif
namespace msra { namespace parallel {

// ===========================================================================
// helpers related to multiprocessing and NUMA
// ===========================================================================

// determine number of CPU cores on this machine
static inline size_t determine_num_cores()
{
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return sysInfo.dwNumberOfProcessors;
}

extern size_t ppl_cores; // number of cores to run on as requested by user

static inline void set_cores(size_t cores)
{
    ppl_cores = cores;
}

static inline size_t get_cores() // if returns 1 then no parallelization will be done
{
    return ppl_cores;
}

#if 0
// execute body() a bunch of times for hopefully each core
// This is not precise. Cores will be hit multiple times, and some cores may not be touched.
template <typename FUNCTION> void for_all_numa_nodes_approximately (const FUNCTION & body)
{
    if (ppl_cores > 1)  // parallel computation (regular)
        parallel_for ((size_t) 0, ppl_cores * 2, (size_t) 1, [&](size_t) { body(); });
    else            // for comparison: single-threaded (this also documents what the above means)
        body();
}
#endif

// wrapper around Concurrency::parallel_for() to allow disabling parallelization altogether
template <typename FUNCTION>
void parallel_for(size_t begin, size_t end, size_t step, const FUNCTION& f)
{
    const size_t cores = ppl_cores;
    if (cores > 1) // parallel computation (regular)
    {
        //fprintf (stderr, "foreach_index_block: computing %d blocks of %d frames on %d cores\n", nblocks, nfwd, determine_num_cores());
        Concurrency::parallel_for(begin, end, step, f);
    }
    else // for comparison: single-threaded (this also documents what the above means)
    {
        //fprintf (stderr, "foreach_index_block: computing %d blocks of %d frames on a single thread\n", nblocks, nfwd);
        for (size_t j0 = begin; j0 < end; j0 += step)
            f(j0);
    }
}

// execute a function 'body (j0, j1)' for j = [0..n) in chunks of ~targetstep in 'cores' cores
// Very similar to parallel_for() except that body function also takes end index,
// and the 'targetsteps' gets rounded a little to better map to 'cores.'
// ... TODO: Currently, 'cores' does not limit the number of threads in parallel_for() (not so critical, fix later or never)
template <typename FUNCTION>
void foreach_index_block(size_t n, size_t targetstep, size_t targetalignment, const FUNCTION& body)
{
    const size_t cores = ppl_cores;
    const size_t maxnfwd = 2 * targetstep;
    size_t nblocks = (n + targetstep / 2) / targetstep;
    if (nblocks == 0)
        nblocks = 1;
    // round to a multiple of the number of cores
    if (nblocks < cores) // less than # cores -> round up
        nblocks = (1 + (nblocks - 1) / cores) * cores;
    else // more: round down (reduce overhead)
        nblocks = nblocks / cores * cores;
    size_t nfwd = 1 + (n - 1) / nblocks;
    assert(nfwd * nblocks >= n);
    if (nfwd > maxnfwd)
        nfwd = maxnfwd; // limit to allocated memory just in case
    // ... TODO: does the above actually do anything/significant? nfwd != targetstep?

    // enforce alignment
    nfwd = (1 + (nfwd - 1) / targetalignment) * targetalignment;

    // execute it!
    parallel_for(0, n, nfwd, [&](size_t j0)
                 {
                    // Take the min(j0+nfwd, n)
                    body(j0, (j0 + nfwd < n) ? j0 + nfwd : n);
                 });
}

}}
