//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// numahelpers.h -- some helpers with NUMA
//

#pragma once

#ifndef __unix__
#include <Windows.h>
#include "pplhelpers.h"
#endif
#include <stdexcept>
#include "simple_checked_arrays.h"
#include "Basics.h" // for FormatWin32Error

namespace msra { namespace numa {

// ... TODO: this can be a 'static', as it should only be set during foreach_node but not outside
extern int node_override; // -1 = normal operation; >= 0: force a specific NUMA node

// force a specific NUMA node (only do this during single-threading!)
static inline void overridenode(int n = -1)
{
    node_override = n;
}

// get the number of NUMA nodes we would like to distinguish
static inline size_t getnumnodes()
{
#ifdef CNTK_UWP
	return 1;
#else
    ULONG n;
    if (!GetNumaHighestNodeNumber(&n))
        return 1;
    return n + 1;
#endif
}

// execute body (node, i, n), i in [0,n) on all NUMA nodes in small chunks
template <typename FUNCTION>
void parallel_for_on_each_numa_node(bool multistep, const FUNCTION &body)
{
    // get our configuration
    const size_t cores = ppl_cores;
    assert(cores > 0);
    const size_t nodes = getnumnodes();
    const size_t corespernode = (cores - 1) / nodes + 1;
    // break into 8 steps per thread
    const size_t stepspernode = multistep ? 16 : 1;
    const size_t steps = corespernode * stepspernode;
    // now run on many threads, hoping to hit all NUMA nodes, until we are done
    hardcoded_array<LONG /*unsigned int*/, 256> nextstepcounters; // next block to run for a NUMA node
    if (nodes > nextstepcounters.size())
        LogicError("parallel_for_on_each_numa_node: nextstepcounters buffer too small, need to increase hard-coded size");
    for (size_t k = 0; k < nodes; k++)
        nextstepcounters[k] = 0;
    overridenode();
    //unsigned int totalloops = 0;    // for debugging only, can be removed later
    msra::parallel::parallel_for(0, nodes * steps /*execute each step on each NUMA node*/, 1, [&](size_t /*dummy*/)
                                 {
                                     const size_t numanodeid = getcurrentnode();
                                     // find a node that still has work left, preferring our own node
                                     // Towards the end we will run on wrong nodes, but what can we do.
                                     for (size_t node1 = numanodeid; node1 < numanodeid + nodes; node1++)
                                     {
                                         const size_t node = node1 % nodes;
                                         const unsigned int step = InterlockedIncrement(&nextstepcounters[node]) - 1; // grab this step
                                         if (step >= steps)                                                           // if done then counter has exceeded the required number of steps
                                             continue;                                                                // so try next NUMA node
                                         // found one: execute and terminate loop
                                         body(node, step, steps);
                                         //InterlockedIncrement (&totalloops);
                                         return; // done
                                     }
                                     // oops??
                                     LogicError("parallel_for_on_each_numa_node: no left-over block found--should not get here!!");
                                 });
    //assert (totalloops == nodes * steps);
}

// execute a passed function once for each NUMA node
// This must be run from the main thread only.
// ... TODO: honor ppl_cores == 1 for comparative measurements against single threads.
template <typename FUNCTION>
static void foreach_node_single_threaded(const FUNCTION &f)
{
    const size_t n = getnumnodes();
    for (size_t i = 0; i < n; i++)
    {
        overridenode((int) i);
        f();
    }
    overridenode(-1);
}

// get the current NUMA node
static inline size_t getcurrentnode()
{
#ifdef CNTK_UWP
	return 0;
#else
    // we can force it to be a certain node, for use in initializations
    if (node_override >= 0)
        return (size_t) node_override;
    // actually use current node
    DWORD i = GetCurrentProcessorNumber(); // note: need to change for >63 processors
    UCHAR n;
    if (!GetNumaProcessorNode((UCHAR) i, &n))
        return 0;
    if (n == 0xff)
        LogicError("GetNumaProcessorNode() failed to determine NUMA node for GetCurrentProcessorNumber()??");
    return n;
#endif
}

// allocate memory
// Allocation seems to be at least on a 512-byte boundary. We nevertheless verify alignment requirements.
typedef LPVOID(WINAPI *VirtualAllocExNuma_t)(HANDLE, LPVOID, SIZE_T, DWORD, DWORD, DWORD);
static VirtualAllocExNuma_t VirtualAllocExNuma = (VirtualAllocExNuma_t) -1;
#ifndef CNTK_UWP
static inline void *malloc(size_t n, size_t align)
{
    // VirtualAllocExNuma() only exists on Vista+, so go through an explicit function pointer
    if (VirtualAllocExNuma == (VirtualAllocExNuma_t) -1)
    {
        VirtualAllocExNuma = (VirtualAllocExNuma_t) GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")), "VirtualAllocExNuma");
    }

    // if we have the function then do a NUMA-aware allocation
    void *p;
    if (VirtualAllocExNuma != NULL)
    {
        size_t node = getcurrentnode();
        // "all Win32 heap allocations that are 1 MB or greater are forwarded directly to NtAllocateVirtualMemory
        // when they are allocated and passed directly to NtFreeVirtualMemory when they are freed" Greg Colombo, 2010/11/17
        if (n < 1024 * 1024)
            n = 1024 * 1024; // -> brings NUMA-optimized code back to Node Interleave level (slightly faster)
        p = VirtualAllocExNuma(GetCurrentProcess(), NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE, (DWORD) node);
    }
    else // on old OS call no-NUMA version
    {
        p = VirtualAllocEx(GetCurrentProcess(), NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    }
    if (p == NULL)
        fprintf(stderr, "numa::malloc: failed allocating %lu bytes with alignment %lu\n", (unsigned long)n, (unsigned long)align);
    if (((size_t) p) % align != 0)
        LogicError("VirtualAllocExNuma() returned an address that does not match the alignment requirement");
    return p;
}

// free memory allocated with numa::malloc()
static inline void free(void *p)
{
    assert(p != NULL);
    if (!VirtualFree(p, 0, MEM_RELEASE))
        LogicError("VirtualFreeEx failure");
}
#endif // CNTK_UWP

// dump memory allocation
static inline void showavailablememory(const char *what)
{
#ifndef CNTK_UWP
    size_t n = getnumnodes();
    for (size_t i = 0; i < n; i++)
    {
        ULONGLONG availbytes = 0;
        BOOL rc = GetNumaAvailableMemoryNode((UCHAR) i, &availbytes);
        const double availmb = availbytes / (1024.0 * 1024.0);
        if (rc)
            fprintf(stderr, "%s: %8.2f MB available on NUMA node %lu\n", what, availmb, (unsigned long)i);
        else
            fprintf(stderr, "%s: error for getting available memory on NUMA node %lu\n", what, (unsigned long)i);
    }
#endif
}

// determine NUMA node with most memory available
static inline size_t getmostspaciousnumanode()
{
#ifdef CNTK_UWP
    return 0;
#else
    size_t n = getnumnodes();
    size_t bestnode = 0;
    ULONGLONG bestavailbytes = 0;
    for (size_t i = 0; i < n; i++)
    {
        ULONGLONG availbytes = 0;
        GetNumaAvailableMemoryNode((UCHAR) i, &availbytes);
        if (availbytes > bestavailbytes)
        {
            bestavailbytes = availbytes;
            bestnode = i;
        }
    }
    return bestnode;
#endif
}

#if 0 // this is no longer used (we now parallelize the big matrix products directly)
// class to manage multiple copies of data on local NUMA nodes
template<class DATATYPE,class CACHEDTYPE> class numalocaldatacache
{
    numalocaldatacache (const numalocaldatacache&); numalocaldatacache & operator= (const numalocaldatacache&);

    // the data set we associate to
    const DATATYPE & data;

    // cached copies of the models for NUMA
    vector<unique_ptr<CACHEDTYPE>> cache;

    // get the pointer to the clone for the NUMA node of the current thread (must exist)
    CACHEDTYPE * getcloneptr()
    {
        return cache[getcurrentnode()].get();
    }
public:
    numalocaldatacache (const DATATYPE & data) : data (data), cache (getnumnodes())
    {
        foreach_node_single_threaded ([&]()
        {
            cache[getcurrentnode()].reset (new CACHEDTYPE (data));
        });
    }

    // this takes the cached versions of the parent model
    template<typename ARGTYPE1,typename ARGTYPE2,typename ARGTYPE3>
    numalocaldatacache (numalocaldatacache<DATATYPE,DATATYPE> & parentcache, const ARGTYPE1 & arg1, const ARGTYPE2 & arg2, const ARGTYPE3 & arg3) : data (*(DATATYPE*)nullptr), cache (getnumnodes())
    {
        foreach_node_single_threaded ([&]()
        {
            const DATATYPE & parent = parentcache.getclone();
            size_t numanodeid = getcurrentnode();
            cache[numanodeid].reset (new CACHEDTYPE (parent, arg1, arg2, arg3));
        });
    }

    // re-clone --update clones from the cached 'data' reference
    // This is only valid if CACHEDTYPE==DATATYPE.
    // ... parallelize this!
    void reclone()
    {
        parallel_for_on_each_numa_node (true, [&] (size_t numanodeid, size_t step, size_t steps)
        {
            if (step != 0)
                return;     // ... TODO: tell parallel_for_on_each_numa_node() to only have one step, or parallelize
            cache[numanodeid].get()->copyfrom (data);    // copy it all over
        });
    }

    // post-process all clones
    // 'numanodeid' is ideally the current NUMA node most of the time, but not required.
    template<typename POSTPROCFUNC>
    void process (const POSTPROCFUNC & postprocess)
    {
        parallel_for_on_each_numa_node (true, [&] (size_t numanodeid, size_t step, size_t steps)
        {
            postprocess (*cache[numanodeid].get(), step, steps);
        });
    }

    // a thread calls this to get the data pre-cloned for its optimal NUMA node
    // (only works for memory allocated through msra::numa::malloc())
    const CACHEDTYPE & getclone() const { return *getcloneptr(); }
    CACHEDTYPE & getclone()             { return *getcloneptr(); }
};
#endif
};
};
