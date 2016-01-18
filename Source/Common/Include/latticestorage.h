//
// <copyright file="latticestorage.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// latticestorage.h -- basic data structures for storing lattices

#pragma once

#include "Basics.h"
#include <string> // for the error message in checkoverflow() only
#include <stdexcept>
#include <stdint.h>
#include <cstdio>

#undef INITIAL_STRANGE // [v-hansu] intialize structs to strange values
#define PARALLEL_SIL   // [v-hansu] process sil on CUDA, used in other files, please search this
#define LOGZERO -1e30f

namespace msra { namespace lattices {

static void checkoverflow(size_t fieldval, size_t targetval, const char* fieldname)
{
    if (fieldval != targetval)
        RuntimeError("lattice: bit field %s too small for value 0x%x (cut from 0x%x)", fieldname, (unsigned int) targetval, (unsigned int) fieldval);
}

struct nodeinfo
{
    //uint64_t firstinedge : 24;  // index of first incoming edge
    //uint64_t firstoutedge : 24; // index of first outgoing edge
    //uint64_t t : 16;            // time associated with this
    unsigned short t; // time associated with this
    nodeinfo(size_t pt)
        : t((unsigned short) pt) //, firstinedge (NOEDGE), firstoutedge (NOEDGE)
    {
        checkoverflow(t, pt, "nodeinfo::t");
        //checkoverflow (firstinedge, NOEDGE, "nodeinfo::firstinedge");
        //checkoverflow (firstoutedge, NOEDGE, "nodeinfo::firstoutedge");
    }
    nodeinfo() // [v-hansu] initialize to impossible values
    {
#ifdef INITIAL_STRANGE
        t = unsigned short(-1);
#endif
    }
};
// V2 format: a and l are stored in separate vectors
struct edgeinfo
{
    uint64_t S : 19;          // start node
    uint64_t unused : 1;      // (for future use)
    uint64_t E : 19;          // end node
    uint64_t implysp : 1;     // 1--alignment ends with a /sp/ that is not stored
    uint64_t firstalign : 24; // index into align for first entry; end is firstalign of next edge
    edgeinfo(size_t pS, size_t pE, size_t pfirstalign)
        : S(pS), E(pE), firstalign(pfirstalign), unused(0), implysp(0)
    {
        checkoverflow(S, pS, "edgeinfowithscores::S");
        checkoverflow(E, pE, "edgeinfowithscores::E");
        checkoverflow(firstalign, pfirstalign, "edgeinfowithscores::firstalign");
    }
    edgeinfo() // [v-hansu] initialize to impossible values
    {
#ifdef INITIAL_STRANGE
        S = uint64_t(-1);
        unused = uint64_t(-1);
        E = uint64_t(-1);
        implysp = uint64_t(-1);
        firstalign = uint64_t(-1);
#endif
    }
};
// V1 format: a and l are included in the edge itself
struct edgeinfowithscores : edgeinfo
{
    float a;
    float l;
    edgeinfowithscores(size_t pS, size_t pE, float a, float l, size_t pfirstalign)
        : edgeinfo(pS, pE, pfirstalign), a(a), l(l)
    {
    }
    edgeinfowithscores() // [v-hansu] initialize to impossible values
    {
#ifdef INITIAL_STRANGE
        a = LOGZERO;
        l = LOGZERO;
#endif
    }
};
struct aligninfo // phonetic alignment
{
    unsigned int unit : 19;   // triphone index
    unsigned int frames : 11; // duration in frames
    // note: V1 did not have the following, which were instead the two 2 bits of 'frames'
    unsigned int unused : 1; // (for future use)
    unsigned int last : 1;   // set for last entry
    aligninfo(size_t punit, size_t pframes)
        : unit((unsigned int) punit), frames((unsigned int) pframes), unused(0), last(0)
    {
        checkoverflow(unit, punit, "aligninfo::unit");
        checkoverflow(frames, pframes, "aligninfo::frames");
    }
    aligninfo() // [v-hansu] initialize to impossible values
    {
#ifdef INITIAL_STRANGE
        unit = unsigned int(-1);
        frames = unsigned int(-1);
        unused = unsigned int(-1);
        last = unsigned int(-1);
#endif
    }
    template <class IDMAP>
    void updateunit(const IDMAP& idmap /*[unit] -> new unit*/) // update 'unit' w.r.t. a different mapping, with bit-field overflow check
    {
        const size_t mappedunit = idmap[unit];
        unit = (unsigned int) mappedunit;
        checkoverflow(unit, mappedunit, "aligninfo::unit");
    }
};
};
};
