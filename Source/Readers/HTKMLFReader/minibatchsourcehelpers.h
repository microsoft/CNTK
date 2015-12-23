//
// <copyright file="minibatchsourcehelpers.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// minibatchsourcehelpers.h -- helper classes for minibatch sources
//

#pragma once

#include "Basics.h"
#include <stdio.h>
#include <vector>
#include <algorithm>
#ifndef __unix__
#include "ssematrix.h"      // for matrix type
#endif

#pragma warning (disable: 4505) // unreferenced local function has been removed

namespace msra { namespace dbn {

// ---------------------------------------------------------------------------
// augmentneighbors() -- augmenting features with their neighbor frames
// ---------------------------------------------------------------------------
    
// implant a sub-vector into a vector, for use in augmentneighbors
template<class INV, class OUTV> static void copytosubvector (const INV & inv, size_t subvecindex, OUTV & outv)
{
    size_t subdim = inv.size();
    assert (outv.size() % subdim == 0);
    size_t k0 = subvecindex * subdim;
    foreach_index (k, inv)
        outv[k + k0] = inv[k];
}

// compute the augmentation extent (how many frames added on each side)
static size_t augmentationextent (size_t featdim/*augment from*/, size_t modeldim/*to*/)
{
    const size_t windowframes = modeldim / featdim;   // total number of frames to generate
    const size_t extent = windowframes / 2;           // extend each side by this

    if (modeldim % featdim != 0)
        RuntimeError("augmentationextent: model vector size not multiple of input features");
    if (windowframes % 2 == 0)
        RuntimeError("augmentationextent: neighbor expansion of input features to %d not symmetrical", (int)windowframes);

    return extent;
}

// augment neighbor frames for a frame (correctly not expanding across utterance boundaries)
// The boundaryflags[] array, if not empty, flags first (-1) and last (+1) frame, i.e. frames to not expand across.
// The output 'v' must have te-ts columns.
template<class MATRIX, class VECTOR> static void augmentneighbors (const MATRIX & frames, const std::vector<char> & boundaryflags, size_t t,
                                                                   VECTOR & v)
{
    // how many frames are we adding on each side
    const size_t extent = augmentationextent (frames[t].size(), v.size());

    // copy the frame and its neighbors
    // Once we hit a boundaryflag in either direction, do not move index beyond.
    copytosubvector (frames[t], extent, v);     // frame[t] sits right in the middle
    size_t t1 = t;  // index for frames on to the left
    size_t t2 = t;  // and right
    for (size_t n = 1; n <= extent; n++)
    {
#ifdef SAMPLING_EXPERIMENT
        if (boundaryflags.empty())  // boundary flags not given: 'frames' is full utterance
        {
            if (t1 >= SAMPLING_EXPERIMENT) t1 -= SAMPLING_EXPERIMENT;                   // index does not move beyond boundary
            if (t2 + SAMPLING_EXPERIMENT < frames.size()) t2 += SAMPLING_EXPERIMENT;
        }
        else
        {
            if (boundaryflags[t1] != -1) t1 -= SAMPLING_EXPERIMENT;  // index does not move beyond a set boundaryflag,
            if (boundaryflags[t2] != 1) t2 += SAMPLING_EXPERIMENT;   // because that's the start/end of the utterance
        }
#else
        if (boundaryflags.empty())  // boundary flags not given: 'frames' is full utterance
        {
            if (t1 > 0) t1--;                   // index does not move beyond boundary
            if (t2 + 1 < frames.size()) t2++;
        }
        else
        {
            if (boundaryflags[t1] != -1) t1--;  // index does not move beyond a set boundaryflag,
            if (boundaryflags[t2] != 1) t2++;   // because that's the start/end of the utterance
        }
#endif
        copytosubvector (frames[t1], extent - n, v);
        copytosubvector (frames[t2], extent + n, v);
    }
}

// augment neighbor frames for a frame (correctly not expanding across utterance boundaries)
// The boundaryflags[] array, if not empty, flags first (-1) and last (+1) frame, i.e. frames to not expand across.
// The output 'v' must have te-ts columns.
template<class MATRIX, class VECTOR> static void augmentneighbors(const MATRIX & frames, const std::vector<char> & boundaryflags, size_t t, const size_t leftextent, const size_t rightextent,
    VECTOR & v)
{

    // copy the frame and its neighbors
    // Once we hit a boundaryflag in either direction, do not move index beyond.
    copytosubvector(frames[t], leftextent, v);     // frame[t] sits right in the middle
    size_t t1 = t;  // index for frames on to the left
    size_t t2 = t;  // and right

    for (size_t n = 1; n <= leftextent; n++)
    {
        if (boundaryflags.empty())  // boundary flags not given: 'frames' is full utterance
        {
            if (t1 > 0) t1--;                   // index does not move beyond boundary
        }
        else
        {
            if (boundaryflags[t1] != -1) t1--;  // index does not move beyond a set boundaryflag,
        }
        copytosubvector(frames[t1], leftextent - n, v);
    }
    for (size_t n = 1; n <= rightextent; n++)
    {
        if (boundaryflags.empty())  // boundary flags not given: 'frames' is full utterance
        {
            if (t2 + 1 < frames.size()) t2++;
        }
        else
        {
            if (boundaryflags[t2] != 1) t2++;   // because that's the start/end of the utterance
        }
        copytosubvector(frames[t2], rightextent + n, v);
    }
}

// augment neighbor frames for one frame t in frames[] according to boundaryflags[]; result returned in column j of v
template<class INMATRIX, class OUTMATRIX> static void augmentneighbors (const INMATRIX & frames, const std::vector<char> & boundaryflags, size_t t,
                                                                        OUTMATRIX & v, size_t j)
{
    auto v_j = v.col(j); // the vector to fill in
    augmentneighbors (frames, boundaryflags, t, v_j);
}

// augment neighbor frames for one frame t in frames[] according to boundaryflags[]; result returned in column j of v
template<class INMATRIX, class OUTMATRIX> static void augmentneighbors(const INMATRIX & frames, const std::vector<char> & boundaryflags, size_t t, size_t leftextent, size_t rightextent,
    OUTMATRIX & v, size_t j)
{
    auto v_j = v.col(j); // the vector to fill in
    augmentneighbors(frames, boundaryflags, t, leftextent, rightextent, v_j);
}

// augment neighbor frames for a sequence of frames (part of an utterance, possibly spanning across boundaries)
template<class MATRIX> static void augmentneighbors (const std::vector<std::vector<float>> & frames, const std::vector<char> & boundaryflags,
                                                     size_t ts, size_t te,  // range [ts,te)
                                                     MATRIX & v)
{
    for (size_t t = ts; t < te; t++)
    {
        auto v_t = v.col(t-ts); // the vector to fill in
        augmentneighbors (frames, boundaryflags, t, v_t);
    }
}


// augment neighbor frames for a sequence of frames (part of an utterance, possibly spanning across boundaries)
template<class MATRIX> static void augmentneighbors(const std::vector<std::vector<float>> & frames, const std::vector<char> & boundaryflags, size_t leftextent, size_t rightextent,
    size_t ts, size_t te,  // range [ts,te)
    MATRIX & v)
{
    for (size_t t = ts; t < te; t++)
    {
        auto v_t = v.col(t - ts); // the vector to fill in
        augmentneighbors(frames, boundaryflags, t, leftextent, rightextent, v_t);
    }
}

// ---------------------------------------------------------------------------
// randomordering -- class to help manage randomization of input data
// ---------------------------------------------------------------------------

static inline size_t rand (const size_t begin, const size_t end)
{
    const size_t randno = ::rand() * RAND_MAX + ::rand();   // BUGBUG: still only covers 32-bit range
    return begin + randno % (end - begin);
}

class randomordering                // note: NOT thread-safe at all
{
    // constants for randomization
    const static size_t randomizeAuto=0;
    const static size_t randomizeDisable=(size_t)-1;

    typedef unsigned int INDEXTYPE; // don't use size_t, as this saves HUGE amounts of RAM
    std::vector<INDEXTYPE> map;          // [t] -> t' indices in randomized order
    size_t currentseed;             // seed for current sequence
    size_t randomizationrange;      // t - randomizationrange/2 <= t' < t + randomizationrange/2 (we support this to enable swapping)
                                    // special values (randomizeAuto, randomizeDisable)
    void invalidate() { currentseed = (size_t) -1; }
public:
    randomordering() { invalidate(); }

    void resize (size_t len, size_t p_randomizationrange) { randomizationrange = p_randomizationrange>0?p_randomizationrange:len; map.resize (len); invalidate(); }

    // return the randomized feature bounds for a time range
    std::pair<size_t,size_t> bounds (size_t ts, size_t te) const
    {
        size_t tbegin = max (ts, randomizationrange/2) - randomizationrange/2;
        size_t tend = min (te + randomizationrange/2, map.size());
        return std::make_pair<size_t, size_t>(std::move(tbegin), std::move(tend));
    }

    // this returns the map directly (read-only) and will lazily initialize it for a given seed
    const std::vector<INDEXTYPE> & operator() (size_t seed) //throw()
    {
        // if wrong seed then lazily recache the sequence
        if (seed != currentseed)
        {
            // test for numeric overflow
            if (map.size()-1 != (INDEXTYPE) (map.size()-1))
                RuntimeError("randomordering: INDEXTYPE has too few bits for this corpus");
            // 0, 1, 2...
            foreach_index (t, map) map[t] = (INDEXTYPE) t;
            // now randomize them
            if (randomizationrange != randomizeDisable)
            {
    #if 1       // change to 0 to disable randomizing
                if (map.size() > RAND_MAX * (size_t) RAND_MAX)
                    RuntimeError("randomordering: too large training set: need to change to different random generator!");
                srand ((unsigned int) seed);
                size_t retries = 0;
                foreach_index (t, map)
                {
                    for (int tries = 0; tries < 5; tries++)
                    {
                        // swap current pos with a random position
                        // Random positions are limited to t+randomizationrange.
                        // This ensures some locality suitable for paging with a sliding window.
                        const size_t tbegin = max ((size_t) t, randomizationrange/2) - randomizationrange/2; // range of window  --TODO: use bounds() function above
                        const size_t tend = min (t + randomizationrange/2, map.size());
                        assert (tend >= tbegin);                    // (guard against potential numeric-wraparound bug)
                        const size_t trand = rand (tbegin, tend);   // random number within windows
                        assert ((size_t) t <= trand + randomizationrange/2 && trand < (size_t) t + randomizationrange/2);
                        // if range condition is fulfilled then swap
                        if (trand <= map[t] + randomizationrange/2 && map[t] < trand + randomizationrange/2
                            && (size_t) t <= map[trand] + randomizationrange/2 && map[trand] < (size_t) t + randomizationrange/2)
                        {
                            std::swap (map[t], map[trand]);
                            break;
                        }
                        // but don't multi-swap stuff out of its range (for swapping positions that have been swapped before)
                        // instead, try again with a different random number
                        retries++;
                    }
                }
                fprintf (stderr, "randomordering: %d retries for %d elements (%.1f%%) to ensure window condition\n", (int)retries, (int)map.size(), 100.0 * retries / map.size());
                // ensure the window condition
                foreach_index (t, map) assert ((size_t) t <= map[t] + randomizationrange/2 && map[t] < (size_t) t + randomizationrange/2);
    #if 1       // and a live check since I don't trust myself here yet
                foreach_index (t, map) if (!((size_t) t <= map[t] + randomizationrange/2 && map[t] < (size_t) t + randomizationrange/2))
                {
                    fprintf (stderr, "randomordering: windowing condition violated %d -> %d\n", t, map[t]);
                    LogicError("randomordering: windowing condition violated");
                }
    #endif
    #endif
    #if 1       // test whether it is indeed a unique complete sequence
                auto map2 = map;
                std::sort (map2.begin(), map2.end());
                foreach_index (t, map2) assert (map2[t] == (size_t) t);
    #endif
                fprintf (stderr, "randomordering: recached sequence for seed %d: %d, %d, ...\n", (int) seed, (int) map[0], (int) map[1]);
            }
            currentseed = seed;
        }
        return map; // caller can now access it through operator[]
    }
};

// Using a smaller sized integer type to store the state ids saves a huge amount of space
// We need more than 2 bytes (i.e. an unsigned int or bigger) to support untied context-dependent phones 
// TODO: This is currently being hardcoded to unsigned short for saving space, which means untied context-dependent phones
// will not work. This needs to be changed to dynamically choose what size to use based on the number of class ids.
typedef unsigned short CLASSIDTYPE;
typedef unsigned short HMMIDTYPE;

#ifndef _MSC_VER
// these needed below with gcc because some regex function is not implemented properly there
struct MatchPathSeparator
{
    bool operator()(char ch) const { return ch == '\\' || ch == '/'; }
};
static inline std::string basename(std::string const& pathname)
{
    return std::string(std::find_if(pathname.rbegin(), pathname.rend(), MatchPathSeparator()).base(), pathname.end());
}
static inline std::string removeExtension(std::string const& filename)
{
    return filename.substr(0, filename.find_last_of("."));
}
static inline std::wstring basename(std::wstring const& pathname)
{
    return std::wstring(std::find_if(pathname.rbegin(), pathname.rend(), MatchPathSeparator()).base(), pathname.end());
}
static inline std::wstring removeExtension(std::wstring const& filename)
{
    return filename.substr(0, filename.find_last_of(L"."));
}
#endif

};};
