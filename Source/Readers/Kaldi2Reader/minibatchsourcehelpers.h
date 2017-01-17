//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// minibatchsourcehelpers.h -- helper classes for minibatch sources
//

#pragma once

#include "basetypes.h"
#include <stdio.h>
#include <vector>
#include <algorithm>

#ifndef __unix__
#include "ssematrix.h" // for matrix type
#endif

namespace msra { namespace dbn {

// ---------------------------------------------------------------------------
// augmentneighbors() -- augmenting features with their neighbor frames
// ---------------------------------------------------------------------------

// implant a sub-vector into a vector, for use in augmentneighbors
template <class INV, class OUTV>
static void copytosubvector(const INV& inv, size_t subvecindex, OUTV& outv)
{
    size_t subdim = inv.size();
    assert(outv.size() % subdim == 0);
    size_t k0 = subvecindex * subdim;
    foreach_index (k, inv)
        outv[k + k0] = inv[k];
}

// compute the augmentation extent (how many frames added on each side)
static size_t augmentationextent(size_t featdim /*augment from*/, size_t modeldim /*to*/)
{
    const size_t windowframes = modeldim / featdim; // total number of frames to generate
    const size_t extent = windowframes / 2;         // extend each side by this

    if (modeldim % featdim != 0)
        throw runtime_error("augmentationextent: model vector size not multiple of input features");
    if (windowframes % 2 == 0)
        throw runtime_error(msra::strfun::strprintf("augmentationextent: neighbor expansion of input features to %d not symmetrical", windowframes));

    return extent;
}

// augment neighbor frames for a frame (correctly not expanding across utterance boundaries)
// The boundaryflags[] array, if not empty, flags first (-1) and last (+1) frame, i.e. frames to not expand across.
// The output 'v' must have te-ts columns.
template <class MATRIX, class VECTOR>
static void augmentneighbors(const MATRIX& frames, const std::vector<char>& boundaryflags, size_t t,
                             VECTOR& v)
{
    // how many frames are we adding on each side
    const size_t extent = augmentationextent(frames[t].size(), v.size());

    // copy the frame and its neighbors
    // Once we hit a boundaryflag in either direction, do not move index beyond.
    copytosubvector(frames[t], extent, v); // frame[t] sits right in the middle
    size_t t1 = t;                         // index for frames on to the left
    size_t t2 = t;                         // and right
    for (size_t n = 1; n <= extent; n++)
    {
#ifdef SAMPLING_EXPERIMENT
        if (boundaryflags.empty()) // boundary flags not given: 'frames' is full utterance
        {
            if (t1 >= SAMPLING_EXPERIMENT)
                t1 -= SAMPLING_EXPERIMENT; // index does not move beyond boundary
            if (t2 + SAMPLING_EXPERIMENT < frames.size())
                t2 += SAMPLING_EXPERIMENT;
        }
        else
        {
            if (boundaryflags[t1] != -1)
                t1 -= SAMPLING_EXPERIMENT; // index does not move beyond a set boundaryflag,
            if (boundaryflags[t2] != 1)
                t2 += SAMPLING_EXPERIMENT; // because that's the start/end of the utterance
        }
#else
        if (boundaryflags.empty()) // boundary flags not given: 'frames' is full utterance
        {
            if (t1 > 0)
                t1--; // index does not move beyond boundary
            if (t2 + 1 < frames.size())
                t2++;
        }
        else
        {
            if (boundaryflags[t1] != -1)
                t1--; // index does not move beyond a set boundaryflag,
            if (boundaryflags[t2] != 1)
                t2++; // because that's the start/end of the utterance
        }
#endif
        copytosubvector(frames[t1], extent - n, v);
        copytosubvector(frames[t2], extent + n, v);
    }
}

// augment neighbor frames for a frame (correctly not expanding across utterance boundaries)
// The boundaryflags[] array, if not empty, flags first (-1) and last (+1) frame, i.e. frames to not expand across.
// The output 'v' must have te-ts columns.
template <class MATRIX, class VECTOR>
static void augmentneighbors(const MATRIX& frames, const std::vector<char>& boundaryflags, size_t t, const size_t leftextent, const size_t rightextent,
                             VECTOR& v)
{

    // copy the frame and its neighbors
    // Once we hit a boundaryflag in either direction, do not move index beyond.
    copytosubvector(frames[t], leftextent, v); // frame[t] sits right in the middle
    size_t t1 = t;                             // index for frames on to the left
    size_t t2 = t;                             // and right

    for (size_t n = 1; n <= leftextent; n++)
    {
        if (boundaryflags.empty()) // boundary flags not given: 'frames' is full utterance
        {
            if (t1 > 0)
                t1--; // index does not move beyond boundary
        }
        else
        {
            if (boundaryflags[t1] != -1)
                t1--; // index does not move beyond a set boundaryflag,
        }
        copytosubvector(frames[t1], leftextent - n, v);
    }
    for (size_t n = 1; n <= rightextent; n++)
    {
        if (boundaryflags.empty()) // boundary flags not given: 'frames' is full utterance
        {
            if (t2 + 1 < frames.size())
                t2++;
        }
        else
        {
            if (boundaryflags[t2] != 1)
                t2++; // because that's the start/end of the utterance
        }
        copytosubvector(frames[t2], rightextent + n, v);
    }
}

// augment neighbor frames for one frame t in frames[] according to boundaryflags[]; result returned in column j of v
template <class INMATRIX, class OUTMATRIX>
static void augmentneighbors(const INMATRIX& frames, const std::vector<char>& boundaryflags, size_t t,
                             OUTMATRIX& v, size_t j)
{
    auto v_j = v.col(j); // the vector to fill in
    augmentneighbors(frames, boundaryflags, t, v_j);
}

// augment neighbor frames for one frame t in frames[] according to boundaryflags[]; result returned in column j of v
template <class INMATRIX, class OUTMATRIX>
static void augmentneighbors(const INMATRIX& frames, const std::vector<char>& boundaryflags, size_t t, size_t leftextent, size_t rightextent,
                             OUTMATRIX& v, size_t j)
{
    auto v_j = v.col(j); // the vector to fill in
    augmentneighbors(frames, boundaryflags, t, leftextent, rightextent, v_j);
}

// augment neighbor frames for a sequence of frames (part of an utterance, possibly spanning across boundaries)
template <class MATRIX>
static void augmentneighbors(const std::vector<std::vector<float>>& frames, const std::vector<char>& boundaryflags,
                             size_t ts, size_t te, // range [ts,te)
                             MATRIX& v)
{
    for (size_t t = ts; t < te; t++)
    {
        auto v_t = v.col(t - ts); // the vector to fill in
        augmentneighbors(frames, boundaryflags, t, v_t);
    }
}

// augment neighbor frames for a sequence of frames (part of an utterance, possibly spanning across boundaries)
template <class MATRIX>
static void augmentneighbors(const std::vector<std::vector<float>>& frames, const std::vector<char>& boundaryflags, size_t leftextent, size_t rightextent,
                             size_t ts, size_t te, // range [ts,te)
                             MATRIX& v)
{
    for (size_t t = ts; t < te; t++)
    {
        auto v_t = v.col(t - ts); // the vector to fill in
        augmentneighbors(frames, boundaryflags, t, leftextent, rightextent, v_t);
    }
}

//typedef unsigned short CLASSIDTYPE; // type to store state ids; don't use size_t --saves HUGE amounts of RAM
typedef unsigned int CLASSIDTYPE; // mseltzer - change to unsigned int for untied context-dependent phones
};
};
