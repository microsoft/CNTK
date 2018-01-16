//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// RandomOrdering.h -- helper classes for randomizing timelines
//

#pragma once

#include "Basics.h"
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <random>

namespace Microsoft { namespace MSR { namespace CNTK {

// ---------------------------------------------------------------------------
// RandomOrdering -- class to help manage randomization of input data
// ---------------------------------------------------------------------------

static inline size_t rand(const size_t begin, const size_t end)
{
    const size_t randno = ::rand() * RAND_MAX + ::rand(); // BUGBUG: still only covers 32-bit range
    return begin + randno % (end - begin);
}

// Rand based on Mersenne Twister.
// We use our own distribution in order to match baselines between different operating systems,
// because uniform_distribution is not guaranteed to provide the same numbers on different platforms.
// TODO: Switching to Boost would eliminate this problem.
static inline size_t RandMT(const size_t begin, const size_t end, std::mt19937_64& rng)
{
    const size_t randomNumber = rng();
    return begin + randomNumber % (end - begin);
}

// Rand based on Mersenne Twister.
// We use our own distribution in order to match baselines between different operating systems,
// instead of using std::shuffle which uses unitform_distribution internally.
// TODO: Switching to Boost would eliminate this problem.
template <typename TVector>
inline void RandomShuffleMT(TVector& v, size_t begin, size_t end, std::mt19937_64& rng)
{
    for(size_t i = begin; i < end; ++i)
    {
        const size_t randomLocation = RandMT(i, end, rng);
        std::swap(v[i], v[randomLocation]);
    }
}

template <typename TVector>
inline void RandomShuffleMT(TVector& v, std::mt19937_64& rng)
{
    foreach_index(currentLocation, v)
    {
        // Pick a random location and swap with current one
        const size_t randomLocation = RandMT(0, v.size(), rng);
        std::swap(v[currentLocation], v[randomLocation]);
    }
}

class RandomOrdering // note: NOT thread-safe at all
{
    // constants for randomization
    const static size_t randomizeDisable = 0;

    typedef unsigned int INDEXTYPE; // don't use size_t, as this saves HUGE amounts of RAM
    std::vector<INDEXTYPE> map;     // [t] -> t' indices in randomized order
    size_t currentseed;             // seed for current sequence
    size_t randomizationrange;      // t - randomizationrange/2 <= t' < t + randomizationrange/2 (we support this to enable swapping)
                                    // special values (randomizeDisable)
    void Invalidate()
    {
        currentseed = (size_t) -1;
    }

public:
    RandomOrdering()
    {
        Invalidate();
        randomizationrange = randomizeDisable;
    }

    void Resize(size_t len, size_t p_randomizationrange)
    {
        randomizationrange = p_randomizationrange;
        if (len > 0)
            map.resize(len);
        Invalidate();
    }

    // return the randomized feature bounds for a time range
    std::pair<size_t, size_t> Bounds(size_t ts, size_t te) const
    {
        size_t tbegin = max(ts, randomizationrange / 2) - randomizationrange / 2;
        size_t tend = min(te + randomizationrange / 2, map.size());
        return std::make_pair<size_t, size_t>(std::move(tbegin), move(tend));
    }

    // this returns the map directly (read-only) and will lazily initialize it for a given seed
    const std::vector<INDEXTYPE>& operator()(size_t seed) // throw()
    {
        // if wrong seed then lazily recache the sequence
        if (seed != currentseed && randomizationrange != randomizeDisable)
        {
            // test for numeric overflow
            if (map.size() - 1 != (INDEXTYPE)(map.size() - 1))
                RuntimeError("RandomOrdering: INDEXTYPE has too few bits for this corpus");
            // 0, 1, 2...
            foreach_index (t, map)
                map[t] = (INDEXTYPE) t;

            if (map.size() > RAND_MAX * (size_t) RAND_MAX)
                RuntimeError("RandomOrdering: too large training set: need to change to different random generator!");
            srand((unsigned int) seed);
            size_t retries = 0;
            foreach_index (t, map)
            {
                for (int tries = 0; tries < 5; tries++)
                {
                    // swap current pos with a random position
                    // Random positions are limited to t+randomizationrange.
                    // This ensures some locality suitable for paging with a sliding window.
                    const size_t tbegin = max((size_t) t, randomizationrange / 2) - randomizationrange / 2; // range of window  --TODO: use bounds() function above
                    const size_t tend = min(t + randomizationrange / 2, map.size());
                    assert(tend >= tbegin);                  // (guard against potential numeric-wraparound bug)
                    const size_t trand = rand(tbegin, tend); // random number within windows
                    assert((size_t) t <= trand + randomizationrange / 2 && trand < (size_t) t + randomizationrange / 2);
                    // if range condition is fulfilled then swap
                    if (trand <= map[t] + randomizationrange / 2 && map[t] < trand + randomizationrange / 2 && (size_t) t <= map[trand] + randomizationrange / 2 && map[trand] < (size_t) t + randomizationrange / 2)
                    {
                        std::swap(map[t], map[trand]);
                        break;
                    }
                    // but don't multi-swap stuff out of its range (for swapping positions that have been swapped before)
                    // instead, try again with a different random number
                    retries++;
                }
            }
            fprintf(stderr, "RandomOrdering: %lu retries for %lu elements (%.1f%%) to ensure window condition\n", (unsigned long) retries, (unsigned long) map.size(), 100.0 * retries / map.size());
            // ensure the window condition
            foreach_index (t, map)
                assert((size_t) t <= map[t] + randomizationrange / 2 && map[t] < (size_t) t + randomizationrange / 2);
#if 0 // and a live check since I don't trust myself here yet
            foreach_index (t, map) if (!((size_t) t <= map[t] + randomizationrange/2 && map[t] < (size_t) t + randomizationrange/2))
            {
                fprintf (stderr, "RandomOrdering: windowing condition violated %d -> %d\n", t, map[t]);
                LogicError("RandomOrdering: windowing condition violated");
            }
#endif
#if 0 // test whether it is indeed a unique complete sequence
            auto map2 = map;
            ::sort (map2.begin(), map2.end());
            foreach_index (t, map2) assert (map2[t] == (size_t) t);
#endif
            fprintf(stderr, "RandomOrdering: recached sequence for seed %d: %d, %d, ...\n", (int) seed, (int) map[0], (int) map[1]);
            currentseed = seed;
        }
        return map; // caller can now access it through operator[]
    }

    size_t CurrentSeed() const
    {
        return currentseed;
    }

    bool IsRandomizationDisabled() const
    {
        return randomizationrange == randomizeDisable;
    }
};

}}}
