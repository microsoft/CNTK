//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include <vector>
#include "Rectangle2D.h"

// TODO: the current implementation is O(N^2), it should be possible to do this in O(N logN),
// see the sweeping line algorithm:  http://algs4.cs.princeton.edu/93intersection/
// Returns first topN elements from the input vector that do not overlap with any preceeding rectangle 
// by more than a threshold ratio.
std::vector<Rectangle2D> NonMaximumSupression(const std::vector<Rectangle2D>& in, float threshold, size_t topN)
{
    std::vector<Rectangle2D> out;
    out.reserve(topN);

    std::vector<bool> suppressed(in.size(), false);

    for (auto i = 0; i < in.size() ; i++)
    {
        const auto& box1 = in[i];
        
        if (suppressed[i])
            continue;

        out.push_back(box1);

        if (out.size() == topN)
           break;

        for (auto j = i+1; j < in.size(); j++)
        {
            if (suppressed[j])
                continue;

            const auto& box2 = in[j];

            float overlap = box1.Overlap(box2);
            float overlapRatio = overlap / (box1.Area() + box2.Area() - overlap);
            if (overlapRatio >= threshold) 
                suppressed[j] = true;
        }
    }

    return out;
}

