//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include <algorithm>
#include <stdexcept>

//
// An immutable data type to encapsulate a two-dimensional axis-aligned rectangle 
// with float-value coordinates. The rectangle is closed — it includes the points on the boundary.
//
// Ported from java version available at http://algs4.cs.princeton.edu/code/edu/princeton/cs/algs4/Rectangle2D.java.html
// 

#pragma pack(push, 1)

struct Point2D {
    float x, y;
};

// Immutable data type for 2D axis-aligned rectangle.
struct Rectangle2D {
    float xmin, ymin;   // minimum x- and y-coordinates
    float xmax, ymax;   // maximum x- and y-coordinates
                            
    Rectangle2D(float xmin, float ymin, float xmax, float ymax) 
        :xmin{ xmin }, ymin{ ymin }, xmax{ xmax }, ymax{ ymax }
    {
        if (xmax < xmin || ymax < ymin)
            throw std::invalid_argument("Invalid rectangle");
    }

    Rectangle2D(float w, float h, const Point2D& center)
        :xmin{ center.x - 0.5f * w }, 
        ymin{ center.y - 0.5f * h }, 
        xmax{ center.x + 0.5f * w - 1.f }, 
        ymax{ center.y + 0.5f * h - 1.f }
    {
        if (xmax < xmin || ymax < ymin)
            throw std::invalid_argument("Invalid rectangle");
    }

    float Width() const { return xmax - xmin + 1.f; }

    float Height() const { return ymax - ymin + 1.f; }

    float Area() const { return Width() * Height(); }
    
    bool Intersects(const Rectangle2D& that) const 
    {
        return xmax >= that.xmin && ymax >= that.ymin
            && that.xmax >= xmin && that.ymax >= ymin;
    }

    float Overlap(const Rectangle2D& that) const 
    {
        if (!Intersects(that))
            return 0;
        Rectangle2D overlap{ std::max(xmin, that.xmin), std::max(ymin, that.ymin), 
                        std::min(xmax, that.xmax), std::min(ymax, that.ymax) };
        return overlap.Area();
    }

    Point2D Center() const {
        return { 0.5f * (xmin + xmax + 1.f), 0.5f * (ymin + ymax + 1.f) };
    }

};

#pragma pack(pop)