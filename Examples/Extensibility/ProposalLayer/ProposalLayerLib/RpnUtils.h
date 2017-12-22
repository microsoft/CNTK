//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include <cmath>
#include <vector>
#include <stdint.h>
#include "Rectangle2D.h"

// Generate anchor(reference) windows by enumerating aspect ratios X scales 
// wrt a reference(0, 0, 15, 15) window.
std::vector<Rectangle2D> GenerateAnchors(const std::vector<uint32_t>& scales, const std::vector<float>& ratios = { 0.5, 1, 2 }, float baseSize = 16.f) 
{
    Rectangle2D base(0.f, 0.f, baseSize-1, baseSize-1);
    
    auto area = base.Area();
    auto center = base.Center();

    std::vector<Rectangle2D> anchors;
    anchors.reserve(ratios.size() * scales.size());

    for (const auto& ratio : ratios)
    {
        auto areaRatio = area / ratio;
        auto w = std::round(sqrt(areaRatio));
        auto h = std::round(w * ratio);

        for (const auto& scale : scales)
        {
            auto scaledW = w * scale;
            auto scaledH = h * scale;
            anchors.emplace_back(scaledW, scaledH, center);
        }
    }

    return anchors;
}

// Deltas contain a tuple of 4 floats (dx, dy, dw, dh) for every box 
// in the boxes vector. Returns a vector of boxes transformed according 
// to the following rule:
// new_center_x = dx * widths + center_x
// new_w = exp(dw) * width
// new_xmin = new_center_x - 0.5 * new_w
std::vector<Rectangle2D> TransformBboxInv(const std::vector<Rectangle2D>& boxes, const float* deltas, const size_t stride)
{
    std::vector<Rectangle2D> predBoxes;
    predBoxes.reserve(boxes.size());

    size_t index = 0;
    for (const auto& box : boxes) 
    {
        auto offset = index++;

        if (index % stride == 0) 
            // each block consists of four strides, index points to the end of the first stride,
            // jump over the remaining three.
            index += 3 * stride;
            

        float dx = deltas[offset]; offset += stride;
        float dy = deltas[offset]; offset += stride;
        float dw = deltas[offset]; offset += stride;
        float dh = deltas[offset];
        auto center = box.Center();
        auto x = std::min(dx, 10.f) * box.Width() + center.x;
        auto y = std::min(dy, 10.f) * box.Height() + center.y;
        auto w = std::exp(std::min(dw, 10.f)) * box.Width();
        auto h = std::exp(std::min(dh, 10.f)) * box.Height();
        // The following seems incorrect, but it matches Caffe implementation.
        predBoxes.emplace_back(x - 0.5f * w, y - 0.5f * h, x + 0.5f * w, y + 0.5f * h);
    }

    return predBoxes;
}

// Clip boxes to image boundaries.
// imInfo - a tuple of 6 floats (pad_width, pad_height, scaled_image_width, scaled_image_height, orig_img_width, orig_img_height)
// e.g.(1000, 1000, 1000, 600, 500, 300) for an original image of 600x300 that is scaled and padded to 1000x1000
void ClipBoxes(std::vector<Rectangle2D>& boxes, const float* imInfo)
{
    auto i = 0;
    auto pad_width = imInfo[i++];
    auto pad_height = imInfo[i++];
    auto scaled_image_width = imInfo[i++];
    auto scaled_image_height = imInfo[i++];

    auto xmin = (pad_width - scaled_image_width) / 2;
    auto xmax = xmin + scaled_image_width - 1;

    auto ymin = (pad_height - scaled_image_height) / 2;
    auto ymax = ymin + scaled_image_height - 1;

    for (auto& box : boxes)
    {
        box.xmin = std::max(std::min(box.xmin, xmax), xmin);
        box.xmax = std::max(std::min(box.xmax, xmax), xmin);
        box.ymin = std::max(std::min(box.ymin, ymax), ymin);
        box.ymax = std::max(std::min(box.ymax, ymax), ymin);
    }
}