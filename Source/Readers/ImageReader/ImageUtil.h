//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <opencv2/opencv.hpp>
#include "Transformer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    inline bool IdentifyElementTypeFromOpenCVType(int openCvType, ElementType& type)
    {
        type = ElementType::tvariant;
        switch (openCvType)
        {
        case CV_64F:
            type = ElementType::tdouble;
            return true;
        case CV_32F:
            type = ElementType::tfloat;
            return true;
        case CV_8U:
            type = ElementType::tuchar;
            return true;
        default:
            return false;
        }
    }

    inline ElementType GetElementTypeFromOpenCVType(int openCvType)
    {
        ElementType result;
        if (!IdentifyElementTypeFromOpenCVType(openCvType, result))
            RuntimeError("Unsupported OpenCV type '%d'", openCvType);
        return result;
    }

}}}
