//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include <memory>
#ifdef _WIN32
#include <typeinfo.h>
#endif
#ifdef __unix__
#include <typeinfo>
#endif
#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Returns the size of the type.
inline size_t GetSizeByType(ElementType type)
{
    switch (type)
    {
    case ElementType::tfloat:
        return sizeof(float);
    case ElementType::tdouble:
        return sizeof(double);
    case ElementType::tatom:
        return sizeof(char);
    default:
        RuntimeError("Unsupported type '%s'", typeid(type).name());
    }
}
} } }
