//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

//
// This is a helper header to resolve some external symbols if the project depends in a header-only way from CNTKLibrary.h
// Please define CNTK_HEADERONLY_DEFINITIONS macros if the project includes CNTKLibrary.h but does not link against CNTKv2Library so/dll.
// Unfortunately CNTKLibrary.h still depends on some external functions, they are defined in this header.
//
#pragma once

#include "Basics.h"

namespace CNTK {

    template <class E>
    __declspec_noreturn void ThrowFormatted(const char* format, ...)
    {
        va_list args;
        va_start(args, format);
        Microsoft::MSR::CNTK::ThrowFormattedVA<E>(format, args);
        va_end(args);
    }

    template __declspec_noreturn void ThrowFormatted<std::runtime_error>(const char* format, ...);
    template __declspec_noreturn void ThrowFormatted<std::logic_error>(const char* format, ...);
    template __declspec_noreturn void ThrowFormatted<std::invalid_argument>(const char* format, ...);

    namespace Internal
    {
        bool IsReversingTensorShapesInErrorMessagesEnabled()
        {
            LogicError("Should not be called, used only for external symbol "
                "resolution if CNTKLibrary.h header is included without linking to CNTKv2Library.");
        }
    }
}
