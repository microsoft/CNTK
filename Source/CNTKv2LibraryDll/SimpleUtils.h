//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include "CNTKLibrary.h"

namespace CNTK
{
    template <typename Container>
    inline std::wstring NamedListString(const Container& namedList)
    {
        std::wstringstream wss;
        bool first = true;
        for (auto namedObject : namedList)
        {
            if (!first)
                wss << L", ";

            wss << namedObject.AsString();
            first = false;
        }

        return wss.str();
    }

    inline Variable PlaceholderLike(const Variable& var)
    {
        return PlaceholderVariable(var.Shape(), var.GetDataType(), var.Name(), var.DynamicAxes());
    }
}
