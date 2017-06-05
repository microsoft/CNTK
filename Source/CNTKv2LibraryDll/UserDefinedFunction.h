//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"

namespace CNTK
{
    class UDFUtils
    {
    public:

        static bool IsUDF(const FunctionPtr& f);

        static bool IsUDF(const Dictionary& dict);

        static Dictionary Serialize(const FunctionPtr& f);

        static FunctionPtr Deserialize(const Dictionary& dictionary,
            const std::unordered_map<std::wstring, Variable>& uidToVariableMap,
            const CNTK::DeviceDescriptor& device);

        static const size_t s_serializationVersion = 0;

    private:
        static bool IsNativeUDF(const Dictionary& dict);
    };
}
