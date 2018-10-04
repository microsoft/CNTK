//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"


namespace CNTK
{
    class ONNXFormat
    {
    public:
        static void Save(const FunctionPtr& src, const std::wstring& filepath);
        static FunctionPtr Load(const std::wstring& filepath, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
    private:
        static void InitializeLotusIR();
        static std::once_flag op_schema_initializer_flag_;
    };
}