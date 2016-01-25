//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Exports.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS
#include "DataReader.h"
#include "ReaderShim.h"
#include "ImageReader.h"
#include "HeapMemoryProvider.h"
#include "CudaMemoryProvider.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// TODO: Memory provider should be injected by SGD.

auto factory = [](const ConfigParameters& parameters) -> ReaderPtr
{
    return std::make_shared<ImageReader>(std::make_shared<HeapMemoryProvider>(), parameters);
};

extern "C" DATAREADER_API void GetReaderF(IDataReader<float>** preader)
{
    *preader = new ReaderShim<float>(factory);
}

extern "C" DATAREADER_API void GetReaderD(IDataReader<double>** preader)
{
    *preader = new ReaderShim<double>(factory);
}
} } }
