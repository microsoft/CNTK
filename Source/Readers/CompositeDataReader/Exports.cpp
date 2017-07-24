//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// DataReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Basics.h"

#define DATAREADER_EXPORTS
#include "DataReader.h"
#include "CompositeDataReader.h"
#include "ReaderShim.h"

namespace CNTK {

using namespace Microsoft::MSR::CNTK;

template<class ElemType>
class CompositeReaderShim : public ReaderShim<ElemType>
{
public:
    explicit CompositeReaderShim(ReaderFactory f) : ReaderShim<ElemType>(f) {}

    // Returning 0 for composite configs.
    // This forbids the use of learning-rate and momentum per MB if truncation is enabled.
    size_t GetNumParallelSequencesForFixingBPTTMode() override
    {
        return 0;
    }
};

auto factory = [](const ConfigParameters& parameters) -> ReaderPtr
{
    return std::make_shared<CompositeDataReader>(parameters);
};

extern "C" DATAREADER_API void GetReaderF(IDataReader** preader)
{
    *preader = new CompositeReaderShim<float>(factory);
}

extern "C" DATAREADER_API void GetReaderD(IDataReader** preader)
{
    *preader = new CompositeReaderShim<double>(factory);
}

extern "C" DATAREADER_API Reader* CreateCompositeDataReader(const ConfigParameters* parameters)
{
    return new CompositeDataReader(*parameters);
}

}
