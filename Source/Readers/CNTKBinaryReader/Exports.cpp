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
#include "CNTKBinaryReader.h"
#include "V2Dependencies.h"
#include "BinaryChunkDeserializer.h"
#include "CorpusDescriptor.h"

namespace CNTK {

using namespace Microsoft::MSR::CNTK;

// TODO: Memory provider should be injected by SGD.

auto factory = [](const ConfigParameters& parameters) -> ReaderPtr
{
    return std::make_shared<CNTKBinaryReader>(parameters);
};

extern "C" DATAREADER_API void GetReaderF(IDataReader** preader)
{
    *preader = new ReaderShim<float>(factory);
}

extern "C" DATAREADER_API void GetReaderD(IDataReader** preader)
{
    *preader = new ReaderShim<double>(factory);
}

extern "C" DATAREADER_API bool CreateDeserializer(DataDeserializerPtr& deserializer, const std::wstring& type, const ConfigParameters& deserializerConfig, CorpusDescriptorPtr corpus, bool primary)
{
    if (corpus && !corpus->IsNumericSequenceKeys())
        InvalidArgument("Binary deserializer does not support non-numeric sequence keys.");

    if (!primary)
        // TODO: do we want to support non-primary binary deserializers?
        InvalidArgument("Binary deserializer can only be used as a primary.");
    
    if (type == L"CNTKBinaryFormatDeserializer")
    {
        deserializer = make_shared<BinaryChunkDeserializer>(BinaryConfigHelper(deserializerConfig));
    }
    else
        InvalidArgument("Unknown deserializer type '%ls'", type.c_str());

    // Deserializer created.
    return true;
}

}
