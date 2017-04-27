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
#include "CNTKTextFormatReader.h"
#include "HeapMemoryProvider.h"
#include "StringUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// TODO: Memory provider should be injected by SGD.

auto factory = [](const ConfigParameters& parameters) -> ReaderPtr
{
    return std::make_shared<CNTKTextFormatReader>(parameters);
};

extern "C" DATAREADER_API void GetReaderF(IDataReader** preader)
{
    *preader = new ReaderShim<float>(factory);
}

extern "C" DATAREADER_API void GetReaderD(IDataReader** preader)
{
    *preader = new ReaderShim<double>(factory);
}

// TODO: Not safe from the ABI perspective. Will be uglified to make the interface ABI.
// A factory method for creating text deserializers.
extern "C" DATAREADER_API bool CreateDeserializer(IDataDeserializer** deserializer, const std::wstring& type, const ConfigParameters& deserializerConfig, CorpusDescriptorPtr corpus, bool primary)
{
    string precision = deserializerConfig.Find("precision", "float");
    if (!AreEqualIgnoreCase(precision, "float") && !AreEqualIgnoreCase(precision, "double"))
    {
        InvalidArgument("Unsupported precision '%s'", precision.c_str());
    }

    // TODO: Remove type from the parser. Current implementation does not support streams of different types.
    if (type == L"CNTKTextFormatDeserializer")
    {
        if (precision == "float")
            *deserializer = new TextParser<float>(corpus, TextConfigHelper(deserializerConfig), primary);
        else // double
            *deserializer = new TextParser<double>(corpus, TextConfigHelper(deserializerConfig), primary);
    }
    else
        InvalidArgument("Unknown deserializer type '%ls'", type.c_str());

    // Deserializer created.
    return true;
}


}}}
