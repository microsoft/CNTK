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
#include "ImageDataDeserializer.h"
#include "ImageTransformers.h"
#include "CorpusDescriptor.h"
#include "Base64ImageDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// TODO: Memory provider should be injected by SGD.

auto factory = [](const ConfigParameters& parameters) -> ReaderPtr
{
    return std::make_shared<ImageReader>(parameters);
};

extern "C" DATAREADER_API void GetReaderF(IDataReader** preader)
{
    *preader = new ReaderShim<float>(factory);
}

extern "C" DATAREADER_API void GetReaderD(IDataReader** preader)
{
    *preader = new ReaderShim<double>(factory);
}

//TODO: Names of transforms and deserializers should be case insensitive.

// TODO: Not safe from the ABI perspective. Will be uglified to make the interface ABI.
// A factory method for creating image deserializers.
extern "C" DATAREADER_API bool CreateDeserializer(IDataDeserializer** deserializer, const std::wstring& type, const ConfigParameters& deserializerConfig, CorpusDescriptorPtr corpus, bool primary)
{
    if (type == L"ImageDeserializer")
        *deserializer = new ImageDataDeserializer(corpus, deserializerConfig, primary);
    else if (type == L"Base64ImageDeserializer")
        *deserializer = new Base64ImageDeserializer(corpus, deserializerConfig, primary);
    else
        // Unknown type.
        return false;

    // Deserializer created.
    return true;
}

// A factory method for creating image transformers.
extern "C" DATAREADER_API bool CreateTransformer(Transformer** transformer, const std::wstring& type, const ConfigParameters& config)
{
    if (type == L"Crop")
        *transformer = new CropTransformer(config);
    else if (type == L"Scale")
        *transformer = new ScaleTransformer(config);
    else if (type == L"Color")
        *transformer = new ColorTransformer(config);
    else if (type == L"Intensity")
        *transformer = new IntensityTransformer(config);
    else if (type == L"Mean")
        *transformer = new MeanTransformer(config);
    else if (type == L"Transpose")
        *transformer = new TransposeTransformer(config);
    else if (type == L"Cast")
        *transformer = new CastTransformer(config);
    else
        // Unknown type.
        return false;

    // Transformer created.
    return true;
}

}}}
