//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <opencv2/opencv.hpp>
#include <numeric>
#include "ImageDataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// "private" helper functions
namespace {

std::unique_ptr<IDataDeserializer> createImageDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, const LabelType& labelType, const ElementType& elementType)
{
    std::unique_ptr<IDataDeserializer> deserializer;
    switch (labelType)
    {
    case LabelType::Classification:
        switch (elementType)
        {
        case ElementType::tfloat: deserializer = std::make_unique<ImageDataDeserializer<LabelType::Classification, float>>(corpus, config); break;
        case ElementType::tdouble: deserializer = std::make_unique<ImageDataDeserializer<LabelType::Classification, double>>(corpus, config); break;
        }
        break;
    case LabelType::Regression:
        switch (elementType)
        {
        case ElementType::tfloat: deserializer = std::make_unique<ImageDataDeserializer<LabelType::Regression, float>>(corpus, config); break;
        case ElementType::tdouble: deserializer = std::make_unique<ImageDataDeserializer<LabelType::Regression, double>>(corpus, config); break;
        }
        break;
    default:
        RuntimeError("Unknown ImageDataDeserializer configuration.");
    }
    return deserializer;
}

// (soon to be) deprecated version
std::unique_ptr<IDataDeserializer> createImageDataDeserializer(const ConfigParameters& config, const LabelType& labelType, const ElementType& elementType)
{
    std::unique_ptr<IDataDeserializer> deserializer;
    switch (labelType)
    {
    case LabelType::Classification:
        switch (elementType)
        {
        case ElementType::tfloat: deserializer = std::make_unique<ImageDataDeserializer<LabelType::Classification, float>>(config); break;
        case ElementType::tdouble: deserializer = std::make_unique<ImageDataDeserializer<LabelType::Classification, double>>(config); break;
        }
        break;
    case LabelType::Regression:
        switch (elementType)
        {
        case ElementType::tfloat: deserializer = std::make_unique<ImageDataDeserializer<LabelType::Regression, float>>(config); break;
        case ElementType::tdouble: deserializer = std::make_unique<ImageDataDeserializer<LabelType::Regression, double>>(config); break;
        }
        break;
    default:
        RuntimeError("Unknown ImageDataDeserializer configuration.");
    }
    return deserializer;
}

}

std::unique_ptr<IDataDeserializer> createImageDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config)
{
    const LabelType labelType = AreEqualIgnoreCase(std::string(config(L"labelType", "classification")), "classification") ? LabelType::Classification : LabelType::Regression;
    const ElementType elementType = AreEqualIgnoreCase(std::string(config(L"precision", "float")), "float") ? ElementType::tfloat : ElementType::tdouble;
    return createImageDataDeserializer(corpus, config, labelType, elementType);
}

// (soon to be) deprecated version
std::unique_ptr<IDataDeserializer> createImageDataDeserializer(const ConfigParameters& config)
{
    const LabelType labelType = AreEqualIgnoreCase(std::string(config(L"labelType", "classification")), "classification") ? LabelType::Classification : LabelType::Regression;
    const ElementType elementType = AreEqualIgnoreCase(std::string(config(L"precision", "float")), "float") ? ElementType::tfloat : ElementType::tdouble;
    return createImageDataDeserializer(config, labelType, elementType);
}

cv::Mat FileByteReader::Read(size_t, const std::string& path, bool grayscale)
{
    assert(!path.empty());

    return cv::imread(path, grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
}

// instantiate templates explicitely
template class ImageDataDeserializer<LabelType::Classification, float>;
template class ImageDataDeserializer<LabelType::Classification, double>;
template class ImageDataDeserializer<LabelType::Regression, float>;
template class ImageDataDeserializer<LabelType::Regression, double>;

}}}
