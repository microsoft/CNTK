//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <opencv2/opencv.hpp>
#include "ImageDeserializerBase.h"
#include "StringUtil.h"
#include "ConfigUtil.h"
#include "ImageTransformers.h"
#include "SequenceData.h"
#include "ImageUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    ImageDeserializerBase::ImageDeserializerBase() 
        : DataDeserializerBase(true),
          m_precision(ElementType::tfloat),
          m_grayscale(false), m_verbosity(0), m_multiViewCrop(false)
    {}

    ImageDeserializerBase::ImageDeserializerBase(CorpusDescriptorPtr corpus, const ConfigParameters& config, bool primary)
        : DataDeserializerBase(primary),
          m_corpus(corpus)
    {
        assert(m_corpus);

        ConfigParameters inputs = config("input");
        std::vector<std::string> featureNames = GetSectionsWithParameter("ImageDeserializerBase", inputs, "transforms");
        std::vector<std::string> labelNames = GetSectionsWithParameter("ImageDeserializerBase", inputs, "labelDim");

        if (featureNames.size() != 1 || labelNames.size() != 1)
            RuntimeError(
            "Please specify a single feature and label stream. '%d' features , '%d' labels found.",
            static_cast<int>(featureNames.size()),
            static_cast<int>(labelNames.size()));

        string precision = config("precision", "float");
        m_precision = AreEqualIgnoreCase(precision, "float") ? ElementType::tfloat : ElementType::tdouble;
        m_verbosity = config(L"verbosity", 0);

        // Feature stream.
        ConfigParameters featureSection = inputs(featureNames[0]);
        auto features = std::make_shared<StreamDescription>();
        features->m_id = 0;
        features->m_name = msra::strfun::utf16(featureSection.ConfigName());
        features->m_storageType = StorageType::dense;
        // Due to performance, now we support images of different types.
        features->m_elementType = ElementType::tvariant;
        m_streams.push_back(features);

        // Label stream.
        ConfigParameters label = inputs(labelNames[0]);
        size_t labelDimension = label("labelDim");
        auto labels = std::make_shared<StreamDescription>();
        labels->m_id = 1;
        labels->m_name = msra::strfun::utf16(label.ConfigName());
        labels->m_sampleLayout = std::make_shared<TensorShape>(labelDimension);
        labels->m_storageType = StorageType::sparse_csc;
        labels->m_elementType = m_precision;
        m_streams.push_back(labels);

        m_labelGenerator = labels->m_elementType == ElementType::tfloat ?
            (LabelGeneratorPtr)std::make_shared<TypedLabelGenerator<float>>(labelDimension) :
            std::make_shared<TypedLabelGenerator<double>>(labelDimension);

        m_grayscale = config(L"grayscale", false);

        // TODO: multiview should be done on the level of randomizer/transformers - it is responsiblity of the
        // TODO: randomizer to collect how many copies each transform needs and request same sequence several times.
        m_multiViewCrop = config(L"multiViewCrop", false);
    }

    void ImageDeserializerBase::PopulateSequenceData(
        cv::Mat image,
        size_t classId,
        size_t copyId,
        const KeyType& sequenceKey,
        std::vector<SequenceDataPtr>& result)
    {
        auto imageData = make_shared<ImageSequenceData>();
        if (!image.data)
        {
            imageData->m_isValid = false;
        }
        else
        {
            ElementType dataType = ConvertImageToSupportedDataType(image, m_precision);
            if (!image.isContinuous())
                image = image.clone();
            assert(image.isContinuous());

            ImageDimensions dimensions(image.cols, image.rows, image.channels());

            imageData->m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(HWC));
            imageData->m_copyIndex = static_cast<uint8_t>(copyId);
            imageData->m_image = image;
            imageData->m_numberOfSamples = 1;
            imageData->m_elementType = dataType;
            imageData->m_isValid = true;
            imageData->m_key = sequenceKey;
        }
        result.push_back(imageData);

        auto label = std::make_shared<CategorySequenceData>();
        m_labelGenerator->CreateLabelFor(classId, *label);
        label->m_numberOfSamples = 1;
        result.push_back(label);
    }
}}}