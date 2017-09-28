//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <opencv2/opencv.hpp>
#include "SequenceData.h"
#include "DataDeserializer.h"
#include <numeric>

namespace CNTK {

    inline bool IdentifyDataTypeFromOpenCVType(int openCvType, DataType& type)
    {
        type = DataType::Unknown;
        switch (openCvType)
        {
        case CV_64F:
            type = DataType::Double;
            return true;
        case CV_32F:
            type = DataType::Float;
            return true;
        case CV_8U:
            type = DataType::UChar;
            return true;
        default:
            return false;
        }
    }

    inline DataType GetDataTypeFromOpenCVType(int openCvType)
    {
        DataType result;
        if (!IdentifyDataTypeFromOpenCVType(openCvType, result))
            RuntimeError("Unsupported OpenCV type '%d'", openCvType);
        return result;
    }

    inline DataType ConvertImageToSupportedDataType(cv::Mat& image, DataType defaultElementType)
    {
        DataType resultType;
        if (!IdentifyDataTypeFromOpenCVType(image.depth(), resultType))
        {
            // Could not identify element type.
            // Natively unsupported image type. Let's convert it to required precision.
            int requiredType = defaultElementType == DataType::Float ? CV_32F : CV_64F;
            image.convertTo(image, requiredType);
            resultType = defaultElementType;
        }
        return resultType;
    }

    // A helper interface to generate a typed label in a sparse format for categories.
    // It is represented as an array indexed by the category, containing zero values for all categories the sequence does not belong to,
    // and a single one for a category it belongs to: [ 0 .. 0.. 1 .. 0 ]
    class LabelGenerator
    {
    public:
        virtual void CreateLabelFor(size_t classId, CategorySequenceData& data) = 0;
        virtual size_t LabelDimension() const = 0;
        virtual ~LabelGenerator() { }
    };
    typedef std::shared_ptr<LabelGenerator> LabelGeneratorPtr;

    // Simple implementation of the LabelGenerator.
    // The class is parameterized because the representation of 1 is type specific.
    template <class TElement>
    class TypedLabelGenerator : public LabelGenerator
    {
    public:
        TypedLabelGenerator(size_t labelDimension) : m_value(1), m_indices(labelDimension)
        {
            if (labelDimension > numeric_limits<SparseIndexType>::max())
            {
                RuntimeError("Label dimension (%d) exceeds the maximum allowed "
                    "value (%d)\n", (int)labelDimension, (int)numeric_limits<SparseIndexType>::max());
            }
            iota(m_indices.begin(), m_indices.end(), 0);
        }

        void CreateLabelFor(size_t classId, CategorySequenceData& data) override
        {
            data.m_nnzCounts.resize(1);
            data.m_nnzCounts[0] = 1;
            data.m_totalNnzCount = 1;
            data.m_data = &m_value;
            data.m_indices = &(m_indices[classId]);
        }

        size_t LabelDimension() const override
        {
            return m_indices.size();
        }

    private:
        TElement m_value;
        std::vector<SparseIndexType> m_indices;
    };
}
