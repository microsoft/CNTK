//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <unordered_map>
#include <random>
#include <opencv2/opencv.hpp>

#include "Transformer.h"
#include "ConcStack.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class ConfigParameters;

// Base class for image transformations based on OpenCV
// that helps to wrap the sequences into OpenCV::Mat class.
class SlimImageTransformerBase : public SlimTransformer
{
public:
    explicit SlimImageTransformerBase(const ConfigParameters& config);

    // Transformation of the stream.
    StreamDescription Transform(const StreamDescription& inputStream) override;

    // Transformation of the sequence.
    SequenceDataPtr Transform(SequenceDataPtr sequence) override;

protected:
    // Seed  getter.
    unsigned int GetSeed() const
    {
        return m_seed;
    }

    using Base = SlimTransformer;
    using UniRealT = std::uniform_real_distribution<double>;
    using UniIntT = std::uniform_int_distribution<int>;

    // The only function that should be redefined by the inherited classes.
    virtual void Apply(size_t id, cv::Mat &from) = 0;

protected:
    StreamDescription m_inputStream;
    StreamDescription m_outputStream;

    unsigned int m_seed;
    int m_imageElementType;
    conc_stack<std::unique_ptr<std::mt19937>> m_rngs;
};

// Crop transformation of the image.
// Can work on images of any size.
class SlimCropTransformer : public SlimImageTransformerBase
{
public:
    explicit SlimCropTransformer(const ConfigParameters& config);

protected:
    virtual void Apply(size_t id, cv::Mat &mat) override;

private:
    enum class CropType
    {
        Center = 0,
        Random = 1,
        MultiView10 = 2
    };
    enum class RatioJitterType
    {
        None = 0,
        UniRatio = 1,
        UniLength = 2,
        UniArea = 3
    };

    CropType ParseCropType(const std::string &src);
    RatioJitterType ParseJitterType(const std::string &src);
    cv::Rect GetCropRect(CropType type, int viewIndex, int crow, int ccol, double cropRatio,
                         std::mt19937 &rng);

    CropType m_cropType;
    double m_cropRatioMin;
    double m_cropRatioMax;
    RatioJitterType m_jitterType;
    bool m_hFlip;
};

// Scale transformation of the image.
// Scales the image to the dimensions requested by the network.
class SlimScaleTransformer : public SlimImageTransformerBase
{
public:
    explicit SlimScaleTransformer(const ConfigParameters& config);

private:
    virtual void Apply(size_t id, cv::Mat &mat) override;

    using StrToIntMapT = std::unordered_map<std::string, int>;
    StrToIntMapT m_interpMap;
    std::vector<int> m_interp;

    size_t m_imgWidth;
    size_t m_imgHeight;
    size_t m_imgChannels;
};

// Mean transformation.
class SlimMeanTransformer : public SlimImageTransformerBase
{
public:
    explicit SlimMeanTransformer(const ConfigParameters& config);

private:
    virtual void Apply(size_t id, cv::Mat &mat) override;

    cv::Mat m_meanImg;
};

// Transpose transformation from HWC to CHW.
class SlimTransposeTransformer : public SlimTransformerPtr
{
public:
    explicit SlimTransposeTransformer(const ConfigParameters& config);

    // Transformation of the stream.
    StreamDescription Transform(const StreamDescription& inputStream);

    // Transformation of the sequence.
    SequenceDataPtr Transform(SequenceDataPtr sequence);


private:
    template <class TElement>
    SequenceDataPtr TypedApply(SequenceDataPtr inputSequence);

    StreamDescription m_inputStream;
    StreamDescription m_outputStream;
};

}}}
