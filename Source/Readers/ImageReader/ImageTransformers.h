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
#include "Config.h"
#include "ImageConfigHelper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class ConfigParameters;

// Base class for image transformations based on OpenCV
// that helps to wrap the sequences into OpenCV::Mat class.
class ImageTransformerBase : public Transformer
{
public:
    explicit ImageTransformerBase(const ConfigParameters& config);

    void StartEpoch(const EpochConfiguration&) override {}

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

    using Base = Transformer;
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
class CropTransformer : public ImageTransformerBase
{
public:
    explicit CropTransformer(const ConfigParameters& config);

private:
    void Apply(size_t id, cv::Mat &mat) override;

private:
    enum class RatioJitterType
    {
        None = 0,
        UniRatio = 1,
        UniLength = 2,
        UniArea = 3
    };

    void StartEpoch(const EpochConfiguration &config) override;

    RatioJitterType ParseJitterType(const std::string &src);
    cv::Rect GetCropRect(CropType type, int viewIndex, int crow, int ccol, double cropRatio, std::mt19937 &rng);

    conc_stack<std::unique_ptr<std::mt19937>> m_rngs;
    CropType m_cropType;
    double m_cropRatioMin;
    double m_cropRatioMax;
    RatioJitterType m_jitterType;
    bool m_hFlip;
    doubleargvector m_aspectRatioRadius;
    double m_curAspectRatioRadius;
};

// Scale transformation of the image.
// Scales the image to the dimensions requested by the network.
class ScaleTransformer : public ImageTransformerBase
{
public:
    explicit ScaleTransformer(const ConfigParameters& config);

    StreamDescription Transform(const StreamDescription& inputStream) override;

private:
    void Apply(size_t id, cv::Mat &mat) override;

    using StrToIntMapT = std::unordered_map<std::string, int>;
    StrToIntMapT m_interpMap;
    std::vector<int> m_interp;

    conc_stack<std::unique_ptr<std::mt19937>> m_rngs;
    size_t m_imgWidth;
    size_t m_imgHeight;
    size_t m_imgChannels;
};

// Mean transformation.
class MeanTransformer : public ImageTransformerBase
{
public:
    explicit MeanTransformer(const ConfigParameters& config);

private:
    void Apply(size_t id, cv::Mat &mat) override;

    cv::Mat m_meanImg;
};

// Transpose transformation from HWC to CHW (note: row-major notation).
class TransposeTransformer : public Transformer
{
public:
    explicit TransposeTransformer(const ConfigParameters&) {}

    void StartEpoch(const EpochConfiguration&) override {}

    // Transformation of the stream.
    StreamDescription Transform(const StreamDescription& inputStream) override;

    // Transformation of the sequence.
    SequenceDataPtr Transform(SequenceDataPtr sequence) override;

private:
    template <class TElement>
    SequenceDataPtr TypedTransform(SequenceDataPtr inputSequence);

    StreamDescription m_inputStream;
    StreamDescription m_outputStream;
};

// Intensity jittering based on PCA transform as described in original AlexNet paper
// (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
// Currently uses precomputed values from 
// https://github.com/facebook/fb.resnet.torch/blob/master/datasets/imagenet.lua
// but should be replaced with per-class values?
class IntensityTransformer : public ImageTransformerBase
{
public:
    explicit IntensityTransformer(const ConfigParameters& config);

private:
    void StartEpoch(const EpochConfiguration &config) override;

    void Apply(size_t id, cv::Mat &mat) override;
    template <typename ElemType>
    void Apply(cv::Mat &mat);

    doubleargvector m_stdDev;
    double m_curStdDev;

    cv::Mat m_eigVal;
    cv::Mat m_eigVec;

    conc_stack<std::unique_ptr<std::mt19937>> m_rngs;
};

// Color jittering transform based on the paper: http://arxiv.org/abs/1312.5402
// In short, the transform randomly changes contrast, brightness and color of the image.
class ColorTransformer : public ImageTransformerBase
{
public:
    explicit ColorTransformer(const ConfigParameters& config);

private:
    void StartEpoch(const EpochConfiguration &config) override;

    void Apply(size_t id, cv::Mat &mat) override;
    template <typename ElemType>
    void Apply(cv::Mat &mat);

    doubleargvector m_brightnessRadius;
    double m_curBrightnessRadius;
    doubleargvector m_contrastRadius;
    double m_curContrastRadius;
    doubleargvector m_saturationRadius;
    double m_curSaturationRadius;

    conc_stack<std::unique_ptr<std::mt19937>> m_rngs;
    conc_stack<std::unique_ptr<cv::Mat>> m_hsvTemp;
};

}}}
