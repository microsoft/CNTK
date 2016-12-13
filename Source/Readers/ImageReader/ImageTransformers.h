//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <unordered_map>
#include <random>
#include <opencv2/opencv.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "Transformer.h"
#include "ConcStack.h"
#include "Config.h"
#include "ImageConfigHelper.h"
#include "TransformBase.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Sequence data that is used for images.
struct ImageSequenceData : DenseSequenceData
{
    cv::Mat m_image;

    const void* GetDataBuffer() override
    {
        if (!m_image.isContinuous())
        {
            // According to the contract, dense sequence data 
            // should return continuous data buffer.
            // TODO: This is potentially an expensive operation. Need to do some logging.
            m_image = m_image.clone();
        }

        return m_image.ptr();
    }
};

class ConfigParameters;

// Base class for image transformations based on OpenCV
// that helps to wrap the sequences into OpenCV::Mat class.
class ImageTransformerBase : public TransformBase
{
public:
    explicit ImageTransformerBase(const ConfigParameters& config) : TransformBase(config)
    {};

    // Transformation of the sequence.
    SequenceDataPtr Transform(SequenceDataPtr sequence) override;

protected:
    using Base = Transformer;
    using UniRealT = boost::random::uniform_real_distribution<double>;
    using UniIntT = boost::random::uniform_int_distribution<int>;

    int ExpectedOpenCVPrecision() const
    {
        assert(m_precision == ElementType::tfloat || m_precision == ElementType::tdouble);
        return m_precision == ElementType::tfloat ? CV_32F : CV_64F;
    }

    void ConvertToFloatingPointIfRequired(cv::Mat& image)
    {
        int depth = ExpectedOpenCVPrecision();
        if (image.depth() != depth)
            image.convertTo(image, depth);
    }

    // The only function that should be redefined by the inherited classes.
    virtual void Apply(size_t id, cv::Mat &from) = 0;

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
class TransposeTransformer : public TransformBase
{
public:
    explicit TransposeTransformer(const ConfigParameters&);

    // Transformation of the stream.
    StreamDescription Transform(const StreamDescription& inputStream) override;

    // Transformation of the sequence.
    SequenceDataPtr Transform(SequenceDataPtr sequence) override;

private:
    // A helper class transposes images using a set of typed memory buffers.
    template <class TElementTo>
    struct TypedTranspose
    {
        TransposeTransformer* m_parent;

        TypedTranspose(TransposeTransformer* parent) : m_parent(parent) {}

        template <class TElementFrom>
        SequenceDataPtr Apply(ImageSequenceData* inputSequence);
        conc_stack<std::vector<TElementTo>> m_memBuffers;
    };

    // Auxiliary buffer to handle images of float type.
    TypedTranspose<float> m_floatTransform;

    // Auxiliary buffer to handle images of double type.
    TypedTranspose<double> m_doubleTransform;
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

// Cast the input to a particular type.
// Images coming from the deserializer/transformers could come in different types,
// i.e. as a uchar due to performance reasons. On the other hand, the packer/network
// currently only supports float and double. This transform is necessary to do a proper
// casting before the sequence data enters the packer.
class CastTransformer : public TransformBase
{
public:
    explicit CastTransformer(const ConfigParameters&);

    // Transformation of the stream.
    StreamDescription Transform(const StreamDescription& inputStream) override;

    // Transformation of the sequence.
    SequenceDataPtr Transform(SequenceDataPtr sequence) override;

private:

    // A helper class casts images using a set of typed memory buffers.
    template <class TElementTo>
    struct TypedCast
    {
        CastTransformer* m_parent;

        TypedCast(CastTransformer* parent) : m_parent(parent) {}

        template <class TElementFrom>
        SequenceDataPtr Apply(SequenceDataPtr inputSequence);
        conc_stack<std::vector<TElementTo>> m_memBuffers;
    };

    TypedCast<float> m_floatTransform;
    TypedCast<double> m_doubleTransform;
};


}}}
