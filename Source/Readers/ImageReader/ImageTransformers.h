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

namespace CNTK {

// Sequence data that is used for images.
struct ImageSequenceData : DenseSequenceData
{
    cv::Mat m_image;

    uint8_t  m_copyIndex;            // Index of the copy. Used in i.e. Multicrop,
                                     // when deserializer provides several copies of the same sequence.

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

    const NDShape& GetSampleShape() override
    {
        return m_sampleShape;
    }

    NDShape m_sampleShape;
};

// Base class for image transformations based on OpenCV
// that helps to wrap the sequences into OpenCV::Mat class.
class ImageTransformerBase : public TransformBase
{
public:
    explicit ImageTransformerBase(const Microsoft::MSR::CNTK::ConfigParameters& config) : TransformBase(config)
    {};

    // Transformation of the sequence.
    SequenceDataPtr Transform(SequenceDataPtr sequence) override;

protected:
    using Base = Transformer;
    using UniRealT = boost::random::uniform_real_distribution<double>;
    using UniIntT = boost::random::uniform_int_distribution<int>;

    int ExpectedOpenCVPrecision() const
    {
        assert(m_precision == DataType::Float || m_precision == DataType::Double);
        return m_precision == DataType::Float ? CV_32F : CV_64F;
    }

    void ConvertToFloatingPointIfRequired(cv::Mat& image)
    {
        int depth = ExpectedOpenCVPrecision();
        if (image.depth() != depth)
            image.convertTo(image, depth);
    }

    // The only function that should be redefined by the inherited classes.
    virtual void Apply(uint8_t copyId, cv::Mat &from) = 0;

    Microsoft::MSR::CNTK::conc_stack<std::unique_ptr<std::mt19937>> m_rngs;
};

// Crop transformation of the image.
// Can work on images of any size.
class CropTransformer : public ImageTransformerBase
{
public:
    explicit CropTransformer(const Microsoft::MSR::CNTK::ConfigParameters& config);

    StreamInformation Transform(const StreamInformation& inputStream);

private:
    void Apply(uint8_t copyId, cv::Mat &mat) override;

private:
    enum class RatioJitterType
    {
        None = 0,
        UniRatio = 1
    };

    void StartEpoch(const EpochConfiguration &config) override;

    RatioJitterType ParseJitterType(const std::string &src);

    // assistent functions for GetCropRect****(). 
    double ApplyRatioJitter(const double minVal, const double maxVal, std::mt19937 &rng);

    cv::Rect GetCropRectCenter(int crow, int ccol, std::mt19937 &rng);
    cv::Rect GetCropRectRandomSide(int crow, int ccol, std::mt19937 &rng);
    cv::Rect GetCropRectRandomArea(int crow, int ccol, std::mt19937 &rng);
    cv::Rect GetCropRectMultiView10(int viewIndex, int crow, int ccol, std::mt19937 &rng);

    Microsoft::MSR::CNTK::conc_stack<std::unique_ptr<std::mt19937>> m_rngs;
    CropType m_cropType; 
    int m_cropWidth; 
    int m_cropHeight; 

    bool m_useSideRatio; 
    double m_sideRatioMin;
    double m_sideRatioMax;
    bool m_useAreaRatio; 
    double m_areaRatioMin;
    double m_areaRatioMax;
    double m_aspectRatioMin;
    double m_aspectRatioMax; 

    RatioJitterType m_jitterType;
    bool m_hFlip;
};

// Scale transformation of the image.
// Scales the image to the dimensions requested by the network.
class ScaleTransformer : public ImageTransformerBase
{
public:
    explicit ScaleTransformer(const Microsoft::MSR::CNTK::ConfigParameters& config);

    StreamInformation Transform(const StreamInformation& inputStream) override;

private:
    enum class ScaleMode
    {
        Fill = 0,
        Crop = 1,
        Pad  = 2
    };
    void Apply(uint8_t copyId, cv::Mat &mat) override;

    size_t m_imgWidth;
    size_t m_imgHeight;
    size_t m_imgChannels;

    ScaleMode m_scaleMode;
    int m_interp;
    int m_borderType;
    int m_padValue;
};

// Mean transformation.
class MeanTransformer : public ImageTransformerBase
{
public:
    explicit MeanTransformer(const Microsoft::MSR::CNTK::ConfigParameters& config);

private:
    void Apply(uint8_t copyId, cv::Mat &mat) override;

    cv::Mat m_meanImg;
};

// Transpose transformation from HWC to CHW (note: row-major notation).
class TransposeTransformer : public TransformBase
{
public:
    explicit TransposeTransformer(const Microsoft::MSR::CNTK::ConfigParameters&);

    // Transformation of the stream.
    StreamInformation Transform(const StreamInformation& inputStream) override;

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
        Microsoft::MSR::CNTK::conc_stack<std::vector<TElementTo>> m_memBuffers;
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
    explicit IntensityTransformer(const Microsoft::MSR::CNTK::ConfigParameters& config);

private:
    void StartEpoch(const EpochConfiguration &config) override;

    void Apply(uint8_t copyId, cv::Mat &mat) override;
    template <typename ElemType>
    void Apply(cv::Mat &mat);

    double m_stdDev;

    cv::Mat m_eigVal;
    cv::Mat m_eigVec;

    Microsoft::MSR::CNTK::conc_stack<std::unique_ptr<std::mt19937>> m_rngs;
};

// Color jittering transform based on the paper: http://arxiv.org/abs/1312.5402
// In short, the transform randomly changes contrast, brightness and color of the image.
class ColorTransformer : public ImageTransformerBase
{
public:
    explicit ColorTransformer(const Microsoft::MSR::CNTK::ConfigParameters& config);

private:
    void StartEpoch(const EpochConfiguration &config) override;

    void Apply(uint8_t copyId, cv::Mat &mat) override;
    template <typename ElemType>
    void Apply(cv::Mat &mat);

    double m_brightnessRadius;
    double m_contrastRadius;
    double m_saturationRadius;

    Microsoft::MSR::CNTK::conc_stack<std::unique_ptr<std::mt19937>> m_rngs;
    Microsoft::MSR::CNTK::conc_stack<std::unique_ptr<cv::Mat>> m_hsvTemp;
};

// Cast the input to a particular type.
// Images coming from the deserializer/transformers could come in different types,
// i.e. as a uchar due to performance reasons. On the other hand, the packer/network
// currently only supports float and double. This transform is necessary to do a proper
// casting before the sequence data enters the packer.
class CastTransformer : public TransformBase
{
public:
    explicit CastTransformer(const Microsoft::MSR::CNTK::ConfigParameters&);

    // Transformation of the stream.
    StreamInformation Transform(const StreamInformation& inputStream) override;

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
        Microsoft::MSR::CNTK::conc_stack<std::vector<TElementTo>> m_memBuffers;
    };

    TypedCast<float> m_floatTransform;
    TypedCast<double> m_doubleTransform;
};

}
