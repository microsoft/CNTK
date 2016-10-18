//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <algorithm>
#include <unordered_map>
#include <random>
#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include "ImageTransformers.h"
#include "Config.h"
#include "ConcStack.h"
#include "StringUtil.h"
#include "SequenceData.h"
#include "ImageUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK 
{

// Transforms a single sequence as open cv dense image. Called once per sequence.
SequenceDataPtr ImageTransformerBase::Transform(SequenceDataPtr sequence)
{
    auto inputSequence = dynamic_cast<ImageSequenceData*>(sequence.get());
    if (inputSequence == nullptr)
        RuntimeError("Unexpected sequence provided");

    auto result = std::make_shared<ImageSequenceData>();
    Apply(sequence->m_id, inputSequence->m_image);

    result->m_image = inputSequence->m_image;
    result->m_numberOfSamples = inputSequence->m_numberOfSamples;
    result->m_elementType = GetElementTypeFromOpenCVType(inputSequence->m_image.depth());

    ImageDimensions outputDimensions(inputSequence->m_image.cols, inputSequence->m_image.rows, inputSequence->m_image.channels());
    result->m_sampleLayout = std::make_shared<TensorShape>(outputDimensions.AsTensorShape(HWC));
    return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CropTransformer::CropTransformer(const ConfigParameters& config) : ImageTransformerBase(config)
{
    floatargvector cropRatio = config(L"cropRatio", "1.0");
    m_cropRatioMin = cropRatio[0];
    m_cropRatioMax = cropRatio[1];

    if (!(0 < m_cropRatioMin && m_cropRatioMin <= 1.0) ||
        !(0 < m_cropRatioMax && m_cropRatioMax <= 1.0) ||
        m_cropRatioMin > m_cropRatioMax)
    {
        RuntimeError("Invalid cropRatio value, must be > 0 and <= 1. cropMin must "
                     "<= cropMax");
    }

    m_jitterType = ParseJitterType(config(L"jitterType", ""));

    m_cropType = ImageConfigHelper::ParseCropType(config(L"cropType", ""));

    if (!config.ExistsCurrent(L"hflip"))
    {
        m_hFlip = m_cropType == CropType::Random;
    }
    else
    {
        m_hFlip = config(L"hflip");
    }

    // for MultiView10 we need to set m_hflip = false, otherwise we might not get 5 unflipped image (see CropTransformer::Apply below)
    if (m_cropType == CropType::MultiView10)
    {
        m_hFlip = false;
    }

    m_aspectRatioRadius = config(L"aspectRatioRadius", ConfigParameters::Array(doubleargvector(vector<double>{0.0})));
}

void CropTransformer::StartEpoch(const EpochConfiguration &config)
{
    m_curAspectRatioRadius = m_aspectRatioRadius[config.m_epochIndex];
    if (!(0 <= m_curAspectRatioRadius && m_curAspectRatioRadius <= 1.0))
        InvalidArgument("aspectRatioRadius must be >= 0.0 and <= 1.0");
    ImageTransformerBase::StartEpoch(config);
}

void CropTransformer::Apply(size_t id, cv::Mat &mat)
{
    auto seed = GetSeed();
    auto rng = m_rngs.pop_or_create([seed]() { return std::make_unique<std::mt19937>(seed); });

    double ratio = 1;
    switch (m_jitterType)
    {
    case RatioJitterType::None:
        ratio = m_cropRatioMin;
        break;
    case RatioJitterType::UniRatio:
        if (m_cropRatioMin == m_cropRatioMax)
        {
            ratio = m_cropRatioMin;
        }
        else
        {
            ratio = UniRealT(m_cropRatioMin, m_cropRatioMax)(*rng);
            assert(m_cropRatioMin <= ratio && ratio < m_cropRatioMax);
        }
        break;
    default:
        RuntimeError("Jitter type currently not implemented.");
    }

    int viewIndex = m_cropType == CropType::MultiView10 ? (int)(id % 10) : 0;

    mat = mat(GetCropRect(m_cropType, viewIndex, mat.rows, mat.cols, ratio, *rng));
    // for MultiView10 m_hFlip is false, hence the first 5 will be unflipped, the later 5 will be flipped
    if ((m_hFlip && boost::random::bernoulli_distribution<>()(*rng)) ||
        viewIndex >= 5)
    {
        cv::flip(mat, mat, 1);
    }

    m_rngs.push(std::move(rng));
}

CropTransformer::RatioJitterType
CropTransformer::ParseJitterType(const std::string &src)
{
    if (src.empty() || AreEqualIgnoreCase(src, "none"))
    {
        return RatioJitterType::None;
    }

    if (AreEqualIgnoreCase(src, "uniratio"))
    {
        return RatioJitterType::UniRatio;
    }

    if (AreEqualIgnoreCase(src, "unilength"))
    {
        return RatioJitterType::UniLength;
    }

    if (AreEqualIgnoreCase(src, "uniarea"))
    {
        return RatioJitterType::UniArea;
    }

    RuntimeError("Invalid jitter type: %s.", src.c_str());
}

cv::Rect CropTransformer::GetCropRect(CropType type, int viewIndex, int crow, int ccol,
                                          double cropRatio, std::mt19937 &rng)
{
    assert(crow > 0);
    assert(ccol > 0);
    assert(0 < cropRatio && cropRatio <= 1.0);

    // Get square crop size that preserves aspect ratio.
    int cropSize = (int)(std::min(crow, ccol) * cropRatio);
    int cropSizeX = cropSize;
    int cropSizeY = cropSize;
    // Change aspect ratio, if this option is enabled.
    if (m_curAspectRatioRadius > 0)
    {
        double factor = 1.0 + UniRealT(-m_curAspectRatioRadius, m_curAspectRatioRadius)(rng);
        double area = cropSize * cropSize;
        double newArea = area * factor;
        if (boost::random::bernoulli_distribution<>()(rng))
        {
            cropSizeX = (int)std::sqrt(newArea);
            cropSizeY = (int)(area / cropSizeX);
        }
        else
        {
            cropSizeY = (int)std::sqrt(newArea);
            cropSizeX = (int)(area / cropSizeY);
        }
        // This clamping should be ok if jittering ratio is not too big.
        cropSizeX = std::min(cropSizeX, ccol);
        cropSizeY = std::min(cropSizeY, crow);
    }

    int xOff = -1;
    int yOff = -1;
    switch (type)
    {
    case CropType::Center:
        assert(viewIndex == 0);
        xOff = (ccol - cropSizeX) / 2;
        yOff = (crow - cropSizeY) / 2;
        break;
    case CropType::Random:
        assert(viewIndex == 0);
        xOff = UniIntT(0, ccol - cropSizeX)(rng);
        yOff = UniIntT(0, crow - cropSizeY)(rng);
        break;
    case CropType::MultiView10:
    {
        assert(0 <= viewIndex && viewIndex < 10);
        // 0 - 4: 4 corners + center crop. 5 - 9: same, but with a flip.
        int isubView = viewIndex % 5;
        switch (isubView)
        {
            // top-left
        case 0:
            xOff = 0;
            yOff = 0;
            break;
            // top-right
        case 1:
            xOff = ccol - cropSizeX;
            yOff = 0;
            break;
            // bottom-left
        case 2:
            xOff = 0;
            yOff = crow - cropSizeY;
            break;
            // bottom-right
        case 3:
            xOff = ccol - cropSizeX;
            yOff = crow - cropSizeY;
            break;
            // center
        case 4:
            xOff = (ccol - cropSizeX) / 2;
            yOff = (crow - cropSizeY) / 2;
            break;
        }
        break;
    }
    default:
        assert(false);
    }

    assert(0 <= xOff && xOff <= ccol - cropSizeX);
    assert(0 <= yOff && yOff <= crow - cropSizeY);
    return cv::Rect(xOff, yOff, cropSizeX, cropSizeY);
}

// scaleMode = "fill" (default) - warp the image to the given target size
// scaleMode = "crop" - resize the image's shorter side to the given target size and crops the overlap
// scaleMode = "pad"  - resize the image's larger side to the given target size, center it and pad the rest
ScaleTransformer::ScaleTransformer(const ConfigParameters& config) : ImageTransformerBase(config)
{
    m_imgWidth    = config(L"width");
    m_imgHeight   = config(L"height");
    m_imgChannels = config(L"channels");

    size_t cfeat = m_imgWidth * m_imgHeight * m_imgChannels;
    if (cfeat == 0 || cfeat > SIZE_MAX / 2)
        RuntimeError("Invalid image dimensions.");

    string scaleMode = config(L"scaleMode", "fill");
    if      (scaleMode == "crop") m_scaleMode = ScaleMode::Crop;
    else if (scaleMode == "pad")  m_scaleMode = ScaleMode::Pad;
    else if (scaleMode == "fill") m_scaleMode = ScaleMode::Fill;
    else RuntimeError("Invalid scaleMode value, must be fill, crop or pad (all lower case)");

    // the pad value used for the 'pad' mode. if set to -1 then the border will be replicated.
    m_padValue = config(L"padValue", -1);
    if (m_padValue >= 0)       m_borderType = cv::BORDER_CONSTANT;
    else if (m_padValue == -1) m_borderType = cv::BORDER_REPLICATE;
    else RuntimeError("Invalid padValue value, must be -1 (replicates border) or >= 0 (constant)");

    // for old config options use case-insensitve comparison ...
    string interpolation = config(L"interpolations", "linear");
    if (AreEqualIgnoreCase(interpolation, "nearest"))      m_interp = cv::INTER_NEAREST;
    else if (AreEqualIgnoreCase(interpolation, "cubic"))   m_interp = cv::INTER_CUBIC;
    else if (AreEqualIgnoreCase(interpolation, "lanczos")) m_interp = cv::INTER_LANCZOS4;
    else if (AreEqualIgnoreCase(interpolation, "linear"))  m_interp = cv::INTER_LINEAR;
    else RuntimeError("Invalid interpolations value, must be nearest, cubic, lanczos or linear");
}

// The method describes how input stream is transformed to the output stream. Called once per applied stream.
// Scale transformer transforms the stream so that all samples are of the same size.
StreamDescription ScaleTransformer::Transform(const StreamDescription& inputStream)
{
    TransformBase::Transform(inputStream);
    m_outputStream.m_sampleLayout = std::make_shared<TensorShape>(ImageDimensions(m_imgWidth, m_imgHeight, m_imgChannels).AsTensorShape(HWC));
    return m_outputStream;
}

void ScaleTransformer::Apply(size_t id, cv::Mat &mat)
{
    UNUSED(id);

    if (m_scaleMode == ScaleMode::Fill)
    { // warp the image to the given target size
        cv::resize(mat, mat, cv::Size((int)m_imgWidth, (int)m_imgHeight), 0, 0, m_interp);
    }
    else
    {
        int height = mat.rows;
        int width = mat.cols;

        // which dimension is our scaled one?
        bool scaleW;
        if (m_scaleMode == ScaleMode::Crop)
            scaleW = width < height; // in "crop" mode we resize the smaller side
        else
            scaleW = width > height; // else we resize the larger side

        size_t targetW, targetH;
        if (scaleW)
        {
            targetW = (size_t)m_imgWidth;
            targetH = (size_t)round(height * m_imgWidth / (double)width);
        }
        else
        {
            targetH = (size_t)m_imgHeight;
            targetW = (size_t)round(width * m_imgHeight / (double)height);
        }

        cv::resize(mat, mat, cv::Size((int)targetW, (int)targetH), 0, 0, m_interp);

        if (m_scaleMode == ScaleMode::Crop)
        { // crop the overlap
            size_t xOff = max((size_t)0, (targetW - m_imgWidth) / 2);
            size_t yOff = max((size_t)0, (targetH - m_imgHeight) / 2);
            mat = mat(cv::Rect((int)xOff, (int)yOff, (int)m_imgWidth, (int)m_imgHeight));
        }
        else
        { // ScaleMode::PAD --> center it and pad the rest
            size_t hdiff = max((size_t)0, (m_imgHeight - mat.rows) / 2);
            size_t wdiff = max((size_t)0, (m_imgWidth - mat.cols) / 2);

            size_t top = hdiff;
            size_t bottom = m_imgHeight - top - mat.rows;
            size_t left = wdiff;
            size_t right = m_imgWidth - left - mat.cols;
            cv::copyMakeBorder(mat, mat, (int)top, (int)bottom, (int)left, (int)right, m_borderType, m_padValue);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MeanTransformer::MeanTransformer(const ConfigParameters& config) : ImageTransformerBase(config)
{
    std::wstring meanFile = config(L"meanFile", L"");
    if (meanFile.empty())
        m_meanImg.release();
    else
    {
        cv::FileStorage fs;
        // REVIEW alexeyk: this sort of defeats the purpose of using wstring at
        // all...  [fseide] no, only OpenCV has this problem.
        fs.open(msra::strfun::utf8(meanFile).c_str(), cv::FileStorage::READ);
        if (!fs.isOpened())
            RuntimeError("Could not open file: %ls", meanFile.c_str());
        fs["MeanImg"] >> m_meanImg;
        int cchan;
        fs["Channel"] >> cchan;
        int crow;
        fs["Row"] >> crow;
        int ccol;
        fs["Col"] >> ccol;
        if (cchan * crow * ccol !=
            m_meanImg.channels() * m_meanImg.rows * m_meanImg.cols)
            RuntimeError("Invalid data in file: %ls", meanFile.c_str());
        fs.release();
        m_meanImg = m_meanImg.reshape(cchan, crow);
    }
}

void MeanTransformer::Apply(size_t id, cv::Mat &mat)
{
    UNUSED(id);
    assert(m_meanImg.size() == cv::Size(0, 0) ||
           (m_meanImg.size() == mat.size() &&
           m_meanImg.channels() == mat.channels()));

    if (m_meanImg.size() == mat.size())
    {
        // If matrix has not been converted to the right type, do it now as maen requires floating point type.
        ConvertToFloatingPointIfRequired(mat);
        mat = mat - m_meanImg;
    }
}

TransposeTransformer::TransposeTransformer(const ConfigParameters& config) : TransformBase(config),
    m_floatTransform(this), m_doubleTransform(this)
{}

// The method describes how input stream is transformed to the output stream. Called once per applied stream.
// Transpose transformer expects the dense input stream with samples as HWC and outputs CHW.
StreamDescription TransposeTransformer::Transform(const StreamDescription& inputStream)
{
    m_outputStream = TransformBase::Transform(inputStream);

    // Changing from NHWC to NCHW
    m_outputStream.m_elementType = m_precision;
    if (m_inputStream.m_sampleLayout != nullptr)
    {
        ImageDimensions dimensions(*m_inputStream.m_sampleLayout, HWC);
        m_outputStream.m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(CHW));
    }

    return m_outputStream;
}

// Transformation of the sequence.
SequenceDataPtr TransposeTransformer::Transform(SequenceDataPtr sequence)
{
    auto inputSequence = dynamic_cast<ImageSequenceData*>(sequence.get());
    if (inputSequence == nullptr)
        RuntimeError("Currently Transpose transform only works with images.");

    ElementType elementType = m_inputStream.m_elementType != ElementType::tvariant ?
        m_inputStream.m_elementType :
        sequence->m_elementType;

    switch (elementType)
    {
    case ElementType::tdouble:
        if (m_precision == ElementType::tfloat)
            return m_floatTransform.Apply<double>(inputSequence);
        if (m_precision == ElementType::tdouble)
            return m_doubleTransform.Apply<double>(inputSequence);
    case ElementType::tfloat:
        if (m_precision == ElementType::tdouble)
            return m_doubleTransform.Apply<float>(inputSequence);
        if (m_precision == ElementType::tfloat)
            return m_floatTransform.Apply<float>(inputSequence);
    case ElementType::tuchar:
        if (m_precision == ElementType::tdouble)
            return m_doubleTransform.Apply<unsigned char>(inputSequence);
        if (m_precision == ElementType::tfloat)
            return m_floatTransform.Apply<unsigned char>(inputSequence);
    default:
        RuntimeError("Unsupported type. Please apply a cast transform with 'double' or 'float' precision.");
    }
    return nullptr; // Make compiler happy
}

template <class TElementTo>
template<class TElementFrom>
SequenceDataPtr TransposeTransformer::TypedTranspose<TElementTo>::Apply(ImageSequenceData* inputSequence)
{
    TensorShapePtr shape = m_parent->m_inputStream.m_sampleLayout;
    if (shape == nullptr) // Taking the shape from the sequence.
        shape = inputSequence->m_sampleLayout;

    if (!shape)
        RuntimeError("Unknown shape of the sample in stream '%ls'.", m_parent->m_inputStream.m_name.c_str());

    assert(inputSequence->m_numberOfSamples == 1);

    size_t count = shape->GetNumElements();
    auto result = std::make_shared<DenseSequenceWithBuffer<TElementTo>>(m_memBuffers, count);

    ImageDimensions dimensions(*shape, ImageLayoutKind::HWC);
    size_t rowCount = dimensions.m_height * dimensions.m_width;
    size_t channelCount = dimensions.m_numChannels;

    auto dst = result->GetBuffer();

    if (channelCount == 3) // Unrolling for BGR, the most common case.
    {
        size_t nRows = inputSequence->m_image.rows;
        size_t nCols = inputSequence->m_image.cols;

        TElementTo* b = dst;
        TElementTo* g = dst + rowCount;
        TElementTo* r = dst + 2 * rowCount;

        for (size_t i = 0; i < nRows; ++i)
        {
            auto* x = inputSequence->m_image.ptr<TElementFrom>((int)i);
            for (size_t j = 0; j < nCols; ++j)
            {
                auto row = j * 3;
                *b++ = static_cast<TElementTo>(x[row]);
                *g++ = static_cast<TElementTo>(x[row + 1]);
                *r++ = static_cast<TElementTo>(x[row + 2]);
            }
        }
    }
    else
    {
        auto src = reinterpret_cast<const TElementFrom*>(inputSequence->GetDataBuffer());
        for (size_t irow = 0; irow < rowCount; irow++)
        {
            for (size_t icol = 0; icol < channelCount; icol++)
            {
                dst[icol * rowCount + irow] = static_cast<TElementTo>(src[irow * channelCount + icol]);
            }
        }
    }

    result->m_sampleLayout = m_parent->m_outputStream.m_sampleLayout != nullptr ?
        m_parent->m_outputStream.m_sampleLayout :
        std::make_shared<TensorShape>(dimensions.AsTensorShape(CHW));
    result->m_numberOfSamples = inputSequence->m_numberOfSamples;
    return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IntensityTransformer::IntensityTransformer(const ConfigParameters &config) : ImageTransformerBase(config)
{
    m_stdDev = config(L"intensityStdDev", ConfigParameters::Array(doubleargvector(vector<double>{0.0})));
    std::wstring intFile = config(L"intensityFile", L"");
    if (intFile.empty())
    {
        m_eigVal.release();
        m_eigVec.release();
    }
    else
    {
        cv::FileStorage fs;
        fs.open(msra::strfun::utf8(intFile).c_str(), cv::FileStorage::READ);
        if (!fs.isOpened())
            RuntimeError("Could not open file: %ls", intFile.c_str());
        fs["EigVal"] >> m_eigVal;
        if (m_eigVal.rows != 1 || m_eigVal.cols != 3 || m_eigVal.channels() != 1)
            RuntimeError("Invalid EigVal data in file: %ls", intFile.c_str());
        fs["EigVec"] >> m_eigVec;
        if (m_eigVec.rows != 3 || m_eigVec.cols != 3 || m_eigVec.channels() != 1)
            RuntimeError("Invalid EigVec data in file: %ls", intFile.c_str());
        fs.release();
    }
}

void IntensityTransformer::StartEpoch(const EpochConfiguration &config)
{
    m_curStdDev = m_stdDev[config.m_epochIndex];
    ImageTransformerBase::StartEpoch(config);
}

void IntensityTransformer::Apply(size_t id, cv::Mat &mat)
{
    UNUSED(id);

    if (m_eigVal.empty() || m_eigVec.empty() || m_curStdDev == 0)
        return;

    // Have to convert to float.
    int type = m_precision == ElementType::tfloat ? CV_32F : CV_64F;
    if (mat.type() != type)
        mat.convertTo(mat, type);

    if (mat.type() == CV_64FC(mat.channels()))
        Apply<double>(mat);
    else if (mat.type() == CV_32FC(mat.channels()))
        Apply<float>(mat);
    else
        RuntimeError("Unsupported type");
}

template <typename ElemType>
void IntensityTransformer::Apply(cv::Mat &mat)
{
    auto seed = GetSeed();
    auto rng = m_rngs.pop_or_create([seed]() { return std::make_unique<std::mt19937>(seed); } );

    // Using single precision as EigVal and EigVec matrices are single precision.
    boost::random::normal_distribution<float> d(0, (float)m_curStdDev);
    cv::Mat alphas(1, 3, CV_32FC1);
    assert(m_eigVal.rows == 1 && m_eigVec.cols == 3);
    alphas.at<float>(0) = d(*rng) * m_eigVal.at<float>(0);
    alphas.at<float>(1) = d(*rng) * m_eigVal.at<float>(1);
    alphas.at<float>(2) = d(*rng) * m_eigVal.at<float>(2);
    m_rngs.push(std::move(rng));

    assert(m_eigVec.rows == 3 && m_eigVec.cols == 3);

    cv::Mat shifts = m_eigVec * alphas.t();

    // For multi-channel images data is in BGR format.
    size_t cdst = mat.rows * mat.cols * mat.channels();
    ElemType* pdstBase = reinterpret_cast<ElemType*>(mat.data);
    for (ElemType* pdst = pdstBase; pdst < pdstBase + cdst;)
    {
        for (int c = 0; c < mat.channels(); c++)
        {
            float shift = shifts.at<float>(mat.channels() - c - 1);
            *pdst = std::min(std::max(*pdst + shift, (ElemType)0), (ElemType)255);
            pdst++;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ColorTransformer::ColorTransformer(const ConfigParameters &config) : ImageTransformerBase(config)
{
    m_brightnessRadius = config(L"brightnessRadius", ConfigParameters::Array(doubleargvector(vector<double>{0.0})));
    m_contrastRadius = config(L"contrastRadius", ConfigParameters::Array(doubleargvector(vector<double>{0.0})));
    m_saturationRadius = config(L"saturationRadius", ConfigParameters::Array(doubleargvector(vector<double>{0.0})));
}

void ColorTransformer::StartEpoch(const EpochConfiguration &config)
{
    m_curBrightnessRadius = m_brightnessRadius[config.m_epochIndex];
    if (!(0 <= m_curBrightnessRadius && m_curBrightnessRadius <= 1.0))
        InvalidArgument("brightnessRadius must be >= 0.0 and <= 1.0");

    m_curContrastRadius = m_contrastRadius[config.m_epochIndex];
    if (!(0 <= m_curContrastRadius && m_curContrastRadius <= 1.0))
        InvalidArgument("contrastRadius must be >= 0.0 and <= 1.0");

    m_curSaturationRadius = m_saturationRadius[config.m_epochIndex];
    if (!(0 <= m_curSaturationRadius && m_curSaturationRadius <= 1.0))
        InvalidArgument("saturationRadius must be >= 0.0 and <= 1.0");

    ImageTransformerBase::StartEpoch(config);
}

void ColorTransformer::Apply(size_t id, cv::Mat &mat)
{
    UNUSED(id);

    if (m_curBrightnessRadius == 0 && m_curContrastRadius == 0 && m_curSaturationRadius == 0)
        return;

    // Have to convert to float
    ConvertToFloatingPointIfRequired(mat);

    if (mat.type() == CV_64FC(mat.channels()))
        Apply<double>(mat);
    else if (mat.type() == CV_32FC(mat.channels()))
        Apply<float>(mat);
    else
        RuntimeError("Unsupported type");
}

template <typename ElemType>
void ColorTransformer::Apply(cv::Mat &mat)
{
    auto seed = GetSeed();
    auto rng = m_rngs.pop_or_create([seed]() { return std::make_unique<std::mt19937>(seed); });

    if (m_curBrightnessRadius > 0 || m_curContrastRadius > 0)
    {
        // To change brightness and/or contrast the following standard transformation is used:
        // Xij = alpha * Xij + beta, where
        // alpha is a contrast adjustment and beta - brightness adjustment.
        ElemType beta = 0;
        if (m_curBrightnessRadius > 0)
        {
            UniRealT d(-m_curBrightnessRadius, m_curBrightnessRadius);
            // Compute mean value of the image.
            cv::Scalar imgMean = cv::sum(cv::sum(mat));
            // Compute beta as a fraction of the mean.
            beta = (ElemType)(d(*rng) * imgMean[0] / (mat.rows * mat.cols * mat.channels()));
        }

        ElemType alpha = 1;
        if (m_curContrastRadius > 0)
        {
            UniRealT d(-m_curContrastRadius, m_curContrastRadius);
            alpha = (ElemType)(1 + d(*rng));
        }

        // Could potentially use mat.convertTo(mat, -1, alpha, beta) 
        // but it does not do range checking for single/double precision matrix. saturate_cast won't work either.
        size_t count = mat.rows * mat.cols * mat.channels();
        ElemType* pbase = reinterpret_cast<ElemType*>(mat.data);
        for (ElemType* p = pbase; p < pbase + count; p++)
        {
            *p = std::min(std::max(*p * alpha + beta, (ElemType)0), (ElemType)255);
        }
    }

    if (m_curSaturationRadius > 0 && mat.channels() == 3)
    {
        UniRealT d(-m_curSaturationRadius, m_curSaturationRadius);
        double ratio = 1.0 + d(*rng);
        assert(0 <= ratio && ratio <= 2);

        auto hsv = m_hsvTemp.pop_or_create([]() { return std::make_unique<cv::Mat>(); });

        // To change saturation, we need to convert the image to HSV format first,
        // the change S channgel and convert the image back to BGR format.
        cv::cvtColor(mat, *hsv, CV_BGR2HSV);
        assert(hsv->rows == mat.rows && hsv->cols == mat.cols);
        size_t count = hsv->rows * hsv->cols * mat.channels();
        ElemType* phsvBase = reinterpret_cast<ElemType*>(hsv->data);
        for (ElemType* phsv = phsvBase; phsv < phsvBase + count; phsv += 3)
        {
            const int HsvIndex = 1;
            phsv[HsvIndex] = std::min((ElemType)(phsv[HsvIndex] * ratio), (ElemType)1);
        }
        cv::cvtColor(*hsv, mat, CV_HSV2BGR);

        m_hsvTemp.push(std::move(hsv));
    }

    m_rngs.push(std::move(rng));
}

CastTransformer::CastTransformer(const ConfigParameters& config) : TransformBase(config), m_floatTransform(this), m_doubleTransform(this)
{
}

StreamDescription CastTransformer::Transform(const StreamDescription& inputStream)
{
    m_outputStream = TransformBase::Transform(inputStream);
    m_outputStream.m_elementType = m_precision;
    return m_outputStream;
}

SequenceDataPtr CastTransformer::Transform(SequenceDataPtr sequence)
{
    if (m_inputStream.m_elementType == m_precision || sequence->m_elementType == m_precision)
    {
        // No need to do anything, exit.
        return sequence;
    }

    SequenceDataPtr result;
    ElementType inputType = m_inputStream.m_elementType != ElementType::tvariant
        ? m_inputStream.m_elementType 
        : sequence->m_elementType;

    switch (m_precision)
    {
    case ElementType::tdouble:
        if (inputType == ElementType::tfloat)
            result = m_doubleTransform.Apply<float>(sequence);
        else if (inputType == ElementType::tuchar)
            result = m_doubleTransform.Apply<unsigned char>(sequence);
        else
            RuntimeError("Unsupported type. Please apply a cast transform with 'double' or 'float' precision.");
        break;
    case ElementType::tfloat:
        if (inputType == ElementType::tdouble)
            result = m_floatTransform.Apply<double>(sequence);
        if (inputType == ElementType::tuchar)
            result = m_floatTransform.Apply<unsigned char>(sequence);
        else
            RuntimeError("Unsupported type. Please apply a cast transform with 'double' or 'float' precision.");
        break;
    default:
        RuntimeError("Unsupported type. Please apply a cast transform with 'double' or 'float' precision.");
    }
    result->m_elementType = m_precision;
    return result;
}

template <class TElementTo>
template<class TElementFrom>
SequenceDataPtr CastTransformer::TypedCast<TElementTo>::Apply(SequenceDataPtr sequence)
{
    TensorShapePtr shape = m_parent->m_inputStream.m_sampleLayout;
    if (!shape) // Taking the shape from the sequence.
        shape = sequence->m_sampleLayout;

    if (!shape)
        RuntimeError("Unknown shape of the sample in stream '%ls'.", m_parent->m_inputStream.m_name.c_str());

    auto& inputSequence = static_cast<DenseSequenceData&>(*sequence);
    size_t count = shape->GetNumElements() * sequence->m_numberOfSamples;
    auto result = std::make_shared<DenseSequenceWithBuffer<TElementTo>>(m_memBuffers, count);

    auto src = reinterpret_cast<const TElementFrom*>(inputSequence.GetDataBuffer());
    auto dst = result->GetBuffer();

    for (size_t i = 0; i < count; i++)
    {
        dst[i] = static_cast<TElementTo>(src[i]);
    }

    result->m_sampleLayout = shape;
    result->m_numberOfSamples = inputSequence.m_numberOfSamples;
    return result;
}

}}}
