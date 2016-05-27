//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <algorithm>
#include <unordered_map>
#include <random>
#include "ImageTransformers.h"
#include "Config.h"
#include "ConcStack.h"
#include "StringUtil.h"
#include "ElementTypeUtils.h"

namespace Microsoft { namespace MSR { namespace CNTK 
{

struct ImageSequenceData : DenseSequenceData
{
    cv::Mat m_image;
    // In case we do not copy data - we have to preserve the original sequence.
    SequenceDataPtr m_original;
};

ImageTransformerBase::ImageTransformerBase(const ConfigParameters& readerConfig) : m_imageElementType(0)
{
    m_seed = readerConfig(L"seed", 0u);
}

// The method describes how input stream is transformed to the output stream. Called once per applied stream.
// Currently for image transformations we only support dense streams of type double or float.
StreamDescription ImageTransformerBase::Transform(const StreamDescription& inputStream)
{
    m_inputStream = inputStream;
    m_outputStream = m_inputStream;

    if (m_inputStream.m_storageType != StorageType::dense)
    {
        LogicError("ImageTransformerBase supports only dense input streams.");
    }

    if (m_inputStream.m_elementType == ElementType::tdouble)
    {
        m_imageElementType = CV_64F;
    }
    else if (m_inputStream.m_elementType == ElementType::tfloat)
    {
        m_imageElementType = CV_32F;
    }
    else
    {
        RuntimeError("Unsupported type");
    }

    return m_outputStream;
}

// Transforms a single sequence as open cv dense image. Called once per sequence.
SequenceDataPtr ImageTransformerBase::Transform(SequenceDataPtr sequence)
{
    auto inputSequence = static_cast<const DenseSequenceData&>(*sequence);

    ImageDimensions dimensions(*inputSequence.m_sampleLayout, HWC);
    int columns = static_cast<int>(dimensions.m_width);
    int rows = static_cast<int>(dimensions.m_height);
    int channels = static_cast<int>(dimensions.m_numChannels);

    auto result = std::make_shared<ImageSequenceData>();
    int type = CV_MAKETYPE(m_imageElementType, channels);
    cv::Mat buffer = cv::Mat(rows, columns, type, inputSequence.m_data);
    Apply(sequence->m_id, buffer);
    if (!buffer.isContinuous())
    {
        buffer = buffer.clone();
    }
    else
    {
        result->m_original = sequence;
    }
    assert(buffer.isContinuous());
    result->m_image = buffer;
    result->m_data = buffer.ptr();
    result->m_numberOfSamples = inputSequence.m_numberOfSamples;

    ImageDimensions outputDimensions(buffer.cols, buffer.rows, buffer.channels());
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
    if ((m_hFlip && std::bernoulli_distribution()(*rng)) ||
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
        if (std::bernoulli_distribution()(rng))
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ScaleTransformer::ScaleTransformer(const ConfigParameters& config) : ImageTransformerBase(config)
{
    m_interpMap.emplace("nearest", cv::INTER_NEAREST);
    m_interpMap.emplace("linear", cv::INTER_LINEAR);
    m_interpMap.emplace("cubic", cv::INTER_CUBIC);
    m_interpMap.emplace("lanczos", cv::INTER_LANCZOS4);

    m_imgWidth = config(L"width");
    m_imgHeight = config(L"height");
    m_imgChannels = config(L"channels");

    size_t cfeat = m_imgWidth * m_imgHeight * m_imgChannels;
    if (cfeat == 0 || cfeat > std::numeric_limits<size_t>().max() / 2)
        RuntimeError("Invalid image dimensions.");

    m_interp.clear();
    std::stringstream ss{config(L"interpolations", "")};
    for (std::string token = ""; std::getline(ss, token, ':');)
    {
        // Explicit cast required for GCC.
        std::transform(token.begin(), token.end(), token.begin(),
                       (int (*) (int)) std::tolower);
        StrToIntMapT::const_iterator res = m_interpMap.find(token);
        if (res != m_interpMap.end())
            m_interp.push_back((*res).second);
    }

    if (m_interp.size() == 0)
        m_interp.push_back(cv::INTER_LINEAR);
}

// The method describes how input stream is transformed to the output stream. Called once per applied stream.
// Scale transformer transforms the stream so that all samples are of the same size.
StreamDescription ScaleTransformer::Transform(const StreamDescription& inputStream)
{
    ImageTransformerBase::Transform(inputStream);
    m_outputStream.m_sampleLayout = std::make_shared<TensorShape>(ImageDimensions(m_imgWidth, m_imgHeight, m_imgChannels).AsTensorShape(HWC));
    return m_outputStream;
}

void ScaleTransformer::Apply(size_t id, cv::Mat &mat)
{
    UNUSED(id);

    // If matrix has not been converted to the right type, do it now as rescaling
    // requires floating point type.
    if (mat.type() != CV_MAKETYPE(m_imageElementType, m_imgChannels))
    {
        mat.convertTo(mat, m_imageElementType);
    }

    auto seed = GetSeed();
    auto rng = m_rngs.pop_or_create([seed]() { return std::make_unique<std::mt19937>(seed); });

    auto index = UniIntT(0, static_cast<int>(m_interp.size()) - 1)(*rng);
    assert(m_interp.size() > 0);
    cv::resize(mat, mat, cv::Size((int)m_imgWidth, (int)m_imgHeight), 0, 0, m_interp[index]);

    m_rngs.push(std::move(rng));
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

    // REVIEW alexeyk: check type conversion (float/double).
    if (m_meanImg.size() == mat.size())
    {
        mat = mat - m_meanImg;
    }
}

// The method describes how input stream is transformed to the output stream. Called once per applied stream.
// Transpose transformer expects the dense input stream with samples as HWC and outputs CHW.
StreamDescription TransposeTransformer::Transform(const StreamDescription& inputStream)
{
    m_inputStream = inputStream;
    if (m_inputStream.m_storageType != StorageType::dense)
    {
        LogicError("Transpose transformer supports only dense streams.");
    }

    // Changing from NHWC to NCHW
    ImageDimensions dimensions(*m_inputStream.m_sampleLayout, HWC);
    m_outputStream = m_inputStream;
    m_outputStream.m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(CHW));
    return m_outputStream;
}

// Transformation of the sequence.
SequenceDataPtr TransposeTransformer::Transform(SequenceDataPtr sequence)
{
    if (m_inputStream.m_elementType == ElementType::tdouble)
    {
        return TypedTransform<double>(sequence);
    }

    if (m_inputStream.m_elementType == ElementType::tfloat)
    {
        return TypedTransform<float>(sequence);
    }

    RuntimeError("Unsupported type");
}

// The class represents a sequence that owns an internal data buffer.
// Passed from the TransposeTransformer.
// TODO: Transposition potentially could be done in place (alexeyk: performance might be much worse than of out-of-place transpose).
struct DenseSequenceWithBuffer : DenseSequenceData
{
    std::vector<char> m_buffer;
};

template <class TElemType>
SequenceDataPtr TransposeTransformer::TypedTransform(SequenceDataPtr sequence)
{
    auto inputSequence = static_cast<DenseSequenceData&>(*sequence);
    assert(inputSequence.m_numberOfSamples == 1);

    size_t count = m_inputStream.m_sampleLayout->GetNumElements() * GetSizeByType(m_inputStream.m_elementType);

    auto result = std::make_shared<DenseSequenceWithBuffer>();
    result->m_buffer.resize(count);

    ImageDimensions dimensions(*m_inputStream.m_sampleLayout, ImageLayoutKind::HWC);
    size_t rowCount = dimensions.m_height * dimensions.m_width;
    size_t channelCount = dimensions.m_numChannels;

    auto src = reinterpret_cast<TElemType*>(inputSequence.m_data);
    auto dst = reinterpret_cast<TElemType*>(result->m_buffer.data());

    for (size_t irow = 0; irow < rowCount; irow++)
    {
        for (size_t icol = 0; icol < channelCount; icol++)
        {
            dst[icol * rowCount + irow] = src[irow * channelCount + icol];
        }
    }

    result->m_sampleLayout = m_outputStream.m_sampleLayout;
    result->m_data = result->m_buffer.data();
    result->m_numberOfSamples = inputSequence.m_numberOfSamples;
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
    std::normal_distribution<float> d(0, (float)m_curStdDev);
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

}}}
