//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#define DATAREADER_EXPORTS // creating the exports here
#include "DataReader.h"
#include "ImageReader.h"
#include "Config.h"
#include "ScriptableObjects.h"
#include "ConcStack.h"
#include <algorithm>
#include <fstream>
#include <sstream> // TODO: this should go away once we update the parameter parsing
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <omp.h>

namespace Microsoft { namespace MSR { namespace CNTK {

static bool AreEqual(const std::string& s1, const std::string& s2)
{
    return std::equal(s1.begin(), s1.end(), s2.begin(), [](const char& a, const char& b)
                      {
                          return std::tolower(a) == std::tolower(b);
                      });
};

//-------------------
// Transforms

class ITransform
{
public:
    virtual void Init(const ConfigParameters& config) = 0;
    //virtual void Init(const ScriptableObjects::IConfigRecord & config) = 0;
    virtual void Apply(cv::Mat& mat) = 0;

    ITransform(){};
    virtual ~ITransform(){};

public:
    ITransform(const ITransform&) = delete;
    ITransform& operator=(const ITransform&) = delete;
    ITransform(ITransform&&) = delete;
    ITransform& operator=(ITransform&&) = delete;
};

class CropTransform : public ITransform
{
public:
    CropTransform(unsigned int seed)
        : m_seed(seed)
    {
    }

    template <class ConfigRecordType>
    void InitFromConfig(const ConfigRecordType& config)
    {
        m_cropType = ParseCropType(config(L"cropType", ""));

#if 1
        floatargvector cropRatio = config(L"cropRatio", "1.0");
        m_cropRatioMin = cropRatio[0];
        m_cropRatioMax = cropRatio[1];
#else
        std::stringstream ss{config(L"cropRatio", "1")};
        std::string token{""};
        if (std::getline(ss, token, ':'))
        {
            m_cropRatioMin = std::stof(token);
            m_cropRatioMax = std::getline(ss, token, ':') ? std::stof(token) : m_cropRatioMin;
        }
#endif

        if (!(0 < m_cropRatioMin && m_cropRatioMin <= 1.0) ||
            !(0 < m_cropRatioMax && m_cropRatioMax <= 1.0) ||
            m_cropRatioMin > m_cropRatioMax)
        {
            RuntimeError("Invalid cropRatio value, must be > 0 and <= 1. cropMin must <= cropMax");
        }

        m_jitterType = ParseJitterType(config(L"jitterType", ""));

        if (!config.ExistsCurrent(L"hflip"))
            m_hFlip = m_cropType == CropType::Random;
        else
            m_hFlip = std::stoi(config(L"hflip")) != 0;
    }
    virtual void Init(const ConfigParameters& config) override
    {
        InitFromConfig(config);
    }
    //virtual void Init(const ScriptableObjects::IConfigRecord & config) override { InitFromConfig(config); }

    void Apply(cv::Mat& mat)
    {
        auto seed = m_seed;
        auto rng = m_rngs.pop_or_create([seed]()
                                        {
                                            return std::make_unique<std::mt19937>(seed);
                                        });

        double ratio = 1;
        switch (m_jitterType)
        {
        case RatioJitterType::None:
            ratio = m_cropRatioMin;
            break;
        case RatioJitterType::UniRatio:
            if (m_cropRatioMin == m_cropRatioMax)
                ratio = m_cropRatioMin;
            else
            {
                ratio = UniRealT(m_cropRatioMin, m_cropRatioMax)(*rng);
                assert(m_cropRatioMin <= ratio && ratio < m_cropRatioMax);
            }
            break;
        default:
            RuntimeError("Jitter type currently not implemented.");
        }
        mat = mat(GetCropRect(m_cropType, mat.rows, mat.cols, ratio, *rng));
        if (m_hFlip && std::bernoulli_distribution()(*rng))
            cv::flip(mat, mat, 1);

        m_rngs.push(std::move(rng));
    }

private:
    using UniRealT = std::uniform_real_distribution<double>;
    using UniIntT = std::uniform_int_distribution<int>;

    enum class CropType
    {
        Center = 0,
        Random = 1
    };
    enum class RatioJitterType
    {
        None = 0,
        UniRatio = 1,
        UniLength = 2,
        UniArea = 3
    };

    CropType ParseCropType(const std::string& src)
    {
        if (src.empty() || AreEqual(src, "center"))
            return CropType::Center;
        if (AreEqual(src, "random"))
            return CropType::Random;

        RuntimeError("Invalid crop type: %s.", src.c_str());
    }

    RatioJitterType ParseJitterType(const std::string& src)
    {
        if (src.empty() || AreEqual(src, "none"))
            return RatioJitterType::None;
        if (AreEqual(src, "uniratio"))
            return RatioJitterType::UniRatio;
        if (AreEqual(src, "unilength"))
            return RatioJitterType::UniLength;
        if (AreEqual(src, "uniarea"))
            return RatioJitterType::UniArea;

        RuntimeError("Invalid jitter type: %s.", src.c_str());
    }

    cv::Rect GetCropRect(CropType type, int crow, int ccol, double cropRatio, std::mt19937& rng)
    {
        assert(crow > 0);
        assert(ccol > 0);
        assert(0 < cropRatio && cropRatio <= 1.0);

        int cropSize = static_cast<int>(std::min(crow, ccol) * cropRatio);
        int xOff = -1;
        int yOff = -1;
        switch (type)
        {
        case CropType::Center:
            xOff = (ccol - cropSize) / 2;
            yOff = (crow - cropSize) / 2;
            break;
        case CropType::Random:
            xOff = UniIntT(0, ccol - cropSize)(rng);
            yOff = UniIntT(0, crow - cropSize)(rng);
            break;
        default:
            assert(false);
        }

        assert(0 <= xOff && xOff <= ccol - cropSize);
        assert(0 <= yOff && yOff <= crow - cropSize);
        return cv::Rect(xOff, yOff, cropSize, cropSize);
    }

private:
    unsigned int m_seed;
    conc_stack<std::unique_ptr<std::mt19937>> m_rngs;

    CropType m_cropType;
    double m_cropRatioMin;
    double m_cropRatioMax;
    RatioJitterType m_jitterType;
    bool m_hFlip;
};

class ScaleTransform : public ITransform
{
public:
    ScaleTransform(int dataType, unsigned int seed)
        : m_dataType(dataType), m_seed(seed)
    {
        assert(m_dataType == CV_32F || m_dataType == CV_64F);

        m_interpMap.emplace("nearest", cv::INTER_NEAREST);
        m_interpMap.emplace("linear", cv::INTER_LINEAR);
        m_interpMap.emplace("cubic", cv::INTER_CUBIC);
        m_interpMap.emplace("lanczos", cv::INTER_LANCZOS4);
    }

    template <class ConfigRecordType>
    void InitFromConfig(const ConfigRecordType& config)
    {
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
            std::transform(token.begin(), token.end(), token.begin(), (int (*) (int)) std::tolower);
            StrToIntMapT::const_iterator res = m_interpMap.find(token);
            if (res != m_interpMap.end())
                m_interp.push_back((*res).second);
        }

        if (m_interp.size() == 0)
            m_interp.push_back(cv::INTER_LINEAR);
    }
    virtual void Init(const ConfigParameters& config) override
    {
        InitFromConfig(config);
    }
    //virtual void Init(const ScriptableObjects::IConfigRecord & config) override { InitFromConfig(config); }

    void Apply(cv::Mat& mat)
    {
        // If matrix has not been converted to the right type, do it now as rescaling requires floating point type.
        if (mat.type() != CV_MAKETYPE(m_dataType, m_imgChannels))
            mat.convertTo(mat, m_dataType);

        auto seed = m_seed;
        auto rng = m_rngs.pop_or_create([seed]()
                                        {
                                            return std::make_unique<std::mt19937>(seed);
                                        });

        assert(m_interp.size() > 0);
        cv::resize(mat, mat, cv::Size(static_cast<int>(m_imgWidth), static_cast<int>(m_imgHeight)), 0, 0,
                   m_interp[UniIntT(0, static_cast<int>(m_interp.size()) - 1)(*rng)]);

        m_rngs.push(std::move(rng));
    }

private:
    using UniIntT = std::uniform_int_distribution<int>;

    unsigned int m_seed;
    conc_stack<std::unique_ptr<std::mt19937>> m_rngs;

    int m_dataType;

    using StrToIntMapT = std::unordered_map<std::string, int>;
    StrToIntMapT m_interpMap;
    std::vector<int> m_interp;

    size_t m_imgWidth;
    size_t m_imgHeight;
    size_t m_imgChannels;
};

class MeanTransform : public ITransform
{
public:
    MeanTransform()
    {
    }

    template <class ConfigRecordType>
    void InitFromConfig(const ConfigRecordType& config)
    {
        std::wstring meanFile = config(L"meanFile", L"");
        if (meanFile.empty())
            m_meanImg.release();
        else
        {
            cv::FileStorage fs;
            // REVIEW alexeyk: this sort of defeats the purpose of using wstring at all...  [fseide] no, only OpenCV has this problem.
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
            if (cchan * crow * ccol != m_meanImg.channels() * m_meanImg.rows * m_meanImg.cols)
                RuntimeError("Invalid data in file: %ls", meanFile.c_str());
            fs.release();
            m_meanImg = m_meanImg.reshape(cchan, crow);
        }
    }
    virtual void Init(const ConfigParameters& config) override
    {
        InitFromConfig(config);
    }
    //virtual void Init(const ScriptableObjects::IConfigRecord & config) override { InitFromConfig(config); }

    void Apply(cv::Mat& mat)
    {
        assert(m_meanImg.size() == cv::Size(0, 0) || (m_meanImg.size() == mat.size() && m_meanImg.channels() == mat.channels()));

        // REVIEW alexeyk: check type conversion (float/double).
        if (m_meanImg.size() == mat.size())
            mat = mat - m_meanImg;
    }

private:
    cv::Mat m_meanImg;
};

//-------------------
// ImageReader

template <class ElemType>
static void CopyFromImage(const cv::Mat& src, std::vector<ElemType>& dst, size_t ivDst, bool transpose);

static std::launch GetLaunchPolicy(bool prefetch)
{
    return prefetch ? std::launch::async : std::launch::deferred;
}

template <class ElemType>
ImageReader<ElemType>::ImageReader()
    : m_seed(0), m_rng(m_seed), m_imgListRand(true), m_pMBLayout(make_shared<MBLayout>()), m_mbFmt(DataFormat::NCHW)
{
    m_transforms.push_back(std::make_unique<CropTransform>(m_seed));
    m_transforms.push_back(std::make_unique<ScaleTransform>(sizeof(ElemType) == 4 ? CV_32F : CV_64F, m_seed));
    m_transforms.push_back(std::make_unique<MeanTransform>());
}

template <class ElemType>
ImageReader<ElemType>::~ImageReader()
{
}

template <class ElemType>
template <class ConfigRecordType>
void ImageReader<ElemType>::InitFromConfig(const ConfigRecordType& config)
{
    using SectionT = std::pair<std::string, ConfigParameters>; // TODO: does not work for BrainScript, since configs cannot be copied
    auto gettter = [&](const std::string& paramName) -> SectionT
    {
        auto sect = std::find_if(config.begin(), config.end(),
                                 [&](const std::pair<std::string, ConfigValue>& p)
                                 {
                                     return ConfigParameters(p.second).ExistsCurrent(paramName);
                                 });
        if (sect == config.end())
            RuntimeError("ImageReader requires %s parameter.", paramName.c_str());
        return {(*sect).first, ConfigParameters((*sect).second)};
    };

    // REVIEW alexeyk: currently support only one feature and label section.
    SectionT featSect{gettter("width")};
    m_featName = msra::strfun::utf16(featSect.first);
    // REVIEW alexeyk: w, h and c will be read again in ScaleTransform.
    size_t w = featSect.second("width");
    size_t h = featSect.second("height");
    size_t c = featSect.second("channels");
    m_featDim = w * h * c;

    // Get mini-batch format.
    std::string mbFmt = featSect.second("mbFormat", "nchw");
    if (AreEqual(mbFmt, "nhwc"))
        m_mbFmt = DataFormat::NHWC;
    else if (!AreEqual(mbFmt, "nchw"))
        RuntimeError("ImageReader does not support the mini-batch format %s.", mbFmt.c_str());

    // Initialize transforms.
    for (auto& t : m_transforms)
        t->Init(featSect.second);

    SectionT labSect{gettter("labelDim")};
    m_labName = msra::strfun::utf16(labSect.first);
    m_labDim = labSect.second("labelDim");

    std::string mapPath = config(L"file");
    std::ifstream mapFile(mapPath);
    if (!mapFile)
        RuntimeError("Could not open %s for reading.", mapPath.c_str());

    std::string line{""};
    for (size_t cline = 0; std::getline(mapFile, line); cline++)
    {
        std::stringstream ss{line};
        std::string imgPath;
        std::string clsId;
        if (!std::getline(ss, imgPath, '\t') || !std::getline(ss, clsId, '\t'))
            RuntimeError("Invalid map file format, must contain 2 tab-delimited columns: %s, line: %d.", mapPath.c_str(), static_cast<int>(cline));
        m_files.push_back({imgPath, std::stoi(clsId)});
    }

    std::string rand = config(L"randomize", "auto");
    if (AreEqual(rand, "none"))
        m_imgListRand = false;
    else if (!AreEqual(rand, "auto"))
        RuntimeError("Only Auto and None are currently supported.");

    m_prefetch = config(L"prefetch", true);

    int cthread = config(L"numCPUThreads", 0);
    if (cthread > 0)
        omp_set_num_threads(cthread);

    m_epochStart = 0;
    m_mbStart = 0;
}
//template<class ElemType> virtual void ImageReader<ElemType>::Init(const ConfigParameters & config);
//template<class ElemType> virtual void ImageReader<ElemType>::Init(const ScriptableObjects::IConfigRecord & config);

template <class ElemType>
void ImageReader<ElemType>::Destroy()
{
}

template <class ElemType>
void ImageReader<ElemType>::StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples)
{
    assert(mbSize > 0);
    assert(numSubsets > 0);
    assert(subsetNum < numSubsets);
    assert(requestedEpochSamples > 0);

    m_subsetNum = subsetNum;
    m_numSubsets = numSubsets;

    if (m_imgListRand)
        std::shuffle(m_files.begin(), m_files.end(), m_rng);

    m_epochSize = (requestedEpochSamples == requestDataSize ? m_files.size() : requestedEpochSamples);
    m_mbSize = mbSize;
    // REVIEW alexeyk: if user provides epoch size explicitly then we assume epoch size is a multiple of mbsize, is this ok?
    assert(requestedEpochSamples == requestDataSize || (m_epochSize % m_mbSize) == 0);
    m_epoch = epoch;
    m_epochStart = m_epoch * m_epochSize;
    if (m_epochStart >= m_files.size())
    {
        m_epochStart = 0;
        m_mbStart = 0;
    }

    m_featBuf.resize(m_mbSize * m_featDim);
    m_labBuf.resize(m_mbSize * m_labDim);
    m_mbPrefetchFut = std::async(GetLaunchPolicy(m_prefetch), [this]()
                                 {
                                     return ReadImages();
                                 });
}

template <class ElemType>
bool ImageReader<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
{
    assert(matrices.size() > 0);
    assert(matrices.find(m_featName) != matrices.end());
    assert(m_mbSize > 0);

    size_t mbSize = m_mbPrefetchFut.valid() ? m_mbPrefetchFut.get() : 0;

    if (mbSize == 0)
        return false;

    Matrix<ElemType>& features = *matrices[m_featName];
    features.SetValue(m_featDim, mbSize, features.GetDeviceId(), m_featBuf.data(), matrixFlagNormal);

    Matrix<ElemType>& labels = *matrices[m_labName];
    labels.SetValue(m_labDim, mbSize, labels.GetDeviceId(), m_labBuf.data(), matrixFlagNormal);

    m_pMBLayout->InitAsFrameMode(mbSize);

    // It is safe to run prefetching with just one buffer as SetValue is synchronous so there will be no race.
    m_mbPrefetchFut = std::async(GetLaunchPolicy(m_prefetch), [this]()
                                 {
                                     return ReadImages();
                                 });

    return true;
}

template <class ElemType>
bool ImageReader<ElemType>::DataEnd(EndDataType endDataType)
{
    bool ret = false;
    switch (endDataType)
    {
    case endDataNull:
        assert(false);
        break;
    case endDataEpoch:
        ret = m_mbStart < m_epochStart + m_epochSize;
        break;
    case endDataSet:
        ret = m_mbStart >= m_files.size();
        break;
    case endDataSentence:
        ret = true;
        break;
    }
    return ret;
}

template <class ElemType>
void ImageReader<ElemType>::SetRandomSeed(unsigned int seed)
{
    m_seed = seed;
    m_rng.seed(m_seed);
}

template <class ElemType>
size_t ImageReader<ElemType>::ReadImages()
{
    if (m_mbStart >= m_files.size() || m_mbStart >= m_epochStart + m_epochSize)
        return 0;

    size_t mbLim = m_mbStart + m_mbSize;
    if (mbLim > m_files.size())
        mbLim = m_files.size();

    std::fill(m_labBuf.begin(), m_labBuf.end(), static_cast<ElemType>(0));

    size_t actualMBSize = mbLim - m_mbStart;
    size_t iStart = actualMBSize * m_subsetNum / m_numSubsets;
    size_t iLim = actualMBSize * (m_subsetNum + 1) / m_numSubsets;
    size_t subsetSize = iLim - iStart;

#pragma omp parallel for ordered schedule(dynamic)
    for (long long i = 0; i < static_cast<long long>(subsetSize); i++)
    {
        const auto& p = m_files[m_mbStart + iStart + i];
        cv::Mat img{cv::imread(p.first, cv::IMREAD_COLOR)};
        if (!img.data)
            RuntimeError("Cannot read image file %s", p.first.c_str());
        for (auto& t : m_transforms)
            t->Apply(img);

        assert(img.rows * img.cols * img.channels() == m_featDim);
        // When IMREAD_COLOR is used, OpenCV stores image in BGR format.
        // Transpose is required if requested mini-batch format is NCHW.
        CopyFromImage(img, m_featBuf, m_featDim * i, m_mbFmt == DataFormat::NCHW);
        m_labBuf[m_labDim * i + p.second] = 1;
    }

    m_mbStart += actualMBSize;
    return subsetSize;
}

template class ImageReader<double>;
template class ImageReader<float>;

template <class ElemType>
static void CopyFromImage(const cv::Mat& src, std::vector<ElemType>& dst, size_t ivDst, bool transpose)
{
    assert(src.isContinuous());
    assert(src.channels() == 3);

    size_t count = src.rows * src.cols * src.channels();
    assert(ivDst + count <= dst.size());

    auto data = reinterpret_cast<const ElemType*>(src.ptr());
    if (!transpose)
        std::copy(data, data + count, dst.begin() + ivDst);
    else
    {
        size_t crow = src.rows * src.cols;
        size_t ccol = src.channels();
        for (size_t irow = 0; irow < crow; irow++)
        {
            for (size_t icol = 0; icol < ccol; icol++)
            {
                dst[ivDst + icol * crow + irow] = data[irow * ccol + icol];
            }
        }
    }
}
} } }
