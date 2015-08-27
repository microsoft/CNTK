#include "stdafx.h"
#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "ImageReader.h"
#include "commandArgUtil.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <random>

#include <opencv2/opencv.hpp>

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
ImageReader<ElemType>::ImageReader() : m_seed(0)
{
}

template<class ElemType>
ImageReader<ElemType>::~ImageReader()
{
}

template<class ElemType>
void ImageReader<ElemType>::Init(const ConfigParameters& config)
{
    using SectionT = std::pair<std::string, ConfigParameters>;
    auto gettter = [&](const std::string& paramName) -> SectionT
    {
        auto sect = std::find_if(config.begin(), config.end(),
            [&](const std::pair<std::string, ConfigValue>& p) { return ConfigParameters(p.second).ExistsCurrent(paramName); });
        if (sect == config.end())
            RuntimeError("ImageReader requires " + paramName + " parameter.");
        return{ (*sect).first, ConfigParameters((*sect).second) };
    };

    // REVIEW alexeyk: currently support only one feature and label section.
    SectionT featSect{ gettter("width") };
    m_featName = msra::strfun::utf16(featSect.first);
    m_imgWidth = featSect.second("width");
    m_imgHeight = featSect.second("height");
    m_imgChannels = featSect.second("channels");
    m_featDim = m_imgWidth * m_imgHeight * m_imgChannels;

    SectionT labSect{ gettter("labelDim") };
    m_labName = msra::strfun::utf16(labSect.first);
    m_labDim = labSect.second("labelDim");

    std::string mapPath = config("file");
    std::ifstream mapFile(mapPath);
    if (!mapFile)
        RuntimeError("Could not open " + mapPath + " for reading.");
    
    std::string line{ "" };
    for (size_t cline = 0; std::getline(mapFile, line); cline++)
    {
        std::stringstream ss{ line };
        std::string imgPath;
        std::string clsId;
        if (!std::getline(ss, imgPath, '\t') || !std::getline(ss, clsId, '\t'))
            RuntimeError("Invalid map file format, must contain 2 tab-delimited columns: %s, line: %d.", mapPath.c_str(), cline);
        files.push_back({ imgPath, std::stoi(clsId) });
    }

    std::default_random_engine rng(m_seed);
    std::shuffle(files.begin(), files.end(), rng);

    m_epochStart = 0;
    m_mbStart = 0;
}

template<class ElemType>
void ImageReader<ElemType>::Destroy()
{
}

template<class ElemType>
void ImageReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize)
{
    assert(mbSize > 0);
    assert(requestedEpochSamples > 0);

    m_epochSize = (requestedEpochSamples == requestDataSize ? files.size() : requestedEpochSamples);
    m_mbSize = mbSize;
    // REVIEW alexeyk: if user provides epoch size explicitly then we assume epoch size is a multiple of mbsize, is this ok?
    assert(requestedEpochSamples == requestDataSize || (m_epochSize % m_mbSize) == 0);
    m_epoch = epoch;
    m_epochStart = m_epoch * m_epochSize;
    if (m_epochStart >= files.size())
    {
        m_epochStart = 0;
        m_mbStart = 0;
    }

    m_featBuf.resize(m_mbSize * m_featDim);
    m_labBuf.resize(m_mbSize * m_labDim);
}

template<class ElemType>
bool ImageReader<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
{
    assert(matrices.size() > 0);
    assert(matrices.find(m_featName) != matrices.end());
    assert(m_mbSize > 0);

    Matrix<ElemType>& features = *matrices[m_featName];
    Matrix<ElemType>& labels = *matrices[m_labName];

    if (m_mbStart >= files.size() || m_mbStart >= m_epochStart + m_epochSize)
        return false;

    size_t mbLim = m_mbStart + m_mbSize;
    if (mbLim > files.size())
        mbLim = files.size();

    std::fill(m_labBuf.begin(), m_labBuf.end(), static_cast<ElemType>(0));
    
#pragma omp parallel for ordered schedule(dynamic)
    for (long long i = 0; i < static_cast<long long>(mbLim - m_mbStart); i++)
    {
        const auto& p = files[i + m_mbStart];
        auto img = cv::imread(p.first, cv::IMREAD_COLOR);
        // Crop
        int w = img.cols;
        int h = img.rows;
        int cropSize = std::min(w, h);
        int xOff = (w - cropSize) / 2;
        int yOff = (h - cropSize) / 2;
        cv::Mat cropped{ img(cv::Rect(xOff, yOff, cropSize, cropSize)) };
        cropped.convertTo(img, CV_32F);
        
        // Scale
        cv::resize(img, img, cv::Size(static_cast<int>(m_imgWidth), static_cast<int>(m_imgHeight)), 0, 0, cv::INTER_LINEAR);

        assert(img.isContinuous());
        auto data = reinterpret_cast<ElemType*>(img.ptr());
        std::copy(data, data + m_featDim, m_featBuf.begin() + m_featDim * i);
        m_labBuf[m_labDim * i + p.second] = 1;
    }

    size_t mbSize = mbLim - m_mbStart;
    features.SetValue(m_featDim, mbSize, m_featBuf.data(), matrixFlagNormal);
    labels.SetValue(m_labDim, mbSize, m_labBuf.data(), matrixFlagNormal);

    m_mbStart = mbLim;
    return true;
}

template<class ElemType>
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
        ret = m_mbStart >= files.size();
        break;
    case endDataSentence:
        ret = true;
        break;
    }
    return ret;
}

template<class ElemType>
void ImageReader<ElemType>::SetRandomSeed(unsigned int seed)
{
    m_seed = seed;
}

template class ImageReader<double>;
template class ImageReader<float>;

}}}
