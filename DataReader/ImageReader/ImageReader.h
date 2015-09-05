//
// <copyright file="UCIFastReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// ImageReader.h - Include file for the image reader 

#pragma once
#include <random>
#include <opencv2/opencv.hpp>
#include "DataReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
class ImageReader : public IDataReader<ElemType>
{
public:
    ImageReader();
    virtual ~ImageReader();

    ImageReader(const ImageReader&) = delete;
    ImageReader& operator=(const ImageReader&) = delete;
    ImageReader(ImageReader&&) = delete;
    ImageReader& operator=(ImageReader&&) = delete;

    void Init(const ConfigParameters& config) override;
    void Destroy() override;
    void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize) override;
    bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices) override;
    bool DataEnd(EndDataType endDataType) override;

    size_t NumberSlicesInEachRecurrentIter() override { return 0; }
    void SetSentenceSegBatch(Matrix<float>&, vector<MinibatchPackingFlag>&) override { };

    void SetRandomSeed(unsigned int seed) override;

private:
    enum class CropType { Center = 0, Random = 1 };

    CropType ParseCropType(const std::string& src);
    cv::Rect GetCropRect(CropType type, int crow, int ccol, float cropRatio);
    void CropTransform(const cv::Mat& src, cv::Mat& dst);

    void SubMeanTransform(const cv::Mat& src, cv::Mat& dst);

private:
    std::default_random_engine m_rng;
    std::uniform_int_distribution<int> m_rndUniInt;

    std::wstring m_featName;
    std::wstring m_labName;

    size_t m_imgWidth;
    size_t m_imgHeight;
    size_t m_imgChannels;
    size_t m_featDim;
    size_t m_labDim;

    using StrIntPairT = std::pair<std::string, int>;
    std::vector<StrIntPairT> files;

    size_t m_epochSize;
    size_t m_mbSize;
    size_t m_epoch;

    size_t m_epochStart;
    size_t m_mbStart;
    std::vector<ElemType> m_featBuf;
    std::vector<ElemType> m_labBuf;

    unsigned int m_seed;

    CropType m_cropType;
    float m_cropRatio;

    std::wstring m_meanFile;
};
}}}
