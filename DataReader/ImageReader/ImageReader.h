//
// <copyright file="UCIFastReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// ImageReader.h - Include file for the image reader 

#pragma once
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
    void SetSentenceSegBatch(Matrix<ElemType> &, vector<MinibatchPackingFlag>&) override { };

    void SetRandomSeed(unsigned int seed) override;

    //virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    //virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);
    //virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart = 0);
    //void SetSentenceSegBatch(Matrix<ElemType>&, Matrix<ElemType>&) { };
    //void SetNbrSlicesEachRecurrentIter(const size_t sz);

private:
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
};
}}}
