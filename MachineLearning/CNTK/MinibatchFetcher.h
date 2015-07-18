//
// <copyright file="MinibatchFetcher.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "ComputationNetwork.h"
#include "DataReader.h"
#include "TimerUtility.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// This base class represent the old, sequential way of fetching a single minibatch of input data.
// Essentially, it simply calls GetMinibatch on the reader.
template<class ElemType>
class MinibatchFetcher
{
public:
    MinibatchFetcher(IDataReader<ElemType>* trainSetDataReader,
                     std::map<std::wstring, Matrix<ElemType>*>* inputMatrices,
                     Matrix<ElemType>* sentenceBegin,
                     vector<MinibatchPackingFlag>* sentenceExistsBeginOrNoLabels) 
                     :
        m_reader(trainSetDataReader),
        m_inputMatrices(inputMatrices),
        m_sentenceBegin(sentenceBegin),
        m_sentenceExistsBeginOrNoLabels(sentenceExistsBeginOrNoLabels)
    {
        assert((m_sentenceBegin != nullptr) && (m_sentenceExistsBeginOrNoLabels != nullptr));
    }

    // This virtual dtor is necessary to allow invocation of derived dtors, which have some required synchronization points
    virtual ~MinibatchFetcher() {}

    virtual bool GetMinibatch()
    {
        bool retVal = m_reader->GetMinibatch(*m_inputMatrices);
        m_reader->SetSentenceSegBatch(*m_sentenceBegin, *m_sentenceExistsBeginOrNoLabels);

        return retVal;
    }

protected:
    IDataReader<ElemType>* m_reader;
    std::map<std::wstring, Matrix<ElemType>*>* m_inputMatrices;
    Matrix<ElemType>* m_sentenceBegin;
    vector<MinibatchPackingFlag>* m_sentenceExistsBeginOrNoLabels;
};

}}}