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
    MinibatchFetcher(IDataReader<ElemType>* trainSetDataReader, const std::map<std::wstring, Matrix<ElemType>*>* inputMatrices) :
        m_reader(trainSetDataReader),
        m_inputMatrices(inputMatrices)
    {
    }

    // This virtual dtor is necessary to allow invocation of derived dtors, which have some required synchronization points
    virtual ~MinibatchFetcher() {}

    virtual bool GetMinibatch()
    {
        return m_reader->GetMinibatch(*const_cast<std::map<std::wstring, Matrix<ElemType>*>*>(m_inputMatrices));
    }

protected:
    IDataReader<ElemType>* m_reader;
    const std::map<std::wstring, Matrix<ElemType>*>* m_inputMatrices;
};

}}}