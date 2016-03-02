//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataWriter.h"
#include "LUSequenceParser.h"
#include <stdio.h>

#ifndef MAX_STRING
#define MAX_STRING 2048
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class LUSequenceWriter : public IDataWriter
{
private:
    std::vector<size_t> outputDims;
    map<wstring, wstring> outputFiles;
    map<wstring, FILE*> outputFileIds;

    std::vector<size_t> udims;
    map<wstring, map<string, int>> word4idx;
    map<wstring, map<int, string>> idx4word;

    map<wstring, int> nBests;
    bool compare_val(const ElemType& first, const ElemType& second);

    void Save(std::wstring& outputFile, const Matrix<ElemType>& outputData, const map<int, string>& idx2wrd, const int& nbest = 1);

    void ReadLabelInfo(const wstring& vocfile,
                       map<string, int>& word4idx,
                       map<int, string>& idx4word);

public:
    ~LUSequenceWriter()
    {
        Destroy();
    }

public:
    void GetSections(std::map<std::wstring, SectionType, nocase_compare>& /*sections*/)
    {
    }
    void SaveMapping(std::wstring saveId, const std::map<LabelIdType, LabelType>& /*labelMapping*/)
    {
    }

public:
    template <class ConfigRecordType>
    void InitFromConfig(const ConfigRecordType& writerConfig);
    virtual void Init(const ConfigParameters& config)
    {
        InitFromConfig(config);
    }
    virtual void Init(const ScriptableObjects::IConfigRecord& config)
    {
        InitFromConfig(config);
    }
    virtual void Destroy();
    virtual bool SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized);
    virtual bool SupportMultiUtterances() const
    {
        return false;
    };
};
} } }
