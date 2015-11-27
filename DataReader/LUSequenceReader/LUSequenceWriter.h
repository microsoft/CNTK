//
// <copyright file="LUSequenceWriter.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once
#include "DataWriter.h"
#include "LUSequenceParser.h"
#include <stdio.h>

#define MAX_STRING 2048

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
class LUSequenceWriter : public IDataWriter<ElemType>
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

    void SaveToFile(std::wstring& outputFile, const Matrix<ElemType>& outputData, const map<int, string>& idx2wrd, const int& nbest = 1);

    void ReadLabelInfo(const wstring & vocfile, 
            map<string, int> & word4idx,
            map<int, string>& idx4word);

public:
    ~LUSequenceWriter(){
        Destroy();
    }

public:
    void GetSections(std::map<std::wstring, SectionType, nocase_compare>& /*sections*/){}
    void SaveMapping(std::wstring saveId, const std::map<typename LabelIdType, typename LabelType>& /*labelMapping*/){}

public:
    template<class ConfigRecordType>
    void InitFromConfig(const ConfigRecordType & writerConfig);
    virtual void Init(const ConfigParameters & config) { InitFromConfig(config); }
    virtual void Init(const ScriptableObjects::IConfigRecord & config) { InitFromConfig(config); }
    virtual void Destroy();
    virtual bool SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized);
};

}}}