//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#ifdef _WIN32
#include <objbase.h>
#endif
#include "Basics.h"
#include <fstream>
#include <algorithm>

#define DATAWRITER_EXPORTS // creating the exports here
#include "DataWriter.h"
#include "LUSequenceWriter.h"
#include "Config.h"
#ifdef LEAKDETECT
#include <vld.h> // for memory leak detection
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

// Create a Data Writer

// comparison, not case sensitive.
template <class ElemType>
bool LUSequenceWriter<ElemType>::compare_val(const ElemType& first, const ElemType& second)
{
    return (first < second);
}

template <class ElemType>
template <class ConfigRecordType>
void LUSequenceWriter<ElemType>::InitFromConfig(const ConfigRecordType& writerConfig)
{
    udims.clear();

    vector<wstring> outputNames = writerConfig(L"outputNodeNames", ConfigRecordType::Array(stringargvector()));
    if (outputNames.size() < 1)
        RuntimeError("writer needs at least one outputNodeName specified in config");

    foreach_index (i, outputNames) // inputNames should map to node names
    {
        const ConfigRecordType& thisOutput = writerConfig(outputNames[i]);
        outputFiles[outputNames[i]] = (const wstring&) thisOutput(L"file");
        int iN = thisOutput(L"nbest", 1);
        nBests[outputNames[i]] = iN;
        wstring fname = thisOutput(L"token");
        ReadLabelInfo(fname, word4idx[outputNames[i]], idx4word[outputNames[i]]);
        size_t dim = idx4word[outputNames[i]].size();
        udims.push_back(dim);
    }
}

template <class ElemType>
void LUSequenceWriter<ElemType>::ReadLabelInfo(const wstring& vocfile,
                                               map<string, int>& word4idx,
                                               map<int, string>& idx4word)
{
    char stmp[MAX_STRING];
    int b;

    FILE* vin = fopenOrDie(vocfile, L"rt");

    if (vin == nullptr)
    {
        RuntimeError("cannot open word class file: %ls", vocfile.c_str());
    }
    b = 0;
    while (!feof(vin))
    {
        fscanf_s(vin, "%s\n", stmp, _countof(stmp));
        word4idx[stmp] = b;
        idx4word[b++] = stmp;
    }
    fclose(vin);
}

template <class ElemType>
void LUSequenceWriter<ElemType>::Destroy()
{
    for (auto ptr = outputFileIds.begin(); ptr != outputFileIds.end(); ptr++)
    {
        fclose(ptr->second);
    }
}

template <class ElemType>
bool LUSequenceWriter<ElemType>::SaveData(size_t /*recordStart*/, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t /*numRecords*/, size_t /*datasetSize*/, size_t /*byteVariableSized*/)
{

    for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
    {
        wstring outputName = iter->first;
        Matrix<ElemType>& outputData = *(static_cast<Matrix<ElemType>*>(iter->second));
        wstring outFile = outputFiles[outputName];

        Save(outFile, outputData, idx4word[outputName], nBests[outputName]);
    }

    return true;
}

template <class ElemType>
void LUSequenceWriter<ElemType>::Save(std::wstring& outputFile, const Matrix<ElemType>& outputData, const map<int, string>& idx2wrd, const int& nbest)
{
    size_t nT = outputData.GetNumCols();
    size_t nD = min(idx2wrd.size(), outputData.GetNumRows());
    FILE* fp = nullptr;
    vector<pair<size_t, ElemType>> lv;

    auto NbestComparator = [](const pair<size_t, ElemType>& lv, const pair<size_t, ElemType>& rv)
    {
        return lv.second > rv.second;
    };

    if (outputFileIds.find(outputFile) == outputFileIds.end())
    {
        FILE* ofs;
        msra::files::make_intermediate_dirs(outputFile);
        string str(outputFile.begin(), outputFile.end());
        ofs = fopen(str.c_str(), "wt");
        if (ofs == nullptr)
            RuntimeError("Cannot open %s for writing", str.c_str());
        outputFileIds[outputFile] = ofs;
        fp = ofs;
    }
    else
        fp = outputFileIds[outputFile];

    for (int j = 0; j < nT; j++)
    {
        int imax = 0;
        ElemType fmax = outputData(imax, j);
        lv.clear();
        if (nbest > 1)
            lv.push_back(pair<size_t, ElemType>(0, fmax));
        for (int i = 1; i < nD; i++)
        {
            if (nbest > 1)
                lv.push_back(pair<size_t, ElemType>(i, outputData(i, j)));
            if (outputData(i, j) > fmax)
            {
                fmax = outputData(i, j);
                imax = i;
            }
        }
        if (nbest > 1)
            sort(lv.begin(), lv.end(), NbestComparator);
        for (int i = 0; i < nbest; i++)
        {
            if (nbest > 1)
            {
                if (lv[i].second != 0)
                {
                    int idx = (int) lv[i].first;
                    string sRes = idx2wrd.find(idx)->second;
                    fprintf(fp, "%s ", sRes.c_str());
                }
            }
            else
            {
                string sRes = idx2wrd.find(imax)->second;
                fprintf(fp, "%s ", sRes.c_str());
                fprintf(stderr, "%s ", sRes.c_str());
            }
        }
    }
    fprintf(fp, "\n");
    fprintf(stderr, "\n");
}

template class LUSequenceWriter<float>;
template class LUSequenceWriter<double>;
} } }
