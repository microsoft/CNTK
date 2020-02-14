//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// HTKMLFReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "basetypes.h"

#include "htkfeatio.h" // for reading HTK features

#include "ssematrix.h"

#define DATAWRITER_EXPORTS // creating the exports here
#include "DataWriter.h"
#include "Config.h"
#include "NbestWriter.h"
#include "Config.h"
#ifdef LEAKDETECT
#include <vld.h> // for memory leak detection
#endif

namespace Microsoft
{
namespace MSR
{
namespace CNTK
{

// Create a Data Writer
//DATAWRITER_API IDataWriter* DataWriterFactory(void)

template <class ElemType>
template <class ConfigRecordType>
void NbestWriter<ElemType>::InitFromConfig(const ConfigRecordType& writerConfig)
{
    wstring scriptpath;
    size_t numFiles;
    size_t firstfilesonly = SIZE_MAX; // set to a lower value for testing

    m_verbosity = writerConfig(L"verbosity", 2);

    ConfigParameters nbestWriterConfig = writerConfig(L"NbestWriter");

    if (nbestWriterConfig.Exists(L"scpFile"))
        scriptpath = nbestWriterConfig(L"scpFile");
    else
        RuntimeError("NbestWriter::Init: writer needs to specify scpFile for output");

    saveType = nbestWriterConfig(L"saveType", NBEST_SAVE_TXT);

    outputFiles.clear();

    fprintf(stderr, "NbestWriter::Init: reading output script file %S ...", scriptPath.c_str());
    size_t n = 0;
    for (msra::files::textreader reader(scriptPath); reader && outputFiles.size() <= firstfilesonly /*optimization*/;)
    {
        std::wstring line = reader.wgetline();
        wstringstream ss(line);
        std::wstring first_col;
        ss >> first_col;
        outputFiles.push_back(first_col);
        n++;
    }

    fprintf(stderr, " %zu entries\n", n);

    numFiles = n;

    outputFileIndex = 0;
}

template <class ElemType>
void NbestWriter<ElemType>::Destroy()
{
}

template <class ElemType>
void NbestWriter<ElemType>::GetSections(std::map<std::wstring, SectionType, nocase_compare>& /*sections*/)
{
}

template <class ElemType>
bool NbestWriter<ElemType>::SaveData(size_t /*recordStart*/, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t /*numRecords*/, size_t /*datasetSize*/, size_t /*byteVariableSized*/)
{
    // matrices should be size 1, matrices->second should be vector<pair<vector<size_t>, ElemType>>*, where pair is [ token ID sequence, log probability ]

    if (matrices.size() != 1)
        RuntimeError("NbestWriter: matrices.size() must equal 1");

    std::vector<std::pair<std::vector<size_t>, ElemType>>& nbest = *(static_cast<std::vector<std::pair<std::vector<size_t>, ElemType>>*>(iter->second));

    wstring outFile = outputFiles[outputFileIndex];

    switch (saveType)
    {
    case NBEST_SAVE_TXT:
        SaveTxt(outFile, nbest);
        break;
    case NBEST_SAVE_HTKLATTICE:
        RuntimeError("NbestWriter: saveType=1 for HTK lattice format not yet implemented");
        break;
    default:
        RuntimeError("NbestWriter: saveType must be 0 for text format or 1 for HTK lattice format");
    }

    outputFileIndex++;
    return true;
}

template <class ElemType>
void NbestWriter::SaveTxt(std::wstring& outputFile, std::vector<std::pair<std::vector<size_t>, ElemType>>& outputData)
{
    std::ofstream fileStream;
    fileStream.open(outputFile);
    for (size_t i = 0; i < outputData.size(); i++)
    {
        ElemType logP = outputData[i].second;
        vector<size_t>* labelseq = &(outputData[i].first);
        fileStream << logP;
        for (size_t j = 0; j < labelseq->size(); j++)
        {
            fileStream << " " << (*labelseq)[j];
        }
        fileStream << "\n";
    }
    fileStream.close();
}

template <class ElemType>
void NbestWriter<ElemType>::SaveMapping(std::wstring saveId, const std::map<LabelIdType, LabelType>& /*labelMapping*/)
{
}

template class NbestWriter<float>;
template class NbestWriter<double>;
} // namespace CNTK
} // namespace MSR
} // namespace Microsoft
