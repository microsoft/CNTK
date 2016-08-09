//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// HTKMLFReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#ifdef _WIN32
#include <objbase.h>
#endif
#include "Basics.h"

#include "htkfeatio.h" // for reading HTK features
#include "ssematrix.h"

#define DATAWRITER_EXPORTS // creating the exports here
#include "DataWriter.h"
#include "Config.h"
#include "HTKMLFWriter.h"
#ifdef LEAKDETECT
#include <vld.h> // for memory leak detection
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

// Create a Data Writer
//DATAWRITER_API IDataWriter* DataWriterFactory(void)

template <class ElemType>
template <class ConfigRecordType>
void HTKMLFWriter<ElemType>::InitFromConfig(const ConfigRecordType& writerConfig)
{
    m_tempArray = nullptr;
    m_tempArraySize = 0;

    vector<wstring> scriptpaths;
    vector<wstring> filelist;
    size_t numFiles;
    size_t firstfilesonly = SIZE_MAX; // set to a lower value for testing

    vector<wstring> outputNames = writerConfig(L"outputNodeNames", ConfigRecordType::Array(stringargvector()));
    if (outputNames.size() < 1)
        RuntimeError("writer needs at least one outputNodeName specified in config");

    foreach_index (i, outputNames) // inputNames should map to node names
    {
        ConfigParameters thisOutput = writerConfig(outputNames[i]);
        if (thisOutput.Exists("dim"))
            m_udims.push_back(thisOutput(L"dim"));
        else
            RuntimeError("HTKMLFWriter::Init: writer need to specify dim of output");

        if (thisOutput.Exists("file"))
            scriptpaths.push_back(thisOutput(L"file"));
        else if (thisOutput.Exists("scpFile"))
            scriptpaths.push_back(thisOutput(L"scpFile"));
        else
            RuntimeError("HTKMLFWriter::Init: writer needs to specify scpFile for output");

        m_outputNameToIdMap[outputNames[i]] = i;
        m_outputNameToDimMap[outputNames[i]] = m_udims[i];
        m_outputNameToTotalSamples[outputNames[i]] = 0;
        m_outputNameToValues[outputNames[i]] = vector<float>();

        wstring type = thisOutput(L"type", "Real");
        if (type == L"Real")
        {
            m_outputNameToTypeMap[outputNames[i]] = OutputTypes::outputReal;
        }
        else
        {
            RuntimeError("HTKMLFWriter::Init: output type for writer output expected to be Real");
        }
    }

    numFiles = 0;
    foreach_index (i, scriptpaths)
    {
        filelist.clear();
        std::wstring scriptPath = scriptpaths[i];
        fprintf(stderr, "HTKMLFWriter::Init: reading output script file %ls ...", scriptPath.c_str());
        size_t n = 0;
        for (msra::files::textreader reader(scriptPath); reader && filelist.size() <= firstfilesonly /*optimization*/;)
        {
            filelist.push_back(reader.wgetline());
            n++;
        }

        fprintf(stderr, " %d entries\n", (int) n);

        if (i == 0)
            numFiles = n;
        else if (n != numFiles)
            RuntimeError("HTKMLFWriter:Init: number of files in each scriptfile inconsistent (%d vs. %d)", (int) numFiles, (int) n);

        m_outputFiles.push_back(filelist);
    }
    outputFileIndex = 0;
    sampPeriod = 100000;
}

template <class ElemType>
void HTKMLFWriter<ElemType>::Destroy()
{
    delete[] m_tempArray;
    m_tempArray = nullptr;
    m_tempArraySize = 0;
}

template <class ElemType>
void HTKMLFWriter<ElemType>::GetSections(std::map<std::wstring, SectionType, nocase_compare>& /*sections*/)
{
}

template <class ElemType>
bool HTKMLFWriter<ElemType>::SaveData(size_t /*recordStart*/, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t /*datasetSize*/, size_t /*byteVariableSized*/)
{

    // std::map<std::wstring, void*, nocase_compare>::iterator iter;
    if (outputFileIndex >= m_outputFiles[0].size())
        RuntimeError("index for output scp file out of range...");

    if (numRecords == 0)  //indicate end of an utterance
    {
        for (auto iter = m_outputNameToValues.begin(); iter != m_outputNameToValues.end(); iter++)
        {
            wstring outputName = iter->first;
            vector<float>& outputData = iter->second;
            size_t id = m_outputNameToIdMap[outputName];
            size_t dim = m_outputNameToDimMap[outputName];
            size_t totalSamples = m_outputNameToTotalSamples[outputName];
            wstring outFile = m_outputFiles[id][outputFileIndex];

            assert(dim * totalSamples == outputData.size());
            dim;
            Save(outFile, outputData, dim, totalSamples);

            outputData.clear();
            m_outputNameToTotalSamples[outputName] = 0;
        }
        outputFileIndex++;
    }
    else  //not end of an utterance, accumulate values
    {
        for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
        {
            wstring outputName = iter->first;
            Matrix<ElemType>& nodeValue = *(static_cast<Matrix<ElemType>*>(iter->second));
            vector<float>& outputData = m_outputNameToValues[outputName];
            size_t dim = m_outputNameToDimMap[outputName];

            assert(dim == nodeValue.GetNumRows()); 
            dim;

            for (int j = 0; j < nodeValue.GetNumCols(); j++)
            {
                for (int i = 0; i < nodeValue.GetNumRows(); i++)
                {
                    outputData.push_back((float)nodeValue(i,j));
                }
            }

            m_outputNameToTotalSamples[outputName] += nodeValue.GetNumCols();
        }
    }


    return true;
}

template <class ElemType>
void HTKMLFWriter<ElemType>::Save(std::wstring& outputFile, vector<float>& outputData, const size_t dim, const size_t totalSamples)
{
    msra::dbn::matrix output;  //TODO: this extra copy is not needed. leave it for now.
    output.resize(dim, totalSamples);

    size_t k = 0;
    for (size_t j = 0; j < totalSamples; j++)
    {
        for (size_t i = 0; i < dim; i++)
        {
            output(i, j) = outputData[k++];
        }
    }

    const size_t nansinf = output.countnaninf();
    if (nansinf > 0)
        fprintf(stderr, "chunkeval: %d NaNs or INF detected in '%ls' (%d frames)\n", (int) nansinf, outputFile.c_str(), (int) output.cols());
    // save it
    msra::files::make_intermediate_dirs(outputFile);
    msra::util::attempt(5, [&]()
                        {
                            msra::asr::htkfeatwriter::write(outputFile, "USER", this->sampPeriod, output);
                        });

    fprintf(stderr, "evaluate: writing %d frames of %ls\n", (int) output.cols(), outputFile.c_str());
}

template <class ElemType>
void HTKMLFWriter<ElemType>::SaveMapping(std::wstring saveId, const std::map<LabelIdType, LabelType>& /*labelMapping*/)
{
}

template class HTKMLFWriter<float>;
template class HTKMLFWriter<double>;
} } }
