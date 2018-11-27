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
#include "HTKMLFWriter.h"
#include "Config.h"
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
    m_overflowWarningCount = 0;

    vector<wstring> scriptpaths;
    vector<wstring> filelist;
    size_t numFiles;
    size_t firstfilesonly = SIZE_MAX; // set to a lower value for testing

    m_verbosity = writerConfig(L"verbosity", 2);
    m_overflowValue = writerConfig(L"overflowValue", 50);
    m_maxNumOverflowWarning = writerConfig(L"maxNumOverflowWarning", 10);

    vector<wstring> outputNames = writerConfig(L"outputNodeNames", ConfigRecordType::Array(stringargvector()));
    if (outputNames.size() < 1)
        RuntimeError("writer needs at least one outputNodeName specified in config");
    int counter = 0;
    foreach_index (i, outputNames) // inputNames should map to node names
    {
        ConfigParameters thisOutput = writerConfig(outputNames[i]);

        if (thisOutput.Exists("dim"))
            udims.push_back(thisOutput(L"dim"));
        else
            RuntimeError("HTKMLFWriter::Init: writer need to specify dim of output");
        if (thisOutput.Exists("file"))
            scriptpaths.push_back(thisOutput(L"file"));
        else if (thisOutput.Exists("scpFile"))
            scriptpaths.push_back(thisOutput(L"scpFile"));
        else
            RuntimeError("HTKMLFWriter::Init: writer needs to specify scpFile for output");

        if (thisOutput.Exists("Kaldicmd"))
        {
            kaldicmd.push_back(thisOutput(L"Kaldicmd"));
            kaldi::BaseFloatMatrixWriter wfea;
            feature_writer.push_back(wfea);
            feature_writer[i].Open(Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(kaldicmd[counter])));
        }

        outputNameToIdMap[outputNames[i]] = i;
        outputNameToDimMap[outputNames[i]] = udims[i];
        wstring type = thisOutput(L"type", "Real");
        if (type == L"Real")
        {
            outputNameToTypeMap[outputNames[i]] = OutputTypes::outputReal;
        }
        else
        {
            throw std::runtime_error("HTKMLFWriter::Init: output type for writer output expected to be Real");
        }
        counter++;
    }

    numFiles = 0;
    foreach_index (i, scriptpaths)
    {
        filelist.clear();
        std::wstring scriptPath = scriptpaths[i];

        // TODO: The format specifier should probably be "%ls" here, but I'm not making that change as part of 
        // this checkin as it is on the larger side and I want to limit its scope.
        fprintf(stderr, "HTKMLFWriter::Init: reading output script file %S ...", scriptPath.c_str());
        size_t n = 0;
        for (msra::files::textreader reader(scriptPath); reader && filelist.size() <= firstfilesonly /*optimization*/;)
        {
            std::wstring line = reader.wgetline();
            wstringstream ss(line);
            std::wstring first_col;
            ss >> first_col;
            filelist.push_back(first_col); // LEOTODO
            n++;
        }

        fprintf(stderr, " %zu entries\n", n);

        if (i == 0)
            numFiles = n;
        else if (n != numFiles)
            throw std::runtime_error(msra::strfun::strprintf("HTKMLFWriter:Init: number of files in each scriptfile inconsistent (%d vs. %d)", numFiles, n));

        outputFiles.push_back(filelist);
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
    for (size_t i = 0; i < feature_writer.size(); i++)
    {
        feature_writer[i].Close();
        fprintf(stderr, "Closed Kaldi writer\n");
    }
}

template <class ElemType>
void HTKMLFWriter<ElemType>::GetSections(std::map<std::wstring, SectionType, nocase_compare>& /*sections*/)
{
}

template <class ElemType>
bool HTKMLFWriter<ElemType>::SaveData(size_t /*recordStart*/, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t /*numRecords*/, size_t /*datasetSize*/, size_t /*byteVariableSized*/)
{

    if (kaldicmd.size() == 0)
    {
        // std::map<std::wstring, void*, nocase_compare>::iterator iter;
        if (outputFileIndex >= outputFiles[0].size())
            RuntimeError("index for output scp file out of range...");

        for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
        {
            wstring outputName = iter->first;
            Matrix<ElemType>& outputData = *(static_cast<Matrix<ElemType>*>(iter->second));
            size_t id = outputNameToIdMap[outputName];
            size_t dim = outputNameToDimMap[outputName];
            wstring outFile = outputFiles[id][outputFileIndex];

            assert(outputData.GetNumRows() == dim);
            dim;

            SaveToKaldiFile(outFile, outputData);
        }

        outputFileIndex++;
    }
    else
    {
        if (outputFileIndex >= outputFiles[0].size())
            RuntimeError("index for output scp file out of range...");
        int i = 0;
        for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
        {
            wstring outputName = iter->first;
            Matrix<ElemType>& outputData = *(static_cast<Matrix<ElemType>*>(iter->second));
            size_t id = outputNameToIdMap[outputName];
            size_t dim = outputNameToDimMap[outputName];
            wstring outFile = outputFiles[id][outputFileIndex];
            string wfea = "ark:" + Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(outFile));

            // wfea = msra::strfun::utf8(kaldicmd[i]);
            // feature_writer[i].Open(wfea);
            kaldi::Matrix<kaldi::BaseFloat> nnet_out_host;

            assert(outputData.GetNumRows() == dim);
            dim;
            const std::string outputPath = Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(outFile));
            const std::string file_key = removeExtension(basename(outputPath));

            nnet_out_host.Resize(outputData.GetNumCols(), outputData.GetNumRows());
            outputData.CopyToArray(m_tempArray, m_tempArraySize);
            ElemType* pValue = m_tempArray;

            for (int j = 0; j < outputData.GetNumCols(); j++)
            {
                for (int i = 0; i < outputData.GetNumRows(); i++)
                {
                    nnet_out_host(j, i) = (float) *pValue++;
                    if (nnet_out_host(j, i) > m_overflowValue)
                    {
                        nnet_out_host(j, i) = -(float) log(1.0 / outputData.GetNumCols());
                        if (m_verbosity > 0 && m_overflowWarningCount < m_maxNumOverflowWarning)
                        {
                            fprintf(stderr, "overflowed!! : %d %d frames of %s\n", i, j, wfea.c_str());
                            m_overflowWarningCount++;
                        }
                    }
                }
            }

            fprintf(stderr, "evaluate: writing %zu frames of %s\n", outputData.GetNumCols(), wfea.c_str());
            feature_writer[i].Write(file_key, nnet_out_host);
            i++;
        }

        outputFileIndex++;
    }
    return true;
}

template <class ElemType>
void HTKMLFWriter<ElemType>::Save(std::wstring& outputFile, Matrix<ElemType>& outputData)
{
    msra::dbn::matrix output;
    output.resize(outputData.GetNumRows(), outputData.GetNumCols());
    outputData.CopyToArray(m_tempArray, m_tempArraySize);
    ElemType* pValue = m_tempArray;

    for (int j = 0; j < outputData.GetNumCols(); j++)
    {
        for (int i = 0; i < outputData.GetNumRows(); i++)
        {
            output(i, j) = (float) *pValue++;
        }
    }

    const size_t nansinf = output.countnaninf();
    if (nansinf > 0)
        fprintf(stderr, "chunkeval: %d NaNs or INF detected in '%S' (%d frames)\n", (int) nansinf, outputFile.c_str(), (int) output.cols());
    // save it
    msra::files::make_intermediate_dirs(outputFile);
    msra::util::attempt(5, [&]()
                        {
                            msra::asr::htkfeatwriter::write(outputFile, "USER", this->sampPeriod, output);
                        });

    fprintf(stderr, "evaluate: writing %zu frames of %S\n", output.cols(), outputFile.c_str());
}

template <class ElemType>
void HTKMLFWriter<ElemType>::SaveToKaldiFile(std::wstring& outputFile, Matrix<ElemType>& outputData)
{
    msra::dbn::matrix output;
    output.resize(outputData.GetNumRows(), outputData.GetNumCols());
    outputData.CopyToArray(m_tempArray, m_tempArraySize);
    ElemType* pValue = m_tempArray;

    for (int j = 0; j < outputData.GetNumCols(); j++)
    {
        for (int i = 0; i < outputData.GetNumRows(); i++)
        {
            output(i, j) = (float) *pValue++;
        }
    }

    const size_t nansinf = output.countnaninf();
    if (nansinf > 0)
        fprintf(stderr, "chunkeval: %d NaNs or INF detected in '%S' (%d frames)\n", (int) nansinf, outputFile.c_str(), (int) output.cols());
    // save it
    msra::files::make_intermediate_dirs(outputFile);
    msra::util::attempt(5, [&]()
                        {
                            msra::asr::htkfeatwriter::writeKaldi(outputFile, "USER", this->sampPeriod, output, sizeof(ElemType));
                        });

    fprintf(stderr, "evaluate: writing %zu frames of %S\n", output.cols(), outputFile.c_str());
}

template <class ElemType>
void HTKMLFWriter<ElemType>::SaveMapping(std::wstring saveId, const std::map<LabelIdType, LabelType>& /*labelMapping*/)
{
}

template class HTKMLFWriter<float>;
template class HTKMLFWriter<double>;
} } }
