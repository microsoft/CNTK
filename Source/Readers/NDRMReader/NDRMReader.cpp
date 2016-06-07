//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// NDRMReader.cpp : Defines the reader for the Neural Document Ranking model.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS // creating the exports here
#include "DataReader.h"
#include "NDRMReader.h"
#ifdef LEAKDETECT
#include <vld.h> // leak detection
#endif
#ifndef SPARSE_PCREADER_USE_WINDOWS_API
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdio.h>
#include <fcntl.h>
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

DWORD HIDWORD(size_t size)
{
    return size >> 32;
}
DWORD LODWORD(size_t size)
{
    return size & 0xFFFFFFFF;
}

template <class ElemType>
NDRMReader<ElemType>::~NDRMReader()
{
#ifdef SPARSE_PCREADER_USE_WINDOWS_API
    if (m_filemap != NULL)
    {
        UnmapViewOfFile(m_filemap);
    }

    if (m_dataBuffer != NULL)
    {
        UnmapViewOfFile(m_dataBuffer);
    }

    CloseHandle(m_hndl);

    if (m_qEmbFilemap != NULL)
    {
        UnmapViewOfFile(m_qEmbFilemap);
    }

    if (m_qEmbDataBuffer != NULL)
    {
        UnmapViewOfFile(m_qEmbDataBuffer);
    }

    CloseHandle(m_qEmbHndl);

    if (m_qEmbeddingsFile != m_dEmbeddingsFile)
    {
        if (m_dEmbFilemap != NULL)
        {
            UnmapViewOfFile(m_dEmbFilemap);
        }

        if (m_dEmbDataBuffer != NULL)
        {
            UnmapViewOfFile(m_dEmbDataBuffer);
        }

        CloseHandle(m_dEmbHndl);
    }
#else
    munmap(m_dataBuffer, m_filePositionMax); 
    close(m_hndl);

    munmap(m_qEmbDataBuffer, m_qEmbFilePositionMax);
    close(m_qEmbHndl);

    if (m_qEmbeddingsFile != m_dEmbeddingsFile)
    {
        munmap(m_dEmbDataBuffer, m_dEmbFilePositionMax);
        close(m_dEmbHndl);
    }
#endif
}

template <class ElemType>
void NDRMReader<ElemType>::Destroy()
{
    delete this;
}

// Init - Reader Initialize for multiple data sets
// config - [in] configuration parameters for the datareader
template <class ElemType>
template <class ConfigRecordType>
void NDRMReader<ElemType>::InitFromConfig(const ConfigRecordType& readerConfig)
{
    m_miniBatchSize = 0;
    m_traceLevel = readerConfig(L"traceLevel", 0);

    m_file = (const wstring&)readerConfig(L"file");
    m_qEmbeddingsFile = (const wstring&)readerConfig(L"qEmbeddingsFile");
    m_dEmbeddingsFile = (const wstring&)readerConfig(L"dEmbeddingsFile");
    m_numDocs = readerConfig(L"numDocs", (size_t)1);
    m_numWordsPerQuery = readerConfig(L"numWordsPerQuery", (size_t)10);
    m_numWordsPerDoc = readerConfig(L"numWordsPerDoc", (size_t)2000);
    m_vocabSize = readerConfig(L"vocabSize", (size_t)2748230);
    m_vectorSize = readerConfig(L"vectorSize", (size_t)200);
    m_bytesPerSample = sizeof(int32_t) * (m_numWordsPerQuery + m_numDocs * m_numWordsPerDoc);
    m_bytesPerVector = sizeof(ElemType) * m_vectorSize;
    m_currOffset = 0;

#ifdef SPARSE_PCREADER_USE_WINDOWS_API
    m_hndl = CreateFile(m_file.c_str(), GENERIC_READ,
                        FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (m_hndl == INVALID_HANDLE_VALUE)
    {
        RuntimeError("Unable to Open/Create file %ls, error %x", m_file.c_str(), GetLastError());
    }

    GetFileSizeEx(m_hndl, (PLARGE_INTEGER) &m_filePositionMax);
    m_filemap = CreateFileMapping(m_hndl, NULL, PAGE_READONLY, 0, 0, NULL);

    m_dataBuffer = (char*) MapViewOfFile(m_filemap,
                                         FILE_MAP_READ,
                                         HIDWORD(0),
                                         LODWORD(0),
                                         0);

    m_qEmbHndl = CreateFile(m_qEmbeddingsFile.c_str(), GENERIC_READ,
        FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (m_qEmbHndl == INVALID_HANDLE_VALUE)
    {
        RuntimeError("Unable to Open/Create file %ls, error %x", m_qEmbeddingsFile.c_str(), GetLastError());
    }

    GetFileSizeEx(m_qEmbHndl, (PLARGE_INTEGER)&m_qEmbFilePositionMax);
    m_qEmbFilemap = CreateFileMapping(m_qEmbHndl, NULL, PAGE_READONLY, 0, 0, NULL);

    m_qEmbDataBuffer = (char*)MapViewOfFile(m_qEmbFilemap,
        FILE_MAP_READ,
        HIDWORD(0),
        LODWORD(0),
        0);

    if (m_qEmbeddingsFile == m_dEmbeddingsFile)
    {
        m_dEmbDataBuffer = m_qEmbDataBuffer;
        m_dEmbFilePositionMax = m_qEmbFilePositionMax;
    }
    else
    {
        m_dEmbHndl = CreateFile(m_dEmbeddingsFile.c_str(), GENERIC_READ,
            FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (m_dEmbHndl == INVALID_HANDLE_VALUE)
        {
            RuntimeError("Unable to Open/Create file %ls, error %x", m_dEmbeddingsFile.c_str(), GetLastError());
        }

        GetFileSizeEx(m_dEmbHndl, (PLARGE_INTEGER)&m_dEmbFilePositionMax);
        m_dEmbFilemap = CreateFileMapping(m_dEmbHndl, NULL, PAGE_READONLY, 0, 0, NULL);

        m_dEmbDataBuffer = (char*)MapViewOfFile(m_dEmbFilemap,
            FILE_MAP_READ,
            HIDWORD(0),
            LODWORD(0),
            0);
    }
#else
    m_hndl = open(msra::strfun::utf8(m_file).c_str(), O_RDONLY);
    if (m_hndl == -1)
        RuntimeError("Unable to Open/Create file %ls", m_file.c_str());
    struct stat sb;
    if (fstat(m_hndl, &sb) == -1)
        RuntimeError("Unable to Retrieve file size for file %ls", m_file.c_str());
    m_filePositionMax = sb.st_size;
    m_dataBuffer = (char*) mmap(nullptr, m_filePositionMax, PROT_READ, MAP_PRIVATE, m_hndl, 0);
    if (m_dataBuffer == MAP_FAILED)
    {
        m_dataBuffer = nullptr;
        RuntimeError("Could not memory map file %ls", m_file.c_str());
    }

    m_qEmbHndl = open(msra::strfun::utf8(m_qEmbeddingsFile).c_str(), O_RDONLY);
    if (m_qEmbHndl == -1)
        RuntimeError("Unable to Open/Create file %ls", m_qEmbeddingsFile.c_str());
    if (fstat(m_qEmbHndl, &sb) == -1)
        RuntimeError("Unable to Retrieve file size for file %ls", m_qEmbeddingsFile.c_str());
    m_qEmbFilePositionMax = sb.st_size;
    m_qEmbDataBuffer = (char*)mmap(nullptr, m_qEmbFilePositionMax, PROT_READ, MAP_PRIVATE, m_qEmbHndl, 0);
    if (m_qEmbDataBuffer == MAP_FAILED)
    {
        m_qEmbDataBuffer = nullptr;
        RuntimeError("Could not memory map file %ls", m_qEmbeddingsFile.c_str());
    }

    if (m_qEmbeddingsFile == m_dEmbeddingsFile)
    {
        m_dEmbDataBuffer = m_qEmbDataBuffer;
    }
    else
    {
        m_dEmbHndl = open(msra::strfun::utf8(m_dEmbeddingsFile).c_str(), O_RDONLY);
        if (m_dEmbHndl == -1)
            RuntimeError("Unable to Open/Create file %ls", m_dEmbeddingsFile.c_str());
        if (fstat(m_dEmbHndl, &sb) == -1)
            RuntimeError("Unable to Retrieve file size for file %ls", m_dEmbeddingsFile.c_str());
        m_dEmbFilePositionMax = sb.st_size;
        m_dEmbDataBuffer = (char*)mmap(nullptr, m_dEmbFilePositionMax, PROT_READ, MAP_PRIVATE, m_dEmbHndl, 0);
        if (m_dEmbDataBuffer == MAP_FAILED)
        {
            m_dEmbDataBuffer = nullptr;
            RuntimeError("Could not memory map file %ls", m_dEmbeddingsFile.c_str());
        }
    }
#endif
}

//StartMinibatchLoop - Startup a minibatch loop
// mbSize - [in] size of the minibatch (number of Samples, etc.)
// epoch - [in] epoch number for this loop --ignored
// requestedEpochSamples - [in] number of samples to randomize --ignored
template <class ElemType>
void NDRMReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t /*epoch*/, size_t requestedEpochSamples)
{
    if (m_miniBatchSize != mbSize || m_qValues == NULL || m_dValues == NULL || m_labels == NULL)
    {
        m_miniBatchSize = mbSize;
        m_qValues = (char*)malloc(m_bytesPerVector * m_numWordsPerQuery * m_miniBatchSize);
        m_dValues = (char*)malloc(m_bytesPerVector * m_numWordsPerDoc * m_miniBatchSize);
        m_labels = (char*)malloc(sizeof(ElemType) * m_numDocs * m_miniBatchSize);

        memset(m_labels, 0, sizeof(ElemType) * m_numDocs * m_miniBatchSize);
        for (int i = 0; i < m_miniBatchSize; i++)
        {
            ((ElemType*)m_labels)[i * m_numDocs] = (ElemType)1;
        }
    }

    m_numSamplesPerEpoch = requestedEpochSamples;
    m_numSamplesCurrEpoch = 0;
}

// GetMinibatch - Get the next minibatch (features and labels)
// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponding matrix,
//             [out] each matrix resized if necessary containing data.
// returns - true if there are more minibatches, false if no more minibatchs remain
template <class ElemType>
bool NDRMReader<ElemType>::TryGetMinibatch(StreamMinibatchInputs& matrices)
{
    // get out if they didn't call StartMinibatchLoop() first
    if (m_miniBatchSize == 0)
        return false;

    if ((m_filePositionMax - m_currOffset) < (int64_t)m_bytesPerSample)
    {
        if (m_currOffset == 0)
            RuntimeError("Not enough data to fill a single minibatch");
        m_currOffset = 0;
        return false;
    }

    size_t actualMiniBatchSize = min(((m_filePositionMax - m_currOffset) / m_bytesPerSample), m_miniBatchSize);

    for (int i = 0; i <= m_numDocs; i++)
    {
        char* srcAddrBase = (char*)(i == 0 ? m_qEmbDataBuffer : m_dEmbDataBuffer);
        char* tgtAddrBase = (char*)(i == 0 ? m_qValues : m_dValues);
        size_t numWordsPerFeatureSample = (i == 0 ? m_numWordsPerQuery : m_numWordsPerDoc);
        size_t numRows = m_vectorSize * numWordsPerFeatureSample;
        std::wstring featureName = (i == 0 ? L"Q" : L"D" + std::to_wstring(i - 1));

        if (!matrices.HasInput(featureName))
            continue;

        Matrix<ElemType>& features = matrices.GetInputMatrix<ElemType>(featureName);
        features.Resize(numRows, actualMiniBatchSize);

        memset(tgtAddrBase, 0, sizeof(ElemType) * numRows * actualMiniBatchSize);

        for (int j = 0; j < actualMiniBatchSize; j++)
        {
            for (int k = 0; k < numWordsPerFeatureSample; k++)
            {
                int32_t wordId = *(int32_t*)((char*)m_dataBuffer
                                                    + m_currOffset
                                                    + j * m_bytesPerSample
                                                    + (i > 0 ? m_numWordsPerQuery + (i - 1) * m_numWordsPerDoc : 0) * sizeof(int32_t)
                                                    + k * sizeof(int32_t));
                if (wordId == 0)
                    continue;

                for (int l = 0; l < m_vectorSize; l++)
                {
                    char* srcAddr = srcAddrBase + (wordId - 1) * m_bytesPerVector + l * sizeof(ElemType);
                    char* tgtAddr = tgtAddrBase + ((k * m_vectorSize + l) * actualMiniBatchSize + j) * sizeof(ElemType);
                    memcpy(tgtAddr, srcAddr, sizeof(ElemType));
                }
            }
        }

        features.SetValue(numRows, actualMiniBatchSize, features.GetDeviceId(), (ElemType*)tgtAddrBase, matrixFlagNormal);
    }

    if (matrices.HasInput(L"L"))
    {
        Matrix<ElemType>& labels = matrices.GetInputMatrix<ElemType>(L"L");
        labels.Resize(m_numDocs, actualMiniBatchSize);
        labels.SetValue(m_numDocs, actualMiniBatchSize, labels.GetDeviceId(), (ElemType*)m_labels, matrixFlagNormal);
    }

    // create the MBLayout
    m_pMBLayout->Init(actualMiniBatchSize, 1);
    for (size_t i = 0; i < actualMiniBatchSize; i++)
        m_pMBLayout->AddSequence(NEW_SEQUENCE_ID, i, 0, 1);

    m_currOffset += (m_bytesPerSample * actualMiniBatchSize);

    m_numSamplesCurrEpoch += actualMiniBatchSize;
    if (m_numSamplesCurrEpoch > m_numSamplesPerEpoch)
    {
        m_numSamplesCurrEpoch = 0;
        return false;
    }

    return true;
}

template <class ElemType>
bool NDRMReader<ElemType>::DataEnd() { return true; }

// GetLabelMapping - Gets the label mapping from integer index to label type
// returns - a map from numeric datatype to native label type
template <class ElemType>
const std::map<IDataReader::LabelIdType, IDataReader::LabelType>& NDRMReader<ElemType>::GetLabelMapping(const std::wstring& /*sectionName*/)
{
    return m_mapIdToLabel;
}

// SetLabelMapping - Sets the label mapping from integer index to label
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run
template <class ElemType>
void NDRMReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<IDataReader::LabelIdType, LabelType>& labelMapping)
{
    m_mapIdToLabel = labelMapping;
    m_mapLabelToId.clear();
    for (std::pair<unsigned, LabelType> var : labelMapping)
    {
        m_mapLabelToId[var.second] = var.first;
    }
}

// instantiate all the combinations we expect to be used
template class NDRMReader<double>;
template class NDRMReader<float>;
} } }
