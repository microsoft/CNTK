//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// SparsePCReader.cpp : Defines the Sparse Parallel Corpus reader.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS // creating the exports here
#include "DataReader.h"
#include "SparsePCReader.h"
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
SparsePCReader<ElemType>::~SparsePCReader()
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
#else
    munmap(m_dataBuffer, m_filePositionMax); 
    close(m_hndl);
#endif
    for (int i = 0; i < m_featureCount; i++)
    {
        if (m_values[i] != NULL)
        {
            free(m_values[i]);
        }

        if (m_rowIndices[i] != NULL)
        {
            free(m_rowIndices[i]);
        }

        if (m_colIndices[i] != NULL)
        {
            free(m_colIndices[i]);
        }
    }

    if (m_labelsBuffer != NULL)
    {
        free(m_labelsBuffer);
    }
}

template <class ElemType>
void SparsePCReader<ElemType>::Destroy()
{
    delete this;
}

// Init - Reader Initialize for multiple data sets
// config - [in] configuration parameters for the datareader
template <class ElemType>
template <class ConfigRecordType>
void SparsePCReader<ElemType>::InitFromConfig(const ConfigRecordType& readerConfig)
{
    // Sparse PC reader considers every consecutive N rows to be part of a single block.
    // This is used later to compute the corss-entropy with softmax per block.
    // Default value is 1 to indicate all rows are independent.
    m_microBatchSize = readerConfig(L"microbatchSize", (size_t) 1);

    m_miniBatchSize = 0;
    m_traceLevel = readerConfig(L"traceLevel", 0);
    m_maxReadData = readerConfig(L"maxReadData", (size_t) 0);
    m_doGradientCheck = readerConfig(L"gradientCheck", false);
    m_returnDense = readerConfig(L"returnDense", false);
    m_sparsenessFactor = readerConfig(L"sparsenessFactor", (size_t) 50); // We don't expect more than one in 50 input positions to have non-zero values
    m_verificationCode = (int32_t) readerConfig(L"verificationCode", (size_t) 0);

    std::vector<std::wstring> featureNames;
    std::vector<std::wstring> labelNames;

    m_file = (const wstring&) readerConfig(L"file");

    // Determine the names of the features and lables sections in the config file.
    // features - [in,out] a vector of feature name strings
    // labels - [in,out] a vector of label name strings
    GetFileConfigNames(readerConfig, featureNames, labelNames);

    if (labelNames.size() != 1)
    {
        RuntimeError("SparsePC requires exactly one label. Their names should match those in NDL definition");
        return;
    }

    m_featureCount = featureNames.size();
    m_labelName = labelNames[0];

    m_featureNames = std::vector<std::wstring>(m_featureCount);
    m_dims = std::vector<size_t>(m_featureCount);
    m_values = std::vector<ElemType*>(m_featureCount);
    m_rowIndices = std::vector<int32_t*>(m_featureCount);
    m_colIndices = std::vector<int32_t*>(m_featureCount);

    for (int i = 0; i < m_featureCount; i++)
    {
        // In the config file, we must specify query features first, then document features. The sequence is different here. Pay attention
        m_featureNames[i] = featureNames[m_featureCount - i - 1];

        ConfigParameters featureConfig = readerConfig(m_featureNames[i]);

        if (featureConfig.size() == 0)
            RuntimeError("features config not found, required in configuration: i.e. 'features=[dim=506530]'");

        m_dims[i] = featureConfig("dim");
    }

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

#endif
}

//StartMinibatchLoop - Startup a minibatch loop
// mbSize - [in] size of the minibatch (number of Samples, etc.)
// epoch - [in] epoch number for this loop --ignored
// requestedEpochSamples - [in] number of samples to randomize --ignored
template <class ElemType>
void SparsePCReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t /*epoch*/, size_t /*requestedEpochSamples*/)
{
    if (m_values[0] == NULL || m_miniBatchSize != mbSize)
    {
        m_miniBatchSize = mbSize;

        for (int i = 0; i < m_featureCount; i++)
        {
            if (m_values[i] != NULL)
            {
                free(m_values[i]);
                free(m_colIndices[i]);
                free(m_rowIndices[i]);
                free(m_labelsBuffer);
            }

            m_values[i] = (ElemType*) malloc(sizeof(ElemType) * m_dims[i] * m_miniBatchSize / m_sparsenessFactor);
            m_rowIndices[i] = (int32_t*) malloc(sizeof(int32_t) * m_dims[i] * m_miniBatchSize / m_sparsenessFactor);
            m_colIndices[i] = (int32_t*) malloc(sizeof(int32_t) * (m_miniBatchSize + 1));
            m_labelsBuffer = (ElemType*) malloc(sizeof(ElemType) * m_miniBatchSize);
        }
    }

    // reset the next read sample
    m_currOffset = 0;
}

// GetMinibatch - Get the next minibatch (features and labels)
// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponding matrix,
//             [out] each matrix resized if necessary containing data.
// returns - true if there are more minibatches, false if no more minibatchs remain
template <class ElemType>
bool SparsePCReader<ElemType>::TryGetMinibatch(StreamMinibatchInputs& matrices)
{
    // get out if they didn't call StartMinibatchLoop() first
    if (m_miniBatchSize == 0)
        return false;

    // Return early (for debugging purposes)
    if (m_maxReadData > 0 && m_currOffset >= m_maxReadData)
        return false;

    if (m_currOffset >= m_filePositionMax)
        return false;

    Matrix<ElemType>* labels = nullptr; // labels to return, or NULL if no labels in matrix set
    if (matrices.HasInput(m_labelName))
    {
        labels = &matrices.GetInputMatrix<ElemType>(m_labelName);
        if (labels->GetNumRows() != 1)
            RuntimeError("SparsePCReader only supports single label value per column but the network expected %d.", (int) labels->GetNumRows());
    }

    std::vector<int32_t> currIndex = std::vector<int32_t>(m_featureCount);
    for (int i = 0; i < m_featureCount; i++)
    {
        currIndex[i] = 0;
    }

    size_t j = 0;

    for (j = 0; j < m_miniBatchSize && m_currOffset < m_filePositionMax; j++)
    {
        for (int i = 0; i < m_featureCount; i++)
        {
            m_colIndices[i][j] = currIndex[i];

            int32_t nnz = *(int32_t*) ((char*) m_dataBuffer + m_currOffset);
            m_currOffset += sizeof(int32_t);

            if (nnz > m_dims[i] / m_sparsenessFactor)
            {
                RuntimeError("Input data is too dense - not enough memory allocated");
            }

            memcpy(m_values[i] + currIndex[i], (char*) m_dataBuffer + m_currOffset, sizeof(ElemType) * nnz);
            m_currOffset += (sizeof(ElemType) * nnz);

            memcpy(m_rowIndices[i] + currIndex[i], (char*) m_dataBuffer + m_currOffset, sizeof(int32_t) * nnz);
            m_currOffset += (sizeof(int32_t) * nnz);

            currIndex[i] += nnz;
        }

        ElemType label = *(ElemType*) ((char*) m_dataBuffer + m_currOffset);
        m_labelsBuffer[j] = label;
        m_currOffset += sizeof(ElemType);

        if (m_verificationCode != 0)
        {
            int32_t verifCode = *(int32_t*) ((char*) m_dataBuffer + m_currOffset);

            if (verifCode != m_verificationCode)
            {
                RuntimeError("Verification code did not match (expected %d) - error in reading data", m_verificationCode);
                return false;
            }

            m_currOffset += sizeof(int32_t);
        }
    }

    for (int i = 0; i < m_featureCount; i++)
    {
        m_colIndices[i][j] = currIndex[i];
        Matrix<ElemType>& features = matrices.GetInputMatrix<ElemType>(m_featureNames[i]);

        if (features.GetFormat() != MatrixFormat::matrixFormatSparseCSC)
            features.SwitchToMatrixType(MatrixType::SPARSE, MatrixFormat::matrixFormatSparseCSC, false);

        features.SetMatrixFromCSCFormat(m_colIndices[i], m_rowIndices[i], m_values[i], currIndex[i], m_dims[i], j);
    }

    if (m_returnDense || m_doGradientCheck)
    {
        for (int i = 0; i < m_featureCount; i++)
            matrices.GetInputMatrix<ElemType>(m_featureNames[i]).SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, true);
    }

    if (labels)
    {
        labels->Resize(1, j);
        labels->SetValue((ElemType) 0);
        labels->SetValue(1, j, labels->GetDeviceId(), m_labelsBuffer, 0);
    }

    // create the MBLayout
    // Each sample consists of a "sequence" of 'm_microBatchSize' samples.
    // TODO: They are not really temporally ordered, so a better way would be to use tensors, once that is ready.
    m_pMBLayout->Init(j / m_microBatchSize, m_microBatchSize);
    for (size_t s = 0; s < j / m_microBatchSize; s++)
        m_pMBLayout->AddSequence(NEW_SEQUENCE_ID, s, 0, m_microBatchSize);

    return true;
}

template <class ElemType>
bool SparsePCReader<ElemType>::DataEnd() { return true; }

// GetLabelMapping - Gets the label mapping from integer index to label type
// returns - a map from numeric datatype to native label type
template <class ElemType>
const std::map<IDataReader::LabelIdType, IDataReader::LabelType>& SparsePCReader<ElemType>::GetLabelMapping(const std::wstring& /*sectionName*/)
{
    return m_mapIdToLabel;
}

// SetLabelMapping - Sets the label mapping from integer index to label
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run
template <class ElemType>
void SparsePCReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<IDataReader::LabelIdType, LabelType>& labelMapping)
{
    m_mapIdToLabel = labelMapping;
    m_mapLabelToId.clear();
    for (std::pair<unsigned, LabelType> var : labelMapping)
    {
        m_mapLabelToId[var.second] = var.first;
    }
}

// instantiate all the combinations we expect to be used
template class SparsePCReader<double>;
template class SparsePCReader<float>;
} } }
