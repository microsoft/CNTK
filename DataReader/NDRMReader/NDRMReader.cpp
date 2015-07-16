//
// <copyright file="NDRMReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// NDRMReader.cpp : Defines the reader for the Neural Document Ranking Model (NDRM).
//

#include "stdafx.h"
#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "NDRMReader.h"
#ifdef LEAKDETECT
#include <vld.h> // leak detection
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

DWORD HIDWORD(size_t size) {return size>>32;}
DWORD LODWORD(size_t size) { return size & 0xFFFFFFFF; }

template<class ElemType>
NDRMReader<ElemType>::~NDRMReader()
{
    if (m_filemap != NULL)
    {
        UnmapViewOfFile(m_filemap);
    }

    if (m_dataBuffer != NULL)
    {
        UnmapViewOfFile(m_dataBuffer);
    }

    CloseHandle(m_hndl);

    for (int i = 0; i < FEATURE_COUNT; i++)
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

template<class ElemType>
void NDRMReader<ElemType>::Destroy()
{
    delete this;
}

// Init - Reader Initialize for multiple data sets
// config - [in] configuration parameters for the datareader
template<class ElemType>
void NDRMReader<ElemType>::Init(const ConfigParameters& readerConfig)
{
    m_miniBatchSize = 0;
    m_traceLevel = readerConfig("traceLevel", "0");
    m_maxReadData = readerConfig("maxReadData", "0");
    m_doGradientCheck = readerConfig("gradientCheck", "false");
    m_returnDense = readerConfig("returnDense", "false");
    m_sparsenessFactor = (m_doGradientCheck ? 1 : SPARSENESS_FACTOR_DEFAULT); // Disable sparseness test if gradient check is enabled

    std::vector<std::wstring> featureNames;
    std::vector<std::wstring> labelNames;

    m_file = readerConfig("file");

    // Determine the names of the features and lables sections in the config file.
    // features - [in,out] a vector of feature name strings
    // labels - [in,out] a vector of label name strings
    GetFileConfigNames(readerConfig, featureNames, labelNames);

    if (featureNames.size() != FEATURE_COUNT || labelNames.size() != 1)
    {
        RuntimeError("NDRM requires exactly two features and one label. Their names should match those in NDL definition");
        return;
    }

    m_labelName = labelNames[0];
    
    for (int i = 0; i < FEATURE_COUNT; i++)
    {
        // In the config file, we must specify query features first, then document features. The sequence is different here. Pay attention
        m_featureNames[i] = featureNames[FEATURE_COUNT - i - 1];

        ConfigParameters featureConfig = readerConfig(m_featureNames[i], "");

        if (featureConfig.size() == 0)
            RuntimeError("features config not found, required in configuration: i.e. 'features=[dim=506530]'");

        m_dims[i] = featureConfig("dim");
    }

    m_hndl = CreateFile(m_file.c_str(), GENERIC_READ,
        FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (m_hndl == INVALID_HANDLE_VALUE)
    {
        char message[256];
        sprintf_s(message, "Unable to Open/Create file %ls, error %x", m_file.c_str(), GetLastError());
        throw runtime_error(message);
    }

    GetFileSizeEx(m_hndl, (PLARGE_INTEGER)&m_filePositionMax);
    m_filemap = CreateFileMapping(m_hndl, NULL, PAGE_READONLY, 0, 0, NULL);

    m_dataBuffer = (char*)MapViewOfFile(m_filemap,
        FILE_MAP_READ,
        HIDWORD(0),
        LODWORD(0),
        0);
}

//StartMinibatchLoop - Startup a minibatch loop 
// mbSize - [in] size of the minibatch (number of Samples, etc.)
// epoch - [in] epoch number for this loop --ignored
// requestedEpochSamples - [in] number of samples to randomize --ignored
template<class ElemType>
void NDRMReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t /*epoch*/, size_t /*requestedEpochSamples*/)
{
    if (m_values[0] == NULL || m_miniBatchSize != mbSize)
    {
        m_miniBatchSize = mbSize;

        for (int i = 0; i < FEATURE_COUNT; i++)
        {
            if (m_values[i] != NULL)
            {
                free(m_values[i]);
                free(m_colIndices[i]);
                free(m_rowIndices[i]);
                free(m_labelsBuffer);
            }

            m_values[i] = (ElemType*)malloc(sizeof(ElemType)* m_dims[i] * m_miniBatchSize / m_sparsenessFactor);
            m_rowIndices[i] = (int32_t*)malloc(sizeof(int32_t)* m_dims[i] * m_miniBatchSize / m_sparsenessFactor);
            m_colIndices[i] = (int32_t*)malloc(sizeof(int32_t)* (m_miniBatchSize + 1));
            m_labelsBuffer = (ElemType*)malloc(sizeof(ElemType)* m_miniBatchSize);
        }
    }

    // reset the next read sample
    m_currOffset = 0;
}

// GetMinibatch - Get the next minibatch (features and labels)
// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponing matrix, 
//             [out] each matrix resized if necessary containing data. 
// returns - true if there are more minibatches, false if no more minibatchs remain
template<class ElemType>
bool NDRMReader<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
{
    // get out if they didn't call StartMinibatchLoop() first
    if (m_miniBatchSize == 0)
        return false;

    // Return early (for debugging purposes)
    if (m_maxReadData > 0 && m_currOffset >= m_maxReadData)
        return false;

    if (m_currOffset >= m_filePositionMax)
    {
        return false;
    }

    Matrix<ElemType>* labels = nullptr;
    auto labelEntry = matrices.find(m_labelName);
    bool useLabels = false;
    if (labelEntry != matrices.end())
    {
        labels = labelEntry->second;
        if (labels != nullptr)
        {
            useLabels = true;
        }
    }
    
    std::vector<int32_t> currIndex = std::vector<int32_t>(FEATURE_COUNT);
    for (int i = 0; i < FEATURE_COUNT; i++)
    {
        currIndex[i] = 0;
    }

    size_t j = 0;

    for (j = 0; j < m_miniBatchSize && m_currOffset < m_filePositionMax; j++)
    {
        for (int i = 0; i < FEATURE_COUNT; i++)
        {
            m_colIndices[i][j] = currIndex[i];

            int32_t nnz = *(int32_t*)((char*)m_dataBuffer + m_currOffset);
            m_currOffset += sizeof(int32_t);

            if (nnz > m_dims[i] / m_sparsenessFactor)
            {
                RuntimeError("Input data is too dense - not enough memory allocated");
            }

            memcpy(m_values[i] + currIndex[i], (char*)m_dataBuffer + m_currOffset, sizeof(ElemType)*nnz);
            m_currOffset += (sizeof(ElemType)*nnz);

            memcpy(m_rowIndices[i] + currIndex[i], (char*)m_dataBuffer + m_currOffset, sizeof(int32_t)*nnz);
            m_currOffset += (sizeof(int32_t)*nnz);

            currIndex[i] += nnz;

            // Debug code - print first feature row
            /*if (j == 0)
            {
                fprintf(stderr, "Feature Type=%S\n", m_featureNames[i].c_str());
                fprintf(stderr, "Vals =\t");
                for (int x = 0; x < nnz; x++)
                {
                    fprintf(stderr, "%d\t", (int)m_values[i][x]);
                }
                fprintf(stderr, "\nRIdx =\t");
                for (int x = 0; x < nnz; x++)
                {
                    fprintf(stderr, "%d\t", m_rowIndices[i][x]);
                }
                fprintf(stderr, "\n");
            }*/
        }
        
        ElemType label = *(ElemType*)((char*)m_dataBuffer + m_currOffset);
        m_labelsBuffer[j] = label;
        m_currOffset += sizeof(ElemType);

        int32_t verifCode = *(int32_t*)((char*)m_dataBuffer + m_currOffset);

        if (verifCode != VERIFICATION_CODE)
        {
            throw runtime_error("Verification code did not match - error in reading data");
            return false;
        }

        m_currOffset += sizeof(int32_t);
    }

    for (int i = 0; i < FEATURE_COUNT; i++)
    {
        m_colIndices[i][j] = currIndex[i];

        Matrix<ElemType>& features = *matrices[m_featureNames[i]];
        
        if (features.GetFormat() != MatrixFormat::matrixFormatSparseCSC)
            features.SwitchToMatrixType(MatrixType::SPARSE, MatrixFormat::matrixFormatSparseCSC, false);

        features.SetMatrixFromCSCFormat(m_colIndices[i], m_rowIndices[i], m_values[i], currIndex[i], m_dims[i], j);
    }

    if (m_returnDense || m_doGradientCheck)
    {
        (*matrices[m_featureNames[0]]).SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, true);
        (*matrices[m_featureNames[1]]).SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, true);
    }

    if (useLabels)
    {
        size_t labelRows = (*labels).GetNumRows();
        size_t labelCols = (*labels).GetNumCols();

        if (labelCols != j)
        {
            (*labels).Resize(labelRows, j);
        }

        (*labels).SetValue((ElemType)0);
        (*labels).SetValue(labelRows, j, m_labelsBuffer, 0, (*labels).GetDeviceId());
    }

    // Debug code - print labels
    /*for (int i = 0; i < j; i++)
    {
        fprintf(stderr, "%d", (int)m_labelsBuffer[i]);
        if (i % 20 == 19)
            fprintf(stderr, "\n");
    }*/

    return true;
}

template<class ElemType>
bool NDRMReader<ElemType>::DataEnd(EndDataType endDataType)
{
    bool ret = false;
    switch (endDataType)
    {
    case endDataNull:
        assert(false);
        break;
    case endDataEpoch:
        ret = (m_currOffset >= m_filePositionMax);
        break;
    case endDataSet:
        ret = (m_currOffset >= m_filePositionMax);
        break;
    case endDataSentence:  // for fast reader each minibatch is considered a "sentence", so always true --huh?
        ret = true;
        break;
    }

    return ret;
}

// GetLabelMapping - Gets the label mapping from integer index to label type 
// returns - a map from numeric datatype to native label type 
template<class ElemType>
const std::map<typename IDataReader<ElemType>::LabelIdType, typename IDataReader<ElemType>::LabelType>& NDRMReader<ElemType>::GetLabelMapping(const std::wstring& /*sectionName*/)
{
    return m_mapIdToLabel;
}

// SetLabelMapping - Sets the label mapping from integer index to label 
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run 
template<class ElemType>
void NDRMReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<typename IDataReader<ElemType>::LabelIdType, typename LabelType>& labelMapping)
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
}}}
