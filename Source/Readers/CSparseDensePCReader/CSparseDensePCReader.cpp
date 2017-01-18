//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CSparseDensePCReader.cpp : Defines the Compressed Sparse Parallel Corpus reader.
//

#include "stdafx.h"
#include <cstdint>
#define DATAREADER_EXPORTS // creating the exports here
#include "DataReader.h"
#include "CSparseDensePCReader.h"
#ifdef LEAKDETECT
#include <vld.h> // leak detection
#endif
// #ifndef SPARSE_PCREADER_USE_WINDOWS_API
// #include <sys/mman.h>
// #include <sys/stat.h>
// #include <stdio.h>
// #include <fcntl.h>
// #endif

namespace Microsoft { namespace MSR { namespace CNTK {

DWORD HIDWORD(size_t size)
{
    return size >> 32;
}
DWORD LODWORD(size_t size)
{
    return size & 0xFFFFFFFF;
}
template<class ElemType>
void SparseDenseMemory<ElemType>::Clear()
{
    this->m_denseIndex = 0;
    this->m_sparseValueIndex = 0;
    this->m_sparseRowIndex = 0;
    this->m_sparseColIndex = 0;
    this->m_sampleNumber = 0;
    this->m_nnZ = 0;

    if (this->m_bIsSparse)
    {
        this->m_colIndices[0] = 0;
    }

}

template<class ElemType>
void SparseDenseMemory<ElemType>::AddDenseData(void* pSource, size_t dim)
{
    this->m_sampleNumber++;
    size_t increase = dim*sizeof(ElemType);

    memcpy((char*)this->m_values + this->m_denseIndex, pSource, increase);
    this->m_denseIndex += increase;

}

template<class ElemType>
void SparseDenseMemory<ElemType>::AddSparseData(void* nnzValue, void* nnzIndex, int32_t length)
{
    size_t increaseOfnnzValue = length* sizeof(ElemType);
    size_t increaseofnnzIndex = length*sizeof(int32_t);

    if (this->m_sampleNumber == 0)
    {
        this->m_colIndices[0] = 0;
    }

    this->m_sampleNumber++;
    memcpy((char*)this->m_values + this->m_sparseValueIndex*sizeof(ElemType), nnzValue, increaseOfnnzValue);
    this->m_sparseValueIndex += length;

    this->m_nnZ += length;

    memcpy((char*)this->m_rowIndices + this->m_sparseRowIndex*sizeof(int32_t), nnzIndex, increaseofnnzIndex);

    this->m_sparseRowIndex += length;

    this->m_colIndices[m_sampleNumber] = m_colIndices[m_sampleNumber - 1] + length;
    this->m_sparseColIndex++;


}


template<class ElemType>
void SparseDenseMemory<ElemType>::FillDenseMatrix(Matrix<ElemType>* matrix)
{

    matrix->Resize(m_Dim, this->m_sampleNumber);
    matrix->SetValue((ElemType)0);

    matrix->SetValue(this->m_Dim, this->m_sampleNumber, matrix->GetDeviceId(), this->m_values, matrixFlagNormal);

    //for (size_t j = 0; j < matrix.GetNumCols(); j++)
    //{
    //    size_t start = matrix.m_compIndex[j];  //ColLocation
    //    size_t end = matrix.m_compIndex[j + 1];
    //    for (size_t p = start; p < end; p++)
    //    {
    //        size_t i = rhs.m_unCompIndex[p]; //RowLocation
    //        ElemType val = rhs.m_pArray[p];

    //        for (size_t h = 0; h < lhs.GetNumRows(); h++)
    //        {
    //            if (h >= c.GetNumRows() || j >= c.GetNumCols() || h >= lhs.GetNumRows() || i >= lhs.GetNumCols())
    //            {
    //                std::cout << "ErrorHere" << std::endl;
    //            }


    //            c(h, j) += alpha * lhs(h, i)*val;
    //        }
    //    }
    //}


}
template<class ElemType>
void SparseDenseMemory<ElemType>::FillSparseMatrix(Matrix<ElemType>* matrix)
{
    if (matrix->GetFormat() != MatrixFormat::matrixFormatSparseCSC)
        matrix->SwitchToMatrixType(MatrixType::SPARSE, MatrixFormat::matrixFormatSparseCSC, false);

    matrix->SetMatrixFromCSCFormat(m_colIndices, m_rowIndices, m_values, m_nnZ, m_Dim, m_sampleNumber);
}
template <class ElemType>
CSparseDensePCReader<ElemType>::~CSparseDensePCReader()
{
}

template <class ElemType>
void CSparseDensePCReader<ElemType>::Destroy()
{
    delete this;
}

// Init - Reader Initialize for multiple data sets
// config - [in] configuration parameters for the datareader
template <class ElemType>
template <class ConfigRecordType>
void CSparseDensePCReader<ElemType>::InitFromConfig(const ConfigRecordType& readerConfig)
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
    m_reshapeInputToRowSize = readerConfig(L"reshapeInputToRowSize", (size_t)0);

    std::vector<std::wstring> featureNames;
    std::vector<std::wstring> labelNames;

    m_file = (const wstring&) readerConfig(L"file");    
    m_mapReaderOrder2FeatureName = std::map<int32_t, std::wstring>();
    for (const auto & id : readerConfig.GetMemberIds())
    {
        if (!readerConfig.CanBeConfigRecord(id))
            continue;
        const ConfigRecordType & temp = readerConfig(id);
        // see if we have a config parameters that contains a "dim" element, it's a sub key, use it
        if (temp.ExistsCurrent(L"dim"))
        {
            bool bIsSparse = temp(L"isSparse", true);
            int32_t featureDim = temp(L"dim", 0);
            int32_t featureSparseFactor = temp(L"sparsenessFactor", 0);
            int32_t readerOrder = temp(L"readerOrder", 0);

            if (featureSparseFactor == 0)
            {
                featureSparseFactor = (int32_t)m_sparsenessFactor;
            }

            SparseDenseFeatureInfo* featureInfo = new SparseDenseFeatureInfo();
            featureInfo->m_bIsSparse = bIsSparse;
            featureInfo->m_Dim = featureDim;
            featureInfo->m_SparseFactor = featureSparseFactor;
            wstring wname = id;
            m_mapSparseDenseInfo[wname] = *featureInfo;
            m_mapReaderOrder2FeatureName[readerOrder] = wname;
        }
    }

    this->m_bSparseDenseInfoInitialized = false;

    std::string name = msra::strfun::utf8(m_file);
    m_inFile.open(name, ifstream::binary | ifstream::in);

    m_inFile.seekg(0, ios::end);
    this->m_filePositionMax = (int64_t)m_inFile.tellg();

    if (m_filePositionMax < 0)
    {
        RuntimeError("Your Data file Does not exists, Check your Path");
    }
    // // Determine the names of the features and lables sections in the config file.
    // // features - [in,out] a vector of feature name strings
    // // labels - [in,out] a vector of label name strings
    // GetFileConfigNames(readerConfig, featureNames, labelNames);

    // if (labelNames.size() != 1)
    // {
        // RuntimeError("CSparsePC requires exactly one label. Their names should match those in NDL definition");
        // return;
    // }

    // m_featureCount = featureNames.size();
    // m_labelName = labelNames[0];

    // m_featureNames = std::vector<std::wstring>(m_featureCount);
    // m_dims = std::vector<size_t>(m_featureCount);
    // m_values = std::vector<ElemType*>(m_featureCount);
    // m_rowIndices = std::vector<int32_t*>(m_featureCount);
    // m_colIndices = std::vector<int32_t*>(m_featureCount);

    // for (int i = 0; i < m_featureCount; i++)
    // {
        // // In the config file, we must specify query features first, then document features. The sequence is different here. Pay attention
        // m_featureNames[i] = featureNames[m_featureCount - i - 1];

        // ConfigParameters featureConfig = readerConfig(m_featureNames[i]);

        // if (featureConfig.size() == 0)
            // RuntimeError("features config not found, required in configuration: i.e. 'features=[dim=506530]'");

        // m_dims[i] = featureConfig("dim");
    // }

// #ifdef SPARSE_PCREADER_USE_WINDOWS_API
    // m_hndl = CreateFile(m_file.c_str(), GENERIC_READ,
                        // FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    // if (m_hndl == INVALID_HANDLE_VALUE)
    // {
        // RuntimeError("Unable to Open/Create file %ls, error %x", m_file.c_str(), GetLastError());
    // }

    // GetFileSizeEx(m_hndl, (PLARGE_INTEGER) &m_filePositionMax);
    // m_filemap = CreateFileMapping(m_hndl, NULL, PAGE_READONLY, 0, 0, NULL);

    // m_dataBuffer = (char*) MapViewOfFile(m_filemap,
                                         // FILE_MAP_READ,
                                         // HIDWORD(0),
                                         // LODWORD(0),
                                         // 0);
// #else
    // m_hndl = open(msra::strfun::utf8(m_file).c_str(), O_RDONLY);
    // if (m_hndl == -1)
        // RuntimeError("Unable to Open/Create file %ls", m_file.c_str());
    // struct stat sb;
    // if (fstat(m_hndl, &sb) == -1)
        // RuntimeError("Unable to Retrieve file size for file %ls", m_file.c_str());
    // m_filePositionMax = sb.st_size;
    // m_dataBuffer = (char*) mmap(nullptr, m_filePositionMax, PROT_READ, MAP_PRIVATE, m_hndl, 0);
    // if (m_dataBuffer == MAP_FAILED)
    // {
        // m_dataBuffer = nullptr;
        // RuntimeError("Could not memory map file %ls", m_file.c_str());
    // }

// #endif
}

//StartMinibatchLoop - Startup a minibatch loop
// mbSize - [in] size of the minibatch (number of Samples, etc.)
// epoch - [in] epoch number for this loop --ignored
// requestedEpochSamples - [in] number of samples to randomize --ignored
template <class ElemType>
void CSparseDensePCReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t /*epoch*/, size_t /*requestedEpochSamples*/)
{
    //Set file offset to zero
    this->m_currOffset = 0;

    this->m_inFile.seekg(ios_base::beg);
    if (m_miniBatchSize != mbSize)
    {
        m_miniBatchSize = mbSize;
    }

    if (!this->m_bSparseDenseInfoInitialized)
    {

        size_t maxMBSize = 0;
        size_t maxMem = 1024 * 1024 * 1024; // 1GB
                                            //size_t maxPointers = maxMem / m_maxMBSize;

        for (int32_t i = 0; i < m_mapReaderOrder2FeatureName.size(); i++)
        {
            auto featureName = m_mapReaderOrder2FeatureName[i];

            SparseDenseFeatureInfo featureInfo = this->m_mapSparseDenseInfo[featureName];

            SparseDenseMemory<ElemType>* pieceMemory = new SparseDenseMemory<ElemType>();

            pieceMemory->m_bIsSparse = featureInfo.m_bIsSparse;
            pieceMemory->m_Dim = featureInfo.m_Dim;


            if (featureInfo.m_bIsSparse)
            {
                size_t valuesSize = sizeof(ElemType)* featureInfo.m_Dim * m_miniBatchSize / featureInfo.m_SparseFactor;
                pieceMemory->m_values = (ElemType*)malloc(valuesSize);

                size_t rowIdSize = sizeof(int32_t)* featureInfo.m_Dim * m_miniBatchSize / featureInfo.m_SparseFactor;

                pieceMemory->m_rowIndices = (int32_t*)malloc(sizeof(int32_t)* featureInfo.m_Dim * m_miniBatchSize / featureInfo.m_SparseFactor);
                size_t columnIdSize = sizeof(int32_t)* (m_miniBatchSize + 1);
                pieceMemory->m_colIndices = (int32_t*)malloc(sizeof(int32_t)* (m_miniBatchSize + 1));

                maxMBSize += valuesSize;
                maxMBSize += rowIdSize;
                maxMBSize += columnIdSize;
            }
            else
            {
                size_t denseValueSize = sizeof(ElemType)* featureInfo.m_Dim * m_miniBatchSize;
                pieceMemory->m_values = (ElemType*)malloc(denseValueSize);
                maxMBSize += denseValueSize;
            }

            this->m_mapWorkingMemory[featureName] = pieceMemory;
        }

        size_t maxPointers = maxMem / maxMBSize;

        for (int i = 0; i < maxPointers; i++)
        {
            std::map<wstring, SparseDenseMemory<ElemType>*> perMap;
            for (int32_t i = 0; i < m_mapReaderOrder2FeatureName.size(); i++)
            {
                auto featureName = m_mapReaderOrder2FeatureName[i];

                SparseDenseFeatureInfo featureInfo = this->m_mapSparseDenseInfo[featureName];

                SparseDenseMemory<ElemType>* pieceMemory = new SparseDenseMemory<ElemType>();

                pieceMemory->m_bIsSparse = featureInfo.m_bIsSparse;
                pieceMemory->m_Dim = featureInfo.m_Dim;


                if (featureInfo.m_bIsSparse)
                {
                    size_t valuesSize = sizeof(ElemType)* featureInfo.m_Dim * m_miniBatchSize / featureInfo.m_SparseFactor;
                    pieceMemory->m_values = (ElemType*)malloc(valuesSize);
                    pieceMemory->m_rowIndices = (int32_t*)malloc(sizeof(int32_t)* featureInfo.m_Dim * m_miniBatchSize / featureInfo.m_SparseFactor);
                    pieceMemory->m_colIndices = (int32_t*)malloc(sizeof(int32_t)* (m_miniBatchSize + 1));

                }
                else
                {
                    size_t denseValueSize = sizeof(ElemType)* featureInfo.m_Dim * m_miniBatchSize;
                    pieceMemory->m_values = (ElemType*)malloc(denseValueSize);
                }

                perMap[featureName] = pieceMemory;


            }
            this->m_dataToProduce.push(perMap);

        }

        m_bSparseDenseInfoInitialized = true;



    }
    // reset the next read sample
    m_currOffset = 0;
    std::thread readData([this] { this->ReadData(); });
    readData.detach();
}

// GetMinibatch - Get the next minibatch (features and labels)
// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponding matrix,
//             [out] each matrix resized if necessary containing data.
// returns - true if there are more minibatches, false if no more minibatchs remain
template <class ElemType>
bool CSparseDensePCReader<ElemType>::TryGetMinibatch(StreamMinibatchInputs& matrices)
{
    // get out if they didn't call StartMinibatchLoop() first
    if (m_miniBatchSize == 0)
        return false;

    //// Return early (for debugging purposes)
    //if (m_maxReadData > 0 && m_currOffset >= m_maxReadData)
    //    return false;

    //if (m_currOffset >= m_filePositionMax)
    //    return false;
    std::map<wstring, SparseDenseMemory<ElemType>*> dataToConsume = this->m_dataToConsume.pop();

    //First test if there is not any data left, when end of data is met, it will send out an empty data.

    wstring firstFeatureName = this->m_mapReaderOrder2FeatureName[0];
    if (dataToConsume[firstFeatureName]->m_sampleNumber == 0)
    {
        for (int32_t i = 0; i < m_mapReaderOrder2FeatureName.size(); i++)
        {
            std::wstring featureName = m_mapReaderOrder2FeatureName[i];
            dataToConsume[featureName]->Clear();
        }
        this->m_dataToProduce.push(dataToConsume);
        return false;
    }


    int32_t sampleSize = 0;

    for (int32_t i = 0; i < m_mapReaderOrder2FeatureName.size(); i++)
    {
        std::wstring featureName = m_mapReaderOrder2FeatureName[i];

        SparseDenseMemory<ElemType>* pMemoryInfo = dataToConsume[featureName];
        if (!matrices.HasInput(featureName))
        {
            pMemoryInfo->Clear();
            //this matrix is not needed in the NDL file, skip it 
            continue;
        }

        Matrix<ElemType>* pMatrix = &matrices.GetInputMatrix<ElemType>(featureName);
        SparseDenseFeatureInfo info = this->m_mapSparseDenseInfo[featureName];

        sampleSize = pMemoryInfo->m_sampleNumber;

        if (info.m_bIsSparse)
        {
            pMemoryInfo->FillSparseMatrix(pMatrix);            

            //This only happens in sparse matrix

            if (m_reshapeInputToRowSize != 0)
            {
                pMatrix->Reshape(m_reshapeInputToRowSize, info.m_Dim * sampleSize / m_reshapeInputToRowSize);
            }

            if (m_returnDense || m_doGradientCheck)
            {
                pMatrix->SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, true);
            }
        }
        else
        {

            pMemoryInfo->FillDenseMatrix(pMatrix);

        }
        pMemoryInfo->Clear();
    }
    this->m_dataToProduce.push(dataToConsume);
    // TODO: They are not really temporally ordered, so a better way would be to use tensors, once that is ready.
    m_pMBLayout->InitAsFrameMode(sampleSize);

    //m_pMBLayout->Init(sampleSize / m_microBatchSize, m_microBatchSize);
    //for (size_t s = 0; s < sampleSize / m_microBatchSize; s++)
    //       m_pMBLayout->AddSequence(NEW_SEQUENCE_ID, s, 0, m_microBatchSize);

    return true;
}
//ReadData, read data into a shared memory, if no data is read, then just break

template<class ElemType>
void CSparseDensePCReader<ElemType>::ReadData()
{
    ///Read data, if file end has been ended. Then insert a Empty one to the producer side.

    size_t currentSampleNumber = 0;
    std::map<wstring, SparseDenseMemory<ElemType>*> workingMemory;
    while (this->m_currOffset<this->m_filePositionMax)
    {
        //Now a new batch
        if (currentSampleNumber == 0)
        {
            workingMemory = this->m_dataToProduce.pop();
        }

        for (int32_t i = 0; i < m_mapReaderOrder2FeatureName.size(); i++)
        {
            std::wstring featureName = m_mapReaderOrder2FeatureName[i];
            auto& featureInfo = this->m_mapSparseDenseInfo[featureName];

            SparseDenseMemory<ElemType>* pmemory = this->m_mapWorkingMemory[featureName];
            SparseDenseMemory<ElemType>* pConsumeMemory = workingMemory[featureName];

            if (featureInfo.m_bIsSparse)
            {
                int32_t nnZ;
                this->m_inFile.read((char*)&nnZ, sizeof(int32_t));
                m_currOffset += sizeof(int32_t);
                this->m_inFile.read((char*)pmemory->m_values, sizeof(ElemType)*nnZ);
                m_currOffset += sizeof(ElemType)*nnZ;
                this->m_inFile.read((char*)pmemory->m_rowIndices, sizeof(int32_t)*nnZ);
                m_currOffset += sizeof(int32_t)*nnZ;
                pConsumeMemory->AddSparseData(pmemory->m_values, pmemory->m_rowIndices, nnZ);

            }
            else
            {
                this->m_inFile.read((char*)pmemory->m_values, sizeof(ElemType)*featureInfo.m_Dim);
                m_currOffset += sizeof(ElemType)*featureInfo.m_Dim;
                pConsumeMemory->AddDenseData(pmemory->m_values, featureInfo.m_Dim);
            }
        }
        int32_t verificationCode;
        m_inFile.read((char*)&verificationCode, sizeof(int32_t));
        m_currOffset += sizeof(int32_t);
        if (verificationCode != this->m_verificationCode)
        {
            RuntimeError("Verification code did not match (expected %d) - error in reading data", m_verificationCode);
            return;
        }
        //Reading one End
        currentSampleNumber++;
        if (currentSampleNumber == this->m_miniBatchSize)
        {
            this->m_dataToConsume.push(workingMemory);
            currentSampleNumber = 0;
        }

    }
    //let's check if there are some left over
    if (currentSampleNumber > 0)
    {
        this->m_dataToConsume.push(workingMemory);
    }
    workingMemory = this->m_dataToProduce.pop();
    wstring firstFeature = this->m_mapReaderOrder2FeatureName[0];
    workingMemory[firstFeature]->Clear();
    this->m_dataToConsume.push(workingMemory);
}
template <class ElemType>
bool CSparseDensePCReader<ElemType>::DataEnd() { return true; }

// GetLabelMapping - Gets the label mapping from integer index to label type
// returns - a map from numeric datatype to native label type
template <class ElemType>
const std::map<IDataReader::LabelIdType, IDataReader::LabelType>& CSparseDensePCReader<ElemType>::GetLabelMapping(const std::wstring& /*sectionName*/)
{
    return m_mapIdToLabel;
}

// SetLabelMapping - Sets the label mapping from integer index to label
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run
template <class ElemType>
void CSparseDensePCReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<IDataReader::LabelIdType, LabelType>& labelMapping)
{
    m_mapIdToLabel = labelMapping;
    m_mapLabelToId.clear();
    for (std::pair<unsigned, LabelType> var : labelMapping)
    {
        m_mapLabelToId[var.second] = var.first;
    }
}

// instantiate all the combinations we expect to be used
template class CSparseDensePCReader<double>;
template class CSparseDensePCReader<float>;
template class SparseDenseMemory<double>;
template class SparseDenseMemory<float>;
} } }
