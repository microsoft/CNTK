//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// DSSMReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS // creating the exports here
#include "DataReader.h"
#include "DSSMReader.h"
#ifdef LEAKDETECT
#include <vld.h> // leak detection
#endif
#include "fileutil.h" // for fexists()

namespace Microsoft { namespace MSR { namespace CNTK {

DWORD HIDWORD(size_t size)
{
    return size >> 32;
}
DWORD LODWORD(size_t size)
{
    return size & 0xFFFFFFFF;
}

std::string ws2s(const std::wstring& wstr)
{
    int size_needed = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), int(wstr.length() + 1), 0, 0, 0, 0);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), int(wstr.length() + 1), &strTo[0], size_needed, 0, 0);
    return strTo;
}

template <class ElemType>
size_t DSSMReader<ElemType>::RandomizeSweep(size_t mbStartSample)
{
    // size_t randomRangePerEpoch = (m_epochSize+m_randomizeRange-1)/m_randomizeRange;
    // return m_epoch*randomRangePerEpoch + epochSample/m_randomizeRange;
    return mbStartSample / m_randomizeRange;
}

// ReadLine - Read a line
// readSample - sample to read in global sample space
// returns - true if we successfully read a record, otherwise false
template <class ElemType>
bool DSSMReader<ElemType>::ReadRecord(size_t /*readSample*/)
{
    return false; // not used
}

// RecordsToRead - Determine number of records to read to populate record buffers
// mbStartSample - the starting sample from which to read
// tail - we are checking for possible remainer records to read (default false)
// returns - true if we have more to read, false if we hit the end of the dataset
template <class ElemType>
size_t DSSMReader<ElemType>::RecordsToRead(size_t mbStartSample, bool tail)
{
    assert(mbStartSample >= m_epochStartSample);
    // determine how far ahead we need to read
    bool randomize = Randomize();
    // need to read to the end of the next minibatch
    size_t epochSample = mbStartSample;
    epochSample %= m_epochSize;

    // determine number left to read for this epoch
    size_t numberToEpoch = m_epochSize - epochSample;
    // we will take either a minibatch or the number left in the epoch
    size_t numberToRead = min(numberToEpoch, m_mbSize);
    if (numberToRead == 0 && !tail)
        numberToRead = m_mbSize;

    if (randomize)
    {
        size_t randomizeSweep = RandomizeSweep(mbStartSample);
        // if first read or read takes us to another randomization range
        // we need to read at least randomization range records
        if (randomizeSweep != m_randomordering.CurrentSeed()) // the range has changed since last time
        {
            numberToRead = RoundUp(epochSample, m_randomizeRange) - epochSample;
            if (numberToRead == 0 && !tail)
                numberToRead = m_randomizeRange;
        }
    }
    return numberToRead;
}

template <class ElemType>
void DSSMReader<ElemType>::WriteLabelFile()
{
    // write out the label file if they don't have one
    if (!m_labelFileToWrite.empty())
    {
        if (m_mapIdToLabel.size() > 0)
        {
            File labelFile(m_labelFileToWrite, fileOptionsWrite | fileOptionsText);
            for (int i = 0; i < m_mapIdToLabel.size(); ++i)
            {
                labelFile << m_mapIdToLabel[i] << '\n';
            }
            fprintf(stderr, "label file %ls written to disk\n", m_labelFileToWrite.c_str());
            m_labelFileToWrite.clear();
        }
        else if (!m_cachingWriter)
        {
            fprintf(stderr, "WARNING: file %ls NOT written to disk yet, will be written the first time the end of the entire dataset is found.\n", m_labelFileToWrite.c_str());
        }
    }
}

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template <class ElemType>
void DSSMReader<ElemType>::Destroy()
{
    delete this;
}

// Init - Reader Initialize for multiple data sets
// config - [in] configuration parameters for the datareader
// Sample format below:
//# Parameter values for the reader
//reader=[
//  # reader to use
//  readerType=DSSMReader
//  miniBatchMode=Partial
//  randomize=None
//  features=[
//    dim=784
//    start=1
//    file=c:\speech\mnist\mnist_test.txt
//  ]
//  labels=[
//    dim=1
//      start=0
//      file=c:\speech\mnist\mnist_test.txt
//      labelMappingFile=c:\speech\mnist\labels.txt
//      labelDim=10
//      labelType=Category
//  ]
//]
template <class ElemType>
template <class ConfigRecordType>
void DSSMReader<ElemType>::InitFromConfig(const ConfigRecordType& readerConfig)
{
    std::vector<std::wstring> features;
    std::vector<std::wstring> labels;

    // Determine the names of the features and lables sections in the config file.
    // features - [in,out] a vector of feature name strings
    // labels - [in,out] a vector of label name strings
    // For DSSM dataset, we only need features. No label is necessary. The following "labels" just serves as a place holder
    GetFileConfigNames(readerConfig, features, labels);

    // For DSSM dataset, it must have exactly two features
    // In the config file, we must specify query features first, then document features. The sequence is different here. Pay attention
    if (features.size() == 2 && labels.size() == 1)
    {
        m_featuresNameQuery = features[1];
        m_featuresNameDoc = features[0];
        m_labelsName = labels[0];
    }
    else
    {
        RuntimeError("DSSM requires exactly two features and one label. Their names should match those in NDL definition");
        return;
    }

    m_mbStartSample = m_epoch = m_totalSamples = m_epochStartSample = 0;
    m_labelIdMax = m_labelDim = 0;
    m_partialMinibatch = m_endReached = false;
    m_labelType = labelCategory;
    m_readNextSample = 0;
    m_traceLevel = readerConfig(L"traceLevel", 0);

    if (readerConfig.Exists(L"randomize"))
    {
        // BUGBUG: reading out string and number... ugh
        wstring randomizeString = readerConfig(L"randomize");
        if (randomizeString == L"None")
        {
            m_randomizeRange = randomizeNone;
        }
        else if (randomizeString == L"Auto")
        {
            m_randomizeRange = randomizeAuto;
        }
        else
        {
            m_randomizeRange = readerConfig(L"randomize");
        }
    }
    else
    {
        m_randomizeRange = randomizeNone;
    }

    std::string minibatchMode(readerConfig(L"minibatchMode", "Partial"));
    m_partialMinibatch = EqualCI(minibatchMode, "Partial");

    // Get the config parameters for query feature and doc feature
    ConfigParameters configFeaturesQuery = readerConfig(m_featuresNameQuery, "");
    ConfigParameters configFeaturesDoc   = readerConfig(m_featuresNameDoc, "");

    if (configFeaturesQuery.size() == 0)
        RuntimeError("features file not found, required in configuration: i.e. 'features=[file=c:\\myfile.txt;start=1;dim=123]'");
    if (configFeaturesDoc.size() == 0)
        RuntimeError("features file not found, required in configuration: i.e. 'features=[file=c:\\myfile.txt;start=1;dim=123]'");

    // Read in feature size information
    // This information will be used to handle OOVs
    m_featuresDimQuery = configFeaturesQuery(L"dim");
    m_featuresDimDoc   = configFeaturesDoc(L"dim");

    std::wstring fileQ = configFeaturesQuery("file");
    std::wstring fileD = configFeaturesDoc("file");

    dssm_queryInput.Init(fileQ, m_featuresDimQuery);
    dssm_docInput.Init(fileD, m_featuresDimDoc);

    m_totalSamples = dssm_queryInput.numRows;
    if (read_order == NULL)
    {
        read_order = new int[m_totalSamples];
        for (int c = 0; c < m_totalSamples; c++)
        {
            read_order[c] = c;
        }
    }
    m_mbSize = 0;
}
// destructor - virtual so it gets called properly
template <class ElemType>
DSSMReader<ElemType>::~DSSMReader()
{
    ReleaseMemory();
}

// ReleaseMemory - release the memory footprint of DSSMReader
// used when the caching reader is taking over
template <class ElemType>
void DSSMReader<ElemType>::ReleaseMemory()
{
    if (m_qfeaturesBuffer != NULL)
        delete[] m_qfeaturesBuffer;
    m_qfeaturesBuffer = NULL;
    if (m_dfeaturesBuffer != NULL)
        delete[] m_dfeaturesBuffer;
    m_dfeaturesBuffer = NULL;
    if (m_labelsBuffer != NULL)
        delete[] m_labelsBuffer;
    m_labelsBuffer = NULL;
    if (m_labelsIdBuffer != NULL)
        delete[] m_labelsIdBuffer;
    m_labelsIdBuffer = NULL;
    m_featureData.clear();
    m_labelIdData.clear();
    m_labelData.clear();
}

//SetupEpoch - Setup the proper position in the file, and other variable settings to start a particular epoch
template <class ElemType>
void DSSMReader<ElemType>::SetupEpoch()
{
}

// utility function to round an integer up to a multiple of size
size_t RoundUp(size_t value, size_t size)
{
    return ((value + size - 1) / size) * size;
}

//StartMinibatchLoop - Startup a minibatch loop
// mbSize - [in] size of the minibatch (number of Samples, etc.)
// epoch - [in] epoch number for this loop, if > 0 the requestedEpochSamples must be specified (unless epoch zero was completed this run)
// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
//   this value must be a multiple of mbSize, if it is not, it will be rounded up to one.
template <class ElemType>
void DSSMReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    size_t mbStartSample = m_epoch * m_epochSize;
    if (m_totalSamples == 0)
    {
        m_totalSamples = dssm_queryInput.numRows;
    }

    size_t fileRecord = m_totalSamples ? mbStartSample % m_totalSamples : 0;
    fprintf(stderr, "starting epoch %lld at record count %lld, and file position %lld\n", m_epoch, mbStartSample, fileRecord);

    // reset the next read sample
    m_readNextSample = 0;
    m_epochStartSample = m_mbStartSample = mbStartSample;
    m_mbSize = mbSize;
    m_epochSize = requestedEpochSamples;
    dssm_queryInput.SetupEpoch(mbSize);
    dssm_docInput.SetupEpoch(mbSize);
    if (m_epochSize > (size_t) dssm_queryInput.numRows)
    {
        m_epochSize = (size_t) dssm_queryInput.numRows;
    }
    if (Randomize())
    {
        random_shuffle(&read_order[0], &read_order[m_epochSize]);
    }
    m_epoch = epoch;
    m_mbStartSample = epoch * m_epochSize;
}

// function to store the LabelType in an ElemType
// required for string labels, which can't be stored in ElemType arrays
template <class ElemType>
void DSSMReader<ElemType>::StoreLabel(ElemType& labelStore, const LabelType& labelValue)
{
    labelStore = (ElemType) m_mapLabelToId[labelValue];
}

// GetMinibatch - Get the next minibatch (features and labels)
// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponding matrix,
//             [out] each matrix resized if necessary containing data.
// returns - true if there are more minibatches, false if no more minibatchs remain
template <class ElemType>
bool DSSMReader<ElemType>::TryGetMinibatch(StreamMinibatchInputs& matrices)
{
    if (m_readNextSample >= m_totalSamples)
    {
        return false;
    }
    // In my unit test example, the input matrices contain 5: N, S, fD, fQ and labels
    // Both N and S serve as a pre-set constant values, no need to change them
    // In this node, we only need to fill in these matrices: fD, fQ, labels
    Matrix<ElemType>& featuresQ = matrices.GetInputMatrix<ElemType>(m_featuresNameQuery);
    Matrix<ElemType>& featuresD = matrices.GetInputMatrix<ElemType>(m_featuresNameDoc);
    Matrix<ElemType>& labels    = matrices.GetInputMatrix<ElemType>(m_labelsName); // will change this part later.  TODO: How?

    size_t actualMBSize = (m_readNextSample + m_mbSize > m_totalSamples) ? m_totalSamples - m_readNextSample : m_mbSize;

    featuresQ.SwitchToMatrixType(MatrixType::SPARSE, MatrixFormat::matrixFormatSparseCSC, false);
    featuresD.SwitchToMatrixType(MatrixType::SPARSE, MatrixFormat::matrixFormatSparseCSC, false);

    /*
    featuresQ.Resize(dssm_queryInput.numRows, actualMBSize);
    featuresD.Resize(dssm_docInput.numRows, actualMBSize);
    */

    // fprintf(stderr, "featuresQ\n");
    dssm_queryInput.Next_Batch(featuresQ, m_readNextSample, actualMBSize, read_order);
    // fprintf(stderr, "\n\n\nfeaturesD\n");
    dssm_docInput.Next_Batch(featuresD, m_readNextSample, actualMBSize, read_order);
    // fprintf(stderr, "\n\n\n\n\n");
    m_readNextSample += actualMBSize;
    /*
                featuresQ.Print("featuresQ");
                fprintf(stderr, "\n");
                featuresD.Print("featuresD");
                fprintf(stderr, "\n");
                */

    /*
    GPUSPARSE_INDEX_TYPE* h_CSCCol;
    GPUSPARSE_INDEX_TYPE* h_Row;
    ElemType* h_val;
    size_t nz;
    size_t nrs;
    size_t ncols;
    featuresQ.GetMatrixFromCSCFormat(&h_CSCCol, &h_Row, &h_val, &nz, &nrs, &ncols);

    for (int j = 0, k=0; j < nz; j++)
    {
        if (h_CSCCol[k] >= j)
        {
            fprintf(stderr, "\n");
            k++;
        }
        fprintf(stderr, "%d:%.f ", h_Row[j], h_val[j]);

    }
    */

    /*
    featuresQ.TransferFromDeviceToDevice(featuresQ.GetDeviceId(), -1);
    featuresQ.SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense);
    featuresQ.Print("featuresQ");

    featuresD.TransferFromDeviceToDevice(featuresD.GetDeviceId(), -1);
    featuresD.SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense);
    featuresD.Print("featuresD");

    exit(1);
    */

    if (actualMBSize > m_mbSize || m_labelsBuffer == NULL)
    {
        size_t rows = labels.GetNumRows();
        m_labelsBuffer = new ElemType[rows * actualMBSize];
        memset(m_labelsBuffer, 0, sizeof(ElemType) * rows * actualMBSize);
        for (int i = 0; i < actualMBSize; i++)
        {
            m_labelsBuffer[i * rows] = 1;
        }
    }
    if (actualMBSize != labels.GetNumCols())
    {
        size_t rows = labels.GetNumRows();
        labels.Resize(rows, actualMBSize);
        labels.SetValue(0.0);
        labels.SetValue(rows, actualMBSize, labels.GetDeviceId(), m_labelsBuffer, 0);
    }
    /*
    featuresQ.Print("featuresQ");
    featuresD.Print("featuresD");
    labels.print("labels");
    */

    return true;
}

// GetLabelMapping - Gets the label mapping from integer index to label type
// returns - a map from numeric datatype to native label type
template <class ElemType>
const std::map<IDataReader::LabelIdType, IDataReader::LabelType>& DSSMReader<ElemType>::GetLabelMapping(const std::wstring& sectionName)
{
    if (m_cachingReader)
    {
        return m_cachingReader->GetLabelMapping(sectionName);
    }
    return m_mapIdToLabel;
}

// SetLabelMapping - Sets the label mapping from integer index to label
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run
template <class ElemType>
void DSSMReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<LabelIdType, LabelType>& labelMapping)
{
    if (m_cachingReader)
    {
        RuntimeError("Cannot set mapping table when the caching reader is being used");
    }
    m_mapIdToLabel = labelMapping;
    m_mapLabelToId.clear();
    for (std::pair<unsigned, LabelType> var : labelMapping)
    {
        m_mapLabelToId[var.second] = var.first;
    }
}

template <class ElemType>
bool DSSMReader<ElemType>::DataEnd() { return true; }

template <class ElemType>
DSSM_BinaryInput<ElemType>::DSSM_BinaryInput()
{
}
template <class ElemType>
DSSM_BinaryInput<ElemType>::~DSSM_BinaryInput()
{
    Dispose();
}
template <class ElemType>
void DSSM_BinaryInput<ElemType>::Init(wstring fileName, size_t dim)
{

    m_dim = dim;
    mbSize = 0;
    /*
    m_hndl = CreateFileA(fileName.c_str(), GENERIC_READ,
        FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    */
    m_hndl = CreateFile(fileName.c_str(), GENERIC_READ,
                        FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (m_hndl == INVALID_HANDLE_VALUE)
    {
        char message[256];
        sprintf_s(message, "Unable to Open/Create file %ls, error %x", fileName.c_str(), GetLastError());
        RuntimeError(message);
    }

    m_filemap = CreateFileMapping(m_hndl, NULL, PAGE_READONLY, 0, 0, NULL);

    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    DWORD sysGran = sysinfo.dwAllocationGranularity;

    header_buffer = MapViewOfFile(m_filemap,     // handle to map object
                                  FILE_MAP_READ, // get correct permissions
                                  HIDWORD(0),
                                  LODWORD(0),
                                  sizeof(int64_t) * 2 + sizeof(int32_t));

    // cout << "After mapviewoffile" << endl;

    memcpy(&numRows, header_buffer, sizeof(int64_t));
    memcpy(&numCols, (char*) header_buffer + sizeof(int64_t), sizeof(int32_t));
    memcpy(&totalNNz, (char*) header_buffer + sizeof(int64_t) + sizeof(int32_t), sizeof(int64_t));

    // cout << "After gotvalues" << endl;
    int64_t base_offset = sizeof(int64_t) * 2 + sizeof(int32_t);

    int64_t offsets_padding = base_offset % sysGran;
    base_offset -= offsets_padding;

    int64_t header_size = numRows * sizeof(int64_t) + offsets_padding;

    void* offsets_orig = MapViewOfFile(m_filemap,     // handle to map object
                                       FILE_MAP_READ, // get correct permissions
                                       HIDWORD(base_offset),
                                       LODWORD(base_offset),
                                       header_size);

    offsets_buffer = (char*) offsets_orig + offsets_padding;

    if (offsets != NULL)
    {
        free(offsets);
    }
    offsets = (int64_t*) malloc(sizeof(int64_t) * numRows);
    memcpy(offsets, offsets_buffer, numRows * sizeof(int64_t));

    int64_t header_offset = base_offset + offsets_padding + numRows * sizeof(int64_t);

    int64_t data_padding = header_offset % sysGran;
    header_offset -= data_padding;

    void* data_orig = MapViewOfFile(m_filemap,     // handle to map object
                                    FILE_MAP_READ, // get correct permissions
                                    HIDWORD(header_offset),
                                    LODWORD(header_offset),
                                    0);
    data_buffer = (char*) data_orig + data_padding;
}
template <class ElemType>
bool DSSM_BinaryInput<ElemType>::SetupEpoch(size_t minibatchSize)
{
    if (values == NULL || mbSize < minibatchSize)
    {
        if (values != NULL)
        {
            free(values);
            free(colIndices);
            free(rowIndices);
        }

        values = (ElemType*) malloc(sizeof(ElemType) * MAX_BUFFER * minibatchSize);
        colIndices = (int32_t*) malloc(sizeof(int32_t) * (minibatchSize + 1));
        rowIndices = (int32_t*) malloc(sizeof(int32_t) * MAX_BUFFER * minibatchSize);
        // fprintf(stderr, "values  size: %d",sizeof(ElemType)*MAX_BUFFER*minibatchSize);
        // fprintf(stderr, "colindi size: %d",sizeof(int32_t)*MAX_BUFFER*(1+minibatchSize));
        // fprintf(stderr, "rowindi size: %d",sizeof(int32_t)*MAX_BUFFER*minibatchSize);
    }
    if (minibatchSize > mbSize)
    {
        mbSize = minibatchSize;
    }

    return true;
}
template <class ElemType>
bool DSSM_BinaryInput<ElemType>::Next_Batch(Matrix<ElemType>& matrices, size_t cur, size_t numToRead, int* /*ordering*/)
{
    /*
    int devId = matrices.GetDeviceId();
    matrices.TransferFromDeviceToDevice(devId, -1);
    */

    int32_t cur_index = 0;

    for (int c = 0; c < numToRead; c++, cur++)
    {
        // int64_t cur_offset = offsets[ordering[cur]];
        int64_t cur_offset = offsets[cur];
        // int64_t cur_offset = offsets[ordering[c]];
        // int32_t nnz;
        colIndices[c] = cur_index;
        int32_t nnz = *(int32_t*) ((char*) data_buffer + cur_offset);
        // memcpy(&nnz, (char*)data_buffer + cur_offset, sizeof(int32_t));
        memcpy(values + cur_index, (char*) data_buffer + cur_offset + sizeof(int32_t), sizeof(ElemType) * nnz);
        memcpy(rowIndices + cur_index, (char*) data_buffer + cur_offset + sizeof(int32_t) + sizeof(ElemType) * nnz, sizeof(int32_t) * nnz);
        /**
        fprintf(stderr, "%4d (%3d, %6d): ", c, nnz, cur_index + nnz);
        for (int i = 0; i < nnz; i++)
        {
            fprintf(stderr, "%d:%.f ", rowIndices[cur_index+i], values[cur_index+i]);
            // matrices.SetValue(rowIndices[cur_index + i], c, values[cur_index + i]);
        }
        fprintf(stderr, "\n");
        **/

        cur_index += nnz;
    }
    colIndices[numToRead] = cur_index;
    /*
    int col = 0;
    for (int c = 0; c < cur_index; c++)
    {
        if (colIndices[col] == c)
        {
            fprintf(stderr, "\n%4d: ", col);
            col++;
        }
        fprintf(stderr, "%d:%.f ", rowIndices[c], values[c]);
    }
    */
    /*
    fprintf(stderr, "\nXXXX nnz: %d\n", cur_index);
    fprintf(stderr, "XXXX max values read: %d vs %d\n", sizeof(ElemType)*cur_index, sizeof(ElemType)*MAX_BUFFER*numToRead);
    fprintf(stderr, "XXXX max indices read: %d vs %d\n", sizeof(int32_t)*cur_index, sizeof(int32_t)*MAX_BUFFER*numToRead);
    fprintf(stderr, "XXXX sizeof(int32_t) = %d, sizeof(int) = %d\n", sizeof(int32_t), sizeof(int));
    */
    /*
        values = (ElemType*)malloc(sizeof(ElemType)*MAX_BUFFER*minibatchSize);
        colIndices = (int32_t*)malloc(sizeof(int32_t)*MAX_BUFFER*(minibatchSize+1));
        rowIndices = (int32_t*)malloc(sizeof(int32_t)*MAX_BUFFER*minibatchSize);
        */

    matrices.SetMatrixFromCSCFormat(colIndices, rowIndices, values, cur_index, m_dim, numToRead);
    // matrices.Print("actual values");
    // exit(1);
    /*
    matrices.SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense);
    matrices.Print("featuresQ");
    exit(1);
    matrices.TransferFromDeviceToDevice(-1,devId);
    */
    return true;
}

template <class ElemType>
void DSSM_BinaryInput<ElemType>::Dispose()
{
    if (offsets_orig != NULL)
    {
        UnmapViewOfFile(offsets_orig);
    }
    if (data_orig != NULL)
    {
        UnmapViewOfFile(data_orig);
    }

    if (offsets != NULL)
    {
        free(offsets); // = (ElemType*)malloc(sizeof(float)* 230 * 1024);
    }
    if (values != NULL)
    {
        free(values); // = (ElemType*)malloc(sizeof(float)* 230 * 1024);
    }
    if (rowIndices != NULL)
    {
        free(rowIndices); // = (int*)malloc(sizeof(float)* 230 * 1024);
    }
    if (colIndices != NULL)
    {
        free(colIndices); // = (int*)malloc(sizeof(float)* 230 * 1024);
    }
}

template <class ElemType>
bool DSSMReader<ElemType>::GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart)
{
    if (m_cachingReader)
    {
        return m_cachingReader->GetData(sectionName, numRecords, data, dataBufferSize, recordStart);
    }
    RuntimeError("GetData not supported in DSSMReader");
}

// instantiate all the combinations we expect to be used
template class DSSMReader<double>;
template class DSSMReader<float>;
} } }
