//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializerBase.h"
#include "BinaryConfigHelper.h"
#include "BinaryDataChunk.h"
#include "FileHelper.h"
#include "Reader.h"


namespace CNTK {


class BinaryDataDeserializer
{
public:

    BinaryDataDeserializer(FILE* file, DataType precision = DataType::Float)
    {
        ReadName(file);
        ReadDataType(file);
        ReadSampleSize(file);

        if (precision != DataType::Float && precision != DataType::Double)
            LogicError("Unsupported precision type %u.", (unsigned int)precision);

        if ((m_dataType == ReaderDataType::tfloat && precision != DataType::Float) ||
            (m_dataType == ReaderDataType::tdouble && precision != DataType::Double))
            LogicError("Unsupported combination of the input data type %u and precision %u. "
                "At the moment, both have to match.", (unsigned int)m_dataType, (unsigned int)precision);

        m_precision = precision;
    }

    virtual size_t GetSequenceDataForChunk(size_t numSequences, void* data, std::vector<SequenceDataPtr>& result) = 0;

    virtual StorageFormat GetStorageFormat() = 0;

    StreamInformation GetStreamDescription() 
    {
        StreamInformation streamDescription;
        streamDescription.m_elementType = m_precision;
        streamDescription.m_storageFormat = GetStorageFormat();
        streamDescription.m_sampleLayout = GetSampleShape();
        streamDescription.m_name = m_name;
        return streamDescription;
    }

    NDShape GetSampleShape()
    {
        return  NDShape({ m_sampleDimension });
    }

    size_t SizeOfDataType()
    {
        if (m_dataType == ReaderDataType::tfloat)
            return sizeof(float);
        if (m_dataType == ReaderDataType::tdouble)
            return sizeof(double);
        
        LogicError("Unsupported input data type %u.", (unsigned int)m_dataType);
    }

protected:

    enum class ReaderDataType : unsigned char
    {
        tfloat = 0,
        tdouble = 1,
        // TODO: 
        // tbool = 2, 1 bit per value (one-hot data)
        // tbyte = 3, 1 byte per value
    };

    virtual ~BinaryDataDeserializer() = default;

    void ReadName(FILE* file)
    {
        uint32_t len;
        // read the name
        CNTKBinaryFileHelper::ReadOrDie(&len, sizeof(len), 1, file);
        vector<char> temp(len + 1 , '\0');
        CNTKBinaryFileHelper::ReadOrDie(temp.data(), sizeof(char), len, file);
        m_name = msra::strfun::utf16(temp.data());
    }

    void ReadDataType(FILE* file)
    {
        CNTKBinaryFileHelper::ReadOrDie(&m_dataType, sizeof(m_dataType), 1, file);
        if (m_dataType> ReaderDataType::tdouble)
            RuntimeError("Unsupported input data type %u.", (unsigned int)m_dataType);
    }

    void ReadSampleSize(FILE* file)
    {
        CNTKBinaryFileHelper::ReadOrDie(&m_sampleDimension, sizeof(m_sampleDimension), 1, file);
    }

    struct DenseInputStreamBuffer : DenseSequenceData
    {
        const void* GetDataBuffer() override
        {
            return m_data;
        }

        const NDShape& GetSampleShape() override
        {
            return m_sampleShape;
        }

        void* m_data;
        DataType m_dataType;
        NDShape m_sampleShape;
    };

    struct SparseInputStreamBuffer : SparseSequenceData
    {
        SparseInputStreamBuffer()
        {
            m_totalNnzCount = 0;
        }

        const void* GetDataBuffer() override
        {
            return m_data;
        }

        const NDShape& GetSampleShape() override
        {
            return m_sampleShape;
        }

        void* m_data;
        NDShape m_sampleShape;
    };

    DataType m_precision;
    ReaderDataType m_dataType;
    uint32_t m_sampleDimension;
    wstring m_name;
};

typedef shared_ptr<BinaryDataDeserializer> BinaryDataDeserializerPtr;
    
class DenseBinaryDataDeserializer : public BinaryDataDeserializer
{
public:
    using BinaryDataDeserializer::BinaryDataDeserializer;

    virtual  StorageFormat GetStorageFormat() override { return StorageFormat::Dense; }

    size_t GetSequenceDataForChunk(size_t numSequences, void* data, std::vector<SequenceDataPtr>& result)
    {
        size_t valueSize = SizeOfDataType();
        result.resize(numSequences);
        size_t offset = 0;
        for (size_t i = 0; i < numSequences; i++)
        {
            shared_ptr<DenseInputStreamBuffer> sequenceDataPtr = make_shared<DenseInputStreamBuffer>();
            sequenceDataPtr->m_numberOfSamples = *(uint32_t*)((char*)data + offset);
            offset += sizeof(uint32_t);
            sequenceDataPtr->m_data = (char*)data + offset;
            sequenceDataPtr->m_sampleShape = GetSampleShape();
            sequenceDataPtr->m_elementType = m_precision;
            result[i]  = sequenceDataPtr;
            offset += m_sampleDimension * valueSize * sequenceDataPtr->m_numberOfSamples;
        }

        return offset;
    }
};

class SparseBinaryDataDeserializer : public BinaryDataDeserializer
{
public:
    SparseBinaryDataDeserializer(FILE* file, DataType precision = DataType::Float)
        :BinaryDataDeserializer(file, precision)
    {
        if (IndexType(m_sampleDimension) < 0)
        {
            RuntimeError("Sample dimension is too large for an IndexType value.");
        }
    }

    virtual  StorageFormat GetStorageFormat() override { return StorageFormat::SparseCSC; }

    // The format of data is: 
    // sequence[numSequences], where each sequence consists of:
    //   uint32_t: numSamples
    //   uint32_t: nnz for the sequence
    //   ElemType[nnz]: the values for the sparse sequences
    //   int32_t[nnz]: the row offsets for the sparse sequences
    //   int32_t[numSamples]: sizes (nnz counts) for each sample in the sequence
    size_t GetSequenceDataForChunk(size_t numSequences, void* data, std::vector<SequenceDataPtr>& result)
    {
        size_t offset = 0;
        result.resize(numSequences);
        for (size_t i = 0; i < numSequences; i++)
        {
            shared_ptr<SparseInputStreamBuffer> sequenceDataPtr = make_shared<SparseInputStreamBuffer>();
            offset += GetSequenceData((char*)data + offset, sequenceDataPtr);
            sequenceDataPtr->m_sampleShape = GetSampleShape();
            sequenceDataPtr->m_elementType = m_precision;
            result[i] = sequenceDataPtr;
        }

        return offset;
    }

    size_t GetSequenceData(void* data, shared_ptr<SparseInputStreamBuffer>& sequence)
    {
        size_t valueSize = SizeOfDataType();
        size_t offset = 0;

        // The very first value in the buffer is the number of samples in this sequence.
        sequence->m_numberOfSamples = *(uint32_t*)data;
        offset += sizeof(uint32_t);

        // Next is the total number of elements in all of the samples.
        uint32_t nnz = *(uint32_t*)((char*)data + offset);
        if (IndexType(nnz) < 0) 
        {
            RuntimeError("NNZ count is too large for an IndexType value.");
        }
        sequence->m_totalNnzCount = nnz;
        offset += sizeof(uint32_t);

        // the rest of this sequence
        // Since we're not templating on ElemType, we use void for the values. Note that this is the only place
        // this deserializer uses ElemType, the rest are int32_t for this deserializer.
        // The data is already properly packed, so just use it.
        sequence->m_data = (char*)data + offset;
        offset += valueSize * sequence->m_totalNnzCount;

        // The indices are supposed to be correctly packed (i.e., in increasing order)
        sequence->m_indices = (int32_t*)((char*)data + offset);
        offset += sizeof(int32_t) * sequence->m_totalNnzCount;
        
        int32_t* begin = (int32_t*)((char*)data + offset);
        offset += sizeof(int32_t) * sequence->m_numberOfSamples;
        int32_t* end = (int32_t*)((char*)data + offset);
        
        sequence->m_nnzCounts.reserve(sequence->m_numberOfSamples);
        sequence->m_nnzCounts.assign(begin, end);

        return offset;
    }
};

    
}
