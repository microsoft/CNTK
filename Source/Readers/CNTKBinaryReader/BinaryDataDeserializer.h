//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializerBase.h"
#include "BinaryConfigHelper.h"
#include "CorpusDescriptor.h"
#include "BinaryDataChunk.h"
#include "FileHelper.h"

namespace Microsoft { namespace MSR { namespace CNTK {


class BinaryDataDeserialzer {
public:
    virtual size_t GetSequenceDataForChunk(size_t numSequences, void* data, std::vector<SequenceDataPtr>& result) = 0;

    StorageType GetStorageType() { return m_storageType; }
    ElementType GetElementType() { return m_elemType; }
    TensorShapePtr GetSampleLayout() { return make_shared<TensorShape>(m_numCols); }
    virtual bool IsSequence() { return false; }

    size_t GetElemSizeBytes()
    {
        if (m_elemType == ElementType::tfloat)
            return sizeof(float);
        else if (m_elemType == ElementType::tdouble)
            return sizeof(double);
        else
            LogicError("Error, elemtype is not defined for BinaryDataDeserializer.");
    }

protected:
    struct DenseInputStreamBuffer : DenseSequenceData
    {
        // capacity = expected number of samples * sample size
        const void* GetDataBuffer() override
        {
            return m_data;
        }

        void* m_data;
    };

    // In case of sparse input, we also need a vector of
    // indices (one index for each input value) and a vector
    // of NNZ counts (one for each sample).
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
        
        std::vector<IndexType> m_indicesBuffer;
        void* m_data;
    };

    
protected:
    StorageType m_storageType;
    ElementType m_elemType;
    size_t m_numCols;

};

typedef shared_ptr<BinaryDataDeserialzer> BinaryDataDeserializerPtr;
    
class DenseBinaryDataDeserializer : public BinaryDataDeserialzer
{
public:
    DenseBinaryDataDeserializer(FILE* infile)
    {
        // We don't have to read the storage type. We know we're dense
        m_storageType = StorageType::dense;

        // Read the element type, note it's stored as an int32
        int32_t elemType;
        CNTKBinaryFileHelper::readOrDie(&elemType, sizeof(elemType), 1, infile);
        if (elemType == 0)
            m_elemType = ElementType::tfloat;
        else if (elemType == 1)
            m_elemType = ElementType::tdouble;
        else
            RuntimeError("Unsupported element type %d.", elemType);

        // Read the number of columns
        int32_t numCols;
        CNTKBinaryFileHelper::readOrDie(&numCols, sizeof(numCols), 1, infile);
        m_numCols = numCols;
    }

    size_t GetSequenceDataForChunk(size_t numSequences, void* data, std::vector<SequenceDataPtr>& result)
    {
        size_t elemSize = GetElemSizeBytes();
        result.resize(numSequences);
        for (size_t c = 0; c < numSequences; c++)
        {
            shared_ptr<DenseInputStreamBuffer> sequence = make_shared<DenseInputStreamBuffer>();
            sequence->m_data            = (char*)data + c*m_numCols*elemSize;
            sequence->m_numberOfSamples = 1;
            sequence->m_sampleLayout    = std::make_shared<TensorShape>(m_numCols);
            result[c]                   = sequence;
        }

        // For dense, the number of bytes processed is just numRows * numCols * elemSize;
        return numSequences * m_numCols * elemSize;
    }

};

class SparseBinaryDataDeserializer : public BinaryDataDeserialzer
{
public:
    SparseBinaryDataDeserializer(FILE* infile)
    {
        // Read the storage type. Currently we only support sparse_csc, 
        // but for future compatability allow it to be a parameter.
        int32_t storageType;
        CNTKBinaryFileHelper::readOrDie(&storageType, sizeof(storageType), 1, infile);
        if (storageType == 0)
            m_storageType = StorageType::sparse_csc;
        else
            RuntimeError("Unsupported storage type %d.", storageType);

        // Read the element type, note it's stored as an int32
        int32_t elemType;
        CNTKBinaryFileHelper::readOrDie(&elemType, sizeof(elemType), 1, infile);
        if (elemType== 0)
            m_elemType = ElementType::tfloat;
        else if (elemType == 1)
            m_elemType = ElementType::tdouble;
        else
            RuntimeError("Unsupported element type %d.", elemType);

        int32_t isSequence;
        CNTKBinaryFileHelper::readOrDie(&isSequence, sizeof(isSequence), 1, infile);
        if (isSequence == 0)
            m_isSequence = false;
        else if (isSequence == 1)
            m_isSequence = true;
        else
            RuntimeError("Unsupported sequence type %d.", isSequence);

        // Read the number of columns
        int32_t numCols;
        CNTKBinaryFileHelper::readOrDie(&numCols, sizeof(numCols), 1, infile);
        m_numCols = numCols;
    }
    
    bool IsSequence() override { return m_isSequence; }

    // The format of data is: 
    // int32_t: nnz for the entire chunk
    // ElemType[nnz]: the values for the sparse sequences
    // int32_t[nnz]: the row offsets for the sparse sequences
    // int32_t[numSequences]: the column offsets for the sparse sequences
    size_t GetSequenceDataForChunk(size_t numSequences, void* data, std::vector<SequenceDataPtr>& result)
    {
        size_t elemSize = GetElemSizeBytes();
        result.resize(numSequences);

        // For sparse, the first int32_t is the number of nnz values in the entire set of sequences
        int32_t totalNNz = *(int32_t*)data;

        // the rest of this chunk
        // Since we're not templating on ElemType, we use void for the values. Note that this is the only place
        // this deserializer uses ElemType, the rest are int32_t for this deserializer.
        void* values = (char*)data + sizeof(int32_t);

        // Now the row offsets
        int32_t* rowOffsets = (int32_t*)((char*)values + elemSize * totalNNz);

        // Now the col offsets
        int32_t* colOffsets = rowOffsets + totalNNz;

        // Now we setup some helper members to process the chunk
        for (size_t colIndex = 0; colIndex < numSequences; colIndex++)
        {
            shared_ptr<SparseInputStreamBuffer> sequence = make_shared<SparseInputStreamBuffer>();
            // We can't popuplate sequence->m_chunk here, so delay that for later

            // We know the number of elements in all of the samples, it's just this:
            sequence->m_totalNnzCount = colOffsets[colIndex + 1] - colOffsets[colIndex];

            // The values array is already properly packed, so just use it.
            sequence->m_data = values;
            
            // The indices are correct (note they MUST BE IN INCREASING ORDER), but we will have to fix them up a 
            // little bit, for now just use them
            sequence->m_indices = rowOffsets;
            for (int32_t curRow = 0; curRow < sequence->m_totalNnzCount; curRow++)
            {
                // Get the sample for the current index
                size_t sampleNum = rowOffsets[curRow] / m_numCols;
                // The current sample might be OOB, if so, fill in the the missing ones.
                while(sequence->m_nnzCounts.size() < sampleNum+1)
                    sequence->m_nnzCounts.push_back(0);
                // Now that we have enough samples, increment the nnz for the sample
                sequence->m_nnzCounts[sampleNum] += 1;
                // Now that we've found it's sample, fix up the index.
                rowOffsets[curRow] %= m_numCols;
            }
            sequence->m_numberOfSamples = (uint32_t)sequence->m_nnzCounts.size();
            // update values, rowOffsets pointers
            values = (char*)values + sequence->m_totalNnzCount * elemSize;
            rowOffsets += sequence->m_totalNnzCount;

            result[colIndex] = sequence;
        }

        // For sparse, we compute how many bytes we processed
        // From the header to this function, we see that is:
        // sizeof(int32_t) + totalNNz * sizeof(ElemType) + totalNNz * sizeof(int32_t) + numSequences * sizeof(int32_t)
        return sizeof(int32_t) + totalNNz * (elemSize + sizeof(int32_t)) + (numSequences + 1) * sizeof(int32_t);
    }

private:
    bool m_isSequence;


};

    
}}}
