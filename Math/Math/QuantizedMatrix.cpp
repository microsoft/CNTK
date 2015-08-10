#include "stdafx.h"
#include "QuantizedMatrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    
    template<class ElemType>
    QuantizedMatrix<ElemType>::QuantizedMatrix(const size_t numRows, const size_t numCols, const size_t nbits, short deviceId, MemAllocator* allocator /* = nullptr */)
        : m_numRows(numRows), m_numCols(numCols), m_numBits(nbits), m_allocator(allocator)
    {
        m_qColSize = QuantizedColumn<ElemType>::QuantizedColumnSize(m_numBits, m_numRows);
        if (((QWordNumBits / m_numBits) * m_numBits) != QWordNumBits)
        {
            throw std::logic_error("Quantization: 'nbits' must be a divisor of 64");
        }
        
        if (m_allocator == nullptr)
        {
            m_quantizedData = new Matrix<char>(m_qColSize, m_numCols, deviceId);
        }
        else
        {
            m_quantizedData = new Matrix<char>(m_qColSize, m_numCols, m_allocator->Malloc(m_qColSize * m_numCols), matrixFlagDontOwnBuffer, deviceId);
        }
    }

    template<class ElemType>
    QuantizedMatrix<ElemType>::QuantizedMatrix(QuantizedMatrix<ElemType>&& moveFrom)
        : m_quantizedData(moveFrom.m_quantizedData), m_allocator(moveFrom.m_allocator),
        m_numRows(moveFrom.m_numRows), m_numCols(moveFrom.m_numCols),
        m_numBits(moveFrom.m_numBits), m_qColSize(moveFrom.m_qColSize)
    {
        moveFrom.m_quantizedData = nullptr;
        moveFrom.m_allocator = nullptr;
    }

    template<class ElemType>
    QuantizedMatrix<ElemType>& QuantizedMatrix<ElemType>::operator=(QuantizedMatrix<ElemType>&& moveFrom)
    {
        assert(this != &moveFrom);

        this->m_quantizedData = moveFrom.m_quantizedData;
        this->m_allocator = moveFrom.m_allocator;
        this->m_numRows = moveFrom.m_numRows;
        this->m_numCols = moveFrom.m_numCols;
        this->m_numBits = moveFrom.m_numBits;
        this->m_qColSize = moveFrom.m_qColSize;

        moveFrom.m_quantizedData = nullptr;
        moveFrom.m_allocator = nullptr;

        return *this;
    }

    template<class ElemType>
    QuantizedMatrix<ElemType>::QuantizedMatrix(const size_t numRows, const size_t numCols, const size_t nbits, Matrix<char>* data)
        : m_numRows(numRows), m_numCols(numCols), m_numBits(nbits), m_quantizedData(data), m_allocator(nullptr)
    {
        m_qColSize = QuantizedColumn<ElemType>::QuantizedColumnSize(m_numBits, m_numRows);
        if (((QWordNumBits / m_numBits) * m_numBits) != QWordNumBits)
        {
            throw std::logic_error("Quantization: 'nbits' must be a divisor of 64");
        }

        // Make sure that the data matrix has enough space
        assert((m_quantizedData->GetNumRows() == m_qColSize) && (m_quantizedData->GetNumCols() >= numCols));
    }

    template<class ElemType>
    QuantizedMatrix<ElemType>::~QuantizedMatrix()
    {
        if (nullptr != m_quantizedData)
        {
            // If we used an external allocator, lets free the backing buffer of the matrix
            if (m_allocator != nullptr)
            {
                assert(!m_quantizedData->OwnBuffer());
                m_allocator->Free(m_quantizedData->BufferPointer());
            }

            delete m_quantizedData;
            m_quantizedData = nullptr;
        }
    }
    
    template<class ElemType>
    int QuantizedMatrix<ElemType>::GetDeviceId() const
    {
        return m_quantizedData->GetDeviceId();
    }    
    
    template<class ElemType>
    size_t QuantizedMatrix<ElemType>::GetSize() const 
    {
        return m_quantizedData->GetNumElements();
    }
    
    template<class ElemType>
    char* QuantizedMatrix<ElemType>::GetArray() const
    {
        return m_quantizedData->BufferPointer();
    }
    
    template<class ElemType>
    QuantizedMatrix<ElemType> QuantizedMatrix<ElemType>::ColumnSlice(size_t startColumn, size_t numCols) const
    {
        auto matrixSliceData = new Matrix<char>(m_quantizedData->ColumnSlice(startColumn, numCols));
        return QuantizedMatrix<ElemType>(this->GetNumRows(), numCols, this->GetNumBits(), matrixSliceData);
    }
    
    // Explicit instantiation
    template class QuantizedMatrix<float>;
    template class QuantizedMatrix<double>;    

}}}
