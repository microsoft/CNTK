#include "stdafx.h"
#include "QuantizedMatrix.h"
#include "ColumnQuantizer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
QuantizedMatrix<ElemType>::QuantizedMatrix(const size_t numRows, const size_t numCols, const size_t nbits, DEVICEID_TYPE deviceId, MemAllocator* allocator /* = nullptr */)
    : m_numRows(numRows), m_numCols(numCols), m_numBits(nbits), m_allocator(allocator)
{
    m_qColSize = QuantizedColumn<ElemType>::QuantizedColumnSize(m_numBits, m_numRows);
    if (((QWordNumBits / m_numBits) * m_numBits) != QWordNumBits)
    {
        LogicError("Quantization: 'nbits' must be a divisor of 64");
    }

    if (m_allocator == nullptr)
    {
        m_quantizedData = new Matrix<char>(m_qColSize, m_numCols, deviceId);
    }
    else
    {
        m_quantizedData = new Matrix<char>(m_qColSize, m_numCols, (char*)m_allocator->Malloc(m_qColSize * m_numCols), deviceId, matrixFlagDontOwnBuffer);
    }
}

template <class ElemType>
QuantizedMatrix<ElemType>::QuantizedMatrix(QuantizedMatrix<ElemType>&& moveFrom)
    : m_quantizedData(moveFrom.m_quantizedData), m_allocator(moveFrom.m_allocator), m_numRows(moveFrom.m_numRows), m_numCols(moveFrom.m_numCols), m_numBits(moveFrom.m_numBits), m_qColSize(moveFrom.m_qColSize)
{
    moveFrom.m_quantizedData = nullptr;
    moveFrom.m_allocator = nullptr;
}

template <class ElemType>
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

template <class ElemType>
QuantizedMatrix<ElemType>::QuantizedMatrix(const size_t numRows, const size_t numCols, const size_t nbits, Matrix<char>* data)
    : m_numRows(numRows), m_numCols(numCols), m_numBits(nbits), m_quantizedData(data), m_allocator(nullptr)
{
    m_qColSize = QuantizedColumn<ElemType>::QuantizedColumnSize(m_numBits, m_numRows);
    if (((QWordNumBits / m_numBits) * m_numBits) != QWordNumBits)
    {
        LogicError("Quantization: 'nbits' must be a divisor of 64");
    }

    // Make sure that the data matrix has enough space
    assert((m_quantizedData->GetNumRows() == m_qColSize) && (m_quantizedData->GetNumCols() >= numCols));
}

template <class ElemType>
QuantizedMatrix<ElemType>::~QuantizedMatrix()
{
    if (nullptr != m_quantizedData)
    {
        // If we used an external allocator, lets free the backing buffer of the matrix
        if (m_allocator != nullptr)
        {
            assert(!m_quantizedData->OwnBuffer());
            m_allocator->Free(m_quantizedData->Data());
        }

        delete m_quantizedData;
        m_quantizedData = nullptr;
    }
}

template <class ElemType>
int QuantizedMatrix<ElemType>::GetDeviceId() const
{
    return m_quantizedData->GetDeviceId();
}

template <class ElemType>
size_t QuantizedMatrix<ElemType>::GetSize() const
{
    return m_quantizedData->GetNumElements();
}

template <class ElemType>
char* QuantizedMatrix<ElemType>::Buffer() const
{
    return m_quantizedData->Data();
}

template <class ElemType>
QuantizedMatrix<ElemType> QuantizedMatrix<ElemType>::ColumnSlice(size_t startColumn, size_t numCols) const
{
    auto matrixSliceData = new Matrix<char>(m_quantizedData->ColumnSlice(startColumn, numCols));
    return QuantizedMatrix<ElemType>(this->GetNumRows(), numCols, this->GetNumBits(), matrixSliceData);
}

template <class ElemType>
void QuantizedMatrix<ElemType>::Print(const char* matrixName, size_t rowStart, size_t rowEnd, size_t colStart, size_t colEnd)
{
    if ((GetNumRows() == 0) || (GetNumCols() == 0))
    {
        LogicError("Print: QuantizedMatrix is empty.");
    }

    if (rowEnd >= GetNumRows() || colEnd >= GetNumCols())
    {
        InvalidArgument("Index out of range.");
    }

    DEVICEID_TYPE orgdevice = this->GetDeviceId();
    CurrentDataLocation curLocation = m_quantizedData->GetCurrentMatrixLocation();
    if (curLocation == CurrentDataLocation::GPU)
    {
        m_quantizedData->_transferToDevice(CPUDEVICE, false, false);
    }

    if (matrixName != nullptr)
        fprintf(stderr, "\n###### %s (%lu, %lu) ######\n", matrixName, GetNumRows(), GetNumCols());
    else
        fprintf(stderr, "\n###### Unnamed Matrix (%lu, %lu) ######\n", GetNumRows(), GetNumCols());

    fprintf(stderr, "\n------ Print Range (%lu:%lu, %lu:%lu) ------\n", rowStart, rowEnd, colStart, colEnd);

    for (size_t j = colStart; j <= colEnd; j++)
    {
        QuantizedColumn<ElemType>* qCol = this->GetQuantizedColumn(j);
        fprintf(stderr, "Lower=%.10f,Upper=%.10f\t", qCol->lower, qCol->upper);
    }
    fprintf(stderr, "\n");

    const size_t ldNbits = ValueQuantizer<ElemType>::ld(this->GetNumBits());
    size_t numQWordsPerCol = ColumnQuantizer<ElemType>::QWordsPerCol(this->GetNumRows(), this->GetNumBits());
    for (size_t i = rowStart; i <= rowEnd; i++)
    {
        size_t qWordIdx = i % numQWordsPerCol;
        size_t offsetInQWord = i / numQWordsPerCol;
        for (size_t j = colStart; j <= colEnd; j++)
        {
            QuantizedColumn<ElemType>* qCol = this->GetQuantizedColumn(j);
            ColumnQuantizer<ElemType> q(ldNbits, qCol->lower, qCol->upper);
            QWord qWord = qCol->bits[qWordIdx];

            QWordVal qVal;
            ElemType val;
            if (this->GetNumBits() == 1)
            {
                ElemType val0 = q.valQ.Unquantize(0);
                ElemType val1 = q.valQ.Unquantize(1);
                qVal = (qWord >> offsetInQWord) & 1;
                val = ValueQuantizer<ElemType>::Unquantize1(qVal != 0, val0, val1);
            }
            else
            {
                const QWordVal bitmask = q.valQ.QuanRangeEnd() - 1;
                qVal = (qWord >> (offsetInQWord * this->GetNumBits())) & bitmask;
                val = q.valQ.Unquantize(qVal);
            }

            fprintf(stderr, "%10d (%.10f)          \t", (int) qVal, val);
        }
        fprintf(stderr, "\n");
    }

    if (curLocation == CurrentDataLocation::GPU)
    {
        m_quantizedData->_transferToDevice(orgdevice, false, false);
    }
}

// Explicit instantiation
template class QuantizedMatrix<float>;
template class QuantizedMatrix<double>;
} } }
