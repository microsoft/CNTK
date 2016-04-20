#pragma once

#include "Matrix.h"
#include "MemAllocator.h"
#include "ValueQuantizer.h"

#ifdef _WIN32
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#else // no DLLs on Linux
#define MATH_API
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

// A QuantizedColumn represents a quantized value of a column as a byte stream of this format:
// A quantized matrix is then
//  - array of                              // one for each column
//     - lower bound: float
//     - upper bound: float
//     - array of 'QWords'                 // same for each column, rounded to multiple of number of bits in a QWord
// This is a variable-length structure.
// A matrix is an array of these.
template <class ElemType>
struct QuantizedColumn
{
    typedef typename ValueQuantizer<ElemType>::QWord QWord;
    static const size_t QWordNumBits = ValueQuantizer<ElemType>::QWordNumBits;

public:
    // quantization range for this column
    ElemType lower;
    ElemType upper;

    // variable-size array to hold the bits, grouped into 'QWords'
    QWord bits[1];

    // required storage size of one columne in bytes for a given column
    cudasharedcode static size_t QuantizedColumnSize(size_t bits, size_t rows)
    {
        // bit array for one column, rounded to multiple of QWord size
        const size_t columnDataSize = (((rows * bits) + (QWordNumBits - 1)) / QWordNumBits) * sizeof(QWord);
        return (2 * sizeof(ElemType)) + columnDataSize;
    }
};

template <class ElemType>
class MATH_API QuantizedMatrix
{
    typedef typename ValueQuantizer<ElemType>::QWord QWord;
    typedef typename ValueQuantizer<ElemType>::QWordVal QWordVal;
    static const size_t QWordNumBits = ValueQuantizer<ElemType>::QWordNumBits;

public:
    QuantizedMatrix(const size_t numRows, const size_t numCols, const size_t nbits, DEVICEID_TYPE deviceId, MemAllocator* allocator = nullptr);

    // Move constructor and assignment
    QuantizedMatrix(QuantizedMatrix<ElemType>&& moveFrom);
    QuantizedMatrix<ElemType>& operator=(QuantizedMatrix<ElemType>&& moveFrom);

    ~QuantizedMatrix();

    int GetDeviceId() const;

    size_t GetNumRows() const
    {
        return m_numRows;
    }

    size_t GetNumCols() const
    {
        return m_numCols;
    }

    size_t GetNumBits() const
    {
        return m_numBits;
    }

    size_t GetSize() const;
    char* Buffer() const;

    QuantizedColumn<ElemType>* GetQuantizedColumn(size_t colIdx)
    {
        return (QuantizedColumn<ElemType>*) (&((this->Buffer())[m_qColSize * colIdx]));
    }

    Matrix<char>* GetQuantizedData() const
    {
        return m_quantizedData;
    }

    QuantizedMatrix<ElemType> ColumnSlice(size_t startColumn, size_t numCols) const;

    void Print(const char* matrixName, size_t rowStart, size_t rowEnd, size_t colStart, size_t colEnd);

private:
    // Private constructor for creating quantized matrix column slices
    QuantizedMatrix(const size_t numRows, const size_t numCols, const size_t nbits, Matrix<char>* data);

    // Disallow copy construction and assignment
    QuantizedMatrix(const QuantizedMatrix<ElemType>&) = delete;
    QuantizedMatrix<ElemType>& operator=(const QuantizedMatrix<ElemType>&) = delete;

private:
    Matrix<char>* m_quantizedData;
    MemAllocator* m_allocator;

    size_t m_numRows;
    size_t m_numCols;
    size_t m_numBits;

    // number of bytes in a quantized column
    size_t m_qColSize;

    template <typename T>
    friend class MatrixQuantizer;
};
} } }
