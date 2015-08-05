#pragma once

#include "Matrix.h"
#include "MemAllocator.h"
#include "ValueQuantizer.h"

#ifdef    _WIN32
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
//     - array of 'qbwords'                 // same for each column, rounded to multiple of 32 bits
// one quantized column with header
// This is a variable-length structure.
// A matrix is an array of these.
template<class ElemType>    
struct QuantizedColumn
{
public:
    ElemType lower;                            // quantization range for this column
    ElemType upper;                             // 
    QBWord bits[1/*variable*/];                  // variable-size array to hold the bits, grouped into 'qbwords'

    // required storage size of one columne in bytes for a given column
    // (incl. header, aligned to 4 bytes for 'float')
    cudasharedcode
    static size_t QuantizedColumnSize (size_t bits, size_t rows) 
    {
        const size_t columnDataSize = (rows * bits + (qbwordbits-1)) / qbwordbits * sizeof(QBWord);       // bit array for one column, rounded to multiple of 4 bytes
        return 2 * sizeof (float) + columnDataSize;
    }
};

template<class ElemType>
class MATH_API QuantizedMatrix
{
public:       
    QuantizedMatrix(const size_t numRows, const size_t numCols, const size_t nbits, short deviceId, MemAllocator* allocator = nullptr);
    
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
    char* GetArray() const;
    
    QuantizedColumn<ElemType>* GetQuantizedColumn(size_t colIdx)
    {
        return (QuantizedColumn<ElemType>*)&((this->GetArray())[m_qColSize * colIdx]);
    }
    
    Matrix<char>* GetQuantizedData() const
    {
        return m_quantizedData;
    }
    
    QuantizedMatrix<ElemType> ColumnSlice(size_t startColumn, size_t numCols) const;

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
    size_t m_qColSize; //number of bytes in a quantized column

    template <typename T>
    friend class MatrixQuantizer;
};
}}}