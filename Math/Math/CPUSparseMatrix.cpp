//
// <copyright file="CPUSparseMatrix.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// Math.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include <assert.h>
#include <stdexcept>
#include <omp.h>
#include <math.h>
#include "CPUMatrix.h"
#include "CPUSparseMatrix.h"
#include <random>
#include <chrono>
#include <Windows.h>
#ifdef LEAKDETECT
#include <vld.h>
#endif

#include "basetypes.h"
#include "fileutil.h"


#ifndef USE_MKL
// use ACML as default. 
// Download ACML 5.3.0 (e.g., acml5.3.0-ifort64.exe) or above 
// from http://developer.amd.com/tools/cpu-development/amd-core-math-library-acml/acml-downloads-resources/
// Install the ifort64 variant (compiled with intel compiler) of the library
// Set Environment variable ACML_PATH to C:\AMD\acml5.3.0\ifort64_mp or the folder you installed acml
// to point to your folder for the include file and link library
#include <acml.h>  // requires ACML 5.3.0 and above
#else
// requires MKL 10.0 and above
#endif

// This is an example of an exported variable
//MATH_API int nMath=0;

// This is an example of an exported function.
//MATH_API int fnMath(void)
//{
//    return 42;
//}

#ifndef USE_MKL  //MKL has one additional parameter for different matrix order
#define BLAS_COLMAJOR 
#else
#define BLAS_COLMAJOR (int)MatrixOrder::ColMajor, 
#endif

#define SWAP(a,b) {(a) ^= (b); (b) ^= (a); (a) ^= (b);}
#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // 0 based indexing
namespace Microsoft { namespace MSR { namespace CNTK {

#pragma region Helpful Enum Definitions
    enum class MatrixOrder
    {
        RowMajor = 101,  // row-major arrays 
        ColMajor = 102  // column-major arrays 
    };

    enum class MatrixTranspose : char
    {
        NoTrans = 'N', // trans='N'
        Trans = 'T', // trans='T' 
        ConjTrans = 'C' // trans='C'
    };

    enum class SymMatrixType : char
    {
        Up = 'U', // symmetric matrix is stored in the upper part
        Low = 'L', // symmetric matrix is stored in thelower part
        Full = 'F', //full populated
        NotSymmetric = 'N' //not a symmetric matrix
    };

    enum class MatrixOpSide : char
    {
        Left = 'L', // left multiply
        Right = 'R', // right multiply
    };
#pragma endregion Helpful Enum Definitions

#pragma region Constructors and Destructor

    //should only be used by constructors.
    template<class ElemType>
    void CPUSparseMatrix<ElemType>::ZeroInit()
    {   
        m_numRows = 0;
        m_numCols = 0;
        m_elemSizeAllocated = 0;
        m_externalBuffer = false;
        m_pArray = NULL;
        m_computeDevice = CPUDEVICE;
        m_nz = 0;
        m_matrixName = NULL;   

        if(m_format == MatrixFormat::matrixFormatSparseCSC || m_format == MatrixFormat::matrixFormatSparseCSR) 
        {
            m_colIdx = -1;
            m_val = NULL;
            m_row = NULL;
            m_pb = NULL;
        } 
        else if (m_format == MatrixFormat::matrixFormatSparseBlockCol || m_format == MatrixFormat::matrixFormatSparseBlockRow) 
        {
            m_blockSize = 0;      
            m_blockVal = NULL;
            m_blockIds = NULL;
        }
    }

    template<class ElemType>
    CPUSparseMatrix<ElemType>::CPUSparseMatrix(const MatrixFormat format)
    {
        if(format != MatrixFormat::matrixFormatSparseCSC && format != MatrixFormat::matrixFormatSparseCSR && format != MatrixFormat::matrixFormatSparseBlockCol && format != MatrixFormat::matrixFormatSparseBlockRow) 
        {
            throw std::logic_error("CPUSparseMatrix:  unsupported sparse matrix format");
        }
        m_format = format;
        ZeroInit();
    }

    template<class ElemType>
    CPUSparseMatrix<ElemType>::CPUSparseMatrix(const MatrixFormat format, const size_t numRows, const size_t numCols, const size_t size)
    {   CPUSparseMatrix<ElemType>::CPUSparseMatrix(format);
        Resize(numRows, numCols, size);
    }

    template<class ElemType>
    CPUSparseMatrix<ElemType>::~CPUSparseMatrix()
    {       
        if (m_matrixName!=NULL) 
        {
            delete[] m_matrixName;
            m_matrixName = nullptr;
        }
        if(m_format == MatrixFormat::matrixFormatSparseCSC || m_format == MatrixFormat::matrixFormatSparseCSR) 
        {
            if(m_val != NULL) 
                delete[] m_val;
            if(m_row != NULL) 
                delete[] m_row;
            if(m_pb != NULL)
                delete[] m_pb;
        }  
        else if (m_format == MatrixFormat::matrixFormatSparseBlockCol || m_format == MatrixFormat::matrixFormatSparseBlockRow) 
        {
            if(m_blockVal != NULL) 
                delete[] m_blockVal;
            if(m_blockIds != NULL) 
                delete[] m_blockIds;
        }
    }



#pragma endregion Constructors and Destructor

#pragma region Basic Operators

    //make sure call order in colume wise for CSC and row wise for CSR
    template<class ElemType>
    void CPUSparseMatrix<ElemType>::SetValue(const size_t rIdx, const size_t cIdx, const ElemType v)
    {
        if(m_format != MatrixFormat::matrixFormatSparseCSC && m_format != MatrixFormat::matrixFormatSparseCSR) 
        {
            throw std::logic_error("CPUSparseMatrix:  unsupported SetValue() call.");
        }

        if(m_elemSizeAllocated < m_nz +1) {
            throw std::logic_error("CPUSparseMatrix:  allocated size is too small.");
        }

        if(rIdx < 0 || rIdx >= m_numRows) {
            throw std::logic_error("CPUSparseMatrix: SetValue() invalid row id");
        }

        if(cIdx < 0 || cIdx >= m_numCols) {
            throw std::logic_error("CPUSparseMatrix: SetValue() invalid column id");
        }

        size_t r = (m_format == matrixFormatSparseCSC) ? rIdx: cIdx;
        size_t c = (m_format == matrixFormatSparseCSC) ? cIdx: rIdx;

        m_val[m_nz] = v;
        m_row[m_nz] = r;

        //consistency check
        if(c == m_colIdx && r <= m_row[m_nz-1]) 
        {
            throw std::logic_error("CPUSparseMatrix:  SetValue is not called properly");
        }

        if (c != m_colIdx) 
        {
            m_pb[c] = m_nz;
            m_colIdx = (int) c;
        } 
        m_pb[c+1] = m_nz+1;
        m_nz++;
    }

    template<class ElemType>
    ElemType* CPUSparseMatrix<ElemType>::BufferPointer() const
    {
        if(m_format == MatrixFormat::matrixFormatSparseCSC || m_format == MatrixFormat::matrixFormatSparseCSR) 
        {
            return m_val;
        }  
        else
        {
            return m_blockVal;
        }
    }

    template<class ElemType>
    void CPUSparseMatrix<ElemType>::Resize(const size_t numRows, const size_t numCols, size_t size)
    {               
        m_nz = 0; 
        m_colIdx = -1;
        m_numRows = numRows;
        m_numCols = numCols;            
        
        if(m_elemSizeAllocated < size) 
        {                
            m_elemSizeAllocated = size;
            if(m_format == MatrixFormat::matrixFormatSparseCSC || m_format == MatrixFormat::matrixFormatSparseCSR) 
            {
                if(m_val != NULL) 
                    delete[] m_val;
                if(m_row != NULL) 
                    delete[] m_row;
                if(m_pb != NULL) 
                    delete[] m_pb; 
                
                //int len = m_format == MatrixFormat::matrixFormatSparseCSC ? numCols : numRows;
                size_t len = numCols > numRows ? numCols : numRows;
                m_val = new ElemType[size];
                m_row = new size_t[size];                
                m_pb = new size_t[len+1];  
                
            } 
            else if(m_format == MatrixFormat::matrixFormatSparseBlockCol || m_format == MatrixFormat::matrixFormatSparseBlockRow) 
            {
                if(m_blockVal != NULL) 
                    delete[] m_blockVal;
                if(m_blockIds != NULL) 
                    delete[] m_blockIds;

                size_t max = numCols > numRows ? numCols : numRows;
                m_blockVal = new ElemType[size];                
                m_blockIds = new size_t[max];
            }
        }
    }

    //Reset matrix so it can be reused
    template<class ElemType>
    void CPUSparseMatrix<ElemType>::Reset()
    {                
        m_nz = 0;
        m_colIdx = -1;
        m_blockSize = 0;
    }

    //c = op(a) * op(this) or c += op(a) * op(this) 
    template<class ElemType>
    void CPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& lhs, const bool transposeA, 
        const CPUSparseMatrix<ElemType>& rhs, const bool transposeB, ElemType beta, CPUMatrix<ElemType>& c)

    {
        if (lhs.IsEmpty() || rhs.IsEmpty())
            throw std::logic_error("LeftMultiplyAndAdd:  one of the input matrix is empty.");

        int m = transposeA? (int)lhs.GetNumCols(): (int)lhs.GetNumRows();
        int k = transposeA? (int)lhs.GetNumRows(): (int)lhs.GetNumCols();
        int l = transposeB? (int)rhs.GetNumCols(): (int)rhs.GetNumRows();
        int n = transposeB? (int)rhs.GetNumRows(): (int)rhs.GetNumCols();

        assert (m>0 && k>0 && l>0 && n>0);  //converting from size_t to int may cause overflow
        assert (k == l);
        if (k != l) 
        {
            throw std::invalid_argument("CPUSparseMatrix::MultiplyAndAdd: The inner dimensions of a and b must match.");
        }

        if (c.GetNumRows() != m || c.GetNumCols() != n) 
        {
            c.Resize(m,n);
        }         

        if (beta == 0)
        {
            memset(c.GetArray(), 0, sizeof(ElemType) * c.GetNumElements());
        }
        else 
        {
#pragma omp parallel for
            foreach_coord(i,j,c)
            {
                c(i,j) = beta * c(i,j); 
            } 
        }

        if (!transposeA && !transposeB)
        {
            for(size_t j = 0; j < rhs.GetNumCols(); j++) 
            {
                size_t start = rhs.m_pb[j];
                size_t end = rhs.m_pb[j+1];
                for(size_t p = start; p < end; p++)
                { 
                    size_t i = rhs.m_row[p];
                    ElemType val = rhs.m_val[p];

                    for(size_t h = 0; h < lhs.GetNumRows(); h++)
                    {
                        c(h,j) += alpha * lhs(h, i)*val; 
                    }
                }
            }
        }
        else if (!transposeA && transposeB)
        {           
            for(size_t j = 0; j < rhs.GetNumCols(); j++)
            { 
                size_t start = rhs.m_pb[j];
                size_t end = rhs.m_pb[j + 1];

                for(size_t p = start; p < end; p++)
                { 
                    size_t i = rhs.m_row[p];
                    ElemType val = rhs.m_val[p];
                    for(size_t h = 0; h < lhs.GetNumRows(); h++)
                    {                     
                        c(h, i) += alpha * lhs(h, j)*val;
                    }
                }
            }           
        }
        else if (transposeA && !transposeB)
        {
            NOT_IMPLEMENTED;
        }
        else 
        {
            NOT_IMPLEMENTED;
        }
    }

    //c = alpha * op(a) * op(this)
    template<class ElemType>
    void CPUSparseMatrix<ElemType>::MultiplyAndAdd(ElemType alpha, const CPUMatrix<ElemType>& lhs, const bool transposeA, 
        const CPUSparseMatrix<ElemType>& rhs, const bool transposeB, CPUSparseMatrix<ElemType>& c)
    {
        if (lhs.IsEmpty() || rhs.IsEmpty())
            throw std::logic_error("LeftMultiplyAndAdd:  one of the input matrix is empty.");

        int m = transposeA? (int)lhs.GetNumCols(): (int)lhs.GetNumRows();
        int k = transposeA? (int)lhs.GetNumRows(): (int)lhs.GetNumCols();
        int l = transposeB? (int)rhs.GetNumCols(): (int)rhs.GetNumRows();
        int n = transposeB? (int)rhs.GetNumRows(): (int)rhs.GetNumCols();

        assert (m>0 && k>0 && l>0 && n>0); m; n;  //converting from size_t to int may cause overflow
        assert (k == l);
        if (k != l) 
        {
            throw std::invalid_argument("CPUSparseMatrix::MultiplyAndAdd: The inner dimensions of a and b must match.");
        }

        c.Reset();

        if (!transposeA && !transposeB)
        {
            NOT_IMPLEMENTED;
        }
        else if (!transposeA && transposeB)
        {           
            //allocate enough memory
            if(c.m_elemSizeAllocated < lhs.GetNumElements()) 
            {
                c.Resize(c.GetNumRows(), c.GetNumCols(), lhs.GetNumElements());
            }

            map<size_t, size_t> w2Id;
            for(size_t j = 0; j < rhs.GetNumCols(); j++)
            { // j ranges over batches
                size_t start = rhs.m_pb[j];
                size_t end = rhs.m_pb[j+1];

                for(size_t p = start; p < end; p++) 
                { 
                    size_t i = rhs.m_row[p]; //i ranges over words
                    ElemType val = rhs.m_val[p]; //1 for(i, j)

                    bool first = true;
                    if(w2Id.find(i) == w2Id.end()) 
                    {
                        w2Id[i] = w2Id.size();
                        c.m_blockIds[c.m_blockSize]=i;
                        c.m_blockSize++;
                    } 
                    else 
                    {
                        first = false;
                    }
                    size_t pos = w2Id[i] * lhs.GetNumRows();
                    for(size_t h = 0; h < lhs.GetNumRows(); h++) 
                    { // h range over hidden layer 
                        if(first == true) 
                        {
                            c.m_blockVal[pos] = alpha*lhs(h, j)*val;
                        } else 
                        {
                            c.m_blockVal[pos] += alpha*lhs(h, j)*val;
                        }
                        pos++;
                    }
                }
            }   
            c.m_nz = c.m_blockSize * lhs.GetNumRows();
            if(c.m_nz > c.GetSizeAllocated()) 
            {
                throw std::logic_error("sparse matrix out of range.");
            }
            c.SetFormat(matrixFormatSparseBlockCol);
        }
        else if (transposeA && !transposeB)
        {
            NOT_IMPLEMENTED;
        }
        else 
        {
            NOT_IMPLEMENTED;
        }
    }

    template<class ElemType>
    void CPUSparseMatrix<ElemType>::ScaleAndAdd(const ElemType alpha, const CPUSparseMatrix<ElemType>& lhs, CPUMatrix<ElemType>& rhs)
    {
        if (lhs.IsEmpty() || rhs.IsEmpty()) 
        {
            throw std::logic_error("ScaleAndAdd:  one of the input matrix is empty.");
        }

        if (lhs.GetNumRows() != rhs.GetNumRows() || lhs.GetNumCols() != rhs.GetNumCols()) 
        {
            throw std::invalid_argument("CPUSparseMatrix::ScaleAndAdd: The dimensions of a and b must match.");
        }

        if(lhs.GetFormat() == MatrixFormat::matrixFormatSparseCSC || lhs.GetFormat() == MatrixFormat::matrixFormatSparseCSR) 
        {
            size_t col_num = (lhs.m_format == MatrixFormat::matrixFormatSparseCSC) ? lhs.GetNumCols(): lhs.GetNumRows();
            for(size_t j = 0; j < col_num; j++) 
            {
                size_t start = lhs.m_pb[j];
                size_t end = lhs.m_pb[j + 1];
                for(size_t p = start; p < end; p++) 
                {
                    size_t i = lhs.m_row[p];
                    ElemType val = lhs.m_val[p];
                    size_t r = (lhs.m_format == MatrixFormat::matrixFormatSparseCSC) ? i : j;
                    size_t c = (lhs.m_format == MatrixFormat::matrixFormatSparseCSC) ? j : i;
                    rhs(r, c) += alpha * val; 
                }
            }
        } 
        else if (lhs.m_format == MatrixFormat::matrixFormatSparseBlockCol || lhs.m_format == MatrixFormat::matrixFormatSparseBlockRow) 
        {
            for(size_t j = 0; j < lhs.m_blockSize; j++) 
            {
                size_t i = lhs.m_blockIds[j];
                size_t len = (lhs.m_format == MatrixFormat::matrixFormatSparseBlockCol) ? lhs.GetNumRows() : lhs.GetNumCols();
                size_t start = j * len;
                for(size_t p = start; p < start+len; p++) 
                {
                    ElemType val = lhs.m_blockVal[p];

                    size_t r = (lhs.m_format == MatrixFormat::matrixFormatSparseBlockCol) ? (p - start) : i;
                    size_t c = (lhs.m_format == MatrixFormat::matrixFormatSparseBlockCol) ? i : (p - start);
                    rhs(r, c) += alpha * val; 
                }
            }
        } 
        else 
        {
            throw std::exception("CPUSparseMatrix:: ScaleAndAdd() Not implemented");
        }
    }


    // a: H x No: H is hidden layer size and No is mini-batch size
    // weight: V x H, V is vocab size
    // label: V x No
    // cls: 2 x Nc, Nc is number of classes, each col is start and end word ids of a class
    // idx2cls: V x 1, mapping from word to class id
    // etp: V x No, stores predicted values
    template<class ElemType>
    void CPUSparseMatrix<ElemType>::ClassEntropy(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& weight,
        const CPUSparseMatrix<ElemType> & label, const CPUMatrix<ElemType>& cls, 
        const CPUMatrix<ElemType>& idx2cls, CPUSparseMatrix<ElemType>& etp, CPUMatrix<ElemType>& entropyScore)
    {
        if (a.IsEmpty() || cls.IsEmpty() || label.IsEmpty() || idx2cls.IsEmpty())
            throw std::logic_error("AssignSoftmaxOf: Matrix a, class, idx2cls or label is empty.");

        if(etp.GetFormat() != MatrixFormat::matrixFormatSparseCSC)
            throw std::exception("CPUSparseMatrix:: ClassEntropy() only support CSC");  

        size_t nC = cls.GetNumCols();
        size_t nV = label.GetNumRows() - nC;

        if (nV != idx2cls.GetNumRows() || idx2cls.GetNumCols() != 1 || cls.GetNumCols() + idx2cls.GetNumRows() != label.GetNumRows())
            throw std::logic_error("ClassEntropy: check matrix dimension");
        
        //allocate enough memory
        if(etp.m_elemSizeAllocated < etp.GetNumElements()) 
        {
            etp.Resize(etp.GetNumRows(), etp.GetNumCols(), etp.GetNumElements());
        }
        entropyScore(0, 0) = 0;
        for(size_t j = 0; j < label.GetNumCols(); j++)
        {
            size_t start = label.m_pb[j];
            size_t end = label.m_pb[j + 1];
            for (size_t p = start; p < end; p++)
            {
                size_t i = label.m_row[p];
                size_t iStt, iEnd;
                if (i < nV)
                {
                    size_t clsid = (size_t)idx2cls(i, 0);
                    iStt = (size_t) cls(0, clsid); //class start word id
                    iEnd = (size_t) cls(1, clsid); //class end word id
                }
                else
                {
                    iStt = nV;
                    iEnd = nV + nC;
                }

                size_t b = etp.m_nz;
                for(size_t ii = iStt; ii < iEnd; ii++) //ii ranges over sub-vocab or class ids
                {
                    ElemType val = 0.0;
                    foreach_row(rw, a) //rw ranges over hidden units
                    {
                        val += weight(ii,rw) * a(rw,j); 
                    }
                    etp.SetValue(ii, j, val); 
                }
                ElemType maxV = LZERO;
                for(size_t ii = b; ii < etp.m_nz; ii++)
                {
                    maxV = (ElemType) logadd(maxV, etp.m_val[ii]);
                }

                for(size_t ii = b; ii < etp.m_nz; ii++)
                {
                    etp.m_val[ii] = etp.m_val[ii] - maxV;
                }

                entropyScore(0, 0) -= etp.m_val[b+i-iStt];
                //negate positive data points
                etp.m_val[b+i-iStt] *=-1;
            }
        }
    }


    template<class ElemType>
    void CPUSparseMatrix<ElemType>::ClassEntropyError(CPUSparseMatrix<ElemType>& a)
    {        
        for(int i = 0; i < a.m_nz; i++) 
        {
            if(a.m_val[i] < 0) 
            {
                a.m_val[i] = exp(a.m_val[i]); //negative;
            } 
            else 
            { 
                a.m_val[i] = exp(-a.m_val[i])-1; //positive
            }
        }       
    }


    template<class ElemType>
    void CPUSparseMatrix<ElemType>::ClassEntropyGradientOfInput(
        const CPUSparseMatrix<ElemType>& error, 
        const CPUMatrix<ElemType>& weight,
        CPUMatrix<ElemType>& grd) 
    {
        grd.SetValue(0);

        for(size_t j = 0; j < error.GetNumCols(); j++) 
        {
            size_t start = error.m_pb[j];
            size_t end = error.m_pb[j+1];
            for(size_t p = start; p < end; p++)
            {
                size_t i = error.m_row[p];
                for(size_t h = 0; h < grd.GetNumRows(); h++)
                { // h ranges over hidden units
                    grd(h,j) += weight(i, h) * error.m_val[p];
                }
            }
        }
    }



    template<class ElemType>
    void CPUSparseMatrix<ElemType>::ClassEntropyGradientOfWeight(
        const CPUSparseMatrix<ElemType>& error, 
        const CPUMatrix<ElemType>& input,
        const CPUSparseMatrix<ElemType> & /*label*/,
        const CPUMatrix<ElemType>& /*cls*/, 
        const CPUMatrix<ElemType>& /*idx2cls*/,
        CPUSparseMatrix<ElemType>& grd) 
    {   
        //allocate enough memory
        if(grd.m_elemSizeAllocated < error.m_nz*input.GetNumRows()) 
        {
            grd.Resize(grd.GetNumRows(), grd.GetNumCols(), error.m_nz*input.GetNumRows());
        }
        grd.Reset();
        map<size_t, size_t> w2Id;
        for(size_t j = 0; j < error.GetNumCols(); j++)
        {
            size_t start = error.m_pb[j];
            size_t end = error.m_pb[j+1];

            for(size_t p = start; p < end; p++)
            {
                size_t i = error.m_row[p]; // i ranges over words
                bool first = true;
                if(w2Id.find(i) == w2Id.end()) 
                {
                    w2Id[i] = w2Id.size();
                    grd.m_blockIds[grd.m_blockSize]=i;
                    grd.m_blockSize++;
                } 
                else 
                {
                    first = false;
                }
                size_t pos = w2Id[i]*input.GetNumRows();
                for(size_t h = 0; h < input.GetNumRows(); h++)
                { // h range over hidden layer 
                    if(first == true) 
                    {
                        grd.m_blockVal[pos] = input(h, j)*error.m_val[p];
                    } 
                    else 
                    {
                        grd.m_blockVal[pos] += input(h, j)*error.m_val[p];
                    }
                    pos++;
                }
            }
        }
        grd.m_nz = grd.m_blockSize * input.GetNumRows();
        if(grd.m_nz > grd.GetSizeAllocated()) 
        {
            throw std::logic_error("sparse matrix out of range.");
        }
        grd.SetFormat(matrixFormatSparseBlockRow);
    }

    // normal update for smoothed gradients c and current gradients (this)
    template<class ElemType> 
    void CPUSparseMatrix<ElemType>::NormalGrad(CPUMatrix<ElemType>& c, const ElemType momentum)
    {
        if (c.IsEmpty())
        {
            c.Resize(this->GetNumRows(), this->GetNumCols());
            c.SetValue(0.0);
        }

        if(m_format == MatrixFormat::matrixFormatSparseBlockCol || m_format == MatrixFormat::matrixFormatSparseBlockRow) 
        {
            for(size_t j = 0; j < m_blockSize; j++) 
            {
                size_t i = m_blockIds[j];
                size_t len = (m_format == MatrixFormat::matrixFormatSparseBlockCol) ? GetNumRows() : GetNumCols();
                size_t start = j* len;
                for(size_t p = start; p < start+len; p++) 
                {
                    ElemType val = m_blockVal[p];
                    size_t row = (m_format == MatrixFormat::matrixFormatSparseBlockCol) ? (p - start) : i;
                    size_t col = (m_format == MatrixFormat::matrixFormatSparseBlockCol) ? i : (p - start);
                    c(row, col) = (1-momentum)*val + momentum*c(row, col);
                    m_blockVal[p] = c(row, col);
                }
            }
        } 
        else 
        {
            throw std::exception("CPUSparseMatrix:: NormalGrad() only support block sparse format");
        }
    }

    // update smoothed gradients c and current gradients (this)
    template<class ElemType> 
    void CPUSparseMatrix<ElemType>::Adagrad(CPUMatrix<ElemType>& c)
    {
        if (c.IsEmpty())
        {
            c.Resize(this->GetNumRows(), this->GetNumCols());
            c.SetValue(0.0);
        }

        const ElemType floor = 1e-16f;
        if(m_format == MatrixFormat::matrixFormatSparseCSC || m_format == MatrixFormat::matrixFormatSparseCSR) 
        {
            size_t col_num = (m_format == MatrixFormat::matrixFormatSparseCSC) ? GetNumCols() : GetNumRows();
            for(size_t j = 0; j < col_num; j++) 
            {
                size_t start = m_pb[j];
                size_t end = m_pb[j+1];
                for(size_t p = start; p < end; p++) 
                {
                    size_t i = m_row[p];
                    ElemType val = m_val[p];

                    size_t row = (m_format == MatrixFormat::matrixFormatSparseCSC) ? i : j;
                    size_t col = (m_format == MatrixFormat::matrixFormatSparseCSC) ? j : i;
                    ElemType adenorm = c(row, col); 
                    adenorm += val * val; 
                    val = val / (floor + sqrt(adenorm)); 
                    m_val[p] = val;
                    c(row, col) = adenorm; 
                }
            }
        } else if(m_format == MatrixFormat::matrixFormatSparseBlockCol || m_format == MatrixFormat::matrixFormatSparseBlockRow) 
        {
            for(size_t j = 0; j < m_blockSize; j++)
            {
                size_t i = m_blockIds[j];
                size_t len = (m_format == MatrixFormat::matrixFormatSparseBlockCol) ? GetNumRows() : GetNumCols();
                size_t start = j* len;
                for(size_t p = start; p < start+len; p++) 
                {
                    ElemType val = m_blockVal[p];

                    size_t row = (m_format == MatrixFormat::matrixFormatSparseBlockCol) ? (p - start) : i;
                    size_t col = (m_format == MatrixFormat::matrixFormatSparseBlockCol) ? i : (p - start);
                    ElemType adenorm = c(row, col); 
                    adenorm += val * val; 
                    val = val / (floor + sqrt(adenorm)); 
                    m_blockVal[p] = val;
                    c(row, col) = adenorm; 
                }
            }
        } 
    }

    template<class ElemType>
    CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::InplaceTruncate (const ElemType threshold)
    {
        if(m_format == MatrixFormat::matrixFormatSparseBlockCol || m_format == MatrixFormat::matrixFormatSparseBlockRow) 
        {
            ElemType locThresholdPos = abs(threshold);
            ElemType locTHresholdNeg = -locThresholdPos; 

            for(size_t j = 0; j < m_blockSize; j++) 
            {
                size_t len = (m_format == MatrixFormat::matrixFormatSparseBlockCol) ? GetNumRows() : GetNumCols();
                size_t start = j* len;
                for (size_t p = start; p < start+len; p++)
                {
                    if (m_blockVal[p] > locThresholdPos)
                    {
                        m_blockVal[p] = locThresholdPos;
                    }
                    else if (m_blockVal[p] < locTHresholdNeg)
                    {
                        m_blockVal[p] = locTHresholdNeg;
                    }
                }
            }
        } 
        else 
        {
            throw std::exception("CPUSparseMatrix:: InplaceTruncate() only support block based sparse matrix");
        }
        return *this;
    }    

    template class CPUSparseMatrix<float>; 
    template class CPUSparseMatrix<double>;

}}}
