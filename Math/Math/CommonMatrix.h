//
// <copyright file="CommonMatrix.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include <string>
#include <stdint.h>

#ifdef	LINUX
#define	wcsnlen_s	wcsnlen			/* Not sure if this is best replacement... Malcolm */
// typedef	char wchar_t;
#endif	

#define AUTOPLACEMATRIX 1000 // used in parameters only
#define MANAGEDEXTERN -2 // managed externally (i.e. PTask)
#define CPUDEVICE -1 // device is the CPU
#define EPS_IN_INVERSE 1e-30f  // min float is 1.4e-45 and max float is 3.4e-38
#define EPS_IN_LOG 1e-40f
#define LOG_OF_EPS_IN_LOG -92.1f // log(EPS_IN_LOG)
#define LOG10_OF_EPS_IN_LOG -40 // log_10(EPS_IN_LOG)

#define NOT_IMPLEMENTED throw std::logic_error("Not implemented.")

namespace Microsoft { namespace MSR { namespace CNTK {    

    enum MatrixFlagBitPosition
    {
        bitPosRowMajor = 0, // row major matrix
        bitPosSparse = 1, // sparse matrix (COO if uncompressed)
        bitPosCompressed = 2, // a compressed sparse format (CSC/CSR)
        bitPosDontOwnBuffer = 3, // buffer is not owned by this matrix
        bitPosSetValueOnDevice = 4, // in a setValue situation, the copy from buffer is already on the device
    };

    enum MatrixFormat
    {
        matrixFormatDense = 0, // default is dense
        matrixFormatColMajor = 0, // default is column major
        matrixFormatRowMajor = 1<<bitPosRowMajor, // row major matrix
        matrixFormatSparse = 1<<bitPosSparse, // sparse matrix
        matrixFormatCompressed = 1<<bitPosCompressed, // a compressed sparse format (CSC/CSR/COO)
        matrixFormatDenseColMajor = matrixFormatDense + matrixFormatColMajor,
        matrixFormatDenseRowMajor = matrixFormatDense + matrixFormatRowMajor,
        matrixFormatSparseCSC = matrixFormatSparse + matrixFormatColMajor + matrixFormatCompressed,
        matrixFormatSparseCSR = matrixFormatSparse + matrixFormatRowMajor + matrixFormatCompressed,
        matrixFormatSparseOther = matrixFormatSparse + matrixFormatRowMajor, // currently used for CPU sparse format, will change to CSC/CSR eventually
        matrixFormatMask = matrixFormatRowMajor + matrixFormatSparse + matrixFormatCompressed,// mask that covers all the 
        matrixFormatSparseBlockCol, //col block based sparse matrix
        matrixFormatSparseBlockRow, //row block based sparse matrix
    };

    // common matrix flags for use on all matrices
    enum MatrixFlags
    {
        // first bits of matrix flags are MatrixFormat
        matrixFlagNormal = 0,
        matrixFlagDontOwnBuffer = 1<<bitPosDontOwnBuffer, // the matrix memory pointers are externally managed, don't allocate/free or attempt to copy to another location
        matrixFlagSetValueOnDevice = 1<<bitPosSetValueOnDevice, // SetValue() call has a buffer that is already on the device
    };


    template<class ElemType>
    class BaseMatrix
    {
    public:
        MatrixFormat GetFormat() const {return m_format;}
        void SetFormat(MatrixFormat format) {m_format = format;}
        size_t GetNumRows() const {return m_numRows;}
        size_t GetNumCols() const {return m_numCols;}
        size_t GetNumElements() const {return m_numRows * m_numCols;}
        bool IsEmpty() const {return m_numRows  == 0 || m_numCols == 0; }
        ElemType* GetArray() {return m_pArray;}
        void SetArray(ElemType *parray) {m_pArray = parray;}
        int GetComputeDeviceId() const {return m_computeDevice;}
        void SetComputeDeviceId(int computeId) {m_computeDevice = computeId;}
        bool OwnBuffer() const {return !m_externalBuffer && m_computeDevice != MANAGEDEXTERN;}
        void SetOwnBuffer(bool own) {m_externalBuffer = !own;}
        wchar_t* GetMatrixName() const { return m_matrixName; }
        size_t NzCount() const {return m_nz;}
        size_t GetSizeAllocated() const {return m_elemSizeAllocated; }
        void SetMatrixName(const wchar_t* s) 
        { 
            Clear();
            if (s!=NULL)
            {
                size_t n = wcsnlen_s(s, SIZE_MAX);
                m_matrixName = new wchar_t[n+1];
                wmemcpy(m_matrixName,s,n+1);
            }
        }

        BaseMatrix()
        {
            m_numRows = m_numCols = m_elemSizeAllocated = 0;
            m_pArray = NULL;
            m_matrixName = NULL;
            m_format = matrixFormatDense;
            m_externalBuffer = false;
            m_nz = 0;
            m_computeDevice = CPUDEVICE;
        }
        ~BaseMatrix()
        {
            Clear();
        }
    protected:
        void Clear()
        {
            if (m_matrixName!=NULL)
            {
                delete[] m_matrixName;
                m_matrixName = NULL;
            }
        }

    protected:
        size_t m_numRows;  
        size_t m_numCols;
        size_t m_elemSizeAllocated;
        MatrixFormat m_format;
        bool m_externalBuffer; // is the buffer used by this matrix, 
        ElemType *m_pArray;
        int m_computeDevice; //current GPU device Id, CPUDEVICE, or MANAGEDEXTERN 
        size_t m_nz; //Number of non-zero elements for sparse matrices (unused in other formats)
        wchar_t* m_matrixName;
    };
}}}
