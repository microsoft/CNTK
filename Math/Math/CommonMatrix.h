//
// <copyright file="CommonMatrix.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include <string>
#include <stdint.h>

#define DEVICEID_TYPE int
// and the following magic values
#define CPUDEVICE                 (DEVICEID_TYPE)-1    // device is the CPU
#define DEVICEID_NOTYETDETERMINED (DEVICEID_TYPE)-3    // not yet set
#define DEVICEID_AUTO             (DEVICEID_TYPE)-4    // device should be picked automatically
#define AUTOPLACEMATRIX           (DEVICEID_TYPE)1000  // used in parameters only

// EnforceOneGPUOnly - enforce that we only use one GPU (because we don't really support more than one at this point in time)
// BUGBUG workaround.
// Call this after every place a device is deviced.
// We have multiple independent mechanisms to pick a device.
// After selecting a device id, always run the result through this function, which will cache the first choice.
// TODO: This is a stop-gap. It will be cleaned up once we also fix the GPU late-locking bug.
//       The correct fix is to always route GPU selection through a single function in the first place.
static inline DEVICEID_TYPE EnforceOneGPUOnly(DEVICEID_TYPE requestedDeviceId)
{
    if (requestedDeviceId < 0)      // only apply this to GPU ids
        return requestedDeviceId;
    static DEVICEID_TYPE theGPUId = DEVICEID_NOTYETDETERMINED;
    if (theGPUId == DEVICEID_NOTYETDETERMINED)
        theGPUId = requestedDeviceId;
    else if (theGPUId != requestedDeviceId)
    {
        static bool shown = false;
        if (!shown)
        {
            fprintf(stderr, "EnforceOneGPUOnly: WARNING: Ignored attempt to change GPU choice from %d now %d. This message will be shown only once.\n", theGPUId, requestedDeviceId);
            shown = true;
        }
    }
    return theGPUId;
}
#define EPS_IN_INVERSE 1e-30f  // 1e-37 is the only guaranteed precision
#define EPS_IN_LOG 1e-37f  // 1e-37 is the only guaranteed precision
#define LOG_OF_EPS_IN_LOG -85.1f // log(EPS_IN_LOG)
#define LOG10_OF_EPS_IN_LOG -37 // log_10(EPS_IN_LOG)
#define LZERO  -10e10
#define MINLOGEXP -9.2103
#define LSMALL -0.5E10

#define NOT_IMPLEMENTED \
    {   \
    fprintf(stderr, "Inside File: %s  Line: %d  Function: %s  -> Feature Not Implemented.\n", __FILE__, __LINE__, __FUNCTION__); \
    LogicError("Not Implemented"); \
    }

#define GPUSPARSE_INDEX_TYPE int  //cuSparse only supports int array indexes
#define CPUSPARSE_INDEX_TYPE int  //to be consistent with cuSparse but limited the possible size of the matrix.

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
        void SetArray(ElemType *parray) { m_pArray = parray; }
        virtual DEVICEID_TYPE GetComputeDeviceId() const {return m_computeDevice;}
        void SetComputeDeviceId(const DEVICEID_TYPE computeId) const {m_computeDevice = computeId;}
        bool OwnBuffer() const {return !m_externalBuffer;}
        void SetOwnBuffer(bool own) {m_externalBuffer = !own;}
        wchar_t* GetMatrixName() const { return m_matrixName; }
        size_t NzCount() const {return m_nz;}
        void SetNzCount(const size_t nz) { m_nz = nz; }
        size_t GetSizeAllocated() const {return m_elemSizeAllocated; }
        void SetMatrixName(const wchar_t* s) 
        { 
            Clear();
            if (s!=nullptr)
            {
                size_t n = wcslen(s);
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
            if (m_matrixName!=nullptr)
            {
                delete[] m_matrixName;
                m_matrixName = nullptr;
            }
        }

    protected:
        size_t m_numRows;  
        size_t m_numCols;
        size_t m_elemSizeAllocated;
        size_t m_sliceViewOffset; // this is used to get a column slice view of a matrix in the Sparse CSC format
        MatrixFormat m_format;
        bool m_externalBuffer; // is the buffer used by this matrix, 
        ElemType *m_pArray;
        mutable DEVICEID_TYPE m_computeDevice; //current GPU device Id or CPUDEVICE
        size_t m_nz; //Number of non-zero elements for sparse matrices (unused in other formats)
        wchar_t* m_matrixName;
    };
}}}
