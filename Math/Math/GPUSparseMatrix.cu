//
// <copyright file="GPUSparseMatrix.cu" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "BestGpu.h"

#ifndef CPUONLY

#include "GPUSparseMatrix.h"
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "cublas_v2.h"
#include "GPUMatrixCUDAKernels.cu"
#include <functional>
#include "CommonMatrix.h"
#include <iostream> // for cout
#include <assert.h>

#pragma warning (disable: 4267) // conversion from 'size_t' to 'unsigned int'; happens in CUDA <<<a,b>>> syntax if a and b are size_t
#pragma warning (disable: 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this

#ifdef    _WIN32
// thread local storage to access the current stream, initalize to default stream
extern __declspec (thread)
#endif
cudaStream_t t_stream;

void CUDACALL(cudaError_t x) 
{
    if(x!=cudaSuccess) 
    { 
        const char* errmsg = cudaGetErrorString(x);
        std::cerr<< "!!!!!!!!CUDA EXCEPTION: " << errmsg << std::endl;

        throw std::runtime_error(errmsg);
    }    
}

void CUSPARSECALL(cusparseStatus_t x) 
{
    if(x!= CUSPARSE_STATUS_SUCCESS) 
    {         
        std::cerr << "!!!!!!!!CUSPARSE EXCEPTION: " << std::endl;
        throw std::runtime_error("CUSPARSE EXCEPTION");
    }    
}

namespace Microsoft { namespace MSR { namespace CNTK {
    void PrepareDevice(DEVICEID_TYPE deviceId);

#pragma region Constructors and Destructor

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::ZeroInit(const MatrixFormat matrixFormat /*= MatrixFormat::matrixFormatSparseCSR*/, 
        const DEVICEID_TYPE computeDevice /*= AUTOPLACEMATRIX*/)
    {
        if (matrixFormat != MatrixFormat::matrixFormatSparseCSC && matrixFormat != MatrixFormat::matrixFormatSparseCSR &&
            matrixFormat != MatrixFormat::matrixFormatSparseBlockCol && matrixFormat != MatrixFormat::matrixFormatSparseBlockRow)
        {
            throw std::logic_error("GPUSparseMatrix:  unsupported sparse matrix format");
        }

        m_computeDevice = computeDevice; //current GPU device Id
        m_numRows=0;  
        m_numCols=0;
        m_elemSizeAllocated = m_nz = 0; //Number of non-zero elements
        m_totalBufferSizeAllocated = 0;
        m_format = matrixFormat;
        m_externalBuffer = false;
        m_pArray=nullptr; 
        m_matrixName=nullptr;

        //MajorIndexLocation() = nullptr;
        //SecondaryIndexLocation() = nullptr;
        m_rowIdx = nullptr;

        m_blockSize = 0;
        m_blockVal = nullptr;
        m_blockIds = nullptr;

        m_expandedSize = 0;
        m_block2Id = nullptr;
        m_block2UniqId = nullptr;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>::GPUSparseMatrix(const MatrixFormat matrixFormat /*= MatrixFormat::matrixFormatSparseCSR*/,
        const DEVICEID_TYPE computeDevice /*= AUTOPLACEMATRIX*/)
    {
        ZeroInit(matrixFormat, computeDevice)
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>::GPUSparseMatrix(const GPUMatrix<ElemType>& deepCopy, const MatrixFormat matrixFormat = MatrixFormat::matrixFormatSparseCSR)
    {
        ZeroInit(matrixFormat, deepCopy.GetComputeDeviceId());
        if (!deepCopy.IsEmpty()) 
            SetValue(deepCopy);
    }


    template<class ElemType>
    GPUSparseMatrix<ElemType>::GPUSparseMatrix(const GPUSparseMatrix<ElemType>& deepCopy)
    {
        DeepCopy(deepCopy);
    }

    // PrepareDevice - Setup the correct cuda context for an operation
    // deviceId - the device on which the operation will take place
    //            defaults to -1, which means use matrices current device
    template<class ElemType>
    void GPUSparseMatrix<ElemType>::PrepareDevice(DEVICEID_TYPE deviceId /*=-1*/) const
    {
        // if default value use current compute device
        if (deviceId == -1)
            deviceId = (DEVICEID_TYPE)m_computeDevice;
        Microsoft::MSR::CNTK::PrepareDevice(deviceId);
    }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::DeepCopy(const GPUSparseMatrix<ElemType>& deepCopy)
    {
        m_computeDevice=deepCopy.m_computeDevice;
        deepCopy.PrepareDevice();

        Resize(deepCopy.m_numRows, deepCopy.m_numCols, deepCopy.m_nz, deepCopy.m_format);
        CUDACALL(cudaMemcpy(NzValues(), deepCopy.NzValues(), NzSize(), cudaMemcpyDeviceToDevice));
        CUDACALL(cudaMemcpy(MajorIndexLocation(), deepCopy.MajorIndexLocation(), MajorIndexSize(), cudaMemcpyDeviceToDevice));
        CUDACALL(cudaMemcpy(SecondaryrIndexLocation(), deepCopy.SecondaryrIndexLocation(), SecondaryIndexSize(), cudaMemcpyDeviceToDevice));

        m_externalBuffer = deepCopy.m_externalBuffer;
        SetMatrixName(deepCopy.m_matrixName);

        //TODO: to copy other varibles used only for class based LM
    }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::SetValue(const GPUSparseMatrix<ElemType>& deepCopy)
    {
        DeepCopy(deepCopy);
    }

    template<class ElemType>
    GPUMatrix<ElemType> GPUSparseMatrix<ElemType>::CopyToDenseMatrix()
    {
        GPUMatrix<ElemType> res(GetComputeDeviceId());
        if (IsEmpty())
            return res;

        PrepareDevice();
        cusparseHandle_t cusparseHandle = 0;
        CUSPARSECALL(cusparseCreate(&cusparseHandle));
        cusparseMatDescr_t descr = 0;
        CUSPARSECALL(cusparseCreateMatDescr(&descr));
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

        res.Resize(m_numRows, m_numCols);

        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));
        CUSPARSECALL(cusparseSetStream(cusparseHandle, t_stream));
        if (m_format == MatrixType::matrixFormatSparseCSR)
        {
            if (sizeof(ElemType) == sizeof(float))
            {
                CUSPARSECALL(cusparseScsr2dense(cusparseHandle, int(m_numRows), int(m_numCols), descr, (float*)NzValues(), RowLocation(), ColLocation(), (float*)res.BufferPointer(), int(m_numRows)));
            }
            else
            {
                CUSPARSECALL(cusparseDcsr2dense(cusparseHandle, int(m_numRows), int(m_numCols), descr, (double*)NzValues(), RowLocation(), ColLocation(), (double*)res.BufferPointer(), int(m_numRows)));
            }
        }
        else if (m_format == MatrixType::matrixFormatSparseCSC)
        {
            if (sizeof(ElemType) == sizeof(float))
            {
                CUSPARSECALL(cusparseScsc2dense(cusparseHandle, int(m_numRows), int(m_numCols), descr, (float*)NzValues(), RowLocation(), ColLocation(), (float*)res.BufferPointer(), int(m_numRows)));
            }
            else
            {
                CUSPARSECALL(cusparseDcsc2dense(cusparseHandle, int(m_numRows), int(m_numCols), descr, (double*)NzValues(), RowLocation(), ColLocation(), (double*)res.BufferPointer(), int(m_numRows)));
            }
        }
        else
        {
            NOT_IMPLEMENTED;
        }

        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));
        CUDACALL(cudaEventDestroy(done));
        CUSPARSECALL(cusparseDestroy(cusparseHandle));

        res.SetMatrixName(m_matrixName);        
        return res;            
    }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::SetValue(const GPUMatrix<ElemType>& denseMatrix, const MatrixFormat matrixFormat/* = matrixFormatSparseCSR*/)
    {
        if (matrixFormat != matrixFormatSparseCSR && matrixFormat != matrixFormatSparseCSC)
        {
            NOT_IMPLEMENTED;
        }

        PrepareDevice();
        cusparseHandle_t cusparseHandle = 0;
        CUSPARSECALL(cusparseCreate(&cusparseHandle));
        cusparseMatDescr_t descr = 0;
        CUSPARSECALL(cusparseCreateMatDescr(&descr));
        cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

        int numRows = (int)denseMatrix.GetNumRows(); //m
        int numCols = (int)denseMatrix.GetNumCols(); //n

        int *nnzPerRowOrCol = nullptr;
        CUDACALL(cudaMalloc((void**)&nnzPerRowOrCol, sizeof(int)*((matrixFormat&matrixFormatRowMajor) ? numRows : numCols)));

        int nnzTotalDevHostPtr = -1;

        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));

        if (sizeof(ElemType)==sizeof(float))
        {
            CUSPARSECALL(cusparseSnnz(cusparseHandle, (matrixFormat&matrixFormatRowMajor) ? CUSPARSE_DIRECTION_ROW : CUSPARSE_DIRECTION_COLUMN, (int)numRows, (int)numCols, descr,
                reinterpret_cast<float*>(denseMatrix.BufferPointer()), (int)numRows, nnzPerRowOrCol, &nnzTotalDevHostPtr));
        }
        else
        {
            CUSPARSECALL(cusparseDnnz(cusparseHandle, (matrixFormat&matrixFormatRowMajor) ? CUSPARSE_DIRECTION_ROW : CUSPARSE_DIRECTION_COLUMN, (int)numRows, (int)numCols, descr,
                reinterpret_cast<double*>(denseMatrix.BufferPointer()), (int)numRows, nnzPerRowOrCol, &nnzTotalDevHostPtr));
        }
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));
        CUDACALL(cudaEventDestroy(done));

        Resize(numRows, numCols, nnzTotalDevHostPtr, matrixFormat);

        CUDACALL(cudaEventCreate(&done));
        if (m_format == MatrixType::matrixFormatSparseCSR)
        {
            if (sizeof(ElemType) == sizeof(float))
            {
                CUSPARSECALL(cusparseSdense2csr(cusparseHandle, (int)m_numRows, (int)m_numCols, descr, reinterpret_cast<float*>(denseMatrix.BufferPointer()),
                    (int)m_numRows, nnzPerRowOrCol, reinterpret_cast<float*>(NzValues()), RowLocation(), ColLocation()));
            }
            else
            {
                CUSPARSECALL(cusparseDdense2csr(cusparseHandle, (int)m_numRows, (int)m_numCols, descr, reinterpret_cast<double*>(denseMatrix.BufferPointer()),
                    (int)m_numRows, nnzPerRowOrCol, reinterpret_cast<double*>(NzValues()), RowLocation(), ColLocation()));
            }
        }
        else if (m_format == MatrixType::matrixFormatSparseCSC)
        {
            if (sizeof(ElemType) == sizeof(float))
            {
                CUSPARSECALL(cusparseSdense2csr(cusparseHandle, (int)m_numRows, (int)m_numCols, descr, reinterpret_cast<float*>(denseMatrix.BufferPointer()),
                    (int)m_numRows, nnzPerRowOrCol, reinterpret_cast<float*>(NzValues()), RowLocation(), ColLocation()));
            }
            else
            {
                CUSPARSECALL(cusparseDdense2csr(cusparseHandle, (int)m_numRows, (int)m_numCols, descr, reinterpret_cast<double*>(denseMatrix.BufferPointer()),
                    (int)m_numRows, nnzPerRowOrCol, reinterpret_cast<double*>(NzValues()), RowLocation(), ColLocation()));
            }
        }
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));
        SetMatrixName(denseMatrix.GetMatrixName());
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::operator=(const GPUSparseMatrix<ElemType>& deepCopy)
    {
        if (this != &deepCopy)
        {
            SetValue(deepCopy);
        }
        return *this;       
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>::GPUSparseMatrix(GPUSparseMatrix<ElemType>&& moveFrom)
    {
        m_computeDevice=moveFrom.m_computeDevice;
        m_numRows=moveFrom.m_numRows;  
        m_numCols=moveFrom.m_numCols;
        m_nz=moveFrom.m_nz; 
        m_elemSizeAllocated = moveFrom.m_elemSizeAllocated;
        m_totalBufferSizeAllocated = moveFrom.m_totalBufferSizeAllocated;
        m_pArray = moveFrom.m_pArray;
        m_format = moveFrom.m_format;
        m_externalBuffer = moveFrom.m_externalBuffer;
        m_matrixName=moveFrom.m_matrixName;

        m_rowIdx = moveFrom.m_rowIdx;

        m_blockSize = moveFrom.m_blockSize;
        m_blockVal = moveFrom.m_blockVal;
        m_blockIds = moveFrom.m_blockIds;

        m_expandedSize = moveFrom.m_expandedSize;
        m_block2Id = moveFrom.m_block2Id;
        m_block2UniqId = moveFrom.m_block2UniqId;

        moveFrom.ZeroInit(moveFrom.m_format, moveFrom.m_computeDevice);
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::operator=(GPUSparseMatrix<ElemType>&& moveFrom)
    {
        Clear();
        m_computeDevice=moveFrom.m_computeDevice;
        m_numRows=moveFrom.m_numRows;
        m_numCols=moveFrom.m_numCols;
        m_nz=moveFrom.m_nz;
        m_elemSizeAllocated = moveFrom.m_elemSizeAllocated;
        m_totalBufferSizeAllocated = moveFrom.m_totalBufferSizeAllocated;
        m_pArray = moveFrom.m_pArray;
        m_format = moveFrom.m_format;
        m_externalBuffer = moveFrom.m_externalBuffer;

        m_matrixName=moveFrom.m_matrixName;

        m_rowIdx = moveFrom.m_rowIdx;

        m_blockSize = moveFrom.m_blockSize;
        m_blockVal = moveFrom.m_blockVal;
        m_blockIds = moveFrom.m_blockIds;

        m_expandedSize = moveFrom.m_expandedSize;
        m_block2Id = moveFrom.m_block2Id;
        m_block2UniqId = moveFrom.m_block2UniqId;

        moveFrom.ZeroInit(moveFrom.m_format, moveFrom.m_computeDevice);

        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>::~GPUSparseMatrix()
    {
        Clear();
    }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::Clear()
    {
        if (m_matrixName!=nullptr) 
        {
            delete[] m_matrixName;
            m_matrixName = NULL;
        }

        if(m_pArray != nullptr) 
            CUDACALL(cudaFree(m_pArray));

        if(m_blockVal != nullptr) 
            CUDACALL(cudaFree(m_blockVal));
        if(m_blockIds != nullptr) 
            CUDACALL(cudaFree(m_blockIds));

        ZeroInit(m_format, m_computeDevice);
    }

    //ResizeAsAndCopyIndexFrom - Resize this sparse matrix to have the same element structure as the passed matrix
    // a - sparse matrix whose structure we want to clone
    // remark: this was done for element wise operations where the structure will be identical after an operation
    template<class ElemType>
    void GPUSparseMatrix<ElemType>::ResizeAsAndCopyIndexFrom(const GPUSparseMatrix<ElemType>& a, const bool growOnly /*= true*/)
    {
        Resize(a.m_numRows, a.m_numCols, a.m_nz, a.m_format, growOnly);

        // copy over the non-zero locations from the source matrix
        CUDACALL(cudaMemcpy(ColLocation(),a.ColLocation(),ColSize(),cudaMemcpyDeviceToDevice));
        CUDACALL(cudaMemcpy(RowLocation(),a.RowLocation(),RowSize(),cudaMemcpyDeviceToDevice));

        CUDACALL(cudaMemcpy(MajorIndexLocation(), a.MajorIndexLocation(), MajorIndexSize(), cudaMemcpyDeviceToDevice));
        CUDACALL(cudaMemcpy(SecondaryrIndexLocation(), a.SecondaryrIndexLocation(), SecondaryIndexSize(), cudaMemcpyDeviceToDevice));
    }

    //-------------------------------------------------------------------------
    // Start of new GPU Sparse Matrix code 
    //-------------------------------------------------------------------------

    template<class ElemType>
    ElemType* GPUSparseMatrix<ElemType>::BufferPointer() const
    {
        if(m_format == matrixFormatSparseCSC || m_format == matrixFormatSparseCSR) 
        {
            return m_pArray;
        }  
        else
        {
            return m_blockVal;
        }
    }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::Resize(const size_t numRows, const size_t numCols, const size_t numNZ, const MatrixFormat matrixFormat, const bool growOnly /*= true*/)
    {               
        m_numRows = numRows;
        m_numCols = numCols; 
        m_nz = numNZ;

        if(m_format == MatrixFormat::matrixFormatSparseCSC || m_format == MatrixFormat::matrixFormatSparseCSR) 
        {
            bool reallocate = (m_totalBufferSizeAllocated < BufferSize() || (!growOnly && m_totalBufferSizeAllocated > BufferSize()));

            m_totalBufferSizeAllocated = BufferSize();
            m_elemSizeAllocated = ElemCountFromBufferSize();

            if (reallocate)
            {
                if (m_pArray != nullptr)
                    CUDACALL(cudaFree(m_pArray));
                if (m_rowIdx != nullptr)
                    CUDACALL(cudaFree(m_rowIdx));
                if (m_block2Id != nullptr)
                    CUDACALL(cudaFree(m_block2Id));
                if (m_block2UniqId != nullptr)
                    CUDACALL(cudaFree(m_block2UniqId));

                PrepareDevice();

                CUDACALL(cudaMalloc((void **)&m_pArray, m_totalBufferSizeAllocated));
                CUDACALL(cudaMalloc((void **)&m_rowIdx, sizeof(size_t)*m_elemSizeAllocated));
                CUDACALL(cudaMalloc((void **)&m_block2Id, sizeof(size_t)*(numCols * 2)));
                CUDACALL(cudaMalloc((void **)&m_block2UniqId, sizeof(size_t)*(numCols * 2)));
            }
        } 
        else if (m_format == MatrixFormat::matrixFormatSparseBlockCol || m_format == MatrixFormat::matrixFormatSparseBlockRow)
        {
            if (m_blockVal != nullptr)
                CUDACALL(cudaFree(m_blockVal));
            if (m_blockIds != nullptr)
                CUDACALL(cudaFree(m_blockIds));
            PrepareDevice();
            CUDACALL(cudaMalloc((void **)&m_blockVal, sizeof(ElemType)*numNZ));
            int max = numCols > numRows ? numCols : numRows;
            CUDACALL(cudaMalloc((void **)&m_blockIds, sizeof(size_t)*max));

        }
        else
            NOT_IMPLEMENTED;
    }

    //Reset matrix so it can be reused
    template<class ElemType>
    void GPUSparseMatrix<ElemType>::Reset()
    {                
        m_nz = 0;
        m_blockSize = 0;
    }

#pragma endregion Constructors and Destructor

#pragma region Static BLAS Functions
    
    // copy features to GPU matrix 
     template<class ElemType>
    void GPUSparseMatrix<ElemType>::SetMatrixFromCSCFormat(size_t *h_row, size_t *h_rowIdx, size_t size, size_t blockSize)
    {
        if(m_format != matrixFormatSparseCSC) 
        {
            throw std::logic_error("CPUSparseMatrix: unsupported SetValue() call.");
        }

        if(m_elemSizeAllocated < size) 
        {
            throw std::logic_error("CPUSparseMatrix:  allocated size is too small.");
        }

        Reset();
        m_nz = size;
        m_blockSize = blockSize;
        PrepareDevice();
        CUDACALL(cudaMemcpy(MajorIndexLocation(), h_row, sizeof(size_t)*size,cudaMemcpyHostToDevice));
        CUDACALL(cudaMemcpy(m_rowIdx, h_rowIdx, sizeof(size_t)*size,cudaMemcpyHostToDevice));   
    }
       
    template<class ElemType>
    void GPUSparseMatrix<ElemType>::SetMatrixFromLabelAndClass(size_t *h_row, size_t *h_block2Id, size_t *h_block2UniqId, size_t labelSize, size_t expandedSize, size_t blockSize)
    {
        if(m_format != matrixFormatSparseCSC) 
        {
            throw std::logic_error("CPUSparseMatrix: unsupported SetValue() call.");
        }

        if(m_elemSizeAllocated < labelSize) 
        {
            throw std::logic_error("CPUSparseMatrix:  allocated size is too small.");
        }
        
        Reset();
        m_nz = labelSize;
        m_expandedSize = expandedSize;
        m_blockSize = blockSize;
        PrepareDevice();
        
        CUDACALL(cudaMemcpy(MajorIndexLocation(), h_row, sizeof(size_t)*labelSize,cudaMemcpyHostToDevice));
        CUDACALL(cudaMemcpy(m_block2Id, h_block2Id, sizeof(size_t)*labelSize,cudaMemcpyHostToDevice));
        CUDACALL(cudaMemcpy(m_block2UniqId, h_block2UniqId, sizeof(size_t)*labelSize,cudaMemcpyHostToDevice));   
    }

    // forward pass from feature to hidden layer
    template<class ElemType>
    void GPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& lhs, const bool transposeA, 
        const GPUSparseMatrix<ElemType>& rhs, const bool transposeB, ElemType beta, GPUMatrix<ElemType>& c)

    {
        if (lhs.GetComputeDeviceId()!=rhs.GetComputeDeviceId()||(lhs.GetComputeDeviceId()!=c.GetComputeDeviceId()))
            throw std::runtime_error("MultiplyAndWeightedAddStD: All matrices must be on the same GPU");

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
            c.SetValue(0.0);
        }
        else 
        {
            c *= beta;
        }

        size_t blocksPerGrid = rhs.m_nz;
        size_t p = (threadsPerBlock < lhs.GetNumRows())? threadsPerBlock : lhs.GetNumRows();
        
        if (!transposeA && !transposeB)
        {
            cudaEvent_t done = nullptr; 
            CUDACALL(cudaEventCreate(&done));
            _denseMulSparseToDense<ElemType><<<blocksPerGrid, p>>>(
                alpha,
                reinterpret_cast<ElemType*>(lhs.BufferPointer()),
                m,
                k,
                rhs.MajorIndexLocation(),
                reinterpret_cast<ElemType*>(c.BufferPointer()));
            CUDACALL(cudaEventRecord(done));        
            CUDACALL(cudaEventSynchronize(done));
            CUDACALL(cudaEventDestroy(done));
        }
        else if (!transposeA && transposeB)
        {           
            NOT_IMPLEMENTED;
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

    // backward pass from hidden layer to feature weight
    template<class ElemType>
    void GPUSparseMatrix<ElemType>::MultiplyAndAdd(ElemType /*alpha*/, const GPUMatrix<ElemType>& lhs, const bool transposeA, 
        const GPUSparseMatrix<ElemType>& rhs, const bool transposeB, GPUSparseMatrix<ElemType>& c)
    {
        if (lhs.GetComputeDeviceId()!=rhs.GetComputeDeviceId())
            throw std::runtime_error("GPUSparseMatrix::MultiplyAndAdd: All matrices must be on the same GPU");
        
        int m = transposeA? (int)lhs.GetNumCols(): (int)lhs.GetNumRows();
        int k = transposeA? (int)lhs.GetNumRows(): (int)lhs.GetNumCols();
        int l = transposeB? (int)rhs.GetNumCols(): (int)rhs.GetNumRows();
        int n = transposeB? (int)rhs.GetNumRows(): (int)rhs.GetNumCols();

        assert(m>0 && k>0 && l>0 && n>0); (void)m; (void)n;  //converting from size_t to int may cause overflow
        assert (k == l);
        if (k != l) 
        {
            throw std::invalid_argument("GPUSparseMatrix::MultiplyAndAdd: The inner dimensions of a and b must match.");
        }

        c.SetFormat(matrixFormatSparseBlockCol);  
        size_t nz = rhs.m_blockSize * c.GetNumRows();        
        //allocate enough memory
        if(c.m_elemSizeAllocated < nz) 
        {
            c.Resize(c.GetNumRows(), c.GetNumCols(), nz);
        }
        c.m_blockSize = rhs.m_blockSize;      
        c.m_nz = nz;
        CUDACALL(cudaMemset(c.m_blockVal,0,sizeof(ElemType)*(c.m_nz)));
        CUDACALL(cudaMemset(c.m_blockIds,0,sizeof(size_t)*(c.m_blockSize)));
                
        if (!transposeA && !transposeB)
        {
            NOT_IMPLEMENTED;
        }
        else if (!transposeA && transposeB)
        {   
            cudaEvent_t done = nullptr;
            CUDACALL(cudaEventCreate(&done));
            size_t blocksPerGrid =rhs.GetNZElements();  
            _denseMulSparseToSparse<ElemType><<<blocksPerGrid, threadsPerBlock>>>(
                lhs.BufferPointer(),
                lhs.GetNumRows(),
                rhs.MajorIndexLocation(),
                rhs.m_rowIdx,          
                c.m_blockVal, 
                c.m_blockIds);
            CUDACALL(cudaEventRecord(done));        
            CUDACALL(cudaEventSynchronize(done));
            CUDACALL(cudaEventDestroy(done));
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

    // used for gradients udpate
    template<class ElemType>
    void GPUSparseMatrix<ElemType>::ScaleAndAdd(const ElemType alpha, const GPUSparseMatrix<ElemType>& lhs, GPUMatrix<ElemType>& rhs)
    {
        if (lhs.GetComputeDeviceId()!=rhs.GetComputeDeviceId())
            throw std::runtime_error("GPUSparseMatrix::ScaleAndAdd: All matrices must be on the same GPU");

        if (lhs.m_format == matrixFormatSparseBlockCol || lhs.m_format == matrixFormatSparseBlockRow) 
        {
            size_t len = (lhs.m_format == matrixFormatSparseBlockCol) ? lhs.GetNumRows(): lhs.GetNumCols();
            bool blockCol = (lhs.m_format == matrixFormatSparseBlockCol);

            cudaEvent_t done = nullptr;
            CUDACALL(cudaEventCreate(&done));
            size_t blocksPerGrid = lhs.m_blockSize;
            _scaleAndAdd<ElemType><<<blocksPerGrid, threadsPerBlock>>>(
                alpha,
                blockCol,
                lhs.m_blockVal,
                lhs.m_blockIds,
                len,
                rhs.BufferPointer(),
                rhs.GetNumRows());
            CUDACALL(cudaEventRecord(done));        
            CUDACALL(cudaEventSynchronize(done));
            CUDACALL(cudaEventDestroy(done));
        } 
        else 
        {
            throw std::runtime_error("GPUSparseMatrix:: ScaleAndAdd() Not implemented");
        }
    }

    // a: H x No: H is hidden layer size and No is mini-batch size
    // weight: V x H, V is vocab size
    // label: V x No
    // cls: 2 x Nc, Nc is number of classes, each col is start and end word ids of a class
    // idx2cls: V x 1, mapping from word to class id
    // etp: V x No, stores predicted values
    template<class ElemType>
    void GPUSparseMatrix<ElemType>::ClassEntropy(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& weight,
        const GPUSparseMatrix<ElemType> & label, const GPUMatrix<ElemType>& cls, 
        const GPUMatrix<ElemType>& idx2cls, GPUSparseMatrix<ElemType>& etp, GPUMatrix<ElemType>& entropyScore)
    {
        int deviceId = a.GetComputeDeviceId();
        if (weight.GetComputeDeviceId()!=deviceId || label.GetComputeDeviceId()!=deviceId || cls.GetComputeDeviceId()!=deviceId 
            || idx2cls.GetComputeDeviceId()!=deviceId || etp.GetComputeDeviceId()!=deviceId )
            throw std::runtime_error("GPUSparseMatrix:: ClassEntropy() All matrices must be on the same GPU");  

        size_t nC = cls.GetNumCols();
        size_t nV = label.GetNumRows() - nC;

        if (nV != idx2cls.GetNumRows() || idx2cls.GetNumCols() != 1 || cls.GetNumCols() + idx2cls.GetNumRows() != label.GetNumRows())
            throw std::logic_error("ClassEntropy: check matrix dimension");        
        
        //allocate enough memory
        if(etp.m_elemSizeAllocated < label.m_expandedSize) 
        {
            etp.Resize(etp.GetNumRows(), etp.GetNumCols(), label.m_expandedSize);
        }
        etp.m_nz = label.m_expandedSize;
        CUDACALL(cudaMemset(etp.m_pArray,0,sizeof(ElemType)*(etp.m_nz)));
        entropyScore.SetValue((ElemType)0);     

        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));
        size_t blocksPerGrid = label.m_expandedSize;

        //_computePrediction<ElemType><<<blocksPerGrid, threadsPerBlock>>>(
        _computePrediction<ElemType><<<blocksPerGrid, 20>>>(
            idx2cls.GetNumRows(),
            a.BufferPointer(),
            a.GetNumRows(),
            weight.BufferPointer(),
            weight.GetNumRows(),
            label.m_nz,
            label.MajorIndexLocation(),
            label.m_block2Id,
            cls.BufferPointer(),
            idx2cls.BufferPointer(),            
            etp.m_pArray,
            etp.MajorIndexLocation(),
            etp.SecondaryIndexLocation());

        blocksPerGrid = label.m_nz;
        _normalizePrediction<ElemType><<<blocksPerGrid, threadsPerBlock>>>(
            label.m_nz,
            label.m_expandedSize,
            label.MajorIndexLocation(),
            label.m_block2Id, 
            etp.MajorIndexLocation(),
            etp.m_pArray,
            entropyScore.BufferPointer());

        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));
        CUDACALL(cudaEventDestroy(done));
   }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::ClassEntropyError(GPUSparseMatrix<ElemType>& a)
    {
        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));

        int N = a.m_nz;
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock); 

        _computePredictionError<ElemType><<<blocksPerGrid, threadsPerBlock>>>(
            a.m_pArray,
            N);

        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));
        CUDACALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::ClassEntropyGradientOfInput(const GPUSparseMatrix<ElemType>& error, const GPUMatrix<ElemType>& weight,  GPUMatrix<ElemType>& grd)
    {
        int deviceId = error.GetComputeDeviceId();
        if (weight.GetComputeDeviceId()!=deviceId || grd.GetComputeDeviceId()!=deviceId )
            throw std::runtime_error("GPUSparseMatrix::ClassEntropyGradientOfInput() All matrices must be on the same GPU");

        grd.SetValue((ElemType)0); 
        cudaEvent_t done = nullptr; 
        CUDACALL(cudaEventCreate(&done));

        size_t blocksPerGrid = grd.GetNumElements();
        //_computeGradientOfInput<ElemType><<<blocksPerGrid, threadsPerBlock>>>(
        _computeGradientOfInput<ElemType><<<blocksPerGrid, 20>>>(
            error.m_pArray,
            error.MajorIndexLocation(),
            error.SecondaryIndexLocation(),
            weight.BufferPointer(),
            weight.GetNumRows(),
            grd.BufferPointer(), 
            grd.GetNumRows());
        CUDACALL(cudaEventRecord(done));  
        CUDACALL(cudaEventSynchronize(done));
        CUDACALL(cudaEventDestroy(done));
    }
    
    template<class ElemType>
    void GPUSparseMatrix<ElemType>::ClassEntropyGradientOfWeight(const GPUSparseMatrix<ElemType>& error,  const GPUMatrix<ElemType>& input, const GPUSparseMatrix<ElemType> & label, const GPUMatrix<ElemType>& cls, 
        const GPUMatrix<ElemType>& idx2cls, GPUSparseMatrix<ElemType>& grd)
    {
        int deviceId = error.GetComputeDeviceId();
        if (input.GetComputeDeviceId()!=deviceId || label.GetComputeDeviceId()!=deviceId || cls.GetComputeDeviceId()!=deviceId  || idx2cls.GetComputeDeviceId()!=deviceId || grd.GetComputeDeviceId()!=deviceId )
            throw std::runtime_error("GPUSparseMatrix::ClassEntropyGradientOfWeight() All matrices must be on the same GPU");

        grd.SetFormat(matrixFormatSparseBlockRow);  
        size_t nz = label.m_blockSize * grd.GetNumCols();        
        //allocate enough memory
        if(grd.m_elemSizeAllocated < nz) 
        {
            grd.Resize(grd.GetNumRows(), grd.GetNumCols(), nz);
        }
        grd.m_blockSize = label.m_blockSize;      
        grd.m_nz = nz;
        CUDACALL(cudaMemset(grd.m_blockVal,0,sizeof(ElemType)*(grd.m_nz)));
        CUDACALL(cudaMemset(grd.m_blockIds,0,sizeof(size_t)*(grd.m_blockSize)));

        cudaEvent_t done = nullptr;  
        CUDACALL(cudaEventCreate(&done));

        size_t blocksPerGrid = error.m_nz;
        _computeGradientOfWeight<ElemType><<<blocksPerGrid, threadsPerBlock>>>(
            error.m_pArray,
            error.MajorIndexLocation(),
            error.SecondaryIndexLocation(),
            input.GetNumCols(),
            idx2cls.GetNumRows(),
            label.MajorIndexLocation(),
            label.m_block2UniqId,
            cls.BufferPointer(),
            idx2cls.BufferPointer(),              
            input.BufferPointer(),
            input.GetNumRows(),
            grd.m_blockVal, 
            grd.m_blockIds);
        CUDACALL(cudaEventRecord(done)); 
        CUDACALL(cudaEventSynchronize(done));
        CUDACALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceTruncate (const ElemType threshold)
    {
        if(m_format == matrixFormatSparseBlockCol || m_format == matrixFormatSparseBlockRow) 
        {
            long N=(long)GetNZElements();
            int blocksPerGrid =(int)ceil(N*1.0/threadsPerBlock);                
            cudaEvent_t done = nullptr;
            CUDACALL(cudaEventCreate(&done));        
            _inplaceTruncate<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_blockVal,threshold,N);
            CUDACALL(cudaEventRecord(done));        
            CUDACALL(cudaEventSynchronize(done));   
            CUDACALL(cudaEventDestroy(done));
        } 
        else 
        {
            throw std::runtime_error("GPUSparseMatrix:: InplaceTruncate() only support block based sparse matrix");
        }
        return *this;
    } 

    // normal update for smoothed gradients c and current gradients (this)
    template<class ElemType> 
    void GPUSparseMatrix<ElemType>::NormalGrad(GPUMatrix<ElemType>& c, const ElemType momentum)
    {
        if (c.IsEmpty())
        {
            c.Resize(GetNumRows(), GetNumCols());
            c.SetValue(0.0);
        }

        if(m_format == matrixFormatSparseBlockCol || m_format == matrixFormatSparseBlockRow) 
        {
            size_t blocksPerGrid = m_blockSize;
            bool isBlockCol = (m_format == MatrixFormat::matrixFormatSparseBlockCol);
            size_t len = isBlockCol ? GetNumRows(): GetNumCols();
            cudaEvent_t done = nullptr;
            CUDACALL(cudaEventCreate(&done));        
            _normalGrad<ElemType><<<blocksPerGrid,threadsPerBlock>>>(
                isBlockCol,
                len,
                momentum,
                m_blockIds,
                m_blockVal,
                c.BufferPointer(),
                c.GetNumRows());                        
            CUDACALL(cudaEventRecord(done));        
            CUDACALL(cudaEventSynchronize(done));    
            CUDACALL(cudaEventDestroy(done));
        } 
        else 
        {
            throw std::runtime_error("GPUSparseMatrix:: NormalGrad() only support block sparse format");
        }
    }

    //-------------------------------------------------------------------------
    // End of new GPU Sparse Matrix code 
    //-------------------------------------------------------------------------

    template<class ElemType>
    void  GPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const GPUSparseMatrix<ElemType>& a, const bool transposeA, 
        const GPUMatrix<ElemType>& b, ElemType beta, GPUMatrix<ElemType>& c)
    {
        if (a.GetComputeDeviceId()!=b.GetComputeDeviceId()||(b.GetComputeDeviceId()!=a.GetComputeDeviceId()))
            throw std::runtime_error("MultiplyAndWeightedAddStD: All matrices must be on the same GPU");
        a.PrepareDevice();
        cusparseHandle_t cusparseHandle = 0;
        CUSPARSECALL(cusparseCreate(&cusparseHandle));
        cusparseMatDescr_t descr = 0;
        CUSPARSECALL(cusparseCreateMatDescr(&descr));
        cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
        cusparseOperation_t oper = transposeA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

        int m = (int)a.GetNumRows();
        int n = (int)b.GetNumCols();
        assert(n==(int)c.GetNumCols());
        int k = (int)a.GetNumCols();

        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));
        if (sizeof(ElemType)==sizeof(float))
        {
            CUSPARSECALL(cusparseScsrmm(cusparseHandle,oper,m,n,k,(int)a.GetNZElements(),reinterpret_cast <float*>(&alpha),descr,reinterpret_cast <const float*>(a.NzValues()),
                a.RowLocation(), a.ColLocation(), reinterpret_cast <float*>(b.BufferPointer()),
                (int)b.GetNumRows(),reinterpret_cast <float*>(&beta),reinterpret_cast <float*>(c.BufferPointer()),(int)c.GetNumRows()));
        }
        else 
        {
            CUSPARSECALL(cusparseDcsrmm(cusparseHandle,oper,m,n,k,(int)a.GetNZElements(),reinterpret_cast <double*>(&alpha),descr,reinterpret_cast <const double*>(a.NzValues()),
                a.RowLocation(), a.ColLocation(), reinterpret_cast <double*>(b.BufferPointer()),
                (int)b.GetNumRows(),reinterpret_cast <double*>(&beta),reinterpret_cast <double*>(c.BufferPointer()),(int)c.GetNumRows()));
        }
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));
        CUDACALL(cudaEventDestroy(done));
        CUSPARSECALL(cusparseDestroy(cusparseHandle));        
    }
       

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::Multiply(const GPUSparseMatrix<ElemType>& S, const GPUMatrix<ElemType>& D, GPUMatrix<ElemType>& C)
    {
        if (C.GetNumRows()!=S.GetNumRows() || C.GetNumCols()!=D.GetNumRows())
        {
            GPUMatrix<ElemType> tmp(S.GetNumRows(),D.GetNumCols(),S.GetComputeDeviceId());
            C=tmp;
        }
        MultiplyAndWeightedAdd(1,S,false,D,0,C);
    }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::Multiply(const GPUMatrix<ElemType>& D, const GPUSparseMatrix<ElemType>& S, GPUMatrix<ElemType>& C)
    {   
        GPUMatrix<ElemType> Res(S.GetNumCols(),D.GetNumRows());
        MultiplyAndWeightedAdd(1,S,true,D.Transpose(),0,Res);
        C.AssignTransposeOf(Res);       
    }

    // ElemCountFromBufferSize - Return the elemCountAllocated for a particular buffersize
    // totalBufferSize - total buffer we have to use
    // return: size of allocated elements/index slots available
    template<class ElemType>
    size_t GPUSparseMatrix<ElemType>::ElemCountFromBufferSize(const size_t totalBufferSize)
    {
        size_t elemSizeAllocated;
        if (m_format & matrixFormatCompressed)
        {
            elemSizeAllocated = (totalBufferSize-SecondaryIndexSize())/(sizeof(int)+sizeof(ElemType));
        }
        else // uncompressed COO format
        {
            elemSizeAllocated = totalBufferSize/(2*sizeof(int)+sizeof(ElemType));
        }
        return elemSizeAllocated;
    }

    // PrepareBuffer - Get the dimensions start buffer, computes the starting row/column of each value
    // m - rows in the source
    // n - cols in the source
    // canReuseBuffer - target matrix can be reused for temporary space
    // func - function to call to count elements in the result (returns count, and fills csrRowPtr array)
    template<class ElemType>
    void GPUSparseMatrix<ElemType>::PrepareBuffer(size_t m, size_t n, bool canReuseBuffer, std::function<size_t (int* csrRowPtrC)> func)
    {
        int* csrRowPtrC=nullptr;
        GPUSparseMatrix<ElemType>& c = *this;
        size_t cSize = c.BufferSize();
        size_t rowBufferRequired = (m + 1)*sizeof(int);
        // determine the size of the buffer and align the final location of the row index buffer
        size_t nzBufSize = cSize-rowBufferRequired;
        nzBufSize -= nzBufSize%(sizeof(int)+sizeof(ElemType));
        bool allocatedBuffer = false;

        // do we have enough memory to store just the row buffer?
        if (cSize >= rowBufferRequired && c.NzValues() != nullptr && canReuseBuffer)
        {
            // determine the final location if we reuse the buffer
            csrRowPtrC = (int*)((byte*)c.NzValues() + nzBufSize);
        }
        else
        {
            CUDACALL(cudaMalloc((void **)&csrRowPtrC,(m+1)*sizeof(int)));
            allocatedBuffer = true;
        }

        // get the non-zero count from the function (and 
        size_t nnzC = func(csrRowPtrC);

        // now we know the number of Non-zeros in the result set, set the output size
        c.m_elemSizeAllocated = c.m_nz = nnzC;
        c.m_numRows = m;
        c.m_numCols = n;
        size_t requiredSize = c.BufferSize();
        // see if the buffer we already have is big enough
        if (cSize >= requiredSize)
        {
            // compute the allocated size, to take up any additional space in the memory block 
            c.m_elemSizeAllocated = c.ElemCountFromBufferSize(cSize);
            // copy the rowPtr array to the proper location
            CUDACALL(cudaMemcpy(c.SecondaryIndexLocation(),csrRowPtrC,c.SecondaryIndexSize(),cudaMemcpyDeviceToDevice));
        }
        else
        {
            void* oldBuffer = c.m_pArray;
            // allocate required array space
            CUDACALL(cudaMalloc((void **)&c.m_pArray,requiredSize));      
            // copy over 
            CUDACALL(cudaMemcpy(c.SecondaryIndexLocation(),csrRowPtrC,c.SecondaryIndexSize(),cudaMemcpyDeviceToDevice));
            // release the previous buffer since we just reallocated it
            if (oldBuffer != nullptr)
                CUDACALL(cudaFree(oldBuffer));
        }
        // if we allocated the buffer, free it here
        if (allocatedBuffer)
            CUDACALL(cudaFree(csrRowPtrC));
    }

    // Multiply - multiply one spares matrix by another sparse matrix
    // S1 - first sparse matrix
    // transposeS1 - transpose first matrix?
    // S2 - second sparse matrix
    // transposeS2 - tanspose second matrix?
    // c - result matrix
    // NOTE: if c has enough space allocated, it will be reused, otherwise it will be freed and a new memory block used
    template<class ElemType>
    void GPUSparseMatrix<ElemType>::Multiply(const GPUSparseMatrix<ElemType>& S1, bool transposeS1, const GPUSparseMatrix<ElemType>& S2, bool transposeS2, GPUSparseMatrix<ElemType> &c)
    {
        if (S1.GetComputeDeviceId()!=S2.GetComputeDeviceId())
            throw std::runtime_error("Sparse matrix multiply: both matrices must be on the same device");

        S1.PrepareDevice();
        cusparseHandle_t cusparseHandle = 0;
        CUSPARSECALL(cusparseCreate(&cusparseHandle));
        cusparseMatDescr_t descrA = 0, descrB = 0, descrC = 0;
        CUSPARSECALL(cusparseCreateMatDescr(&descrA)); CUSPARSECALL(cusparseCreateMatDescr(&descrB)); CUSPARSECALL(cusparseCreateMatDescr(&descrC));        
        cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL); cusparseSetMatType(descrB,CUSPARSE_MATRIX_TYPE_GENERAL); cusparseSetMatType(descrC,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO); cusparseSetMatIndexBase(descrB,CUSPARSE_INDEX_BASE_ZERO); cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ZERO);
        cusparseOperation_t operA = transposeS1 ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
        cusparseOperation_t operB = transposeS2 ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

        int m = int(transposeS1 ? S1.GetNumCols() : S1.GetNumRows());
        int n = int(transposeS2 ? S2.GetNumRows() : S2.GetNumCols());
        int k = int(transposeS1 ? S1.GetNumRows() : S1.GetNumCols());
        int l = int(transposeS2 ? S2.GetNumCols() : S2.GetNumRows());
        if (k!=l)
            throw std::runtime_error("Sparse matrix multiply: dimensionality mismatch");

        int nnzA = (int)S1.GetNZElements();
        int nnzB = (int)S2.GetNZElements();

        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));
        //Step 1 
        c.PrepareBuffer(m, n, true, // true means we can reuse the "c" buffer if it exists for temporaries
            [&](int* csrRowPtrC) -> size_t
        {
            int nnzTotal = -1; 
            CUSPARSECALL(cusparseXcsrgemmNnz(cusparseHandle,operA,operB,m,n,k,descrA,nnzA,S1.RowLocation(),S1.ColLocation(),descrB,nnzB,
                S2.RowLocation(),S2.ColLocation(),descrC,csrRowPtrC,&nnzTotal));
            return nnzTotal;
        });


        //Step 2
        if (sizeof(float)==sizeof(ElemType))
        {
            CUSPARSECALL(cusparseScsrgemm(cusparseHandle,operA,operB,m,n,k,descrA,nnzA,(const float*)S1.NzValues(),S1.RowLocation(),S1.ColLocation(),
                descrB,nnzB,(const float*)S2.NzValues(),S2.RowLocation(),S2.ColLocation(),
                descrC,(float*)c.NzValues(),c.RowLocation(),c.ColLocation()));
        }
        else
        {
            CUSPARSECALL(cusparseDcsrgemm(cusparseHandle,operA,operB,m,n,k,descrA,nnzA,(const double*)S1.NzValues(),S1.RowLocation(),S1.ColLocation(),
                descrB,nnzB,(const double*)S2.NzValues(),S2.RowLocation(),S2.ColLocation(),
                descrC,(double*)c.NzValues(),c.RowLocation(),c.ColLocation()));
        }
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));
        CUDACALL(cudaEventDestroy(done));
        cusparseDestroy(cusparseHandle);   
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignProductOf(const GPUSparseMatrix<ElemType>& a, const bool transposeA, const GPUSparseMatrix<ElemType>& b, const bool transposeB)
    {
        Multiply(a,transposeA,b,transposeB,*this);
        return *this;
    }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::ScaleAndAdd(ElemType alpha,const GPUSparseMatrix<ElemType>& a, ElemType beta, const GPUSparseMatrix<ElemType>& b, GPUSparseMatrix<ElemType>& c)
    {
        if (a.GetNumCols()!=b.GetNumCols() || a.GetNumRows()!=b.GetNumRows())
            throw std::runtime_error("Dimensions mismatch in ScaleAndAdd");
        if (a.GetComputeDeviceId()!=b.GetComputeDeviceId())
            throw std::runtime_error("ScaleAndAdd: matrices must be on the same device");

        int m = (int)a.GetNumRows();
        int n = (int)a.GetNumCols();
        int nnzA = (int)a.GetNZElements();
        int nnzB = (int)b.GetNZElements();

        a.PrepareDevice();
        cusparseHandle_t cusparseHandle = 0;
        CUSPARSECALL(cusparseCreate(&cusparseHandle));
        cusparseMatDescr_t descrA = 0, descrB = 0, descrC = 0;
        CUSPARSECALL(cusparseCreateMatDescr(&descrA)); CUSPARSECALL(cusparseCreateMatDescr(&descrB)); CUSPARSECALL(cusparseCreateMatDescr(&descrC));
        cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL); cusparseSetMatType(descrB,CUSPARSE_MATRIX_TYPE_GENERAL); cusparseSetMatType(descrB,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO); cusparseSetMatIndexBase(descrB,CUSPARSE_INDEX_BASE_ZERO); cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ZERO);

        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));
        //Step 1 
        bool inOutParameter = (&b == &c);
        c.PrepareBuffer(m, n, !inOutParameter, [&] (int* csrRowPtrC) -> size_t
        {
            int nnzTotal = -1;
            CUSPARSECALL(cusparseXcsrgeamNnz(cusparseHandle,m,n,descrA,nnzA,a.RowLocation(),a.ColLocation(),descrB,nnzB,b.RowLocation(),b.ColLocation(),descrC,csrRowPtrC,&nnzTotal));
            return nnzTotal;
        });

        //Step 2
        if (sizeof(ElemType)==sizeof(float))
        {
            CUSPARSECALL(cusparseScsrgeam(cusparseHandle,m,n,reinterpret_cast <const float*>(&alpha),descrA,nnzA,reinterpret_cast <const float*>(a.NzValues()),a.RowLocation(),a.ColLocation(),
                reinterpret_cast <const float*>(&beta),descrB,nnzB,reinterpret_cast <const float*>(b.NzValues()),b.RowLocation(),b.ColLocation(),descrC,reinterpret_cast <float*>(c.NzValues()),c.RowLocation(),c.ColLocation()));
        }
        else
        {
            CUSPARSECALL(cusparseDcsrgeam(cusparseHandle,m,n,reinterpret_cast <const double*>(&alpha),descrA,nnzA,reinterpret_cast <const double*>(a.NzValues()),a.RowLocation(),a.ColLocation(),
                reinterpret_cast <const double*>(&beta),descrB,nnzB,reinterpret_cast <const double*>(b.NzValues()),b.RowLocation(),b.ColLocation(),descrC,reinterpret_cast <double*>(c.NzValues()),c.RowLocation(),c.ColLocation()));
        }
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));
        CUDACALL(cudaEventDestroy(done));
        cusparseDestroy(cusparseHandle);   
    }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::ScaleAndAdd(ElemType alpha,const GPUSparseMatrix<ElemType>& a, ElemType beta, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
    {
        if (a.GetNumRows()!=b.GetNumRows()||a.GetNumRows()!=c.GetNumRows()||a.GetNumCols()!=b.GetNumCols()||a.GetNumCols()!=c.GetNumCols())
            throw std::logic_error("ScaleAndAdd: dimension mismatch");
        if (a.GetComputeDeviceId()!=b.GetComputeDeviceId()||a.GetComputeDeviceId()!=c.GetComputeDeviceId())
            throw std::runtime_error("ScaleAndAdd: matrices must be on the same device");
        b.PrepareDevice();
        //copy b to c
        CUDACALL(cudaMemcpy(c.BufferPointer(),b.BufferPointer(),sizeof(ElemType)*b.GetNumElements(),cudaMemcpyDeviceToDevice));
        if (beta!=1)
        {
            c*=beta;
        }
        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));
        long M=(long)a.GetNumRows();
        int blocksPerGrid =(int)ceil(1.0*M/threadsPerBlock);        
        _sparsePlusDense<ElemType><<<blocksPerGrid,threadsPerBlock>>>(alpha,a.NzValues(),a.RowLocation(),a.ColLocation(),c.BufferPointer(),M);
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));
        CUDACALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::ScaleAndAdd(ElemType alpha,const GPUMatrix<ElemType>& a, ElemType beta, const GPUSparseMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
    {
        ScaleAndAdd(beta,b,alpha,a,c);
    }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::Scale(ElemType alpha, GPUSparseMatrix<ElemType>& a)
    {
        if (a.IsEmpty())
            return;

        long N=(long)a.GetNZElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);                
        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));        
        _scaleArray<ElemType><<<blocksPerGrid,threadsPerBlock>>>(alpha,a.NzValues(),N);
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));        
        CUDACALL(cudaEventDestroy(done));        
    }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::ElementWisePower (ElemType alpha, const GPUSparseMatrix<ElemType>& a, GPUSparseMatrix<ElemType>& c)
    {
        if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
        {
            throw std::invalid_argument("All matrices must be on the same GPU");
        }
        else 
        {
            if (a.IsEmpty())
                throw std::logic_error("ElementWisePower:  The input matrix a is empty.");
            if (a.GetNumRows()!=c.GetNumRows() || a.GetNumCols()!=c.GetNumCols() || a.GetNZElements()!=c.GetNZElements())
                c.ResizeAsAndCopyIndexFrom(a);

            cudaEvent_t done = nullptr;
            CUDACALL(cudaEventCreate(&done));
            a.PrepareDevice();
            long N=(long)a.GetNZElements();
            int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);                
            _elementWisePowerOnCuda<ElemType><<<blocksPerGrid,threadsPerBlock>>>(alpha,a.NzValues(),c.NzValues(),N);             
            CUDACALL(cudaEventRecord(done));        
            CUDACALL(cudaEventSynchronize(done));   
        }
    }

    template<class ElemType>
    ElemType GPUSparseMatrix<ElemType>::InnerProductOfMatrices(const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
    {
        if (a.GetComputeDeviceId()!=b.GetComputeDeviceId())
            throw std::runtime_error("a and b must be on the same device");

        //This implementation requires additional memory
        //need to put a in ColumnMajor format
        int m = (int)a.GetNumRows();
        int n = (int)a.GetNumCols();
        int nnz = (int)a.GetNZElements();
        cusparseAction_t cpVals = CUSPARSE_ACTION_NUMERIC;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
        ElemType* cscValA = nullptr;
        int* cscRowIndA = nullptr;
        int* cscColPtrA = nullptr;
        a.PrepareDevice();
        CUDACALL(cudaMalloc((void **)&cscValA,nnz*sizeof(ElemType)));
        CUDACALL(cudaMalloc((void **)&cscRowIndA,nnz*sizeof(int)));        
        CUDACALL(cudaMalloc((void **)&cscColPtrA,(n+1)*sizeof(int)));
        cusparseHandle_t cusparseHandle = 0;
        CUSPARSECALL(cusparseCreate(&cusparseHandle));
        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));
        if (sizeof(ElemType)==sizeof(float))
        {
            CUSPARSECALL(cusparseScsr2csc(cusparseHandle,m,n,nnz,reinterpret_cast<const float*>(a.NzValues()),a.RowLocation(),a.ColLocation(),reinterpret_cast<float*>(cscValA),cscRowIndA,cscColPtrA,cpVals,idxBase));
        }
        else
        {
            CUSPARSECALL(cusparseDcsr2csc(cusparseHandle,m,n,nnz,reinterpret_cast<const double*>(a.NzValues()),a.RowLocation(),a.ColLocation(),reinterpret_cast<double*>(cscValA),cscRowIndA,cscColPtrA,cpVals,idxBase));
        }
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done)); 
        CUDACALL(cudaEventDestroy(done));

        //Given sparse matrix in column major format, calculate indices for corresponding sparse vector
        int* vectArray=nullptr;
        CUDACALL(cudaMalloc((void**)&vectArray,sizeof(int)*a.m_nz));
        long M=n;
        long N=m;
        //int* h_vectArray= new int[a.m_nz];
        int blocksPerGrid =(int)ceil(1.0*M/threadsPerBlock);   
        CUDACALL(cudaEventCreate(&done));
        _getSparseVectorRepresntationForMatrix<ElemType><<<blocksPerGrid,threadsPerBlock>>>(cscColPtrA,cscRowIndA,vectArray,M,N);
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));
        CUDACALL(cudaEventDestroy(done));
        CUDACALL(cudaFree(cscRowIndA));
        CUDACALL(cudaFree(cscColPtrA));
        //CUDACALL(cudaMemcpy(h_vectArray,vectArray,sizeof(int)*a.m_nz,cudaMemcpyDeviceToHost));    

        //Actual dot product
        ElemType res=0;
        if (sizeof(ElemType)==sizeof(float))
        {
            CUSPARSECALL(cusparseSdoti(cusparseHandle,(int)a.m_nz,reinterpret_cast<float*>(cscValA),vectArray,
                reinterpret_cast<float*>(b.BufferPointer()),
                reinterpret_cast<float*>(&res),idxBase));
        }
        else
        {
            CUSPARSECALL(cusparseDdoti(cusparseHandle,(int)a.m_nz,reinterpret_cast<double*>(cscValA),vectArray,
                reinterpret_cast<double*>(b.BufferPointer()),
                reinterpret_cast<double*>(&res),idxBase));
        }       
        CUDACALL(cudaFree(vectArray));
        CUDACALL(cudaFree(cscValA));
        CUSPARSECALL(cusparseDestroy(cusparseHandle));   
        return res;        
    }

    template<class ElemType>
    ElemType GPUSparseMatrix<ElemType>::InnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b)
    {
        return GPUSparseMatrix<ElemType>::InnerProductOfMatrices(b,a);
    }

    template<class ElemType>
    bool GPUSparseMatrix<ElemType>::AreEqual(const GPUSparseMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b, 
        const ElemType threshold)
    {
        if (a.GetNZElements()!=b.GetNZElements() || a.GetNumRows()  != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
            return false;

        a.PrepareDevice();
        long *res = new long[3];
        res[0]=1;
        res[1]=1;
        res[2]=1;
        long *d_res = nullptr;
        CUDACALL(cudaMalloc((void**)&d_res,sizeof(long)*3)); 
        CUDACALL(cudaMemcpy(d_res,res,sizeof(long)*3,cudaMemcpyHostToDevice));

        int blocksPerGrid =(int)ceil(1.0*a.GetNZElements()/threadsPerBlock); 
        _areEqual<ElemType><<<blocksPerGrid,threadsPerBlock>>>(a.NzValues(),b.NzValues(),(long)a.GetNZElements(),threshold,d_res);        
        _areEqual<int><<<blocksPerGrid,threadsPerBlock>>>(a.ColLocation(),b.ColLocation(),(long)a.GetNZElements(),(int)threshold,d_res+1);
        blocksPerGrid =(int)ceil((1.0*a.GetNumRows()+1.0)/threadsPerBlock); 
        _areEqual<int><<<blocksPerGrid,threadsPerBlock>>>(a.RowLocation(),b.RowLocation(),(long)a.GetNumRows()+1,(int)threshold,d_res+2);

        CUDACALL(cudaMemcpy(res,d_res,sizeof(long)*3,cudaMemcpyDeviceToHost));        
        if (res[0]*res[1]*res[2]==1)
            return true;
        else
            return false;
    }

    template<class ElemType>
    bool GPUSparseMatrix<ElemType>::AreEqual(const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b, 
        const ElemType threshold)
    {
        if (a.GetNumElements()!=b.GetNZElements() || a.GetNumRows()  != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
            return false;
        GPUSparseMatrix<ElemType> c;
        c.SetValue(a);
        return AreEqual(c,b,threshold);
    }

    template<class ElemType>
    bool GPUSparseMatrix<ElemType>::AreEqual(const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, 
        const ElemType threshold)
    {
        if (a.GetNZElements()!=b.GetNumElements() || a.GetNumRows()  != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
            return false;
        GPUSparseMatrix<ElemType> c;
        c.SetValue(b);
        return AreEqual(a,c,threshold);
    }

    template<class ElemType>
    bool GPUSparseMatrix<ElemType>::IsEqualTo(const GPUSparseMatrix<ElemType>& a, const ElemType threshold) const
    {
        return AreEqual(*this,a,threshold);
    }

    template<class ElemType>
    bool GPUSparseMatrix<ElemType>::IsEqualTo(const GPUMatrix<ElemType>& a, const ElemType threshold) const
    {
        return AreEqual(*this,a,threshold);
    }
#pragma endregion Static BLAS Functions

#pragma region Member BLAS Functions

    template<class ElemType>
    int GPUSparseMatrix<ElemType>::GetComputeDeviceId() const 
    {
        // for externally managed memory the CUDA context will have the current device
        if (m_computeDevice == MANAGEDEXTERN)
        {
            int devId;
            assert(m_externalBuffer);
            CUDACALL(cudaGetDevice(&devId));
            return devId;
        }
        return m_computeDevice;
    }

    template<class ElemType>
    GPUMatrix<ElemType> GPUSparseMatrix<ElemType>::ElementProductOf (const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
    {
        if (a.GetNumRows()!=b.GetNumRows()||a.GetNumCols()!=b.GetNumCols())
            throw std::logic_error("ElementProductOf: matrix dimensions mismatch");

        b.PrepareDevice();        
        GPUMatrix<ElemType> c(b.GetNumRows(),b.GetNumCols(),b.GetComputeDeviceId());

        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));
        long M=(long)a.GetNumRows();
        int blocksPerGrid =(int)ceil(1.0*M/threadsPerBlock);        
        _sparseMulDense<ElemType><<<blocksPerGrid,threadsPerBlock>>>(a.NzValues(),a.RowLocation(),a.ColLocation(),b.BufferPointer(),c.BufferPointer(),M);
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));
        CUDACALL(cudaEventDestroy(done));
        return c;
    }

    template<class ElemType>
    GPUMatrix<ElemType> GPUSparseMatrix<ElemType>::ElementProductOf (const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b)
    {
        return GPUSparseMatrix<ElemType>::ElementProductOf(b,a);        
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType> GPUSparseMatrix<ElemType>::operator+ (const GPUSparseMatrix<ElemType>& a) const
    {
        GPUSparseMatrix<ElemType> res;
        GPUSparseMatrix<ElemType>::ScaleAndAdd(1,*this,1,a,res);
        return res;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType> GPUSparseMatrix<ElemType>::operator- (const GPUSparseMatrix<ElemType>& a) const
    {
        GPUSparseMatrix<ElemType> res;
        GPUSparseMatrix<ElemType>::ScaleAndAdd(1,*this,-1,a,res);
        return res;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::operator^=(ElemType alpha)
    {
        GPUSparseMatrix<ElemType>& us = *this;
        ElementWisePower(alpha, us, us);
        return us;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType> GPUSparseMatrix<ElemType>::operator^ (ElemType alpha) const
    {
        GPUSparseMatrix<ElemType> c;
        c.ResizeAsAndCopyIndexFrom(*this);
        ElementWisePower(alpha, *this, c);
        return c;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::operator*=(ElemType alpha)
    {
        GPUSparseMatrix<ElemType>& us = *this;
        if (alpha!=1)            
            Scale(alpha,us);
        return us;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType> GPUSparseMatrix<ElemType>::operator* (ElemType alpha) const
    {
        GPUSparseMatrix<ElemType> c(*this);
        if (alpha!=1)
            Scale(alpha, c);
        return c;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignElementPowerOf(const GPUSparseMatrix<ElemType>& a, const ElemType power)
    {
        ElementWisePower(power, a, *this);
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType> GPUSparseMatrix<ElemType>::Transpose() const
    {
        int m = (int)GetNumRows();
        int n = (int)GetNumCols();
        int nnz = (int)GetNZElements();
        cusparseAction_t cpVals = CUSPARSE_ACTION_NUMERIC;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;

        assert(GetFormat()&matrixFormatCompressed); // for now this only supports compressed formats
        PrepareDevice();
        GPUSparseMatrix c(n, m, nnz, nullptr, GetFormat(), GetComputeDeviceId(), m_elemSizeAllocated);
        CUDACALL(cudaMalloc((void **)&c.m_pArray,c.BufferSize()));

        cusparseHandle_t cusparseHandle = 0;
        CUSPARSECALL(cusparseCreate(&cusparseHandle));

        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));
        if (sizeof(ElemType)==sizeof(float))
        {
            CUSPARSECALL(cusparseScsr2csc(cusparseHandle,m,n,nnz,reinterpret_cast<const float*>(this->NzValues()),this->SecondaryIndexLocation(),this->MajorIndexLocation(),
                reinterpret_cast<float*>(c.NzValues()),c.MajorIndexLocation(),c.SecondaryIndexLocation(),cpVals,idxBase));
        }
        else
        {
            CUSPARSECALL(cusparseDcsr2csc(cusparseHandle,m,n,nnz,reinterpret_cast<const double*>(this->NzValues()),this->SecondaryIndexLocation(),this->MajorIndexLocation(),
                reinterpret_cast<double*>(c.NzValues()),c.MajorIndexLocation(),c.SecondaryIndexLocation(),cpVals,idxBase));
        }
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done)); 
        CUDACALL(cudaEventDestroy(done));
        CUSPARSECALL(cusparseDestroy(cusparseHandle));        
        return c;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignTransposeOf(const GPUSparseMatrix<ElemType>& a)
    {
        if (this == &a)
            throw std::logic_error("AssignTransposeOf: a is the same as [this]. Does not support inplace transpose.");

        if (a.IsEmpty())
            throw std::logic_error("AssignTransposeOf: Matrix a is empty.");

        *this = a.Transpose();
        return *this;
    }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::InplaceTranspose()
    {
        if (IsEmpty())
            return;
        // transfer converted block over to this pointer
        *this = std::move(Transpose());
    }

    template<class ElemType>
    ElemType GPUSparseMatrix<ElemType>::SumOfAbsElements() const
    {
        if (IsEmpty())
            throw std::logic_error("SumOfAbsElements: Matrix is empty");

        cublasHandle_t cuHandle = GPUMatrix<ElemType>::GetCublasHandle(GetComputeDeviceId());
        if (sizeof(ElemType)==sizeof(float))
        {
            float res=0;
            cublasSasum(cuHandle,(int)GetNZElements(),reinterpret_cast<float*>(m_pArray),1,&res);
            return res;
        }
        else
        {
            double res=0;
            cublasDasum(cuHandle,(int)GetNZElements(),reinterpret_cast<double*>(m_pArray),1,&res);
            return ElemType(res);
        }         
    }

    template<class ElemType>
    ElemType GPUSparseMatrix<ElemType>::SumOfElements() const
    {
        if (IsEmpty())
            throw std::logic_error("SumOfElements: Matrix is empty");

        PrepareDevice();
        ElemType* d_sum = nullptr;
        ElemType h_sum;
        CUDACALL(cudaMalloc((void**)&d_sum,sizeof(ElemType)));
        //WARNING: THIS kernel is not the most efficient way!
        _reductionSum<ElemType><<<1,1024>>>(m_pArray,d_sum,(LONG64)GetNZElements());
        CUDACALL(cudaMemcpy(&h_sum,d_sum,sizeof(ElemType),cudaMemcpyDeviceToHost));
        CUDACALL(cudaFree(d_sum));               
        return h_sum;        
    }


    template<class ElemType>
    ElemType GPUSparseMatrix<ElemType>::FrobeniusNorm() const 
    {
        if (IsEmpty())
            throw std::logic_error("FrobeniusNorm: Matrix is empty.");

        ElemType* d_sum = nullptr;
        ElemType h_sum=0;
        CUDACALL(cudaMalloc((void**)&d_sum,sizeof(ElemType)));
        //WARNING: THIS kernel is not the most efficient way!
        _reductionSum2<ElemType><<<1,1024>>>(m_pArray,d_sum,(int)GetNZElements());
        CUDACALL(cudaMemcpy(&h_sum,d_sum,sizeof(ElemType),cudaMemcpyDeviceToHost));
        CUDACALL(cudaFree(d_sum));               
        if (sizeof(ElemType)==sizeof(float))
            return (ElemType)sqrtf((float)h_sum);
        else
            return (ElemType)sqrt((double)h_sum);
    }

    template<class ElemType>
    ElemType GPUSparseMatrix<ElemType>::MatrixNormInf() const
    {
        if (IsEmpty())
            throw std::logic_error("MatrixNorm1: Matrix is empty.");

        ElemType* d_maxAbs = nullptr;
        ElemType h_maxAbs=0;
        CUDACALL(cudaMalloc((void**)&d_maxAbs,sizeof(ElemType)));
        //WARNING: THIS kernel is not the most efficient way!
        _reductionMatrixNormInf<ElemType><<<1,1024>>>(m_pArray,d_maxAbs,(int)GetNZElements());
        CUDACALL(cudaMemcpy(&h_maxAbs,d_maxAbs,sizeof(ElemType),cudaMemcpyDeviceToHost));
        CUDACALL(cudaFree(d_maxAbs));               
        if (sizeof(ElemType)==sizeof(float))
            return h_maxAbs;
        else
            return h_maxAbs; 
    }

    template<class ElemType>
    ElemType GPUSparseMatrix<ElemType>::MatrixNorm1() const
    {
        if (IsEmpty())
            throw std::logic_error("MatrixNorm1: Matrix is empty.");
        return SumOfAbsElements();              
    }

#pragma endregion Member BLAS Functions

#pragma region Other Functions

    template<class ElemType>    
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::ElementInverse ()
    {
        if (IsEmpty())
            throw std::logic_error("ElementInverse: Matrix is empty.");

        long N=(long)GetNZElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);                
        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));        
        _elemInverse<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_pArray,N);
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));        
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignElementInverseOf (const GPUSparseMatrix<ElemType>& a)
    {
        SetValue(a);
        return ElementInverse();
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceSigmoid()
    {
        performInplaceFunction(0);                    
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignSigmoidOf (const GPUSparseMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceSigmoid();
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceLinearRectifierDerivative()
    {
        performInplaceFunction(6);                    
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignLinearRectifierDerivativeOf (const GPUSparseMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceLinearRectifierDerivative();
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceTanh()
    {
        performInplaceFunction(1);
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignTanhOf (const GPUSparseMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceTanh();
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceSqrt()
    {
        performInplaceFunction(2);        
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignSqrtOf (const GPUSparseMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceSqrt();
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceExp()
    {
        performInplaceFunction(3);        
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignExpOf (const GPUSparseMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceExp();
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceLog()
    {
        performInplaceFunction(4);        
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignLogOf (const GPUSparseMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceLog();
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceAbs()
    {
        performInplaceFunction(5);        
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignAbsOf (const GPUSparseMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceAbs();
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceTruncateBottom (const ElemType threshold)
    {
        if (IsEmpty())
            throw std::logic_error("InplaceTruncateBottom: Matrix is empty.");
        long N=(long)GetNZElements();
        int blocksPerGrid =(int)ceil(N*1.0/threadsPerBlock);                
        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));        
        _inplaceTruncateBottom<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_pArray,threshold,N);
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done)); 
        return *this;
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignTruncateBottomOf (const GPUSparseMatrix<ElemType>& a, const ElemType threshold)
    {
        if (a.IsEmpty())
            throw std::logic_error("AssignTruncateBottomOf: Matrix a is empty.");

        if (this!=&a)
        {
            //Resize(a.GetNumRows(), a.GetNumCols());           
            ResizeAsAndCopyIndexFrom(a);  
        }
        long N=(long)GetNZElements();
        int blocksPerGrid =(int)ceil(N*1.0/threadsPerBlock);                
        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));        
        _assignTruncateBottom<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_pArray,a.NzValues(),threshold,N);                        
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));
        return *this;
    }   

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceTruncateTop (const ElemType threshold)
    {
        if (IsEmpty())
            throw std::logic_error("InplaceTruncateTop: Matrix is empty.");
        long N=(long)GetNZElements();
        int blocksPerGrid =(int)ceil(N*1.0/threadsPerBlock);                
        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));        
        _inplaceTruncateTop<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_pArray,threshold,N);
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done)); 
        return *this;        
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignTruncateTopOf (const GPUSparseMatrix<ElemType>& a, const ElemType threshold)
    {
        if (a.IsEmpty())
            throw std::logic_error("AssignTruncateTopOf: Matrix a is empty.");

        if (this!=&a)
        {
            ResizeAsAndCopyIndexFrom(a);
        }

        long N=(long)GetNZElements();
        int blocksPerGrid =(int)ceil(N*1.0/threadsPerBlock);                
        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));        
        _assignTruncateTop<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_pArray,a.NzValues(),threshold,N);                        
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));
        return *this;        
    }

    template<class ElemType>
    GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::SetToZeroIfAbsLessThan (const ElemType threshold)
    {
        if (IsEmpty())
            throw std::logic_error("SetToZeroIfAbsLessThan: Matrix is empty.");
        long N=(long)GetNZElements();
        int blocksPerGrid =(int)ceil(N*1.0/threadsPerBlock);                
        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));        
        _setToZeroIfAbsLessThan<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_pArray,threshold,N);
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done)); 
        return *this;  
    }
    template<class ElemType>
    void GPUSparseMatrix<ElemType>::Unrolling (//GPUSparseMatrix<ElemType>& debugMatrix, 
        GPUMatrix<ElemType>& /*UnrolledMatrix*/, const GPUMatrix<ElemType>& /*InMatrix*/, GPUSparseMatrix<ElemType>& /*UnrollMapping*/, 
        const int /*inputWidth*/, const int /*inputHeight*/, const int /*inputChannelNum*/,
        const int /*FltWidth*/,const int /*FltHeight*/, const int /*FltChannel*/,
        const int /*FltStepW*/,  const int /*FltStepH*/)
    {
        ////if ((UnrolledMatrix.m_computeDevice!=InMatrix.m_computeDevice) ||(InMatrix.m_computeDevice!=UnrollMapping.m_computeDevice)) //different GPUs
        ////{
        ////    throw std::invalid_argument("All matrices must be on the same GPU");
        ////}
        ////else
        ////{ 
        //    //m_computeDevice = deviceId;

        //    const int inPatchSize = inputWidth * inputHeight;// * inputChannelNum;
        //    const int inRowHeight = InMatrix.GetNumRows();//m_inSampleNum;
        //    const int inColWidth = InMatrix.GetNumCols();
        //    const int inChannelNum = inputChannelNum;//column as sample VS column as channel//inColWidth;
        //    const int inSampleNum = inColWidth;// //inRowHeight / inPatchSize ;
        //    const int filterPatchSize = FltWidth * FltHeight;
        //    const int outWidth = inputWidth + 2 * (FltWidth - 1); // - FltWidth + 1; // Filter Width Step = 1; with padding
        //    const int outHeight = inputHeight + 2 * (FltHeight -1);//inputHeight - FltHeight + 1; 
        //    const int outWidthFltNum = ceil( double(outWidth - FltWidth + 1) / FltStepW);
        //    const int outHeightFltNum = ceil( double(outHeight - FltHeight + 1) /FltStepH);
        //    //const int convNum = outWidth * outHeight;
        //    //auto& UnrolledMatrix=*this;

        //    const int unrolledRowNum = outHeightFltNum * outWidthFltNum * inChannelNum;//Number of Filters Per Sample//outHeightFltNum * outWidthFltNum;
        //    const int unrolledColNum = filterPatchSize * inSampleNum;//filterPatchSize * inChannelNum;
        //    if (UnrolledMatrix.IsEmpty())
        //        UnrolledMatrix = GPUMatrix<ElemType>::Zeros(unrolledRowNum, unrolledColNum);//UnrolledMatrix.ZeroInit();
        //    //UnrollMapping.SetValue(-1);
        //    long N = inRowHeight * inColWidth; //total number of threads
        //    int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);
        //    //CUDA_CALL(cudaSetDevice(InMatrix.m_computeDevice));
        //    ElemType* d_unrolledMatrix;
        //    ElemType* d_unrollMapping;
        //    const int outArraySize = unrolledRowNum * unrolledColNum;
        //    UnrollMapping.ZeroInit();

        //    //GPUSparseMatrix<ElemType>UnrollMapping;// = ZeroInit();//GPUSparseMatrix(InMatrix.GetNumElements(), UnrolledMatrix.GetNumElements());

        //    //const int _debugSize = unrolledRowNum * unrolledColNum;

        //    //int* d_debugArray; 
        //    //CUDA_CALL(cudaMalloc((void**)&d_debugArray, _debugSize * sizeof(int)));
        //    //CUDA_CALL(cudaMemcpy(d_debugArray, debugMatrix, _debugSize *sizeof(int),cudaMemcpyHostToDevice)); 



        //    if (FltStepW == 1 && FltStepH == 1)
        //        _unrollElem_noStride<ElemType><<<blocksPerGrid, threadsPerBlock>>>(
        //        UnrolledMatrix.BufferPointer(), InMatrix.BufferPointer(), UnrollMapping.m_pArray,
        //        inRowHeight, inColWidth, 
        //        inputWidth, inputHeight, inputChannelNum,
        //        FltWidth,FltHeight, FltChannel,
        //        inPatchSize, outWidthFltNum,outHeightFltNum,
        //        unrolledRowNum, unrolledColNum);
        //    else
        //    {
        //        _unrollElem_Stride<ElemType><<<blocksPerGrid, threadsPerBlock>>> (
        //            UnrolledMatrix.BufferPointer(), InMatrix.BufferPointer(), UnrollMapping.m_pArray,
        //            inRowHeight, inColWidth, 
        //            inputWidth, inputHeight, inputChannelNum,
        //            FltWidth,FltHeight, FltChannel,
        //            outWidthFltNum, outHeightFltNum,
        //            FltStepW,  FltStepH,
        //            unrolledRowNum, unrolledColNum);
        //    }           
        //    //CUDA_CALL(cudaMemcpy(debugMatrix, d_debugArray, _debugSize *sizeof(int),cudaMemcpyDeviceToHost)); 

        ////}
    }

#pragma endregion

#pragma region Helper Functions

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::performInplaceFunction(int kind)
    {        
        long N=(long)GetNZElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);                
        cudaEvent_t done = nullptr;
        CUDACALL(cudaEventCreate(&done));        
        switch (kind)
        {
        case 0:
            _inplaceSigmoidOnCuda<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_pArray,N);
            break;
        case 1:
            _inplaceTanhOnCuda<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_pArray,N);
            break;
        case 2:
            _inplaceSqrtOnCuda<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_pArray,N);
            break;
        case 3:
            _inplaceExpOnCuda<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_pArray,N);
            break;
        case 4:
            _inplaceLogOnCuda<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_pArray,N);
            break;
        case 5:
            _inplaceAbsOnCuda<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_pArray,N);
            break;
        case 6:
            _inplaceLinRectDerivative<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_pArray,N);
        } 
        CUDACALL(cudaEventRecord(done));        
        CUDACALL(cudaEventSynchronize(done));        
    }

    template<class ElemType>
    void GPUSparseMatrix<ElemType>::SetMatrixFromCSRFormat(const GPUSPARSE_INDEX_TYPE *h_CSRRow, const GPUSPARSE_INDEX_TYPE *h_Col, const ElemType *h_Val,
        const size_t nz, const size_t numRows, const size_t numCols, const bool IsOnDevice /*= false*/, const DEVICEID_TYPE devId /*= 0*/)
    {
        m_computeDevice = devId;
        m_elemSizeAllocated = m_nz = nz;
        m_numCols=numCols;
        m_numRows=numRows;  
        m_format=matrixFormatSparseCSR;
        m_externalBuffer = false;

        if (OwnBuffer() && m_pArray != NULL)
        {
            CUDACALL(cudaFree(m_pArray));            
        }

        PrepareDevice();
        CUDACALL(cudaMalloc((void **)&m_pArray,BufferSize()));

        cudaMemcpyKind kind = IsOnDevice?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice;
        CUDACALL(cudaMemcpy(RowLocation(),h_CSRRow,RowSize(),kind));
        CUDACALL(cudaMemcpy(ColLocation(),h_Col,ColSize(),kind));
        CUDACALL(cudaMemcpy(NzValues(),h_Val,NzSize(),kind));
    }

    // NOTE: we should change this to just use a single buffer, and return pointers into it
    template<class ElemType>
    void GPUSparseMatrix<ElemType>::GetMatrixFromCSRFormat(int*& h_CSRRow, int*& h_Col, ElemType*& h_Val, size_t &nz, size_t &numRows, size_t &numCols) const
    {
        if (h_CSRRow!=nullptr || h_Col!=nullptr || h_Val!=nullptr)
            throw std::exception("Passed pointers must be nullptr");
        nz = GetNZElements();
        numRows = GetNumRows();
        numCols = GetNumCols();

        if (IsEmpty())
            return;
        else
        {
            PrepareDevice();
            h_Val = new ElemType[nz];
            h_CSRRow = new int[m_numRows + 1];
            h_Col = new int[nz];

            CUDACALL(cudaMemcpy(h_CSRRow,RowLocation(),RowSize(),cudaMemcpyDeviceToHost));
            CUDACALL(cudaMemcpy(h_Col,   ColLocation(),ColSize(),cudaMemcpyDeviceToHost));
            CUDACALL(cudaMemcpy(h_Val,   NzValues(), NzSize(), cudaMemcpyDeviceToHost));
        }
    }

#pragma endregion Helper Functions

    template class GPUSparseMatrix<float>; 
    template class GPUSparseMatrix<double>;    

    template <class ElemType>
    MATH_API File& operator>>(File& stream, GPUSparseMatrix<ElemType>& us)
    {
        stream.GetMarker(fileMarkerBeginSection, std::wstring(L"BMAT"));
        size_t elsize;
        stream>>elsize;
        if (sizeof(ElemType)!=elsize)
            throw std::runtime_error("Template argument size doesn't match those in file");
        std::wstring matrixName;

        // save off the buffer size being passed in
        ElemType* deviceBuffer = us.m_pArray;
        size_t deviceBufferSize = us.BufferSize();

        // now prepare this header to receive the data being read
        // Once CPUSpareMatrix uses same format, should use that class
        size_t nz, colnum, rownum;
        int format;

        // read in the header information
        stream>>matrixName>>format>>nz>>colnum>>rownum;
        us.m_format = (MatrixFormat)format;
        us.m_numCols = colnum;
        us.m_numRows = rownum;
        us.m_elemSizeAllocated = us.m_nz = nz;
        us.m_externalBuffer = false;

        // temporarily allocate a CPU side array here (could use CPUSparseMatrix when has same format)
        ElemType* hostBuffer = new ElemType[us.BufferSize()];
        us.m_pArray = hostBuffer;
        ElemType *dVal=us.NzValues();
        int* idx=us.MajorIndexLocation();
        int* cidx=us.SecondaryIndexLocation();
        size_t ncidx = us.SecondaryIndexCount();

        // read in the sparse matrix info
        for (int i=0;i<nz;++i)
        {
            stream>>dVal[i];
        }
        for (int i=0;i<nz;++i)
        {
            stream>>idx[i];
        }
        for (int i=0;i<ncidx;++i)
        {
            stream>>cidx[i];
        }  

        // decide if we have enough room in the current buffer
        if (deviceBufferSize >= us.BufferSize())
        {
            us.m_elemSizeAllocated = us.ElemCountFromBufferSize(deviceBufferSize);
        }
        else
        {
            us.PrepareDevice();
            if (deviceBufferSize > 0)
                CUDACALL(cudaFree((void **)&deviceBuffer));
            CUDACALL(cudaMalloc((void **)&us.m_pArray, us.BufferSize()));
        }

        // copy over the different sections data
        CUDACALL(cudaMemcpy(us.NzValues(),dVal,us.NzSize(),cudaMemcpyHostToDevice));
        CUDACALL(cudaMemcpy(us.MajorIndexLocation(),idx,us.MajorIndexSize(),cudaMemcpyHostToDevice));
        CUDACALL(cudaMemcpy(us.SecondaryIndexLocation(),cidx,us.SecondaryIndexSize(),cudaMemcpyHostToDevice));

        // copy over the name if necessary
        if (us.m_matrixName != nullptr)
            delete us.m_matrixName;
        us.m_matrixName = new wchar_t[matrixName.length()+1];
        wmemcpy(us.m_matrixName,matrixName.c_str(),matrixName.length()+1);

        return stream;
    }

    template MATH_API File& operator>>(File& stream, GPUSparseMatrix<float>& us);
    template MATH_API File& operator>>(File& stream, GPUSparseMatrix<double>& us);

    template <class ElemType>
    MATH_API File& operator<<(File& stream, const GPUSparseMatrix<ElemType>& us)
    {
        stream.PutMarker(fileMarkerBeginSection, std::wstring(L"BMAT"));
        stream<<sizeof(ElemType);
        if (us.GetMatrixName()==nullptr)
        {
            std::wstring s(L"nnmatrix");
            stream<<s;
        }
        else
        {
            stream<<us.GetMatrixName();
        }

        // What we would like to do here, is transfer to CPUSparse and save, do that when the format is the same
        char* hostBuffer = new char[us.BufferSize()];       // TODO: use std::shared_ptr
        GPUSparseMatrix<ElemType> hostSide(us.GetNumRows(), us.GetNumCols(), us.NzCount(), (ElemType*)hostBuffer, us.GetFormat());
        CUDACALL(cudaMemcpy(hostBuffer, us.NzValues(),us.BufferSize(),cudaMemcpyDeviceToHost));

        // now setup all the stuff pointing to the CPU side info
        const ElemType *dVal=hostSide.NzValues();
        int* idx=hostSide.MajorIndexLocation();
        int* cidx=hostSide.SecondaryIndexLocation();
        size_t nz=us.NzCount();
        size_t ncidx=us.SecondaryIndexCount();
        int format = us.GetFormat();
        stream<<format<<nz<<us.GetNumCols()<<us.GetNumRows();
        for (int i=0;i<nz;++i)
        {
            stream<<dVal[i];
        }
        for (int i=0;i<nz;++i)
        {
            stream<<idx[i];
        }
        for (int i=0;i<ncidx;++i)
        {
            stream<<cidx[i];
        }
        stream.PutMarker(fileMarkerEndSection, std::wstring(L"EMAT"));

        // now release the hostSide buffer
        delete hostBuffer;
        hostSide.m_pArray = nullptr;

        return stream;
    }
    template MATH_API File& operator<<(File& stream, const GPUSparseMatrix<float>& us);
    template MATH_API File& operator<<(File& stream, const GPUSparseMatrix<double>& us);

}}}

#endif  // CPUONLY
