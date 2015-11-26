//
// <copyright file="GPUMatrix.cu" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#include "BestGpu.h"
#include "DebugUtil.h"

#ifndef CPUONLY

#include "cublas_v2.h"
#include "Basics.h"
#include "GPUMatrix.h"
#include "GPUMatrixCUDAKernels.cuh"
#include "GPUSparseMatrix.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#pragma comment (lib, "cudart.lib")     // instruct linker to reference these libs
#pragma comment (lib, "cublas.lib")
#pragma comment (lib, "cusparse.lib")
#pragma comment (lib, "curand.lib")

#pragma warning (disable: 4267) // conversion from 'size_t' to 'unsigned int'; happens in CUDA <<<a,b>>> syntax if a and b are size_t
#pragma warning (disable: 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this
#pragma warning (disable: 4702) // unreachable code; triggered for unknown reasons

#ifdef NO_SYNC
bool do_sync = false;
#else
bool do_sync = true;
#endif

#define DEFAULT_THREAD_PER_DIM		16

#define UNCONST(t,c,uc)  GPUMatrix<t> &uc = const_cast<GPUMatrix<t>&>(c);

#ifdef _WIN32
// thread local storage to access the current stream, initalize to default stream
__declspec (thread)
#else
static
#endif
cudaStream_t t_stream = cudaStreamDefault;

#define DEFAULT_THREAD_PER_DIM		16

extern int _ConvertSMVer2Cores(int major, int minor);   // forward declaration

// SetStream - set the stream that will be used by the GPU routines
void MATH_API SetStream(cudaStream_t stream)
{
    t_stream = stream;
}

// GetStream - get the stream that will be used by the GPU routines
cudaStream_t MATH_API GetStream()
{
    return t_stream;
}


static const char * CudaErrString(cudaError_t x)  { cudaDeviceSynchronize(); return cudaGetErrorString(x); }
static const char * CudaErrString(cublasStatus_t) { cudaDeviceSynchronize(); return "(see cublas_api.h & look for cublasStatus_t or CUBLAS_STATUS_xxx)"; }
static const char * CudaErrString(curandStatus)   { cudaDeviceSynchronize(); return "(see curand.h & look for curandStatus or CURAND_STATUS_xxx)"; }

namespace Microsoft { namespace MSR { namespace CNTK {

    // PrepareDevice - Setup the correct cuda context for an operation
    // deviceId - the device on which the operation will take place
    void PrepareDevice(DEVICEID_TYPE deviceId)
    {
        static DEVICEID_TYPE currentDevice = AUTOPLACEMATRIX; // set to anything valid
        // and if we last set the device to be this device we are good
        if (deviceId == currentDevice)
            return;
        CUDA_CALL(cudaSetDevice(deviceId));
        currentDevice=deviceId;
    }

#pragma region DeviceBoundNumber class

    template<class ElemType>
    DeviceBoundNumber<ElemType>::DeviceBoundNumber(const DeviceBoundNumber<ElemType> &/*deepCopy*/)
    {
        NOT_IMPLEMENTED;
    }

    template<class ElemType>
    DeviceBoundNumber<ElemType>::DeviceBoundNumber(DeviceBoundNumber<ElemType> &&shallowCopy)
    {
        ShallowCopyFrom(shallowCopy.m_data,shallowCopy.m_computeDevice);
        shallowCopy.m_data=NULL;
    }

    template<class ElemType>
    void DeviceBoundNumber<ElemType>::ShallowCopyFrom(ElemType* newVal,int newValsDevceId)
    {
        m_computeDevice = newValsDevceId;
        m_data = newVal;
    }

    template<class ElemType>
    DeviceBoundNumber<ElemType>::~DeviceBoundNumber()
    {
        if (m_data!=NULL)
        {
            if (m_computeDevice<0)
            {
                delete m_data;
                m_data = NULL;
            }
            else
                CUDA_CALL(cudaFree(m_data));
        }
    }

#pragma endregion DeviceBoundNumber class

#pragma region Helper functions
    template<class ElemType>    
    cublasHandle_t _initCUBLAS(int devId)
    {
        PrepareDevice((DEVICEID_TYPE)devId);
        cublasHandle_t cuHandle;
        CUBLAS_CALL(cublasCreate(&cuHandle));
        return cuHandle;
    }

    // GetBestGPUDeviceId - Get the best GPU DeviceId, based on cuda information
    // Returns -1 if no GPUs can be used.
    //  TODO: should be replaced by BestGpu class instead, it's much better
    static DEVICEID_TYPE SelectBestGPUDeviceId()        // this is an internal version that is wrapped by GPUMatrix<ElemType>::GetBestGPUDeviceId() below
    {      
        // currently there is little point in giving out different device IDs each time ask for a matrix, 
        // we really want them all on the same device eventually
        static int chosenDeviceId = AUTOPLACEMATRIX;
        if (chosenDeviceId != AUTOPLACEMATRIX)
            return chosenDeviceId;
#ifdef __WINDOWS__
        __try
#endif
        {
            // stash previous device state
            // if there was one on entry:
            int nPrevDev = -1;
            cudaError_t ePrevDev = cudaGetDevice(&nPrevDev);
 
            int deviceCount = -1;
            cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
            if (error_id != cudaSuccess || deviceCount==0) 
            { 
                return -1;            
            }

            int setDev = -1;
            int curDev=0;
            CUDA_LONG curPower = 0;
            for (DEVICEID_TYPE dev = 0; dev < deviceCount; ++dev)
            {
                CUDA_CALL(cudaSetDevice(dev));
                setDev = dev;
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, dev);
                int power = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
                //CUDA_LONG power = _GetFreeMemoryOnCUDADevice(dev);
                if (power>curPower)
                {
                    curPower=power;
                    curDev = dev;
                }
            }

            if(nPrevDev >= 0 && ePrevDev == cudaSuccess && 
                setDev >= 0 && setDev != nPrevDev) {
                // restore current context to the one we entered with
                // if there was one the caller might want unchanged.
                cudaSetDevice(nPrevDev);
            }
            chosenDeviceId = curDev;
            return curDev;
        }
#ifdef __WINDOWS__
        __except (1)
        {
            return -1; // CPU
        }
#endif
    }
    template<class ElemType>
    /*static*/ DEVICEID_TYPE GPUMatrix<ElemType>::GetBestGPUDeviceId() //returns -1 if no GPUs can be used
    {
        // route the result through EnforceOneGPUOnly() which only lets the first choice through (see comment there)
        return EnforceOneGPUOnly(SelectBestGPUDeviceId());
    }

    // PrepareDevice - Setup the correct cuda context for an operation
    // deviceId - the device on which the operation will take place
    //            defaults to -1, which means use matrices current device
    template<class ElemType>
    DEVICEID_TYPE GPUMatrix<ElemType>::PrepareDevice(DEVICEID_TYPE deviceId /*=-1*/) const
    {
        // if default value use current compute device
        DEVICEID_TYPE newId = deviceId >= 0 ? deviceId : m_computeDevice;

        Microsoft::MSR::CNTK::PrepareDevice(newId);
        return newId;
    }

    template<class ElemType>
    ElemType* GPUMatrix<ElemType>::CopyToArray() const
    {
        size_t numElements = GetNumElements();
        if (numElements != 0)
        {
            PrepareDevice();
            ElemType* pArray = new ElemType[numElements];                    
            CUDA_CALL(cudaMemcpy(pArray,m_pArray,sizeof(ElemType)*m_numRows*m_numCols,cudaMemcpyDeviceToHost));
            return pArray;
        }
        else
        {
            return NULL;
        }
    }

    //memory will be allocated by the callee if not enough but need to be deleted by the caller after it's done
    //return number of elements copied
    template<class ElemType>
    size_t  GPUMatrix<ElemType>::CopyToArray(ElemType*& arrayCopyTo, size_t& currentArraySize) const
    {
        size_t numElements = GetNumElements();

        if (numElements > currentArraySize)
        {
            delete arrayCopyTo;
            arrayCopyTo = new ElemType[numElements];  
            currentArraySize = numElements;
        }

        if (numElements != 0)
        {
            PrepareDevice();
            CUDA_CALL(cudaMemcpy(arrayCopyTo, m_pArray, sizeof(ElemType)*numElements, cudaMemcpyDeviceToHost));
        }

        return numElements;
    }

    template <typename ElemType>
    void GPUMatrix<ElemType>::CopySection(size_t numRows, size_t numCols, ElemType* dst, size_t colStride) const
    {
        CUBLAS_CALL(cublasGetMatrix((int)numRows, (int)numCols, sizeof(ElemType),
            m_pArray, (int)GetNumRows(), dst, (int)colStride));
    }
    template<class ElemType>
    void GPUMatrix<ElemType>::ChangeDeviceTo(DEVICEID_TYPE to_id)
    {
        if (!OwnBuffer())
            LogicError("Cannot change device on Managed external matrix");
        if (to_id == CPUDEVICE)
            LogicError("to_id must be valid GPU");
        if (m_computeDevice==to_id) 
            return;

        PrepareDevice((DEVICEID_TYPE)to_id);       
        ElemType* d_dst=NULL;
        CUDA_CALL(cudaMalloc((void**)&d_dst,sizeof(ElemType)*m_numRows*m_numCols));

        m_elemSizeAllocated = m_numRows*m_numCols;

        // check to make sure we have something to copy (on init we often have zero sized allocations)
        if (m_elemSizeAllocated > 0)
        {
            // first try peer access
            int canAccessPeer = false;
            CUDA_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, to_id, m_computeDevice));
            if (canAccessPeer)
            {
                cudaError_t cudaStatus = cudaDeviceEnablePeerAccess(m_computeDevice, 0);
                if (cudaStatus != cudaErrorPeerAccessAlreadyEnabled)
                {
                    CUDA_CALL(cudaStatus);
                }
                CUDA_CALL(cudaMemcpyPeer(d_dst,to_id,m_pArray,m_computeDevice,sizeof(ElemType)*m_numRows*m_numCols));  
            }
            else
            {
                // peer access didn't work, just copy normal
                // make this more efficient by keeping some buffers available for each copy
                ElemType* h_dst=NULL;
                PrepareDevice();
                CUDA_CALL(cudaMallocHost((void**)&h_dst,sizeof(ElemType)*m_numRows*m_numCols));
                CUDA_CALL(cudaMemcpy(h_dst,m_pArray,sizeof(ElemType)*m_numRows*m_numCols, cudaMemcpyDeviceToHost));  
                PrepareDevice((DEVICEID_TYPE)to_id);       
                CUDA_CALL(cudaMemcpy(d_dst,h_dst,sizeof(ElemType)*m_numRows*m_numCols, cudaMemcpyHostToDevice)); 
                CUDA_CALL(cudaFreeHost(h_dst));  
            }
        }
        PrepareDevice();
        CUDA_CALL(cudaFree(m_pArray));
        m_pArray=d_dst;

        PrepareDevice((DEVICEID_TYPE)to_id);       
        m_computeDevice=to_id;
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::performInplaceFunction(int kind)    
    {        
        PrepareDevice();
        CUDA_LONG N= (CUDA_LONG) GetNumElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);                
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        switch (kind)
        {
        case 0:
            _inplaceSigmoidOnCuda<ElemType><<<blocksPerGrid, threadsPerBlock, 0, t_stream>>>(m_pArray, N);
            break;
        case 1:
            _inplaceTanhOnCuda<ElemType><<<blocksPerGrid, threadsPerBlock, 0, t_stream>>>(m_pArray, N);
            break;
        case 2:
            _inplaceSqrtOnCuda<ElemType><<<blocksPerGrid, threadsPerBlock, 0, t_stream>>>(m_pArray, N);
            break;
        case 3:
            _inplaceExpOnCuda<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,N);
            break;
        case 4:
            _inplaceLogOnCuda<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,N);
            break;
        case 5:
            _inplaceAbsOnCuda<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,N);
            break;
        case 6:
            _inplaceLinRectDerivative<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,N);
            break;
        case 7:
            _inplaceCosineOnCuda<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,N);
            break;
        case 8:
            _inplaceNegativeSineOnCuda<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,N);
            break;
        } 
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));       
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }


#pragma endregion Helper functions

#pragma region Constructors and Destructor

   //should only be used by constructors.
    template<class ElemType>
    void GPUMatrix<ElemType>::ZeroInit(int deviceId)
    {
        m_computeDevice = deviceId;
        m_pArray = nullptr;
        m_numRows = 0;
        m_numCols = 0;
        m_elemSizeAllocated = 0;
        m_matrixName=NULL;
        m_format = matrixFormatDense; 
        m_externalBuffer = false;
        m_workspace = nullptr;
    }

    template<class ElemType>
    GPUMatrix<ElemType>::GPUMatrix(int deviceId) 
    {
        ZeroInit(deviceId);
    };

    //matrixName is used to verify that correct matrix is read.
    template<class ElemType>
    GPUMatrix<ElemType>::GPUMatrix(FILE* f, const char * matrixName, int /*deviceId*/)
    {
        ReadFromFile(f, matrixName);
    }

    template<class ElemType>
    GPUMatrix<ElemType>::GPUMatrix(const size_t numRows, const size_t numCols,int deviceId)
    {
        ZeroInit(deviceId);
        m_numRows = numRows;
        m_numCols = numCols;
        m_elemSizeAllocated = GetNumElements();

        if (m_elemSizeAllocated != 0)
        {
            PrepareDevice();        
            CUDA_CALL(cudaMalloc((void**)&m_pArray,sizeof(ElemType)*m_elemSizeAllocated));      
            CUDA_CALL(cudaMemset(m_pArray,0,sizeof(ElemType)*m_elemSizeAllocated));  
        }
    };

    template<class ElemType>
    GPUMatrix<ElemType>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId, ElemType *pArray, const size_t matrixFlags)
    {
        ZeroInit(deviceId);
        SetValue(numRows, numCols, deviceId, pArray, matrixFlags);
    };               

    template<class ElemType>
    GPUMatrix<ElemType>::GPUMatrix(const GPUMatrix<ElemType>& deepCopyFrom)
    {
        ZeroInit(deepCopyFrom.m_computeDevice);
        SetValue(deepCopyFrom);
        SetMatrixName(deepCopyFrom.m_matrixName);       
    }

    template<class ElemType>
    GPUMatrix<ElemType>::GPUMatrix(GPUMatrix<ElemType>&& moveFrom)
    {
        m_numRows = moveFrom.m_numRows;
        m_numCols = moveFrom.m_numCols;
        m_computeDevice = moveFrom.m_computeDevice;
        m_pArray = moveFrom.m_pArray;  //shallow copy the pointer       
        m_matrixName=moveFrom.m_matrixName;
        m_elemSizeAllocated = moveFrom.m_elemSizeAllocated;
        m_format = moveFrom.m_format;
        m_externalBuffer = moveFrom.m_externalBuffer;

        //release the pointer from the source object so that the destructor won't release it twice
        moveFrom.ZeroInit(0);       
    }

    //assignment operator, deep copy
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator=(const GPUMatrix<ElemType>& deepCopyFrom)  
    {
        if (this != &deepCopyFrom)
        {
            SetValue(deepCopyFrom);
            SetMatrixName(deepCopyFrom.m_matrixName);       
        }
        return *this;
    }

    //move assignment operator, shallow copy
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator=(GPUMatrix<ElemType>&& moveFrom)  
    {
        if (this != &moveFrom)
        {
            if (OwnBuffer() && m_pArray!=NULL)
            {
                CUDA_CALL(cudaFree(m_pArray));  
            }

            m_numRows = moveFrom.m_numRows;
            m_numCols = moveFrom.m_numCols;
            m_elemSizeAllocated =  moveFrom.m_elemSizeAllocated;
            m_pArray = moveFrom.m_pArray;
            m_computeDevice = moveFrom.m_computeDevice;
            m_format = moveFrom.m_format;
            m_externalBuffer = moveFrom.m_externalBuffer;

            //release the pointer from the source object so that the destructor won't release it twice
            moveFrom.ZeroInit(0);
        }
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>::~GPUMatrix(void)
    {
        Clear();
        if (m_workspace != nullptr)
            delete m_workspace;
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::Clear()
    {
        if (OwnBuffer() && m_pArray!=NULL)
        {
            if (m_computeDevice>=0)
            {            
                PrepareDevice();
                cudaFree(m_pArray);
                m_pArray = NULL;
                m_elemSizeAllocated = 0;
            }        
        }
        BaseMatrix<ElemType>::Clear();

        ZeroInit(m_computeDevice);
    }
#pragma endregion Constructors and Destructor 

    template<class ElemType>
    int GPUMatrix<ElemType>::GetComputeDeviceId() const 
    {
        return m_computeDevice;
    }

    template<class ElemType>
    std::unique_ptr<GPUMatrix<ElemType>> GPUMatrix<ElemType>::GetOrCreateWorkspace() const
    {
        // REVIEW alexeyk: not thread-safe, fine for now.
        if (m_workspace == nullptr)
            m_workspace = new conc_stack<std::unique_ptr<GPUMatrix<ElemType>>>();
        assert(m_workspace != nullptr);
        auto deviceId = m_computeDevice;
        return m_workspace->pop_or_create([deviceId]() { return std::make_unique<GPUMatrix<ElemType>>(deviceId); });
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::ReleaseWorkspace(std::unique_ptr<GPUMatrix<ElemType>> src) const
    {
        assert(m_workspace != nullptr);
        m_workspace->push(std::move(src));
    }

#pragma region Basic Operators
    template<class ElemType>
    GPUMatrix<ElemType> GPUMatrix<ElemType>::ColumnSlice(size_t startColumn, size_t numCols) const
    {
        //if (numCols == 0)
        //    LogicError("The slice cannot have 0 columns.");

        if (startColumn + numCols > m_numCols)
            InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int)startColumn, (int)numCols, (int)m_numCols);
            
        GPUMatrix<ElemType> slice(m_numRows, numCols, m_computeDevice, m_pArray + startColumn * m_numRows, matrixFlagDontOwnBuffer);

        return slice;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignColumnSlice(const GPUMatrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols)
    {
        if (numCols == 0)
            LogicError("The slice cannot have 0 columns.");

        if (startColumn + numCols > fromMatrix.m_numCols)
            InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int)startColumn, (int)numCols, (int)fromMatrix.m_numCols);
        
        Clear();

        m_computeDevice=fromMatrix.m_computeDevice;
        m_externalBuffer=true;
        m_numRows = fromMatrix.m_numRows;
        m_numCols = numCols;
        m_pArray=fromMatrix.m_pArray + startColumn * m_numRows;

        m_elemSizeAllocated = GetNumElements();
        m_matrixName=NULL;
        m_format = fromMatrix.m_format;

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::SetColumnSlice(const GPUMatrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols)
    {
        //if (numCols == 0)
        //    LogicError("The slice cannot have 0 columns.");
        if (startColumn + numCols > m_numCols)
            LogicError("The slice is out of range of the destination matrix.");
        if (numCols > fromMatrix.GetNumCols())
            InvalidArgument("The slice (%d) is out of range of the source matrix (%d).", (int)numCols, (int)fromMatrix.GetNumCols());
        if (m_numRows != fromMatrix.m_numRows)
            LogicError("The number of rows in source and destination matrices do not match");

        if (m_numRows*numCols > 0)  // TODO: remove if unnecessary
        CUDA_CALL(cudaMemcpy(m_pArray + LocateColumn(startColumn), fromMatrix.m_pArray, sizeof(ElemType)*m_numRows*numCols, cudaMemcpyDeviceToDevice));
        return *this;
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::CopyColumnsStrided(const GPUMatrix<ElemType>& fromMatrix, size_t numCols, size_t srcNumColsStride, size_t destNumColsStride)
    {
        if ((((numCols - 1) * srcNumColsStride) + 1) > fromMatrix.m_numCols)
            LogicError("The numCols to copy and srcNumColsStride specified is out of range of the source matrix.");
        if ((((numCols - 1) * destNumColsStride) + 1) > m_numCols)
            LogicError("The numCols to copy and srcNumColsStride specified is out of range of the destination matrix.");
        if (m_numRows != fromMatrix.m_numRows)
            LogicError("The number of rows in source and destination matrices do not match");

        if ((m_numRows * numCols) > 0)
        {
            // Launch a kernel to do the strided copy
            CUDA_LONG N = (CUDA_LONG)(m_numRows * numCols);
            int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
            PrepareDevice();
            cudaEvent_t done = nullptr;
            if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
            _copyColumnsStrided<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(m_pArray, fromMatrix.m_pArray, N, (CUDA_LONG)m_numRows, (CUDA_LONG)destNumColsStride, (CUDA_LONG)srcNumColsStride);
            if (do_sync)    CUDA_CALL(cudaEventRecord(done));
            if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
            if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        }
    }

    //for each column of a, we assign all rows of a to this starting from startIndex
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignToRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
    {
        if (a.IsEmpty())
            LogicError("AddToRowSliceValuesOf: input matrix a is empty.");

        if (a.GetNumRows() != numRows)
            LogicError("AddToRowSliceValuesOf: a.GetNumRows() != numRows.");

        if (startIndex + numRows > GetNumRows())
            LogicError("AddToRowSliceValuesOf: startIndex + numRows exceeds GetNumRows().");

        if (a.GetNumCols() != GetNumCols())
            LogicError("AddToRowSliceValuesOf: columns does not match.");

        CUDA_LONG N = (CUDA_LONG)a.GetNumElements();
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _assignToRowSliceValuesOf<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(m_pArray, a.m_pArray, N, (CUDA_LONG)startIndex, (CUDA_LONG)GetNumRows(), (CUDA_LONG)a.GetNumRows());
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    //for each column of a, we assign numRows starting from startIndex to this
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
    {
        if (a.IsEmpty())
            LogicError("AssignRowSliceValuesOf: input matrix a is empty.");

        if (startIndex + numRows > a.GetNumRows())
            LogicError("AssignRowSliceValuesOf: startIndex + numRows exceeds a.GetNumRows().");

        Resize(numRows, a.GetNumCols());

        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _assignRowSliceValuesOf<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray, a.m_pArray, N, (CUDA_LONG)startIndex, (CUDA_LONG)numRows, (CUDA_LONG)a.GetNumRows());
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    //for the row slice of this starting from startIndex we add a to it.
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddToRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
    {
        if (a.IsEmpty())
            LogicError("AddToRowSliceValuesOf: input matrix a is empty.");

        if (a.GetNumRows() != numRows)
            LogicError("AddToRowSliceValuesOf: a.GetNumRows() != numRows.");

        if (startIndex + numRows > GetNumRows())
            LogicError("AddToRowSliceValuesOf: startIndex + numRows exceeds GetNumRows().");

        if (a.GetNumCols() != GetNumCols())
            LogicError("AddToRowSliceValuesOf: columns does not match.");

        CUDA_LONG N=(CUDA_LONG)a.GetNumElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _addToRowSliceValuesOf<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray, a.m_pArray, N, (CUDA_LONG)startIndex, (CUDA_LONG)GetNumRows(), (CUDA_LONG)a.GetNumRows());
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    //for each column of this, we add row slice of a starting from startIndex
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddWithRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
    {
        if (a.IsEmpty())
            LogicError("AddWithRowSliceValuesOf: input matrix a is empty.");

        if (GetNumRows() != numRows)
            LogicError("AddWithRowSliceValuesOf: GetNumRows() != numRows.");

        if (startIndex + numRows > a.GetNumRows())
            LogicError("AddWithRowSliceValuesOf: startIndex + numRows exceeds a.GetNumRows().");

        if (a.GetNumCols() != GetNumCols())
            LogicError("AddWithRowSliceValuesOf: columns does not match.");

        CUDA_LONG N = (CUDA_LONG)GetNumElements();
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _addWithRowSliceValuesOf<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(m_pArray, a.m_pArray, N, (CUDA_LONG)startIndex, (CUDA_LONG)GetNumRows(), (CUDA_LONG)a.GetNumRows());
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType> GPUMatrix<ElemType>::Diagonal() const
    {
        size_t m = GetNumRows();
        size_t n = GetNumCols();
        if (m != n)
            LogicError("Diagonal can be called only for square matrix. (rows=%d, cols=%d)", (int)m, (int)n);

        GPUMatrix<ElemType> diag(1, n, m_computeDevice);

        CUDA_LONG N = (CUDA_LONG)GetNumElements();
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _assignToDiagonalValuesOf<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(diag.m_pArray, m_pArray, N, (CUDA_LONG)n);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return diag;
    }
#if 0
    //stack the columns in inputMatrices (starting from sliceStartCol for sliceNumCols columns) and assign it to [this] object.
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignRowStackValuesOf(const std::vector<const GPUMatrix<ElemType>*>& inputMatrices, const size_t sliceStartCol, const size_t sliceNumCols)
    {
        if (sliceNumCols == 0)
            LogicError("AssignRowStackValuesOf: sliceNumCols should > 0.");

        size_t totalRows = 0;
        size_t* startRowIndeces = new size_t[inputMatrices.size()+1];
        ElemType ** bufferPointersInInputMatrices = new ElemType*[inputMatrices.size()];

        startRowIndeces[0] = 0;       

        for (int i = 0; i < inputMatrices.size(); i++)
        {
            const GPUMatrix<ElemType>& a = *inputMatrices[i];
            if (a.IsEmpty())
                LogicError("AssignRowStackValuesOf: input matrix (%d) is empty.", i);

            if (a.GetNumCols() < sliceStartCol + sliceNumCols)
                LogicError("AssignRowStackValuesOf: input matrix (%d) GetNumCols() < sliceStartCol + sliceNumCols.", i);

            totalRows += a.GetNumRows();
            startRowIndeces[i + 1] = startRowIndeces[i] + a.GetNumRows();

            bufferPointersInInputMatrices[i] = a.m_pArray + a.LocateColumn(sliceStartCol);
        }

        Resize(totalRows, sliceNumCols);

        PrepareDevice();

        ElemType** bufferPointersInGPU = NULL;
        CUDA_CALL(cudaMalloc((void***)&bufferPointersInGPU, inputMatrices.size()*sizeof(ElemType*)));
        CUDA_CALL(cudaMemcpy(bufferPointersInGPU, bufferPointersInInputMatrices, inputMatrices.size()*sizeof(ElemType*), cudaMemcpyHostToDevice));
        delete[] bufferPointersInInputMatrices;

        size_t* startRowIndecesInGPU = NULL;
        CUDA_CALL(cudaMalloc((void**)&startRowIndecesInGPU, (1+inputMatrices.size())*sizeof(size_t)));
        CUDA_CALL(cudaMemcpy(startRowIndecesInGPU, startRowIndeces, (1+inputMatrices.size())*sizeof(size_t), cudaMemcpyHostToDevice));
        delete[] startRowIndeces;

        CUDA_LONG N = (CUDA_LONG)GetNumElements();
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _assignRowStackValuesOf<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(m_pArray, bufferPointersInGPU, startRowIndecesInGPU, (CUDA_LONG) inputMatrices.size(), N, (CUDA_LONG)GetNumRows(), (CUDA_LONG)GetNumCols());
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        CUDA_CALL(cudaFree(bufferPointersInGPU));
        CUDA_CALL(cudaFree(startRowIndecesInGPU));

        return *this;
    }
#endif

    /// c = c - 1.0 for a specific position
    template<class ElemType>
    void GPUMatrix<ElemType>::MinusOneAt(GPUMatrix<ElemType>& c, const size_t position)
    {
        assert(position < c.GetNumElements());

        cudaEvent_t done = nullptr;
        CUDA_LONG n = (CUDA_LONG)c.GetNumElements();
        CUDA_LONG p = (CUDA_LONG)position;

        int blocksPerGrid = (int)ceil(1.0*n / threadsPerBlock);
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _minusOneAt<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(c.m_pArray, p, n);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    GPUMatrix<ElemType>&  GPUMatrix<ElemType>::AssignRepeatOf(const GPUMatrix<ElemType>& a, const size_t numRowRepeats, const size_t numColRepeats)
    {
        if (this == &a)
            LogicError("AssignRepeatOf: a is the same as [this]. Does not support inplace repeat.");

        if (a.IsEmpty())
            LogicError("AssignRepeatOf: Matrix a is empty.");

        Resize(a.GetNumRows() * numRowRepeats, a.GetNumCols() * numColRepeats);

        CUDA_LONG N = (CUDA_LONG)GetNumElements();
        CUDA_LONG n = (CUDA_LONG)a.GetNumCols(), m = (CUDA_LONG)a.GetNumRows();
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _assignRepeatOf<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(m_pArray, a.m_pArray, N, m, n, (CUDA_LONG)GetNumRows());
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddToRowRepeatValuesOf(const GPUMatrix<ElemType>& a, const size_t numRepeats)
    {
        if (a.IsEmpty())
            LogicError("AddToRowRepeatValuesOf: input matrix a is empty.");

        if (a.GetNumRows() != GetNumRows() * numRepeats)
            LogicError("AddToRowSliceValuesOf: a.GetNumRows() != GetNumRows() * numRepeats.");

        Resize(a.GetNumRows() / numRepeats, a.GetNumCols());

        CUDA_LONG N = (CUDA_LONG)a.GetNumElements();
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _addToRowRepeatValuesOf<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(m_pArray, a.m_pArray, N, (CUDA_LONG)a.GetNumRows(), (CUDA_LONG)a.GetNumCols(), (CUDA_LONG)GetNumRows());
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>&  GPUMatrix<ElemType>::AssignPositiveAndShiftedNegSample(const GPUMatrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber)
    {
        if (this == &a)
            LogicError("AssignPositiveAndShiftedNegSample: a is the same as [this]. Does not support inplace assignment.");

        if (a.IsEmpty())
            LogicError("AssignPositiveAndShiftedNegSample: Matrix a is empty.");

        Resize(a.GetNumRows() * (posNumber + negNumber), a.GetNumCols());

        CUDA_LONG N = (CUDA_LONG)GetNumElements();
        CUDA_LONG n = (CUDA_LONG)a.GetNumCols(), m = (CUDA_LONG)a.GetNumRows();
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _assignPositiveAndShiftedNegSample<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(m_pArray, a.m_pArray, N, m, n, (CUDA_LONG)GetNumRows(), posNumber, shiftNumber);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }
    
    template<class ElemType>
    GPUMatrix<ElemType>&  GPUMatrix<ElemType>::AddFoldedPositiveAndShiftedNegSample(const GPUMatrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber)
    {
        if (this == &a)
            LogicError("AddFoldedPositiveAndShiftedNegSample: a is the same as [this]. Does not support inplace assignment.");

        if (a.IsEmpty())
            LogicError("AddFoldedPositiveAndShiftedNegSample: Matrix a is empty.");

        if (a.GetNumRows() != GetNumRows() * (posNumber + negNumber) || a.GetNumCols() != GetNumCols())
            LogicError("AddFoldedPositiveAndShiftedNegSample: dimensions mismatch.");

        CUDA_LONG N = (CUDA_LONG)a.GetNumElements();
        CUDA_LONG n = (CUDA_LONG)a.GetNumCols(), m = (CUDA_LONG)a.GetNumRows();
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _addFoldedPositiveAndShiftedNegSample<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(m_pArray, a.m_pArray, N, m, n, (CUDA_LONG)GetNumRows(), posNumber, shiftNumber);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }


    template<class ElemType>
    GPUMatrix<ElemType> GPUMatrix<ElemType>::Transpose() const
    {
        if (IsEmpty())
            LogicError("Transpose: Matrix is empty.");

        GPUMatrix<ElemType> c(GetComputeDeviceId());
        c.AssignTransposeOf(*this);
        return c;
    }

    // GetCublasHandle - get a cublas handle for the given GPU, should only need one per GPU
    // computeDevice - The compute device for which the cublas handle is desired
    // returns: cublas handle
    // NOTE: we currently don't bother to ever free the CUBLAS handle, it will be freed automatically by CUDA when the process ends
    template<class ElemType>
    cublasHandle_t GPUMatrix<ElemType>::GetCublasHandle(int computeDevice/*=-1*/)
    {
        // if the compute device is not passed, get the current device from CUDA
        if (computeDevice < 0)
            cudaGetDevice(&computeDevice);

        if (computeDevice < 0 || computeDevice >= MaxGpus)
            LogicError("GetCublasHandle: Maximum GPU exceeded");
        cublasHandle_t cuHandle = s_cuHandle[computeDevice];
        if (cuHandle == NULL)
        {
            s_cuHandle[computeDevice] = cuHandle = _initCUBLAS<ElemType>(computeDevice);
        }
        CUBLAS_CALL(cublasSetStream(cuHandle, t_stream));

        return cuHandle;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignTransposeOf (const GPUMatrix<ElemType>& a)
    {
        if (this == &a)
            LogicError("AssignTransposeOf: a is the same as [this]. Does not support inplace transpose.");

        if (a.IsEmpty())
            LogicError("AssignTransposeOf: Matrix a is empty.");

        if (GetNumRows()!=a.GetNumCols() || GetNumCols()!=a.GetNumRows())
            Resize(a.GetNumCols(), a.GetNumRows());

        cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
        cublasOperation_t transA =  CUBLAS_OP_T;
        cublasOperation_t transB =  CUBLAS_OP_T;
        int m = (int)a.m_numCols;
        int n = (int)a.m_numRows;                
        ElemType alpha=1;
        ElemType beta=0;
        cublasStatus_t st;
        if (sizeof(ElemType)==sizeof(float))
        {
            st = cublasSgeam(cuHandle,transA,transB,m,n,reinterpret_cast<float*>(&alpha),reinterpret_cast<float*>(a.m_pArray),(int)a.m_numRows,reinterpret_cast<float*>(&beta),reinterpret_cast<float*>(a.m_pArray),(int)a.m_numRows,reinterpret_cast<float*>(m_pArray),(int)m_numRows);
        }
        else if (sizeof(ElemType)==sizeof(double))
        {            
            st = cublasDgeam(cuHandle,transA,transB,m,n,reinterpret_cast<double*>(&alpha),reinterpret_cast<double*>(a.m_pArray),(int)a.m_numRows,reinterpret_cast<double*>(&beta),reinterpret_cast<double*>(a.m_pArray),(int)a.m_numRows,reinterpret_cast<double*>(m_pArray),(int)m_numRows);
        }
        else  
        {
            RuntimeError("Unsupported template argument in GPUMatrix"); 
        }
        if (st!=CUBLAS_STATUS_SUCCESS)
        {
            RuntimeError("AssignTransposeOf failed");     
        }
        m_numRows=a.m_numCols;
        m_numCols=a.m_numRows;
        SetMatrixName(a.GetMatrixName());
        return *this;
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::SetValue(const ElemType v)
    {
        if (IsEmpty())
            LogicError("SetValue: Matrix is empty.");

        CUDA_LONG N = (CUDA_LONG) GetNumElements();

        // Check if value is zero, which can be set using cudaMemset
        bool isZero = true;
        const char * valArray = reinterpret_cast<const char *>(&v);

        for (int i = 0; i < sizeof(ElemType); i++)
        {
            if (valArray[i] != 0)
            {
                isZero = false;
                break;
            }
        }

        if (isZero)
        {
            CUDA_CALL(cudaMemset(m_pArray, 0, N * sizeof(ElemType)));
        }
        else
        {
            int blocksPerGrid = (int) ceil(1.0 * N / threadsPerBlock);
            PrepareDevice();
            cudaEvent_t done = nullptr;
            if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
            _setValue<ElemType> <<<blocksPerGrid, threadsPerBlock, 0, t_stream>>>(m_pArray, v, N);
            if (do_sync)    CUDA_CALL(cudaEventRecord(done));
            if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
            if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        }
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::SetValue(const ElemType* d_v) //d_v is pointer to the the value in GPU memory
    {
        if (IsEmpty())
            LogicError("SetValue: Matrix is empty.");

        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _setValue<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,d_v,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done)); 
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::MaskColumnsValue(const GPUMatrix<char>& columnsMask, ElemType val)
    {
        if (GetNumCols() != columnsMask.GetNumCols())
            RuntimeError("Matrix and column mask must have equal number of columns");

        if (GetComputeDeviceId() != columnsMask.GetComputeDeviceId())
            RuntimeError("Matrix and column mask must be on the same device");

        int blocksPerGrid = (int)GetNumCols();
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _maskColumnsValue<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(m_pArray, columnsMask.m_pArray, (CUDA_LONG)GetNumCols(), (CUDA_LONG)GetNumRows(), val);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::SetColumn(const ElemType* colPointer, size_t colInd)
    {
        if (IsEmpty())
            LogicError("SetValue: Matrix is empty.");
        if (colPointer==NULL)
            return;
        CUDA_CALL(cudaMemcpy(m_pArray+LocateColumn(colInd),colPointer,sizeof(ElemType)*m_numRows,cudaMemcpyHostToDevice));
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::SetColumn(const GPUMatrix<ElemType>& valMat, size_t colInd)
    {
        if (IsEmpty())
            LogicError("SetColumn: Matrix is empty.");
        if (valMat.GetNumCols() != 1)
            LogicError("SetColumn: only support one column matrix now.");
        CUDA_CALL(cudaMemcpy(m_pArray + LocateColumn(colInd), valMat.m_pArray, sizeof(ElemType)*m_numRows, cudaMemcpyDeviceToDevice));
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::SetValue(const GPUMatrix<ElemType>& deepCopyFrom)
    {
        if (this == &deepCopyFrom)
            return;

        Resize(deepCopyFrom.GetNumRows(), deepCopyFrom.GetNumCols());
        m_format = deepCopyFrom.m_format; // copy the format over just to be sure
        size_t cpSize = deepCopyFrom.GetNumRows() * deepCopyFrom.GetNumCols();
        if (cpSize != 0)
            CUDA_CALL(cudaMemcpy(m_pArray,deepCopyFrom.m_pArray,cpSize*sizeof(ElemType),cudaMemcpyDeviceToDevice));        
    }

    template<class ElemType>    
    void GPUMatrix<ElemType>::SetValue(const size_t numRows, const size_t numCols, int deviceId, ElemType *pArray, size_t matrixFlags)
    {
        // handle externally managed case
        if (matrixFlags&matrixFlagDontOwnBuffer)
        {
            // free the existing array if it used to be an owned array
            if (OwnBuffer() && m_pArray!=NULL)
            {
                PrepareDevice();
                CUDA_CALL(cudaFree(m_pArray));
            }
            m_numRows = numRows;
            m_numCols = numCols;
            m_pArray = pArray;
            m_elemSizeAllocated = GetNumElements();
            m_matrixName = NULL;
            m_format = matrixFormatDense;
            m_externalBuffer = true;
            m_computeDevice = deviceId;
        }
        else 
        {
            // if didn't previously own the buffer, wipe it clean 
            if (!OwnBuffer())
            {
                ZeroInit(deviceId);
            }

            // if the devices are different move it now
            if (m_computeDevice != deviceId && deviceId >= 0)
            {
                Clear();
                ZeroInit(deviceId);
            }

            // now resize/allocate as necessary
            Resize(numRows, numCols);
            m_externalBuffer = false;

            // copy over the content to the buffer
            PrepareDevice();
            if (pArray!=NULL) 
            {
                if (!(matrixFlags&matrixFormatRowMajor))
                {
                    CUDA_CALL(cudaMemcpy(m_pArray, pArray, sizeof(ElemType)*GetNumElements(), 
                        (matrixFlags&matrixFlagSetValueOnDevice)?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice));
                }
                else
                {
                    RuntimeError("Row major isn't implemented");
                }
            }
        }
        m_format = matrixFormatDense;
    }


    template<class ElemType>
    void GPUMatrix<ElemType>::SetDiagonalValue(const ElemType v)
    {
        CUDA_LONG N=(CUDA_LONG)GetNumRows();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _setDiagonalValue<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,v,N,(CUDA_LONG)GetNumRows());
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::SetDiagonalValue(const GPUMatrix<ElemType>& vector)
    {
        if (IsEmpty() || vector.IsEmpty())
            LogicError("SetDiagonalValue: Matrix is empty.");

        if (GetNumRows() != GetNumCols())
            LogicError("SetDiagonalValue: NumRows and NumCols do not agree.");

        if (vector.GetNumRows() != 1 && vector.GetNumCols() != 1)
            LogicError("SetDiagonalValue: input vector must be a vector.");

        if (vector.GetNumElements() == 1) //reduce to simple form
            SetDiagonalValue(vector.m_pArray[0]);

        else if (vector.GetNumRows() != GetNumRows())
            LogicError("SetDiagonalValue: input vector's dimension does not agree with [this].");
        else
        {
            CUDA_LONG N=(CUDA_LONG)GetNumRows();
            int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);
            PrepareDevice();
            cudaEvent_t done = nullptr;
            if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
            _setDiagonalValueFromVector<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,vector.m_pArray,N);
            if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
            if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
            if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        }
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::SetUniformRandomValue(const ElemType low, const ElemType high, unsigned long seed)
    {
        PrepareDevice();
#if 0   // to change the seed, we must tear down the random generator
        // This is not efficient, but for correctness, we must do it.
        if (s_curandGenerator && (seed != USE_TIME_BASED_SEED))
        {
            fprintf(stderr, "SetUniformRandomValue (GPU): destroying curand object\n");
            CURAND_CALL(curandDestroyGenerator(((curandGenerator_t*)s_curandGenerator)[0]));    // TODO: what is this typecast business??
            delete s_curandGenerator;
            s_curandGenerator = NULL;
        }
#endif
        if (s_curandGenerator==NULL)
        {
            unsigned long long cudaSeed = (seed == USE_TIME_BASED_SEED) ? time(NULL) : seed;
            fprintf(stderr, "SetUniformRandomValue (GPU): creating curand object with seed %llu\n", cudaSeed);
            s_curandGenerator = new curandGenerator_t;
            /* Create pseudo-random number generator */
            CURAND_CALL(curandCreateGenerator(&(((curandGenerator_t*)s_curandGenerator)[0]),CURAND_RNG_PSEUDO_XORWOW));
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(((curandGenerator_t*)s_curandGenerator)[0], cudaSeed));
            CURAND_CALL(curandSetGeneratorOrdering(((curandGenerator_t*)s_curandGenerator)[0],CURAND_ORDERING_PSEUDO_SEEDED));
        }

        cudaEvent_t done = nullptr;
        CUDA_CALL(cudaEventCreate(&done));
        if (sizeof(ElemType)==sizeof(float))
        {
            CURAND_CALL(curandGenerateUniform(((curandGenerator_t*)s_curandGenerator)[0], reinterpret_cast<float*>(m_pArray), GetNumElements()));
        }
        else
        {
            CURAND_CALL(curandGenerateUniformDouble(((curandGenerator_t*)s_curandGenerator)[0], reinterpret_cast<double*>(m_pArray), GetNumElements()));
        }
        CUDA_CALL(cudaEventRecord(done));        
        CUDA_CALL(cudaEventSynchronize(done)); 
        //CURAND_CALL(curandDestroyGenerator(gen));
        CUDA_CALL(cudaEventDestroy(done));

        size_t N=GetNumElements();
        size_t blocksPerGrid = (size_t)ceil(N/(double)threadsPerBlock);

        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _rescaleToRange<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,N,low,high);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::SetGaussianRandomValue(const ElemType mean, const ElemType sigma, unsigned long seed)
    {
        PrepareDevice();
#if 0   // to change the seed, we must tear down the random generator
        // This is not efficient, but for correctness, we must do it.
        if (s_curandGenerator && (seed != USE_TIME_BASED_SEED))
        {
            fprintf(stderr, "SetGaussianRandomValue (GPU): destroying curand object\n");
            CURAND_CALL(curandDestroyGenerator(((curandGenerator_t*)s_curandGenerator)[0]));    // TODO: what is this typecast business??
            delete s_curandGenerator;
            s_curandGenerator = NULL;
        }
#endif
        if (s_curandGenerator==NULL)
        {
            unsigned long long cudaSeed = (seed == USE_TIME_BASED_SEED) ? time(NULL) : seed;
            fprintf(stderr, "SetGaussianRandomValue (GPU): creating curand object with seed %llu\n", cudaSeed);
            s_curandGenerator = new curandGenerator_t;
            /* Create pseudo-random number generator */        
            CURAND_CALL(curandCreateGenerator(&(((curandGenerator_t*)s_curandGenerator)[0]),CURAND_RNG_PSEUDO_XORWOW)); 
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(((curandGenerator_t*)s_curandGenerator)[0], cudaSeed));
            CURAND_CALL(curandSetGeneratorOrdering(((curandGenerator_t*)s_curandGenerator)[0],CURAND_ORDERING_PSEUDO_SEEDED));
        }

        if (sizeof(ElemType)==sizeof(float))
        {
            CURAND_CALL(curandGenerateNormal(((curandGenerator_t*)s_curandGenerator)[0], reinterpret_cast<float*>(m_pArray), GetNumElements(), (float)mean, (float)sigma));
        }
        else
        {
            CURAND_CALL(curandGenerateNormalDouble(((curandGenerator_t*)s_curandGenerator)[0], reinterpret_cast<double*>(m_pArray), GetNumElements(), (double)mean, (double)sigma));
        }
        //CURAND_CALL(curandDestroyGenerator(gen));
    }

    //maskRate: percentage of values masked out (similar to dropout rate)
    //scaleValue: which scale value to set to the left ones (unmasked items).
    template<class ElemType>
    void GPUMatrix<ElemType>::SetUniformRandomMask(const ElemType maskRate, const ElemType scaleValue, unsigned long seed)
    {
        PrepareDevice();
        if (s_curandGenerator==NULL)
        {            
            s_curandGenerator = new curandGenerator_t;
            /* Create pseudo-random number generator */        
            CURAND_CALL(curandCreateGenerator(&(((curandGenerator_t*)s_curandGenerator)[0]),CURAND_RNG_PSEUDO_XORWOW));        
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(((curandGenerator_t*)s_curandGenerator)[0], seed==USE_TIME_BASED_SEED ? time(NULL) : seed));       
            CURAND_CALL(curandSetGeneratorOrdering(((curandGenerator_t*)s_curandGenerator)[0],CURAND_ORDERING_PSEUDO_SEEDED));
        }

        cudaEvent_t done = nullptr;
        CUDA_CALL(cudaEventCreate(&done));
        if (sizeof(ElemType)==sizeof(float))
        {
            CURAND_CALL(curandGenerateUniform((((curandGenerator_t*)s_curandGenerator)[0]), reinterpret_cast<float*>(m_pArray), GetNumElements()));
        }
        else
        {
            CURAND_CALL(curandGenerateUniformDouble((((curandGenerator_t*)s_curandGenerator)[0]), reinterpret_cast<double*>(m_pArray), GetNumElements()));
        }
        CUDA_CALL(cudaEventRecord(done));        
        CUDA_CALL(cudaEventSynchronize(done)); 
        CUDA_CALL(cudaEventDestroy(done));
        //CURAND_CALL(curandDestroyGenerator(gen));

        size_t N=GetNumElements();
        size_t blocksPerGrid = (size_t)ceil(N/(double)threadsPerBlock);        
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _setMaskAndScale<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,N,maskRate,scaleValue);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    ElemType GPUMatrix<ElemType>::Adagrad(GPUMatrix<ElemType>& gradients, const bool needAveMultiplier)
    {
        size_t numColsNeeded = gradients.GetNumCols();
        if (needAveMultiplier)
            numColsNeeded += gradients.GetNumCols();

        if (IsEmpty() || GetNumCols() < numColsNeeded)
        {
            Resize(gradients.GetNumRows(), numColsNeeded);
            SetValue(0.0);
        }

        assert(GetNumRows() == gradients.GetNumRows() && GetNumCols() == numColsNeeded);

        size_t n = gradients.GetNumElements();

        ElemType *multipliers = nullptr;
        if (needAveMultiplier)
            multipliers = m_pArray + n; // temp memory used to store multipliers,

        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        _adagrad<ElemType> << <blocksPerGrid, threadsPerBlock >> >(m_pArray, gradients.m_pArray, n, multipliers);

        if (!needAveMultiplier)
            return 1;

        cublasHandle_t cuHandle = GetCublasHandle(GetComputeDeviceId());
        if (sizeof(ElemType) == sizeof(float))
        {
            float aveMultiplier = 0;
            CUBLAS_CALL(cublasSasum(cuHandle, (CUDA_LONG)n, reinterpret_cast<float*>(multipliers), 1, &aveMultiplier));
            return (ElemType)aveMultiplier / n;
        }
        else
        {
            double aveMultiplier = 0;
            CUBLAS_CALL(cublasDasum(cuHandle, (CUDA_LONG)n, reinterpret_cast<double*>(multipliers), 1, &aveMultiplier));
            return (ElemType)aveMultiplier / n;
        }
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::FSAdagrad(GPUMatrix<ElemType>& gradients,
                                        GPUMatrix<ElemType>& functionValues, 
                                        ElemType learnRatePerSample,
                                        ElemType momentum,
                                        ElemType adaWeight,
                                        ElemType adaMul)
    {
        size_t numColsNeeded = 2 * gradients.GetNumCols();

        if (IsEmpty() || (GetNumCols() < numColsNeeded))
        {
            Resize(gradients.GetNumRows(), numColsNeeded);
            SetValue(0.0);
        }

        assert((GetNumRows() == gradients.GetNumRows()) && (GetNumCols() == numColsNeeded));

        size_t n = gradients.GetNumElements();
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        _fsadagrad<ElemType><<<blocksPerGrid, threadsPerBlock>>>(n, gradients.m_pArray, m_pArray, m_pArray + n, functionValues.m_pArray,
                                                                 learnRatePerSample, momentum, adaWeight, adaMul);
    }

    template<class ElemType>
    ElemType GPUMatrix<ElemType>::RmsProp(GPUMatrix<ElemType>& gradients,
        ElemType RMS_GAMMA,
        ElemType RMS_WGT_INC,
        ElemType RMS_WGT_MAX,
        ElemType RMS_WGT_DEC,
        ElemType RMS_WGT_MIN,
        const bool needAveMultiplier
        )
    {
        const ElemType floor = 1e-6f;
        static ElemType *upd_gpu = (ElemType*)0;

        size_t n = gradients.GetNumElements();
        int blocksPerGrid = (GetNumElements() + threadsPerBlock -1 )/threadsPerBlock;

        size_t numColsNeeded = gradients.GetNumCols() * 3;
        if (needAveMultiplier)
            numColsNeeded += gradients.GetNumCols();

        if (IsEmpty() || GetNumCols() < numColsNeeded)
        {
            Resize(gradients.GetNumRows(), numColsNeeded);
            SetValue(0.0);

            ElemType *avars=m_pArray; // accumulated variances for RMS scaling
            ElemType *signs=m_pArray+n; // sign of previous gradient
            ElemType *steps=m_pArray+2*n; // current step size
            //m_pArray+3*n is temp memory used to store multipliers, no need to initialize

            _rmsprop_init<ElemType><<<blocksPerGrid, threadsPerBlock>>>(avars,signs,steps,gradients.m_pArray,n);

        }
        assert(GetNumRows() == gradients.GetNumRows() && GetNumCols() == numColsNeeded);

        ElemType *avars=m_pArray; // accumulated variances for RMS scaling
        ElemType *signs=m_pArray+n; // sign of previous gradient
        ElemType *steps=m_pArray+2*n; // current step size

        ElemType *multipliers = nullptr;
        if (needAveMultiplier)
            multipliers = m_pArray + 3 * n; // temp memory used to store multipliers,

        if( !upd_gpu )
        {
            ElemType upd[] = {
                2,2,0,
                2,2,0,
                1,1,1,
                2,2,0,
                1,2,1,
                0,2,2,
                1,1,1,
                0,2,2,
                0,2,2,
            };

            CUDA_CALL(cudaMalloc((void**)&upd_gpu,sizeof(ElemType)*27));
            CUDA_CALL(cudaMemcpy(upd_gpu,upd,sizeof(ElemType)*27,cudaMemcpyHostToDevice));
        }

        _rmsprop<ElemType> << <blocksPerGrid, threadsPerBlock >> >(avars, signs, steps, gradients.m_pArray, n,
            RMS_GAMMA, RMS_WGT_INC, RMS_WGT_MAX, RMS_WGT_DEC, RMS_WGT_MIN,
            floor, upd_gpu, multipliers);

        if (!needAveMultiplier)
            return 1;

        cublasHandle_t cuHandle = GetCublasHandle(GetComputeDeviceId());
        if (sizeof(ElemType) == sizeof(float))
        {
            float aveMultiplier = 0;
            CUBLAS_CALL(cublasSasum(cuHandle, (CUDA_LONG)n, reinterpret_cast<float*>(multipliers), 1, &aveMultiplier));
            return aveMultiplier / n;
        }
        else
        {
            double aveMultiplier = 0;
            CUBLAS_CALL(cublasDasum(cuHandle, (CUDA_LONG)n, reinterpret_cast<double*>(multipliers), 1, &aveMultiplier));
            return (ElemType) aveMultiplier / n;
        }
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::Reshape(const size_t numRows, const size_t numCols)
    {
        assert (numRows*numCols == GetNumElements());
        if (numRows*numCols != GetNumElements())
            InvalidArgument("Reshape: total number of elements does not match.");

        m_numRows = numRows;
        m_numCols = numCols;
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::Resize(const size_t numRows, const size_t numCols, bool growOnly)
    {
        if (m_numRows==numRows && m_numCols==numCols)
            return;   
        if (!OwnBuffer())
            InvalidArgument("Can't resize a externally managed matrix");

        m_numRows = numRows;
        m_numCols = numCols;

        size_t numElements = GetNumElements();
        if (numElements > m_elemSizeAllocated || (!growOnly && numElements != m_elemSizeAllocated))
        {
            if (IsEmpty())
            {
                m_elemSizeAllocated = 0;
                m_pArray = NULL;
            }
            else
            {
                //if (!OwnBuffer())
                //    InvalidArgument("Can't resize a externally managed matrix");
                PrepareDevice();
                if (m_pArray)
                    CUDA_CALL(cudaFree(m_pArray)); //delete and reallocate                            
                m_elemSizeAllocated = numElements;
                CUDA_CALL(cudaMalloc((void**)&m_pArray,sizeof(ElemType)*m_elemSizeAllocated));
                CUDA_CALL(cudaMemset(m_pArray,0,sizeof(ElemType)*m_elemSizeAllocated));
            }
        }
    }

    template<class ElemType>
    size_t GPUMatrix<ElemType>::LocateElement (const size_t row, const size_t col) const 
    { 
        assert (row < m_numRows && col < m_numCols); 
        return col * m_numRows  + row;  // matrix in column-wise storage
    }  

    template<class ElemType>
    size_t GPUMatrix<ElemType>::LocateColumn (const size_t col) const 
    { 
        assert (col < m_numCols); 
        return col * m_numRows;  // matrix in column-wise storage
    }  

    template<class ElemType>
    ElemType GPUMatrix<ElemType>::Get00Element() const 
    {        
        ElemType res=0;        
        CUDA_CALL(cudaMemcpy(&res,m_pArray,sizeof(ElemType),cudaMemcpyDeviceToHost));
        return res;
    }
#pragma endregion Basic Operators

#pragma region Member BLAS Functions
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator+= (ElemType alpha) 
    {
        if (IsEmpty())
            LogicError("operator+=: Matrix is empty.");
        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _addValue<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,alpha,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType> GPUMatrix<ElemType>::operator+ (ElemType alpha) const
    {
        if (IsEmpty())
            LogicError("operator+: Matrix is empty.");

        const GPUMatrix<ElemType>& us=*this;
        GPUMatrix<ElemType> c(us);
        c+=alpha;
        return c;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSumOf(const ElemType alpha, const GPUMatrix<ElemType>& a)
    {
        SetValue(a);
        (*this)+=alpha;
        return (*this);
    }


    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator+= (const GPUMatrix<ElemType>& a) 
    {
        //if (a.GetNumElements()==1)
        //{
        //    //*this += a.Get00Element();
        //    CUDA_LONG N=(CUDA_LONG)GetNumElements();
        //    int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);
        //    cudaEvent_t done = nullptr;
        //    if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        //    _addValue<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,a.m_pArray,N);
        //    if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        //    if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        //    if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        //}
        //else 
        //{
            ScaleAndAdd(1, a, *this);
        //}
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType> GPUMatrix<ElemType>::operator+ (const GPUMatrix<ElemType>& a) const
    {
        if (GetNumElements()==1)
        {
            GPUMatrix<ElemType> c(a);
            c+=Get00Element();
            return c;
        }
        else if (a.GetNumElements()==1)
        {
            GPUMatrix<ElemType> c(*this);
            c+=a.Get00Element();
            return c;
        }
        else
        {
            GPUMatrix<ElemType> c(*this); //this implementation will introduce a copy overhead. but make resue of the code
            c += a;
            return c;
        }
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSumOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
    {
        SetValue(a);
        (*this)+=b;
        return (*this);
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator-= (ElemType alpha) 
    {
        if (IsEmpty())
            LogicError("operato-=: Matrix is empty.");
        return operator+=(-1*alpha);        
    }

    template<class ElemType>
    GPUMatrix<ElemType> GPUMatrix<ElemType>::operator- (ElemType alpha) const
    {
        if (IsEmpty())
            LogicError("operator-: Matrix is empty.");
        return operator+(-1*alpha);
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignDifferenceOf(const ElemType alpha, const GPUMatrix<ElemType>& a)
    {
        Resize(a.m_numRows,a.m_numCols);
        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);
        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _assignDifferenceOf1<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,alpha,a.m_pArray,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
        /*Resize(a.m_numRows,a.m_numCols);
        SetValue(alpha);
        (*this)-=a;
        return *this;*/
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignDifferenceOf(const GPUMatrix<ElemType>& a, const ElemType alpha)
    {
        Resize(a.m_numRows,a.m_numCols);
        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);
        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _assignDifferenceOf2<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,alpha,a.m_pArray,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
        /*SetValue(a);
        (*this)-=alpha;
        return *this;*/
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator-= (const GPUMatrix<ElemType>& a)
    {
        //if (a.GetNumElements() == 1)
        //    AssignDifferenceOf(*this, a.Get00Element());
        //else if (GetNumElements() == 1)
        //    AssignDifferenceOf(Get00Element(), a);
        //else
            ScaleAndAdd(-1, a, *this);

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType> GPUMatrix<ElemType>::operator- (const GPUMatrix<ElemType>& a) const
    {
        GPUMatrix<ElemType> c(*this); //this implementation will introduce a copy overhead. but make resue of the code
        c -= a;
        return c;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignDifferenceOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
    {
        if (this != &a)
        {
            Resize(a.GetNumRows(), a.GetNumCols());
            SetValue(a);
        }
        (*this) -= b;
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator*= (ElemType alpha)
    {
        Scale(alpha, *this);
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType> GPUMatrix<ElemType>::operator* (ElemType alpha) const
    {
        GPUMatrix<ElemType> c(GetNumRows(), GetNumCols(), GetComputeDeviceId());
        Scale(alpha, *this, c);
        return c;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignProductOf(const ElemType alpha, const GPUMatrix<ElemType>& a)
    {
        Scale(alpha, a, *this);
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignProductOf (const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB)
    {
        if (a.GetNumElements() == 1)
        {  
            if (transposeB)
                AssignTransposeOf(b);
            (*this) *= a.Get00Element();
        }
        else if (b.GetNumElements() == 1)
        { 
            if (transposeA)
                AssignTransposeOf(a);
            (*this) *= b.Get00Element();
        }
        else
            Multiply(a, transposeA, b, transposeB, *this);
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType> GPUMatrix<ElemType>::operator* (const GPUMatrix<ElemType>& a) const
    {
        const GPUMatrix<ElemType>& us = *this;
        if (GetNumElements() == 1)
        {
            GPUMatrix<ElemType> c(GetComputeDeviceId());
            c.AssignProductOf(Get00Element(), a);
            return c;
        }
        else if (a.GetNumElements() == 1)
        {
            GPUMatrix<ElemType> c(GetComputeDeviceId());
            c.AssignProductOf(a.Get00Element(), us);
            return c;
        }
        else
        {
            GPUMatrix<ElemType> c(GetNumRows(),a.GetNumCols(),GetComputeDeviceId());
            Multiply(*this, a, c);
            return c;
        }
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator/= (ElemType alpha)
    {
        (*this) *= 1/alpha;
        return (*this);
    }

    template<class ElemType>
    GPUMatrix<ElemType> GPUMatrix<ElemType>::operator/ (ElemType alpha) const
    {
        return ((*this) * (1/alpha));
    }

    //element-wise power
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator^= (ElemType alpha)
    {
        GPUMatrix<ElemType>& us = *this;
        ElementWisePower(alpha, us, us);
        return us;
    }

    template<class ElemType>
    GPUMatrix<ElemType> GPUMatrix<ElemType>::operator^ (ElemType alpha) const
    {
        GPUMatrix<ElemType> c(GetNumRows(), GetNumCols(), GetComputeDeviceId());
        ElementWisePower(alpha, *this, c);
        return c;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementPowerOf(const GPUMatrix<ElemType>& a, const ElemType power)
    {
        ElementWisePower(power, a, *this);
        return *this;
    }


    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddElementProductOf (const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
    {
        if (a.IsEmpty() || b.IsEmpty())
            LogicError("AddElementProductOf: Matrix is empty.");

        assert (a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
        if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
            InvalidArgument("The input matrix dimensions do not match.");

        if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == GetNumCols()))
            InvalidArgument("The input matrix dimensions do not match [this].");

        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);    
        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _addElementProductOf<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,a.m_pArray,b.m_pArray,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));      
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::ColumnElementMultiplyWith(const GPUMatrix<ElemType>& a)
    {
        if (a.IsEmpty() || IsEmpty())
            LogicError("ColumnElementMultiplyWith: Matrix is empty.");

        if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == 1))
            InvalidArgument("ColumnElementMultiplyWith: The input matrix should be a col vector and match [this]'s rows.");

        CUDA_LONG N=(CUDA_LONG)a.GetNumRows();
        CUDA_LONG M=(CUDA_LONG)GetNumCols();        
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);  
        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _columnElementMultiplyWith<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,a.m_pArray,N,M);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));      
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::RowElementMultiplyWith(const GPUMatrix<ElemType>& a)
    {
        if (a.IsEmpty() || IsEmpty())
            LogicError("RowElementMultiplyWith: Matrix is empty.");

        if (!(a.GetNumRows() == 1 && a.GetNumCols() == GetNumCols()))
            InvalidArgument("RowElementMultiplyWith: The input matrix should be a row vector and match [this]'s columns.");

        CUDA_LONG N = (CUDA_LONG)GetNumRows();
        CUDA_LONG M = (CUDA_LONG)a.GetNumCols();
        int blocksPerGrid = (int)ceil(1.0*M / threadsPerBlock);
        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _rowElementMultiplyWith<ElemType><<<blocksPerGrid,threadsPerBlock>>>(m_pArray,a.m_pArray,N,M);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));      
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::RowElementDivideBy(const GPUMatrix<ElemType>& a)
    {
        if (a.IsEmpty() || IsEmpty())
            LogicError("RowElementDivideBy: Matrix is empty.");

        if (!(a.GetNumRows() == 1 && a.GetNumCols() == GetNumCols()))
            InvalidArgument("RowElementDivideBy: The input matrix should be a row vector and match [this]'s columns.");

        CUDA_LONG N = (CUDA_LONG)GetNumRows();
        CUDA_LONG M = (CUDA_LONG)a.GetNumCols();
        int blocksPerGrid = (int)ceil(1.0*M / threadsPerBlock);
        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _rowElementDivideBy<ElemType> << <blocksPerGrid, threadsPerBlock >> >(m_pArray, a.m_pArray, N, M);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::ColumnElementDivideBy(const GPUMatrix<ElemType>& a)
    {
        if (a.IsEmpty() || IsEmpty())
            LogicError("ColumnElementDivideBy: Matrix is empty.");

        if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == 1))
            InvalidArgument("ColumnElementDivideBy: The input matrix should be a col vector and match [this]'s rows.");

        CUDA_LONG N = (CUDA_LONG)a.GetNumRows();
        CUDA_LONG M = (CUDA_LONG)GetNumCols();
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _ColumnElementDivideBy<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,a.m_pArray,N,M);                        
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));      
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::ElementInverse ()
    {
        if (IsEmpty())
            LogicError("ElementInverse: Matrix is empty.");

        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);  
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _elemInverse<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));     
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementInverseOf (const GPUMatrix<ElemType>& a)
    {
        SetValue(a);
        return ElementInverse();
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceSigmoid()
    {
        performInplaceFunction(0);                    
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSigmoidOf (const GPUMatrix<ElemType>& a)
    {
        Resize(a.GetNumRows(),a.GetNumCols());
        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _assignSigmoidOf<<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(a.m_pArray,m_pArray,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        /*SetValue(a);
        InplaceSigmoid();*/
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceSigmoidDerivative()
    {
        AssignSigmoidDerivativeOf(*this);                    
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSigmoidDerivativeOf (const GPUMatrix<ElemType>& a)
    {
        if (a.IsEmpty())
            LogicError("AssignSigmoidDerivativeOf: Matrix a is empty.");

        //auto& us=*this;
        if (this != &a)
            Resize(a.GetNumRows(), a.GetNumCols());

        PrepareDevice();
        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);                
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        

        _assignSigmoidDerivative<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(a.m_pArray, m_pArray, N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::AssignNoiseContrastiveEstimation(const GPUMatrix<ElemType>& a,
        const GPUMatrix<ElemType>& b, const GPUMatrix<ElemType>& bias, size_t sampleCount, GPUMatrix<ElemType>& tmp, GPUMatrix<ElemType>& c)
        //this: samples+probs
        // a:   hidden
        // b:   embedding
        // tmp:  softmax
        //  c: loglikelihood
    {
        UNCONST(ElemType, a, my_a);
        UNCONST(ElemType, b, my_b);
        UNCONST(ElemType, bias, my_bias);
        cudaEvent_t done = nullptr;
        if (do_sync) CUDA_CALL(cudaEventCreate(&done));
        //a: dim * minibatch
        //b: dim * |vocab|
        int p = 512;
        int width = a.GetNumRows(); //dimension of hidden vector
        
        while (p / 2 > width) p = p / 2;

        _computeNceOutput<ElemType> << <this->GetNumElements() / 2, p >> >(
            this->GetArray(), 
            sampleCount,
            m_numRows / 2,
            my_a.GetArray(),//a
            a.GetNumRows(),
            my_b.GetArray(),//b
            my_bias.GetArray(),
            tmp.GetArray());//tmp
     
        p = 512;
        while (p / 2 > this->GetNumElements() / 2) p = p / 2;
        // summing up objective must be done in one block
        _assignNoiseContrastiveEstimation<ElemType> << <1, p >> >(
            this->GetArray(),
            sampleCount,
            m_numRows / 2,
            my_a.GetArray(),
            a.GetNumCols(),
            my_b.GetArray(),
            tmp.GetArray(),
            c.GetArray());
        
        if (do_sync) CUDA_CALL(cudaEventRecord(done));
        if (do_sync) CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync) CUDA_CALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::AssignNCEDerivative(GPUMatrix<ElemType>& tmp, const GPUMatrix<ElemType>& a,
        const GPUMatrix<ElemType>& b, size_t inputIndex, GPUMatrix<ElemType>& c)
    {
        UNCONST(ElemType, a, my_a);
        UNCONST(ElemType, b, my_b);
        cudaEvent_t done = nullptr;
        if (do_sync) CUDA_CALL(cudaEventCreate(&done));
        int p = 512;
        int width = a.GetNumRows();
        while (p / 2 > width) p = p / 2;

        _assignNceDerivativeNew<ElemType> << < (tmp.GetNumElements() + p - 1) / p, p >> >(
            GetArray(),
            tmp.GetNumCols(),
            m_numRows / 2,
            my_a.GetArray(),
            a.GetNumRows(),
            my_b.GetArray(),
            tmp.GetArray(),
            c.GetArray(),
            inputIndex);

        if (do_sync) CUDA_CALL(cudaEventRecord(done));
        if (do_sync) CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync) CUDA_CALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::AssignSoftmaxSum(const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c)
    {
        UNCONST(ElemType, a, my_a);
        cudaEvent_t done = nullptr;
        if (do_sync) CUDA_CALL(cudaEventCreate(&done));
        int p = 512;
        int width = a.GetNumRows();
        while (p / 2 > width) p = p / 2;

        _assignSoftmaxSum<ElemType> << <1, p >> >(
            my_a.GetArray(),
            width,
            GetArray(),
            c.GetArray()
            );

        if (do_sync) CUDA_CALL(cudaEventRecord(done));
        if (do_sync) CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync) CUDA_CALL(cudaEventDestroy(done));
    }
    template<class ElemType>
    void GPUMatrix<ElemType>::AssignNCEUnnormalizedEval(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
    {
        assert(a.GetComputeDeviceId() == b.GetComputeDeviceId());
        assert(GetNumRows() == a.GetNumRows());
        assert(GetNumCols() == b.GetNumRows());
        assert(a.GetNumCols() == b.GetNumRows());
        UNUSED(a); UNUSED(b); UNUSED(c);  // TODO: this function seems like a stub
        /*
        EnsureAuxMemory();
        int p = 512;
        int width = a.GetNumCols();
        while (p / 2 > width) p = p / 2;

        // this kernel need be launched in nnz blocks
        _sparseInnerProductDenseTimesDense<ElemType> << <m_nz, p >> >(
        m_dVal,
        m_buf,
        m_dCol,
        m_nz,
        GetNumRows(),
        a.GetArray(),
        b.GetArray(),
        b.GetNumRows(),
        m_res);

        //sum up the results
        _reductionSum32<ElemType> << <1, 32 >> >(m_res, c.GetArray(), m_nz);*/
    }


    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceTanh()
    {
        performInplaceFunction(1);
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignTanhOf (const GPUMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceTanh();
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceLogSoftmax (const bool isColWise)
    {
        if (IsEmpty())
            LogicError("InplaceLogSoftmax: Matrix is empty.");

        PrepareDevice();
        if (isColWise)
        {
            CUDA_LONG N=(CUDA_LONG)GetNumCols(); //one kernel per column
            int blocksPerGrid =(int)ceil(N*1.0/threadsPerBlock);             
            cudaEvent_t done = nullptr;
            if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
            _logSoftMaxColWise<<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,(CUDA_LONG)m_numCols,(CUDA_LONG)m_numRows);
            if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
            if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));  
            if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        }
        else
        {
            CUDA_LONG N=(CUDA_LONG)GetNumRows(); //one kernel per column
            int blocksPerGrid =(int)ceil(N*1.0/threadsPerBlock);                
            cudaEvent_t done = nullptr;
            if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
            _logSoftMaxRowWise<<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,(CUDA_LONG)m_numCols,(CUDA_LONG)m_numRows);
            if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
            if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
            if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        }
        return *this; 
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignLogSoftmaxOf (const GPUMatrix<ElemType>& a, const bool isColWise)
    {
        Resize(a.GetNumRows(),a.GetNumCols());        
        if (isColWise)
        {            
            PrepareDevice();
            CUDA_LONG N = (CUDA_LONG)GetNumCols();
            CUDA_LONG M = (CUDA_LONG)GetNumRows();
            cudaEvent_t done = nullptr;
            if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
            _assignColumnwiseLogSoftmaxOf<<<N,512,0,t_stream>>>(a.m_pArray,m_pArray,N,M);
            if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
            if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
            if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        }
        else
        {
            NOT_IMPLEMENTED;
        }

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceHardmax(const bool isColWise)
    {
        return AssignHardmaxOf(*this, isColWise);
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignHardmaxOf(const GPUMatrix<ElemType>& a, const bool isColWise)
    {
        Resize(a.GetNumRows(), a.GetNumCols());
        if (isColWise)
        {
            PrepareDevice();
            CUDA_LONG N = (CUDA_LONG)GetNumCols();
            CUDA_LONG M = (CUDA_LONG)GetNumRows();
            cudaEvent_t done = nullptr;
            if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
            _assignColumnwiseHardmaxOf << <N, 512, 0, t_stream >> >(a.m_pArray, m_pArray, N, M);
            if (do_sync)    CUDA_CALL(cudaEventRecord(done));
            if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
            if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        }
        else
        {
            NOT_IMPLEMENTED;
        }

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceSqrt()
    {
        performInplaceFunction(2);        
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSqrtOf (const GPUMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceSqrt();
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceExp()
    {
        performInplaceFunction(3);        
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignExpOf (const GPUMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceExp();
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceLog()
    {
        performInplaceFunction(4);        
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignLogOf (const GPUMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceLog();
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceAbs()
    {
        performInplaceFunction(5);        
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignAbsOf (const GPUMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceAbs();
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceLinearRectifierDerivative()
    {
        performInplaceFunction(6);                    
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignLinearRectifierDerivativeOf (const GPUMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceLinearRectifierDerivative();
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceCosine()
    {
        performInplaceFunction(7);        
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignCosineOf (const GPUMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceCosine();
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceNegativeSine()
    {
        performInplaceFunction(8);        
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignNegativeSineOf (const GPUMatrix<ElemType>& a)
    {
        SetValue(a);
        InplaceNegativeSine();
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceTruncateBottom (const ElemType threshold)
    {
        if (IsEmpty())
            LogicError("InplaceTruncateBottom: Matrix is empty.");    

        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(N*1.0/threadsPerBlock); 
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _inplaceTruncateBottom<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,threshold,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignTruncateBottomOf (const GPUMatrix<ElemType>& a, const ElemType threshold)
    {
        if (a.IsEmpty())
            LogicError("AssignTruncateBottomOf: Matrix a is empty.");

        if (this!=&a)
        {
            Resize(a.GetNumRows(), a.GetNumCols());
        }

        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(N*1.0/threadsPerBlock);      
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _assignTruncateBottom<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,a.m_pArray,threshold,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceTruncateTop (const ElemType threshold)
    {
        if (IsEmpty())
            LogicError("InplaceTruncateTop: Matrix is empty.");
        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(N*1.0/threadsPerBlock);      
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _inplaceTruncateTop<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,threshold,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;        
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignTruncateTopOf (const GPUMatrix<ElemType>& a, const ElemType threshold)
    {
        if (a.IsEmpty())
            LogicError("AssignTruncateTopOf: Matrix a is empty.");

        if (this!=&a)
        {
            Resize(a.GetNumRows(), a.GetNumCols());
        }

        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(N*1.0/threadsPerBlock); 
        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _assignTruncateTop<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,a.m_pArray,threshold,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;        
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceTruncate(const ElemType threshold)
    {
        if (IsEmpty())
            LogicError("InplaceTruncate: Matrix is empty.");

        CUDA_LONG N = (CUDA_LONG)GetNumElements();
        int blocksPerGrid = (int)ceil(N*1.0 / threadsPerBlock);
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _inplaceTruncate<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(m_pArray, threshold, N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceSoftThreshold(const ElemType threshold)
    {
        if (IsEmpty())
            LogicError("InplaceSoftThreshold: Matrix is empty.");

        CUDA_LONG N = (CUDA_LONG)GetNumElements();
        int blocksPerGrid = (int)ceil(N*1.0 / threadsPerBlock);
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _inplaceSoftThreshold<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(m_pArray, threshold, N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
    }
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::SetToZeroIfAbsLessThan (const ElemType threshold)
    {
        if (IsEmpty())
            LogicError("SetToZeroIfAbsLessThan: Matrix is empty.");
        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(N*1.0/threadsPerBlock); 
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _setToZeroIfAbsLessThan<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,threshold,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;  
    }

    template<class ElemType>
    ElemType GPUMatrix<ElemType>::SumOfAbsElements() const
    {
        if (IsEmpty())
            LogicError("SumOfAbsElements: Matrix is empty");

        cublasHandle_t cuHandle = GetCublasHandle(GetComputeDeviceId());          
        if (sizeof(ElemType)==sizeof(float))
        {
            float res=0;
            CUBLAS_CALL(cublasSasum(cuHandle, (CUDA_LONG)GetNumElements(), reinterpret_cast<float*>(m_pArray), 1, &res));
            return res;
        }
        else
        {
            double res=0;
            CUBLAS_CALL(cublasDasum(cuHandle, (CUDA_LONG)GetNumElements(), reinterpret_cast<double*>(m_pArray), 1, &res));
            return ElemType(res);
        }         
    }

    template<class ElemType>
    ElemType GPUMatrix<ElemType>::SumOfElements() const
    {
        if (IsEmpty())
            LogicError("SumOfElements: Matrix is empty");

        PrepareDevice();
        ElemType* d_sum = NULL;
        ElemType h_sum;
        CUDA_CALL(cudaMalloc((void**)&d_sum,sizeof(ElemType)));
        //WARNING: THIS kernel is not the most efficient way!
        _reductionSum<ElemType><<<1,1024,0,t_stream>>>(m_pArray,d_sum,(CUDA_LONG)GetNumElements());
        CUDA_CALL(cudaMemcpy(&h_sum,d_sum,sizeof(ElemType),cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_sum));               
        return h_sum;        
    }

    
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSumOfElements(const GPUMatrix<ElemType>& a)
    {
        if (a.IsEmpty())
            LogicError("AssignSumOfElements: Matrix a is empty");

        Resize(1,1);

        PrepareDevice();     
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        //WARNING: THIS kernel is not the most efficient way!
        _reductionSumAndAssign<ElemType><<<1,1024>>>(m_pArray,a.m_pArray,(CUDA_LONG)a.GetNumElements(),(CUDA_LONG)GetNumElements());
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return (*this);
    }

    template<class ElemType>
    DeviceBoundNumber<ElemType> GPUMatrix<ElemType>::Sum_AsDeviceBoundNum() const
    {
        if (IsEmpty())
            LogicError("Matrix is empty");
        PrepareDevice();
        ElemType* d_sum = NULL;        
        CUDA_CALL(cudaMalloc((void**)&d_sum,sizeof(ElemType)));
        //WARNING: THIS kernel is not the most efficient way!
        _reductionSum<ElemType><<<1,1024,0,t_stream>>>(m_pArray,d_sum,(CUDA_LONG)GetNumElements());
        DeviceBoundNumber<ElemType> result;
        result.ShallowCopyFrom(d_sum,GetComputeDeviceId());
        return result;
    }

    template<class ElemType>
    ElemType GPUMatrix<ElemType>::Max() const
    {
        cublasHandle_t cuHandle = GetCublasHandle(GetComputeDeviceId());   
        ElemType res;
        if (sizeof(ElemType)==sizeof(float))
        {
            int resInd=0;
            cublasIsamax(cuHandle,(CUDA_LONG)GetNumElements(),reinterpret_cast<float*>(m_pArray),1,&resInd); 
            resInd--;
            CUDA_CALL(cudaMemcpy(reinterpret_cast<float*>(&res),reinterpret_cast<float*>(m_pArray+resInd),sizeof(float),cudaMemcpyDeviceToHost));
            return res;
        }
        else
        {
            int resInd=0;
            cublasIdamax(cuHandle,(CUDA_LONG)GetNumElements(),reinterpret_cast<double*>(m_pArray),1,&resInd);
            resInd--;
            CUDA_CALL(cudaMemcpy(reinterpret_cast<double*>(&res),m_pArray+resInd,sizeof(float),cudaMemcpyDeviceToHost));
            return res;
        }        
    }


    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::ElementMultiplyWith (const GPUMatrix<ElemType>& a)
    {
        if (IsEmpty() || a.IsEmpty())
            LogicError("ElementMultiplyWith: Matrix is empty.");

        GPUMatrix<ElemType>& us=*this;
        assert (us.GetNumRows() == a.GetNumRows() && us.GetNumCols() == a.GetNumCols());
        if (us.GetNumRows() != a.GetNumRows() || us.GetNumCols() != a.GetNumCols())
            InvalidArgument("The matrix dimensions do not match.");

        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(((double)N)/threadsPerBlock); 
        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _elemMul<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,a.m_pArray,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementProductOf (const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
    {
        if (a.IsEmpty() || b.IsEmpty())
            LogicError("AssignElementProductOf: Matrix is empty.");

        assert (a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
        if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
            InvalidArgument("The input matrix dimensions do not match.");

        Resize(a.GetNumRows(), a.GetNumCols());
        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(((double)N)/threadsPerBlock);  
        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _assignElementProductOf<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,a.m_pArray,b.m_pArray,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::ElementDivideBy(const GPUMatrix<ElemType>& a)
    {
        return AssignElementDivisionOf(*this, a);
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementDivisionOf (const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
    {
        if (a.IsEmpty() || b.IsEmpty())
            LogicError("AssignElementDivisionOf: Matrix is empty.");

        assert (a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
        if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
            InvalidArgument("The input matrix dimensions do not match.");

        Resize(a.GetNumRows(), a.GetNumCols());
        CUDA_LONG N=(CUDA_LONG)GetNumElements();
        int blocksPerGrid =(int)ceil(((double)N)/threadsPerBlock);  
        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _assignElementDivisionOf<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,a.m_pArray,b.m_pArray,N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
    }

    template<class ElemType>
    bool GPUMatrix<ElemType>::IsEqualTo(const GPUMatrix<ElemType>& a, const ElemType threshold /*= 1e-8*/) const
    {
        return AreEqual(*this, a, threshold);
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::VectorSum(const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c, const bool isColWise)
    {
        if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
        {
            InvalidArgument("All matrices must be on the same GPU");
        }

        a.PrepareDevice();

        if (a.IsEmpty())
            LogicError("VectorSum:  Input matrix is empty.");

        const CUDA_LONG n = (CUDA_LONG)a.GetNumRows();
        const CUDA_LONG m = (CUDA_LONG)a.GetNumCols();
        assert(m>0 && n>0); //converting from size_t to int may cause overflow

        cudaEvent_t done = nullptr;

        int blocksPerGrid = 0;
        if (isColWise)  //col-wise
        {
            c.Resize(1, m);
            blocksPerGrid = (int)ceil(1.0*m / threadsPerBlock);
        }
        else
        {
            c.Resize(n, 1);
            blocksPerGrid = (int)ceil(1.0*n / threadsPerBlock);
        }

        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _vectorSum<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(c.m_pArray, a.m_pArray, n, m, isColWise);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }
    template<class ElemType>
    void GPUMatrix<ElemType>::VectorNorm1(GPUMatrix<ElemType>& c, const bool isColWise) const
    {
        if (IsEmpty())
            LogicError("VectorNorm1: Matrix is empty.");

        const CUDA_LONG n = (CUDA_LONG)GetNumRows();
        const CUDA_LONG m = (CUDA_LONG)GetNumCols();
        assert (m>0 && n>0); //converting from size_t to int may cause overflow

        cudaEvent_t done = nullptr;  
        PrepareDevice();
        c.ChangeDeviceTo(GetComputeDeviceId());

        int blocksPerGrid=0;
        if (isColWise)  //col-wise
        {
            c.Resize(1,m);   
            blocksPerGrid =(int)ceil(1.0*m/threadsPerBlock);                                        
        }
        else
        {
            c.Resize(n, 1);
            blocksPerGrid =(int)ceil(1.0*n/threadsPerBlock);                        
        }       

        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));  
        _vectorNorm1<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(c.m_pArray, m_pArray,n,m,isColWise);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignVectorNorm1Of(GPUMatrix<ElemType>& a, const bool isColWise)
    {
        a.VectorNorm1(*this, isColWise);
        return *this;
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::VectorNorm2(GPUMatrix<ElemType>& c, const bool isColWise) const
    {
        if (IsEmpty())
            LogicError("VectorNorm2: Matrix is empty.");

        const CUDA_LONG n = (CUDA_LONG)GetNumRows();
        const CUDA_LONG m = (CUDA_LONG)GetNumCols();
        assert (m>0 && n>0); //converting from size_t to int may cause overflow

        cudaEvent_t done = nullptr;  
        PrepareDevice();
        c.ChangeDeviceTo(GetComputeDeviceId());

        int blocksPerGrid=0;
        if (isColWise)  //col-wise
        {
            c.Resize(1,m);   
            blocksPerGrid =(int)ceil(1.0*m/threadsPerBlock);                                        
        }
        else
        {
            c.Resize(n, 1);
            c.ChangeDeviceTo(GetComputeDeviceId());
            blocksPerGrid =(int)ceil(1.0*n/threadsPerBlock);                        
        }       

        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));  
        _vectorNorm2<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(c.m_pArray, m_pArray,n,m,isColWise);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignVectorNorm2Of(GPUMatrix<ElemType>& a, const bool isColWise)
    {
        a.VectorNorm2(*this, isColWise);
        return *this;
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::VectorNormInf(GPUMatrix<ElemType>& c, const bool isColWise) const
    {
        if (IsEmpty())
            LogicError("VectorMax: Matrix is empty.");

        //this implementation is not efficient
        GPUMatrix<ElemType> tmp(GetComputeDeviceId());
        GPUMatrix<ElemType> tmp1(GetComputeDeviceId());
        tmp.AssignAbsOf((*this));
        tmp.VectorMax(tmp1,c,isColWise);
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignVectorNormInfOf(GPUMatrix<ElemType>& a, const bool isColWise)
    {
        a.VectorNormInf(*this, isColWise);
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignInnerProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const bool isColWise)
    {
        InnerProduct (a, b, *this,isColWise);
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignKhatriRaoProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
    {
        if (a.IsEmpty() || b.IsEmpty())
            LogicError("AssignKhatriRaoProductOf: Matrix is empty.");

        CUDA_LONG cols = a.GetNumCols();
        assert (cols == b.GetNumCols());
        if (!(cols == b.GetNumCols()))
            InvalidArgument("AssignKhatriRaoProductOf: The input matrix dimensions do not match.");

        CUDA_LONG rowsA = (CUDA_LONG)a.GetNumRows();
        CUDA_LONG rowsB = (CUDA_LONG)b.GetNumRows();
        Resize(rowsA * rowsB, cols);
        float N=(float)GetNumElements();
        int blocksPerGrid =(int)ceil(N/threadsPerBlock);  
        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _assignKhatriRaoProductOf<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,a.m_pArray,b.m_pArray,rowsA, rowsB, cols);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    //column-wise reshaped product. Used to compute KhatriRaoProduct Gradient
    //   this = reshape each column of a from (K1xK2,1) to (K1, K2) 
    //   if each column of a is not transposed, each (K1, K2) times each column of b (K2, frames).
    //   the output is a (K1, frames) matrix
    //   if each column of a is tranposed, each (K1, K2)^T times each column of b(K1, frames) and output is (K2, frames)
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddColumnReshapeProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const bool transposeAColumn)
    {
        if (a.IsEmpty() || b.IsEmpty())
            LogicError("AddColumnReshapeProductOf: Matrix is empty.");

        CUDA_LONG cols = a.GetNumCols();
        assert (cols == b.GetNumCols());
        if (!(cols == b.GetNumCols()))
            InvalidArgument("AddColumnReshapeProductOf: The input matrix dimensions do not match.");

        CUDA_LONG rowsA = (CUDA_LONG)a.GetNumRows();
        CUDA_LONG rowsB = (CUDA_LONG)b.GetNumRows();
        if (rowsA % rowsB != 0)
            InvalidArgument("AddColumnReshapeProductOf: number of rows in a should be multiples of that in b.");

        CUDA_LONG rowsC = rowsA / rowsB;
        if (rowsC != GetNumRows() || cols != GetNumCols())
            InvalidArgument("AddColumnReshapeProductOf: This matrix does not have the right size.");

        float N=(float)GetNumElements();
        int blocksPerGrid =(int)ceil(N/threadsPerBlock);  
        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _addColumnReshapeProductOf<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray,a.m_pArray,b.m_pArray, rowsB, rowsC, cols, transposeAColumn);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddWithScaleOf(ElemType alpha, const GPUMatrix<ElemType>& a)
    {
        ScaleAndAdd(alpha, a, *this);
        return *this;
    }

    template<class ElemType>
    ElemType GPUMatrix<ElemType>::FrobeniusNorm() const
    {
        if (IsEmpty())
            LogicError("FrobeniusNorm: Matrix is empty.");

        PrepareDevice();
        ElemType* d_sum = NULL;
        ElemType h_sum=0;
        CUDA_CALL(cudaMalloc((void**)&d_sum,sizeof(ElemType)));
        //WARNING: THIS kernel is not the most efficient way!
        _reductionSum2<ElemType><<<1,1024,0,t_stream>>>(m_pArray,d_sum,(CUDA_LONG)GetNumElements(), true);
        CUDA_CALL(cudaMemcpy(&h_sum,d_sum,sizeof(ElemType),cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_sum));               

        return (h_sum); 
    }
    
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignFrobeniusNormOf (const GPUMatrix<ElemType>& a)
    {
        if (a.IsEmpty())
            LogicError("AssignFrobeniusNormOf: Matrix a is empty.");

        Resize(1,1);        
    
        PrepareDevice();
        //WARNING: THIS kernel is not the most efficient way!
        _reductionSum2<ElemType><<<1,1024,0,t_stream>>>(a.m_pArray,m_pArray,(CUDA_LONG)a.GetNumElements(), true);

        return *this;
    }

    template<class ElemType>
    ElemType GPUMatrix<ElemType>::MatrixNormInf() const
    {
        if (IsEmpty())
            LogicError("MatrixNorm1: Matrix is empty.");

        PrepareDevice();
        ElemType* d_maxAbs = NULL;
        ElemType h_maxAbs=0;
        CUDA_CALL(cudaMalloc((void**)&d_maxAbs,sizeof(ElemType)));
        //WARNING: THIS kernel is not the most efficient way!
        _reductionMatrixNormInf<ElemType><<<1,1024,0,t_stream>>>(m_pArray,d_maxAbs,(CUDA_LONG)GetNumElements());
        CUDA_CALL(cudaMemcpy(&h_maxAbs,d_maxAbs,sizeof(ElemType),cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_maxAbs));               
        return h_maxAbs; 
    }

    template<class ElemType>
    ElemType GPUMatrix<ElemType>::MatrixNorm1() const
    {
        if (IsEmpty())
            LogicError("MatrixNorm1: Matrix is empty.");
        return SumOfAbsElements();              
    }

    template<class ElemType>
    ElemType GPUMatrix<ElemType>::MatrixNorm0() const
    {
        if (IsEmpty())
            LogicError("MatrixNorm0: Matrix is empty.");

        PrepareDevice();
        ElemType* d_nz = NULL;
        ElemType h_nz=0;
        CUDA_CALL(cudaMalloc((void**)&d_nz,sizeof(ElemType)));
        //WARNING: THIS kernel is not the most efficient way!
        _reductionMatrixNorm0<ElemType><<<1,1024,0,t_stream>>>(m_pArray,d_nz,(CUDA_LONG)GetNumElements());
        CUDA_CALL(cudaMemcpy(&h_nz,d_nz,sizeof(ElemType),cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_nz));               
        return h_nz; 
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSignOf(const GPUMatrix<ElemType>& a)
    {
        if (a.IsEmpty())
            LogicError("AssignSignOf: Matrix a is empty.");

        if (this != &a)
            Resize(a.GetNumRows(), a.GetNumCols());

        PrepareDevice();
        cudaEvent_t done = nullptr;
        int blocksPerGrid=(int)ceil(1.0*GetNumElements()/threadsPerBlock);  
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _assignSignOf<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray, a.m_pArray, (CUDA_LONG)GetNumElements());
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));    
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddSignOf(const GPUMatrix<ElemType>& a)
    {
        if (a.IsEmpty())
            LogicError("AddSignOf: Matrix a is empty.");

        if (this != &a)
            Resize(a.GetNumRows(), a.GetNumCols());

        PrepareDevice();
        cudaEvent_t done = nullptr;
        int blocksPerGrid=(int)ceil(1.0*GetNumElements()/threadsPerBlock);  
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _addSignOf<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(m_pArray, a.m_pArray, (CUDA_LONG)GetNumElements());
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));    
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::VectorMax(GPUMatrix<ElemType>& maxIndexes, GPUMatrix<ElemType>& maxValues, const bool isColWise) const
    {
        if (IsEmpty())
            LogicError("VectorMax: Matrix is empty.");

        const GPUMatrix<ElemType>& us = *this;
        const CUDA_LONG m = (CUDA_LONG)GetNumRows();
        const CUDA_LONG n = (CUDA_LONG)GetNumCols();
        assert(m > 0 && n > 0); //converting from size_t to int may cause overflow

        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)     CUDA_CALL(cudaEventCreate(&done));
        if (isColWise)
        {
            maxValues.Resize(1, n);
            maxIndexes.Resize(1, n);

            int blocksPerGrid = n; //we'll have 1 block processing 1 column
            _vectorMaxMinReduce<ElemType, true><<<blocksPerGrid, threadsPerBlock, 0, t_stream>>>(us.m_pArray, maxIndexes.m_pArray, maxValues.m_pArray, m, n);

            /*int blocksPerGrid=(int)ceil(1.0*n/threadsPerBlock);
            _vectorMax<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(us.m_pArray,maxIndexes.m_pArray,maxValues.m_pArray,m,n,isColWise);*/
        }
        else
        {
            maxValues.Resize(m, 1);
            maxIndexes.Resize(m, 1);
            int blocksPerGrid = (int)ceil(1.0*m / threadsPerBlock);
            _vectorMax<ElemType><<<blocksPerGrid, threadsPerBlock, 0, t_stream>>>(us.m_pArray, maxIndexes.m_pArray, maxValues.m_pArray, m, n, isColWise);
        }
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }
    
    __global__ void _initIndicesForSort(uint64_t* indexes, CUDA_LONG crow, CUDA_LONG ccol)
    {
        CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
        if (id >= crow * ccol)
            return;
        uint32_t irow = id % crow;
        uint32_t icol = id / crow;
        indexes[id] = (static_cast<uint64_t>(irow) << 32) | icol;
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::VectorMax(GPUMatrix<ElemType>& maxIndexes, GPUMatrix<ElemType>& maxValues, const bool isColWise, int topK) const
    {
        if (IsEmpty())
            LogicError("VectorMax: Matrix is empty.");

        if (topK == 1)
        {
            VectorMax(maxIndexes, maxValues, isColWise);
            return;
        }

        if (!isColWise)
            RuntimeError("Row-wise TopK max is not supported.");

        const GPUMatrix<ElemType>& us = *this;
        const CUDA_LONG m = (CUDA_LONG)GetNumRows();
        const CUDA_LONG n = (CUDA_LONG)GetNumCols();
        assert(topK <= m);
        assert(m > 0 && n > 0); //converting from size_t to int may cause overflow

        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)     CUDA_CALL(cudaEventCreate(&done));
        maxValues.Resize(topK, n);
        maxIndexes.Resize(topK, n);

        // To sort matrix columns we use 2-pass _stable_ sort algorithm:
        // 1. Sort by values (descending) with corresponding row/col indexes.
        // 2. Sort by col indices (ascending) with corresponding values/row indices.
        // Indices are stored as 64-bit ints where low 32 bits represent column and high 32 bits - row index.
        // On the second pass only first 32 bits of the index are used in sorting, so SortPairs has
        // begin_bit and end_bit set accordingly.

        CUDA_LONG celt = static_cast<CUDA_LONG>(GetNumElements());
        ElemType* inVal = us.m_pArray;
        ElemType* outVal1 = nullptr;
        ElemType* outVal2 = nullptr;
        uint64_t* inIdx = nullptr;
        uint64_t* outIdx = nullptr;
        // Determine temp buffer size needed for SortPairsDescending to sort values on the first pass.
        size_t cbtemp = 0;
        // If first param is nullptr then no actual work is done except writing result to cbtemp.
        CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(nullptr, cbtemp, inVal, outVal1, inIdx, outIdx, celt, 0, sizeof(ElemType) * 8, t_stream));
        size_t ctemp1 = (cbtemp + sizeof(ElemType) - 1) / sizeof(ElemType);
        // Determine temp buffer size needed for SortPairs to sort indices on the second pass.
        cbtemp = 0;
        CUDA_CALL(cub::DeviceRadixSort::SortPairs(nullptr, cbtemp, outIdx, inIdx, outVal1, outVal2, celt, 0, 32, t_stream));
        size_t ctemp2 = (cbtemp + sizeof(ElemType) - 1) / sizeof(ElemType);
        size_t ctemp = std::max(ctemp1, ctemp2);
        cbtemp = ctemp * sizeof(ElemType);
        // ElemType count needed to store indices, accounting for natural alignment for uint64_t type.
        size_t cidx = ((celt + 1) * sizeof(uint64_t) - 1 + sizeof(ElemType) - 1) / sizeof(ElemType);
        // Get temp workspace.
        auto workspace = GetOrCreateWorkspace();
        // Resize to store: output values for the 1st and 2nd passes, input indices, output indices, and temp storage.
        workspace->Resize(m, 2 * n + (2 * cidx + ctemp + m - 1) / m);
        outVal1 = workspace->m_pArray;
        outVal2 = outVal1 + celt;
        inIdx = reinterpret_cast<uint64_t*>(outVal2 + celt);
        // Align indices pointer if needed.
        size_t cbAlign = reinterpret_cast<size_t>(inIdx) % sizeof(uint64_t);
        if (cbAlign != 0)
            reinterpret_cast<uint8_t*&>(inIdx) += sizeof(uint64_t) - cbAlign;
        outIdx = inIdx + celt;
        void* ptmp = outIdx + celt;
        assert(reinterpret_cast<ElemType*>(reinterpret_cast<uint8_t*>(ptmp) + cbtemp) <= workspace->m_pArray + workspace->GetNumElements());

        // Initialize indices.
        const int ThreadsPerBlock = 128;
        int cblock = (celt + ThreadsPerBlock - 1) / ThreadsPerBlock;
        _initIndicesForSort<<<cblock, ThreadsPerBlock, 0, t_stream>>>(inIdx, m, n);
        // Sort by values.
        CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(ptmp, cbtemp, inVal, outVal1, inIdx, outIdx, celt, 0, sizeof(ElemType) * 8, t_stream));
        // Sort by column indices. outIdx contains indices after the first pass so it's used as an input.
        CUDA_CALL(cub::DeviceRadixSort::SortPairs(ptmp, cbtemp, outIdx, inIdx, outVal1, outVal2, celt, 0, 32, t_stream));
        // Copy results.
        cblock = (topK * n + ThreadsPerBlock - 1) / ThreadsPerBlock;
        _copyTopKResults<<<cblock, ThreadsPerBlock, 0, t_stream>>>(inIdx, outVal2, maxIndexes.m_pArray, maxValues.m_pArray, m, n, topK);

        ReleaseWorkspace(std::move(workspace));

        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::VectorMin(GPUMatrix<ElemType>& minIndexes, GPUMatrix<ElemType>& minValues, const bool isColWise) const
    {
        if (IsEmpty())
            LogicError("VectorMax: Matrix is empty.");

        const GPUMatrix<ElemType>& us=*this;
        const int m = (int)GetNumRows();
        const int n = (int)GetNumCols();

        assert (m>0 && n>0); //converting from size_t to int may cause overflow
        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));                
        if (isColWise)
        {
            minValues.Resize(1, n);
            minIndexes.Resize(1, n);

            int blocksPerGrid = n; //we'll have 1 block processing 1 column
            _vectorMaxMinReduce<ElemType, false><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(us.m_pArray,minIndexes.m_pArray,minValues.m_pArray,m,n);

            /*
            int blocksPerGrid=(int)ceil(1.0*n/threadsPerBlock);  
            _vectorMin<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(us.m_pArray,minIndexes.m_pArray,minValues.m_pArray,m,n,isColWise);*/
        }
        else
        {
            minValues.Resize(m, 1);
            minIndexes.Resize(m, 1);
            int blocksPerGrid=(int)ceil(1.0*m/threadsPerBlock);  
            _vectorMin<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(us.m_pArray,minIndexes.m_pArray,minValues.m_pArray,m,n,isColWise);
        }
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    GPUMatrix<ElemType>&  GPUMatrix<ElemType>::AssignNumOfDiff(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, bool searchInCol)
    {
        if (a.GetNumCols() != b.GetNumCols())
            InvalidArgument("AssignNumOfDiff: a and b must have the same number of columns.");
        if (!searchInCol && a.GetNumRows() != b.GetNumRows())
            InvalidArgument("AssignNumOfDiff: a and b must have the same number of rows.");

        Resize(1, 1); //result should be one element

        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        if (!searchInCol)
        {
            //int blocksPerGrid=(int)ceil(1.0*a.GetNumElements()/threadsPerBlock);  
            //_assignNumOfDiff<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(a.m_pArray, b.m_pArray, m_pArray, a.GetNumElements());
            _assignNumOfDiff<ElemType><<<1, 1024, 0, t_stream>>>(a.m_pArray, b.m_pArray, m_pArray, (CUDA_LONG)a.GetNumElements());
        }
        else
        {
            const int blockSize = 1024;
            _assignNumOfDiffCol<blockSize><<<1, blockSize, 0, t_stream>>>(a.m_pArray, b.m_pArray, m_pArray, 
                static_cast<CUDA_LONG>(b.GetNumRows()), static_cast<CUDA_LONG>(a.GetNumCols()));
        }
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
    }

#pragma endregion Member BLAS Functions    

#pragma region Other helper functions
    template<class ElemType>
    void GPUMatrix<ElemType>::Print(const char* /*matrixName*/, size_t /*rowStart*/, size_t /*rowEnd*/, size_t /*colStart*/, size_t /*colEnd*/) const
    {
        NOT_IMPLEMENTED;
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::Print(const char* matrixName /*=nullptr*/) const
    {
        Print(matrixName, 0, GetNumRows()-1, 0, GetNumCols()-1);
    }

    // file I/O
    //matrixName is used to verify that correct matrix is read.
    template<class ElemType>
    void GPUMatrix<ElemType>::ReadFromFile(FILE*, const char * /*matrixName*/)
    {
        NOT_IMPLEMENTED;
    }

    //matrixName is used to verify that correct matrix is read.
    template<class ElemType>
    void GPUMatrix<ElemType>::WriteToFile(FILE*, const char * /*matrixName*/)
    {
        NOT_IMPLEMENTED;
    }

    //helpfer function used for convolution neural network 
    template<class ElemType>
    GPUMatrix<ElemType>&  GPUMatrix<ElemType>::AssignPackedConvolutionInput(const GPUMatrix<ElemType>& inputSubBatch, 
                                            const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                            const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                            const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample, 
                                            const bool zeroPadding)
    {
        assert (verticalSubsample <= kernelHeight && horizontalSubsample <= kernelWidth);

        size_t packedInputRows = kernelWidth * kernelHeight * inputChannels;
        size_t packedInputColsPerSample = outputWidth * outputHeight;
        size_t smallBatchSize = inputSubBatch.GetNumCols();
        Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
        if (zeroPadding) 
            SetValue((ElemType)0);

        PrepareDevice();
        int numThreadPerBlock = threadsPerBlock; 
#if 1
        int blocksPerGrid = (smallBatchSize * inputWidth*inputHeight*inputChannels + numThreadPerBlock - 1)/numThreadPerBlock; 
#else
        dim3 blocksPerGrid((inputWidth*inputHeight*inputChannels + numThreadPerBlock - 1)/numThreadPerBlock, smallBatchSize);
#endif
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _assignPackedConvolutionInput<<<blocksPerGrid, numThreadPerBlock,0,t_stream>>>(m_pArray, 
                                            inputSubBatch.m_pArray, 
                                            smallBatchSize,
                                            inputWidth, inputHeight, inputChannels,
                                            outputWidth, outputHeight, outputChannels,
                                            kernelWidth, kernelHeight, horizontalSubsample, verticalSubsample, zeroPadding);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    //helpfer function used for convolution neural network 
    template<class ElemType>
    GPUMatrix<ElemType>&  GPUMatrix<ElemType>::UnpackConvolutionInput(GPUMatrix<ElemType>& inputSubBatch, 
                                            const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                            const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                            const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample, 
                                            const bool zeroPadding) const
    {
        assert (verticalSubsample <= kernelHeight && horizontalSubsample <= kernelWidth);

        size_t smallBatchSize = inputSubBatch.GetNumCols();

        PrepareDevice();
        int numThreadPerBlock = threadsPerBlock; 
#if 1
        int blocksPerGrid = (smallBatchSize * inputWidth*inputHeight*inputChannels + numThreadPerBlock - 1)/numThreadPerBlock; 
#else
        dim3 blocksPerGrid((inputWidth*inputHeight*inputChannels + numThreadPerBlock - 1)/numThreadPerBlock, smallBatchSize);
#endif
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _unpackConvolutionInput<<<blocksPerGrid, numThreadPerBlock,0,t_stream>>>(m_pArray, 
                                            inputSubBatch.m_pArray, 
                                            smallBatchSize,
                                            inputWidth, inputHeight, inputChannels,
                                            outputWidth, outputHeight, outputChannels,
                                            kernelWidth, kernelHeight, horizontalSubsample, verticalSubsample, zeroPadding);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return inputSubBatch;
    }

    template<class ElemType>
    GPUMatrix<ElemType>&  GPUMatrix<ElemType>::AssignMaxPoolingResult(const GPUMatrix<ElemType>& inputBatch, const size_t channels, 
                                                const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample, 
                                                const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample, 
                                                const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
    {
        assert (verticalSubsample <= windowHeight && horizontalSubsample <= windowWidth);

        unsigned int batchSize = inputBatch.GetNumCols();
        Resize(outputSizePerSample, batchSize);

        int numThreadPerBlock = threadsPerBlock; 
        int blocksPerGrid = (batchSize * outputSizePerSample + numThreadPerBlock - 1)/numThreadPerBlock; 

        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _assignMaxPoolingResult<<<blocksPerGrid, numThreadPerBlock,0,t_stream>>>(m_pArray, inputBatch.m_pArray, batchSize, channels,
                                                 inputWidth, inputHeight,inputSizePerSample, 
                                                 outputWidth, outputHeight, outputSizePerSample, 
                                                 windowWidth, windowHeight, horizontalSubsample, verticalSubsample);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>&  GPUMatrix<ElemType>::AddMaxPoolingGradient(const GPUMatrix<ElemType>& outputGradientBatch, const GPUMatrix<ElemType>& inputBatch, const GPUMatrix<ElemType>& outputBatch, 
                                                const size_t channels, 
                                                const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample, 
                                                const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample, 
                                                const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
    {
        assert (verticalSubsample <= windowHeight && horizontalSubsample <= windowWidth);

        unsigned int batchSize = outputGradientBatch.GetNumCols();
        int numThreadPerBlock = threadsPerBlock; 

        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));

        int blocksPerGrid = (batchSize * inputSizePerSample + numThreadPerBlock - 1)/numThreadPerBlock; 
        _addMaxPoolingGradient<<<blocksPerGrid, numThreadPerBlock,0,t_stream>>>(m_pArray, outputGradientBatch.m_pArray, inputBatch.m_pArray, outputBatch.m_pArray, batchSize, channels,
                                                 inputWidth, inputHeight,inputSizePerSample, 
                                                 outputWidth, outputHeight,  outputSizePerSample, 
                                                 windowWidth, windowHeight, horizontalSubsample, verticalSubsample);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>&  GPUMatrix<ElemType>::AssignAveragePoolingResult(const GPUMatrix<ElemType>& inputBatch, const size_t channels, 
                                                const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample, 
                                                const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample, 
                                                const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
    {
        assert (verticalSubsample <= windowHeight && horizontalSubsample <= windowWidth);

        unsigned int batchSize = inputBatch.GetNumCols();
        Resize(outputSizePerSample, batchSize);

        int numThreadPerBlock = threadsPerBlock; 
        int blocksPerGrid = (batchSize * outputSizePerSample + numThreadPerBlock - 1)/numThreadPerBlock; 

        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _assignAveragePoolingResult<<<blocksPerGrid, numThreadPerBlock,0,t_stream>>>(m_pArray, inputBatch.m_pArray, batchSize, channels,
                                                 inputWidth, inputHeight,inputSizePerSample, 
                                                 outputWidth, outputHeight, outputSizePerSample, 
                                                 windowWidth, windowHeight, horizontalSubsample, verticalSubsample);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>&  GPUMatrix<ElemType>::AddAveragePoolingGradient(const GPUMatrix<ElemType>& outputGradientBatch, 
                                                const size_t channels, 
                                                const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample, 
                                                const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample, 
                                                const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
    {
        assert (verticalSubsample <= windowHeight && horizontalSubsample <= windowWidth);

        size_t batchSize = outputGradientBatch.GetNumCols();
        int numThreadPerBlock = threadsPerBlock; 

        PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));

        size_t blocksPerGrid = (batchSize * inputSizePerSample + numThreadPerBlock - 1)/numThreadPerBlock; 
        _addAveragePoolingGradient<<<blocksPerGrid, numThreadPerBlock,0,t_stream>>>(m_pArray, outputGradientBatch.m_pArray, (CUDA_LONG)batchSize, channels,
                                                 inputWidth, inputHeight,inputSizePerSample, 
                                                 outputWidth, outputHeight,  outputSizePerSample, 
                                                 windowWidth, windowHeight, horizontalSubsample, verticalSubsample);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

#pragma endregion Other helper functions

#pragma region Static BLAS Functions
    template<class ElemType>
    void GPUMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB, 
        ElemType beta, GPUMatrix<ElemType>& c)
    {
        a.PrepareDevice();
        if ((a.GetComputeDeviceId()!=b.GetComputeDeviceId()) || (b.GetComputeDeviceId()!=c.GetComputeDeviceId())) //different GPUs
            InvalidArgument("All matrices must be on the same GPU");

        cublasHandle_t cuHandle = GetCublasHandle(b.GetComputeDeviceId());
        cublasOperation_t transA =  transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t transB =  transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
        int m = int(transposeA ? a.m_numCols : a.m_numRows);
        int n = int(transposeB ? b.m_numRows : b.m_numCols);
        int k = int(transposeA ? a.m_numRows : a.m_numCols);
        int l = int(transposeB ? b.m_numCols : b.m_numRows);
        c.Resize(m,n);

        if (!(m>0 && k>0 && l>0 && n>0)) 
        {
            RuntimeError("!(m>0 && k>0 && l>0 && n>0)");  //converting from size_t to int may cause overflow
        }
        if (k!=l) 
        {
            RuntimeError("matrix dim mismatch in MultiplyAndWeightedAdd");
        }
        if (sizeof(ElemType)==sizeof(float))
        {
            CUBLAS_CALL(cublasSgemm(cuHandle,transA,transB,m,n,k,reinterpret_cast<float*>(&alpha),reinterpret_cast<float*>(a.m_pArray),(int)a.m_numRows,reinterpret_cast<float*>(b.m_pArray),(int)b.m_numRows,reinterpret_cast<float*>(&beta),reinterpret_cast<float*>(c.m_pArray),(int)c.m_numRows));
        }
        else if (sizeof(ElemType)==sizeof(double))
        {            
            CUBLAS_CALL(cublasDgemm(cuHandle,transA,transB,m,n,k,reinterpret_cast<double*>(&alpha),reinterpret_cast<double*>(a.m_pArray),(int)a.m_numRows,reinterpret_cast<double*>(b.m_pArray),(int)b.m_numRows,reinterpret_cast<double*>(&beta),reinterpret_cast<double*>(c.m_pArray),(int)c.m_numRows));
        }
        else 
        {
            RuntimeError("Unsupported template argument in GPUMatrix");             
        }
        c.m_numRows=m;
        c.m_numCols=n;
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::Multiply1x1AndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, ElemType beta, GPUMatrix<ElemType>& c)
    {
        a.PrepareDevice();
        if ((a.GetComputeDeviceId() != b.GetComputeDeviceId()) || (b.GetComputeDeviceId() != c.GetComputeDeviceId())) //different GPUs
            InvalidArgument("All matrices must be on the same GPU");
        CUDA_LONG N = (CUDA_LONG)c.GetNumElements();
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _multiply1x1AndWeightedAdd<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(alpha, a.m_pArray, b.m_pArray, beta, c.m_pArray, N);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::MultiplyAndAdd(const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB, GPUMatrix<ElemType>& c)
    {
        return GPUMatrix<ElemType>::MultiplyAndWeightedAdd(1, a, transposeA, b, transposeB, 1, c);
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::Multiply(const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB, GPUMatrix<ElemType>& c)
    {    
        return GPUMatrix<ElemType>::MultiplyAndWeightedAdd(1, a, transposeA, b, transposeB, 0, c);
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::Multiply(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
    {
        return GPUMatrix<ElemType>::MultiplyAndWeightedAdd(1, a, false, b, false, 0, c);
    }

    /// <summary>Matrix-scalar multiply with col-major matrices: c = alpha * a + c</summary>
    /// if a is a column vector, add to all columns of c 
    /// if a is a row vector, add to all rows of c    
    /// if a is a scalar, add to all elements of c
    /// <param name="alpha">Scalar</param>
    /// <param name="a">Input matrix</param>
    /// <param name="c">Resulting matrix, user is responsible for allocating this</param>
    template<class ElemType>
    void GPUMatrix<ElemType>::ScaleAndAdd(ElemType alpha,const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c)
    {
        if (a.GetComputeDeviceId()!=c.GetComputeDeviceId())
        {
            InvalidArgument("All matrices must be on the same GPU");
        }
        else
        {
            a.PrepareDevice();
            if (a.IsEmpty() || c.IsEmpty())
                LogicError("ScaleAndAdd:  one of the input matrices is empty.");
            //if (a.GetNumRows() != 1 && a.GetNumCols() != 1) // a is not a col or row vector
            if (a.GetNumRows()==c.GetNumRows() && a.GetNumCols()==c.GetNumCols()) // dimensions match
            {
                const int m = (int)a.GetNumRows();
                const int n = (int)a.GetNumCols();
                const int len = m * n;
                const int incx = 1;
                const int incy = 1;

                assert (m>0 && n>0 && len>0); //converting from size_t to int may cause overflow
                assert ((int)c.GetNumRows() == m && (int)c.GetNumCols() == n);
                if ((int)c.GetNumRows() != m || (int)c.GetNumCols() != n)
                    InvalidArgument("dimension of matrix c does not match dimension of matrix a.");

                cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
                if (sizeof(ElemType) == sizeof(float))
                {
                    CUBLAS_CALL(cublasSaxpy(cuHandle,len,reinterpret_cast <float*>(&alpha),reinterpret_cast <float*>(a.m_pArray),incx,reinterpret_cast <float*>(c.m_pArray) ,incy));                
                }
                else if (sizeof(ElemType) == sizeof(double))
                {   
                    CUBLAS_CALL(cublasDaxpy(cuHandle,len,reinterpret_cast <double*>(&alpha),reinterpret_cast <double*>(a.m_pArray),incx,reinterpret_cast <double*>(c.m_pArray) ,incy)); 
                }
                else 
                {
                    RuntimeError("Unsupported template argument in GPUMatrix"); 
                }
            }
            else if (a.GetNumElements() == 1)
            {
                CUDA_LONG N=(CUDA_LONG)c.GetNumElements();
                int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);
                c.PrepareDevice();
                cudaEvent_t done = nullptr;
                if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
                _scaleAndAddScalar<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(c.m_pArray, N, alpha, a.m_pArray);
                if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
                if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
                if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
            }
            else if (a.GetNumCols() == 1) //col vector, add it to all columns
            {                
                CUDA_LONG m = (CUDA_LONG)c.GetNumRows();
                CUDA_LONG n = (CUDA_LONG)c.GetNumCols();                
                if (m != (CUDA_LONG)a.GetNumRows())
                    InvalidArgument("To add column vector, rows should match.");

                cudaEvent_t done = nullptr;
                int blocksPerGrid = (int)(ceil(1.0*m*n / threadsPerBlock));
                if (do_sync)    CUDA_CALL(cudaEventCreate(&done));   
#ifdef VALIDATION
                printf(">>>> CUDA compute device is %d\n", a.GetComputeDeviceId());
                printf(">>>> a.m_pArray = %p, c.m_pArray = %p, alpha = %f, m = %ld, n = %ld\n", a.m_pArray,c.m_pArray,alpha,m,n);   
                for (int i=0; i < 2; i++)
                {
                    ElemType buffer[10] = {-1.234f};
                    cudaError_t error = cudaMemcpy(buffer, !i?a.m_pArray:c.m_pArray, sizeof(buffer), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                    if (error == cudaError::cudaSuccess)
                        printf("buffer valid\n"); 
                }
#endif

                _matrixVectorColumnWiseAddWithThreadPerElem<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(a.m_pArray,c.m_pArray,alpha,m,n);


                if (do_sync)    CUDA_CALL(cudaEventRecord(done));
                if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));   
                if (do_sync)    CUDA_CALL(cudaEventDestroy(done));                
            }
            else  if (a.GetNumRows()==1)  //row vector, add it to all rows
            {
                cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
                int m = (int)c.GetNumRows();
                int n = (int)c.GetNumCols();
                assert (n == (int)a.GetNumCols());
                if (n != (int)a.GetNumCols())
                    InvalidArgument("To add row vector, cols should match.");

                if (sizeof(ElemType) == sizeof(double))
                {
                    foreach_row(i,c)
                    {
                        CUBLAS_CALL(cublasDaxpy(cuHandle,n,reinterpret_cast <double*>(&alpha),reinterpret_cast <double*>(a.m_pArray),1,reinterpret_cast <double*>(c.m_pArray+i),m));
                    }                    
                }
                else
                {
                    foreach_row(i,c)
                    {
                        CUBLAS_CALL(cublasSaxpy(cuHandle,n,reinterpret_cast <float*>(&alpha),reinterpret_cast <float*>(a.m_pArray),1,reinterpret_cast <float*>(c.m_pArray+i),m));
                    }                    
                }
            }
            else
                InvalidArgument("dimension of matrix c does not match dimension of matrix a.");
        }
    }

    /// <summary>c += alpha * (a-b)</summary>
    /// if a, b, c  must have same dim 
    /// <param name="alpha">Scalar</param>
    /// <param name="a">Input matrix</param>
    /// <param name="b">Input matrix</param>
    /// <param name="c">Resulting matrix, user is responsible for allocating this</param>
    template<class ElemType>
    void GPUMatrix<ElemType>::AddScaledDifference(const ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
    {
        if (a.GetComputeDeviceId()!=c.GetComputeDeviceId())
        {
            InvalidArgument("All matrices must be on the same GPU");
        }
        else
        {
            a.PrepareDevice();

            assert(a.GetNumRows() == b.GetNumRows() && a.GetNumRows() == c.GetNumRows() &&
                a.GetNumCols() == b.GetNumCols() && a.GetNumCols() == c.GetNumCols());

            if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumRows() == c.GetNumRows() &&
                a.GetNumCols() == b.GetNumCols() && a.GetNumCols() == c.GetNumCols()))
            {
                InvalidArgument("AddScaledDifference:  a, b, and c must have same dimension.");
            }

            if (a.IsEmpty())
                LogicError("AddScaledDifference:  Input matrix a is empty.");

            cudaEvent_t done = nullptr;
            CUDA_LONG n=(CUDA_LONG)a.GetNumElements();            
            int blocksPerGrid=(int)ceil(1.0*n/threadsPerBlock);  
            if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
            _addScaledDifference<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(alpha, a.m_pArray, b.m_pArray, c.m_pArray, n);
            if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
            if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));   
            if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        }
    }

    /// <summary> c = alpha * (a-b)</summary>
    /// if a, b, c  must have same dim 
    /// <param name="alpha">Scalar</param>
    /// <param name="a">Input matrix</param>
    /// <param name="b">Input matrix</param>
    /// <param name="c">Resulting matrix, user is responsible for allocating this</param>
    template<class ElemType>    
    void GPUMatrix<ElemType>::AssignScaledDifference(const ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
    {
        if (a.GetComputeDeviceId()!=c.GetComputeDeviceId())
        {
            InvalidArgument("All matrices must be on the same GPU");
        }
        else
        {
            a.PrepareDevice();

            assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols() );

            if (!(a.GetNumRows() == b.GetNumRows()  && a.GetNumCols() == b.GetNumCols()))
            {
                InvalidArgument("AssignScaledDifference:  a, b must have same dimension.");
            }

            if (a.IsEmpty())
                LogicError("AssignScaledDifference:  Input matrix a is empty.");

            if (&c != &a && &c != &b)
                c.Resize(a.GetNumRows(), a.GetNumCols());

            cudaEvent_t done = nullptr;
            CUDA_LONG n=(CUDA_LONG)a.GetNumElements();            
            int blocksPerGrid=(int)ceil(1.0*n/threadsPerBlock);  
            if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
            _assignScaledDifference<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(alpha, a.m_pArray, b.m_pArray, c.m_pArray, n);
            if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
            if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));   
            if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        }
    }

    /// <summary>c += alpha * (a-b)</summary>
    /// if a, b, c  must have same dim 
    /// <param name="alpha">1X1 matrix</param>
    /// <param name="a">Input matrix</param>
    /// <param name="b">Input matrix</param>
    /// <param name="c">Resulting matrix, user is responsible for allocating this</param>
    template<class ElemType>
    void GPUMatrix<ElemType>::AddScaledDifference(const GPUMatrix<ElemType>& alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
    {
        assert(alpha.GetNumElements() == 1);
        if (!(alpha.GetNumElements() == 1))
            InvalidArgument("AddScaledDifference:  alpha must be a 1X1 matrix.");

        if (a.GetComputeDeviceId()!=c.GetComputeDeviceId())
        {
            InvalidArgument("All matrices must be on the same GPU");
        }
        else
        {
            a.PrepareDevice();

            assert(a.GetNumRows() == b.GetNumRows() && a.GetNumRows() == c.GetNumRows() &&
                a.GetNumCols() == b.GetNumCols() && a.GetNumCols() == c.GetNumCols());

            if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumRows() == c.GetNumRows() &&
                a.GetNumCols() == b.GetNumCols() && a.GetNumCols() == c.GetNumCols()))
            {
                InvalidArgument("AddScaledDifference:  a, b, and c must have same dimension.");
            }

            if (a.IsEmpty())
                LogicError("AddScaledDifference:  Input matrix a is empty.");

            cudaEvent_t done = nullptr;
            CUDA_LONG n=(CUDA_LONG)a.GetNumElements();            
            int blocksPerGrid=(int)ceil(1.0*n/threadsPerBlock);  
            if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
            _addScaledDifference<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(alpha.m_pArray, a.m_pArray, b.m_pArray, c.m_pArray, n);
            if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
            if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));   
            if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        }
    }

    /// <summary> c = alpha * (a-b)</summary>
    /// if a, b, c  must have same dim 
    /// <param name="alpha">Scalar</param>
    /// <param name="a">Input matrix</param>
    /// <param name="b">Input matrix</param>
    /// <param name="c">Resulting matrix, user is responsible for allocating this</param>
    template<class ElemType>    
    void GPUMatrix<ElemType>::AssignScaledDifference(const GPUMatrix<ElemType>& alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
    {
        assert(alpha.GetNumElements() == 1);
        if (!(alpha.GetNumElements() == 1))
            InvalidArgument("AddScaledDifference:  alpha must be a 1X1 matrix.");

        if (a.GetComputeDeviceId()!=c.GetComputeDeviceId())
        {
            InvalidArgument("All matrices must be on the same GPU");
        }
        else
        {
            a.PrepareDevice();

            assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols() );

            if (!(a.GetNumRows() == b.GetNumRows()  && a.GetNumCols() == b.GetNumCols()))
            {
                InvalidArgument("AssignScaledDifference:  a, b must have same dimension.");
            }

            if (a.IsEmpty())
                LogicError("AssignScaledDifference:  Input matrix a is empty.");

            c.Resize(a.GetNumRows(), a.GetNumCols());

            cudaEvent_t done = nullptr;
            CUDA_LONG n=(CUDA_LONG)a.GetNumElements();            
            int blocksPerGrid=(int)ceil(1.0*n/threadsPerBlock);  
            if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
            _assignScaledDifference<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(alpha.m_pArray, a.m_pArray, b.m_pArray, c.m_pArray, n);
            if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
            if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
            if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        }
    }

    //c[ci,cj] += a[ai,aj]
    template<class ElemType>
    void GPUMatrix<ElemType>::AddElementToElement(const GPUMatrix<ElemType>& a, const size_t ai, const size_t aj, GPUMatrix<ElemType>& c, const size_t ci, const size_t cj)
    {
        if (ai >= a.GetNumRows() || aj >=a.GetNumCols() ||
            ci >= c.GetNumRows() || cj >=c.GetNumCols())
            InvalidArgument("AddElementToElement:  index out of range.");

        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        int blocksPerGrid=1;  //only one element
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));        
        _addElementToElement<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(a.m_pArray, (CUDA_LONG)a.LocateElement(ai, aj), c.m_pArray, (CUDA_LONG)c.LocateElement(ci, cj));
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));  
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::Scale(ElemType alpha, GPUMatrix<ElemType>& a)
    {   
        cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
        if (sizeof(ElemType)==sizeof(float))
        {
            float alph = (float)alpha;            
            CUBLAS_CALL(cublasSscal(cuHandle,int(a.m_numRows*a.m_numCols),&alph,(float*)a.m_pArray,1));
        }
        else if (sizeof(ElemType)==sizeof(double))
        {
            double alph = alpha;
            CUBLAS_CALL(cublasDscal(cuHandle,int(a.m_numRows*a.m_numCols),&alph,(double*)a.m_pArray,1));
        }
        else 
        {
            RuntimeError("Unsupported template argument in GPUMatrix");            
        }
    }


    template<class ElemType>
    void GPUMatrix<ElemType>::Scale(GPUMatrix<ElemType>& alpha, GPUMatrix<ElemType>& a)
    {           
        if (alpha.GetNumElements()!=1)
        {
            RuntimeError("Matrix alpha must be 1x1");
        }
        cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
        cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_DEVICE);
        if (sizeof(ElemType)==sizeof(float))
        {
            CUBLAS_CALL(cublasSscal(cuHandle,int(a.m_numRows*a.m_numCols),(float*)alpha.m_pArray,(float*)a.m_pArray,1));
        }
        else if (sizeof(ElemType)==sizeof(double))
        {            
            CUBLAS_CALL(cublasDscal(cuHandle,int(a.m_numRows*a.m_numCols),(double*)alpha.m_pArray,(double*)a.m_pArray,1));
        }
        else 
        {
            cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_HOST);
            RuntimeError("Unsupported template argument in GPUMatrix");            
        }
        cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_HOST);
    }

    template<class ElemType> //c = alpha * a
    void GPUMatrix<ElemType>::Scale(ElemType alpha, const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c)
    {
        if (a.IsEmpty())
            LogicError("Scale:  Input matrix a is empty.");

        c=a;
        Scale(alpha,c);
    }


    template<class ElemType>
    void GPUMatrix<ElemType>::InnerProduct (const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const bool isColWise)
    {
        if (a.GetComputeDeviceId()!=b.GetComputeDeviceId() || b.GetComputeDeviceId()!=c.GetComputeDeviceId()) //different GPUs
            InvalidArgument("All matrices must be on the same GPU");

        if (a.IsEmpty() || b.IsEmpty())
            LogicError("Scale:  one of the input matrices is empty.");

        const int m = (int)a.GetNumRows();
        const int n = (int)a.GetNumCols();
        const int k = (int)b.GetNumRows();
        const int l = (int)b.GetNumCols();

        assert (m>0 && n>0 && k>0 && l>0); //converting from size_t to int may cause overflow
        assert (m==k && n==l); //converting from size_t to int may cause overflow
        if (m!=k || n!=l)
            InvalidArgument("Matrices a and b should have same dimension.");

        if (isColWise)
            c.Resize(1,n);
        else
            c.Resize(m,1);

        if ((isColWise && m == 1) || !isColWise && n == 1)  //in this case it's equivalent to element-wise product
        {
            c.AssignElementProductOf(a, b);
        }
        else 
        {
            cudaEvent_t done = nullptr;  
            c.PrepareDevice();

            int blocksPerGrid=0;
            if (isColWise)  //col-wise
            {
                c.Resize(1,n);   
                blocksPerGrid =(int)ceil(1.0*n/threadsPerBlock);                                        
            }
            else
            {
                c.Resize(m, 1);
                blocksPerGrid =(int)ceil(1.0*m/threadsPerBlock);                        
            }       

            if (do_sync)    CUDA_CALL(cudaEventCreate(&done));  
            _innerProduct<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(c.m_pArray, a.m_pArray,b.m_pArray,m,n,isColWise);
            if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
            if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
            if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        }             
    }

    template<class ElemType>
    ElemType GPUMatrix<ElemType>::InnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
    {
        if (a.IsEmpty() || b.IsEmpty())
            LogicError("InnerProductOfMatrices:  one of the input matrices is empty.");

        const int m = (int)a.GetNumRows();
        const int n = (int)a.GetNumCols();
        const int k = (int)b.GetNumRows();
        const int l = (int)b.GetNumCols();

        assert (m>0 && n>0 && k>0 && l>0); //converting from size_t to int may cause overflow
        assert (m==k && n==l); //converting from size_t to int may cause overflow
        if (m!=k || n!=l)
            InvalidArgument("InnerProductOfMatrices: Matrices a and b should have same dimension.");

        cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
        if (sizeof(ElemType) == sizeof(double))
        {
            double tmp=0;                        
            CUBLAS_CALL(cublasDdot(cuHandle,m*n, reinterpret_cast <double*>(a.m_pArray), 1, reinterpret_cast <double*>(b.m_pArray), 1,&tmp));
            return ElemType(tmp);
            //return (ElemType)ddot((int)a.GetNumElements(), reinterpret_cast <double*>(a.m_pArray), 1, reinterpret_cast <double*>(b.m_pArray), 1);
        }
        else
        {
            float tmp=0;                        
            CUBLAS_CALL(cublasSdot(cuHandle,m*n, reinterpret_cast <float*>(a.m_pArray), 1, reinterpret_cast <float*>(b.m_pArray), 1,&tmp));
            return tmp;
            //return (ElemType)sdot((int)a.GetNumElements(), reinterpret_cast <float*>(a.m_pArray), 1, reinterpret_cast <float*>(b.m_pArray), 1);
        }
    }


    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignInnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
    {
        if (a.IsEmpty() || b.IsEmpty())
            LogicError("InnerProductOfMatrices:  one of the input matrices is empty.");        

        Resize(1,1);

        const int m = (int)a.GetNumRows();
        const int n = (int)a.GetNumCols();
        const int k = (int)b.GetNumRows();
        const int l = (int)b.GetNumCols();

        assert (m>0 && n>0 && k>0 && l>0); //converting from size_t to int may cause overflow
        assert (m==k && n==l); //converting from size_t to int may cause overflow
        if (m!=k || n!=l)
            InvalidArgument("InnerProductOfMatrices: Matrices a and b should have same dimension.");

        cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
        cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_DEVICE);
        if (sizeof(ElemType) == sizeof(double))
        {   
            CUBLAS_CALL(cublasDdot(cuHandle,m*n, reinterpret_cast <double*>(a.m_pArray), 1, reinterpret_cast <double*>(b.m_pArray), 1,reinterpret_cast <double*>(m_pArray)));                    
        }
        else
        {   
            CUBLAS_CALL(cublasSdot(cuHandle,m*n, reinterpret_cast <float*>(a.m_pArray), 1, reinterpret_cast <float*>(b.m_pArray), 1,reinterpret_cast <float*>(m_pArray)));                      
        }
        cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_HOST);
        return *this;
    }


    template<class ElemType>
    void GPUMatrix<ElemType>::ElementWisePower(ElemType alpha, const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c)
    {
        if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
        {
            InvalidArgument("All matrices must be on the same GPU");
        }
        else 
        {
            if (a.IsEmpty())
                LogicError("ElementWisePower:  The input matrix a is empty.");

            c.Resize(a.GetNumRows(), a.GetNumCols());

            cudaEvent_t done = nullptr;
            a.PrepareDevice();
            if (do_sync)    CUDA_CALL(cudaEventCreate(&done));            
            CUDA_LONG N=(CUDA_LONG)a.GetNumElements();
            int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);                
            _elementWisePowerOnCuda<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(alpha,a.m_pArray,c.m_pArray,N);
            if (do_sync)    CUDA_CALL(cudaEventRecord(done));        
            if (do_sync)    CUDA_CALL(cudaEventSynchronize(done)); 
            if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        }
    }

    template<class ElemType>
    bool GPUMatrix<ElemType>::AreEqual(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const ElemType threshold /*= 1e-8*/)
    {
        if (a.IsEmpty() || b.IsEmpty())
            LogicError("AreEqual: one of the input matrices is empty.");

        if (a.GetNumRows()  != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
            return false;

        bool bResult = false;

        a.PrepareDevice();
        long *res = new long[1];
        res[0]=1;
        long *d_res = NULL;
        CUDA_CALL(cudaMalloc((void**)&d_res,sizeof(long)*1));
        CUDA_CALL(cudaMemcpy(d_res,res,sizeof(long)*1,cudaMemcpyHostToDevice));
        CUDA_LONG N=(CUDA_LONG)a.GetNumElements();
        int blocksPerGrid =(int)ceil(1.0*N/threadsPerBlock);
        _areEqual<ElemType><<<blocksPerGrid,threadsPerBlock,0,t_stream>>>(a.m_pArray,b.m_pArray,N,threshold,d_res);
        CUDA_CALL(cudaMemcpy(res,d_res,sizeof(long)*1,cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_res));
        if (res[0]!=0)
            bResult = true;
        delete [] res;
        return bResult;
    }

    // see Matrix<ElemType>::TensorShuffleScaleAndAdd() for comments
    template<class ElemType>
    void GPUMatrix<ElemType>::TensorShuffleScaleAndAdd(ElemType keepWeight, const GPUMatrix<ElemType>& a, size_t D, size_t S, size_t M, size_t K, size_t T, ElemType scaleFactor, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
    {
        CUDA_LONG N = (CUDA_LONG)c.GetNumElements();
        assert(N == (CUDA_LONG)a.GetNumElements() && N == (CUDA_LONG)b.GetNumElements());
        assert(a.GetComputeDeviceId() == c.GetComputeDeviceId() && b.GetComputeDeviceId() == c.GetComputeDeviceId());
        a.PrepareDevice();
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        _tensorShuffleScaleAndAdd<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(keepWeight, a.m_pArray, D, S, M, K, T, scaleFactor, b.m_pArray, c.m_pArray);
        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
    }

    template<class ElemType>
    bool GPUMatrix<ElemType>::HasElement(const GPUMatrix<ElemType>& a, const ElemType v)
    {
        if (a.IsEmpty())
            LogicError("HasElement: the input matrix is empty.");

        bool bResult = false; 
        a.PrepareDevice();
        ElemType *res = new ElemType[2];
        res[0] = v;
        res[1] = 0;
        ElemType *d_res = NULL;
        CUDA_CALL(cudaMalloc((void**)&d_res, sizeof(ElemType) * 2));
        CUDA_CALL(cudaMemcpy(d_res, res, sizeof(ElemType) * 2, cudaMemcpyHostToDevice));
        CUDA_LONG N = (CUDA_LONG)a.GetNumElements();
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        _hasElement<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(a.m_pArray, N, d_res);
        CUDA_CALL(cudaMemcpy(res, d_res, sizeof(ElemType) * 2, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_res));
        if (res[1] != 0)
            bResult = true; 
        else
            bResult = false;

        delete [] res;
        return bResult;
    }

    template<class ElemType>
    GPUMatrix<ElemType>  GPUMatrix<ElemType>::Ones(const size_t rows, const size_t cols, int deviceId)
    {
        GPUMatrix<ElemType> c(rows, cols, deviceId); //will initialize to 0
        c.SetValue(1);
        return c;
    }

    template<class ElemType>
    GPUMatrix<ElemType>  GPUMatrix<ElemType>::Zeros(const size_t rows, const size_t cols, int deviceId)
    {
        GPUMatrix<ElemType> c(rows, cols, deviceId); //will initialize to 0
        //c.SetValue(0);
        return c;
    }

    template<class ElemType>
    GPUMatrix<ElemType>  GPUMatrix<ElemType>::Eye(const size_t rows, int deviceId)
    {
        GPUMatrix<ElemType> c(rows, rows, deviceId); //will initialize to 0
        c.SetDiagonalValue(1);
        return c;
    }

    template<class ElemType>
    GPUMatrix<ElemType>  GPUMatrix<ElemType>::RandomUniform(const size_t rows, const size_t cols, int deviceId, const ElemType low, const ElemType high, unsigned long seed)
    {
        GPUMatrix<ElemType> c(rows, cols, deviceId); //will initialize to 0
        c.SetUniformRandomValue(low, high, seed);
        return c;
    }

    template<class ElemType>
    GPUMatrix<ElemType> GPUMatrix<ElemType>::RandomGaussian(const size_t rows, const size_t cols, int deviceId, const ElemType mean, const ElemType sigma, unsigned long seed)
    {
        GPUMatrix<ElemType> c(rows, cols, deviceId); //will initialize to 0
        c.SetGaussianRandomValue(mean, sigma, seed);
        return c;
    }

    template<class ElemType>
    ElemType GPUMatrix<ElemType>::GetLearnRateForBlock_Helper(const GPUMatrix<ElemType> &Gradients, const GPUMatrix<ElemType> &SmoothedGradients)
    {                
        Gradients.PrepareDevice();
        ElemType* d_res=NULL;
        CUDA_CALL(cudaMalloc((void**)&d_res,sizeof(ElemType))); //we allocate memory on the device

        //Compute inner product of matrices and keep it on device
        const int m = (int)Gradients.GetNumRows();
        const int n = (int)Gradients.GetNumCols();
        const int k = (int)SmoothedGradients.GetNumRows();
        const int l = (int)SmoothedGradients.GetNumCols();
        assert (m>0 && n>0 && k>0 && l>0); //converting from size_t to int may cause overflow
        assert (m==k && n==l); //converting from size_t to int may cause overflow
        if (m!=k || n!=l) InvalidArgument("InnerProductOfMatrices: Matrices a and b should have same dimension.");

        if (sizeof(ElemType) == sizeof(double))
        {                 
            cublasHandle_t cuHandle = GetCublasHandle(Gradients.GetComputeDeviceId());
            cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_DEVICE);
            CUBLAS_CALL(cublasDdot(cuHandle,m*n, reinterpret_cast <double*>(Gradients.m_pArray), 1, reinterpret_cast <double*>(SmoothedGradients.m_pArray), 1,reinterpret_cast <double*>(d_res)));
            cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_HOST);
        }
        else
        {            
            cublasHandle_t cuHandle = GetCublasHandle(Gradients.GetComputeDeviceId());
            cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_DEVICE);
            CUBLAS_CALL(cublasSdot(cuHandle,m*n, reinterpret_cast <float*>(Gradients.m_pArray), 1, reinterpret_cast <float*>(SmoothedGradients.m_pArray), 1,reinterpret_cast <float*>(d_res)));
            cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_HOST);
        }
        // d_res[0] should now contain inner product of matrices
        // Compute squared Frobenius norms (squared sums of elements)       
        _lrHelper<ElemType><<<1,512,0,t_stream>>>(Gradients.m_pArray,SmoothedGradients.m_pArray, (CUDA_LONG)Gradients.GetNumElements(), d_res);
        ElemType res;
        CUDA_CALL(cudaMemcpy(&res,d_res,sizeof(ElemType),cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_res));
        return res;
    }
    // The inputs are two row vectors [a1 a2 a3 a4] [b1 b2 b3 b4]
    // The outputs are one matrix of size (nt+1)*4
    // The first row is just element multiplication
    // The rest rows will be with shift
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementProductOfWithShiftNeg(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const size_t shift, const size_t nt)
    {
        if (a.IsEmpty() || b.IsEmpty())
            LogicError("AssignElementProductOf: Matrix is empty.");

        assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
        if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
            InvalidArgument("The input matrix dimensions do not match.");

        if (!(a.GetNumRows() == 1))
            InvalidArgument("The input matrix must be a row vector.");

        Resize(nt + 1, a.GetNumCols());
        int BS = a.GetNumCols();

        // the output matrix is of size (nt+1, BS)
        dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
        dim3 block_tail((nt + 1 + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (BS + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

        a.PrepareDevice();
        cudaEvent_t done = nullptr;;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _assignElementProductOfWithShiftNeg<ElemType> << < block_tail, thread_tail, 0, t_stream >> >(m_pArray, a.m_pArray, b.m_pArray, shift, nt + 1, BS);
        //		_assignElementProductOf<ElemType> << <block_tail, thread_tail, 0, t_stream >> >(m_pArray, a.m_pArray, b.m_pArray, nt);

        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
        return *this;
    }

	template<class ElemType>
	void GPUMatrix<ElemType>::InnerProductWithShiftNeg(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const size_t shift, const size_t nt)
	{
		if (a.GetComputeDeviceId() != b.GetComputeDeviceId() || b.GetComputeDeviceId() != c.GetComputeDeviceId()) //different GPUs
			InvalidArgument("All matrices must be on the same GPU");

		if (a.IsEmpty() || b.IsEmpty())
			LogicError("Scale:  one of the input matrices is empty.");

		const int m = (int)a.GetNumRows();
		const int n = (int)a.GetNumCols();
		const int k = (int)b.GetNumRows();
		const int l = (int)b.GetNumCols();

		assert(m>0 && n>0 && k>0 && l>0); //converting from size_t to int may cause overflow
		assert(m == k && n == l); //converting from size_t to int may cause overflow
		if (m != k || n != l)
			InvalidArgument("Matrices a and b should have same dimension.");

		c.Resize(nt + 1, n);

		if (true)
		{
            cudaEvent_t done = nullptr;;
			c.PrepareDevice();

			dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
			dim3 block_tail((nt + 1 + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (n + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);


			if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
			_innerProductWithShiftNeg<ElemType> << <block_tail, thread_tail, 0, t_stream >> >(c.m_pArray, a.m_pArray, b.m_pArray, m, n, shift, nt + 1);
			if (do_sync)    CUDA_CALL(cudaEventRecord(done));
			if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
			if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
		}
	}

	template<class ElemType>
	GPUMatrix<ElemType>& GPUMatrix<ElemType>::GetARowByIndex(const GPUMatrix<ElemType>& a, const size_t m)
	{
		if (a.IsEmpty())
			LogicError("GetARowByIndex: Matrix is empty.");

		Resize(1, a.GetNumCols());

		int n = a.GetNumRows();
		int P = a.GetNumCols();

		if (m >= n)
			LogicError("GetARowByIndex: m is out of range.");


		int blocksPerGrid = (int)ceil(((double)P) / threadsPerBlock);

		a.PrepareDevice();
        cudaEvent_t done = nullptr;;
		if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
		_getARowByIndex<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(m_pArray, a.m_pArray, n, P, m);
		//		_assignElementProductOf<ElemType> << <block_tail, thread_tail, 0, t_stream >> >(m_pArray, a.m_pArray, b.m_pArray, nt);

		if (do_sync)    CUDA_CALL(cudaEventRecord(done));
		if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
		if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
		return *this;
	}


	template<class ElemType>
	void GPUMatrix<ElemType>::ConductRowElementMultiplyWithShift(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const size_t shift, const bool isafixed)
	{
		if (a.GetComputeDeviceId() != b.GetComputeDeviceId() || b.GetComputeDeviceId() != c.GetComputeDeviceId()) //different GPUs
			InvalidArgument("All matrices must be on the same GPU");

		if (a.IsEmpty() || b.IsEmpty())
			LogicError("Scale:  one of the input matrices is empty.");

		const int m = (int)a.GetNumRows();
		const int n = (int)a.GetNumCols();
		const int O = (int)b.GetNumRows();
		const int P = (int)b.GetNumCols();

		assert(m>0 && n>0 && O>0 && P>0); //converting from size_t to int may cause overflow
		if (m != 1 || n != P)
			InvalidArgument("Matrices a and b should have same dimension.");

		c.Resize(O, P);

		if (true)
		{
            cudaEvent_t done = nullptr;;
			c.PrepareDevice();

			dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
			dim3 block_tail((O + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (P + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);


			if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
			_conductRowElementMultiplyWithShift<ElemType> << <block_tail, thread_tail, 0, t_stream >> >(c.m_pArray, a.m_pArray, b.m_pArray, O, P, shift, isafixed);
			if (do_sync)    CUDA_CALL(cudaEventRecord(done));
			if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
			if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
		}
	}



	template<class ElemType>
	GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementProductOfWithShift(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const size_t shift)
	{
		if (a.IsEmpty() || b.IsEmpty())
			LogicError("AssignElementProductOfWithShift: Matrix is empty.");

		assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
		if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
			InvalidArgument("The input matrix dimensions do not match.");

		//int O = a.GetNumRows();
		int P = a.GetNumCols();

		Resize(1, P);
		CUDA_LONG N = (CUDA_LONG)GetNumElements();
		int blocksPerGrid = (int)ceil(((double)N) / threadsPerBlock);
		a.PrepareDevice();
        cudaEvent_t done = nullptr;;
		if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
		_assignElementProductOfWithShift<ElemType> << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(m_pArray, a.m_pArray, b.m_pArray, shift, N);
		if (do_sync)    CUDA_CALL(cudaEventRecord(done));
		if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
		if (do_sync)    CUDA_CALL(cudaEventDestroy(done));
		return *this;
	}

    //sequence training
    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::DropFrame(const GPUMatrix<ElemType>& label, const GPUMatrix<ElemType>& gamma, const ElemType & threshhold)
    {
        if (IsEmpty())
            LogicError("DropFrame: Matrix is empty.");

        PrepareDevice();

        long N = (long)GetNumCols(); //one kernel per column
        int blocksPerGrid = (int)ceil(N*1.0 / threadsPerBlock);
        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        _DropFrame << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(m_pArray, label.m_pArray, gamma.m_pArray, threshhold, (long)m_numCols, (long)m_numRows);

        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

    template<class ElemType>
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSequenceError(const ElemType hsmoothingWeight, const GPUMatrix<ElemType>& label,
        const GPUMatrix<ElemType>& dnnoutput, const GPUMatrix<ElemType>& gamma, ElemType alpha)
    {
        if (IsEmpty())
            LogicError("AssignSequenceError: Matrix is empty.");

        PrepareDevice();

        cudaEvent_t done = nullptr;
        if (do_sync)    CUDA_CALL(cudaEventCreate(&done));
        long N = (LONG64)label.GetNumElements();
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        _AssignSequenceError << <blocksPerGrid, threadsPerBlock, 0, t_stream >> >(hsmoothingWeight, m_pArray, label.m_pArray, dnnoutput.m_pArray, gamma.m_pArray, alpha, N);

        if (do_sync)    CUDA_CALL(cudaEventRecord(done));
        if (do_sync)    CUDA_CALL(cudaEventSynchronize(done));
        if (do_sync)    CUDA_CALL(cudaEventDestroy(done));

        return *this;
    }

#pragma endregion Static BLAS Functions


    /// f = logadd(f, vec) to get the logadd sum of vector elments
    template<class ElemType>
    ElemType GPUMatrix<ElemType>::LogAddSumOfElements() const
    {
        if (this->IsEmpty())
            LogicError("SumOfElements: Matrix is empty");

        PrepareDevice();
        ElemType* d_sum = NULL;
        ElemType h_sum;
        CUDA_LONG N = (CUDA_LONG)GetNumElements();
        CUDA_CALL(cudaMalloc((void**)&d_sum, sizeof(ElemType)));
        int blocksPerGrid = (int)ceil(((double)N) / threadsPerBlock);

        _reductionLogAddSum<ElemType> << < blocksPerGrid, threadsPerBlock >> > (this->m_pArray,
            d_sum, 1, N);
        CUDA_CALL(cudaMemcpy(&h_sum, d_sum, sizeof(ElemType), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_sum));

        return h_sum;
    }

    template<class ElemType>
    void GPUMatrix<ElemType>::RCRFBackwardCompute(
        const GPUMatrix<ElemType>& alpha, GPUMatrix<ElemType>& beta,
        const GPUMatrix<ElemType>& /*lbls*/,
        const GPUMatrix<ElemType>& pos_scores, const GPUMatrix<ElemType>& pair_scores, const int shift)
    {
        if (alpha.IsEmpty() || pos_scores.IsEmpty() || pair_scores.IsEmpty())
            LogicError("RCRFBackwardCompute: one of the input matrices is empty.");

        if (alpha.GetNumRows() != pos_scores.GetNumRows() || alpha.GetNumCols() != pos_scores.GetNumCols())
            LogicError("RCRFBackwardCompute: matrix dimensions mismatched.");

        size_t iNumLab = alpha.GetNumRows();
        size_t iNumPos = alpha.GetNumCols();

        alpha.PrepareDevice();
        beta.Resize(iNumLab, iNumPos);

        ElemType* d_zeta = NULL;
        CUDA_CALL(cudaMalloc((void**)&d_zeta, sizeof(ElemType)* iNumLab)); //we allocate memory on the device

        CUDA_LONG N = iNumLab;
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        size_t szMemSize;
        for (int t = iNumPos - 1; t >= 0; t--)
        {
            szMemSize = sizeof(ElemType)* iNumLab;
            _rcrfBackwardComputeZeta<ElemType> << <blocksPerGrid, threadsPerBlock, szMemSize >> >(t, iNumPos, alpha.m_pArray, d_zeta, pair_scores.m_pArray, iNumLab, shift);
            szMemSize = iNumLab * 3;
            szMemSize *= sizeof(ElemType);
            _rcrfBackwardCompute<ElemType> << <blocksPerGrid, threadsPerBlock, szMemSize >> >(t, iNumPos, alpha.m_pArray, beta.m_pArray,
                d_zeta, pair_scores.m_pArray, iNumLab, shift);
        }
        /*
        error = cudaGetErrorString(cudaPeekAtLastError());
        printf("%s\n", error);
        error = cudaGetErrorString(cudaThreadSynchronize());
        printf("%s\n", error);
        */

        CUDA_CALL(cudaFree(d_zeta));
    }

    /**
    Compute the gradient for the first order Markov transition probabilities
    It uses equations derived in R. Collobert's paper "Natural lanugage processing (almost) from scratch"
    */
    template<class ElemType>
    void GPUMatrix<ElemType>::RCRFTransGrdCompute(const GPUMatrix<ElemType>& lbls,
        const GPUMatrix<ElemType>&   alpha,
        const GPUMatrix<ElemType>& beta,
        const GPUMatrix<ElemType>& pair_scores,
        GPUMatrix<ElemType>& grd,
        const int startLbl,
        const int shift)
    {
        assert(shift == 1);
        int iNumPos = alpha.GetNumCols();
        int iNumLab = alpha.GetNumRows();

        ElemType* d_zeta = NULL;
        CUDA_CALL(cudaMalloc((void**)&d_zeta, sizeof(ElemType)* iNumLab)); //we allocate memory on the device
        CUDA_LONG N = iNumLab;
        int blocksPerGrid = (int)ceil(1.0*N / threadsPerBlock);
        size_t szMemSize;
        for (int t = 0; t < iNumPos; t++)
        {
            szMemSize = sizeof(ElemType)* iNumLab;
            _rcrfTransGrdComputeZeta<ElemType> << <blocksPerGrid, threadsPerBlock, szMemSize >> >(t - 1, iNumPos, alpha.m_pArray, d_zeta, pair_scores.m_pArray, iNumLab, startLbl, shift);
            szMemSize = iNumLab * 3;
            szMemSize *= sizeof(ElemType);
            _rcrfTransGrdCompute<ElemType> << <blocksPerGrid, threadsPerBlock, szMemSize >> >(t, startLbl, alpha.m_pArray, beta.m_pArray,
                d_zeta, pair_scores.m_pArray, lbls.m_pArray, grd.m_pArray, iNumPos, iNumLab, shift);
        }
        CUDA_CALL(cudaFree(d_zeta));
    };

    template class GPUMatrix<float>;
    template class GPUMatrix<double>;
    template class DeviceBoundNumber<float>;
    template class DeviceBoundNumber<double>;

    template<class ElemType>
    cublasHandle_t GPUMatrix<ElemType>::s_cuHandle[GPUMatrix<ElemType>::MaxGpus]={0};

    template<class ElemType>
    void* GPUMatrix<ElemType>::s_curandGenerator=NULL;    

    // We use Matrix<char> as the backing store for QuantizedMatrix
    // Let's explicitly instantiate the methods we need for that purpose
    template GPUMatrix<char>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId);
    template GPUMatrix<char>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId, char *pArray, const size_t matrixFlags);
    template GPUMatrix<char>::GPUMatrix(const GPUMatrix<char>&);
    template GPUMatrix<char>::GPUMatrix(GPUMatrix<char>&&);
    template char* GPUMatrix<char>::CopyToArray() const;
    template void GPUMatrix<char>::ChangeDeviceTo(int);
    template void GPUMatrix<char>::Resize(size_t, size_t, bool);

    template GPUMatrix<char>::~GPUMatrix();
    template DEVICEID_TYPE GPUMatrix<char>::GetBestGPUDeviceId();
    template GPUMatrix<char> GPUMatrix<char>::ColumnSlice(size_t startColumn, size_t numCols) const;
    template GPUMatrix<char>& GPUMatrix<char>::operator=(GPUMatrix<char>&&);
    template GPUMatrix<char>::GPUMatrix(int);
    template void GPUMatrix<char>::SetValue(const char);
    template void GPUMatrix<char>::SetValue(const size_t numRows, const size_t numCols, int deviceId, char *pArray, size_t matrixFlags);
    
}}}

// !!!!This is from helper_cuda.h which comes with CUDA samples!!!! Consider if it is beneficial to just include all helper_cuda.h
// TODO: This is duplicated in BestGpu.cpp
// Beginning of GPU Architecture definitions
int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }
    return nGpuArchCoresPerSM[7].Cores;
};
// end of GPU Architecture definitions

//inline CUDA_LONG _GetFreeMemoryOnCUDADevice(int devId)
//{   
//    CUdevice cudaDevice;  
//    CUresult result = cuDeviceGet(&cudaDevice, devId);  
//    if(result!= CUDA_SUCCESS)  
//    {          
//        return 0;         
//    }  
//  
//    //create cuda context  
//    CUcontext cudaContext;    
//    result = cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice);  
//    if(result != CUDA_SUCCESS)  
//    {          
//        return 0;         
//    }  
//  
//    //get the amount of free memory on the graphics card  
//    size_t free;  
//    size_t total;  
//    result = cuMemGetInfo(&free, &total);  
//    if (result!=CUDA_SUCCESS)
//    {
//        return 0;
//    }
//    else
//        return (CUDA_LONG)free;
//}


#endif // CPUONLY
