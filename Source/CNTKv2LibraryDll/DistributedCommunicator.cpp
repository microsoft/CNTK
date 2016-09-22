//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "DistributedCommunicator.h"
#include <functional>


using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    MPICommunicatorImpl::Buffer MPICommunicatorImpl::AllocateIntermediateBuffer(int deviceID, size_t totalSize)
    {
        assert(deviceID >= 0);
        Buffer buffer;
        // Use pinned memory for GPU devices for better copy performance
        buffer.totalSize = totalSize;
        buffer.data = std::shared_ptr<void>(CUDAPageLockedMemAllocator::Malloc(totalSize, deviceID), 
                                            [deviceID](void* p)
                                            {
                                                CUDAPageLockedMemAllocator::Free(p, deviceID);
                                            });
        return buffer;
    }

    inline size_t GetBufferSize(const NDArrayViewPtr& viewPtr)
    {
        return viewPtr->Shape().TotalSize() * DataTypeSize(viewPtr->GetDataType());
    }

    inline void* GetDataBuffer(const NDArrayViewPtr& viewPtr)
    {
       if (viewPtr->GetDataType() == DataType::Float)
        {
            return (void*) viewPtr->DataBuffer<float>();
        }
        else if (viewPtr->GetDataType() == DataType::Double)
        {
            return (void*) viewPtr->DataBuffer<double>();
        }
        else
            LogicError("Unknown DataType");
    }

    MPICommunicatorImpl::MPICommunicatorImpl()
    {
        m_mpi = MPIWrapper::GetInstance(true);
    }

    /*virtual*/ std::vector<ValuePtr> MPICommunicatorImpl::Aggregate(const std::vector<ValuePtr>& values,
                                                                     const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        std::vector<ValuePtr> outputValues(values.size());
        for (const auto& inputValue : values)
        {
            outputValues.push_back(inputValue->DeepClone()); // move to cpu -- do async call  here + .then(xxx());
        }
        AggregateInPlace(outputValues, sendToWorkers);
    }

    std::vector<size_t> MPICommunicatorImpl::Prepare(const std::vector<ValuePtr>& values)
    {
        std::vector<size_t> indices(values.size(), -1);
        int numGPUValues = 0;
        for (auto i = 0; i < values.size(); ++i)
        {
            auto& value = values[i];
            auto& view = value->Data();
            auto device= view->Device();
            if (device.Type() == DeviceKind::GPU)
            {
                if (device.Id() != values.front()->Data()->Device().Id())
                {
                    // For the time being, assume all devices have the same id.
                    LogicError("Not all values share the same GPU device id"); 
                }

                auto index = numGPUValues++;
                if (m_gpuDataTransferers.size() < numGPUValues)
                {
                     m_gpuDataTransferers.push_back(std::unique_ptr<GPUDataTransferer>(new GPUDataTransferer(device.Id(), true)));
                }

                if (m_intermediateCPUBuffers.size() < numGPUValues)
                {
                    m_intermediateCPUBuffers.push_back(Buffer());
                }

                auto requiredSize = GetBufferSize(view);
                if (m_intermediateCPUBuffers[index].totalSize < requiredSize)
                {
                    m_intermediateCPUBuffers[index] = AllocateIntermediateBuffer(device.Id(), requiredSize);
                }

                indices[i] = index;
            }
        }

        return indices;
    }

    /*virtual*/ void MPICommunicatorImpl::AggregateInPlace(const std::vector<ValuePtr>& values,
                                                           const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {


        // if a value is on a gpu -> copy to cpu -> aggregate -> copy back.
        // if value is on a gpu, copy it to cpu
        auto numValues = values.size();
        if (numValues == 0)
        {
            return;
        }
        
        // launch async thread here, acquire a mutex.
        auto gpuIndexVector = Prepare(values);
        
        for (auto i = 0; i < numValues; ++i)
        {
            auto gpuIndex = gpuIndexVector[i];

            m_gpuDataTransferers[gpuIndex]->CopyGPUToCPUAsync(GetDataBuffer(view), requiredSize, m_intermediateCPUBuffers[i].data.get());
        }
        
        
       
        std::vector<MPI_Request> allReduceRequests(numValues);
        // this has to be done by a single thread. 
        // use a shared_future here to wait on?
        for (auto i = 0; i < numValues; ++i)
        {
            auto gpuIndex = gpuIndexVector[i];
           
            if (gpuIndex >= 0)
            {
                // TODO: actually, we can start reducing all cpu values first, and then wait for the gpu->cpu transfer to finish.
                m_gpuDataTransferers[gpuIndex]->WaitForCopyGPUToCPUAsync();
            }

            auto& value = values[i];
            auto& view = value->Data();
            auto numElements = view->Shape().TotalSize();
            auto dataType = view->GetDataType();

            if (dataType == DataType::Float)
            {
                float* data = view->WritableDataBuffer<float>();
                m_mpi->AllReduceAsync<float>(data, numElements, &allReduceRequests[i]);
            }
            else if (dataType == DataType::Double)
            {
                double* data = view->WritableDataBuffer<double>();
                m_mpi->AllReduceAsync<double>(data, numElements, &allReduceRequests[i]);
            }
            else
                LogicError("Unknown DataType");
        }

        size_t numAllReduceRequestsCompleted = 0;
        while (numAllReduceRequestsCompleted < numValues)
        {
            int idx = MPI_UNDEFINED;
            m_mpi->WaitAny(allReduceRequests.data(), (int)allReduceRequests.size(), &idx);
            if (idx == MPI_UNDEFINED)
            {
                break;
            }

            numAllReduceRequestsCompleted++;
            
             auto gpuIndex = gpuIndexVector[idx];

            if (gpuIndex >= 0)
            {
                auto& view = values[idx]->Data();
                auto size = GetBufferSize(view);
                m_gpuDataTransferers[gpuIndex]->CopyCPUToGPUAsync(m_intermediateCPUBuffers[gpuIndex].data.get(), GetRequi, GetWritableDataBuffer(view));
            }
        }

        // Wait for the allreduce operations to finish and initiate transfer back to the GPU if needed
    }



    /*virtual*/ std::future<void> MPICommunicatorImpl::AggregateInPlaceAsync(const std::vector<ValuePtr>& values,
                                                                             const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        return std::async(std::launch::async, [this, &values, &sendToWorkers](){ /*before starting a new aggregation, make sure the old one is complete*/ this->AggregateInPlace(values, sendToWorkers); });
    }
}