//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "Basics.h"
#include "MPIWrapper.h"
#include "CNTKLibrary.h"
#include "DistributedCommunicator.h"
#include "CUDAPageLockedMemAllocator.h"
#include "MatrixQuantizerImpl.h"
#include <functional>
#include <mutex>

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    static std::once_flag s_initMPICommunicatorFlag;
    static DistributedCommunicatorPtr s_MPICommunicator;

    DistributedCommunicatorPtr MPICommunicator()
    {
        std::call_once(s_initMPICommunicatorFlag, [=]{
            s_MPICommunicator.reset(new MPICommunicatorImpl());
        });
        return s_MPICommunicator;
    }

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
        m_currentWorker.m_globalRank = m_mpi->CurrentNodeRank();
        m_currentWorker.m_hostId = std::wstring(m_mpi->CurrentNodeName());
        for (auto i = 0; i < m_mpi->NumNodesInUse(); ++i)
        {
            if (i == m_currentWorker.m_globalRank)
            {
                m_workers.insert(m_currentWorker);
            }
            else
            {
                m_workers.insert({ i, L"" });
            }
        }
    }


    /*virtual*/ std::unordered_set<DistributedWorkerDescriptor> MPICommunicatorImpl::Workers() const
    {
        return m_workers;
    }

    /*virtual*/ const DistributedWorkerDescriptor& MPICommunicatorImpl::CurrentWorker() const
    {
        return m_currentWorker;
    }

    /*virtual*/ std::vector<ValuePtr> MPICommunicatorImpl::Aggregate(const std::vector<ValuePtr>& values,
                                                                     const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        std::vector<ValuePtr> outputValues;
        for (const auto& inputValue : values)
        {
            const auto inputView = inputValue->Data();
            const auto& outputView = MakeSharedObject<NDArrayView>(inputView->GetDataType(), inputView->Shape(), inputView->Device());
            const auto& inputMask = inputValue->Mask();
            const auto& outputMask = MakeSharedObject<NDMask>(inputMask->Shape(), inputMask->Device());
            outputValues.push_back(MakeSharedObject<Value>(outputView, outputMask)); 
        }
        AggregateImpl(values, outputValues, sendToWorkers);
       
        return outputValues;
    }

    /*virtual*/ DistributedCommunicatorPtr MPICommunicatorImpl::SubGroup(const std::unordered_set<DistributedWorkerDescriptor>& subGroupWorkers) const 
    {
        UNUSED(subGroupWorkers);
        NOT_IMPLEMENTED;
    }

    /*virtual*/ std::unordered_set<ValuePtr> MPICommunicatorImpl::Concatenate(const std::unordered_set<ValuePtr>& values, const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        UNUSED(values);
        UNUSED(sendToWorkers);
        NOT_IMPLEMENTED;
    }
    
    /*virtual*/ std::future<std::vector<ValuePtr>> MPICommunicatorImpl::AggregateAsync(const std::vector<ValuePtr>& values,
                                                                                       const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        return std::async(std::launch::async, [this, &values, &sendToWorkers]() { 

            for (const auto& value : values)
            {
                const auto& device = value->Data()->Device();
                if (device.Type() == DeviceKind::GPU)
                {
                    // We are starting on a new thread. Make sure the new thread is
                    // setup to use the right device
                    // TODO: SetDevice is type agnostic, move it to the base matrix class. 
                    Matrix<float>::SetDevice(device.Id());
                    // For the time being, we assume that all gpu-bound values reside on the same device.
                    break;
                }
            }

            return this->Aggregate(values, sendToWorkers);
        });
    }

    std::vector<int> MPICommunicatorImpl::Prepare(const std::vector<ValuePtr>& values)
    {
        assert(CPUDEVICE < 0); // just in case somebody decides to change CPUDEVICE macro.
        std::vector<int> indices(values.size(), CPUDEVICE);
        int numGPUValues = 0;
        for (auto i = 0; i < values.size(); ++i)
        {
            auto& value = values[i];
            auto view = value->Data();
            auto device= view->Device();

            // Make sure none of the values are sparse - we currently do not support aggregation of sparse matrices
            if (view->GetStorageFormat() != StorageFormat::Dense)
            {
                RuntimeError("Aggregation for sparse matrices is currently not supported!");
            }
            
            DeviceDescriptor lastGpuDevice = DeviceDescriptor::CPUDevice();

            if (device.Type() == DeviceKind::GPU)
            {
                if (lastGpuDevice.Type() == DeviceKind::CPU)
                {
                    lastGpuDevice = device;
                } 
                else if (device.Id() != lastGpuDevice.Id())
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
        AggregateImpl(values, values, sendToWorkers);
    }

    void  MPICommunicatorImpl::AggregateImpl(const std::vector<ValuePtr>& inputValues,
                                             const std::vector<ValuePtr>& outputValues,
                                             const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        UNUSED(sendToWorkers);
        assert(inputValues.size() == outputValues.size());

        auto numValues = inputValues.size();
        if (numValues == 0)
        {
            return;
        }
        
        const auto& gpuIndexVector = Prepare(inputValues);

        if (gpuIndexVector.size() > 0)
        {
            // Before initiating gpu-to-cpu transfer, make sure that all the computations on the main GPU
            // stream are finished.
            const auto& gpuValue = inputValues[gpuIndexVector[0]];
            const auto& device = gpuValue->Data()->Device();
            std::unique_ptr<MatrixComputeStreamEvent> mainStreamSyncEvent(MatrixComputeStreamEvent::Create(device.Id()));
            // TODO: the method below is actually type-agnostic, it needs refactoring.
            mainStreamSyncEvent->SynchronizeDataTransferFetchStreamWithEvent<float>();
        }

        // for all values residing on GPU initiate async transfer to CPU buffers.
        for (auto i = 0; i < numValues; ++i)
        {
            auto& value = inputValues[i];
            auto view = value->Data();
            auto gpuIndex = gpuIndexVector[i];
            if (gpuIndex != CPUDEVICE)
            {
                auto& transferer = m_gpuDataTransferers[gpuIndex];
                auto& buffer = m_intermediateCPUBuffers[gpuIndex];
                transferer->CopyGPUToCPUAsync(GetDataBuffer(view), GetBufferSize(view), buffer.data.get());
            }
        }
        
        std::vector<MPI_Request> allReduceRequests(numValues);
        for (auto i = 0; i < numValues; ++i)
        {
            auto gpuIndex = gpuIndexVector[i];
            if (gpuIndex != CPUDEVICE)
            {
                // TODO: actually, we can start reducing all cpu values first, and then wait for the gpu->cpu transfer to finish.
                m_gpuDataTransferers[gpuIndex]->WaitForCopyGPUToCPUAsync();
            }

            auto& inputValue = inputValues[i];
            auto inputView = inputValue->Data();
            auto numElements = inputView->Shape().TotalSize();
            auto dataType = inputView->GetDataType();

            auto& outputValue = outputValues[i];
            auto outputView = outputValue->Data();

            assert(numElements == outputView->Shape().TotalSize());
            assert(dataType == outputView->GetDataType());
            assert(inputView->Device() == outputView->Device());

            void* inputData = (gpuIndex != CPUDEVICE) ? m_intermediateCPUBuffers[gpuIndex].data.get() : GetDataBuffer(inputView);
            void* outputData = (gpuIndex != CPUDEVICE) ? m_intermediateCPUBuffers[gpuIndex].data.get() : GetDataBuffer(outputView);


            if (dataType == DataType::Float)
            {          
                m_mpi->AllReduceAsync<float>(static_cast<float*>(inputData), static_cast<float*>(outputData), numElements, &allReduceRequests[i]);
            }
            else if (dataType == DataType::Double)
            {
                m_mpi->AllReduceAsync<double>(static_cast<double*>(inputData), static_cast<double*>(outputData), numElements, &allReduceRequests[i]);
            }
            else
                LogicError("Unknown DataType");
        }

        // wait for async all reduce to complete. As soon as one of the requests is finished,
        // check if corresponding value is gpu bound and, if it is the case, initiate a cpu-to-gpu transfer.
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

            if (gpuIndex != CPUDEVICE)
            {
                auto view = outputValues[idx]->Data();
                auto size = GetBufferSize(view);
                auto& transferer = m_gpuDataTransferers[gpuIndex];
                auto& buffer = m_intermediateCPUBuffers[gpuIndex];
                transferer->CopyCPUToGPUAsync(buffer.data.get(), size, GetDataBuffer(view));
            }
        }

        for (auto gpuIndex : gpuIndexVector)
        {
            m_gpuDataTransferers[gpuIndex]->WaitForCopyCPUToGPUAsync();
        }
    }

    /*virtual*/ void MPICommunicatorImpl::QuantizedAggregate(const std::vector<ValuePtr>& inValues,
                                        const std::unordered_set<ValuePtr>& inPreviousQuantizationResidues,
                                        const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers,
                                        const std::unordered_set<ValuePtr>& aggregatedOutputs,
                                        const std::unordered_set<ValuePtr>& newQuantizationResidues)
    {
        UNUSED(inValues);
        UNUSED(inPreviousQuantizationResidues);
        UNUSED(sendToWorkers);
        UNUSED(aggregatedOutputs);
        UNUSED(newQuantizationResidues);
        NOT_IMPLEMENTED;
    }
}