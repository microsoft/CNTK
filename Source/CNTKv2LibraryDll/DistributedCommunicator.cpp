//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <functional>
#include "Basics.h"
#include "MPIWrapper.h"
#include "CNTKLibrary.h"
#include "DistributedCommunicator.h"
#include "CUDAPageLockedMemAllocator.h"
#include "MatrixQuantizerImpl.h"
#include "GPUDataTransferer.h"

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    DistributedCommunicatorPtr MPICommunicator()
    {
        return std::make_shared<MPICommunicatorImpl>();
    }

    MPICommunicatorImpl::Buffer MPICommunicatorImpl::AllocateIntermediateBuffer(int deviceID, size_t totalSize)
    {
        assert(deviceID >= 0);
        Buffer buffer;
        buffer.totalSize = totalSize;
        buffer.data = std::shared_ptr<void>(
            CUDAPageLockedMemAllocator::Malloc(totalSize, deviceID),
            [deviceID](void* p) { CUDAPageLockedMemAllocator::Free(p, deviceID); });
        return buffer;
    }

    inline size_t GetBufferSize(const NDArrayViewPtr& viewPtr)
    {
        return viewPtr->Shape().TotalSize() * DataTypeSize(viewPtr->GetDataType());
    }

    inline void* GetDataBuffer(const NDArrayViewPtr& viewPtr)
    {
        if (viewPtr->GetDataType() == DataType::Float)
            return viewPtr->WritableDataBuffer<float>();
        if (viewPtr->GetDataType() == DataType::Double)
            return viewPtr->WritableDataBuffer<double>();
        LogicError("Unknown DataType");
        return nullptr; // Make compiler happy.
    }

    inline DeviceDescriptor GetNonCPUDevice(const std::vector<NDArrayViewPtr>& values)
    {
        auto device = std::find_if(values.begin(), values.end(), [](const NDArrayViewPtr v) { return v ->Device().Type() != DeviceKind::CPU; });
        return values.end() == device ? DeviceDescriptor::CPUDevice() : (*device)->Device();
    }

    MPICommunicatorImpl::MPICommunicatorImpl()
    {
        m_mpi = MPIWrapper::s_initialized ? MPIWrapper::GetInstance() : std::make_shared<MPIWrapper>();;
        m_currentWorker.m_globalRank = m_mpi->CurrentNodeRank();
        m_currentWorker.m_hostId = std::wstring(m_mpi->CurrentNodeName());
        for (size_t i = 0; i < m_mpi->NumNodesInUse(); ++i)
        {
            if (i == m_currentWorker.m_globalRank)
                m_workers.insert(m_currentWorker);
            else
                // TOOD: Nodes have to exchange their names.
                m_workers.insert({ i,  L"" });
        }
    }

    void MPICommunicatorImpl::Initialize(const std::vector<NDArrayViewPtr>& values)
    {
        assert(CPUDEVICE < 0); // just in case somebody decides to change CPUDEVICE macro.
        DeviceDescriptor lastGpuDevice = DeviceDescriptor::CPUDevice();
        m_gpuDataTransferers.resize(values.size());
        m_intermediateCPUBuffers.resize(values.size());
        for (auto i = 0; i < values.size(); ++i)
        {
            auto view = values[i];
            auto device = view->Device();

            // Make sure none of the values are sparse - we currently do not support aggregation of sparse matrices
            if (view->GetStorageFormat() != StorageFormat::Dense)
                RuntimeError("Aggregation for sparse matrices is currently not supported!");

            // TODO: device.Type should be called Kind.
            if (device.Type() != DeviceKind::GPU)
            {
                m_intermediateCPUBuffers[i] = Buffer();
                m_gpuDataTransferers[i] = nullptr;
            }
            else
            {
                if (lastGpuDevice.Type() == DeviceKind::CPU)
                    lastGpuDevice = device;
                else if (device.Id() != lastGpuDevice.Id()) // For the time being, assume all devices have the same id.
                    LogicError("Not all values are on the same GPU device id");

                auto requiredSize = GetBufferSize(view);
                m_gpuDataTransferers[i] = std::make_shared<GPUDataTransferer>(device.Id(), true);
                if (m_intermediateCPUBuffers[i].totalSize < requiredSize)
                    m_intermediateCPUBuffers[i] = AllocateIntermediateBuffer(device.Id(), requiredSize);
            }
        }
    }

    const std::unordered_set<DistributedWorkerDescriptor>& MPICommunicatorImpl::Workers() const
    {
        return m_workers;
    }

    const DistributedWorkerDescriptor& MPICommunicatorImpl::CurrentWorker() const
    {
        return m_currentWorker;
    }

    void MPICommunicatorImpl::CheckWorkers(const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        // Currently all operations should be executed on all workers, we do not support subgroups.
        if (sendToWorkers != m_workers)
            NOT_IMPLEMENTED;
    }

    void MPICommunicatorImpl::Aggregate(const std::vector<NDArrayViewPtr>& values,
        std::vector<NDArrayViewPtr>& outputValues,
        const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        if (outputValues.empty())
        {
            for (const auto& inputValue : values)
            {
                const auto& outputView = MakeSharedObject<NDArrayView>(inputValue->GetDataType(), inputValue->Shape(), inputValue->Device());
                outputValues.push_back(outputView);
            }
        }
        else if (outputValues.size() != values.size())
        {
            NOT_IMPLEMENTED;
        }

        auto device = GetNonCPUDevice(values);
        if (device.Type() != DeviceKind::CPU)
        {
            // Since we will be copying the gradients asynchronously, let us
            // ensure that the gradient matrices have been computed before starting to aggregate
            // them asynchronously on another thread. This essentially means that when we are using
            // a GPU device, we will synchronize on the main GPU compute stream before starting
            // the gradient aggregation asynchronously on a separate stream
            std::unique_ptr<MatrixComputeStreamEvent> mainStreamSyncEvent(MatrixComputeStreamEvent::Create(device.Id()));
            mainStreamSyncEvent->SynchronizeDataTransferFetchStreamWithEvent<float>();
        }
        AggregateImpl(values, outputValues, sendToWorkers);
    }

    DistributedCommunicatorPtr MPICommunicatorImpl::SubGroup(const std::unordered_set<DistributedWorkerDescriptor>&) const
    {
        NOT_IMPLEMENTED;
    }

    void MPICommunicatorImpl::Concatenate(const std::vector<ValuePtr>&, std::vector<ValuePtr>&, const std::unordered_set<DistributedWorkerDescriptor>&)
    {
        NOT_IMPLEMENTED;
    }

    std::future<std::vector<NDArrayViewPtr>> MPICommunicatorImpl::AggregateAsync(
        const std::vector<NDArrayViewPtr>& values,
        const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        auto device = GetNonCPUDevice(values);

        std::shared_ptr<MatrixComputeStreamEvent> mainStreamSyncEvent;
        if (device.Type() != DeviceKind::CPU)
            mainStreamSyncEvent.reset(MatrixComputeStreamEvent::Create(device.Id()));

        return std::async(std::launch::async, [this, &values, &sendToWorkers, device, mainStreamSyncEvent]()
        {
            if (device.Type() != DeviceKind::CPU)
            {
                // We are starting on a new thread. Make sure the new thread is setup to use the right device
                // TODO: SetDevice is type agnostic, move it to the base matrix class. 
                Matrix<float>::SetDevice(device.Id());

                // Since we will be copying the gradients asynchronously, let us
                // ensure that the gradient matrices have been computed before starting to aggregate
                // them asynchronously on another thread. This essentially means that when we are using
                // a GPU device, we will synchronize on the main GPU compute stream before starting
                // the gradient aggregation asynchronously on a separate stream
                mainStreamSyncEvent->SynchronizeDataTransferFetchStreamWithEvent<float>();
            }

            std::vector<NDArrayViewPtr> output;
            Aggregate(values, output, sendToWorkers);
            return output;
        });
    }

    void MPICommunicatorImpl::AggregateInPlace(
        const std::vector<NDArrayViewPtr>& values,
        const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        auto device = GetNonCPUDevice(values);
        if (device.Type() != DeviceKind::CPU)
        {
            // Since we will be copying the gradients asynchronously, let us
            // ensure that the gradient matrices have been computed before starting to aggregate
            // them asynchronously on another thread. This essentially means that when we are using
            // a GPU device, we will synchronize on the main GPU compute stream before starting
            // the gradient aggregation asynchronously on a separate stream
            std::unique_ptr<MatrixComputeStreamEvent> mainStreamSyncEvent(MatrixComputeStreamEvent::Create(device.Id()));
            mainStreamSyncEvent->SynchronizeDataTransferFetchStreamWithEvent<float>();
        }
        AggregateImpl(values, values, sendToWorkers);
    }

    void  MPICommunicatorImpl::AggregateImpl(
        const std::vector<NDArrayViewPtr>& inputValues,
        const std::vector<NDArrayViewPtr>& outputValues,
        const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        CheckWorkers(sendToWorkers);

        if (m_mpi->NumNodesInUse() == 1) // No need to aggregate anything.
            return;

        assert(inputValues.size() == outputValues.size());

        auto numValues = inputValues.size();
        if (numValues == 0)
            return;

        Initialize(inputValues);

        // for all values residing on GPU initiate async transfer to CPU buffers.
        for (auto i = 0; i < numValues; ++i)
        {
            auto view = inputValues[i];
            if (view->Device() != DeviceDescriptor::CPUDevice())
            {
                auto& transferer = m_gpuDataTransferers[i];
                auto& buffer = m_intermediateCPUBuffers[i];
                transferer->CopyGPUToCPUAsync(GetDataBuffer(view), GetBufferSize(view), buffer.data.get());
            }
        }

        std::vector<MPI_Request> allReduceRequests(numValues);
        for (auto i = 0; i < numValues; ++i)
        {
            auto inputValue = inputValues[i];

            if (inputValue->Device() != DeviceDescriptor::CPUDevice())
            {
                // TODO: actually, we can start reducing all cpu values first, and then wait for the gpu->cpu transfer to finish.
                m_gpuDataTransferers[i]->WaitForCopyGPUToCPUAsync();
            }

            auto numElements = inputValue->Shape().TotalSize();
            auto dataType = inputValue->GetDataType();

            auto& outputValue = outputValues[i];

            assert(numElements == outputValue->Shape().TotalSize());
            assert(dataType == outputValue->GetDataType());
            assert(inputValue->Device() == outputValue->Device());

            void* inputData = (inputValue->Device() != DeviceDescriptor::CPUDevice()) ? m_intermediateCPUBuffers[i].data.get() : GetDataBuffer(inputValue);
            void* outputData = (inputValue->Device() != DeviceDescriptor::CPUDevice()) ? m_intermediateCPUBuffers[i].data.get() : GetDataBuffer(outputValue);

            if (dataType == DataType::Float)
            {
                if (inputData == outputData)
                    m_mpi->AllReduceAsync<float>(static_cast<float*>(outputData), numElements, &allReduceRequests[i]);
                else
                    m_mpi->AllReduceAsync<float>(static_cast<float*>(inputData), static_cast<float*>(outputData), numElements, &allReduceRequests[i]);
            }
            else if (dataType == DataType::Double)
            {
                if (inputData == outputData)
                    m_mpi->AllReduceAsync<double>(static_cast<double*>(outputData), numElements, &allReduceRequests[i]);
                else
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

            assert(idx < inputValues.size());
            auto value = inputValues[idx];

            if (value->Device() != DeviceDescriptor::CPUDevice())
            {
                auto view = outputValues[idx];
                auto size = GetBufferSize(view);
                auto& transferer = m_gpuDataTransferers[idx];
                auto& buffer = m_intermediateCPUBuffers[idx];
                transferer->CopyCPUToGPUAsync(buffer.data.get(), size, GetDataBuffer(view));
            }
        }

        // TODO: Should not wait, simply publishing event on the compute stream should be sufficient.
        for (auto i = 0; i < numValues; ++i)
        {
            if (inputValues[i]->Device() != DeviceDescriptor::CPUDevice())
                m_gpuDataTransferers[i]->WaitForCopyCPUToGPUAsync();
        }
    }

    void MPICommunicatorImpl::QuantizedAggregate(const std::vector<NDArrayViewPtr>& /*inValues*/,
        const std::vector<NDArrayViewPtr>& /*inPreviousQuantizationResidues*/,
        const std::unordered_set<DistributedWorkerDescriptor>& /*sendToWorkers*/,
        const std::vector<NDArrayViewPtr>& /*aggregatedOutputs*/,
        const std::vector<NDArrayViewPtr>& /*newQuantizationResidues*/)
    {
        NOT_IMPLEMENTED;
    }
}