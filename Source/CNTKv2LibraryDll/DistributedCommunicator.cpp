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
#include <numeric>

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    void Recreate(const std::vector<NDArrayViewPtr>& values, std::vector<NDArrayViewPtr>& output)
    {
        output.resize(values.size());
        for (size_t i = 0; i < values.size(); ++i)
        {
            const auto inputView = values[i];
            output[i] = MakeSharedObject<NDArrayView>(inputView->GetDataType(), inputView->Shape(), inputView->Device());
        }
    }

    DistributedCommunicatorPtr MPICommunicator()
    {
        return std::make_shared<MPICommunicatorImpl>();
    }

    void DistributedCommunicator::Finalize()
    {
        MPIWrapper::DeleteInstance();
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

    MPICommunicatorImpl::MPICommunicatorImpl()
    {
        m_mpi = MPIWrapper::GetInstance();
        if (m_mpi == nullptr)
        {
            m_mpi = MPIWrapper::GetInstance(true /*create*/);
        }
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
                RuntimeError("MPICommunicator: Aggregation for sparse matrices is currently not supported.");

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
                    LogicError("MPICommunicator: Not all values are on the same GPU device id");

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
            Recreate(values, outputValues);
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

    void MPICommunicatorImpl::Gather(
        const Dictionary& input,
        std::vector<std::shared_ptr<Dictionary>>& output,
        const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        CheckWorkers(sendToWorkers);

        std::stringstream dict;
        dict << input;
        std::string encoded = dict.str();

        // Exchange data sizes.
        int encodedSizeInBytes = (int)encoded.size();
        std::vector<int> othersSize;
        othersSize.resize(m_mpi->NumNodesInUse());
        m_mpi->Gather(&encodedSizeInBytes, 1, &othersSize[0], 1, 0);

        output.resize(m_mpi->NumNodesInUse(), std::make_shared<Dictionary>());

        int totalSizeInBytes = std::accumulate(othersSize.begin(), othersSize.end(), 0);

        // Exchange actual data
        std::vector<char> gathered;
        gathered.resize(std::max(totalSizeInBytes, 1)); // buffer should be at least of size 1.
        std::vector<int> offsets;
        offsets.resize(m_mpi->NumNodesInUse());
        int currentOffset = 0;
        for (size_t i = 0; i < offsets.size(); ++i)
        {
            offsets[i] = currentOffset;
            currentOffset += othersSize[i];
        }

        m_mpi->Gatherv(&encoded[0], encoded.size(), &gathered[0], &othersSize[0], &offsets[0], 0);
        if (CurrentWorker().m_globalRank != 0)
            return;

        offsets.push_back(totalSizeInBytes);
        output.resize(m_workers.size());
        for (size_t i = 0; i < offsets.size() - 1; ++i)
        {
            size_t startOffset = offsets[i];
            size_t size = offsets[i + 1] - startOffset;
            std::stringstream ss;
            ss.write(&gathered[startOffset], size);
            output[i] = std::make_shared<Dictionary>();
            ss >> *output[i];
        }
    }

    void MPICommunicatorImpl::Concatenate(const std::vector<NDArrayViewPtr>& input, std::vector<NDArrayViewPtr>& output, const std::unordered_set<DistributedWorkerDescriptor>& workers)
    {
        // TODO: Currently we only support concatenation of inputs of the same size.
        CheckWorkers(workers);

        // Check inputs, currently we support only CPU
        auto nonCpu = std::find_if(input.begin(), input.end(), [](const NDArrayViewPtr& v) { return v->Device() != DeviceDescriptor::CPUDevice(); });
        if (nonCpu != input.end())
            LogicError("MPICommunicator: Currently only NDArrayViews located on CPU are supported for concatenation.");

        output.resize(input.size());
        // Currently we only support concatenation of input of the same size.
        // Gathering blocks sequentially.
        for (size_t i = 0; i < input.size(); ++i)
        {
            if (output[i] == nullptr || 
                output[i]->Shape().TotalSize() != m_mpi->NumNodesInUse() * input[i]->Shape().TotalSize() ||
                output[i]->GetDataType() != input[i]->GetDataType())
            {
                // Allocating flat array for all ranks.
                output[i] = std::make_shared<NDArrayView>(input[i]->GetDataType(), NDShape{ input[i]->Shape().TotalSize() * m_mpi->NumNodesInUse() }, DeviceDescriptor::CPUDevice());
            }
        }

        // Initiate concatenation.
        std::vector<MPI_Request> allReduceRequests(input.size());
        for (size_t i = 0; i < input.size(); ++i)
        {
            auto& in = input[i];
            auto& out = output[i];

            if (input[i]->GetDataType() == DataType::Float)
                m_mpi->AllGatherAsync(in->DataBuffer<float>(), in->Shape().TotalSize(), out->WritableDataBuffer<float>(), in->Shape().TotalSize(), &allReduceRequests[i]);
            else if (input[i]->GetDataType() == DataType::Double)
                m_mpi->AllGatherAsync(in->DataBuffer<double>(), in->Shape().TotalSize(), out->WritableDataBuffer<double>(), in->Shape().TotalSize(), &allReduceRequests[i]);
            else
                LogicError("MPICommunicator: input DataType is not supported.");
        }

        // Wait till all requests are finished.
        m_mpi->WaitAll(allReduceRequests);
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
                    m_mpi->AllReduceAsync(static_cast<float*>(outputData), numElements, &allReduceRequests[i]);
                else
                    m_mpi->AllReduceAsync(static_cast<float*>(inputData), static_cast<float*>(outputData), numElements, &allReduceRequests[i]);
            }
            else if (dataType == DataType::Double)
            {
                if (inputData == outputData)
                    m_mpi->AllReduceAsync(static_cast<double*>(outputData), numElements, &allReduceRequests[i]);
                else
                    m_mpi->AllReduceAsync(static_cast<double*>(inputData), static_cast<double*>(outputData), numElements, &allReduceRequests[i]);
            }
            else
                LogicError("MPICommunicator: Unknown DataType.");
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

    void  MPICommunicatorImpl::Barrier()
    {
        m_mpi->WaitAll();
    }
}