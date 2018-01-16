//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <functional>
#include "Basics.h"
#include "Constants.h"
#include "CNTKLibrary.h"
#include "DistributedCommunicator.h"
#include "CUDAPageLockedMemAllocator.h"
#include "MatrixQuantizerImpl.h"
#include "GPUDataTransferer.h"
#include <numeric>
#include "Utils.h"

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

    DistributedCommunicatorPtr MPICommunicator(size_t packThresholdSizeInBytes)
    {
        return std::make_shared<MPICommunicatorImpl>(packThresholdSizeInBytes);
    }

    void DistributedCommunicator::Finalize()
    {
        auto mpi = MPIWrapper::GetInstance(false);
        if (mpi)
            mpi->Finalize();
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

    MPICommunicatorImpl::MPICommunicatorImpl(size_t packThresholdSizeInBytes)
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
        m_packThresholdSizeInBytes = packThresholdSizeInBytes;
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
            if (device.Type() == DeviceKind::CPU)
            {
                m_intermediateCPUBuffers[i] = Buffer();
                m_gpuDataTransferers[i] = nullptr;
            }
            else if (device.Type() == DeviceKind::GPU)
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
            else
            {
                LogicError("Invalid device type (%u).", (unsigned int)device.Type());
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
        if (device.Type() == DeviceKind::GPU)
        {
            // Since we will be copying the gradients asynchronously, let us
            // ensure that the gradient matrices have been computed before starting to aggregate
            // them asynchronously on another thread. This essentially means that when we are using
            // a GPU device, we will synchronize on the main GPU compute stream before starting
            // the gradient aggregation asynchronously on a separate stream
            std::unique_ptr<MatrixComputeStreamEvent> mainStreamSyncEvent(MatrixComputeStreamEvent::Create(device.Id()));
            mainStreamSyncEvent->SynchronizeDataTransferFetchStreamWithEvent<float>();
        }
        else
        {
            LogicError("Invalid device type (%u).", (unsigned int)device.Type());
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
        AggregateImpl(values, values, sendToWorkers);
    }

    void MPICommunicatorImpl::AggregateImpl(
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

        std::vector<NDArrayViewPtr> valuesToAggregate; // Corresponding to inputValues
        std::vector<NDArrayViewPtr> valuesAfterAggregate; // Corresponding to outputValues
        size_t packedFloatGradientsSizeInBytes = 0;
        size_t packedDoubleGradientsSizeInBytes = 0;
        std::vector<size_t> packedFloatGradientsIndex;
        std::vector<size_t> packedDoubleGradientsIndex;
        for (auto i = 0; i < numValues; i++)
        {
            // Push index to packing queue if the gradient's size is less than threshold size
            if (!inputValues[i]->IsSliceView() && GetBufferSize(inputValues[i]) < m_packThresholdSizeInBytes && (inputValues[i]->GetDataType() == DataType::Float))
            {
                packedFloatGradientsSizeInBytes += GetBufferSize(inputValues[i]);
                packedFloatGradientsIndex.push_back(i);
            }
            else if (!inputValues[i]->IsSliceView() && GetBufferSize(inputValues[i]) < m_packThresholdSizeInBytes && (inputValues[i]->GetDataType() == DataType::Double))
            {
                packedDoubleGradientsSizeInBytes += GetBufferSize(inputValues[i]);
                packedDoubleGradientsIndex.push_back(i);
            }
            else
            {
                valuesToAggregate.push_back(inputValues[i]);
                valuesAfterAggregate.push_back(outputValues[i]);
            }
        }

        // Do the packing to reduce the number of MPI requests.
        // Do not re-allocating the continous buffer if existing buffer size equals to required one.
        m_aggregationBufferFloat = SetContinuousBuffer<float>(packedFloatGradientsIndex, packedFloatGradientsSizeInBytes, inputValues, outputValues,
            valuesToAggregate, valuesAfterAggregate);
        m_aggregationBufferDouble = SetContinuousBuffer<double>(packedDoubleGradientsIndex, packedDoubleGradientsSizeInBytes, inputValues, outputValues,
            valuesToAggregate, valuesAfterAggregate);

        PackToContinuousBuffer(m_aggregationBufferFloat.get(), packedFloatGradientsIndex, inputValues, outputValues, valuesToAggregate, valuesAfterAggregate);
        PackToContinuousBuffer(m_aggregationBufferDouble.get(), packedDoubleGradientsIndex, inputValues, outputValues, valuesToAggregate, valuesAfterAggregate);

        numValues = valuesToAggregate.size();

        Initialize(valuesToAggregate);

        // We need to make sure no compuatation happens on the main CUDA stream.
        auto device = GetNonCPUDevice(valuesToAggregate);
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

        // BUGBUG: assuming the all values on the same device
        if (m_nccl == nullptr)
        {
            m_nccl.reset(new NcclComm(AsCNTKImplDeviceId(inputValues[0]->Device()), m_mpi));
        }

        // For all values residing on GPU initiate async transfer to CPU buffers if needed
        CopyDataFromGPUToCPU(valuesToAggregate);

        std::vector<MPI_Request> allReduceRequests;
        for (auto i = 0; i < numValues; ++i)
        {
            auto inputValue = valuesToAggregate[i];

            if (ShouldCopyDataToCPU(inputValue))
            {
                // TODO: actually, we can start reducing all cpu values first, and then wait for the gpu->cpu transfer to finish.
                m_gpuDataTransferers[i]->WaitForCopyGPUToCPUAsync();
            }

            auto numElements = inputValue->Shape().TotalSize();
            auto dataType = inputValue->GetDataType();

            auto& outputValue = valuesAfterAggregate[i];

            assert(numElements == outputValue->Shape().TotalSize());
            assert(dataType == outputValue->GetDataType());
            assert(inputValue->Device() == outputValue->Device());

            void* inputData = (ShouldCopyDataToCPU(inputValue)) ? m_intermediateCPUBuffers[i].data.get() : GetDataBuffer(inputValue);
            void* outputData = (ShouldCopyDataToCPU(inputValue)) ? m_intermediateCPUBuffers[i].data.get() : GetDataBuffer(outputValue);

            if (dataType == DataType::Float)
            {
                AllReduceData(static_cast<float*>(inputData), static_cast<float*>(outputData), numElements,
                    &allReduceRequests, (inputValue->Device() == DeviceDescriptor::CPUDevice()));
            }
            else if (dataType == DataType::Double)
            {
                AllReduceData(static_cast<double*>(inputData), static_cast<double*>(outputData), numElements,
                    &allReduceRequests, (inputValue->Device() == DeviceDescriptor::CPUDevice()));
            }
            else
                LogicError("MPICommunicator: Unknown DataType.");
        }

        if (m_nccl->IsSupported())
        {
            m_nccl->Sync();
        }

        // wait for async all reduce to complete. As soon as one of the requests is finished,
        // check if corresponding value is gpu bound and, if it is the case, initiate a cpu-to-gpu transfer.
        size_t numAllReduceRequestsCompleted = 0;
        while (numAllReduceRequestsCompleted < allReduceRequests.size())
        {
            int idx = MPI_UNDEFINED;
            m_mpi->WaitAny(allReduceRequests.data(), (int)allReduceRequests.size(), &idx);
            if (idx == MPI_UNDEFINED)
            {
                break;
            }

            numAllReduceRequestsCompleted++;

            assert(idx < valuesToAggregate.size());
            auto value = valuesToAggregate[idx];

            if (ShouldCopyDataToCPU(value))
            {
                auto view = valuesAfterAggregate[idx];
                auto size = GetBufferSize(view);
                auto& transferer = m_gpuDataTransferers[idx];
                auto& buffer = m_intermediateCPUBuffers[idx];
                transferer->CopyCPUToGPUAsync(buffer.data.get(), size, GetDataBuffer(view));
            }
        }

        // TODO: Should not wait, simply publishing event on the compute stream should be sufficient
        for (auto i = 0; i < numValues; ++i)
        {
            if (ShouldCopyDataToCPU(valuesToAggregate[i]))
                m_gpuDataTransferers[i]->WaitForCopyCPUToGPUAsync();
        }

        // Unpack the continuous buffer
        UnpackFromContinuousBuffer(m_aggregationBufferFloat.get(), outputValues, packedFloatGradientsIndex);
        UnpackFromContinuousBuffer(m_aggregationBufferDouble.get(), outputValues, packedDoubleGradientsIndex);
    }

    void MPICommunicatorImpl::AllReduceSparseBlockColumn(
        std::vector<NDArrayViewPtr>& sbcValues)
    {
        if (m_mpi->NumNodesInUse() == 1) // No need to aggregate anything.
            return;
#ifdef CPUONLY
        LogicError("Sparse block column aggregation on CPUDevice not implemented");
#else
        // a handy struct to access sparse block column matrix internal data
        struct SBCInfo
        {
            const void* nz;
            const SparseIndexType* blockId2Col;
            const SparseIndexType* col2BlockId;
            size_t numBlocks;
            size_t numRows;
            size_t numCols;

            SBCInfo(const NDArrayViewPtr& sbc)
            {
                if (sbc->Device() == DeviceDescriptor::CPUDevice())
                    LogicError("Unimplmented sparse block column aggregation on CPUDevice. Please cntk.cntk_py.use_sparse_gradient_aggregation_in_data_parallel_sgd(False) to avoid this error.");

                if (sbc->GetDataType() == DataType::Float)
                {
                    auto tuple = sbc->SparseBlockColumnDataBuffers<float>();
                    nz = std::get<0>(tuple);
                    blockId2Col = std::get<1>(tuple);
                    col2BlockId = std::get<2>(tuple);
                    numBlocks = std::get<3>(tuple);
                    numRows = std::get<4>(tuple);
                    numCols = std::get<5>(tuple);
                }
                else if (sbc->GetDataType() == DataType::Double)
                {
                    auto tuple = sbc->SparseBlockColumnDataBuffers<double>();
                    nz = std::get<0>(tuple);
                    blockId2Col = std::get<1>(tuple);
                    col2BlockId = std::get<2>(tuple);
                    numBlocks = std::get<3>(tuple);
                    numRows = std::get<4>(tuple);
                    numCols = std::get<5>(tuple);
                }
                else
                    LogicError("MPICommunicator: Unknown DataType.");
            }
        };

        m_intermediateSBCIndexCPUBuffers.resize(sbcValues.size());
        m_intermediateSBCValueCPUBuffers.resize(sbcValues.size());

        // First, AllReduce(Max) to get the aggregated non-zero columns
        bool aggregateOnCPU = !(m_nccl->IsSupported() || m_mpi->UseGpuGdr());

        std::vector<SBCInfo> sbcInfos;
        for (size_t idx = 0; idx < sbcValues.size(); idx++)
        {
            sbcInfos.emplace_back(SBCInfo(sbcValues[idx]));
            auto& sbcInfo = sbcInfos[idx];
            size_t requiredSize = sbcInfo.numCols * sizeof(SparseIndexType);
            if (m_intermediateSBCIndexCPUBuffers[idx].totalSize < requiredSize)
                m_intermediateSBCIndexCPUBuffers[idx] = AllocateIntermediateBuffer(sbcValues[idx]->Device().Id(), requiredSize);

            SparseIndexType* pCol2BlockId = nullptr;
            if (aggregateOnCPU)
            {
                pCol2BlockId = reinterpret_cast<SparseIndexType*>(m_intermediateSBCIndexCPUBuffers[idx].data.get());
                cudaMemcpy(pCol2BlockId, sbcInfo.col2BlockId, sizeof(SparseIndexType) * sbcInfo.numCols, cudaMemcpyDeviceToHost);
            }
            else
            {
                // aggregate on GPU, since we'll do inplace aggregation for col2BlockId, remember the original one in blockId2Col
                pCol2BlockId = const_cast<SparseIndexType*>(sbcInfo.col2BlockId);
                cudaMemcpy(const_cast<SparseIndexType*>(sbcInfo.blockId2Col), pCol2BlockId, sizeof(SparseIndexType) * sbcInfo.numCols, cudaMemcpyDeviceToDevice);
            }

            // all-reduce max to find out the columns that would have value after aggregation
            AllReduceData<SparseIndexType>(pCol2BlockId, pCol2BlockId, sbcInfo.numCols, nullptr, aggregateOnCPU, MPI_MAX, true);
        }

        if (m_nccl->IsSupported())
        {
            m_nccl->Sync();
        }

        for (size_t idx = 0; idx < sbcInfos.size(); idx++)
        {
            auto sbc = sbcValues[idx];
            auto& sbcInfo = sbcInfos[idx];

            // copy to CPU to count aggregated columns and allocate space for values
            SparseIndexType* aggregatedCol2BlockId = reinterpret_cast<SparseIndexType*>(m_intermediateSBCIndexCPUBuffers[idx].data.get());

            // if aggregation is done on CPU, the buffer already has valid data, otherwise, copy from gpu
            if (!aggregateOnCPU)
            {
                cudaMemcpy(aggregatedCol2BlockId, sbcInfo.col2BlockId, sbcInfo.numCols * sizeof(SparseIndexType), cudaMemcpyDeviceToHost);
            }

            // update col2blockId and count new blocks
            size_t numBlocks = 0;
            for (size_t col = 0; col < sbcInfo.numCols; col++)
            {
                if (aggregatedCol2BlockId[col] != SparseIndex_NotAssigned)
                {
                    // note that the order has been changed after aggregation. This is to make sure the indices are the same for all workers
                    aggregatedCol2BlockId[col] = numBlocks;
                    numBlocks++;
                }
            }

            // adjust sbc with the new col2BlockId. old nz would be copied to the new nz buffer according to the new Col2BlockId,
            // and the rest of nz buffer would be filled with zero. BlockId2Col would be set accordingly too.
            // after this, all nz buffers in workers are aligned and ready for aggregation
            sbc->AdjustSparseBlockColumn(aggregatedCol2BlockId, numBlocks, /*useBlockId2Col*/ !aggregateOnCPU);

            // update the info as nzvalue may got reallocated
            sbcInfo = SBCInfo(sbc);
            size_t requiredElements = sbcInfo.numRows * numBlocks;
            size_t requiredSize = requiredElements * DataTypeSize(sbc->GetDataType());

            void* nzGPU = const_cast<void*>(sbcInfo.nz);
            void* nz = nzGPU;
            if (aggregateOnCPU)
            {
                // if aggregating on CPU, copy nz from GPU first
                if (m_intermediateSBCValueCPUBuffers[idx].totalSize < requiredSize)
                    m_intermediateSBCValueCPUBuffers[idx] = AllocateIntermediateBuffer(sbcValues[idx]->Device().Id(), requiredSize);
                void* nzCPU = m_intermediateSBCValueCPUBuffers[idx].data.get();
                cudaMemcpy(nzCPU, nz, requiredSize, cudaMemcpyDeviceToHost);
                nz = nzCPU;
            }

            if (sbc->GetDataType() == DataType::Float)
                AllReduceData<float>((float*)nz, (float*)nz, requiredElements, nullptr, aggregateOnCPU, MPI_SUM, true);
            else
                AllReduceData<double>((double*)nz, (double*)nz, requiredElements, nullptr, aggregateOnCPU, MPI_SUM, true);

            if (aggregateOnCPU)
            {
                // since only GPU sparse block column is supported, copy aggregated nz back to GPU
                cudaMemcpy(nzGPU, nz, requiredSize, cudaMemcpyHostToDevice);
            }
        }
#endif
    }

    void MPICommunicatorImpl::Barrier()
    {
        m_mpi->WaitAll();
    }

    bool MPICommunicatorImpl::ShouldCopyDataToCPU(NDArrayViewPtr inputValue)
    {
        if (inputValue->Device() == DeviceDescriptor::CPUDevice())
            return false;

        // Donot copy if NCCL is supported or GPUDirect RDMA is used
        if (m_nccl->IsSupported() || m_mpi->UseGpuGdr())
            return false;

        return true;
    }

    void MPICommunicatorImpl::CopyDataFromGPUToCPU(std::vector<NDArrayViewPtr>& inputValues)
    {
        for (auto i = 0; i < inputValues.size(); ++i)
        {
            auto view = inputValues[i];
            if (ShouldCopyDataToCPU(inputValues[i]))
            {
                auto& transferer = m_gpuDataTransferers[i];
                auto& buffer = m_intermediateCPUBuffers[i];
                transferer->CopyGPUToCPUAsync(GetDataBuffer(view), GetBufferSize(view), buffer.data.get());
            }
        }
    }

    template <typename ElemType>
    std::unique_ptr<Matrix<ElemType>> MPICommunicatorImpl::SetContinuousBuffer(std::vector<size_t>& packedGradientsIndex, size_t packedGradientsSizeInBytes,
        const std::vector<NDArrayViewPtr>& inputValues, const std::vector<NDArrayViewPtr>& outputValues,
        std::vector<NDArrayViewPtr>& valuesToAggregate, std::vector<NDArrayViewPtr>& valuesAfterAggregate)
    {
        if (packedGradientsIndex.size() > 1)
        {
            return std::unique_ptr<Matrix<ElemType>>{new (std::nothrow) Matrix<ElemType>(1, packedGradientsSizeInBytes / sizeof(ElemType),
                AsCNTKImplDeviceId(inputValues[packedGradientsIndex[0]]->Device()))};
        }
        else if (packedGradientsIndex.size() == 1)
        {
            valuesToAggregate.push_back(inputValues[packedGradientsIndex.front()]);
            valuesAfterAggregate.push_back(outputValues[packedGradientsIndex.front()]);
            packedGradientsIndex.clear();
        }
        return std::unique_ptr<Matrix<ElemType>>{ nullptr };
    }

    template <typename ElemType>
    void MPICommunicatorImpl::PackToContinuousBuffer(Matrix<ElemType>* aggregationBuffer, std::vector<size_t>& packedGradientsIndex,
        const std::vector<NDArrayViewPtr>& inputValues, const std::vector<NDArrayViewPtr>& outputValues, std::vector<NDArrayViewPtr>& valuesToAggregate, std::vector<NDArrayViewPtr>& valuesAfterAggregate)
    {
        if (packedGradientsIndex.size() < 1)
        {
            return;
        }

        if (aggregationBuffer == nullptr || packedGradientsIndex.size() == 1)
        {
            for (size_t i : packedGradientsIndex)
            {
                valuesToAggregate.push_back(inputValues[i]);
                valuesAfterAggregate.push_back(outputValues[i]);
            }
            packedGradientsIndex.clear();
            return;
        }

        size_t offset = 0;
        for (size_t i : packedGradientsIndex)
        {
            auto gradient = GetWritableMatrix<ElemType>(inputValues[i]);
            aggregationBuffer->ColumnSlice(offset, gradient->GetNumElements()).AssignValuesOf(gradient->Reshaped(1, gradient->GetNumElements()));
            offset += gradient->GetNumElements();
        }
        ::CNTK::NDShape shape{ aggregationBuffer->GetNumElements() };
        auto data = ::CNTK::MakeSharedObject<::CNTK::NDArrayView>(inputValues[packedGradientsIndex[0]]->GetDataType(), shape, aggregationBuffer->Data(),
            offset * sizeof(ElemType), inputValues[packedGradientsIndex[0]]->Device());
        valuesToAggregate.push_back(data);
        valuesAfterAggregate.push_back(data);
    }

    template <typename ElemType>
    void MPICommunicatorImpl::UnpackFromContinuousBuffer(Matrix<ElemType>* aggregationBuffer, const std::vector<NDArrayViewPtr>& outputValues,
        std::vector<size_t>& packedGradientsIndex)
    {
        if (packedGradientsIndex.size() != 0)
        {
            size_t offset = 0;
            for (size_t i : packedGradientsIndex)
            {
                auto gradient = GetWritableMatrix<ElemType>(outputValues[i]);
                gradient->AssignValuesOf(aggregationBuffer->ColumnSlice(offset, gradient->GetNumElements()).Reshaped(gradient->GetNumRows(), gradient->GetNumCols()));
                offset += gradient->GetNumElements();
            }
        }
    }

    template <typename ElemType>
    void MPICommunicatorImpl::AllReduceData(ElemType* inputData, ElemType* outputData, size_t numElements, std::vector<MPI_Request>* pAllReduceRequests, bool dataOnCPU, MPI_Op op, bool forceSync)
    {
        if (m_nccl->IsSupported() && !dataOnCPU)
        {
            m_nccl->AllReduce(inputData, outputData, numElements, op);

            return;
        }

        if (m_mpi->UseGpuGdr() || forceSync)
        {
            if (inputData == outputData)
                m_mpi->AllReduce(outputData, numElements, op);
            else
                m_mpi->AllReduce(inputData, outputData, numElements, op);

            return;
        }

        pAllReduceRequests->push_back(MPI_Request());
        if (inputData == outputData)
            m_mpi->AllReduceAsync(outputData, numElements, &(pAllReduceRequests->back()), op);
        else
            m_mpi->AllReduceAsync(inputData, outputData, numElements, &(pAllReduceRequests->back()), op);
    }
}
