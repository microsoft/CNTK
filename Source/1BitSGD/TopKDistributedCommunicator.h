//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Basics.h"
#include "MPIWrapper.h"
#include "CNTKLibrary.h"
#include "MatrixQuantizerImpl.h"
#include "MatrixCompressor.h"
#include "CUDAPageLockedMemAllocator.h"
#include "Utils.h"
#include "DistributedCommunicator.h"
#include "GPUDataTransferer.h"
#include "c_allreduce_ring.h"
#include "Constants.h"
#include "NcclComm.h"
#include <iostream>
#include "libnbc/nbc.h"

namespace Microsoft {namespace MSR {namespace CNTK {

class SparcStreamAllocBase
{};

template <class ElemType>
class SparcStreamAlloc final : public SparcStreamAllocBase
{
public:
    struct stream* m_buffer;
    Matrix<char>* m_data;
    MemAllocator* m_allocator;

    SparcStreamAlloc(MemAllocator* allocator, size_t size) : m_allocator(allocator)
    {
        if (m_allocator == nullptr)
        {
            m_data = new Matrix<char>(1, size, CPUDEVICE);
        }
        else
        {
            m_data = new Matrix<char>(1, size, (char*)m_allocator->Malloc(size), CPUDEVICE, matrixFlagDontOwnBuffer);
        }
        m_buffer = (struct stream*)m_data->Data();
    }

    ~SparcStreamAlloc()
    {
        if (nullptr != m_data)
        {
            if (m_allocator != nullptr) {
                m_allocator->Free(m_data->Data());
            }
            delete m_data;
            m_data = nullptr;
        }
    }
};
}}}

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
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
        if (viewPtr->GetDataType() == DataType::Float16)
            return viewPtr->WritableDataBuffer<float16>();

        LogicError("Unknown DataType");
        return nullptr; // Make compiler happy.
    }

    class TopkMPICommunicatorImpl final : public MPICommunicatorImpl, public TopkDistributedCommunicator
    {
        using Base = MPICommunicatorImpl;

        template<class T> using vector = std::vector<T>;
        template<class T> using shared_ptr = std::shared_ptr<T>;
        template<class T> using unordered_set = std::unordered_set<T>;

        using MpiFail = Microsoft::MSR::CNTK::MpiFail;
        using QuantizedMatrixBase = Microsoft::MSR::CNTK::QuantizedMatrixBase;
        using QuantizedMatrixBasePtr = shared_ptr<QuantizedMatrixBase>;
        using MatrixCompressorBase = Microsoft::MSR::CNTK::MatrixCompressorBase;
        using SparcStreamAllocBase = Microsoft::MSR::CNTK::SparcStreamAllocBase;
        using CUDAPageLockedMemAllocator = Microsoft::MSR::CNTK::CUDAPageLockedMemAllocator;

        template<class T> using MatrixCompressor = Microsoft::MSR::CNTK::MatrixCompressor<T>;
        template<class T> using SparcStreamAlloc = Microsoft::MSR::CNTK::SparcStreamAlloc<T>;
        template<class T> using QuantizedMatrix = Microsoft::MSR::CNTK::QuantizedMatrix<T>;
        template<class T> using Matrix = Microsoft::MSR::CNTK::Matrix<T>;

    public:
        TopkMPICommunicatorImpl(size_t topK = 0) : m_topK(topK)
        {
            std::cout << "Constructing TopkMPICommunicatorImpl with topk = " << m_topK << endl;
        }

        void TopKAggregateInPlace(
            std::vector<NDArrayViewPtr>& inValues,
            std::vector<NDArrayViewPtr>& residues,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override
        {
            InitializeResiduesIfEmpty(inValues, residues);
            TopKAggregateImpl(inValues, residues, inValues, residues, sendToWorkers);
        }

        const std::unordered_set<DistributedWorkerDescriptor>& Workers() const override { return Base::Workers(); }
        const DistributedWorkerDescriptor& CurrentWorker() const override { return Base::CurrentWorker(); }
        DistributedCommunicatorPtr SubGroup(const std::unordered_set<DistributedWorkerDescriptor>& g) const override { return Base::SubGroup(g); }
        void Concatenate(
            const std::vector<ValuePtr>& in,
            std::vector<ValuePtr>& out,
            const std::unordered_set<DistributedWorkerDescriptor>& w) override
        {
            Base::Concatenate(in, out, w);
        }

        void AggregateInPlace(
            const std::vector<NDArrayViewPtr>& values,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override
        {
            Base::AggregateInPlace(values, sendToWorkers);
        }

        void Aggregate(
            const std::vector<NDArrayViewPtr>& values,
            std::vector<NDArrayViewPtr>& outputValues,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override
        {
            Base::Aggregate(values, outputValues, sendToWorkers);
        }

        void Barrier() override
        {
            Base::Barrier();
        }

        virtual void Concatenate(
            const std::vector<NDArrayViewPtr>& input,
            std::vector<NDArrayViewPtr>& output,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override
        {
            Base::Concatenate(input, output, sendToWorkers);
        }

        virtual void Gather(
            const Dictionary& input,
            std::vector<DictionaryPtr>& output,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override
        {
            Base::Gather(input, output, sendToWorkers);
        }

    private:

        void InitializeResiduesIfEmpty(
            const std::vector<NDArrayViewPtr>& values,
            vector<NDArrayViewPtr>& residues)
        {
            if (!residues.empty())
                return;

            cout << "InitializeResiduesIfEmpty" << endl;
            residues.resize(values.size());

            for (auto i = 0; i < values.size(); ++i)
            {
                auto view = values[i];
                auto device = view->Device();

                if (view->GetDataType() == DataType::Float)
                    InitializeResidue<float>(view, residues[i]);
                else if (view->GetDataType() == DataType::Double)
                    InitializeResidue<double>(view, residues[i]);
            }
        }

        template<class ElemType>
        void InitializeResidue(const NDArrayViewPtr& view, NDArrayViewPtr& residue)
        {
            auto v = GetMatrix<ElemType>(view);
            size_t nRow = v->GetNumRows();
            size_t nCol = v->GetNumCols();
            residue = MakeSharedObject<NDArrayView>(AsDataType<ElemType>(), NDShape{ nRow, nCol }, AsDeviceDescriptor(v->GetDeviceId()));
        }

        void Initialize(const std::vector<NDArrayViewPtr>& values)
        {
            assert(CPUDEVICE < 0); // just in case somebody decides to change CPUDEVICE macro.
            DeviceDescriptor lastGpuDevice = DeviceDescriptor::CPUDevice();
            m_gpuDataTransferers.resize(values.size());
            m_intermediateCPUBuffers.resize(values.size());
            m_preAggregatedGradientCompressors.resize(values.size());
            m_sendbufs.resize(values.size());
            m_recvbufs.resize(values.size());

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

                    if (!ShouldUseTopK(view))
                    {
                        auto requiredSize = GetBufferSize(view);
                        m_gpuDataTransferers[i] = std::make_shared<GPUDataTransferer>(device.Id(), true);
                        if (m_intermediateCPUBuffers[i].totalSize < requiredSize)
                            m_intermediateCPUBuffers[i] = AllocateIntermediateBuffer(device.Id(), requiredSize);
                    }
                    else
                    {
                        if (view->GetDataType() == DataType::Float)
                            Initialize_Buffer<float>(i, view);
                        else if (view->GetDataType() == DataType::Double)
                            Initialize_Buffer<double>(i, view);
                    }
                }
                else
                {
                    LogicError("Invalid device type (%u).", (unsigned int)device.Type());
                }
            }
        }

        template<class ElemType>
        void Initialize_Buffer(const size_t i, const NDArrayViewPtr& view)
        {
            auto v = GetMatrix<ElemType>(view);
            size_t nRow = v->GetNumRows();
            size_t nCol = v->GetNumCols();
            size_t dim = nRow * nCol;

            size_t topK = GetTopK(DEFAULT_BUCKET_SIZE, m_topK);
        
            // size_t numBuckets = dim / DEFAULT_BUCKET_SIZE;
            size_t numBuckets = (dim + (DEFAULT_BUCKET_SIZE - 1)) / DEFAULT_BUCKET_SIZE;

            m_preAggregatedGradientCompressors[i] = std::make_shared<MatrixCompressor<ElemType>>(view->Device().Id(), true);

            size_t size_s = sizeof(unsigned) + topK * numBuckets * (sizeof(unsigned) + sizeof(ElemType));
            m_sendbufs[i] = std::make_shared<SparcStreamAlloc<ElemType>>(m_allocator.get(), size_s);
            GetSparcStreamAlloc<ElemType>(m_sendbufs[i]).m_buffer->nofitems = topK * numBuckets;

            size_t size_r = sizeof(unsigned) + (dim * sizeof(ElemType));
            m_recvbufs[i] = std::make_shared<SparcStreamAlloc<ElemType>>(m_allocator.get(), size_r);
        }

        void TopKAggregateImpl(
            const std::vector<NDArrayViewPtr>& inputValues,
            const std::vector<NDArrayViewPtr>& inputResidules,
            const std::vector<NDArrayViewPtr>& outputValues,
            const std::vector<NDArrayViewPtr>& outputResidules,
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
            std::vector<NDArrayViewPtr> residulesBeforeAggregate; // Corresponding to inputResidules
            std::vector<NDArrayViewPtr> residulesAfterAggregate; // Corresponding to outputResidules
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
                    residulesBeforeAggregate.push_back(inputResidules[i]);
                    residulesAfterAggregate.push_back(outputResidules[i]);
                }
            }

            // Do the packing to reduce the number of MPI requests.
            // Do not re-allocating the continous buffer if existing buffer size equals to required one.
            m_aggregationBufferFloat = SetContinuousBuffer<float>(packedFloatGradientsIndex, packedFloatGradientsSizeInBytes,
                inputValues, outputValues, inputResidules, outputResidules,
                valuesToAggregate, valuesAfterAggregate, residulesBeforeAggregate, residulesAfterAggregate);
            m_aggregationBufferDouble = SetContinuousBuffer<double>(packedDoubleGradientsIndex, packedDoubleGradientsSizeInBytes,
                inputValues, outputValues, inputResidules, outputResidules,
                valuesToAggregate, valuesAfterAggregate, residulesBeforeAggregate, residulesAfterAggregate);

            PackToContinuousBuffer(m_aggregationBufferFloat.get(), packedFloatGradientsIndex,
                inputValues, outputValues, inputResidules, outputResidules,
                valuesToAggregate, valuesAfterAggregate, residulesBeforeAggregate, residulesAfterAggregate);
            PackToContinuousBuffer(m_aggregationBufferDouble.get(), packedDoubleGradientsIndex,
                inputValues, outputValues, inputResidules, outputResidules,
                valuesToAggregate, valuesAfterAggregate, residulesBeforeAggregate, residulesAfterAggregate);

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

            if (m_nccl == nullptr)
            {
                m_nccl.reset(new NcclComm(DeviceDescriptor::UseDefaultDevice().Id(), m_mpi));
            }

            // For all values residing on GPU initiate async transfer to CPU buffers if needed
            CopyDataFromGPUToCPU(valuesToAggregate, residulesBeforeAggregate);

            std::vector<MPI_Request> allReduceRequests;
            std::vector<NBC_Handle> sparseAllReduceHandles;
            for (auto i = 0; i < numValues; ++i)
            {
                auto inputValue = valuesToAggregate[i];

                if (ShouldCopyDataToCPU(inputValue))
                {
                    if (!ShouldUseTopK(inputValue))
                    {
                        // TODO: actually, we can start reducing all cpu values first, and then wait for the gpu->cpu transfer to finish.
                        m_gpuDataTransferers[i]->WaitForCopyGPUToCPUAsync();
                    }
                    else
                    {
                        if (inputValue->GetDataType() == DataType::Float)
                            GetCompressor<float>(m_preAggregatedGradientCompressors[i]).WaitTopKAsyncDone();
                        else if (inputValue->GetDataType() == DataType::Double)
                            GetCompressor<double>(m_preAggregatedGradientCompressors[i]).WaitTopKAsyncDone();
                    }
                }

                auto numElements = inputValue->Shape().TotalSize();
                auto dataType = inputValue->GetDataType();

                auto& outputValue = valuesAfterAggregate[i];

                assert(numElements == outputValue->Shape().TotalSize());
                assert(dataType == outputValue->GetDataType());
                assert(inputValue->Device() == outputValue->Device());

                if (!ShouldCopyDataToCPU(inputValue) || !ShouldUseTopK(inputValue))
                {
                    // NO TOPK
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
                    else if (dataType == DataType::Float16)
                    {
                        AllReduceDataHalf(static_cast<half*>(inputData), static_cast<half*>(outputData), numElements,
                            &allReduceRequests, (inputValue->Device() == DeviceDescriptor::CPUDevice()));
                    }
                    else
                        LogicError("MPICommunicator: Unknown DataType.");
                }
                else
                {
                    // TOPK
                    bool useAsyncReduce = true;
                    if (useAsyncReduce)
                    {
                        if (inputValue->GetDataType() == DataType::Float)
                            DoAllReduceAsync<float>(i, inputValue, &sparseAllReduceHandles);
                        else if (inputValue->GetDataType() == DataType::Double)
                            DoAllReduceAsync<double>(i, inputValue, &sparseAllReduceHandles);
                    }
                    else
                    {
                        // blocking call of all reduce
                        if (inputValue->GetDataType() == DataType::Float)
                            DoAllReduceAndUnTopKAsync<float>(i, inputValue, outputValue);
                        else if (inputValue->GetDataType() == DataType::Double)
                            DoAllReduceAndUnTopKAsync<double>(i, inputValue, outputValue);
                    }
                }
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
                    if (!ShouldUseTopK(value))
                    {
                        auto view = valuesAfterAggregate[idx];
                        auto size = GetBufferSize(view);
                        auto& transferer = m_gpuDataTransferers[idx];
                        auto& buffer = m_intermediateCPUBuffers[idx];
                        transferer->CopyCPUToGPUAsync(buffer.data.get(), size, GetDataBuffer(view));
                    }
                    else
                    {
                        // TODO, currently this shouldn't be hit, cause topk allreduce is blocking call
                        if (value->GetDataType() == DataType::Float)
                            UnTopKAsync<float>(idx, value);
                        else if (value->GetDataType() == DataType::Double)
                            UnTopKAsync<double>(idx, value);
                    }
                }
            }

            for (size_t idx = 0; idx < sparseAllReduceHandles.size(); idx++)
            {
                // There is no NBC_WaitAny, only NBC_Wait
                NBC_Wait(&sparseAllReduceHandles[idx], MPI_STATUS_IGNORE) || MpiFail("NBC_Wait");

                assert(idx < valuesToAggregate.size());
                auto value = valuesToAggregate[idx];
                if (value->GetDataType() == DataType::Float)
                    UnTopKAsync<float>(idx, value);
                else if (value->GetDataType() == DataType::Double)
                    UnTopKAsync<double>(idx, value);
            }

            // TODO: Should not wait, simply publishing event on the compute stream should be sufficient
            for (auto i = 0; i < numValues; ++i)
            {
                if (ShouldCopyDataToCPU(valuesToAggregate[i]))
                {
                    if (!ShouldUseTopK(valuesToAggregate[i]))
                    {
                        m_gpuDataTransferers[i]->WaitForCopyCPUToGPUAsync();
                    }
                    else
                    {
                        // TODO, currently this shouldn't be hit, cause topk allreduce is blocking call
                        if (valuesToAggregate[i]->GetDataType() == DataType::Float)
                            GetCompressor<float>(m_preAggregatedGradientCompressors[i]).WaitUnTopKAsyncDone();
                        else if (valuesToAggregate[i]->GetDataType() == DataType::Double)
                            GetCompressor<double>(m_preAggregatedGradientCompressors[i]).WaitUnTopKAsyncDone();
                    }
                }
            }

            // Unpack the continuous buffer
            UnpackFromContinuousBuffer(m_aggregationBufferFloat.get(), outputValues, packedFloatGradientsIndex);
            UnpackFromContinuousBuffer(m_aggregationBufferDouble.get(), outputValues, packedDoubleGradientsIndex);
        }

        template <typename ElemType>
        std::unique_ptr<Matrix<ElemType>> SetContinuousBuffer(std::vector<size_t>& packedGradientsIndex, size_t packedGradientsSizeInBytes,
            const std::vector<NDArrayViewPtr>& inputValues, const std::vector<NDArrayViewPtr>& outputValues,
            const std::vector<NDArrayViewPtr>& inputResidules, const std::vector<NDArrayViewPtr>& outputResidules,
            std::vector<NDArrayViewPtr>& valuesToAggregate, std::vector<NDArrayViewPtr>& valuesAfterAggregate,
            std::vector<NDArrayViewPtr>& residulesBeforeAggregate, std::vector<NDArrayViewPtr>& residulesAfterAggregate)
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
                residulesBeforeAggregate.push_back(inputResidules[packedGradientsIndex.front()]);
                residulesAfterAggregate.push_back(outputResidules[packedGradientsIndex.front()]);
                packedGradientsIndex.clear();
            }
            return std::unique_ptr<Matrix<ElemType>>{ nullptr };
        }

        template <typename ElemType>
        void PackToContinuousBuffer(Matrix<ElemType>* aggregationBuffer, std::vector<size_t>& packedGradientsIndex,
            const std::vector<NDArrayViewPtr>& inputValues, const std::vector<NDArrayViewPtr>& outputValues,
            const std::vector<NDArrayViewPtr>& inputResidules, const std::vector<NDArrayViewPtr>& outputResidules,
            std::vector<NDArrayViewPtr>& valuesToAggregate, std::vector<NDArrayViewPtr>& valuesAfterAggregate,
            std::vector<NDArrayViewPtr>& residulesBeforeAggregate, std::vector<NDArrayViewPtr>& residulesAfterAggregate)
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
                    residulesBeforeAggregate.push_back(inputResidules[i]);
                    residulesAfterAggregate.push_back(outputResidules[i]);
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
            data->SetIsPacked(true);
            valuesToAggregate.push_back(data);
            valuesAfterAggregate.push_back(data);
            // No need to update residules for packed data
        }

        virtual void CopyDataFromGPUToCPU(std::vector<NDArrayViewPtr>& inputValues, std::vector<NDArrayViewPtr>& inputResidules)
        {
            for (auto i = 0; i < inputValues.size(); ++i)
            {
                auto view = inputValues[i];
                auto residule = inputResidules[i];
                if (ShouldCopyDataToCPU(view))
                {
                    if (!ShouldUseTopK(view))
                    {
                        auto& transferer = m_gpuDataTransferers[i];
                        auto& buffer = m_intermediateCPUBuffers[i];
                        transferer->CopyGPUToCPUAsync(GetDataBuffer(view), GetBufferSize(view), buffer.data.get());
                    }
                    else
                    {
                        if (view->GetDataType() == DataType::Float)
                            TopKAsync<float>(i, view, residule);
                        else if (view->GetDataType() == DataType::Double)
                            TopKAsync<double>(i, view, residule);
                    }
                }
            }
        }

        template<class ElemType>
        void TopKAsync(size_t i, NDArrayViewPtr& value, NDArrayViewPtr& residule)
        {
            GetCompressor<ElemType>(m_preAggregatedGradientCompressors[i]).TopKAsync(
                *(GetWritableMatrix<ElemType>(value)),
                *(GetWritableMatrix<ElemType>(residule)),
                *(GetSparcStreamAlloc<ElemType>(m_sendbufs[i]).m_buffer),
                *(GetWritableMatrix<ElemType>(residule)),
                m_topK);
        }

        template<class ElemType>
        void UnTopKAsync(size_t i, NDArrayViewPtr& value)
        {
            GetCompressor<ElemType>(m_preAggregatedGradientCompressors[i]).UnTopKAsync(
                *(GetSparcStreamAlloc<ElemType>(m_recvbufs[i]).m_buffer),
                *(GetWritableMatrix<ElemType>(value)));
        }

        template<class ElemType>
        void DoAllReduceAndUnTopKAsync(size_t i, NDArrayViewPtr& inputValue, NDArrayViewPtr& outputValue)
        {
            DoAllReduce<ElemType>(i, inputValue);
            UnTopKAsync<ElemType>(i, outputValue);
        }

        template<class ElemType>
        void DoAllReduce(size_t i, NDArrayViewPtr& inputValue)
        {
            auto v = GetMatrix<ElemType>(inputValue);
            size_t dim = v->GetNumRows() * v->GetNumCols();
            GetCompressor<ElemType>(m_preAggregatedGradientCompressors[i]).AllReduce(GetSparcStreamAlloc<ElemType>(m_sendbufs[i]).m_buffer, GetSparcStreamAlloc<ElemType>(m_recvbufs[i]).m_buffer, dim);
        }

        template <class ElemType>
        void DoAllReduceAsync(size_t i, NDArrayViewPtr& inputValue, std::vector<NBC_Handle>* pAllReduceRequests)
        {
            auto v = GetMatrix<ElemType>(inputValue);
            size_t nRow = v->GetNumRows();
            size_t nCol = v->GetNumCols();
            size_t dim = nRow * nCol;
            size_t topK = GetTopK(DEFAULT_BUCKET_SIZE, m_topK);
            //size_t numBuckets = dim / DEFAULT_BUCKET_SIZE;
            size_t numBuckets = (dim + (DEFAULT_BUCKET_SIZE - 1)) / DEFAULT_BUCKET_SIZE;

            pAllReduceRequests->push_back(NBC_Handle());

            GetCompressor<ElemType>(m_preAggregatedGradientCompressors[i]).IallReduce(
                GetSparcStreamAlloc<ElemType>(m_sendbufs[i]).m_buffer,
                GetSparcStreamAlloc<ElemType>(m_recvbufs[i]).m_buffer,
                topK * numBuckets,
                dim,
                m_mpi->Communicator(),
                &(pAllReduceRequests->back()));
        }

        bool ShouldUseTopK(const NDArrayViewPtr& view)
        {
            // Skip top K for packed gradients
            if (view->IsPacked())
                return false;

            size_t topK = GetTopK(DEFAULT_BUCKET_SIZE, m_topK);
            if (view->GetDataType() == DataType::Float)
                return ((sizeof(unsigned) + sizeof(float)) * topK < DEFAULT_BUCKET_SIZE * sizeof(float));
            else if (view->GetDataType() == DataType::Double)
                return ((sizeof(unsigned) + sizeof(double)) * topK < DEFAULT_BUCKET_SIZE * sizeof(double));

            // Anything else fallback to false
            return false;
        }

        static size_t GetTopK(int currNumElementsPerBuckets, size_t topK)
        {
            if (topK == 0) return currNumElementsPerBuckets;
            return topK;
        }

        template <typename ElementType>
        MatrixCompressor<ElementType>& GetCompressor(const shared_ptr<MatrixCompressorBase>& compressor)
        {
            return static_cast<MatrixCompressor<ElementType>&>(*compressor);
        }

        template <typename ElementType>
        SparcStreamAlloc<ElementType>& GetSparcStreamAlloc(const shared_ptr<SparcStreamAllocBase>& msa)
        {
            return static_cast<SparcStreamAlloc<ElementType>&>(*msa);
        }

        size_t m_topK;

        vector<shared_ptr<MatrixCompressorBase>> m_preAggregatedGradientCompressors;
        vector<shared_ptr<SparcStreamAllocBase>> m_sendbufs;
        vector<shared_ptr<SparcStreamAllocBase>> m_recvbufs;

        const unique_ptr<CUDAPageLockedMemAllocator> m_allocator;
    };
}
