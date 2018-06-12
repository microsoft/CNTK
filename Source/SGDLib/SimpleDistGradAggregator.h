//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#undef _SCL_SECURE_NO_WARNINGS
#include "Constants.h"
#include "CNTKLibrary.h"
#include "IDistGradAggregator.h"
#include "CUDAPageLockedMemAllocator.h"
#include "NcclComm.h"
#include <future>
#include "GPUDataTransferer.h"
#include "TimerUtility.h"
#include "MatrixQuantizerImpl.h"
#include "c_allreduce_ring.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    template <class ElemType>
        class MyStreamAlloc
        {
            public:
                struct stream* m_buffer;
                Matrix<char>* m_data;
                MemAllocator* m_allocator;

                MyStreamAlloc(MemAllocator* allocator, size_t size) : m_allocator(allocator) {
                    m_data = new Matrix<char>(1, size, (char*)m_allocator->Malloc(size), CPUDEVICE, matrixFlagDontOwnBuffer);
                    m_buffer = (struct stream*)m_data->Data();
                }

                ~MyStreamAlloc() {
                    if (nullptr != m_data)
                    {
                        if(m_allocator != nullptr) {
                            m_allocator->Free(m_data->Data());
                        }
                        delete m_data;
                        m_data = nullptr;
                    }
                }
        };

    template <class ElemType>
        class SimpleDistGradAggregator : public IDistGradAggregator<ElemType>
    {
        UsingIDistGradAggregatorMembers;

        public:
        SimpleDistGradAggregator(const MPIWrapperPtr& mpi, bool useAsyncAggregation, int deviceId, int syncStatsTrace, size_t packThresholdSizeInBytes = DEFAULT_PACK_THRESHOLD_SIZE_IN_BYTES, int topK = -1)
            : IDistGradAggregator<ElemType>(mpi), m_useAsyncAggregation(useAsyncAggregation), m_initialized(false), m_bufferedGradHeader(nullptr), m_syncStatsTrace(syncStatsTrace),
            m_iterationCount(0), m_packThresholdSizeInBytes(packThresholdSizeInBytes), m_topK(topK)
        {}

        ~SimpleDistGradAggregator()
        {
            for (size_t i = 0; i < m_recvHeaders.size(); ++i)
                DistGradHeader::Destroy(m_recvHeaders[i]);

            if (m_bufferedGradHeader != nullptr)
                DistGradHeader::Destroy(m_bufferedGradHeader);
        }

        static size_t GetNumElementsPerBuckets(int currNumElementsPerBuckets, size_t numRows)
        {
            if (currNumElementsPerBuckets != -1) return currNumElementsPerBuckets;
            return numRows;
        }

        static size_t GetTopK(int currNumElementsPerBuckets, int m_topK)
        {
            if (m_topK == -1) return currNumElementsPerBuckets;
            return m_topK;
        }

        // Aggregate the gradient matrices across all nodes
        bool AggregateGradients(const std::vector<Matrix<ElemType>*>& gradients, DistGradHeader* headerCPU, bool resetState) override
        {
            if (m_mpi->NumNodesInUse() == 1) // No need to aggregate anything.
                return (headerCPU->numSamples != 0);


            // Initialize NCCL
            if (m_nccl == nullptr)
                m_nccl.reset(new NcclComm(::CNTK::DeviceDescriptor::UseDefaultDevice().Id(), m_mpi));

            ResetState(gradients, headerCPU->numEvalNode, resetState);
            bool showSyncPerfStats = (m_syncStatsTrace > 0) && ((m_iterationCount % m_syncStatsTrace) == 0);
            m_iterationCount++;

            if (m_useAsyncAggregation)
            {
                // If we are performing async gradient aggregation, let's wait for the pending gradient aggregation to finish
                // then swap the contents of the buffered gradients and the new gradient matrices and fire an async aggreagation
                // of the new gradient matrices
                if (m_pendingAsyncAggregation.valid())
                {
                    Timer aggregationTimer;
                    if (showSyncPerfStats)
                        aggregationTimer.Start();

                    m_pendingAsyncAggregation.get();

                    if (showSyncPerfStats)
                    {
                        aggregationTimer.Stop();
                        double gradientAggregationTime = aggregationTimer.ElapsedSeconds();
                        fprintf(stderr, "Async gradient aggregation wait time: %.6g\n", gradientAggregationTime);
                    }
                }

                std::vector<Matrix<ElemType>*> newGradients;
                size_t numGradMatrices = gradients.size();
                for (size_t i = 0; i < numGradMatrices; i++)
                {
                    Matrix<ElemType>* bufferedGradientMatrix = m_bufferedGradients[gradients[i]].get();
                    if ((bufferedGradientMatrix == nullptr) ||
                            (bufferedGradientMatrix->GetNumCols() != gradients[i]->GetNumCols()) ||
                            (bufferedGradientMatrix->GetNumRows() != gradients[i]->GetNumRows()) ||
                            (bufferedGradientMatrix->GetDeviceId() != gradients[i]->GetDeviceId()))
                    {
                        LogicError("No buffered gradient matrix found corresponding to a gradient matrix to be aggregated!");
                    }

                    // Swap the gradient matrix contents with the buffered matrices
                    std::swap(*(gradients[i]), *bufferedGradientMatrix);

                    newGradients.push_back(bufferedGradientMatrix);
                }

                // Swap the grad header contents with the buffered grad header
                swap(*headerCPU, *m_bufferedGradHeader);

                // Initiate aggregation only if any samples were processed in previous iteration
                if (resetState || (headerCPU->numSamples != 0))
                {
                    int deviceId = gradients[0]->GetDeviceId();
                    DistGradHeader* newGradHeader = m_bufferedGradHeader;

                    // Since we will be aggregating the gradients assynchronously, let us
                    // ensure that the gradient matrices have been computed before starting to aggregate
                    // them asynchronously on another thread. This essentially means that when we are using
                    // a GPU device, we will synchronize on the main GPU compute stream before starting
                    // the gradient aggregation asynchronously on a separate stream
                    MatrixComputeStreamEvent* mainStreamSyncEvent = MatrixComputeStreamEvent::Create(deviceId);

                    m_pendingAsyncAggregation = std::async(std::launch::async, [=] {
                            // We are starting on a new thread. Make sure the new thread is
                            // setup to use the right device
                            Matrix<ElemType>::SetDevice(deviceId);

                            // Synchronize the Quantization compute stream with the completion of
                            // compute of the gradient matrices on the main compute stream
                            mainStreamSyncEvent->SynchronizeDataTransferFetchStreamWithEvent<ElemType>();
                            delete mainStreamSyncEvent;

                            AggregateGradientsImpl(newGradients, newGradHeader, showSyncPerfStats);
                            });

                    return true;
                }

                return false;
            }
            else
            {
                AggregateGradientsImpl(gradients, headerCPU, showSyncPerfStats);
                return (headerCPU->numSamples != 0);
            }
        }

        private:
        std::shared_ptr<ElemType> AllocateIntermediateBuffer(int deviceID, size_t numElements)
        {
            assert(deviceID >= 0);

            // Use pinned memory for GPU devices for better copy performance
            size_t totalSize = sizeof(ElemType) * numElements;
            return std::shared_ptr<ElemType>((ElemType*) m_allocator->Malloc(totalSize), [this, deviceID](ElemType* p)
                    {
                    m_allocator->Free(p);
                    });
        }

        bool ShouldCopyDataToCPU(int deviceId)
        {
            // Do not copy if data is on CPU
            if (deviceId == CPUDEVICE)
                return false;

            // Do not copy if NCCL is supported or GPUDirect RDMA is used
            if (m_nccl->IsSupported() || m_mpi->UseGpuGdr() == true)
                return false;

            return true;
        }

        void ResetState(const std::vector<Matrix<ElemType>*>& gradients, int numEvalNodes, bool resetState)
        {
            // When called the first time let's setup the intermediateCPU buffers for gradient aggregation if needed
            if (!m_initialized)
            {
                m_initialized = true;
                int deviceId = gradients[0]->GetDeviceId();

                // Initial preparation for data copy from GPU to CPU
                if (ShouldCopyDataToCPU(deviceId))
                {
                    m_allocator.reset(new CUDAPageLockedMemAllocator(deviceId));
                }

                size_t packedGradientsSizeInElements = 0;
                for (size_t i = 0; i < gradients.size(); i++)
                {
                    if (!m_useAsyncAggregation && sizeof(ElemType) * gradients[i]->GetNumElements() <= m_packThresholdSizeInBytes)
                    {
                        packedGradientsSizeInElements += gradients[i]->GetNumElements();
                        m_packedGradientsIndex.push_back(i);
                    }
                    else
                    {
                        m_gradientIndexToAggregate.push_back(i);
                    }

                    // Make sure none of the gradient matrixes are sparse - we currently do not support aggregation of sparse gradient matrices
                    if (gradients[i]->GetMatrixType() != DENSE)
                        RuntimeError("Gradient aggregation for sparse gradient matrices is currently unsupported!");

                    if (m_useAsyncAggregation)
                        m_bufferedGradients[gradients[i]].reset(new Matrix<ElemType>(gradients[i]->GetNumRows(), gradients[i]->GetNumCols(), deviceId));
                }

                // Packing matrices into continous buffer if not doing async aggregation
                m_aggregationBuffer.reset();
                if (packedGradientsSizeInElements > 0)
                {
                    m_aggregationBuffer.reset(new (std::nothrow) Matrix<ElemType>(1, packedGradientsSizeInElements, deviceId));
                }
                // If no extra continous buffer allocated or using async aggregation
                if (m_aggregationBuffer == nullptr)
                {
                    m_gradientIndexToAggregate.clear();
                    m_packedGradientsIndex.clear();
                    packedGradientsSizeInElements = 0;
                    // Reuse "@param m_gradientIndexToAggregate" for following code, if no continous buffer allocated
                    for (size_t i = 0; i < gradients.size(); i++)
                    {
                        m_gradientIndexToAggregate.push_back(i);
                    }
                }
                else
                {
                    // First element is reserved for continous buffer
                    m_gradientIndexToAggregate.insert(m_gradientIndexToAggregate.begin(), 1, (size_t)-1);
                }

                if (ShouldCopyDataToCPU(deviceId))
                {
                    size_t topK = GetTopK(DEFAULT_BUCKET_SIZE, m_topK);
                    if ((sizeof(unsigned) + sizeof(ElemType)) * topK >= DEFAULT_BUCKET_SIZE * sizeof(ElemType))
                    {
                        // NO TOPK
                        for (size_t i : m_gradientIndexToAggregate)
                        {
                            m_gpuDataTransferers.push_back(std::make_unique<GPUDataTransferer>(deviceId, m_useAsyncAggregation));
                            m_intermediateCPUBuffers.push_back(AllocateIntermediateBuffer(deviceId,
                                        (i == -1) ? packedGradientsSizeInElements : gradients[i]->GetNumElements()));
                        }
                    }
                    else
                    {
                        if(m_gradientIndexToAggregate[0] == -1) {
                            m_gpuDataTransferer = std::make_unique<GPUDataTransferer>(deviceId, m_useAsyncAggregation);
                            m_intermediateCPUBuffer = AllocateIntermediateBuffer(deviceId, packedGradientsSizeInElements);
                        }

                        // WITH TOPK
                        int cnt = 0;
                        for (size_t i : m_gradientIndexToAggregate)
                        {
                            if(i == -1) continue;

                             Matrix<ElemType>* gpuCopyBuffer = gradients[i];

                            size_t nRow = gpuCopyBuffer->GetNumRows();
                            size_t nCol = gpuCopyBuffer->GetNumCols();
                            size_t dim = nRow * nCol;

#if defined(_MSC_VER)
                            topK = GetTopK(DEFAULT_BUCKET_SIZE, m_topK);
#else
                            size_t topK = GetTopK(DEFAULT_BUCKET_SIZE, m_topK);
#endif
                            size_t numBuckets = dim / DEFAULT_BUCKET_SIZE;

                            m_preAggGradQuantizers.push_back(std::unique_ptr<MatrixQuantizerImpl<ElemType>>(MatrixQuantizerImpl<ElemType>::Create(deviceId, m_useAsyncAggregation)));
                            m_residuals.push_back(std::make_shared<Matrix<ElemType>>(nRow, nCol, deviceId, DENSE));

                            m_sendbufs.push_back(std::unique_ptr<MyStreamAlloc<ElemType>>(new MyStreamAlloc<ElemType>(m_allocator.get(), sizeof(unsigned) + topK * numBuckets * (sizeof(unsigned) + sizeof(ElemType)))));
                            m_sendbufs[cnt]->m_buffer->nofitems = topK * numBuckets;

                            m_recvbufs.push_back(std::unique_ptr<MyStreamAlloc<ElemType>>(new MyStreamAlloc<ElemType>(m_allocator.get(), sizeof(unsigned) + (dim * sizeof(ElemType)))));

                            cnt++;
                        }
                    }
                }

                if (m_useAsyncAggregation)
                {
                    m_bufferedGradHeader = DistGradHeader::Create(numEvalNodes);
                    m_bufferedGradHeader->Clear();
                }

                if (m_mpi->IsMainNode())
                {
                    for (size_t i = 0; i < NumProc() - 1; ++i)
                        m_recvHeaders.push_back(DistGradHeader::Create(numEvalNodes));
                }
            }
            else if (resetState)
            {
                // Make sure there is no pending async aggregation
                if (m_useAsyncAggregation && m_pendingAsyncAggregation.valid())
                    LogicError("Unexpected pending async gradient aggregation found when resetting aggregator state!");

                for (size_t i = 0; i < m_residuals.size(); ++i)
                    m_residuals[i]->SetValue(0.0);

                // Zero out the buffered gradients if resetting state
                if (m_useAsyncAggregation)
                {
                    for (size_t i = 0; i < gradients.size(); i++)
                        m_bufferedGradients[gradients[i]]->SetValue(0);

                    m_bufferedGradHeader->Clear();
                }
            }
        }

        void AggregateGradientsImpl(const std::vector<Matrix<ElemType>*>& gradients, DistGradHeader* headerCPU, bool showSyncPerfStats)
        {
            Timer aggregationTimer;
            int deviceId = gradients[0]->GetDeviceId();
            if (showSyncPerfStats)
            {
                std::unique_ptr<MatrixComputeStreamEvent> mainStreamSyncEvent(MatrixComputeStreamEvent::Create(deviceId));
                mainStreamSyncEvent->SynchronizeEvent();
                aggregationTimer.Start();
            }

            size_t numGradMatrices = gradients.size();

            if (headerCPU->numSamples == 0)
            {
                assert(headerCPU->criterion == 0.0);
                assert(headerCPU->numSamplesWithLabel == 0);
                for (int i = 0; i < headerCPU->numEvalNode; ++i)
                    assert(headerCPU->evalErrors[i].first == 0 && headerCPU->evalErrors[i].second == 0);

                // If the current node did not process any samples, the gradients should be zero'd
                for (size_t i = 0; i < numGradMatrices; ++i)
                    gradients[i]->SetValue(0);

                if (m_useAsyncAggregation)
                {
                    std::unique_ptr<MatrixComputeStreamEvent> mainStreamSyncEvent(MatrixComputeStreamEvent::Create(deviceId));
                    mainStreamSyncEvent->SynchronizeDataTransferFetchStreamWithEvent<ElemType>();
                }
            }

            // Copy all gradient data into a single contiguous buffer, if additional continous buffer allocated
            size_t offset = 0;
            for (size_t i : m_packedGradientsIndex)
            {
                m_aggregationBuffer->ColumnSlice(offset, gradients[i]->GetNumElements()).AssignValuesOf(gradients[i]->Reshaped(1, gradients[i]->GetNumElements()));
                offset += gradients[i]->GetNumElements();
            }

            // Initiate receive of the header on the main node
            std::vector<MPI_Request> recvHeaderRequests(NumProc() - 1);
            if (m_mpi->IsMainNode())
            {
                for (size_t j = 0; j < NumProc() - 1; ++j)
                {
                    int source = (j >= MyRank()) ? (j + 1) : j;
                    // We use a tag of 'numGradMatrices' for the pre-aggregation header
                    m_mpi->Irecv(m_recvHeaders[j], m_recvHeaders[j]->Size(), MPI_CHAR, source, numGradMatrices, &(recvHeaderRequests[j])) || MpiFail("MPI_Irecv");
                }
            }

            // Send the headers from all nodes but the main node
            MPI_Request sendHeaderRequest;
            if (!m_mpi->IsMainNode())
                m_mpi->Isend(headerCPU, headerCPU->Size(), MPI_CHAR, m_mpi->MainNodeRank(), numGradMatrices, &sendHeaderRequest) || MpiFail("MPI_Isend");


            // New aggregation pipeline for non-GDR, perform sync allreduce on the gradient data
            // For CPU, still use async allreduce
            std::vector<MPI_Request> allReduceRequests;
            size_t allReduceIndex = 0;
            size_t gpuToCpuIndex = 0;
            size_t cpuToGpuIndex = 0;
            size_t numGradientIndex = m_gradientIndexToAggregate.size();
            if (numGradientIndex > 0)
            {
                // non-GDR && GPU && non-NCCL: need to copy data from GPU to CPU
                if ((m_mpi->UseGpuGdr() == 0) && (deviceId != CPUDEVICE) && !m_nccl->IsSupported())
                {
                    size_t topK = GetTopK(DEFAULT_BUCKET_SIZE, m_topK);
                    if ((sizeof(unsigned) + sizeof(ElemType)) * topK >= DEFAULT_BUCKET_SIZE * sizeof(ElemType))
                    {
                        // NO TOPK

                        Matrix<ElemType>* gpuCopyBuffer = m_aggregationBuffer.get();

                        ElemType* reductionBuffer;
                        // currentGradientIndex will load the index from m_gradientIndexToAggregate
                        size_t currentGradientIndex = m_gradientIndexToAggregate[0];
                        size_t nextGradientIndex = 0; // 0 is for initialization only
                        // Get the first Gradient, and do async D-to-H copy
                        if (currentGradientIndex != -1)
                        {
                            gpuCopyBuffer = gradients[currentGradientIndex];
                        }
                        else
                        {
                            // currentGradientIndex == -1, first element is for packed gradients, which should not be with AsyncAggregation
                            assert(m_useAsyncAggregation == false);
                        }
                        // First sync_g_to_c_copy
                        // TODO: we need a CopyGPUToCPUSync
#ifndef CPUONLY
                        cudaMemcpy(m_intermediateCPUBuffers[gpuToCpuIndex].get(), gpuCopyBuffer->Data(), gpuCopyBuffer->GetNumElements() * sizeof(ElemType), cudaMemcpyDeviceToHost);
#endif
                        gpuToCpuIndex++;

                        for (size_t i = 1; i <= numGradientIndex; i ++)
                        {
                            // Get next gradient
                            if (i < numGradientIndex)
                            {
                                nextGradientIndex = m_gradientIndexToAggregate[i];
                                if (nextGradientIndex != -1)
                                {
                                    gpuCopyBuffer = gradients[nextGradientIndex];
                                }
                                else
                                {
                                    // currentGradientIndex == -1, first element is for packed gradients, which should not be with AsyncAggregation
                                    assert(m_useAsyncAggregation == false);
                                }
                                // Async D-to-H copy (next gradient)
                                m_gpuDataTransferers[gpuToCpuIndex]->CopyGPUToCPUAsync(gpuCopyBuffer->Data(), gpuCopyBuffer->GetNumElements(), m_intermediateCPUBuffers[gpuToCpuIndex].get());
                            }
                            // Wait for previous copy
                            m_gpuDataTransferers[allReduceIndex]->WaitForCopyGPUToCPUAsync();

                            // Allreduce
                            reductionBuffer = m_intermediateCPUBuffers[allReduceIndex].get();
                            m_mpi->AllReduce(reductionBuffer, (currentGradientIndex == -1) ? m_aggregationBuffer->GetNumElements() : gradients[currentGradientIndex]->GetNumElements());

                            // Create async H-to-G copy
                            cpuToGpuIndex = allReduceIndex;
                            m_gpuDataTransferers[cpuToGpuIndex]->CopyCPUToGPUAsync(m_intermediateCPUBuffers[cpuToGpuIndex].get(),
                                    (currentGradientIndex == -1) ? m_aggregationBuffer->GetNumElements() : gradients[currentGradientIndex]->GetNumElements(),
                                    (currentGradientIndex == -1) ? m_aggregationBuffer->Data() : gradients[currentGradientIndex]->Data());
                            allReduceIndex = gpuToCpuIndex;
                            gpuToCpuIndex ++;
                            currentGradientIndex = nextGradientIndex;
                        }

                    }
                    else
                    {
                        // WITH TOPK

                        Matrix<ElemType>* gpuCopyBuffer = m_aggregationBuffer.get();
                        size_t currentGradientIndex = m_gradientIndexToAggregate[0];
                        // Check if first is aggregated buffer
                        size_t i = 0;
                        if (currentGradientIndex == -1) {
                            i++;
                            cudaMemcpy(m_intermediateCPUBuffer.get(), gpuCopyBuffer->Data(), gpuCopyBuffer->GetNumElements() * sizeof(ElemType), cudaMemcpyDeviceToHost);
                        }

                        size_t nextGradientIndex  = 0; // Only for initialization
                        if(i < numGradientIndex) {
                            // Start async copy of next gradient
                            nextGradientIndex = m_gradientIndexToAggregate[i];
                            if (nextGradientIndex == -1)
                            {
                                RuntimeError("Unallowed.");
                            }

#if defined(_MSC_VER)
                            topK = GetTopK(DEFAULT_BUCKET_SIZE, m_topK);
#else
                            size_t topK = GetTopK(DEFAULT_BUCKET_SIZE, m_topK);
#endif
                            m_preAggGradQuantizers[gpuToCpuIndex]->TopKAsync(*(gradients[nextGradientIndex]), *m_residuals[gpuToCpuIndex], *(m_sendbufs[gpuToCpuIndex]->m_buffer), *m_residuals[gpuToCpuIndex], topK);
                        }

                        if (currentGradientIndex == -1) {
                            // Allreduce
                            ElemType* reductionBuffer = m_intermediateCPUBuffer.get();
                            m_mpi->AllReduce(reductionBuffer, m_aggregationBuffer->GetNumElements());

                            // Async copy back
                            m_gpuDataTransferer->CopyCPUToGPUAsync(m_intermediateCPUBuffer.get(),
                                    m_aggregationBuffer->GetNumElements(), 
                                    m_aggregationBuffer->Data());
                            if(i < numGradientIndex) {
                                currentGradientIndex = m_gradientIndexToAggregate[i];
                            }
                        }

                        gpuToCpuIndex++;
                        i++;

                        for(; i <= numGradientIndex; ++i) {
                            // Get next gradient
                            if (i < numGradientIndex)
                            {
                                nextGradientIndex = m_gradientIndexToAggregate[i];
                                if (nextGradientIndex == -1)
                                {
                                    RuntimeError("Unallowed.");
                                }
                                // Async D-to-H copy (next gradient)
#if defined(_MSC_VER)
                                topK = GetTopK(DEFAULT_BUCKET_SIZE, m_topK);
#else
                                size_t topK = GetTopK(DEFAULT_BUCKET_SIZE, m_topK);
#endif
                                m_preAggGradQuantizers[gpuToCpuIndex]->TopKAsync(*(gradients[nextGradientIndex]), *m_residuals[gpuToCpuIndex], *(m_sendbufs[gpuToCpuIndex]->m_buffer), *m_residuals[gpuToCpuIndex], topK);
                            }

                            m_preAggGradQuantizers[allReduceIndex]->WaitTopKAsyncDone();

                            size_t nRow = gradients[currentGradientIndex]->GetNumRows();
                            size_t nCol = gradients[currentGradientIndex]->GetNumCols();
                            size_t dim = nRow * nCol;

                            // ReC: TODO Sparse AllReduce
                            c_allreduce_ring<unsigned, ElemType>(m_sendbufs[allReduceIndex]->m_buffer, m_recvbufs[allReduceIndex]->m_buffer, dim);

                            // Create async H-to-G copy
                            cpuToGpuIndex = allReduceIndex;
                            m_preAggGradQuantizers[cpuToGpuIndex]->UnTopKAsync(*(m_recvbufs[cpuToGpuIndex]->m_buffer), *(gradients[currentGradientIndex]));
                            allReduceIndex = gpuToCpuIndex;
                            gpuToCpuIndex++;
                            currentGradientIndex = nextGradientIndex;
                        }

                    }
                }
                // non-NCCL, using CPU, using GDR
                else if (!m_nccl->IsSupported())
                {
                    ElemType* reductionBuffer;
                    for (size_t i : m_gradientIndexToAggregate)
                    {
                        allReduceRequests.push_back(MPI_Request());
                        reductionBuffer = (i == -1)? m_aggregationBuffer->Data() : gradients[i]->Data();
                        // CPU
                        if (m_mpi->UseGpuGdr() == 0)
                        {
                            m_mpi->Iallreduce(MPI_IN_PLACE, reductionBuffer, (i == -1) ? m_aggregationBuffer->GetNumElements() : gradients[i]->GetNumElements(),
                                    MPIWrapper::GetDataType(reductionBuffer), MPI_SUM, &allReduceRequests.back()) || MpiFail("MPI_Iallreduce");
                            allReduceIndex++;
                        }
                        // GDR && GPU
                        else if (deviceId != CPUDEVICE)
                        {
                            m_mpi->AllReduce(reductionBuffer, (i == -1) ? m_aggregationBuffer->GetNumElements() : gradients[i]->GetNumElements());
                        }
                    }
                } 
                else if (m_nccl->IsSupported())
                {
                    std::vector<Matrix<ElemType>*> ncclReduceGradients;
                    for (size_t i : m_gradientIndexToAggregate)
                    {
                        ncclReduceGradients.push_back((i == -1) ? m_aggregationBuffer.get() : gradients[i]);
                    }
                    m_nccl->AllReduce(ncclReduceGradients);
                }
            }

            // On the main node wait for the headers to arrive and aggregate
            if (m_mpi->IsMainNode())
            {
                size_t numNodesHeadersReceivedFrom = 0;
                while (numNodesHeadersReceivedFrom < (NumProc() - 1))
                {
                    int idx = MPI_UNDEFINED;
                    m_mpi->Waitany(recvHeaderRequests.size(), recvHeaderRequests.data(), &idx, MPI_STATUS_IGNORE) || MpiFail("MPI_Waitany");
                    if (idx == MPI_UNDEFINED)
                    {
                        break;
                    }

                    numNodesHeadersReceivedFrom++;

                    headerCPU->Aggregate(m_recvHeaders[idx], true);
                }

                assert(numNodesHeadersReceivedFrom == (NumProc() - 1));
            }

            // Broadcast the aggregated header to all nodes
            m_mpi->Bcast(headerCPU, headerCPU->Size(), MPI_CHAR, m_mpi->MainNodeRank());

            if (m_nccl->IsSupported())
            {
                m_nccl->Sync();
            }
            // Non-GDR && GPU
            else if ((m_mpi->UseGpuGdr() == 0) && (deviceId != CPUDEVICE))
            {
                size_t topK = GetTopK(DEFAULT_BUCKET_SIZE, m_topK);
                if ((sizeof(unsigned) + sizeof(ElemType)) * topK >= DEFAULT_BUCKET_SIZE * sizeof(ElemType))
                {
                    // NO TOPK

                    // Wait for async CPU-to-GPU copy (non-GDR)
                    for (size_t i = 0; i < allReduceIndex; i++)
                        m_gpuDataTransferers[i]->WaitForCopyCPUToGPUAsync();
                }
                else
                {
                    // WITH TOPK 
                    
                    // Wait for async CPU-to-GPU copy (non-GDR)
                    for (size_t i = 0; i < allReduceIndex; i++)
                        m_preAggGradQuantizers[i]->WaitUnTopKAsyncDone();
                }
            }
            // CPU
            else if (m_mpi->UseGpuGdr() == 0)
            {
                // Wait for the Iallreduce operations to finish
                for (size_t i = 0; i < allReduceIndex; i++)
                {
                    m_mpi->Wait(&allReduceRequests[i], MPI_STATUSES_IGNORE) || MpiFail("MPI_Wait");
                }
            }

            // Copy data back to the packed gradients from the continous buffer
            offset = 0;
            for (size_t i : m_packedGradientsIndex)
            {
                gradients[i]->AssignValuesOf(m_aggregationBuffer->ColumnSlice(offset, gradients[i]->GetNumElements()).Reshaped(gradients[i]->GetNumRows(), gradients[i]->GetNumCols()));
                offset += gradients[i]->GetNumElements();
            }

            // Wait for completion of the async send requests
            if (!m_mpi->IsMainNode())
                m_mpi->Wait(&sendHeaderRequest, MPI_STATUSES_IGNORE) || MpiFail("MPI_Wait");

            if (showSyncPerfStats)
            {
                aggregationTimer.Stop();
                double gradientAggregationTime = aggregationTimer.ElapsedSeconds();
                fprintf(stderr, "Actual gradient aggregation time: %.6g\n", gradientAggregationTime);
            }
        }

        private:
        std::unique_ptr<CUDAPageLockedMemAllocator> m_allocator;

        std::vector<std::unique_ptr<MatrixQuantizerImpl<ElemType>>> m_preAggGradQuantizers;
        std::vector<std::shared_ptr<Matrix<ElemType>>> m_residuals;
        std::vector<std::unique_ptr<MyStreamAlloc<ElemType>>> m_sendbufs;
        std::vector<std::unique_ptr<MyStreamAlloc<ElemType>>> m_recvbufs;

        std::shared_ptr<ElemType> m_intermediateCPUBuffer;
        std::unique_ptr<GPUDataTransferer> m_gpuDataTransferer;

        std::vector<std::shared_ptr<ElemType>> m_intermediateCPUBuffers;
        std::vector<std::unique_ptr<GPUDataTransferer>> m_gpuDataTransferers;

        std::vector<DistGradHeader*> m_recvHeaders;

        // Perform aysnchronous gradient aggregation using double buffering of the gradient matrices
        bool m_useAsyncAggregation;
        int m_topK;

        // Future corresponding to the current in-flight async gradient aggregation
        std::future<void> m_pendingAsyncAggregation;

        // Buffered gradients that we asynchronously aggregate
        std::unordered_map<Matrix<ElemType>*, std::unique_ptr<Matrix<ElemType>>> m_bufferedGradients;
        DistGradHeader* m_bufferedGradHeader;

        // Packing small gradients (size not larger than threshold size) into a continous buffer to reduce MPI calls.
        // Threshold size to pack a gradient into the continous buffer, default 32KB (tunable by define "packThresholdSizeInKB=[value]")
        const size_t m_packThresholdSizeInBytes;
        std::unique_ptr<Matrix<ElemType>> m_aggregationBuffer;
        std::vector<size_t> m_packedGradientsIndex;
        std::vector<size_t> m_gradientIndexToAggregate;

        int m_syncStatsTrace;

        // Only used for controlling frequency of measuring/showing gradient aggregation perf stats
        size_t m_iterationCount;

        bool m_initialized;

        std::unique_ptr<NcclComm> m_nccl;
    };
} } }
