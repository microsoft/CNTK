#pragma once

#include "IDistGradAggregator.h"
#include "CUDAPageLockedMemAllocator.h"
#include <future>
#include "GPUDataTransferer.h"
#include "TimerUtility.h"
#include "MatrixQuantizerImpl.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class SimpleDistGradAggregator : public IDistGradAggregator<ElemType>
{
    UsingIDistGradAggregatorMembers;

public:
    SimpleDistGradAggregator(const MPIWrapperPtr& mpi, bool useAsyncAggregation, int syncStatsTrace)
        : IDistGradAggregator<ElemType>(mpi), m_useAsyncAggregation(useAsyncAggregation), m_initialized(false), m_bufferedGradHeader(nullptr), m_syncStatsTrace(syncStatsTrace), m_iterationCount(0)
    {}

    ~SimpleDistGradAggregator()
    {
        for (size_t i = 0; i < m_recvHeaders.size(); ++i)
            DistGradHeader::Destroy(m_recvHeaders[i]);

        if (m_bufferedGradHeader != nullptr)
            DistGradHeader::Destroy(m_bufferedGradHeader);
    }

    // Aggregate the gradient matrices across all nodes
    bool AggregateGradients(const std::vector<Matrix<ElemType>*>& gradients, DistGradHeader* headerCPU, bool resetState) override
    {
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

    void ResetState(const std::vector<Matrix<ElemType>*>& gradients, int numEvalNodes, bool resetState)
    {
        // When called the first time let's setup the intermediateCPU buffers for gradient aggregation if needed
        if (!m_initialized)
        {
            m_initialized = true;
            int deviceId = gradients[0]->GetDeviceId();
            if (deviceId != CPUDEVICE)
                m_allocator.reset(new CUDAPageLockedMemAllocator(deviceId));

            for (size_t i = 0; i < gradients.size(); i++)
            {
                // Make sure none of the gradient matrixes are sparse - we currently do not support aggregation of sparse gradient matrices
                if (gradients[i]->GetMatrixType() != DENSE)
                    RuntimeError("Gradient aggregation for sparse gradient matrices is currently unsupported!");

                if (deviceId != CPUDEVICE)
                {
                    m_gpuDataTransferers.push_back(std::unique_ptr<GPUDataTransferer<ElemType>>(new GPUDataTransferer<ElemType>(deviceId, m_useAsyncAggregation)));
                    m_intermediateCPUBuffers.push_back(AllocateIntermediateBuffer(deviceId, gradients[i]->GetNumElements()));
                }

                if (m_useAsyncAggregation)
                    m_bufferedGradients[gradients[i]].reset(new Matrix<ElemType>(gradients[i]->GetNumRows(), gradients[i]->GetNumCols(), deviceId));
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

        // Initiate transfer of the gradient matrices to the CPU if needed
        if (deviceId >= 0)
        {
            for (size_t i = 0; i < numGradMatrices; ++i)
                m_gpuDataTransferers[i]->CopyGPUToCPUAsync(gradients[i]->Data(), gradients[i]->GetNumElements(), m_intermediateCPUBuffers[i].get());
        }

        // Initiate receive of the header on the main node
        std::vector<MPI_Request> recvHeaderRequests(NumProc() - 1);
        if (m_mpi->IsMainNode())
        {
            for (size_t j = 0; j < NumProc() - 1; ++j)
            {
                int source = (j >= MyRank()) ? (j + 1) : j;
                // We use a tag of 'numGradMatrices' for the pre-aggregation header
                MPI_Irecv(m_recvHeaders[j], m_recvHeaders[j]->Size(), MPI_CHAR, source, numGradMatrices, m_mpi->Communicator(), &(recvHeaderRequests[j])) || MpiFail("MPI_Irecv");
            }
        }

        // Send the headers from all nodes but the main node
        MPI_Request sendHeaderRequest;
        if (!m_mpi->IsMainNode())
            MPI_Isend(headerCPU, headerCPU->Size(), MPI_CHAR, m_mpi->MainNodeRank(), numGradMatrices, m_mpi->Communicator(), &sendHeaderRequest) || MpiFail("MPI_Isend");

        // Perform MPI async allreduce on the gradient data
        std::vector<MPI_Request> allReduceRequests(numGradMatrices);
        for (size_t i = 0; i < numGradMatrices; ++i)
        {
            ElemType* reductionBuffer = gradients[i]->Data();
            if (deviceId >= 0)
            {
                m_gpuDataTransferers[i]->WaitForCopyGPUToCPUAsync();
                reductionBuffer = m_intermediateCPUBuffers[i].get();
            }

            // On Windows this async MPI_Iallreduce call requires MS MPI v7 or higher to be installed
            MPI_Iallreduce(MPI_IN_PLACE, reductionBuffer, gradients[i]->GetNumElements(), MPIWrapper::GetDataType(reductionBuffer), MPI_SUM, m_mpi->Communicator(), &allReduceRequests[i]) || MpiFail("MPI_Iallreduce");
        }

        // On the main node wait for the headers to arrive and aggregate
        if (m_mpi->IsMainNode())
        {
            size_t numNodesHeadersReceivedFrom = 0;
            while (numNodesHeadersReceivedFrom < (NumProc() - 1))
            {
                int idx = MPI_UNDEFINED;
                MPI_Waitany(recvHeaderRequests.size(), recvHeaderRequests.data(), &idx, MPI_STATUS_IGNORE) || MpiFail("MPI_Waitany");
                if (idx == MPI_UNDEFINED)
                {
                    break;
                }

                numNodesHeadersReceivedFrom++;

                headerCPU->Aggregate(m_recvHeaders[idx], true);
            }

            assert(numNodesHeadersReceivedFrom == (NumProc() - 1));
        }

        // Initiate receive of the aggregate header
        MPI_Request recvAggHeaderRequest;
        if (!m_mpi->IsMainNode())
            MPI_Irecv(headerCPU, headerCPU->Size(), MPI_CHAR, m_mpi->MainNodeRank(), numGradMatrices + 1 + numGradMatrices, m_mpi->Communicator(), &recvAggHeaderRequest) || MpiFail("MPI_Irecv");

        // Intiate send of the aggregate header from main node
        std::vector<MPI_Request> sendAggHeaderRequests(NumProc() - 1);
        if (m_mpi->IsMainNode())
        {
            for (size_t j = 0; j < NumProc() - 1; ++j)
            {
                int dest = (j >= MyRank()) ? (j + 1) : j;
                // TODO: Should we use MPI_Bcast instead for better performance
                MPI_Isend(headerCPU, headerCPU->Size(), MPI_CHAR, dest, numGradMatrices + 1 + numGradMatrices, m_mpi->Communicator(), &(sendAggHeaderRequests[j])) || MpiFail("MPI_Isend");
            }
        }

        // Wait for the allreduce operations to finish and initiate transfer back to the GPU if needed
        for (size_t i = 0; i < numGradMatrices; ++i)
        {
            MPI_Wait(&allReduceRequests[i], MPI_STATUSES_IGNORE) || MpiFail("MPI_Wait");
            if (deviceId >= 0)
                m_gpuDataTransferers[i]->CopyCPUToGPUAsync(m_intermediateCPUBuffers[i].get(), gradients[i]->GetNumElements(), gradients[i]->Data());
        }

        // Wait to receive aggregate header
        if (!m_mpi->IsMainNode())
            MPI_Wait(&recvAggHeaderRequest, MPI_STATUSES_IGNORE) || MpiFail("MPI_Wait");

        // Wait for all the transfers to finish
        if (deviceId >= 0)
        {
            for (size_t i = 0; i < numGradMatrices; ++i)
                m_gpuDataTransferers[i]->WaitForCopyCPUToGPUAsync();
        }

        // Wait for completion of the async send requests
        if (!m_mpi->IsMainNode())
            MPI_Wait(&sendHeaderRequest, MPI_STATUSES_IGNORE) || MpiFail("MPI_Wait");
        else
            MPI_Waitall(sendAggHeaderRequests.size(), sendAggHeaderRequests.data(), MPI_STATUSES_IGNORE) || MpiFail("MPI_Waitall");

        if (showSyncPerfStats)
        {
            aggregationTimer.Stop();
            double gradientAggregationTime = aggregationTimer.ElapsedSeconds();
            fprintf(stderr, "Actual gradient aggregation time: %.6g\n", gradientAggregationTime);
        }
    }

private:
    std::unique_ptr<CUDAPageLockedMemAllocator> m_allocator;
    std::vector<std::shared_ptr<ElemType>> m_intermediateCPUBuffers;

    std::vector<std::unique_ptr<GPUDataTransferer<ElemType>>> m_gpuDataTransferers;

    std::vector<DistGradHeader*> m_recvHeaders;

    // Perform aysnchronous gradient aggregation using double buffering of the gradient matrices
    bool m_useAsyncAggregation;

    // Future corresponding to the current in-flight async gradient aggregation
    std::future<void> m_pendingAsyncAggregation;

    // Buffered gradients that we asynchronously aggregate
    std::unordered_map<Matrix<ElemType>*, std::unique_ptr<Matrix<ElemType>>> m_bufferedGradients;
    DistGradHeader* m_bufferedGradHeader;

    int m_syncStatsTrace;

    // Only used for controlling frequency of measuring/showing gradient aggregation perf stats
    size_t m_iterationCount;

    bool m_initialized;
};
} } }
