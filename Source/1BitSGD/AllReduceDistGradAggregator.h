//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "IDistGradAggregator.h"
#include "CUDAPageLockedMemAllocator.h"
#include "QuantizedMatrix.h"
#include "MatrixQuantizer.h"
#include "MatrixQuantizerGPU.h"
#include <future>
#include "TimerUtility.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// =======================================================================
// AllReduceDistGradAggregator -- 1-bit SGD.
// This implements
//    Frank Seide, Hao Fu, Jasha Droppo, Gang Li, and Dong Yu:
//    "1-bit stochastic gradient descent and its application to data-parallel distributed training of speech DNNs"
//    In Proc. Interspeech 2014.
// =======================================================================

template <class ElemType>
class AllReduceDistGradAggregator : public IDistGradAggregator<ElemType>
{
    struct Stripe
    {
        size_t m_startCol;
        size_t m_numCols;
    };

    UsingIDistGradAggregatorMembers;

    static const int DEBUG_OUTPUT_TRACE_LEVEL = 3;

public:
    AllReduceDistGradAggregator(const std::shared_ptr<MPIWrapper>& mpi, int nBits, bool zeroThresholdFor1Bit, bool useQuantizationForSelfStripe, bool useAsyncAggregation, int traceLevel, int syncStatsTrace)
        : IDistGradAggregator<ElemType>(mpi), m_numQuantizationBits(nBits), m_zeroThresholdFor1Bit(zeroThresholdFor1Bit), m_useQuantizationForSelfStripe(useQuantizationForSelfStripe),
        m_traceLevel(traceLevel), m_initialized(false), m_useAsyncAggregation(useAsyncAggregation), m_bufferedGradHeader(nullptr), m_syncStatsTrace(syncStatsTrace), m_iterationCount(0)
    {}

    ~AllReduceDistGradAggregator()
    {
        for (size_t i = 0; i < m_recvHeaders.size(); ++i)
            DistGradHeader::Destroy(m_recvHeaders[i]);

        if (m_bufferedGradHeader != nullptr)
            DistGradHeader::Destroy(m_bufferedGradHeader);
    }

    // Gets the range of columns to be processed by the node with the specified rank
    // when parallel processing using 'numNodes' nodes
    static Stripe GetStripeForNode(size_t numCols, size_t nodeRank, size_t numNodes)
    {
        // Determine which stripe of the gradient is this node responsible for
        size_t numColsPerNode = numCols / numNodes;
        size_t residue = numCols % numNodes;
        size_t startColNumofStripe = (numColsPerNode * nodeRank) + min(residue, nodeRank);
        size_t numColsinStripe = numColsPerNode + ((nodeRank < residue) ? 1 : 0);

        return Stripe({startColNumofStripe, numColsinStripe});
    }

    void ResetState(const std::vector<Matrix<ElemType>*>& gradients, int numEvalNodes, bool resetState)
    {
        // When called the first time let's setup the quantizers and matrices for holding quantized values.
        // These can live for the lifetime of the aggregator since the gradient matrix dimensions for learnable parameters
        // do not change
        if (!m_initialized)
        {
            m_initialized = true;
            int deviceId = gradients[0]->GetDeviceId();
            if (deviceId != CPUDEVICE)
                m_allocator.reset(new CUDAPageLockedMemAllocator(deviceId));

            for (size_t i = 0; i < gradients.size(); i++)
            {
                // Make sure none of the gradient matrices are sparse - we currently do not support aggregation of sparse gradient matrices
                if (gradients[i]->GetMatrixType() != DENSE)
                    RuntimeError("Gradient aggregation for sparse gradient matrices is currently unsupported!");

                size_t nRow = gradients[i]->GetNumRows();
                size_t nCol = gradients[i]->GetNumCols();
                m_preAggGradQuantizers.push_back(std::unique_ptr<MatrixQuantizer<ElemType>>(new MatrixQuantizer<ElemType>(nRow, nCol, deviceId, m_useAsyncAggregation)));
                m_gradQuantized.push_back(std::unique_ptr<QuantizedMatrix<ElemType>>(new QuantizedMatrix<ElemType>(nRow, nCol, m_numQuantizationBits, CPUDEVICE, m_allocator.get())));

                // Determine which stripe of the gradient is this node responsible for
                Stripe stripe = GetStripeForNode(nCol, MyRank(), NumProc());

                MatrixQuantizer<ElemType>* currAggGradQuantizer = nullptr;
                std::vector<std::unique_ptr<QuantizedMatrix<ElemType>>> currRecvGradStripesQuantized;
                if (stripe.m_numCols > 0)
                {
                    currAggGradQuantizer = new MatrixQuantizer<ElemType>(nRow, stripe.m_numCols, deviceId, m_useAsyncAggregation);
                    for (size_t j = 0; j < NumProc() - 1; ++j)
                        currRecvGradStripesQuantized.push_back(std::unique_ptr<QuantizedMatrix<ElemType>>(new QuantizedMatrix<ElemType>(nRow, stripe.m_numCols, m_numQuantizationBits, CPUDEVICE, m_allocator.get())));
                }

                m_aggGradStripeQuantizers.push_back(std::unique_ptr<MatrixQuantizer<ElemType>>(currAggGradQuantizer));
                m_recvGradStripesQuantized.push_back(std::move(currRecvGradStripesQuantized));

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
            // If we are resetting state, let's clear previous quantization residues

            // Make sure there is no pending async aggregation
            if (m_useAsyncAggregation && m_pendingAsyncAggregation.valid())
                LogicError("Unexpected pending async gradient aggregation found when resetting aggregator state!");

            for (size_t i = 0; i < m_preAggGradQuantizers.size(); ++i)
                m_preAggGradQuantizers[i]->ResetResidue();

            for (size_t i = 0; i < m_aggGradStripeQuantizers.size(); ++i)
            {
                if (m_aggGradStripeQuantizers[i] != nullptr)
                    m_aggGradStripeQuantizers[i]->ResetResidue();
            }

            // Zero out the buffered gradients if resetting state
            if (m_useAsyncAggregation)
            {
                for (size_t i = 0; i < gradients.size(); i++)
                    m_bufferedGradients[gradients[i]]->SetValue(0);

                m_bufferedGradHeader->Clear();
            }
        }
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

                // Since we will be aggregating the gradients asynchronously, let us
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
                    mainStreamSyncEvent->SynchronizeQuantizationComputeStreamWithEvent<ElemType>();
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
                mainStreamSyncEvent->SynchronizeQuantizationComputeStreamWithEvent<ElemType>();
            }
        }

        std::vector<std::unique_ptr<Matrix<ElemType>>> aggGradStripes;
        std::vector<std::unique_ptr<QuantizedMatrix<ElemType>>> aggGradStripesQuantized;
        for (size_t i = 0; i < gradients.size(); i++)
        {
            size_t nCol = gradients[i]->GetNumCols();

            // Determine which stripe of the gradient is this node responsible for
            Stripe stripe = GetStripeForNode(nCol, MyRank(), NumProc());

            Matrix<ElemType>* currAggGradStripe = nullptr;
            QuantizedMatrix<ElemType>* currAggGradStripeQuantized = nullptr;
            if (stripe.m_numCols > 0)
            {
                currAggGradStripe = new Matrix<ElemType>(gradients[i]->ColumnSlice(stripe.m_startCol, stripe.m_numCols));
                currAggGradStripeQuantized = new QuantizedMatrix<ElemType>(m_gradQuantized[i]->ColumnSlice(stripe.m_startCol, stripe.m_numCols));
            }

            aggGradStripes.push_back(std::unique_ptr<Matrix<ElemType>>(currAggGradStripe));
            aggGradStripesQuantized.push_back(std::unique_ptr<QuantizedMatrix<ElemType>>(currAggGradStripeQuantized));
        }

        // Initiate quantization of the gradient matrices
        for (size_t i = 0; i < numGradMatrices; ++i)
        {
            if (m_traceLevel >= DEBUG_OUTPUT_TRACE_LEVEL)
            {
                char printHeaderBuf[1024];
                sprintf(printHeaderBuf, "MPI Rank: %d, Original Gradient Matrix No. %d", (int) MyRank(), (int) i);
                PrintMatrix(printHeaderBuf, gradients[i]);
            }

            m_preAggGradQuantizers[i]->QuantizeAsync(*(gradients[i]), *(m_gradQuantized[i]), m_zeroThresholdFor1Bit);
        }

        // Initiate receive of the stripe to be aggregated by the current node, from all other nodes
        std::vector<MPI_Request> recvGradStripesQuantizedRequests;
        std::vector<int> recvRequestIdxToGradientMatrixIdxMap;
        for (size_t i = 0; i < numGradMatrices; ++i)
        {
            Stripe stripe = GetStripeForNode(gradients[i]->GetNumCols(), MyRank(), NumProc());
            if (stripe.m_numCols > 0)
            {
                recvRequestIdxToGradientMatrixIdxMap.push_back(i);
                for (size_t j = 0; j < NumProc() - 1; ++j)
                {
                    int source = (j >= MyRank()) ? (j + 1) : j;

                    recvGradStripesQuantizedRequests.push_back(MPI_Request());
                    int recvRequestIdx = recvGradStripesQuantizedRequests.size() - 1;

                    m_mpi->Irecv(m_recvGradStripesQuantized[i][j]->Buffer(), m_recvGradStripesQuantized[i][j]->GetSize(), MPI_CHAR, source, i, &(recvGradStripesQuantizedRequests[recvRequestIdx])) || MpiFail("MPI_Irecv");
                }
            }
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

        // Asynchronously send stripes of the quantized gradient matrices to the respective nodes that own aggregation of that stripe
        std::vector<std::vector<MPI_Request>> sendGradStripesQuantizedRequests(numGradMatrices);
        for (size_t i = 0; i < numGradMatrices; ++i)
        {
            m_preAggGradQuantizers[i]->WaitQuantizeAsyncDone();
            size_t sendRequestIdx = 0;
            for (size_t j = 0; j < NumProc(); ++j)
            {
                Stripe stripe = GetStripeForNode(gradients[i]->GetNumCols(), j, NumProc());
                if (stripe.m_numCols > 0)
                {
                    // Do not send stripe for self
                    if (j != MyRank())
                    {
                        sendGradStripesQuantizedRequests[i].push_back(MPI_Request());
                        QuantizedMatrix<ElemType> quantizedStripe = m_gradQuantized[i]->ColumnSlice(stripe.m_startCol, stripe.m_numCols);
                        if (m_traceLevel >= DEBUG_OUTPUT_TRACE_LEVEL)
                        {
                            char printHeaderBuf[1024];
                            sprintf(printHeaderBuf, "MPI Rank: %d, Sending Gradient Matrix No. %d slice", (int) MyRank(), (int) i);
                            const size_t numRowsToPeek = 3;
                            const size_t numColsToPeek = 3;
                            size_t numRowsToPrint = (std::min)(numRowsToPeek, quantizedStripe.GetNumRows());
                            size_t numColsToPrint = (std::min)(numColsToPeek, quantizedStripe.GetNumCols());

                            quantizedStripe.Print(printHeaderBuf, 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
                        }

                        m_mpi->Isend(quantizedStripe.Buffer(), quantizedStripe.GetSize(), MPI_CHAR, j, i, &(sendGradStripesQuantizedRequests[i][sendRequestIdx])) || MpiFail("MPI_Isend");
                        sendRequestIdx++;
                    }
                    else
                    {
                        // Initialize the aggregate for the stripe with the quantized gradients instead of the original
                        // gradients themselves, if so desired
                        if (m_useQuantizationForSelfStripe)
                        {
                            QuantizedMatrix<ElemType> preAggGradSelfStripeQuantized = m_gradQuantized[i]->ColumnSlice(stripe.m_startCol, stripe.m_numCols);
                            m_aggGradStripeQuantizers[i]->UnquantizeAsync(preAggGradSelfStripeQuantized, *(aggGradStripes[i]), false);
                        }
                    }
                }
            }
        }

        // Send the headers from all nodes but the main node
        MPI_Request sendHeaderRequest;
        if (!m_mpi->IsMainNode())
            m_mpi->Isend(headerCPU, headerCPU->Size(), MPI_CHAR, m_mpi->MainNodeRank(), numGradMatrices, &sendHeaderRequest) || MpiFail("MPI_Isend");

        // Wait for the stripes to arrive from each node and unquantize and aggregate
        size_t numReceivesExpected = recvGradStripesQuantizedRequests.size();
        size_t numActualReceives = 0;
        std::vector<int> perGradMatrixReceiveCount(recvRequestIdxToGradientMatrixIdxMap.size(), 0);
        while (numActualReceives < numReceivesExpected)
        {
            int idx = MPI_UNDEFINED;
            m_mpi->Waitany(recvGradStripesQuantizedRequests.size(), recvGradStripesQuantizedRequests.data(), &idx, MPI_STATUS_IGNORE) || MpiFail("MPI_Waitany");
            if (idx == MPI_UNDEFINED)
            {
                break;
            }

            numActualReceives++;

            int gradMatrixIdxPosition = idx / (NumProc() - 1);
            int recvBufferSubIndex = idx % (NumProc() - 1);
            // Map idx back to the actual gradient matrix index
            int gradMatrixIdx = recvRequestIdxToGradientMatrixIdxMap[gradMatrixIdxPosition];

            // Wait for the previous Unquantize to finish before issuing a new one
            if (m_useQuantizationForSelfStripe || (perGradMatrixReceiveCount[gradMatrixIdxPosition] > 0))
                m_aggGradStripeQuantizers[gradMatrixIdx]->WaitUnquantizeAsyncDone();

            if (m_traceLevel >= DEBUG_OUTPUT_TRACE_LEVEL)
            {
                char printHeaderBuf[1024];
                sprintf(printHeaderBuf, "MPI Rank: %d, Received Gradient Matrix No. %d slice", (int) MyRank(), gradMatrixIdx);
                const size_t numRowsToPeek = 3;
                const size_t numColsToPeek = 3;
                size_t numRowsToPrint = (std::min)(numRowsToPeek, m_recvGradStripesQuantized[gradMatrixIdx][recvBufferSubIndex]->GetNumRows());
                size_t numColsToPrint = (std::min)(numColsToPeek, m_recvGradStripesQuantized[gradMatrixIdx][recvBufferSubIndex]->GetNumCols());

                m_recvGradStripesQuantized[gradMatrixIdx][recvBufferSubIndex]->Print(printHeaderBuf, 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
            }

            m_aggGradStripeQuantizers[gradMatrixIdx]->UnquantizeAsync(*(m_recvGradStripesQuantized[gradMatrixIdx][recvBufferSubIndex]), *(aggGradStripes[gradMatrixIdx]), true);

            perGradMatrixReceiveCount[gradMatrixIdxPosition]++;

            // Also issue the quantization if this stripe was the last one expected for this matrix
            // Note: We issue the quantization without waiting for the unquantization since the same stream
            // is used for both and they are implicitly sequenced
            // We reuse the buffer that we used for quantizing and sending out the pre-aggregation gradient
            if (perGradMatrixReceiveCount[gradMatrixIdxPosition] == (NumProc() - 1))
            {
                Stripe stripe = GetStripeForNode(gradients[gradMatrixIdx]->GetNumCols(), MyRank(), NumProc());
                UNUSED(stripe);
                assert(stripe.m_numCols > 0);
                m_aggGradStripeQuantizers[gradMatrixIdx]->QuantizeAsync(*(aggGradStripes[gradMatrixIdx]), *(aggGradStripesQuantized[gradMatrixIdx]), m_zeroThresholdFor1Bit);
            }
        }

        assert(numActualReceives == numReceivesExpected);

        // On the main node wait for the headers to arrive and aggregate
        if (m_mpi->IsMainNode())
        {
            size_t numNodesHeadersReceivedFrom = 0;
            while (numNodesHeadersReceivedFrom < (NumProc() - 1))
            {
                int idx = MPI_UNDEFINED;
                m_mpi->Waitany(recvHeaderRequests.size(), recvHeaderRequests.data(), &idx, MPI_STATUS_IGNORE) || MpiFail("MPI_Waitany");
                if (idx == MPI_UNDEFINED)
                    break;

                numNodesHeadersReceivedFrom++;

                headerCPU->Aggregate(m_recvHeaders[idx], true);
            }

            assert(numNodesHeadersReceivedFrom == (NumProc() - 1));
        }

        std::vector<std::vector<MPI_Request>> recvAggGradStripesQuantizedRequests(numGradMatrices);
        // Initiate receive of stripes of quantized aggregated gradients from different nodes
        for (size_t i = 0; i < numGradMatrices; ++i)
        {
            size_t recvRequestIdx = 0;
            for (size_t j = 0; j < NumProc(); ++j)
            {
                // Do not recv stripe for self
                if (j != MyRank())
                {
                    Stripe stripe = GetStripeForNode(gradients[i]->GetNumCols(), j, NumProc());
                    if (stripe.m_numCols > 0)
                    {
                        recvAggGradStripesQuantizedRequests[i].push_back(MPI_Request());
                        QuantizedMatrix<ElemType> quantizedStripe = m_gradQuantized[i]->ColumnSlice(stripe.m_startCol, stripe.m_numCols);
                        m_mpi->Irecv(quantizedStripe.Buffer(), quantizedStripe.GetSize(), MPI_CHAR, j, numGradMatrices + 1 + i, &(recvAggGradStripesQuantizedRequests[i][recvRequestIdx])) || MpiFail("MPI_Irecv");
                        recvRequestIdx++;
                    }
                }
            }
        }

        MPI_Request recvAggHeaderRequest;
        // Initiate receive of the aggregate header
        if (!m_mpi->IsMainNode())
            m_mpi->Irecv(headerCPU, headerCPU->Size(), MPI_CHAR, m_mpi->MainNodeRank(), numGradMatrices + 1 + numGradMatrices, &recvAggHeaderRequest) || MpiFail("MPI_Irecv");

        // Initiate broadcast of quantized aggregated gradient stripes to all other nodes
        std::vector<std::vector<MPI_Request>> sendAggGradStripeQuantizedRequests(numGradMatrices);
        for (size_t i = 0; i < numGradMatrices; ++i)
        {
            Stripe stripe = GetStripeForNode(gradients[i]->GetNumCols(), MyRank(), NumProc());
            if (stripe.m_numCols > 0)
            {
                sendAggGradStripeQuantizedRequests[i] = std::vector<MPI_Request>(NumProc() - 1);
                m_aggGradStripeQuantizers[i]->WaitQuantizeAsyncDone();
                for (size_t j = 0; j < NumProc() - 1; ++j)
                {
                    int dest = (j >= MyRank()) ? (j + 1) : j;
                    // TODO: Should we use MPI_Bcast instead for better performance
                    m_mpi->Isend(aggGradStripesQuantized[i]->Buffer(), aggGradStripesQuantized[i]->GetSize(), MPI_CHAR, dest, numGradMatrices + 1 + i, &(sendAggGradStripeQuantizedRequests[i][j])) || MpiFail("MPI_Irecv");
                }
            }
        }

        // Initiate send of the aggregate header from main node
        std::vector<MPI_Request> sendAggHeaderRequests(NumProc() - 1);
        if (m_mpi->IsMainNode())
        {
            for (size_t j = 0; j < NumProc() - 1; ++j)
            {
                int dest = (j >= MyRank()) ? (j + 1) : j;
                // TODO: Should we use MPI_Bcast instead for better performance
                m_mpi->Isend(headerCPU, headerCPU->Size(), MPI_CHAR, dest, numGradMatrices + 1 + numGradMatrices, &(sendAggHeaderRequests[j])) || MpiFail("MPI_Isend");
            }
        }

        // Wait to receive all aggregated stripes and unquantize
        for (size_t i = 0; i < numGradMatrices; ++i)
        {
            m_mpi->Waitall(recvAggGradStripesQuantizedRequests[i].size(), recvAggGradStripesQuantizedRequests[i].data(), MPI_STATUSES_IGNORE) || MpiFail("MPI_Waitall");

            m_preAggGradQuantizers[i]->UnquantizeAsync(*(m_gradQuantized[i]), *(gradients[i]), false);
        }

        // Wait to receive aggregate header
        if (!m_mpi->IsMainNode())
            m_mpi->Wait(&recvAggHeaderRequest, MPI_STATUSES_IGNORE) || MpiFail("MPI_Wait");

        // Wait for all the unquantizations to finish
        for (size_t i = 0; i < numGradMatrices; ++i)
        {
            m_preAggGradQuantizers[i]->WaitUnquantizeAsyncDone();

            if (m_traceLevel >= DEBUG_OUTPUT_TRACE_LEVEL)
            {
                char printHeaderBuf[1024];
                sprintf(printHeaderBuf, "MPI Rank: %d, Aggregated Gradient Matrix No. %d", (int) MyRank(), (int) i);
                PrintMatrix(printHeaderBuf, gradients[i]);
            }
        }

        // Wait for completion of the async send requests
        for (int i = 0; i < sendGradStripesQuantizedRequests.size(); ++i)
        {
            if (sendGradStripesQuantizedRequests[i].size() > 0)
                m_mpi->Waitall(sendGradStripesQuantizedRequests[i].size(), sendGradStripesQuantizedRequests[i].data(), MPI_STATUSES_IGNORE) || MpiFail("MPI_Waitall");
        }

        if (!m_mpi->IsMainNode())
            m_mpi->Wait(&sendHeaderRequest, MPI_STATUSES_IGNORE) || MpiFail("MPI_Wait");

        for (int i = 0; i < sendAggGradStripeQuantizedRequests.size(); ++i)
        {
            if (sendAggGradStripeQuantizedRequests[i].size() > 0)
                m_mpi->Waitall(sendAggGradStripeQuantizedRequests[i].size(), sendAggGradStripeQuantizedRequests[i].data(), MPI_STATUSES_IGNORE) || MpiFail("MPI_Waitall");
        }

        if (m_mpi->IsMainNode())
            m_mpi->Waitall(sendAggHeaderRequests.size(), sendAggHeaderRequests.data(), MPI_STATUSES_IGNORE) || MpiFail("MPI_Waitall");

        if (showSyncPerfStats)
        {
            aggregationTimer.Stop();
            double gradientAggregationTime = aggregationTimer.ElapsedSeconds();
            fprintf(stderr, "Actual gradient aggregation time: %.6g\n", gradientAggregationTime);
        }
    }

    // Debug helper to print matrix contents
    static void PrintMatrix(const char* printHeader, Matrix<ElemType>* matrixToPrint, bool peek = true)
    {
        if (peek)
        {
            const size_t numRowsToPeek = 3;
            const size_t numColsToPeek = 3;

            size_t numRowsToPrint = (std::min)(numRowsToPeek, matrixToPrint->GetNumRows());
            size_t numColsToPrint = (std::min)(numColsToPeek, matrixToPrint->GetNumCols());

            matrixToPrint->Print(printHeader, 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
        }
        else
        {
            matrixToPrint->Print(printHeader);
        }

        fflush(stderr);
    }

private:
    std::unique_ptr<CUDAPageLockedMemAllocator> m_allocator;

    std::vector<std::unique_ptr<MatrixQuantizer<ElemType>>> m_preAggGradQuantizers;
    std::vector<std::unique_ptr<QuantizedMatrix<ElemType>>> m_gradQuantized;

    std::vector<std::unique_ptr<MatrixQuantizer<ElemType>>> m_aggGradStripeQuantizers;
    std::vector<std::vector<std::unique_ptr<QuantizedMatrix<ElemType>>>> m_recvGradStripesQuantized;
    std::vector<DistGradHeader*> m_recvHeaders;

    // Number of bits that each gradient value is quantized to before communication
    // with other nodes
    int m_numQuantizationBits;

    // option for handling the mean for 1-bit quantization
    // force 1-bit quant to threshold against 0 rather than the midpoint between lower and upper
    bool m_zeroThresholdFor1Bit;

    // Since the self-stripe in an all-reduce is not communicated, there is really no reason to
    // quantize it for reduced communication. However, we add this as an option for for consistency
    // across all stripes if desired
    bool m_useQuantizationForSelfStripe;

    // Perform asynchronous gradient aggregation using double buffering of the gradient matrices
    bool m_useAsyncAggregation;

    // Future corresponding to the current in-flight async gradient aggregation
    std::future<void> m_pendingAsyncAggregation;

    // Buffered gradients that we asynchronously aggregate
    std::unordered_map<Matrix<ElemType>*, std::unique_ptr<Matrix<ElemType>>> m_bufferedGradients;
    DistGradHeader* m_bufferedGradHeader;

    int m_traceLevel;
    int m_syncStatsTrace;

    // Only used for controlling frequency of measuring/showing gradient aggregation perf stats
    size_t m_iterationCount;

    bool m_initialized;
};

} } }
