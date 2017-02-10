//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <future>

#undef _SCL_SECURE_NO_WARNINGS
#include "Constants.h"
#include "CNTKLibrary.h"
#include "IDistGradAggregator.h"
#include "TimerUtility.h"
#include "MatrixQuantizerImpl.h"
#include "Utils.h"
#include "NcclComm.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class V2SimpleDistGradAggregator : public IDistGradAggregator<ElemType>
{
    UsingIDistGradAggregatorMembers;
    ::CNTK::DistributedCommunicatorPtr m_communicator;
    NcclComm m_nccl;

public:
    V2SimpleDistGradAggregator(const MPIWrapperPtr& mpi, bool useAsyncAggregation, int deviceId, int syncStatsTrace, ::CNTK::DistributedCommunicatorPtr communicator, size_t packThresholdSizeInBytes = DEFAULT_PACK_THRESHOLD_SIZE_IN_BYTES)
        : IDistGradAggregator<ElemType>(mpi), m_useAsyncAggregation(useAsyncAggregation), m_initialized(false), m_bufferedGradHeader(nullptr), m_syncStatsTrace(syncStatsTrace), m_iterationCount(0),
        m_communicator(communicator), m_nccl(deviceId, mpi), m_packThresholdSizeInBytes(packThresholdSizeInBytes)
    {}

    ~V2SimpleDistGradAggregator()
    {
        if (m_bufferedGradHeader != nullptr)
            DistGradHeader::Destroy(m_bufferedGradHeader);
    }

    // Aggregate the gradient matrices across all nodes
    bool AggregateGradients(const std::vector<Matrix<ElemType>*>& gradients, DistGradHeader* headerCPU, bool resetState) override
    {
        if (!IsInitialized())
            Initialize(gradients, headerCPU->numEvalNode);
        else if (resetState)
            ResetState(gradients);

        bool showSyncPerfStats = (m_syncStatsTrace > 0) && ((m_iterationCount % m_syncStatsTrace) == 0);
        m_iterationCount++;

        if (!m_useAsyncAggregation) // In case we do not use asyn aggregation, simply aggregate.
        {
            AggregateGradientsImpl(gradients, headerCPU, showSyncPerfStats);
            return (headerCPU->numSamples != 0);
        }

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
        size_t offset = 0;
        int deviceId = gradients[0]->GetDeviceId();
        // If no additional continous buffer allocated
        for (size_t i : m_PackedGradientsIndex)
        {
            if (m_continousGradients.find(gradients[i]) == m_continousGradients.end() ||
                m_continousGradients[gradients[i]].second != gradients[i]->GetNumElements())
            {
                LogicError("No buffered gradients matrix found corresponding to a gradient matrix to be aggregated!");
            }
            offset = m_continousGradients[gradients[i]].first;
            // Swap the gradient matrix contents with the buffered contents in the continous buffer
            std::unique_ptr<Matrix<ElemType>> tempGradient;
            tempGradient.reset(new Matrix<ElemType>(gradients[i]->GetNumRows(), gradients[i]->GetNumCols(), deviceId));
            tempGradient->AssignValuesOf(m_AggregationBuffer->ColumnSlice(offset, gradients[i]->GetNumElements()).Reshaped(gradients[i]->GetNumRows(), gradients[i]->GetNumCols()));
            m_AggregationBuffer->ColumnSlice(offset, gradients[i]->GetNumElements()).AssignValuesOf(gradients[i]->Reshaped(1, gradients[i]->GetNumElements()));
            gradients[i]->AssignValuesOf(*tempGradient);
        }
        if (m_AggregationBuffer != 0)
        {
            newGradients.push_back(m_AggregationBuffer.get());
        }

        for (size_t i : m_noPackedGradientsIndex)
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
            DistGradHeader* newGradHeader = m_bufferedGradHeader;
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

private:
    bool IsInitialized() const { return m_initialized; }
    void Initialize(const std::vector<Matrix<ElemType>*>& gradients, int numEvalNodes)
    {
        int deviceId = gradients[0]->GetDeviceId();
        size_t PackedGradientsSizeInElements = 0;
        for (size_t i = 0; i < gradients.size(); i++)
        {
            if (sizeof(ElemType) * gradients[i]->GetNumElements() <= m_packThresholdSizeInBytes)
            {
                PackedGradientsSizeInElements += gradients[i]->GetNumElements();
                m_PackedGradientsIndex.push_back(i);
            }
            else
            {
                m_noPackedGradientsIndex.push_back(i);
            }

            // Make sure none of the gradient matrixes are sparse - we currently do not support aggregation of sparse gradient matrices
            if (gradients[i]->GetMatrixType() != DENSE)
                RuntimeError("Gradient aggregation for sparse gradient matrices is currently unsupported!");
        }

        m_AggregationBuffer.reset();
        m_AggregationBuffer.reset(new (std::nothrow) Matrix<ElemType>(1, PackedGradientsSizeInElements, deviceId));
        // Failed to allocate extra continous buffer
        if (m_AggregationBuffer == 0)
        {
            m_noPackedGradientsIndex.clear();
            m_PackedGradientsIndex.clear();
            PackedGradientsSizeInElements = 0;
            // Reuse "@param m_noPackedGradientsIndex" for following code, if no continous buffer allocated
            for (size_t i = 0; i < gradients.size(); i++)
            {
                m_noPackedGradientsIndex.push_back(i);
            }
        }

        if (m_useAsyncAggregation)
        {
            size_t offset = 0;
            for (size_t i : m_PackedGradientsIndex)
            {
                m_continousGradients[gradients[i]] = std::make_pair(offset, gradients[i]->GetNumElements());
                offset += gradients[i]->GetNumElements();
            }
            for (size_t i : m_noPackedGradientsIndex)
            {
                m_bufferedGradients[gradients[i]].reset(new Matrix<ElemType>(gradients[i]->GetNumRows(), gradients[i]->GetNumCols(), deviceId));
            }
        }

        if (m_useAsyncAggregation)
        {
            m_bufferedGradHeader = DistGradHeader::Create(numEvalNodes);
            m_bufferedGradHeader->Clear();
        }
        m_initialized = true;
    }

    void ResetState(const std::vector<Matrix<ElemType>*>& gradients)
    {
        if (!m_useAsyncAggregation)
            return;

        // Make sure there is no pending async aggregation
        if (m_pendingAsyncAggregation.valid())
            LogicError("Unexpected pending async gradient aggregation found when resetting aggregator state!");

        // Zero out the buffered gradients if resetting state
        for (size_t i = 0; i < gradients.size(); i++)
            m_bufferedGradients[gradients[i]]->SetValue(0);

        m_AggregationBuffer.reset();
        m_bufferedGradHeader->Clear();
    }

    void AggregateGradientsImpl(const std::vector<Matrix<ElemType>*>& gradients, DistGradHeader* headerCPU, bool showSyncPerfStats)
    {
        Timer aggregationTimer;
        int deviceId = gradients.front()->GetDeviceId();
        if (showSyncPerfStats)
        {
            std::unique_ptr<MatrixComputeStreamEvent> mainStreamSyncEvent(MatrixComputeStreamEvent::Create(deviceId));
            mainStreamSyncEvent->SynchronizeEvent();
            aggregationTimer.Start();
        }

        if (headerCPU->numSamples == 0)
        {
            assert(headerCPU->criterion == 0.0);
            assert(headerCPU->numSamplesWithLabel == 0);
            for (int i = 0; i < headerCPU->numEvalNode; ++i)
                assert(headerCPU->evalErrors[i].first == 0 && headerCPU->evalErrors[i].second == 0);

            // If the current node did not process any samples, the gradients should be zero'd
            for (size_t i = 0; i < gradients.size(); ++i)
                gradients[i]->SetValue(0);

            if (m_useAsyncAggregation)
            {
                std::unique_ptr<MatrixComputeStreamEvent> mainStreamSyncEvent(MatrixComputeStreamEvent::Create(deviceId));
                mainStreamSyncEvent->SynchronizeDataTransferFetchStreamWithEvent<ElemType>();
            }
        }

        // Prepare gradients.
        std::vector<::CNTK::NDArrayViewPtr> valuesToAggregate;
        if (!m_nccl.IsSupported())
        {
            // Pack the gradients to continous buffer
            size_t offset = 0;
            for (size_t i : m_PackedGradientsIndex)
            {
                m_AggregationBuffer->ColumnSlice(offset, gradients[i]->GetNumElements()).AssignValuesOf(gradients[i]->Reshaped(1, gradients[i]->GetNumElements()));
                offset += gradients[i]->GetNumElements();
            }
            // Push packed continous buffer to valuesToAggregate
            if (m_AggregationBuffer != 0)
            {
                ::CNTK::NDShape shape{ m_AggregationBuffer->GetNumElements() };
                auto data = ::CNTK::MakeSharedObject<::CNTK::NDArrayView>(::CNTK::AsDataType<ElemType>(), shape, m_AggregationBuffer->Data(), m_AggregationBuffer->GetNumElements() * sizeof(ElemType), ::CNTK::AsDeviceDescriptor(m_AggregationBuffer->GetDeviceId()));
                valuesToAggregate.push_back(data);
            }

            // Push un-packed gradients to valuesToAggregate
            for (size_t i : m_noPackedGradientsIndex)
            {
                ::CNTK::NDShape shapet{ gradients[i]->GetNumElements() };
                auto datat = ::CNTK::MakeSharedObject<::CNTK::NDArrayView>(::CNTK::AsDataType<ElemType>(), shapet, gradients[i]->Data(), gradients[i]->GetNumElements() * sizeof(ElemType), ::CNTK::AsDeviceDescriptor(gradients[i]->GetDeviceId()));
                valuesToAggregate.push_back(datat);
            }
        }
        else
        {
            // nccl is only enabled if all ranks have net on GPUs.
            // we assume in this case all grad layers are on the GPU too.
            if (m_useAsyncAggregation || m_AggregationBuffer == 0)
            {
                m_nccl.AllReduce(gradients);
            }
            else
            {
                // All reduce the continous buffer
                size_t offset = 0;
                for (size_t i : m_PackedGradientsIndex)
                {
                    m_AggregationBuffer->ColumnSlice(offset, gradients[i]->GetNumElements()).AssignValuesOf(gradients[i]->Reshaped(1, gradients[i]->GetNumElements()));
                    offset += gradients[i]->GetNumElements();
                }
                std::vector<Matrix<ElemType>*> ncclReduceGradients;
                ncclReduceGradients.push_back(m_AggregationBuffer.get());
                for (size_t i : m_noPackedGradientsIndex)
                {
                    ncclReduceGradients.push_back(gradients[i]);
                }
                m_nccl.AllReduce(ncclReduceGradients);
            }
        }

        // Prepare header.
        size_t numberOfElements = 1 + 1 + 1 + headerCPU->numEvalNode * 2;
        std::unique_ptr<double[]> headerBuffer(new double[numberOfElements]);
        headerBuffer[0] = headerCPU->criterion;
        headerBuffer[1] = static_cast<double>(headerCPU->numSamples);
        headerBuffer[2] = static_cast<double>(headerCPU->numSamplesWithLabel);
        for (size_t i = 0; i < headerCPU->numEvalNode; ++i)
        {
            headerBuffer[3 + 2 * i] = headerCPU->evalErrors[i].first;
            headerBuffer[3 + 2 * i + 1] = static_cast<double>(headerCPU->evalErrors[i].second);
        }

        auto headerData = ::CNTK::MakeSharedObject<::CNTK::NDArrayView>(::CNTK::DataType::Double, ::CNTK::NDShape{ numberOfElements }, headerBuffer.get(), numberOfElements * sizeof(double), ::CNTK::DeviceDescriptor::CPUDevice());
        valuesToAggregate.push_back(headerData);

        // Do the aggregation
        m_communicator->AggregateInPlace(valuesToAggregate, m_communicator->Workers());

        if (m_nccl.IsSupported())
            m_nccl.Sync();

        // Copy data back to the gradients if packed and using Sync Aggregation
        // The packed gradients are exchanged during AsyncAggregation
        if (m_AggregationBuffer != 0 && !m_useAsyncAggregation)
        {
            size_t offset = 0;
            for (size_t i : m_PackedGradientsIndex)
            {
                if (gradients[i]->Data() == nullptr) // Hack in case of eval.
                    continue;

                gradients[i]->AssignValuesOf(m_AggregationBuffer->ColumnSlice(offset, gradients[i]->GetNumElements()).Reshaped(gradients[i]->GetNumRows(), gradients[i]->GetNumCols()));
                offset += gradients[i]->GetNumElements();
            }
        }

        // Copy data back to the header
        headerCPU->criterion = headerBuffer[0];
        headerCPU->numSamples = static_cast<size_t>(headerBuffer[1]);
        headerCPU->numSamplesWithLabel = static_cast<size_t>(headerBuffer[2]);
        for (size_t i = 0; i < headerCPU->numEvalNode; ++i)
        {
            headerCPU->evalErrors[i].first = headerBuffer[3 + 2 * i];
            headerCPU->evalErrors[i].second = static_cast<size_t>(headerBuffer[3 + 2 * i + 1]);
        }

        if (showSyncPerfStats)
        {
            aggregationTimer.Stop();
            double gradientAggregationTime = aggregationTimer.ElapsedSeconds();
            fprintf(stderr, "Actual gradient aggregation time: %.6g\n", gradientAggregationTime);
        }
    }

private:
    // Perform aysnchronous gradient aggregation using double buffering of the gradient matrices
    bool m_useAsyncAggregation;

    // Future corresponding to the current in-flight async gradient aggregation
    std::future<void> m_pendingAsyncAggregation;

    // Threshold size to pack all gradients in a continous buffer
    size_t m_packThresholdSizeInBytes;
    std::unique_ptr<Matrix<ElemType>> m_AggregationBuffer;
    std::unordered_map<Matrix<ElemType>*, std::pair<size_t, size_t>> m_continousGradients;
    std::vector<size_t> m_PackedGradientsIndex;
    std::vector<size_t> m_noPackedGradientsIndex;

    // Buffered gradients that we asynchronously aggregate
    std::unordered_map<Matrix<ElemType>*, std::unique_ptr<Matrix<ElemType>>> m_bufferedGradients;
    DistGradHeader* m_bufferedGradHeader;

    // Only used for controlling frequency of measuring/showing gradient aggregation perf stats
    int m_syncStatsTrace;
    size_t m_iterationCount;
    bool m_initialized;
};

}}}
