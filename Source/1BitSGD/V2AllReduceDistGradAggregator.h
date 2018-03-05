//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#undef _SCL_SECURE_NO_WARNINGS
#include "CNTKLibrary.h"
#include "Utils.h"

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
class V2AllReduceDistGradAggregator : public IDistGradAggregator<ElemType>
{
    UsingIDistGradAggregatorMembers;

    static const int DEBUG_OUTPUT_TRACE_LEVEL = 3;
    ::CNTK::QuantizedDistributedCommunicatorPtr m_communicator;

public:
    V2AllReduceDistGradAggregator(::CNTK::QuantizedDistributedCommunicatorPtr communicator, bool useAsyncAggregation, int traceLevel, int syncStatsTrace)
        : IDistGradAggregator<ElemType>(nullptr), m_traceLevel(traceLevel), m_initialized(false), m_useAsyncAggregation(useAsyncAggregation), m_bufferedGradHeader(nullptr), m_syncStatsTrace(syncStatsTrace), m_iterationCount(0),
        m_communicator(communicator)
    {}

    ~V2AllReduceDistGradAggregator()
    {
        if (m_bufferedGradHeader != nullptr)
            DistGradHeader::Destroy(m_bufferedGradHeader);
    }

    void Initialize(const std::vector<Matrix<ElemType>*>& gradients, int numEvalNodes)
    {
        // When called the first time let's setup the quantizers and matrices for holding quantized values.
        // These can live for the lifetime of the aggregator since the gradient matrix dimensions for learnable parameters
        // do not change
        m_initialized = true;
        int deviceId = gradients[0]->GetDeviceId();

        for (size_t i = 0; i < gradients.size(); i++)
        {
            // Make sure none of the gradient matrices are sparse - we currently do not support aggregation of sparse gradient matrices
            if (gradients[i]->GetMatrixType() != DENSE)
                RuntimeError("Gradient aggregation for sparse gradient matrices is currently unsupported!");

            if (m_useAsyncAggregation)
                m_bufferedGradients[gradients[i]].reset(new Matrix<ElemType>(gradients[i]->GetNumRows(), gradients[i]->GetNumCols(), deviceId));
        }

        if (m_useAsyncAggregation)
        {
            m_bufferedGradHeader = DistGradHeader::Create(numEvalNodes);
            m_bufferedGradHeader->Clear();
        }
    }

    void ResetState(const std::vector<Matrix<ElemType>*>& gradients)
    {
        // If we are resetting state, let's clear previous quantization residues
        // Make sure there is no pending async aggregation
        if (m_useAsyncAggregation && m_pendingAsyncAggregation.valid())
            LogicError("Unexpected pending async gradient aggregation found when resetting aggregator state!");

        for (size_t i = 0; i < m_residuals.size(); ++i)
            m_residuals[i]->SetValue(static_cast<ElemType>(0.0));

        for (size_t i = 0; i < m_stripeResiduals.size(); ++i)
            if (m_stripeResiduals[i])
                m_stripeResiduals[i]->SetValue(static_cast<ElemType>(0.0));

        // Zero out the buffered gradients if resetting state
        if (m_useAsyncAggregation)
        {
            for (size_t i = 0; i < gradients.size(); i++)
                m_bufferedGradients[gradients[i]]->SetValue(static_cast<ElemType>(0));

            m_bufferedGradHeader->Clear();
        }
    }

    // Aggregate the gradient matrices across all nodes
    bool AggregateGradients(const std::vector<Matrix<ElemType>*>& gradients, DistGradHeader* headerCPU, bool resetState) override
    {
        if (!m_initialized)
            Initialize(gradients, headerCPU->numEvalNode);
        else if (resetState)
            ResetState(gradients);

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
                gradients[i]->SetValue(static_cast<ElemType>(0));

            if (m_useAsyncAggregation)
            {
                std::unique_ptr<MatrixComputeStreamEvent> mainStreamSyncEvent(MatrixComputeStreamEvent::Create(deviceId));
                mainStreamSyncEvent->SynchronizeQuantizationComputeStreamWithEvent<ElemType>();
            }
        }

        // Aggregate header.
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
        std::vector<::CNTK::NDArrayViewPtr> valuesToAggregate{ headerData };

        // TODO: Should be async
        m_communicator->AggregateInPlace(valuesToAggregate, m_communicator->Workers());

        // Copy data back to the header
        headerCPU->criterion = headerBuffer[0];
        headerCPU->numSamples = static_cast<size_t>(headerBuffer[1]);
        headerCPU->numSamplesWithLabel = static_cast<size_t>(headerBuffer[2]);
        for (size_t i = 0; i < headerCPU->numEvalNode; ++i)
        {
            headerCPU->evalErrors[i].first = headerBuffer[3 + 2 * i];
            headerCPU->evalErrors[i].second = static_cast<size_t>(headerBuffer[3 + 2 * i + 1]);
        }

        // Aggregate gradients.
        std::vector<::CNTK::NDArrayViewPtr> gradientValues;
        for (size_t i = 0; i < gradients.size(); ++i)
        {
            assert(gradients[i]->Data() != nullptr);
            ::CNTK::NDShape shape{ gradients[i]->GetNumRows(), gradients[i]->GetNumCols() };
            auto data = ::CNTK::MakeSharedObject<::CNTK::NDArrayView>(::CNTK::AsDataType<ElemType>(), shape, gradients[i]->Data(), gradients[i]->GetNumElements() * sizeof(ElemType), ::CNTK::AsDeviceDescriptor(gradients[i]->GetDeviceId()));
            gradientValues.push_back(data);
        }

        m_communicator->QuantizedAggregateInPlace(
            gradientValues,
            m_residuals,
            m_stripeResiduals,
            m_communicator->Workers());

        if (showSyncPerfStats)
        {
            aggregationTimer.Stop();
            double gradientAggregationTime = aggregationTimer.ElapsedSeconds();
            fprintf(stderr, "Actual gradient aggregation time: %.6g\n", gradientAggregationTime);
        }
    }

private:
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

    // Residuals of quantized gradients.
    std::vector<::CNTK::NDArrayViewPtr> m_residuals;
    // Residuals of quantized aggregated stripes this node is responsible for.
    std::vector<::CNTK::NDArrayViewPtr> m_stripeResiduals;
};

} } }
