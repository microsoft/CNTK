//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Basics.h"
#include "MPIWrapper.h"
#include "CNTKLibrary.h"
#include "MatrixQuantizerImpl.h"
#include "MatrixQuantizer.h"
#include "CUDAPageLockedMemAllocator.h"
#include "Utils.h"
#include "DistributedCommunicator.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    class MatrixQuantizerBase;

    class QuantizedMatrixBase;
    std::shared_ptr<QuantizedMatrixBase> QuantizedMatrixBasePtr;

    class CUDAPageLockedMemAllocator;
} } }

namespace CNTK
{
    class QuantizedMPICommunicatorImpl final : public MPICommunicatorImpl, public QuantizedDistributedCommunicator
    {
        using Base = MPICommunicatorImpl;

        template<class T> using vector = std::vector<T>;
        template<class T> using shared_ptr = std::shared_ptr<T>;
        template<class T> using unordered_set = std::unordered_set<T>;

        using MpiFail = Microsoft::MSR::CNTK::MpiFail;
        using QuantizedMatrixBase = Microsoft::MSR::CNTK::QuantizedMatrixBase;
        using QuantizedMatrixBasePtr = shared_ptr<QuantizedMatrixBase>;
        using MatrixQuantizerBase = Microsoft::MSR::CNTK::MatrixQuantizerBase;
        using CUDAPageLockedMemAllocator = Microsoft::MSR::CNTK::CUDAPageLockedMemAllocator;

        template<class T> using MatrixQuantizer = Microsoft::MSR::CNTK::MatrixQuantizer<T>;
        template<class T> using QuantizedMatrix = Microsoft::MSR::CNTK::QuantizedMatrix<T>;
        template<class T> using Matrix = Microsoft::MSR::CNTK::Matrix<T>;

    public:
        QuantizedMPICommunicatorImpl(bool zeroThresholdFor1Bit, bool useQuantizationForSelfStripe, size_t numQuantizationBits)
            : m_zeroThresholdFor1Bit(zeroThresholdFor1Bit), m_useQuantizationForSelfStripe(useQuantizationForSelfStripe), m_numQuantizationBits(numQuantizationBits)
        {}

        void QuantizedAggregateInPlace(
            std::vector<NDArrayViewPtr>& inValues,
            std::vector<NDArrayViewPtr>& valueQuantizationResidues,
            std::vector<NDArrayViewPtr>& stripeQuantizationResidues,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override
        {
            QuantizedAggregate(
                inValues, valueQuantizationResidues, stripeQuantizationResidues,
                inValues, valueQuantizationResidues, stripeQuantizationResidues,
                sendToWorkers);
        }

        // A collective communication API to perform quantized aggregation of values across all workers of this communicator
        void QuantizedAggregate(
            const vector<NDArrayViewPtr>& inValues,
            const vector<NDArrayViewPtr>& valueQuantizationResidues,
            const vector<NDArrayViewPtr>& stripeQuantizationResidues,
            vector<NDArrayViewPtr>& aggregatedOutputs,
            vector<NDArrayViewPtr>& newQuantizationResidues,
            vector<NDArrayViewPtr>& newStripeQuantizationResidues,
            const unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override
        {
            CheckWorkers(sendToWorkers);

            if (Workers().size() == 1) // No need to aggregate anything.
            {
                aggregatedOutputs = inValues;
                newQuantizationResidues = valueQuantizationResidues;
                newStripeQuantizationResidues = stripeQuantizationResidues;
                return;
            }

            if (inValues.empty())
                return;

            DataType dataType = inValues.front()->GetDataType();
            for (const auto& v : inValues)
            {
                if (v->GetDataType() != dataType)
                    RuntimeError("Currently values of different types are not supported for quantize.");
            }

            if (dataType == DataType::Float)
                QuantizedAggregate<float>(inValues, valueQuantizationResidues, stripeQuantizationResidues, aggregatedOutputs, newQuantizationResidues, newStripeQuantizationResidues, sendToWorkers);
            else if (dataType == DataType::Double)
                QuantizedAggregate<double>(inValues, valueQuantizationResidues, stripeQuantizationResidues, aggregatedOutputs, newQuantizationResidues, newStripeQuantizationResidues, sendToWorkers);
            else
                LogicError("Unexpected type value.");
        }

        // Redefining inherited members.
        // TODO: Use using and virtual inheritance after switching to VS2015.
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
        struct Stripe
        {
            size_t m_startCol;
            size_t m_numCols;
        };

        // Determine which stripe of the gradient is this node responsible for
        Stripe GetStripeForNode(size_t numCols, size_t nodeRank, size_t numNodes)
        {
            size_t numColsPerNode = numCols / numNodes;
            size_t residue = numCols % numNodes;
            size_t startColNumofStripe = (numColsPerNode * nodeRank) + min(residue, nodeRank);
            size_t numColsinStripe = numColsPerNode + ((nodeRank < residue) ? 1 : 0);
            return Stripe{ startColNumofStripe, numColsinStripe };
        }

        template <typename ElementType>
        MatrixQuantizer<ElementType>& GetQuantizer(const shared_ptr<MatrixQuantizerBase>& quantizer)
        {
            return static_cast<MatrixQuantizer<ElementType>&>(*quantizer);
        }

        template <typename ElementType>
        QuantizedMatrix<ElementType>& GetQuantizedMatrix(QuantizedMatrixBase& matrix)
        {
            return static_cast<QuantizedMatrix<ElementType>&>(matrix);
        }

        void InitializeBuffers(
            const vector<NDArrayViewPtr>& inValues,
            vector<NDArrayViewPtr>& valueQuantizationResidues,
            vector<NDArrayViewPtr>& stripeQuantizationResidues,
            vector<NDArrayViewPtr>& aggregatedOutputs,
            vector<NDArrayViewPtr>& newQuantizationResidues,
            vector<NDArrayViewPtr>& newStripeQuantizationResidues)
        {
            m_preAggregatedGradientQuantizers.resize(std::max(inValues.size(), valueQuantizationResidues.size()));
            if (inValues.size() != m_preAggregatedGradientQuantizers.size())
                LogicError("Number of aggregated values should be equal number of quantized residuals.");

            m_quantizedGradients.resize(inValues.size());
            m_aggregatedGradientStripeQuantizers.resize(std::max(inValues.size(), stripeQuantizationResidues.size()));
            if (inValues.size() != m_aggregatedGradientStripeQuantizers.size())
                LogicError("Number of aggregated values should be equal number of striped quantized residuals.");

            m_recvGradientStripesQuantized.resize(inValues.size());

            if (valueQuantizationResidues.empty())
                valueQuantizationResidues.resize(inValues.size());

            if (stripeQuantizationResidues.empty())
                stripeQuantizationResidues.resize(inValues.size());

            if (newQuantizationResidues.empty())
                newQuantizationResidues.resize(inValues.size());

            if (newStripeQuantizationResidues.empty())
                newStripeQuantizationResidues.resize(inValues.size());

            for (auto i = 0; i < inValues.size(); ++i)
            {
                auto view = inValues[i];

                // Make sure none of the values are sparse - we currently do not support aggregation of sparse matrices
                if (view->GetStorageFormat() != StorageFormat::Dense)
                    RuntimeError("Aggregation for sparse matrices is currently not supported!");

                // Currently we always use async aggregation. Is this correct?
                if (view->GetDataType() == DataType::Float)
                    InitializeBuffer<float>(inValues, valueQuantizationResidues, stripeQuantizationResidues, aggregatedOutputs, newQuantizationResidues, newStripeQuantizationResidues, i);
                else if (view->GetDataType() == DataType::Double)
                    InitializeBuffer<double>(inValues, valueQuantizationResidues, stripeQuantizationResidues, aggregatedOutputs, newQuantizationResidues, newStripeQuantizationResidues, i);
                else
                    LogicError("Unsupported type");
            }
        }

        template<class ElemType>
        void InitializeBuffer(
            const vector<NDArrayViewPtr>& inValues,
            vector<NDArrayViewPtr>& valueQuantizationResidues,
            vector<NDArrayViewPtr>& stripeQuantizationResidues,
            vector<NDArrayViewPtr>& /*aggregatedOutputs*/,
            vector<NDArrayViewPtr>& newQuantizationResidues,
            vector<NDArrayViewPtr>& newStripeQuantizationResidues,
            size_t index)
        {
            int rank = static_cast<int>(CurrentWorker().m_globalRank);
            int numWorkers = static_cast<int>(Workers().size());

            auto value = inValues[index];
            auto v = GetMatrix<ElemType>(value);
            size_t nRow = v->GetNumRows();
            size_t nCol = v->GetNumCols();

            if (!valueQuantizationResidues[index])
            {
                auto residual = MakeSharedObject<NDArrayView>(AsDataType<ElemType>(), NDShape{ nRow, nCol }, AsDeviceDescriptor(v->GetDeviceId()));
                auto outputResidual = MakeSharedObject<NDArrayView>(AsDataType<ElemType>(), NDShape{ nRow, nCol }, AsDeviceDescriptor(v->GetDeviceId()));
                valueQuantizationResidues[index] = residual;
                newQuantizationResidues[index] = outputResidual;
            }

            Stripe stripe = GetStripeForNode(v->GetNumCols(), rank, numWorkers);
            if (!stripeQuantizationResidues[index] && stripe.m_numCols > 0)
            {
                auto residual = MakeSharedObject<NDArrayView>(::CNTK::AsDataType<ElemType>(), NDShape{ nRow, stripe.m_numCols }, AsDeviceDescriptor(v->GetDeviceId()));
                auto outputResidual = MakeSharedObject<NDArrayView>(::CNTK::AsDataType<ElemType>(), NDShape{ nRow, stripe.m_numCols }, AsDeviceDescriptor(v->GetDeviceId()));
                stripeQuantizationResidues[index] = residual;
                newStripeQuantizationResidues[index] = outputResidual;
            }

            auto inResidual = valueQuantizationResidues[index];

            // Initialize buffer.
            m_quantizedGradients[index] = std::make_shared<QuantizedMatrix<ElemType>>(v->GetNumRows(), v->GetNumCols(), m_numQuantizationBits, CPUDEVICE, m_allocator.get());

            // Initialize gradient quantizer.
            m_preAggregatedGradientQuantizers[index] = std::make_shared<MatrixQuantizer<ElemType>>(GetMatrix<ElemType>(inResidual)->GetDeviceId(), true);

            // Determine which stripe of the gradient is this node responsible for
            MatrixQuantizer<ElemType>* aggregatedGradientStripeQuantizers = nullptr;
            if (stripe.m_numCols > 0)
            {
                // Initialize quantizer
                aggregatedGradientStripeQuantizers = new MatrixQuantizer<ElemType>(GetMatrix<ElemType>(inResidual)->GetDeviceId(), true);
                m_recvGradientStripesQuantized[index].resize(numWorkers - 1);
                for (size_t j = 0; j < numWorkers - 1; ++j)
                    m_recvGradientStripesQuantized[index][j]= std::unique_ptr<QuantizedMatrix<ElemType>>(new QuantizedMatrix<ElemType>(v->GetNumRows(), stripe.m_numCols, m_numQuantizationBits, CPUDEVICE, m_allocator.get()));
            }

            m_aggregatedGradientStripeQuantizers[index] = std::unique_ptr<MatrixQuantizer<ElemType>>(aggregatedGradientStripeQuantizers);
        }

        template<class ElemType>
        void QuantizedAggregate(
            const vector<NDArrayViewPtr>& inValues,
            const vector<NDArrayViewPtr>& formalValueQuantizationResidues,
            const vector<NDArrayViewPtr>& formalStripeQuantizationResidues,
            vector<NDArrayViewPtr>& aggregatedOutputs,
            vector<NDArrayViewPtr>& newQuantizationResidues,
            vector<NDArrayViewPtr>& newStripeQuantizationResidues,
            const unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
        {
            CheckWorkers(sendToWorkers);

            const int numWorkers = static_cast<int>(Workers().size());
            const int rank = static_cast<int>(CurrentWorker().m_globalRank);

            auto valueQuantizationResidues = formalValueQuantizationResidues;
            auto stripeQuantizationResidues = formalStripeQuantizationResidues;

            InitializeBuffers(
                inValues,
                valueQuantizationResidues,
                stripeQuantizationResidues,
                aggregatedOutputs,
                newQuantizationResidues,
                newStripeQuantizationResidues);

            vector<shared_ptr<Matrix<ElemType>>> inputValues;
            vector<shared_ptr<Matrix<ElemType>>> outputValues;
            vector<shared_ptr<Matrix<ElemType>>> inputResiduals;
            vector<shared_ptr<Matrix<ElemType>>> outputResiduals;
            vector<shared_ptr<Matrix<ElemType>>> inputStripeResiduals;
            vector<shared_ptr<Matrix<ElemType>>> outputStripeResiduals;

            // Check that input corresponds to output and covert NDArrayViews to the corresponding matrices.
            for (size_t i = 0; i < inValues.size(); i++)
            {
                assert(inValues[i]->Shape().TotalSize() == aggregatedOutputs[i]->Shape().TotalSize());
                assert(inValues[i]->GetDataType() == aggregatedOutputs[i]->GetDataType());
                assert(inValues[i]->Device() == aggregatedOutputs[i]->Device());

                assert(inValues[i] != nullptr);
                inputValues.push_back(GetWritableMatrix<ElemType>(inValues[i]));

                assert(aggregatedOutputs[i] != nullptr);
                outputValues.push_back(GetWritableMatrix<ElemType>(aggregatedOutputs[i]));

                assert(valueQuantizationResidues[i] != nullptr);
                inputResiduals.push_back(GetWritableMatrix<ElemType>(valueQuantizationResidues[i]));

                assert(newQuantizationResidues[i] != nullptr);
                outputResiduals.push_back(GetWritableMatrix<ElemType>(newQuantizationResidues[i]));;

                // Stripe residuals can be null in case when the stripe does not belong to this node.
                inputStripeResiduals.push_back(stripeQuantizationResidues[i] ? GetWritableMatrix<ElemType>(stripeQuantizationResidues[i]) : nullptr);;
                outputStripeResiduals.push_back(newStripeQuantizationResidues[i]? GetWritableMatrix<ElemType>(newStripeQuantizationResidues[i]) : nullptr);
            }

            // Prepare receiving buffers.
            vector<std::unique_ptr<Matrix<ElemType>>> aggGradStripes;
            vector<std::unique_ptr<QuantizedMatrix<ElemType>>> aggGradStripesQuantized;
            for (size_t i = 0; i < inputValues.size(); i++)
            {
                size_t nCol = inputValues[i]->GetNumCols();

                // Determine which stripe of the gradient is this node responsible for
                Stripe stripe = GetStripeForNode(nCol, rank, numWorkers);
                Matrix<ElemType>* currAggGradStripe = nullptr;
                QuantizedMatrix<ElemType>* currAggGradStripeQuantized = nullptr;
                if (stripe.m_numCols > 0)
                {
                    currAggGradStripe = new Matrix<ElemType>(inputValues[i]->ColumnSlice(stripe.m_startCol, stripe.m_numCols));
                    currAggGradStripeQuantized = new QuantizedMatrix<ElemType>(GetQuantizedMatrix<ElemType>(*m_quantizedGradients[i]).ColumnSlice(stripe.m_startCol, stripe.m_numCols));
                }

                aggGradStripes.push_back(std::unique_ptr<Matrix<ElemType>>(currAggGradStripe));
                aggGradStripesQuantized.push_back(std::unique_ptr<QuantizedMatrix<ElemType>>(currAggGradStripeQuantized));
            }

            // Initiate quantization of the gradient matrices
            for (size_t i = 0; i < inValues.size(); ++i)
                GetQuantizer<ElemType>(m_preAggregatedGradientQuantizers[i]).QuantizeAsync(*(inputValues[i]), *(inputResiduals[i]), GetQuantizedMatrix<ElemType>(*(m_quantizedGradients[i])), *(outputResiduals[i]), m_zeroThresholdFor1Bit);

            // Initiate receive of the stripe to be aggregated by the current node, from all other nodes
            vector<MPI_Request> recvGradStripesQuantizedRequests;
            vector<int> recvRequestIdxToGradientMatrixIdxMap;
            for (int i = 0; i < inputValues.size(); ++i)
            {
                Stripe stripe = GetStripeForNode(inputValues[i]->GetNumCols(), rank, numWorkers);
                if (stripe.m_numCols > 0)
                {
                    recvRequestIdxToGradientMatrixIdxMap.push_back(i);
                    for (int j = 0; j < numWorkers - 1; ++j)
                    {
                        int source = (j >= rank) ? (j + 1) : j;

                        recvGradStripesQuantizedRequests.push_back(MPI_Request());
                        int recvRequestIdx = (int)recvGradStripesQuantizedRequests.size() - 1;

                        m_mpi->Irecv(GetQuantizedMatrix<ElemType>(*m_recvGradientStripesQuantized[i][j]).Buffer(), (int)GetQuantizedMatrix<ElemType>(*m_recvGradientStripesQuantized[i][j]).GetSize(), MPI_CHAR, source, i, &(recvGradStripesQuantizedRequests[recvRequestIdx])) || MpiFail("MPI_Irecv");
                    }
                }
            }

            // Asynchronously send stripes of the quantized gradient matrices to the respective nodes that own aggregation of that stripe
            std::vector<std::vector<MPI_Request>> sendGradStripesQuantizedRequests(inValues.size());
            for (int i = 0; i < inValues.size(); ++i)
            {
                GetQuantizer<ElemType>(m_preAggregatedGradientQuantizers[i]).WaitQuantizeAsyncDone();

                size_t sendRequestIdx = 0;
                for (int j = 0; j < numWorkers; ++j)
                {
                    Stripe stripe = GetStripeForNode(inputValues[i]->GetNumCols(), j, numWorkers);
                    if (stripe.m_numCols > 0)
                    {
                        // Do not send stripe for self
                        if (j != rank)
                        {
                            sendGradStripesQuantizedRequests[i].push_back(MPI_Request());
                            QuantizedMatrix<ElemType> quantizedStripe = GetQuantizedMatrix<ElemType>(*m_quantizedGradients[i]).ColumnSlice(stripe.m_startCol, stripe.m_numCols);

                            m_mpi->Isend(quantizedStripe.Buffer(), (int)quantizedStripe.GetSize(), MPI_CHAR, j, i, &(sendGradStripesQuantizedRequests[i][sendRequestIdx])) || MpiFail("MPI_Isend");
                            sendRequestIdx++;
                        }
                        else
                        {
                            // Initialize the aggregate for the stripe with the quantized gradients instead of the original
                            // gradients themselves, if so desired
                            if (m_useQuantizationForSelfStripe)
                            {
                                QuantizedMatrix<ElemType> preAggGradSelfStripeQuantized = GetQuantizedMatrix<ElemType>(*m_quantizedGradients[i]).ColumnSlice(stripe.m_startCol, stripe.m_numCols);
                                GetQuantizer<ElemType>(m_aggregatedGradientStripeQuantizers[i]).UnquantizeAsync(preAggGradSelfStripeQuantized, *(aggGradStripes[i]), false);
                            }
                        }
                    }
                }
            }

            // Wait for the stripes to arrive from each node and unquantize and aggregate
            size_t numReceivesExpected = recvGradStripesQuantizedRequests.size();
            size_t numActualReceives = 0;
            std::vector<int> perGradMatrixReceiveCount(recvRequestIdxToGradientMatrixIdxMap.size(), 0);
            while (numActualReceives < numReceivesExpected)
            {
                int idx = MPI_UNDEFINED;
                m_mpi->Waitany((int)recvGradStripesQuantizedRequests.size(), recvGradStripesQuantizedRequests.data(), &idx, MPI_STATUS_IGNORE) || MpiFail("MPI_Waitany");
                if (idx == MPI_UNDEFINED)
                {
                    break;
                }

                numActualReceives++;

                int gradMatrixIdxPosition = idx / (numWorkers - 1);
                int recvBufferSubIndex = idx % (numWorkers - 1);

                // Map idx back to the actual gradient matrix index
                int gradMatrixIdx = recvRequestIdxToGradientMatrixIdxMap[gradMatrixIdxPosition];

                // Wait for the previous Unquantize to finish before issuing a new one
                if (m_useQuantizationForSelfStripe || (perGradMatrixReceiveCount[gradMatrixIdxPosition] > 0))
                    GetQuantizer<ElemType>(m_aggregatedGradientStripeQuantizers[gradMatrixIdx]).WaitUnquantizeAsyncDone();

                GetQuantizer<ElemType>(m_aggregatedGradientStripeQuantizers[gradMatrixIdx]).UnquantizeAsync(
                    GetQuantizedMatrix<ElemType>(*m_recvGradientStripesQuantized[gradMatrixIdx][recvBufferSubIndex]),
                    *(aggGradStripes[gradMatrixIdx]),
                    true);

                perGradMatrixReceiveCount[gradMatrixIdxPosition]++;

                // Also issue the quantization if this stripe was the last one expected for this matrix
                // Note: We issue the quantization without waiting for the unquantization since the same stream
                // is used for both and they are implicitly sequenced
                // We reuse the buffer that we used for quantizing and sending out the pre-aggregation gradient
                if (perGradMatrixReceiveCount[gradMatrixIdxPosition] == (numWorkers - 1))
                {
                    Stripe stripe = GetStripeForNode(inputValues[gradMatrixIdx]->GetNumCols(), rank, numWorkers);
                    UNUSED(stripe);
                    assert(stripe.m_numCols > 0);
                    GetQuantizer<ElemType>(m_aggregatedGradientStripeQuantizers[gradMatrixIdx]).QuantizeAsync(
                        *(aggGradStripes[gradMatrixIdx]),
                        *(inputStripeResiduals[gradMatrixIdx]),
                        *(aggGradStripesQuantized[gradMatrixIdx]),
                        *(outputStripeResiduals[gradMatrixIdx]),
                        m_zeroThresholdFor1Bit);
                }
            }

            assert(numActualReceives == numReceivesExpected);

            vector<vector<MPI_Request>> recvAggGradStripesQuantizedRequests(inValues.size());
            // Initiate receive of stripes of quantized aggregated gradients from different nodes
            for (int i = 0; i < inValues.size(); ++i)
            {
                int recvRequestIdx = 0;
                for (int j = 0; j < numWorkers; ++j)
                {
                    // Do not recv stripe for self
                    if (j != rank)
                    {
                        Stripe stripe = GetStripeForNode(inputValues[i]->GetNumCols(), j, numWorkers);
                        if (stripe.m_numCols > 0)
                        {
                            recvAggGradStripesQuantizedRequests[i].push_back(MPI_Request());
                            QuantizedMatrix<ElemType> quantizedStripe = GetQuantizedMatrix<ElemType>(*m_quantizedGradients[i]).ColumnSlice(stripe.m_startCol, stripe.m_numCols);
                            m_mpi->Irecv(quantizedStripe.Buffer(), (int)quantizedStripe.GetSize(), MPI_CHAR, j, (int)inValues.size() + 1 + i, &(recvAggGradStripesQuantizedRequests[i][recvRequestIdx])) || MpiFail("MPI_Irecv");
                            recvRequestIdx++;
                        }
                    }
                }
            }

            // Initiate broadcast of quantized aggregated gradient stripes to all other nodes
            vector<vector<MPI_Request>> sendAggGradStripeQuantizedRequests(inValues.size());
            for (int i = 0; i < inValues.size(); ++i)
            {
                Stripe stripe = GetStripeForNode(inputValues[i]->GetNumCols(), rank, numWorkers);
                if (stripe.m_numCols > 0)
                {
                    sendAggGradStripeQuantizedRequests[i] = std::vector<MPI_Request>(numWorkers - 1);
                    GetQuantizer<ElemType>(m_aggregatedGradientStripeQuantizers[i]).WaitQuantizeAsyncDone();
                    for (int j = 0; j < numWorkers - 1; ++j)
                    {
                        int dest = (j >= rank) ? (j + 1) : j;

                        // TODO: Should we use MPI_Bcast instead for better performance
                        m_mpi->Isend(aggGradStripesQuantized[i]->Buffer(), (int)aggGradStripesQuantized[i]->GetSize(), MPI_CHAR, dest, (int)inValues.size() + 1 + i, &(sendAggGradStripeQuantizedRequests[i][j])) || MpiFail("MPI_Irecv");
                    }
                }
            }

            // Wait to receive all aggregated stripes and unquantize
            for (size_t i = 0; i < inValues.size(); ++i)
            {
                m_mpi->Waitall((int)recvAggGradStripesQuantizedRequests[i].size(), recvAggGradStripesQuantizedRequests[i].data(), MPI_STATUSES_IGNORE) || MpiFail("MPI_Waitall");
                GetQuantizer<ElemType>(m_preAggregatedGradientQuantizers[i]).UnquantizeAsync(GetQuantizedMatrix<ElemType>(*m_quantizedGradients[i]), *(outputValues[i]), false);
            }

            // Wait for all the unquantizations to finish
            for (size_t i = 0; i < inValues.size(); ++i)
                GetQuantizer<ElemType>(m_preAggregatedGradientQuantizers[i]).WaitUnquantizeAsyncDone();

            // Wait for completion of the async send requests
            for (int i = 0; i < sendGradStripesQuantizedRequests.size(); ++i)
            {
                if (sendGradStripesQuantizedRequests[i].size() > 0)
                    m_mpi->Waitall((int)sendGradStripesQuantizedRequests[i].size(), sendGradStripesQuantizedRequests[i].data(), MPI_STATUSES_IGNORE) || MpiFail("MPI_Waitall");
            }

            for (int i = 0; i < sendAggGradStripeQuantizedRequests.size(); ++i)
            {
                if (sendAggGradStripeQuantizedRequests[i].size() > 0)
                    m_mpi->Waitall((int)sendAggGradStripeQuantizedRequests[i].size(), sendAggGradStripeQuantizedRequests[i].data(), MPI_STATUSES_IGNORE) || MpiFail("MPI_Waitall");
            }
        }

        // option for handling the mean for 1-bit quantization
        // force 1-bit quant to threshold against 0 rather than the midpoint between lower and upper
        const bool m_zeroThresholdFor1Bit;

        // Number of bits that each gradient value is quantized to before communication with other nodes.
        const size_t m_numQuantizationBits;

        // Since the self-stripe in an all-reduce is not communicated, there is really no reason to
        // quantize it for reduced communication. However, we add this as an option for for consistency
        // across all stripes if desired
        const bool m_useQuantizationForSelfStripe;

        const std::unique_ptr<CUDAPageLockedMemAllocator> m_allocator;

        // Buffer for quantized gradients.
        vector<QuantizedMatrixBasePtr> m_quantizedGradients;

        // Buffer for quantized stripes.
        vector<vector<QuantizedMatrixBasePtr>> m_recvGradientStripesQuantized;

        // Quantizers to quantize initial gradients.
        vector<shared_ptr<MatrixQuantizerBase>> m_preAggregatedGradientQuantizers;

        // Quantizers to quantize aggregated stripes.
        vector<shared_ptr<MatrixQuantizerBase>> m_aggregatedGradientStripeQuantizers;
    };
}
