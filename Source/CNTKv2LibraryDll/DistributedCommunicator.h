//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "GPUDataTransferer.h"

namespace CNTK
{
    class Microsoft::MSR::CNTK::MPIWrapper;
    typedef std::shared_ptr<Microsoft::MSR::CNTK::MPIWrapper> MPIWrapperPtr;

    class MPICommunicatorImpl final : public DistributedCommunicator, public std::enable_shared_from_this<MPICommunicatorImpl>
    {
    public:
        MPICommunicatorImpl();
        
        virtual std::unordered_set<DistributedWorkerDescriptor> Workers() const override;

        virtual const DistributedWorkerDescriptor& CurrentWorker() const override;

        // Creates a new distributed communicator comprising of a subset of the workers in this communicator
        virtual DistributedCommunicatorPtr SubGroup(const std::unordered_set<DistributedWorkerDescriptor>& subGroupWorkers) const override;

        // A collective communication API to concatenate values across each worker of this communicator. The concatenated values are only sent to the specified workers; for all others the returned Values are null
        // TODO: Add an async variant of the Concatenate method
        virtual std::unordered_set<ValuePtr> Concatenate(const std::unordered_set<ValuePtr>& values, const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        // A collective communication API to aggregate values across each worker of this communicator. The aggregated values are only sent to the specified workers; for all others the returned Values are null
        virtual void AggregateInPlace(const std::vector<ValuePtr>& values,
                                      const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        virtual std::vector<ValuePtr> Aggregate(const std::vector<ValuePtr>& inValues,
                                                const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        virtual std::future<std::vector<ValuePtr>> AggregateAsync(const std::vector<ValuePtr>& inValues,
                                                    const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        // A collective communication API to perform quantized aggregation of values across all workers of this communicator
        // TODO: Add an async variant of the QuantizedAggregate method
        virtual void QuantizedAggregate(const std::vector<ValuePtr>& inValues,
                                const std::unordered_set<ValuePtr>& inPreviousQuantizationResidues,
                                const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers,
                                const std::unordered_set<ValuePtr>& aggregatedOutputs,
                                const std::unordered_set<ValuePtr>& newQuantizationResidues) override;
    private:

        std::vector<int> Prepare(const std::vector<ValuePtr>& values);

        void AggregateImpl(const std::vector<ValuePtr>& inputValues,
                           const std::vector<ValuePtr>& outputValues,
                           const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers);

        struct Buffer
        {
            std::shared_ptr<void> data = nullptr;
            size_t totalSize = 0;
        };

        static Buffer AllocateIntermediateBuffer(int deviceID, size_t totalSize);

        Microsoft::MSR::CNTK::MPIWrapperPtr m_mpi;
        DistributedWorkerDescriptor m_currentWorker;
        std::unordered_set<DistributedWorkerDescriptor> m_workers;

        // these two are always parallel, merge them together?
        std::vector<std::unique_ptr<Microsoft::MSR::CNTK::GPUDataTransferer>> m_gpuDataTransferers;
        std::vector<Buffer> m_intermediateCPUBuffers;
    };
}