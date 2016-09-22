//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "MPIWrapper.h"
#include "GPUDataTransferer.h"
#include "CUDAPageLockedMemAllocator.h"

namespace CNTK
{

    class MPICommunicatorImpl final : public DistributedCommunicator, std::enable_shared_from_this<MPICommunicatorImpl>
    {
    public:
        MPICommunicatorImpl();
        
        virtual std::unordered_set<DistributedWorkerDescriptor> Workers() const override;

        virtual const DistributedWorkerDescriptor& CurrentWorker() const override;

        // Creates a new distributed communicator comprising of a subset of the workers in this communicator
        virtual DistributedCommunicatorPtr SubGroup(const std::unordered_set<DistributedWorkerDescriptor>& subGroupWorkers) const override;

        // A collective communication API to concatenate values across each worker of this communicator. The concatenated values are only sent to the specified workers; for all others the returned Values are null
        // TODO: Add an async variant of the Concatenate method
        virtual std::unordered_set<Value> Concatenate(const std::unordered_set<Value>& values, const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers, DeviceDescriptor device = DeviceDescriptor::DefaultDevice()) override;

        // A collective communication API to aggregate values across each worker of this communicator. The aggregated values are only sent to the specified workers; for all others the returned Values are null
        virtual void AggregateInPlace(const std::vector<ValuePtr>& values,
                                               const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        virtual std::future<void> AggregateInPlaceAsync(const std::vector<ValuePtr>& values,
                                               const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        virtual std::vector<ValuePtr> Aggregate(const std::vector<ValuePtr>& inValues,
                                                const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        // A collective communication API to perform quantized aggregation of values across all workers of this communicator
        // TODO: Add an async variant of the QuantizedAggregate method
        virtual void QuantizedAggregate(const std::vector<Value>& inValues,
                                const std::unordered_set<Value>& inPreviousQuantizationResidues,
                                const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers,
                                const std::unordered_set<Value>& aggregatedOutputs,
                                const std::unordered_set<Value>& newQuantizationResidues) override;
    private:

        std::vector<size_t> Prepare(const std::vector<ValuePtr>& values);

        struct Buffer
        {
            std::shared_ptr<void> data = nullptr;
            size_t totalSize = 0;
        };

        static Buffer AllocateIntermediateBuffer(int deviceID, size_t totalSize);

        Microsoft::MSR::CNTK::MPIWrapperPtr m_mpi;

        // these two are always parallel, merge them together?
        std::vector<std::unique_ptr<GPUDataTransferer>> m_gpuDataTransferers;
        std::vector<Buffer> m_intermediateCPUBuffers;
    };
}