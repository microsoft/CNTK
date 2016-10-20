//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "CNTKLibrary.h"
#include <MatrixQuantizerImpl.h>

namespace Microsoft { namespace MSR { namespace CNTK {
    class GPUDataTransferer;

    class MPIWrapper;
    typedef std::shared_ptr<MPIWrapper> MPIWrapperPtr;
}}}

namespace CNTK
{
    class MPICommunicatorImpl final : public DistributedCommunicator, public std::enable_shared_from_this<MPICommunicatorImpl>
    {
    public:
        MPICommunicatorImpl();

        virtual const std::unordered_set<DistributedWorkerDescriptor>& Workers() const override;

        virtual const DistributedWorkerDescriptor& CurrentWorker() const override;

        // Creates a new distributed communicator comprising of a subset of the workers in this communicator
        virtual DistributedCommunicatorPtr SubGroup(const std::unordered_set<DistributedWorkerDescriptor>& subGroupWorkers) const override;

        // A collective communication API to concatenate values across each worker of this communicator. The concatenated values are only sent to the specified workers; for all others the returned Values are null
        // TODO: Add an async variant of the Concatenate method
        virtual void Concatenate(
            const std::vector<ValuePtr>& values,
            std::vector<ValuePtr>& outValues,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        // A collective communication API to aggregate values across each worker of this communicator. The aggregated values are only sent to the specified workers; for all others the returned Values are null
        virtual void AggregateInPlace(const std::vector<NDArrayViewPtr>& values,
                                      const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        virtual void Aggregate(
            const std::vector<NDArrayViewPtr>& inValues,
            std::vector<NDArrayViewPtr>& outValues,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        virtual std::future<std::vector<NDArrayViewPtr>> AggregateAsync(const std::vector<NDArrayViewPtr>& inValues,
                                                    const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        // A collective communication API to perform quantized aggregation of values across all workers of this communicator
        // TODO: Add an async variant of the QuantizedAggregate method
        virtual void QuantizedAggregate(const std::vector<NDArrayViewPtr>& inValues,
            const std::vector<NDArrayViewPtr>& inPreviousQuantizationResidues,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers,
            const std::vector<NDArrayViewPtr>& aggregatedOutputs,
            const std::vector<NDArrayViewPtr>& newQuantizationResidues) override;

    private:
        void CheckWorkers(const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers);
        void Initialize(const std::vector<NDArrayViewPtr>& values);

        void AggregateImpl(
            const std::vector<NDArrayViewPtr>& inputValues,
            const std::vector<NDArrayViewPtr>& outputValues,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers);

        struct Buffer
        {
            std::shared_ptr<void> data = nullptr;
            size_t totalSize = 0;
        };

        static Buffer AllocateIntermediateBuffer(int deviceID, size_t totalSize);

        DistributedWorkerDescriptor m_currentWorker;
        std::unordered_set<DistributedWorkerDescriptor> m_workers;

        // TODO: these two are always parallel, merge them together?
        std::vector<std::shared_ptr<Microsoft::MSR::CNTK::GPUDataTransferer>> m_gpuDataTransferers;
        std::vector<Buffer> m_intermediateCPUBuffers;

        Microsoft::MSR::CNTK::MPIWrapperPtr m_mpi;
    };
}