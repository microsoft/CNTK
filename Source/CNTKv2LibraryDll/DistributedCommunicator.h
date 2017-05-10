//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "CNTKLibrary.h"
#include "Constants.h"
#include "NcclComm.h"
#include "MPIWrapper.h"
#include <MatrixQuantizerImpl.h>

namespace Microsoft { namespace MSR { namespace CNTK {
    class GPUDataTransferer;

    class MPIWrapper;
    typedef std::shared_ptr<MPIWrapper> MPIWrapperPtr;
}}}

namespace CNTK
{
    class MPICommunicatorImpl : public DistributedCommunicator, public std::enable_shared_from_this<MPICommunicatorImpl>
    {
    public:
        MPICommunicatorImpl(size_t packThresholdSizeInBytes = DEFAULT_PACK_THRESHOLD_SIZE_IN_BYTES);

        virtual const std::unordered_set<DistributedWorkerDescriptor>& Workers() const override;

        virtual const DistributedWorkerDescriptor& CurrentWorker() const override;

        // Creates a new distributed communicator comprising of a subset of the workers in this communicator
        virtual DistributedCommunicatorPtr SubGroup(const std::unordered_set<DistributedWorkerDescriptor>& subGroupWorkers) const override;

        // A collective communication API to concatenate values across each worker of this communicator. The concatenated values are only sent to the specified workers; for all others the returned Values are null
        virtual void Concatenate(
            const std::vector<ValuePtr>& values,
            std::vector<ValuePtr>& outValues,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        virtual void Concatenate(
            const std::vector<NDArrayViewPtr>& input,
            std::vector<NDArrayViewPtr>& output, const
            std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        virtual void Gather(
            const Dictionary& input,
            std::vector<DictionaryPtr>& output,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        // A collective communication API to aggregate values across each worker of this communicator. The aggregated values are only sent to the specified workers; for all others the returned Values are null
        virtual void AggregateInPlace(
            const std::vector<NDArrayViewPtr>& values,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        virtual void Aggregate(
            const std::vector<NDArrayViewPtr>& inValues,
            std::vector<NDArrayViewPtr>& outValues,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) override;

        virtual void Barrier() override;

        virtual ~MPICommunicatorImpl() {}

    private:
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
        std::vector<Buffer> m_intermediateCPUBuffers;

        DistributedWorkerDescriptor m_currentWorker;
        std::unordered_set<DistributedWorkerDescriptor> m_workers;

        // TODO: these two are always parallel, merge them together?
        std::vector<std::shared_ptr<Microsoft::MSR::CNTK::GPUDataTransferer>> m_gpuDataTransferers;

        // Threshold size of a gradient to be packed
        size_t m_packThresholdSizeInBytes;
        std::unique_ptr<Microsoft::MSR::CNTK::Matrix<float>> m_aggregationBufferFloat;
        std::unique_ptr<Microsoft::MSR::CNTK::Matrix<double>> m_aggregationBufferDouble;

        // NcclComm
        std::unique_ptr<Microsoft::MSR::CNTK::NcclComm> m_nccl;

    protected:
        DeviceDescriptor GetNonCPUDevice(const std::vector<NDArrayViewPtr>& values)
        {
            auto device = std::find_if(values.begin(), values.end(), [](const NDArrayViewPtr v) { return v->Device().Type() != DeviceKind::CPU; });
            return values.end() == device ? DeviceDescriptor::CPUDevice() : (*device)->Device();
        }

        size_t GetBufferSize(const NDArrayViewPtr& viewPtr)
        {
            return viewPtr->Shape().TotalSize() * DataTypeSize(viewPtr->GetDataType());
        }

        template <typename ElementType>
        std::shared_ptr<const Microsoft::MSR::CNTK::Matrix<ElementType>> GetMatrix(const NDArrayViewPtr& arrayView)
        {
            return arrayView->GetMatrix<ElementType>();
        }

        template <typename ElementType>
        std::shared_ptr<Microsoft::MSR::CNTK::Matrix<ElementType>> GetWritableMatrix(const NDArrayViewPtr& arrayView)
        {
            return arrayView->GetWritableMatrix<ElementType>();
        }

        void CheckWorkers(const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers);

        Microsoft::MSR::CNTK::MPIWrapperPtr m_mpi;

        bool ShouldCopyDataToCPU(NDArrayViewPtr inputValue);
        void CopyDataFromGPUToCPU(std::vector<NDArrayViewPtr>& inputValues);

        template <typename ElemType>
        std::unique_ptr<Microsoft::MSR::CNTK::Matrix<ElemType>> SetContinuousBuffer(std::vector<size_t>& packedGradientsIndex, size_t packedGradientsSizeInBytes,
            const std::vector<NDArrayViewPtr>& inputValues, const std::vector<NDArrayViewPtr>& outputValues,
            std::vector<NDArrayViewPtr>& valuesToAggregate, std::vector<NDArrayViewPtr>& valuesAfterAggregate);

        template <typename ElemType>
        void PackToContinuousBuffer(Microsoft::MSR::CNTK::Matrix<ElemType>* aggregationBuffer, std::vector<size_t>& packedGradientsIndex,
            const std::vector<NDArrayViewPtr>& inputValues, const std::vector<NDArrayViewPtr>& outputValues, std::vector<NDArrayViewPtr>& valuesToAggregate, std::vector<NDArrayViewPtr>& valuesAfterAggregate);

        template <typename ElemType>
        void UnpackFromContinuousBuffer(Microsoft::MSR::CNTK::Matrix<ElemType>* aggregationBuffer, const std::vector<NDArrayViewPtr>& outputValues, std::vector<size_t>& packedGradientsIndex);

        template <typename ElemType>
        void AllReduceGradients(ElemType* inputData, ElemType* outputData, size_t numElements, std::vector<MPI_Request> &allReduceRequests, bool dataOnCPU);
    };
}
