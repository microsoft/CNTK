//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "DistributedCommunicator.h"
#include <functional>


using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    MPICommunicatorImpl::MPICommunicatorImpl()
    {
        m_mpi = MPIWrapper::GetInstance(true);
    }

    /*virtual*/ std::vector<ValuePtr> MPICommunicatorImpl::Aggregate(const std::vector<ValuePtr>& values,
                                                                   const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        std::vector<ValuePtr> outputValues(values.size());
        for (const auto& inputValue : values)
        {
            outputValues.push_back(inputValue->DeepClone()); // move to cpu -- do async call  here + .then(xxx());
        }
        AggregateInPlace(outputValues, sendToWorkers);
    }

    /*virtual*/ void MPICommunicatorImpl::AggregateInPlace(const std::vector<ValuePtr>& values,
                                                           const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        // if a value is on a gpu -> copy to cpu -> aggregate -> copy back.
        auto numValues = values.size();
        std::vector<std::future<void>> futures(numValues);
        for (auto i = 0; i < numValues; ++i)
        {
            auto& value = values[i];
            // refactor this lambda into a separate methods XXX
            futures.push_back(std::async(std::launch::async,
                [this, &value]()
                {
                    // if value is on a gpu, copy it to cpu
                    MPI_Request request;
                    auto& view = value->Data();
                    auto numElements = view->Shape().TotalSize();
                    auto dataType = view->GetDataType();

                    if (dataType == DataType::Float)
                    {
                        float* data = view->WritableDataBuffer<float>();
                        m_mpi->AllReduceAsync<float>(data, numElements, &request);
                    }
                    else if (dataType == DataType::Double)
                    {
                        double* data = view->WritableDataBuffer<double>();
                        m_mpi->AllReduceAsync<double>(data, numElements, &request);
                    }
                    else
                        LogicError("Unknown DataType");

                    // Wait for the allreduce operations to finish
                    m_mpi->Wait(&request);

                   // if needed copy from cpu to gpu.
                }));
        }

        // Wait for the allreduce operations to finish and initiate transfer back to the GPU if needed
        for (auto i = 0; i < numValues; ++i)
        {
            // check valid
            futures[i].wait();
        }
    }

    /*virtual*/ std::future<void> MPICommunicatorImpl::AggregateInPlaceAsync(const std::vector<ValuePtr>& values,
                                                                             const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        return std::async(std::launch::async, [this, &values, &sendToWorkers](){ /*before starting a new aggregation, make sure the old one is complete*/ this->AggregateInPlace(values, sendToWorkers); });
    }
}