//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Reader.h"
#include "ReaderShim.h"
#include "DataReader.h"

namespace CNTK
{
    class CompositeMinibatchSource final : public MinibatchSource
    {
        static const std::wstring PositionAttributeName;
        static const std::wstring DistributedAfterSampleCountAttributeName;

    public:
        CompositeMinibatchSource(const Dictionary& configuration);

        virtual const std::unordered_set<StreamInformation>& StreamInfos() override { return m_streamInfos; }

        virtual const std::unordered_map<StreamInformation, MinibatchData>& GetNextMinibatch(size_t minibatchSizeInSamples,
                                                                                             size_t minibatchSizeInSequences,
                                                                                             const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice()) override;

        virtual Dictionary GetCheckpointState() const override;
        virtual void RestoreFromCheckpoint(const Dictionary& checkpoint) override;

        virtual bool IsDistributed() const override
        {
            return m_shim->GetCurrentSamplePosition() >= m_distributedAfterSampleCount;
        }

    private:
        static Microsoft::MSR::CNTK::InputStreamDescription GetInputStreamDescription(const StreamInformation& s, const DeviceDescriptor& device)
        {
            assert(s.m_storageFormat == StorageFormat::Dense || s.m_storageFormat == StorageFormat::SparseCSC);
            auto CNTKdeviceId = AsCNTKImplDeviceId(device);
            auto CNTKMatrixType = s.m_storageFormat == StorageFormat::Dense ? Microsoft::MSR::CNTK::MatrixType::DENSE : Microsoft::MSR::CNTK::MatrixType::SPARSE;
            auto CNTKMatrixFormat = AsCNTKImplMatrixFormat(s.m_storageFormat);
            return Microsoft::MSR::CNTK::InputStreamDescription(s.m_name, CNTKdeviceId, CNTKMatrixType, CNTKMatrixFormat);
        }

    private: 
        std::unordered_set<StreamInformation> m_streamInfos;
        bool m_epochEndReached;
        bool m_distributed;
        size_t m_numWorkers;
        size_t m_workerRank;
        size_t m_distributedAfterSampleCount;
        size_t m_prevMinibatchSize;
        size_t m_epochSize;
		size_t m_randomizedWindow;
        size_t m_truncationLength;
        std::unordered_map<StreamInformation, MinibatchData> m_minibatchData;
        std::vector<Microsoft::MSR::CNTK::StreamDescriptionPtr> m_compositeDataReaderStreamDescs;

        // For now reusing the shim to allow prefetch.
        // Please only use a subset of the shim interface that includes
        // Init()/StartEpoch()/GetMinibatch()/IsEndOfEpoch()
        // Shim will be deleted in the future versions.
        std::shared_ptr<Microsoft::MSR::CNTK::ReaderShim<float>> m_shim;
        Microsoft::MSR::CNTK::StreamMinibatchInputs m_matrices;
    };
}
