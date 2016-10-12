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

namespace CNTK
{
    class CompositeMinibatchSource final : public MinibatchSource
    {
    public:
        CompositeMinibatchSource(const Dictionary& configuration);

        virtual const std::unordered_set<StreamInformation>& StreamInfos() override { return m_streamInfos; }

        virtual const std::unordered_map<StreamInformation, MinibatchData>& GetNextMinibatch(size_t minibatchSizeInSamples,
                                                                                             size_t minibatchSizeInSequences,
                                                                                             const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice()) override;

    private: 
        std::unordered_set<StreamInformation> m_streamInfos;
        bool m_epochEndReached;
        size_t m_prevMinibatchSize;
        size_t m_epochSize;
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
