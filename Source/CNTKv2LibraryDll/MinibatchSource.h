//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Reader.h"

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
        std::shared_ptr<Microsoft::MSR::CNTK::Reader> m_compositeDataReader;
        bool m_epochEndReached;
        size_t m_prevMinibatchSize;
        size_t m_epochSize;
        std::unordered_map<StreamInformation, MinibatchData> m_minibatchData;
    };
}
