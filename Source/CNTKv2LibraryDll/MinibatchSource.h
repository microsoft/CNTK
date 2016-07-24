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

        virtual const std::unordered_set<StreamInfo>& StreamInfos() override { return m_streamInfos; }

        virtual bool GetNextMinibatch(std::unordered_map<StreamInfo, std::pair<size_t, ValuePtr>>& minibatchData) override;

    private: 
        std::unordered_set<StreamInfo> m_streamInfos;
        std::shared_ptr<Microsoft::MSR::CNTK::Reader> m_compositeDataReader;
        bool m_startNewEpoch;
        size_t m_nextEpochIndex;
        size_t m_prevMinibatchSize;
        size_t m_epochSize;
    };
}

