//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Config.h"
#include "MinibatchSource.h"
#include "HeapMemoryProvider.h"
#include "ReaderShim.h"
#include "Function.h"

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    MinibatchSourcePtr CreateCompositeMinibatchSource(const Dictionary& configuration)
    {
        return MinibatchSourcePtr(new CompositeMinibatchSource(configuration));
    }

    CompositeMinibatchSource::CompositeMinibatchSource(const Dictionary& configuration)
        : m_startNewEpoch(true), m_nextEpochIndex(0), m_prevMinibatchSize(0)
    {
        ConfigParameters config;
        std::wstringstream s;
        for (const auto& keyValuePair : *(configuration.m_dictionaryData))
        {
            AddConfigString(s, keyValuePair.first, keyValuePair.second, 0);
        }
        config.Parse(msra::strfun::utf8(s.str()));

        const wchar_t* epochSizeConfigurationKey = L"epochSize";
        if (!configuration.Contains(epochSizeConfigurationKey))
            InvalidArgument("'epochSize' value must be configured when constructing a CNTK built-in composite MinibatchSource!");

        m_epochSize = configuration[epochSizeConfigurationKey].GetValue<size_t>();

        typedef Reader*(*CreateCompositeDataReaderProc)(const ConfigParameters* parameters);
        CreateCompositeDataReaderProc createReaderProc = (CreateCompositeDataReaderProc)Plugin().Load(L"CompositeDataReader", "CreateCompositeDataReader");
        m_compositeDataReader.reset(createReaderProc(&config));

        auto compositeDataReaderStreamDescs = m_compositeDataReader->GetStreamDescriptions();
        for (auto streamDesc : compositeDataReaderStreamDescs)
            m_streamInfos.insert({ streamDesc->m_name, streamDesc->m_id, AsStorageFormat(streamDesc->m_storageType), AsDataType(streamDesc->m_elementType), AsNDShape(*(streamDesc->m_sampleLayout)) });
    }

    /*virtual*/ bool CompositeMinibatchSource::GetNextMinibatch(std::unordered_map<StreamInfo, std::pair<size_t, ValuePtr>>& minibatchData) /*override*/
    {
        // TODO: Support different minibatch sizes for different streams
        size_t requestedMinibatchSize = 0;
        for (const auto& val : minibatchData)
        {
            if (requestedMinibatchSize == 0)
                requestedMinibatchSize = val.second.first;
            else
            {
                if (requestedMinibatchSize != val.second.first)
                    LogicError("Different minibatch sizes across different input streams is currently unsupported!");
            }
        }

        if (requestedMinibatchSize == 0)
            InvalidArgument("GetNextMinibatch: Requested minibatch sizes must be > 0");

        if (m_startNewEpoch)
        {
            // TODO: Add support for distributed reading
            EpochConfiguration epochConfig = { 1, 0, requestedMinibatchSize, m_epochSize, m_nextEpochIndex, 0 };
            m_compositeDataReader->StartEpoch(epochConfig);
            m_prevMinibatchSize = requestedMinibatchSize;
        }

        if (requestedMinibatchSize != m_prevMinibatchSize)
            LogicError("GetNextMinibatch: Changing minibatch sizes across calls is currently unsupported");

        auto compositeReaderMinibatchData = m_compositeDataReader->ReadMinibatch();
        m_startNewEpoch = compositeReaderMinibatchData.m_endOfEpoch;
        if (m_startNewEpoch)
            m_nextEpochIndex++;

        auto compositeDataReaderStreamDescs = m_compositeDataReader->GetStreamDescriptions();
        size_t numStreams = compositeDataReaderStreamDescs.size();
        for (size_t i = 0; i < numStreams; ++i)
        {
            auto currentStreamDesc = compositeDataReaderStreamDescs[i];
            auto sampleShape = AsNDShape(*(currentStreamDesc->m_sampleLayout));
            auto minibatchDataEntryForCurrentStream = std::find_if(minibatchData.begin(), minibatchData.end(), [currentStreamDesc](const std::pair<StreamInfo, std::pair<size_t, ValuePtr>>& entry) {
                return entry.first.m_id == currentStreamDesc->m_id;
            });

            auto minibatchValuePtr = minibatchDataEntryForCurrentStream->second.second;
            if (compositeReaderMinibatchData.m_data.empty())
            {
                minibatchValuePtr = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(minibatchDataEntryForCurrentStream->first.m_elementType, sampleShape.AppendShape({ 0, 0 }), DeviceDescriptor::CPUDevice()));
                continue;
            }

            auto currentStreamMinibatchData = compositeReaderMinibatchData.m_data[i];

            if (currentStreamDesc->m_elementType == ElementType::tfloat)
            {
                auto dataMatrix = std::make_shared<Matrix<float>>(CPUDEVICE);
                size_t sampleSize = currentStreamDesc->m_sampleLayout->GetNumElements();

                // TODO: Eliminate the unnecessary CPU to CPU copy
                ReaderShim<float>::FillMatrixFromStream(currentStreamDesc->m_storageType, dataMatrix.get(), sampleSize, currentStreamMinibatchData);
                auto minibatchValueObject = CompositeFunction::GetValueObjectFromCNTKImplMatrixAndMBLayout<float>(sampleShape, *dataMatrix, currentStreamMinibatchData->m_layout, false);

                // TODO: Should slice off the supplied Value object instead of reallocating, in cases the actual minibatch 
                // size is smaller than the supplied storage in the Value object
                if ((minibatchValuePtr == nullptr) || (minibatchValuePtr->Data()->Shape() != minibatchValueObject->Data()->Shape()))
                    minibatchData[minibatchDataEntryForCurrentStream->first].second = minibatchValueObject;
                else
                    minibatchValuePtr->CopyFrom(*minibatchValueObject);
            }
            else
                LogicError("Double precision input data is currently unsupported by the CNTK built-in composite MinibatchSource!");
        }

        return true;
    }
}
