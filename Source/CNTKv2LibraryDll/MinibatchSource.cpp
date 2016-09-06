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
#include <tuple>

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    const std::unordered_map<StreamInformation, MinibatchData>& MinibatchSource::GetNextMinibatch(size_t minibatchSizeInSamples, const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        return GetNextMinibatch(0, minibatchSizeInSamples, device);
    }

    const StreamInformation& MinibatchSource::StreamInfo(const std::wstring& streamName)
    {
        std::unordered_set<const StreamInformation*> matchingStreamInfos;
        auto& allStreamInfos = StreamInfos();
        for (auto& streamInfo : allStreamInfos)
        {
            if (streamInfo.m_name == streamName)
                matchingStreamInfos.insert(&streamInfo);
        }

        if (matchingStreamInfos.empty())
            RuntimeError("No stream found matching given name");

        if (matchingStreamInfos.size() > 1)
            RuntimeError("Multiple streams found matching given name");

        return *(*(matchingStreamInfos.begin()));
    }

    const StreamInformation& MinibatchSource::StreamInfo(const Variable& variableToMatch)
    {
        std::unordered_set<const StreamInformation*> matchingStreamInfos;
        auto& allStreamInfos = StreamInfos();
        for (auto& streamInfo : allStreamInfos)
        {
            bool streamHasSparseData = (streamInfo.m_storageFormat != StorageFormat::Dense);
            if ((streamInfo.m_elementType == variableToMatch.GetDataType()) && (streamInfo.m_sampleLayout == variableToMatch.Shape()) && (streamHasSparseData == variableToMatch.IsSparse()))
                matchingStreamInfos.insert(&streamInfo);
        }

        if (matchingStreamInfos.empty())
            RuntimeError("No stream found matching given Variable's attributes");

        if (matchingStreamInfos.size() > 1)
            RuntimeError("Multiple streams found matching given Variable's attributes");

        return *(*(matchingStreamInfos.begin()));
    }

    MinibatchSourcePtr CreateCompositeMinibatchSource(const Dictionary& configuration)
    {
        return MinibatchSourcePtr(new CompositeMinibatchSource(configuration));
    }

    CompositeMinibatchSource::CompositeMinibatchSource(const Dictionary& configuration)
        : m_epochEndReached(false), m_prevMinibatchSize(0), m_epochSize(SIZE_MAX)
    {
        // The CNTK reader implementation requires for each deserializer both the module and deserializer type be specified
        // This is redundant and the V2 API users will just specify type from which the module is automatically inferred
        Dictionary augmentedConfiguration = configuration;
        auto& deserializerConfigurations = augmentedConfiguration[L"deserializers"].Value<std::vector<DictionaryValue>>();
        for (auto& deserializerConfig : deserializerConfigurations)
        {
            static const std::unordered_map<std::wstring, std::wstring> deserializerTypeNameToModuleNameMap = {
                { L"CNTKTextFormatDeserializer", L"CNTKTextFormatReader" },
                { L"ImageDeserializer",          L"ImageReader"          },
            };

            auto& deserializerConfigDict = deserializerConfig.Value<Dictionary>();
            auto deserializerTypeName = deserializerConfigDict[L"type"].Value<std::wstring>();
            deserializerConfigDict[L"module"] = deserializerTypeNameToModuleNameMap.at(deserializerTypeName);
        }

        ConfigParameters config;
        std::wstringstream s;
        for (const auto& keyValuePair : *(augmentedConfiguration.m_dictionaryData))
            AddConfigString(s, keyValuePair.first, keyValuePair.second, 0);

        config.Parse(msra::strfun::utf8(s.str()));

        const wchar_t* epochSizeConfigurationKey = L"epochSize";
        if (augmentedConfiguration.Contains(epochSizeConfigurationKey))
            m_epochSize = augmentedConfiguration[epochSizeConfigurationKey].Value<size_t>();

        if (m_epochSize == 0)
            m_epochSize = Microsoft::MSR::CNTK::requestDataSize;

        typedef Reader*(*CreateCompositeDataReaderProc)(const ConfigParameters* parameters);
        CreateCompositeDataReaderProc createReaderProc = (CreateCompositeDataReaderProc)Plugin().Load(L"CompositeDataReader", "CreateCompositeDataReader");
        m_compositeDataReader.reset(createReaderProc(&config));

        auto compositeDataReaderStreamDescs = m_compositeDataReader->GetStreamDescriptions();
        for (auto streamDesc : compositeDataReaderStreamDescs)
            m_streamInfos.insert({ streamDesc->m_name, streamDesc->m_id, AsStorageFormat(streamDesc->m_storageType), AsDataType(streamDesc->m_elementType), AsNDShape(*(streamDesc->m_sampleLayout)) });
    }

    /*virtual*/ const std::unordered_map<StreamInformation, MinibatchData>&
    CompositeMinibatchSource::GetNextMinibatch(size_t minibatchSizeInSequences,
                                               size_t minibatchSizeInSamples,
                                               const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/) /*override*/
    {
        m_minibatchData.clear();

        if (!m_epochEndReached)
        {
            if (minibatchSizeInSequences != 0)
                LogicError("Specifying minibatch size in #sequences is currently unsupported");

            if (minibatchSizeInSamples == 0)
                InvalidArgument("GetNextMinibatch: Requested minibatch sizes must be > 0");

            if (m_prevMinibatchSize == 0)
            {
                // TODO: Add support for distributed reading
                EpochConfiguration epochConfig = { 1, 0, minibatchSizeInSamples, m_epochSize, 0, 0 };

                std::map<std::wstring, int> requiredStreams;
                for (const auto& s : m_streamInfos)
                    // Allocating all on CPU for now.
                    requiredStreams[s.m_name] = -1;

                m_compositeDataReader->StartEpoch(epochConfig, requiredStreams);
                m_prevMinibatchSize = minibatchSizeInSamples;
            }

            if (minibatchSizeInSamples != m_prevMinibatchSize)
                LogicError("GetNextMinibatch: Changing minibatch sizes across calls is currently unsupported");

            auto compositeReaderMinibatchData = m_compositeDataReader->ReadMinibatch();
            m_epochEndReached = compositeReaderMinibatchData.m_endOfEpoch;

            auto& streamInfos = StreamInfos();
            auto compositeDataReaderStreamDescs = m_compositeDataReader->GetStreamDescriptions();
            size_t numStreams = compositeDataReaderStreamDescs.size();
            for (size_t i = 0; i < numStreams; ++i)
            {
                auto currentStreamDesc = compositeDataReaderStreamDescs[i];
                auto iter = std::find_if(streamInfos.begin(), streamInfos.end(), [currentStreamDesc](const StreamInformation& streamInfo) {
                    return streamInfo.m_id == currentStreamDesc->m_id;
                });

                if (iter == streamInfos.end())
                    continue;

                auto& currentStreamInfo = *iter;
                auto sampleShape = AsNDShape(*(currentStreamDesc->m_sampleLayout));

                ValuePtr minibatchValuePtr;
                if (compositeReaderMinibatchData.m_data.empty())
                {
                    minibatchValuePtr = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(currentStreamInfo.m_elementType, sampleShape.AppendShape({ 0, 0 }), DeviceDescriptor::CPUDevice()));
                    continue;
                }

                auto currentStreamMinibatchData = compositeReaderMinibatchData.m_data[i];
                if (currentStreamDesc->m_elementType == ElementType::tfloat)
                {
                    auto CNTKMatrixType = (currentStreamDesc->m_storageType == StorageType::dense) ? DENSE : SPARSE;
                    auto CNTKMatrixFormat = (currentStreamDesc->m_storageType == StorageType::dense) ? matrixFormatDense : matrixFormatSparseCSC;
                    auto dataMatrix = std::make_shared<Matrix<float>>(0, 0, CPUDEVICE, CNTKMatrixType, CNTKMatrixFormat);
                    size_t sampleSize = currentStreamDesc->m_sampleLayout->GetNumElements();

                    // TODO: Eliminate the unnecessary CPU to CPU copy
                    ReaderShim<float>::FillMatrixFromStream(currentStreamDesc->m_storageType, dataMatrix.get(), sampleSize, currentStreamMinibatchData);
                    minibatchValuePtr = CompositeFunction::GetValueObjectFromCNTKImplMatrixAndMBLayout<float>(sampleShape, *dataMatrix, currentStreamMinibatchData->m_layout, false);

                    size_t numSamples = currentStreamMinibatchData->m_layout->GetActualNumSamples();
                    size_t numSequences = currentStreamMinibatchData->m_layout->GetNumSequences();

                    m_minibatchData[currentStreamInfo] = { numSequences, numSamples, minibatchValuePtr };
                }
                else
                    LogicError("Input data of type other than DataType::Float is currently unsupported by the CNTK built-in composite MinibatchSource!");
            }
        }

        return m_minibatchData;
    }
}
