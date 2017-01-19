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
#include <tuple>
#include "Value.h"
#include "MPIWrapper.h"

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    const std::unordered_map<StreamInformation, MinibatchData>& MinibatchSource::GetNextMinibatch(size_t minibatchSizeInSamples, const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        return GetNextMinibatch(0, minibatchSizeInSamples, device);
    }

    const std::unordered_map<StreamInformation, MinibatchData>& MinibatchSource::GetNextMinibatch(size_t minibatchSizeInSequences, size_t minibatchSizeInSamples, const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        return GetNextMinibatch(minibatchSizeInSequences, minibatchSizeInSamples, 1, 0, device);
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

    /*static*/ const std::wstring CompositeMinibatchSource::PositionAttributeName = L"minibatchSourcePosition";

    CompositeMinibatchSource::CompositeMinibatchSource(const Dictionary& configuration)
        : m_epochEndReached(false),
          m_prevMinibatchSize(0),
          m_maxNumSamplesToRead(MinibatchSource::InfinitelyRepeat),
          m_randomizedWindow(MinibatchSource::DefaultRandomizationWindow),
          m_truncationLength(0),
          m_numWorkers(1),
          m_workerRank(0)
    {
        // The CNTK reader implementation requires for each deserializer both the module and deserializer type be specified
        // This is redundant and the V2 API users will just specify type from which the module is automatically inferred
        // TODO: This should be done in the same manner for CNTK exe as well.
        Dictionary augmentedConfiguration = configuration;
        auto& deserializerConfigurations = augmentedConfiguration[L"deserializers"].Value<std::vector<DictionaryValue>>();
        for (auto& deserializerConfig : deserializerConfigurations)
        {
            static const std::unordered_map<std::wstring, std::wstring> deserializerTypeNameToModuleNameMap = {
                { L"CNTKTextFormatDeserializer", L"CNTKTextFormatReader" },
                { L"ImageDeserializer",          L"ImageReader"          },
                { L"HTKFeatureDeserializer",     L"HTKDeserializers"     },
                { L"HTKMLFDeserializer",         L"HTKDeserializers"     },
            };

            auto& deserializerConfigDict = deserializerConfig.Value<Dictionary>();
            auto deserializerTypeName = deserializerConfigDict[L"type"].Value<std::wstring>();
            if (deserializerTypeName == L"ImageDeserializer")
            {
                // Add a transpose transform since the image data in read in HWC (CWH in column major format) form while 
                // the CNTK convolution engive supports WHC (in column-major format)
                auto& inputStreamsConfig = deserializerConfigDict[L"input"].Value<Dictionary>();
                auto& streamsMap = *(inputStreamsConfig.m_dictionaryData);
                for (auto& inputStreamEntry : streamsMap)
                {
                    auto& inputStreamConfig = inputStreamEntry.second.Value<Dictionary>();
                    if (inputStreamConfig.Contains(L"transforms"))
                    {
                        auto& transforms = inputStreamConfig[L"transforms"].Value<std::vector<DictionaryValue>>();

                        // Add the transpose transform
                        Dictionary transposeTransform;
                        transposeTransform[L"type"] = L"Transpose";
                        transforms.push_back(transposeTransform);
                    }
                }

            }

            if (deserializerTypeNameToModuleNameMap.find(deserializerTypeName) == deserializerTypeNameToModuleNameMap.end())
                InvalidArgument("Unknown deserializer type (%S)", deserializerTypeName.c_str());

            deserializerConfigDict[L"module"] = deserializerTypeNameToModuleNameMap.at(deserializerTypeName);
        }

        ConfigParameters config;
        std::wstringstream s;
        for (const auto& keyValuePair : *(augmentedConfiguration.m_dictionaryData))
            AddConfigString(s, keyValuePair.first, keyValuePair.second, 0);

        config.Parse(msra::strfun::utf8(s.str()));

        const wchar_t* epochSizeConfigurationKey = L"epochSize";
        if (augmentedConfiguration.Contains(epochSizeConfigurationKey))
            m_maxNumSamplesToRead = augmentedConfiguration[epochSizeConfigurationKey].Value<size_t>();

        const wchar_t* randomizedWindowConfigurationKey = L"randomizationWindow";
        if (augmentedConfiguration.Contains(randomizedWindowConfigurationKey))
            m_randomizedWindow = augmentedConfiguration[randomizedWindowConfigurationKey].Value<size_t>();

        if (m_randomizedWindow == MinibatchSource::DefaultRandomizationWindow)
            m_randomizedWindow = randomizeAuto;

        const wchar_t* truncatedConfigurationKey = L"truncated";
        const wchar_t* truncationLengthConfigurationKey = L"truncationLength";
        if (augmentedConfiguration.Contains(truncatedConfigurationKey) &&
            augmentedConfiguration[truncatedConfigurationKey].Value<bool>() &&
            augmentedConfiguration.Contains(truncationLengthConfigurationKey))
        {
            m_truncationLength = augmentedConfiguration[truncationLengthConfigurationKey].Value<size_t>();
        }

        typedef Reader*(*CreateCompositeDataReaderProc)(const ConfigParameters* parameters);
        CreateCompositeDataReaderProc createReaderProc = (CreateCompositeDataReaderProc)Plugin().Load(L"CompositeDataReader", "CreateCompositeDataReader");
        std::shared_ptr<Microsoft::MSR::CNTK::Reader> compositeDataReader(createReaderProc(&config));

        m_compositeDataReaderStreamDescs = compositeDataReader->GetStreamDescriptions();
        for (auto streamDesc : m_compositeDataReaderStreamDescs)
            m_streamInfos.insert({ streamDesc->m_name, streamDesc->m_id, AsStorageFormat(streamDesc->m_storageType), AsDataType(streamDesc->m_elementType), AsNDShape(*(streamDesc->m_sampleLayout)) });

        m_shim = std::shared_ptr<ReaderShim<float>>(new ReaderShim<float>(compositeDataReader), [](ReaderShim<float>* x) { x->Destroy(); });
        m_shim->Init(config);

        const wchar_t* numWorkersConfigurationKey = L"numWorkers";
        if (configuration.Contains(numWorkersConfigurationKey))
        {
            m_numWorkers = configuration[numWorkersConfigurationKey].Value<size_t>();

            const wchar_t* workerRankConfigurationKey = L"workerRank";
            if (configuration.Contains(workerRankConfigurationKey))
            {
                m_workerRank = configuration[workerRankConfigurationKey].Value<size_t>();
            }
            if (m_workerRank > m_numWorkers - 1)
            {
                LogicError("Invalid worker rank %lu (numWorkers %lu)", m_workerRank, m_numWorkers);
            }
        }
    }

    /*virtual*/ const std::unordered_map<StreamInformation, MinibatchData>&
    CompositeMinibatchSource::GetNextMinibatch(size_t minibatchSizeInSequences,
                                               size_t minibatchSizeInSamples,
                                               size_t numberOfWorkers,
                                               size_t workerRank,
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
                EpochConfiguration epochConfig;
                epochConfig.m_numberOfWorkers = numberOfWorkers;
                epochConfig.m_workerRank = workerRank;
                epochConfig.m_minibatchSizeInSamples = minibatchSizeInSamples;
                epochConfig.m_truncationSize = m_truncationLength;
                epochConfig.m_allowMinibatchesToCrossSweepBoundaries = true;

                if (m_maxNumSamplesToRead == MinibatchSource::FullDataSweep)
                {
                    epochConfig.m_totalEpochSizeInSamples = Microsoft::MSR::CNTK::requestDataSize;
                }
                else if (m_maxNumSamplesToRead == MinibatchSource::InfinitelyRepeat)
                {
                    // Setting big value, but not the max in order to aviod bit overflow.
                    epochConfig.m_totalEpochSizeInSamples = std::numeric_limits<size_t>::max() / 2;
                }
                else 
                {
                    epochConfig.m_totalEpochSizeInSamples = m_maxNumSamplesToRead;
                }

                epochConfig.m_epochIndex = 0;

                m_matrices.clear();

                std::unordered_set<InputStreamDescription> inputs;
                for (const auto& s : m_streamInfos)
                {
                    auto inputStreamDescription = GetInputStreamDescription(s, device);
                    inputs.insert(inputStreamDescription);

                    if (s.m_elementType == DataType::Float)
                    {
                        auto iter = std::find_if(m_compositeDataReaderStreamDescs.begin(), m_compositeDataReaderStreamDescs.end(), [s](StreamDescriptionPtr& streamInfo) {
                            return streamInfo->m_id == s.m_id;
                        });
                        assert(iter != m_compositeDataReaderStreamDescs.end());

                        m_matrices.AddInput(
                            s.m_name,
                            std::make_shared<Matrix<float>>(0, 0, inputStreamDescription.GetDeviceId(), inputStreamDescription.GetMatrixType(), inputStreamDescription.GetMatrixFormat()),
                            std::make_shared<MBLayout>(),
                            *(*iter)->m_sampleLayout);
                    }
                    else
                        LogicError("Input data of type other than DataType::Float is currently unsupported by the CNTK built-in composite MinibatchSource!");
                }

                m_shim->StartEpoch(epochConfig, inputs);
                m_prevMinibatchSize = minibatchSizeInSamples;
                m_workerRank = workerRank;
                m_numWorkers = numberOfWorkers;
            }

            if (minibatchSizeInSamples != m_prevMinibatchSize || m_workerRank != workerRank || m_numWorkers != numberOfWorkers)
            {
                std::map<std::wstring, int> inputDescriptions;
                for (const auto& s : m_streamInfos)
                    inputDescriptions[s.m_name] = AsCNTKImplDeviceId(device);

                ReaderConfiguration newConfig;
                newConfig.m_numberOfWorkers = numberOfWorkers;
                newConfig.m_workerRank = workerRank;
                newConfig.m_minibatchSizeInSamples = minibatchSizeInSamples;
                newConfig.m_truncationSize = m_truncationLength;
                newConfig.m_allowMinibatchesToCrossSweepBoundaries = true;

                m_shim->SetConfiguration(newConfig, inputDescriptions);

                m_prevMinibatchSize = minibatchSizeInSamples;
                m_workerRank = workerRank;
                m_numWorkers = numberOfWorkers;
            }

            auto hasData = m_shim->GetMinibatch(m_matrices);
            m_epochEndReached = m_shim->IsEndOfEpoch();

            if (m_epochEndReached && !hasData)
                return m_minibatchData;

            bool hasReachedSweepEnd = m_shim->IsEndOfSweep();

            for (const auto& s: m_streamInfos)
            {
                auto input = m_matrices.GetInput(s.m_name);
                auto& currentStreamInfo = s;

                ValuePtr minibatchValuePtr;
                if (!hasData)
                {
                    m_minibatchData[currentStreamInfo] = { 0, 0, nullptr };
                    continue;
                }

                if (s.m_elementType == DataType::Float)
                {
                    auto matrix = dynamic_pointer_cast<Matrix<float>>(input.matrix);
                    if (!matrix)
                        LogicError("Invalid matrix type.");

                    minibatchValuePtr = MakeSharedObject<PackedValue>(s.m_sampleLayout, matrix, input.pMBLayout, /*readOnly =*/ false);

                    size_t numSamples = input.pMBLayout->GetActualNumSamples();
                    size_t numSequences = input.pMBLayout->GetNumSequences();

                    m_minibatchData[currentStreamInfo] = { minibatchValuePtr, numSequences, numSamples, hasReachedSweepEnd };
                }
                else
                    LogicError("Input data of type other than DataType::Float is currently unsupported by the CNTK built-in composite MinibatchSource!");
            }
        }

        return m_minibatchData;
    }

    /*virtual*/ Dictionary CompositeMinibatchSource::GetCheckpointState() const /*override*/
    {
        Dictionary checkpointState;
        checkpointState[PositionAttributeName] = m_shim->GetCurrentSamplePosition();
        return checkpointState;
    }

    /*virtual*/ void CompositeMinibatchSource::RestoreFromCheckpoint(const Dictionary& checkpoint) /*override*/
    {
        auto checkpointedMinibatchSourcePosition = checkpoint[PositionAttributeName].Value<size_t>();
        m_shim->SetCurrentSamplePosition(checkpointedMinibatchSourcePosition);
    }
}
