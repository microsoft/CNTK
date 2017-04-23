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
#include "ReaderConstants.h"
#include <tuple>
#include "Value.h"
#include "MPIWrapper.h"
#include "PerformanceProfiler.h"

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    const size_t MinibatchSource::DefaultRandomizationWindowInChunks = g_4GB / g_32MB;
    const size_t  MinibatchSource::InfinitelyRepeat = g_infinity;
    const size_t  MinibatchSource::FullDataSweep = g_dataSweep;


    const std::unordered_map<StreamInformation, MinibatchData>& MinibatchSource::GetNextMinibatch(size_t minibatchSizeInSamples, const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        return GetNextMinibatch(0, minibatchSizeInSamples, device);
    }

    const std::unordered_map<StreamInformation, MinibatchData>& MinibatchSource::GetNextMinibatch(size_t minibatchSizeInSequences, size_t minibatchSizeInSamples, const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        return GetNextMinibatch(minibatchSizeInSequences, minibatchSizeInSamples, 1, 0, device);
    }

    MinibatchSourceConfig::MinibatchSourceConfig(const std::vector<Deserializer>& deserializers, bool randomize/* = true*/) 
        : deserializers(deserializers)
    {
        if (!randomize) 
        {
            randomizationWindowInChunks = 0;
            randomizationWindowInSamples = 0;
        }
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
            RuntimeError("No stream found matching given name '%S'.", streamName.c_str());

        if (matchingStreamInfos.size() > 1)
            RuntimeError("Multiple streams (%d) found matching given name '%S'.", (int)matchingStreamInfos.size(), streamName.c_str());

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
            RuntimeError("No stream found matching given Variable '%S'.", variableToMatch.AsString().c_str());

        if (matchingStreamInfos.size() > 1)
            RuntimeError("Multiple streams (%d) found matching given Variable '%S'.", (int)matchingStreamInfos.size(), variableToMatch.AsString().c_str());

        return *(*(matchingStreamInfos.begin()));
    }

    MinibatchSourcePtr CreateCompositeMinibatchSource(const MinibatchSourceConfig& configuration)
    {
        return MinibatchSourcePtr(new CompositeMinibatchSource(configuration));
    }

    /*static*/ const std::wstring CompositeMinibatchSource::PositionAttributeName = L"minibatchSourcePosition";

    CompositeMinibatchSource::CompositeMinibatchSource(const MinibatchSourceConfig& configuration)
        : m_epochEndReached(false),
          m_prevMinibatchSize(0),
          m_maxNumSamplesToRead(configuration.maxSamples),
          m_maxNumSweepsToRead(configuration.maxSweeps),
          m_truncationLength(0),
          m_numWorkers(1),
          m_workerRank(0),
          m_restorePosition(0)
    {
        m_truncationLength = configuration.truncationLength;

        auto augmentedConfiguration = Internal::ToDictionary(configuration);

        ConfigParameters config;
        std::wstringstream s;
        for (const auto& keyValuePair : *(augmentedConfiguration.m_dictionaryData))
            AddConfigString(s, keyValuePair.first, keyValuePair.second, 0);

        config.Parse(msra::strfun::utf8(s.str()));

        typedef Reader*(*CreateCompositeDataReaderProc)(const ConfigParameters* parameters);
        CreateCompositeDataReaderProc createReaderProc = (CreateCompositeDataReaderProc)Plugin().Load(L"CompositeDataReader", "CreateCompositeDataReader");
        std::shared_ptr<Microsoft::MSR::CNTK::Reader> compositeDataReader(createReaderProc(&config));

        m_compositeDataReaderStreamDescs = compositeDataReader->GetStreamDescriptions();
        for (auto streamDesc : m_compositeDataReaderStreamDescs)
            m_streamInfos.insert({ streamDesc->m_name, streamDesc->m_id, AsStorageFormat(streamDesc->m_storageType), AsDataType(streamDesc->m_elementType), AsNDShape(*(streamDesc->m_sampleLayout)) });

        m_shim = std::shared_ptr<ReaderShim<float>>(new ReaderShim<float>(compositeDataReader), [](ReaderShim<float>* x) { x->Destroy(); });
        m_shim->Init(config);
    }

    /*virtual*/ const std::unordered_map<StreamInformation, MinibatchData>&
    CompositeMinibatchSource::GetNextMinibatch(size_t minibatchSizeInSequences,
                                               size_t minibatchSizeInSamples,
                                               size_t numberOfWorkers,
                                               size_t workerRank,
                                               const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/) /*override*/
    {
        auto profGetMinibatch = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainGetMinibatch);

        m_minibatchData.clear();

        if (!m_epochEndReached)
        {
            if (minibatchSizeInSequences != 0)
                LogicError("GetNextMinibatch: Specifying minibatch size in #sequences is currently unsupported");

            if (minibatchSizeInSamples == 0)
                InvalidArgument("GetNextMinibatch: Requested minibatch size must be > 0.");

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

                epochConfig.m_totalEpochSizeInSweeps = m_maxNumSweepsToRead;

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
                        LogicError("GetNextMinibatch: Input of type other than DataType::Float is currently unsupported by the CNTK built-in composite MinibatchSource!");
                }

                m_shim->StartEpoch(epochConfig, inputs);

                m_prevMinibatchSize = minibatchSizeInSamples;
                m_workerRank = workerRank;
                m_numWorkers = numberOfWorkers;
            }

            if (minibatchSizeInSamples != m_prevMinibatchSize || m_workerRank != workerRank || m_numWorkers != numberOfWorkers || m_restorePosition != 0)
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

                if (m_restorePosition != 0)
                {
                    m_shim->SetCurrentSamplePosition(m_restorePosition);
                    m_restorePosition = 0;
                }

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
                    m_minibatchData[currentStreamInfo] = { nullptr, 0, 0 };
                    continue;
                }

                if (s.m_elementType == DataType::Float)
                {
                    auto matrix = dynamic_pointer_cast<Matrix<float>>(input.matrix);
                    if (!matrix)
                        LogicError("GetNextMinibatch: Invalid matrix type.");

                    minibatchValuePtr = MakeSharedObject<PackedValue>(s.m_sampleLayout, Axis::DefaultInputVariableDynamicAxes(), matrix, input.pMBLayout, /*readOnly =*/ false);

                    size_t numSamples = input.pMBLayout->GetActualNumSamples();
                    size_t numSequences = input.pMBLayout->GetNumSequences();

                    m_minibatchData[currentStreamInfo] = { minibatchValuePtr, numSequences, numSamples, hasReachedSweepEnd };
                }
                else
                    LogicError("GetNextMinibatch: Input of type other than DataType::Float is currently unsupported by the CNTK built-in composite MinibatchSource!");
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

        // Need to reinitialize, we also have to remember the current position because StartEpoch
        // effectively resets it.
        // TODO: Remove call to StartEpoch - this API is legacy.
        m_restorePosition = checkpointedMinibatchSourcePosition;
        m_epochEndReached = false;
        m_prevMinibatchSize = 0;
    }

    /* static */ ImageTransform ReaderCrop(const wchar_t* cropType,
            int cropSize, float sideRatio, float areaRatio,
            float aspectRatio, const wchar_t* jitterType)
    {
        ImageTransform crop;
        crop.Add(L"type", L"Crop",
            L"cropType", cropType,
            L"cropSize", cropSize,
            L"sideRatio", sideRatio,
            L"areaRatio", areaRatio,
            L"aspectRatio", aspectRatio,
            L"jitterType", jitterType);
        return crop;
    }

    /* static */ ImageTransform ReaderScale(int width,
            int height, int channels, const wchar_t* interpolations,
            const wchar_t* scaleMode, int padValue)
    {
        ImageTransform scale;
        scale.Add(L"type", L"Scale",
            L"width", width,
            L"height", height,
            L"channels", channels,
            L"interpolations", interpolations,
            L"scaleMode", scaleMode,
            L"padValue", padValue);
        return scale;
    }

    /* static */ ImageTransform ReaderMean(const wchar_t* meanFile)
    {
        ImageTransform mean;
        mean.Add(L"type", L"Mean", L"meanFile", meanFile);
        return mean;
    }

    /* static */ ImageTransform ReaderColor(float brightnessRadius,
            float contrastRadius, float saturationRadius)
    {
        ImageTransform color;
        color.Add(L"type", L"Color",
            L"brightnessRadius", brightnessRadius,
            L"contrastRadius", contrastRadius,
            L"saturationRadius", saturationRadius);
        return color;
    }

    Deserializer ImageDeserializer(const std::wstring& fileName, const std::wstring& labelStreamName, size_t numLabels, const std::wstring& imageStreamName, const std::vector<ImageTransform>& transforms)
    {
        Deserializer img;
        std::vector<DictionaryValue> actualTransforms;
        std::transform(transforms.begin(), transforms.end(), std::back_inserter(actualTransforms), [](ImageTransform t) { return static_cast<DictionaryValue>(t); });
        Dictionary labeldim;
        labeldim[L"labelDim"] = numLabels;
        Dictionary xforms;
        xforms[L"transforms"] = actualTransforms;
        Dictionary input;
        input.Add(imageStreamName.c_str(), xforms, labelStreamName.c_str(), labeldim);
        img.Add(L"type", L"ImageDeserializer", L"file", fileName, L"input", input);
        return img;
    }

    Deserializer CTFDeserializer(const std::wstring& fileName, const std::vector<StreamConfiguration>& streams)
    {
        Deserializer ctf;
        Dictionary input;
        for (const auto& s : streams)
        {
            const auto& key = s.m_streamName;
            Dictionary stream;
            stream[L"dim"] = s.m_dim;
            stream[L"format"] = s.m_isSparse ? L"sparse" : L"dense";
            if (!s.m_streamAlias.empty())
                stream[L"alias"] = s.m_streamAlias;
            input[key] = stream;
        }
        ctf.Add(L"type", L"CNTKTextFormatDeserializer", L"file", fileName, L"input", input);
        return ctf;
    }

    Deserializer HTKFeatureDeserializer(const std::vector<HTKFeatureConfiguration>& streams)
    {
        Deserializer htk;
        Dictionary input;
        for (const auto& s : streams)
        {
            const auto& key = s.m_streamName;
            Dictionary stream;
            std::vector<DictionaryValue> ctxWindow = { DictionaryValue(s.m_left), DictionaryValue(s.m_right) };
            stream.Add(L"scpFile", s.m_scp, L"dim", s.m_dim, L"contextWindow", ctxWindow, L"expandToUtterance", s.m_broadcast);
            input[key] = stream;
        }
        htk.Add(L"type", L"HTKFeatureDeserializer", L"input", input);
        return htk;
    }

    Deserializer HTKMLFDeserializer(const std::wstring& streamName, const std::wstring& labelMappingFile, size_t dimension, const std::vector<std::wstring>& mlfFiles)
    {
        Deserializer htk;
        Dictionary stream;
        Dictionary labels;
        labels.Add(L"labelMappingFile", labelMappingFile, L"dim", dimension);
        std::vector<DictionaryValue> actualFiles;
        std::transform(mlfFiles.begin(), mlfFiles.end(), std::back_inserter(actualFiles), [](const std::wstring& s) {return static_cast<DictionaryValue>(s); });
        if (actualFiles.size() > 1)
            labels[L"mlfFileList"] = actualFiles;
        else if (actualFiles.size() == 1)
            labels[L"mlfFile"] = actualFiles[0];
        else
            LogicError("HTKMLFDeserializer: No mlf files were specified");
        stream[streamName] = labels;
        htk.Add(L"type", L"HTKMLFDeserializer", L"input", stream);
        return htk;
    }

    namespace Internal 
    {

        void Validate(const MinibatchSourceConfig& configuration)
        {
            if (configuration.maxSamples != MinibatchSource::InfinitelyRepeat && configuration.maxSweeps != MinibatchSource::InfinitelyRepeat)
                LogicError("MinibatchSourceConfig: max samples and max sweeps are mutually exclusive options"
                    " and cannot have non-default values at the same time.");

            if (configuration.randomizationWindowInChunks != 0 && configuration.randomizationWindowInSamples != 0)
                LogicError("MinibatchSourceConfig: randomization window in chunks and randomization window in samples"
                    " are mutually exclusive options and cannot have non-zero values at the same time.");

            if (configuration.isFrameModeEnabled && configuration.truncationLength != 0)
                LogicError("MinibatchSourceConfig: truncation and frame mode are mutually exclusive options.");
        }

        Dictionary ToDictionary(const ::CNTK::MinibatchSourceConfig& configuration)
        {
            Validate(configuration);

            Dictionary augmentedConfiguration;

            if (configuration.randomizationWindowInSamples != 0)
            {
                augmentedConfiguration[L"randomize"] = true;
                augmentedConfiguration[L"randomizationWindow"] = configuration.randomizationWindowInSamples;
                augmentedConfiguration[L"sampleBasedRandomizationWindow"] = true;
            }
            else if (configuration.randomizationWindowInChunks != 0) 
            {
                augmentedConfiguration[L"randomize"] = true;
                augmentedConfiguration[L"randomizationWindow"] = configuration.randomizationWindowInChunks;
                augmentedConfiguration[L"sampleBasedRandomizationWindow"] = false;
            }
            else 
            {
                augmentedConfiguration[L"randomize"] = false;
            }

            if (configuration.truncationLength != 0)
            {
                augmentedConfiguration[L"truncated"] = true;
                augmentedConfiguration[L"truncationLength"] = configuration.truncationLength;
            }

            augmentedConfiguration[L"frameMode"] = configuration.isFrameModeEnabled;
            augmentedConfiguration[L"multiThreadedDeserialization"] = configuration.isMultithreaded;
            augmentedConfiguration[L"traceLevel"] = static_cast<size_t>(configuration.traceLevel);

            // The CNTK reader implementation requires for each deserializer both the module and deserializer type be specified
            // This is redundant and the V2 API users will just specify type from which the module is automatically inferred
            // TODO: This should be done in the same manner for CNTK exe as well.
            vector<DictionaryValue> deserializers;
            for (auto deserializerConfig : configuration.deserializers)
            {
                static const std::unordered_map<std::wstring, std::wstring> deserializerTypeNameToModuleNameMap = {
                    { L"CNTKTextFormatDeserializer", L"CNTKTextFormatReader" },
                    { L"ImageDeserializer",          L"ImageReader" },
                    { L"HTKFeatureDeserializer",     L"HTKDeserializers" },
                    { L"HTKMLFDeserializer",         L"HTKDeserializers" },
                };

                auto deserializerTypeName = deserializerConfig[L"type"].Value<std::wstring>();
                if (deserializerTypeName == L"ImageDeserializer")
                {
                    // Add a transpose transform since the image data in read in HWC (CWH in column major format) form while 
                    // the CNTK convolution engive supports WHC (in column-major format)
                    auto& inputStreamsConfig = deserializerConfig[L"input"].Value<Dictionary>();
                    for (auto& inputStreamEntry : inputStreamsConfig)
                    {
                        auto& inputStreamConfig = inputStreamEntry.second.Value<Dictionary>();
                        if (inputStreamConfig.Contains(L"transforms"))
                        {
                            auto& transforms = inputStreamConfig[L"transforms"].Value<std::vector<DictionaryValue>>();

                            // Add the transpose transform
                            Dictionary transposeTransform;
                            transposeTransform[L"type"] = L"Transpose";
                            transforms.push_back(DictionaryValue(transposeTransform));
                        }
                    }
                }

                if (deserializerTypeNameToModuleNameMap.find(deserializerTypeName) == deserializerTypeNameToModuleNameMap.end())
                    InvalidArgument("Unknown deserializer type '%S' specified for CNTK built-in composite MinibatchSource construction.", deserializerTypeName.c_str());

                deserializerConfig[L"module"] = deserializerTypeNameToModuleNameMap.at(deserializerTypeName);
                deserializers.push_back(deserializerConfig);
            }

            augmentedConfiguration[L"deserializers"] = deserializers;

            return augmentedConfiguration;
        }
    }
}
