//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Config.h"
#include "MinibatchSource.h"
#include "ReaderShim.h"
#include "Reader.h"
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
    const static std::wstring DeserializersProperty = L"_deserializers";

    const std::unordered_map<StreamInformation, MinibatchData>& MinibatchSource::GetNextMinibatch(size_t minibatchSizeInSamples, const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        return GetNextMinibatch(0, minibatchSizeInSamples, device);
    }

    const std::unordered_map<StreamInformation, MinibatchData>& MinibatchSource::GetNextMinibatch(size_t minibatchSizeInSequences, size_t minibatchSizeInSamples, const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        return GetNextMinibatch(minibatchSizeInSequences, minibatchSizeInSamples, 1, 0, device);
    }

    template<typename DataType>
    std::wstring pair_to_colon_format(const pair<DataType, DataType>& pair)
    {
        std::wostringstream str;
        str << pair.first << L":" << pair.second;
        return str.str();
    }

    // Some deserializers can contains other deserializers inside (i.e. htk), this function will flatten them.
    inline static std::vector<Deserializer> Flatten(const std::vector<Deserializer>& deserializers)
    {
        std::vector<Deserializer> flattened;
        for (const auto& d : deserializers)
        {
            if (d.Contains(DeserializersProperty) && d.Size() == 1)
            {
                auto inner = d[DeserializersProperty].Value<std::vector<DictionaryValue>>();
                for (auto& i : inner)
                    flattened.push_back(i.Value<Dictionary>());
            }
            else
                flattened.push_back(d);
        }

        return flattened;
    }

    MinibatchSourceConfig::MinibatchSourceConfig(const std::vector<Deserializer>& deserializers, bool randomize/* = true*/)
        : deserializers(Flatten(deserializers))
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

    inline std::map<std::wstring, size_t> ToMap(const Dictionary& d)
    {
        std::map<std::wstring, size_t> result;
        for (auto i = d.begin(); i != d.end(); ++i)
        {
            result[i->first] = (i->second.ValueType() == DictionaryValue::Type::Int) ?
                (size_t)i->second.Value<int>() :
                i->second.Value<size_t>();
        }
        return result;
    }

    CompositeMinibatchSource::CompositeMinibatchSource(const MinibatchSourceConfig& configuration)
        : m_epochEndReached(false),
          m_prevMinibatchSize(0),
          m_maxNumSamplesToRead(configuration.maxSamples),
          m_maxNumSweepsToRead(configuration.maxSweeps),
          m_truncationLength(0),
          m_numWorkers(1),
          m_workerRank(0)
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
        std::shared_ptr<Reader> compositeDataReader(createReaderProc(&config));

        auto compositeDataReaderStreamDescs = compositeDataReader->GetStreamDescriptions();
        m_streamInfos.insert(compositeDataReaderStreamDescs.begin(), compositeDataReaderStreamDescs.end());

        m_shim = std::shared_ptr<ReaderShim<float>>(new ReaderShim<float>(compositeDataReader), [](ReaderShim<float>* x) { x->Destroy(); });
        m_shim->Init(config);
    }

    bool CompositeMinibatchSource::IsInfinite()
    {
        return m_maxNumSamplesToRead == MinibatchSource::InfinitelyRepeat &&
               m_maxNumSweepsToRead == MinibatchSource::InfinitelyRepeat;
    }

    /*virtual*/ const std::unordered_map<StreamInformation, MinibatchData>&
    CompositeMinibatchSource::GetNextMinibatch(size_t minibatchSizeInSequences,
                                               size_t minibatchSizeInSamples,
                                               size_t numberOfWorkers,
                                               size_t workerRank,
                                               const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/) /*override*/
    {
#ifndef  CNTK_UWP
        auto profGetMinibatch = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainGetMinibatch);
#endif

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
                    // Setting big value, but not the max in order to avoid bit overflow.
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
                        m_matrices.AddInput(
                            s.m_name,
                            std::make_shared<Matrix<float>>(0, 0, inputStreamDescription.GetDeviceId(), inputStreamDescription.GetMatrixType(), inputStreamDescription.GetMatrixFormat()),
                            std::make_shared<MBLayout>(),
                            AsTensorShape(s.m_sampleLayout));
                    }
                    else
                        LogicError("GetNextMinibatch: Input of type other than DataType::Float is currently unsupported by the CNTK built-in composite MinibatchSource!");
                }

                m_shim->StartEpoch(epochConfig, inputs);

                m_prevMinibatchSize = minibatchSizeInSamples;
                m_workerRank = workerRank;
                m_numWorkers = numberOfWorkers;
            }

            if (minibatchSizeInSamples != m_prevMinibatchSize || m_workerRank != workerRank || m_numWorkers != numberOfWorkers || m_state.IsInitialized())
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

                if (m_state.IsInitialized())
                {
                    m_shim->SetState(ToMap(m_state.Get()));
                    m_state.Reset();
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

                    minibatchValuePtr = MakeSharedObject<PackedValue>(AsNDShape(input.sampleLayout), Axis::DefaultInputVariableDynamicAxes(), matrix, input.pMBLayout, /*readOnly =*/ false);

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
        auto state = m_shim->GetState();
        Dictionary result;
        for (const auto& p : state)
        {
            result[p.first] = p.second;
        }

        return result;
    }

    /*virtual*/ void CompositeMinibatchSource::RestoreFromCheckpoint(const Dictionary& checkpoint) /*override*/
    {
        m_shim->SetState(ToMap(checkpoint));

        // Need to reinitialize, we also have to remember the current position because StartEpoch
        // effectively resets it.
        // TODO: Remove call to StartEpoch - this API is legacy.
        m_state = checkpoint;
        m_epochEndReached = false;
        m_prevMinibatchSize = 0;
    }

    /* static */ ImageTransform ReaderCrop(const wchar_t* cropType,
            std::pair<int, int> cropSize, std::pair<float, float> sideRatio, std::pair<float, float> areaRatio,
            std::pair<float, float> aspectRatio, const wchar_t* jitterType)
    {
        ImageTransform crop;

        if (sideRatio.first > sideRatio.second)
            RuntimeError("For sideRatio values: the first number must be smaller than or equal to the second number.");
        if (areaRatio.first > areaRatio.second)
            RuntimeError("For areaRatio values: the first number must be smaller than or equal to the second number.");
        if (aspectRatio.first > aspectRatio.second)
            RuntimeError("For aspectRatio values: the first number must be smaller than or equal to the second number.");

        crop.Add(L"type", L"Crop",
            L"cropType", cropType,
            L"cropSize", pair_to_colon_format(cropSize),
            L"sideRatio", pair_to_colon_format(sideRatio),
            L"areaRatio", pair_to_colon_format(areaRatio),
            L"aspectRatio", pair_to_colon_format(aspectRatio),
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

    Deserializer BuildImageDeserializer(const std::wstring deserializer,
        const std::wstring& fileName, const std::wstring& labelStreamName, size_t numLabels,
        const std::wstring& imageStreamName, const std::vector<ImageTransform>& transforms) 
    {
        Deserializer img;
        std::vector<DictionaryValue> actualTransforms;
        std::transform(transforms.begin(), transforms.end(), std::back_inserter(actualTransforms), [](ImageTransform t) { return static_cast<DictionaryValue>(t); });

        // Add the transpose transform by default.
        Dictionary transposeTransform;
        transposeTransform[L"type"] = L"Transpose";
        actualTransforms.push_back(DictionaryValue(transposeTransform));

        Dictionary labeldim;
        labeldim[L"labelDim"] = numLabels;

        Dictionary xforms;
        xforms[L"transforms"] = actualTransforms;
        Dictionary input;
        input.Add(imageStreamName.c_str(), xforms, labelStreamName.c_str(), labeldim);
        img.Add(L"type", deserializer, L"file", fileName, L"input", input);
        return img;
    }

    Deserializer ImageDeserializer(const std::wstring& fileName, const std::wstring& labelStreamName, size_t numLabels, 
        const std::wstring& imageStreamName, const std::vector<ImageTransform>& transforms)
    {
        return BuildImageDeserializer(L"ImageDeserializer", fileName, labelStreamName, numLabels, imageStreamName, transforms);
    }

    Deserializer Base64ImageDeserializer(const std::wstring& fileName, const std::wstring& labelStreamName, size_t numLabels, 
        const std::wstring& imageStreamName, const std::vector<ImageTransform>& transforms)
    {
        return BuildImageDeserializer(L"Base64ImageDeserializer", fileName, labelStreamName, numLabels, imageStreamName, transforms);
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
            stream[L"definesMBSize"] = s.m_definesMbSize;
            if (!s.m_streamAlias.empty())
                stream[L"alias"] = s.m_streamAlias;
            input[key] = stream;
        }
        ctf.Add(L"type", L"CNTKTextFormatDeserializer", L"file", fileName, L"input", input);
        return ctf;
    }

    Deserializer CBFDeserializer(const std::wstring& fileName, const std::vector<StreamConfiguration>& streams)
    {
        Deserializer config;
        Dictionary input;
        for (const auto& s : streams)
        {
            if (s.m_streamAlias != s.m_streamName) 
            {
                Dictionary stream;
                stream[L"alias"] = s.m_streamAlias;
                input[s.m_streamName] = stream;
            }
        }
        config.Add(L"type", L"CNTKBinaryFormatDeserializer", L"file", fileName, L"input", input);
        return config;
    }

    Deserializer HTKFeatureDeserializer(const std::vector<HTKFeatureConfiguration>& streams)
    {
        if (streams.empty())
            InvalidArgument("HTK deserializer expects at least one stream.");

        std::vector<DictionaryValue> deserializers;
        for (const auto& s : streams)
        {
            Deserializer htk;
            Dictionary input;
            const auto& key = s.m_streamName;
            Dictionary stream;
            std::vector<DictionaryValue> ctxWindow = { DictionaryValue(s.m_left), DictionaryValue(s.m_right) };
            stream.Add(L"scpFile", s.m_scp, L"dim", s.m_dim, L"contextWindow", ctxWindow, L"expandToUtterance", s.m_broadcast);
            stream[L"definesMBSize"] = s.m_definesMbSize;
            input[key] = stream;
            htk.Add(L"type", L"HTKFeatureDeserializer", L"input", input);
            deserializers.push_back(htk);
        }

        if (deserializers.size() == 1)
            return deserializers.front().Value<Dictionary>();

        Dictionary result;
        result[DeserializersProperty] = deserializers;
        return result;
    }

    Deserializer HTKMLFDeserializer(const std::wstring& streamName, const std::wstring& labelMappingFile, size_t dimension, const std::vector<std::wstring>& mlfFiles, bool phoneBoundaries)
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
        if (phoneBoundaries)
            labels[L"phoneBoundaries"] = L"true";
        else
            labels[L"phoneBoundaries"] = L"false";
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
                augmentedConfiguration[L"randomizationSeed"] = configuration.randomizationSeed;
            }
            else if (configuration.randomizationWindowInChunks != 0) 
            {
                augmentedConfiguration[L"randomize"] = true;
                augmentedConfiguration[L"randomizationWindow"] = configuration.randomizationWindowInChunks;
                augmentedConfiguration[L"sampleBasedRandomizationWindow"] = false;
                augmentedConfiguration[L"randomizationSeed"] = configuration.randomizationSeed;
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
            augmentedConfiguration[L"traceLevel"] = static_cast<size_t>(configuration.traceLevel);

            bool defaultMultithreaded = false;
            // The CNTK reader implementation requires for each deserializer both the module and deserializer type be specified
            // This is redundant and the V2 API users will just specify type from which the module is automatically inferred
            // TODO: This should be done in the same manner for CNTK exe as well.
            vector<DictionaryValue> deserializers;
            for (auto deserializerConfig : configuration.deserializers)
            {
                static const std::unordered_map<std::wstring, std::wstring> deserializerTypeToModule = {
                    { L"CNTKTextFormatDeserializer",   L"CNTKTextFormatReader" },
                    { L"CNTKBinaryFormatDeserializer", L"CNTKBinaryReader" },
                    { L"ImageDeserializer",            L"ImageReader" },
                    { L"Base64ImageDeserializer",      L"ImageReader" },
                    { L"HTKFeatureDeserializer",       L"HTKDeserializers" },
                    { L"HTKMLFDeserializer",           L"HTKDeserializers" },
                };

                auto deserializerTypeName = deserializerConfig[L"type"].Value<std::wstring>();
                if (deserializerTypeName == L"ImageDeserializer" || deserializerTypeName == L"Base64ImageDeserializer")
                {
                    defaultMultithreaded = true;
                }

                if (deserializerTypeToModule.find(deserializerTypeName) == deserializerTypeToModule.end())
                {
                    if (!deserializerConfig.Contains(L"module"))
                        InvalidArgument("Unknown deserializer type '%S' specified for CNTK built-in composite MinibatchSource construction.", deserializerTypeName.c_str());
                }
                else
                    deserializerConfig[L"module"] = deserializerTypeToModule.at(deserializerTypeName);
                deserializers.push_back(deserializerConfig);
            }

            augmentedConfiguration[L"multiThreadedDeserialization"] = 
                (configuration.isMultithreaded.IsInitialized()) ? configuration.isMultithreaded.Get() : defaultMultithreaded;

            augmentedConfiguration[L"deserializers"] = deserializers;

            return augmentedConfiguration;
        }
    }
}
