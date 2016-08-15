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
#include "ComputationNetworkBuilder.h"

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    MinibatchSourcePtr CreateCompositeMinibatchSource(const Dictionary& configuration)
    {
        return MinibatchSourcePtr(new CompositeMinibatchSource(configuration));
    }

    CompositeMinibatchSource::CompositeMinibatchSource(const Dictionary& configuration)
        : m_epochEndReached(false), m_prevMinibatchSize(0), m_epochSize(SIZE_MAX)
    {
        ConfigParameters config;
        std::wstringstream s;
        for (const auto& keyValuePair : *(configuration.m_dictionaryData))
            AddConfigString(s, keyValuePair.first, keyValuePair.second, 0);

        config.Parse(msra::strfun::utf8(s.str()));

        const wchar_t* epochSizeConfigurationKey = L"epochSize";
        if (configuration.Contains(epochSizeConfigurationKey))
            m_epochSize = configuration[epochSizeConfigurationKey].GetValue<size_t>();

        if (m_epochSize == 0)
            m_epochSize = Microsoft::MSR::CNTK::requestDataSize;

        typedef Reader*(*CreateCompositeDataReaderProc)(const ConfigParameters* parameters);
        CreateCompositeDataReaderProc createReaderProc = (CreateCompositeDataReaderProc)Plugin().Load(L"CompositeDataReader", "CreateCompositeDataReader");
        m_compositeDataReader.reset(createReaderProc(&config));

        auto compositeDataReaderStreamDescs = m_compositeDataReader->GetStreamDescriptions();
        for (auto streamDesc : compositeDataReaderStreamDescs)
            m_streamInfos.insert({ streamDesc->m_name, streamDesc->m_id, AsStorageFormat(streamDesc->m_storageType), AsDataType(streamDesc->m_elementType), AsNDShape(*(streamDesc->m_sampleLayout)) });
    }

    /*virtual*/ std::unordered_map<StreamInfo, MinibatchData> CompositeMinibatchSource::GetNextMinibatch(const std::unordered_map<StreamInfo, std::pair<size_t, size_t>>& perStreamMBSizeLimits,
                                                                                                         const DeviceDescriptor& device /*= DeviceDescriptor::DefaultDevice()*/) /*override*/
    {
        std::unordered_map<StreamInfo, MinibatchData> minibatchData;
        if (!m_epochEndReached)
        {
            // TODO: Support different minibatch sizes for different streams
            size_t requestedMinibatchSizeInSamples = 0;
            for (const auto& val : perStreamMBSizeLimits)
            {
                size_t maxNumSequencesRequested = val.second.first;
                size_t maxNumSamplesRequested = val.second.second;

                // TODO: Specifying minibatch size in #sequences is currently unsupported
                if (maxNumSequencesRequested != 0)
                    LogicError("Specifying minibatch size in #sequences is currently unsupported");

                if (requestedMinibatchSizeInSamples == 0)
                    requestedMinibatchSizeInSamples = maxNumSamplesRequested;
                else
                {
                    if (requestedMinibatchSizeInSamples != maxNumSamplesRequested)
                        LogicError("Different minibatch sizes across different input streams is currently unsupported!");
                }
            }

            if (requestedMinibatchSizeInSamples == 0)
                InvalidArgument("GetNextMinibatch: Requested minibatch sizes must be > 0");

            if (m_prevMinibatchSize == 0)
            {
                // TODO: Add support for distributed reading
                EpochConfiguration epochConfig = { 1, 0, requestedMinibatchSizeInSamples, m_epochSize, 0, 0 };
                m_compositeDataReader->StartEpoch(epochConfig);
                m_prevMinibatchSize = requestedMinibatchSizeInSamples;
            }

            if (requestedMinibatchSizeInSamples != m_prevMinibatchSize)
                LogicError("GetNextMinibatch: Changing minibatch sizes across calls is currently unsupported");

            auto compositeReaderMinibatchData = m_compositeDataReader->ReadMinibatch();
            m_epochEndReached = compositeReaderMinibatchData.m_endOfEpoch;

            auto compositeDataReaderStreamDescs = m_compositeDataReader->GetStreamDescriptions();
            size_t numStreams = compositeDataReaderStreamDescs.size();
            for (size_t i = 0; i < numStreams; ++i)
            {
                auto currentStreamDesc = compositeDataReaderStreamDescs[i];
                auto iter = std::find_if(perStreamMBSizeLimits.begin(), perStreamMBSizeLimits.end(), [currentStreamDesc](const std::pair<StreamInfo, std::pair<size_t, size_t>>& entry) {
                    return entry.first.m_id == currentStreamDesc->m_id;
                });

                if (iter == perStreamMBSizeLimits.end())
                    continue;

                auto& currentStreamInfo = iter->first;
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
                    auto dataMatrix = std::make_shared<Matrix<float>>(CPUDEVICE);
                    size_t sampleSize = currentStreamDesc->m_sampleLayout->GetNumElements();

                    // TODO: Eliminate the unnecessary CPU to CPU copy
                    ReaderShim<float>::FillMatrixFromStream(currentStreamDesc->m_storageType, dataMatrix.get(), sampleSize, currentStreamMinibatchData);
                    minibatchValuePtr = CompositeFunction::GetValueObjectFromCNTKImplMatrixAndMBLayout<float>(sampleShape, *dataMatrix, currentStreamMinibatchData->m_layout, false);

                    size_t numSamples = currentStreamMinibatchData->m_layout->GetActualNumSamples();
                    size_t numSequences = currentStreamMinibatchData->m_layout->GetNumSequences();

                    minibatchData[currentStreamInfo] = { numSequences, numSamples, minibatchValuePtr };
                }
                else
                    LogicError("Input data of type other than DataType::Float is currently unsupported by the CNTK built-in composite MinibatchSource!");
            }
        }

        return minibatchData;
    }

    void ComputeInputPerDimMeansAndInvStdDevs(const MinibatchSourcePtr& minibatchSource,
                                              std::unordered_map<StreamInfo, std::pair<NDArrayViewPtr, NDArrayViewPtr>>& computedMeanAndInvStdDevs,
                                              const DeviceDescriptor& device /*= DeviceDescriptor::CPUDevice()*/)
    {
        typedef std::shared_ptr<ComputationNode<float>> ComputationNodePtr;
        const auto& minibatchSourceStreams = minibatchSource->StreamInfos();

        auto computationNetwork = std::make_shared<ComputationNetwork>(AsCNTKImplDeviceId(device));
        ComputationNetworkBuilder<float> builder(*computationNetwork);

        std::vector<ComputationNodeBasePtr> allInputNodes;
        std::unordered_map<StreamInfo, ComputationNodeBasePtr> streamToInputNodeMap;
        std::unordered_map<StreamInfo, Variable> streamToDummyInputVariableMap;
        std::unordered_map<StreamInfo, ComputationNodeBasePtr> streamToMeanNodeMap;
        std::unordered_map<StreamInfo, ComputationNodeBasePtr> streamToInvStdDevNodeMap;

        size_t totalSizePerSample = 0;
        for (auto& currentStreamKV : computedMeanAndInvStdDevs)
        {
            auto currentStreamInfo = currentStreamKV.first;
            if (minibatchSourceStreams.find(currentStreamInfo) == minibatchSourceStreams.end())
                InvalidArgument("ComputeMeanAndVariance: Stream for which mean and variance is to be computed is not supported by the specified minibatchSource");

            if (currentStreamInfo.m_elementType != DataType::Float)
                LogicError("Input data of type other than DataType::Float is currently unsupported by the CNTK built-in composite MinibatchSource!");

            auto inputVariableShape = currentStreamInfo.m_sampleLayout;
            auto inputTensorShape = AsTensorShape(inputVariableShape);
            totalSizePerSample += (inputVariableShape.TotalSize() * sizeof(float));

            ComputationNodePtr inputNode;
            Variable inputVariable;
            if (currentStreamInfo.m_storageFormat != StorageFormat::Dense)
            {
                inputNode = builder.CreateSparseInputNode(currentStreamInfo.m_name, inputTensorShape);
                inputVariable = Variable(inputVariableShape, true, DataType::Float, currentStreamInfo.m_name);
            }
            else
            {
                inputNode = builder.CreateInputNode(currentStreamInfo.m_name, inputTensorShape);
                inputVariable = Variable(inputVariableShape, DataType::Float, currentStreamInfo.m_name);
            }

            allInputNodes.push_back(inputNode);
            streamToInputNodeMap[currentStreamInfo] = inputNode;
            streamToDummyInputVariableMap[currentStreamInfo] = inputVariable;
            streamToMeanNodeMap[currentStreamInfo] = builder.Mean(inputNode);
            streamToInvStdDevNodeMap[currentStreamInfo] = builder.InvStdDev(inputNode);
        }

        computationNetwork->CompileNetwork();
        computationNetwork->AllocateAllMatrices(computationNetwork->RootNodes(), {}, nullptr);

        ScopedNetworkOperationMode modeGuard(computationNetwork, NetworkOperationMode::preComputing);

        // initialize
        auto preComputeNodes = computationNetwork->GetNodesRequiringPreComputation();
        for (auto & preComputeNode : preComputeNodes)
            dynamic_pointer_cast<IPreComputeNode>(preComputeNode)->MarkComputed(false /*begin accumulating*/);

        const size_t maxMinibatchDataSize = (1 << 27); // 128 MB
        const size_t minibatchSize = maxMinibatchDataSize / totalSizePerSample;
        std::unordered_map<StreamInfo, std::pair<size_t, size_t>> minibatchSizeLimits;
        for (auto& currentStreamKV : computedMeanAndInvStdDevs)
            minibatchSizeLimits.insert(std::make_pair(currentStreamKV.first, std::make_pair((size_t)0, minibatchSize)));

        for (;;)
        {
            auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSizeLimits, device);
            if (minibatchData.empty())
                break;

            for (auto& currentStreamKV : computedMeanAndInvStdDevs)
                CompositeFunction::PopulateComputationNodeValue<float>({ streamToDummyInputVariableMap[currentStreamKV.first], minibatchData[currentStreamKV.first].m_data }, streamToInputNodeMap[currentStreamKV.first]);

            ComputationNetwork::BumpEvalTimeStamp(allInputNodes);

            computationNetwork->ForwardProp(preComputeNodes);
        }

        // finalize
        for (auto & preComputeNode : preComputeNodes)
            dynamic_pointer_cast<IPreComputeNode>(preComputeNode)->MarkComputed(true /*done accumulating*/);

        // Copy out the results
        for (auto& currentStreamKV : computedMeanAndInvStdDevs)
        {
            ValuePtr mean, invStdDev;
            if (computedMeanAndInvStdDevs[currentStreamKV.first].first != nullptr)
                mean = MakeSharedObject<Value>(computedMeanAndInvStdDevs[currentStreamKV.first].first);

            if (computedMeanAndInvStdDevs[currentStreamKV.first].second != nullptr)
                invStdDev = MakeSharedObject<Value>(computedMeanAndInvStdDevs[currentStreamKV.first].second);

            CompositeFunction::GetNodeOutputOrGradient(streamToDummyInputVariableMap[currentStreamKV.first], mean, streamToMeanNodeMap[currentStreamKV.first], false /*getGradient*/);
            CompositeFunction::GetNodeOutputOrGradient(streamToDummyInputVariableMap[currentStreamKV.first], invStdDev, streamToInvStdDevNodeMap[currentStreamKV.first], false /*getGradient*/);

            if (computedMeanAndInvStdDevs[currentStreamKV.first].first == nullptr)
                computedMeanAndInvStdDevs[currentStreamKV.first].first = mean->Data();

            if (computedMeanAndInvStdDevs[currentStreamKV.first].second == nullptr)
                computedMeanAndInvStdDevs[currentStreamKV.first].second = invStdDev->Data();

        }
    }
}

