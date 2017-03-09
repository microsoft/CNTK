//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "DataReader.h"
#include "ComputationNetwork.h"
#include "DataReaderHelpers.h"
#include "Helpers.h"
#include "File.h"
#include "fileutil.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <cstdio>
#include "ProgressTracing.h"
#include "ComputationNetworkBuilder.h"

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {


template <class ElemType>
class SimpleOutputWriter
{
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;

public:
    SimpleOutputWriter(ComputationNetworkPtr net, int verbosity = 0)
        : m_net(net), m_verbosity(verbosity)
    {
    }

    void WriteOutput(IDataReader& dataReader, size_t mbSize, IDataWriter& dataWriter, const std::vector<std::wstring>& outputNodeNames, size_t numOutputSamples = requestDataSize, bool doWriterUnitTest = false)
    {
        ScopedNetworkOperationMode modeGuard(m_net, NetworkOperationMode::inferring);

        if (outputNodeNames.size() == 0 && m_verbosity > 0)
            fprintf(stderr, "OutputNodeNames are not specified, using the default outputnodes.\n");

        std::vector<ComputationNodeBasePtr> outputNodes = m_net->OutputNodesByName(outputNodeNames);
        std::vector<ComputationNodeBasePtr> inputNodes  = m_net->InputNodesForOutputs(outputNodeNames);

        // allocate memory for forward computation
        m_net->AllocateAllMatrices({}, outputNodes, nullptr);

        StreamMinibatchInputs inputMatrices = DataReaderHelpers::RetrieveInputMatrices(inputNodes);

        // evaluate with minibatches
        dataReader.StartMinibatchLoop(mbSize, 0, inputMatrices.GetStreamDescriptions(), numOutputSamples);
        if (!dataWriter.SupportMultiUtterances())
            dataReader.SetNumParallelSequences(1);
        m_net->StartEvaluateMinibatchLoop(outputNodes);

        size_t totalEpochSamples = 0;
        std::map<std::wstring, void*, nocase_compare> outputMatrices;

        const size_t numIterationsBeforePrintingProgress = 100;
        size_t numItersSinceLastPrintOfProgress = 0;
        size_t actualMBSize;
        while (DataReaderHelpers::GetMinibatchIntoNetwork<ElemType>(dataReader, m_net, nullptr, false, false, inputMatrices, actualMBSize, nullptr))
        {
            ComputationNetwork::BumpEvalTimeStamp(inputNodes);
            m_net->ForwardProp(outputNodes);

            for (int i = 0; i < outputNodes.size(); i++)
                outputMatrices[outputNodes[i]->NodeName()] = (void*) (&dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[i])->Value());

            if (doWriterUnitTest)
            {
                std::map<std::wstring, void*, nocase_compare> inputMatricesUnitTest;
                for (auto& iter : inputMatrices)
                    inputMatricesUnitTest[iter.first] = (void*) iter.second.matrix.get();  // BUGBUG: void* are evil
                dataWriter.SaveData(0, inputMatricesUnitTest, actualMBSize, actualMBSize, 0);
            }
            else
                dataWriter.SaveData(0, outputMatrices, actualMBSize, actualMBSize, 0);

            totalEpochSamples += actualMBSize;

            numItersSinceLastPrintOfProgress = ProgressTracing::TraceFakeProgress(numIterationsBeforePrintingProgress, numItersSinceLastPrintOfProgress);

            // call DataEnd function in dataReader to do
            // reader specific process if sentence ending is reached
            dataReader.DataEnd();
        }

        if (m_verbosity > 0)
            fprintf(stderr, "Total Samples Evaluated = %lu\n", (unsigned long)totalEpochSamples);

        // clean up
    }

    // Perform a single forward pass to obtain the output values from a network
    void WriteOutput(IDataWriter& dataWriter, const std::vector<std::wstring>& outputNodeNames, size_t numOutputSamples = requestDataSize, bool doUnitTest = false)
    {
        std::vector<ComputationNodeBasePtr> outputNodes = m_net->OutputNodesByName(outputNodeNames);

        // allocate memory for forward computation
        m_net->AllocateAllMatrices({}, outputNodes, nullptr);

        m_net->StartEvaluateMinibatchLoop(outputNodes);

        std::map<std::wstring, void*, nocase_compare> outputMatrices;

        m_net->ForwardProp(outputNodes);
        for (int i = 0; i < outputNodes.size(); i++)
            outputMatrices[outputNodes[i]->NodeName()] = (void*)(&dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[i])->Value());

        // TODO: What should the data size be?
        dataWriter.SaveData(0, outputMatrices, 1, 1, 0);
    }

    void WriteMinibatch(FILE* f, ComputationNodePtr node,
        const WriteFormattingOptions & formattingOptions, char formatChar, std::string valueFormatString, std::vector<std::string>& labelMapping,
        size_t numMBsRun, bool gradient, std::function<std::string(size_t)>& idToKeyMapping)
    {
        const auto sequenceSeparator = formattingOptions.Processed(node->NodeName(), formattingOptions.sequenceSeparator, numMBsRun);
        const auto sequencePrologue =  formattingOptions.Processed(node->NodeName(), formattingOptions.sequencePrologue,  numMBsRun);
        const auto sequenceEpilogue =  formattingOptions.Processed(node->NodeName(), formattingOptions.sequenceEpilogue,  numMBsRun);
        const auto elementSeparator =  formattingOptions.Processed(node->NodeName(), formattingOptions.elementSeparator,  numMBsRun);
        const auto sampleSeparator =   formattingOptions.Processed(node->NodeName(), formattingOptions.sampleSeparator,   numMBsRun);

        node->WriteMinibatchWithFormatting(f, FrameRange(), SIZE_MAX, SIZE_MAX, formattingOptions.transpose, formattingOptions.isCategoryLabel, formattingOptions.isSparse, labelMapping,
            sequenceSeparator, sequencePrologue, sequenceEpilogue, elementSeparator, sampleSeparator,
            valueFormatString, gradient, false, idToKeyMapping);
    }

    void InsertNode(std::vector<ComputationNodeBasePtr>& allNodes, ComputationNodeBasePtr parent, ComputationNodeBasePtr newNode)
    {
        newNode->SetInput(0, parent);
        for (auto node : allNodes)
        {
            size_t i = 0;
            for (auto n : node->GetInputs())
            {
                if (n == parent)
                    node->SetInput(i, newNode);
                ++i;
            }
        }
    }

    // TODO: Remove code dup with above function by creating a fake Writer object and then calling the other function.
    void WriteOutput(IDataReader& dataReader, size_t mbSize, std::wstring outputPath, const std::vector<std::wstring>& outputNodeNames, const WriteFormattingOptions& formattingOptions, size_t numOutputSamples = requestDataSize, bool nodeUnitTest = false, bool writeSequenceKey = false)
    {
        // In case of unit test, make sure backprop works
        ScopedNetworkOperationMode modeGuard(m_net, nodeUnitTest ? NetworkOperationMode::training : NetworkOperationMode::inferring);

        std::vector<ComputationNodeBasePtr> outputNodes = m_net->OutputNodesByName(outputNodeNames);
        std::vector<ComputationNodeBasePtr> inputNodes = m_net->InputNodesForOutputs(outputNodeNames);
        std::vector<ComputationNodePtr> gradientNodes;
        std::vector<ComputationNodeBasePtr> allOutputNodes = outputNodes;

        if (!nodeUnitTest)                                        // regular operation
        {
            m_net->AllocateAllMatrices({}, outputNodes, nullptr); // don't allocate for backward pass
        }
        else                                                      // we mis-appropriate this code for unit testing of the back-prop path
        {
            // Unit test only makes sense for one output node.
            if (outputNodes.size() != 1)
                RuntimeError("Expected exactly 1 output node for unit test, got %d.", (int)outputNodes.size());

            // Set up machinery to output gradients alongside forward pass output
            // Gradients are not passed on to inputs. Need to hook an identity function in between.
            ComputationNetworkBuilder<ElemType> builder(*m_net);
            auto allInputs = inputNodes;
            auto allParameters = m_net->LearnableParameterNodes(outputNodes[0]);
            allInputs.insert(allInputs.end(), allParameters.begin(), allParameters.end());
            auto allNodes = m_net->GetAllNodes();

            for (auto inputNode : allInputs)
            {
                auto parent = dynamic_pointer_cast<ComputationNode<ElemType>>(inputNode);
                auto newNode = builder.Pass(parent, inputNode->NodeName() + L".grad");
                newNode->SetLearningRateMultiplier(1.0); // Forces gradient update. Otherwise, backprop might get pruned from this path.
                InsertNode(allNodes, parent, newNode);
                gradientNodes.push_back(dynamic_pointer_cast<ComputationNode<ElemType>>(newNode));
                allOutputNodes.push_back(newNode);
            }

            // Update the evaluation order, and other things.
            m_net->CompileNetwork();
            
            // Allocate memory for forward and backward computation. In case of unit test, treat the output node
            // like a criterion node. Submitting a node as parameter 3 here will allocate the gradients.
            m_net->AllocateAllMatrices({}, outputNodes, outputNodes[0]);
        }

        StreamMinibatchInputs inputMatrices = DataReaderHelpers::RetrieveInputMatrices(inputNodes);
        
        // load a label mapping if requested
        std::vector<std::string> labelMapping;
        if ((formattingOptions.isCategoryLabel || formattingOptions.isSparse) && !formattingOptions.labelMappingFile.empty())
            File::LoadLabelFile(formattingOptions.labelMappingFile, labelMapping);

        // open output files
        File::MakeIntermediateDirs(outputPath);
        std::map<ComputationNodeBasePtr, shared_ptr<File>> outputStreams; // TODO: why does unique_ptr not work here? Complains about non-existent default_delete()
        for (auto & onode : allOutputNodes)
        {
            std::wstring nodeOutputPath = outputPath;
            if (nodeOutputPath != L"-")
                nodeOutputPath += L"." + onode->NodeName();
            auto f = make_shared<File>(nodeOutputPath, fileOptionsWrite | fileOptionsText);
            outputStreams[onode] = f;
        }

        // evaluate with minibatches
        dataReader.StartMinibatchLoop(mbSize, 0, inputMatrices.GetStreamDescriptions(), numOutputSamples);

        m_net->StartEvaluateMinibatchLoop(outputNodes);

        size_t totalEpochSamples = 0;

        for (auto & onode : outputNodes)
        {
            FILE* f = *outputStreams[onode];
            fprintfOrDie(f, "%s", formattingOptions.prologue.c_str());
        }

        size_t actualMBSize;
        const size_t numIterationsBeforePrintingProgress = 100;
        size_t numItersSinceLastPrintOfProgress = 0;
        char formatChar = !formattingOptions.isCategoryLabel ? 'f' : !formattingOptions.labelMappingFile.empty() ? 's' : 'u';
        std::string valueFormatString = "%" + formattingOptions.precisionFormat + formatChar; // format string used in fprintf() for formatting the values

        for (size_t numMBsRun = 0; DataReaderHelpers::GetMinibatchIntoNetwork<ElemType>(dataReader, m_net, nullptr, false, false, inputMatrices, actualMBSize, nullptr); numMBsRun++)
        {
            ComputationNetwork::BumpEvalTimeStamp(inputNodes);
            m_net->ForwardProp(outputNodes);

            for (auto & onode : outputNodes)
            {
                // compute the node value
                // Note: Intermediate values are memoized, so in case of multiple output nodes, we only compute what has not been computed already.

                FILE* file = *outputStreams[onode];
                auto getKeyById = writeSequenceKey ? inputMatrices.m_getKeyById : std::function<std::string(size_t)>();
                WriteMinibatch(file, dynamic_pointer_cast<ComputationNode<ElemType>>(onode), formattingOptions, formatChar, valueFormatString, labelMapping, numMBsRun, /* gradient */ false, getKeyById);

                if (nodeUnitTest)
                    m_net->Backprop(onode);
            } // end loop over nodes

            if (nodeUnitTest)
            {
                for (auto & node : gradientNodes)
                {
                    FILE* file = *outputStreams[node];
                    if (!node->GradientPtr())
                    {
                        fprintf(stderr, "Warning: Gradient of node '%s' is empty. Not used in backward pass?", msra::strfun::utf8(node->NodeName().c_str()).c_str());
                    }
                    else
                    {
                        auto idToKeyMapping = std::function<std::string(size_t)>();
                        WriteMinibatch(file, node, formattingOptions, formatChar, valueFormatString, labelMapping, numMBsRun, /* gradient */ true, idToKeyMapping);
                    }
                }
            }
            totalEpochSamples += actualMBSize;

            fprintf(stderr, "Minibatch[%lu]: ActualMBSize = %lu\n", (unsigned long)numMBsRun, (unsigned long)actualMBSize);
            if (outputPath == L"-") // if we mush all nodes together on stdout, add some visual separator
                fprintf(stdout, "\n");

            numItersSinceLastPrintOfProgress = ProgressTracing::TraceFakeProgress(numIterationsBeforePrintingProgress, numItersSinceLastPrintOfProgress);

            // call DataEnd function in dataReader to do
            // reader specific process if sentence ending is reached
            dataReader.DataEnd();
        } // end loop over minibatches

        for (auto & stream : outputStreams)
        {
            FILE* f = *stream.second;
            fprintfOrDie(f, "%s", formattingOptions.epilogue.c_str());
        }

        fprintf(stderr, "Written to %ls*\nTotal Samples Evaluated = %lu\n", outputPath.c_str(), (unsigned long)totalEpochSamples);

        // flush all files (where we can catch errors) so that we can then destruct the handle cleanly without error
        for (auto & iter : outputStreams)
            iter.second->Flush();
    }

private:
    ComputationNetworkPtr m_net;
    int m_verbosity;
    void operator=(const SimpleOutputWriter&); // (not assignable)
};

}}}
