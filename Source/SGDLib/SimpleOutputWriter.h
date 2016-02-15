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

    void WriteOutput(IDataReader<ElemType>& dataReader, size_t mbSize, IDataWriter<ElemType>& dataWriter, const std::vector<std::wstring>& outputNodeNames, size_t numOutputSamples = requestDataSize, bool doUnitTest = false)
    {
        // specify output nodes and files
        std::vector<ComputationNodeBasePtr> outputNodes;
        if (outputNodeNames.size() == 0)
        {
            if (m_verbosity > 0)
                fprintf(stderr, "OutputNodeNames are not specified, using the default outputnodes.\n");
            if (m_net->OutputNodes().size() == 0)
                LogicError("There is no default output node specified in the network.");

            outputNodes = m_net->OutputNodes();
        }
        else
        {
            for (int i = 0; i < outputNodeNames.size(); i++)
                outputNodes.push_back(m_net->GetNodeFromName(outputNodeNames[i]));
        }

        // allocate memory for forward computation
        m_net->AllocateAllMatrices({}, outputNodes, nullptr);

        // specify feature value nodes
        std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
        for (auto& onode : outputNodes)
            for (auto& inode : m_net->InputNodes(onode))
                inputMatrices[inode->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(inode)->Value();

        // Matrix<ElemType> endOfFile =  Matrix<ElemType>((size_t)1,(size_t)1);
        // endOfFile(0,0)=0;

        // evaluate with minibatches
        dataReader.StartMinibatchLoop(mbSize, 0, numOutputSamples);
        dataReader.SetNumParallelSequences(1);

        m_net->StartEvaluateMinibatchLoop(outputNodes);

        size_t totalEpochSamples = 0;
        std::map<std::wstring, void*, nocase_compare> outputMatrices;

        size_t actualMBSize;
        while (DataReaderHelpers::GetMinibatchIntoNetwork(dataReader, m_net, nullptr, false, false, inputMatrices, actualMBSize))
        {
            // Update timestamp for all input nodes ancestors of the output nodes
            for (auto& onode : outputNodes)
                for (auto& inode : m_net->InputNodes(onode))
                    inode->BumpEvalTimeStamp();

            for (int i = 0; i < outputNodes.size(); i++)
            {
                m_net->ForwardProp(outputNodes[i]);
                outputMatrices[outputNodes[i]->NodeName()] = (void*) (&dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[i])->Value());
            }

            if (doUnitTest)
            {
                std::map<std::wstring, void*, nocase_compare> inputMatricesUnitTest;
                for (auto iter = inputMatrices.begin(); iter != inputMatrices.end(); iter++)
                    inputMatricesUnitTest[iter->first] = (void*) (iter->second);
                dataWriter.SaveData(0, inputMatricesUnitTest, actualMBSize, actualMBSize, 0);
            }
            else
                dataWriter.SaveData(0, outputMatrices, actualMBSize, actualMBSize, 0);

            totalEpochSamples += actualMBSize;

            // call DataEnd function in dataReader to do
            // reader specific process if sentence ending is reached
            dataReader.DataEnd(endDataSentence);
        }

        if (m_verbosity > 0)
            fprintf(stderr, "Total Samples Evaluated = %lu\n", totalEpochSamples);

        // clean up
    }

    // pass this to WriteOutput() (to file-path, below) to specify how the output should be formatted
    struct WriteFormattingOptions
    {
        // How to interpret the data:
        bool isCategoryLabel;          // true: find max value in column and output the index instead of the entire vector
        std::wstring labelMappingFile; // optional dictionary for pretty-printing category labels
        bool transpose;                // true: one line per sample, each sample (column vector) forms one line; false: one column per sample
        // The following strings are interspersed with the data:
        // overall
        std::string prologue; // print this at the start (e.g. a global header or opening bracket)
        std::string epilogue; // and this at the end
        // sequences
        std::string sequenceSeparator; // print this between sequences (i.e. before all sequences but the first)
        std::string sequencePrologue;  // print this before each sequence (after sequenceSeparator)
        std::string sequenceEpilogue;  // and this after each sequence
        // elements
        std::string elementSeparator;  // print this between elements on a row
        std::string sampleSeparator;   // and this between rows
        // Optional printf precision parameter:
        std::string precisionFormat;        // printf precision, e.g. ".2" to get a "%.2f"

        WriteFormattingOptions() :
            isCategoryLabel(false), transpose(true), sequenceEpilogue("\n"), elementSeparator(" "), sampleSeparator("\n")
        { }
    };

    // TODO: Remove code dup with above function by creating a fake Writer object and then calling the other function.
    void WriteOutput(IDataReader<ElemType>& dataReader, size_t mbSize, std::wstring outputPath, const std::vector<std::wstring>& outputNodeNames, const WriteFormattingOptions & formattingOptions, size_t numOutputSamples = requestDataSize)
    {
        // load a label mapping if requested
        std::vector<std::string> labelMapping;
        if (formattingOptions.isCategoryLabel && !formattingOptions.labelMappingFile.empty())
            File::LoadLabelFile(formattingOptions.labelMappingFile, labelMapping);

        // specify output nodes and files
        std::vector<ComputationNodeBasePtr> outputNodes;
        if (outputNodeNames.size() == 0)
        {
            fprintf(stderr, "OutputNodeNames are not specified, using the default outputnodes.\n");
            if (m_net->OutputNodes().size() == 0)
                LogicError("There is no default output node specified in the network.");

            outputNodes = m_net->OutputNodes();
        }
        else
        {
            for (int i = 0; i < outputNodeNames.size(); i++)
                outputNodes.push_back(m_net->GetNodeFromName(outputNodeNames[i]));
        }

        // open output files
        File::MakeIntermediateDirs(outputPath);
        std::map<ComputationNodeBasePtr, shared_ptr<File>> outputStreams; // TODO: why does unique_ptr not work here? Complains about non-existent default_delete()
        for (auto & onode : outputNodes)
        {
            std::wstring nodeOutputPath = outputPath;
            if (nodeOutputPath != L"-")
                nodeOutputPath += L"." + onode->NodeName();
            auto f = make_shared<File>(nodeOutputPath, fileOptionsWrite | fileOptionsText);
            outputStreams[onode] = f;
        }

        // allocate memory for forward computation
        m_net->AllocateAllMatrices({}, outputNodes, nullptr);

        // specify feature value nodes
        auto& featureNodes = m_net->FeatureNodes();
        std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
        // BUGBUG: This loop is inconsistent with the above version of this function in that it does not handle label nodes.
        for (size_t i = 0; i < featureNodes.size(); i++)
            inputMatrices[featureNodes[i]->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(featureNodes[i])->Value();

        // evaluate with minibatches
        dataReader.StartMinibatchLoop(mbSize, 0, numOutputSamples);

        m_net->StartEvaluateMinibatchLoop(outputNodes);

        size_t totalEpochSamples = 0;
        size_t numMBsRun = 0;
        size_t tempArraySize = 0;
        ElemType* tempArray = nullptr;

        for (auto & onode : outputNodes)
        {
            FILE * f = *outputStreams[onode];
            fprintfOrDie(f, "%s", formattingOptions.prologue.c_str());
        }

        char formatChar = !formattingOptions.isCategoryLabel ? 'f' : !formattingOptions.labelMappingFile.empty() ? 's' : 'u';
        std::string valueFormatString = "%" + formattingOptions.precisionFormat + formatChar; // format string used in fprintf() for formatting the values

        size_t actualMBSize;
        while (DataReaderHelpers::GetMinibatchIntoNetwork(dataReader, m_net, nullptr, false, false, inputMatrices, actualMBSize))
        {
            // BUGBUG: This loop is inconsistent with the above version of this function in that it does not handle label nodes.
            ComputationNetwork::BumpEvalTimeStamp(featureNodes);

            for (auto & onode : outputNodes)
            {
                // compute the node value
                // Note: Intermediate values are memoized, so in case of multiple output nodes, we only compute what has not been computed already.
                m_net->ForwardProp(onode);

                // get it (into a flat CPU-side vector)
                Matrix<ElemType>& outputValues = dynamic_pointer_cast<ComputationNode<ElemType>>(onode)->Value();
                outputValues.CopyToArray(tempArray, tempArraySize);
                ElemType* pCurValue = tempArray;

                // sequence separator
                FILE * f = *outputStreams[onode];
                if (numMBsRun > 0 && !formattingOptions.sequenceSeparator.empty())
                    fprintfOrDie(f, "%s", formattingOptions.sequenceSeparator.c_str());
                fprintfOrDie(f, "%s", formattingOptions.sequencePrologue.c_str());

                // output it according to our format specification
                size_t T   = outputValues.GetNumCols();
                size_t dim = outputValues.GetNumRows();
                if (formattingOptions.isCategoryLabel)
                {
                    if (formatChar == 's') // verify label dimension
                    {
                        if (dim != labelMapping.size())
                            InvalidArgument("write: Row dimension %d does not match number of entries %d in labelMappingFile '%ls'", (int)dim, (int)labelMapping.size(), formattingOptions.labelMappingFile.c_str());
                    }
                    // update the matrix in-place from one-hot (or max) to index
                    // find the max in each column
                    foreach_column(j, outputValues)
                    {
                        double maxPos = -1;
                        double maxVal = 0;
                        foreach_row(i, outputValues)
                        {
                            double val = pCurValue[i + j * dim];
                            if (maxPos < 0 || val >= maxVal)
                            {
                                maxPos = (double)i;
                                maxVal = val;
                            }
                        }
                        pCurValue[j] = (ElemType) maxPos; // overwrite in-place, assuming a flat vector
                    }
                    dim = 1;
                }
                size_t iend    = formattingOptions.transpose ? dim  : T;
                size_t jend    = formattingOptions.transpose ? T    : dim;
                size_t istride = formattingOptions.transpose ? 1    : jend;
                size_t jstride = formattingOptions.transpose ? iend : 1;
                for (size_t j = 0; j < jend; j++)
                {
                    if (j > 0)
                        fprintfOrDie(f, "%s", formattingOptions.sampleSeparator.c_str());
                    for (size_t i = 0; i < iend; i++)
                    {
                        if (i > 0)
                            fprintfOrDie(f, "%s", formattingOptions.elementSeparator.c_str());
                        if (formatChar == 'f') // print as real number
                        {
                            double dval = pCurValue[i * istride + j * jstride];
                            fprintfOrDie(f, valueFormatString.c_str(), dval);
                        }
                        else if (formatChar == 'u') // print category as integer index
                        {
                            unsigned int uval = (unsigned int) pCurValue[i * istride + j * jstride];
                            fprintfOrDie(f, valueFormatString.c_str(), uval);
                        }
                        else if (formatChar == 's') // print category as a label string
                        {
                            size_t uval = (size_t) pCurValue[i * istride + j * jstride];
                            assert(uval < labelMapping.size());
                            const char * sval = labelMapping[uval].c_str();
                            fprintfOrDie(f, valueFormatString.c_str(), sval);
                        }
                    }
                }
                fprintfOrDie(f, "%s", formattingOptions.sequenceEpilogue.c_str());
            }

            totalEpochSamples += actualMBSize;

            fprintf(stderr, "Minibatch[%lu]: ActualMBSize = %lu\n", ++numMBsRun, actualMBSize);
        }

        for (auto & onode : outputNodes)
        {
            FILE * f = *outputStreams[onode];
            fprintfOrDie(f, "%s", formattingOptions.epilogue.c_str());
        }

        delete[] tempArray;

        fprintf(stderr, "Total Samples Evaluated = %lu\n", totalEpochSamples);

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
