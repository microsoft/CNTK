//
// <copyright file="SimpleOutputWriter.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
#include "DataReader.h"
#include "ComputationNetwork.h"
#include "DataReaderHelpers.h"
#include "fileutil.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    class SimpleOutputWriter
    {
        typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;

    public:

        SimpleOutputWriter(ComputationNetworkPtr net, int verbosity = 0) :
            m_net(net), m_verbosity(verbosity)
        { }

        void WriteOutput(IDataReader<ElemType>& dataReader, size_t mbSize, IDataWriter<ElemType>& dataWriter, const std::vector<std::wstring>& outputNodeNames, size_t numOutputSamples=requestDataSize, bool doUnitTest = false)
        {
            
            //specify output nodes and files
            std::vector<ComputationNodeBasePtr> outputNodes;
            if (outputNodeNames.size() == 0)
            {
                if (m_verbosity > 0)
                    fprintf (stderr, "OutputNodeNames are not specified, using the default outputnodes.\n");
                if (m_net->OutputNodes().size() == 0)
                    LogicError("There is no default output node specified in the network.");

                outputNodes = m_net->OutputNodes();
            }
            else
            {
                for (int i=0; i<outputNodeNames.size(); i++)
                    outputNodes.push_back(m_net->GetNodeFromName(outputNodeNames[i]));
            }

            //specify feature value nodes
            std::vector<ComputationNodeBasePtr>& featureNodes = m_net->FeatureNodes();
            std::vector<ComputationNodeBasePtr>& labelNodes = m_net->LabelNodes();
            std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
            for (size_t i=0; i<featureNodes.size(); i++)
                inputMatrices[featureNodes[i]->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(featureNodes[i])->FunctionValues();
            for (size_t i=0; i<labelNodes.size(); i++)
                inputMatrices[labelNodes[i]->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(labelNodes[i])->FunctionValues();
            //Matrix<ElemType> endOfFile =  Matrix<ElemType>((size_t)1,(size_t)1);
            //endOfFile(0,0)=0;

            // evaluate with minibatches
            dataReader.StartMinibatchLoop(mbSize, 0, numOutputSamples);
            dataReader.SetNumParallelSequences(1);

            m_net->StartEvaluateMinibatchLoop(outputNodes);

            size_t totalEpochSamples = 0;
            std::map<std::wstring, void *, nocase_compare> outputMatrices;

            size_t actualMBSize;
            while (DataReaderHelpers::GetMinibatchIntoNetwork(dataReader, m_net, nullptr, false, false, inputMatrices, actualMBSize))
            {
                ComputationNetwork::UpdateEvalTimeStamps(featureNodes);
                ComputationNetwork::UpdateEvalTimeStamps(labelNodes);

                //size_t actualMBSize = m_net->SetActualMiniBatchSizeFromFeatures();
                //dataReader.CopyMBLayoutTo(m_net->GetMBLayoutPtr());
                //m_net->VerifyActualNumParallelSequences(dataReader.GetNumParallelSequences());

                for (int i=0; i<outputNodes.size(); i++)
                {
                    m_net->Evaluate(outputNodes[i]);
                    outputMatrices[outputNodes[i]->NodeName()] = (void *)(&dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[i])->FunctionValues());
                }

                if (doUnitTest) 
                {
                    std::map<std::wstring, void *, nocase_compare> inputMatricesUnitTest;
                    for (auto iter = inputMatrices.begin(); iter!= inputMatrices.end(); iter++)
                        inputMatricesUnitTest[iter->first] = (void *)(iter->second);
                    dataWriter.SaveData(0, inputMatricesUnitTest, actualMBSize, actualMBSize, 0);
                }
                else 
                    dataWriter.SaveData(0, outputMatrices, actualMBSize, actualMBSize, 0);

                totalEpochSamples += actualMBSize;
            
                /// call DataEnd function in dataReader to do
                /// reader specific process if sentence ending is reached
                dataReader.DataEnd(endDataSentence);
            }           

            if (m_verbosity > 0)
                fprintf(stderr,"Total Samples Evaluated = %lu\n", totalEpochSamples);

            //clean up
            
        }

        void WriteOutput(IDataReader<ElemType>& dataReader, size_t mbSize, std::wstring outputPath, const std::vector<std::wstring>& outputNodeNames, size_t numOutputSamples=requestDataSize)
        {
            msra::files::make_intermediate_dirs (outputPath);

            //specify output nodes and files
            std::vector<ComputationNodeBasePtr> outputNodes;
            if (outputNodeNames.size() == 0)
            {
                fprintf (stderr, "OutputNodeNames are not specified, using the default outputnodes.\n");
                if (m_net->OutputNodes().size() == 0)
                    LogicError("There is no default output node specified in the network.");

                outputNodes = m_net->OutputNodes();
            }
            else
            {
                for (int i=0; i<outputNodeNames.size(); i++)
                    outputNodes.push_back(m_net->GetNodeFromName(outputNodeNames[i]));
            }

            std::vector<ofstream *> outputStreams;
            for (int i=0; i<outputNodes.size(); i++)
#ifdef _MSC_VER
                outputStreams.push_back(new ofstream ((outputPath + L"." + outputNodes[i]->NodeName()).c_str()));
#else
                outputStreams.push_back(new ofstream(wtocharpath(outputPath + L"." + outputNodes[i]->NodeName()).c_str()));
#endif

            //specify feature value nodes
            auto & featureNodes = m_net->FeatureNodes();
            std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
            for (size_t i=0; i<featureNodes.size(); i++)
                inputMatrices[featureNodes[i]->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(featureNodes[i])->FunctionValues();

            // evaluate with minibatches
            dataReader.StartMinibatchLoop(mbSize, 0, numOutputSamples);

            m_net->StartEvaluateMinibatchLoop(outputNodes);

            size_t totalEpochSamples = 0;
            size_t numMBsRun = 0;
            size_t tempArraySize = 0;
            ElemType* tempArray = nullptr;

            size_t actualMBSize;
            while (DataReaderHelpers::GetMinibatchIntoNetwork(dataReader, m_net, nullptr, false, false, inputMatrices, actualMBSize))
            {
                ComputationNetwork::UpdateEvalTimeStamps(featureNodes);

                //size_t actualMBSize = m_net->SetActualMiniBatchSizeFromFeatures();
                //dataReader.CopyMBLayoutTo(m_net->GetMBLayoutPtr());
                //m_net->VerifyActualNumParallelSequences(dataReader.GetNumParallelSequences());  // TODO: This was added by my (fseide) but UNTESTED. If this fails, comment out and let me know.

                for (int i=0; i<outputNodes.size(); i++)
                {
                    m_net->Evaluate(outputNodes[i]);
                    
                    Matrix<ElemType> & outputValues = dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[i])->FunctionValues();
                    ofstream & outputStream = *outputStreams[i];
                    outputValues.CopyToArray(tempArray, tempArraySize);
                    ElemType * pCurValue = tempArray;
                    foreach_column(j, outputValues)
                    {
                        foreach_row(k,outputValues)
                        {
                            outputStream << *pCurValue++ << " ";
                        }
                        outputStream << endl;
                    }
                }

                totalEpochSamples += actualMBSize;
            
                fprintf(stderr,"Minibatch[%lu]: ActualMBSize = %lu\n", ++numMBsRun, actualMBSize);
            }           

            fprintf(stderr,"Total Samples Evaluated = %lu\n", totalEpochSamples);

            //clean up
            for (int i=0; i<outputStreams.size(); i++)
            {
                outputStreams[i]->close();
                delete outputStreams[i];
            }

            delete [] tempArray;
        }
    private:
        ComputationNetworkPtr m_net;
        int m_verbosity;
        void operator=(const SimpleOutputWriter&); // (not assignable)
    };

}}}
