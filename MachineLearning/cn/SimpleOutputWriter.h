//
// <copyright file="SimpleOutputWriter.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "ComputationNetwork.h"
#include "ComputationNetworkHelper.h"
#include "DataReader.h"
#include <vector>
#include <string>
#include <stdexcept>
#include "basetypes.h"
#include "fileutil.h"
#include "commandArgUtil.h"
#include <Windows.h>
#include <WinBase.h>
#include <fstream>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    class SimpleOutputWriter : ComputationNetworkHelper<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr;

    public:

        SimpleOutputWriter(ComputationNetwork<ElemType>& net, int verbosity=0)
            : m_net(net), m_verbosity(verbosity)
        {

        }

		void WriteOutput(IDataReader<ElemType>& dataReader, size_t mbSize, IDataWriter<ElemType>& dataWriter, const std::vector<std::wstring>& outputNodeNames, size_t numOutputSamples=requestDataSize, bool doUnitTest = false)
        {
            
            //specify output nodes and files
            std::vector<ComputationNodePtr> outputNodes;
            if (outputNodeNames.size() == 0)
            {
                if (m_verbosity > 0)
                    fprintf (stderr, "OutputNodeNames are not specified, using the default outputnodes.\n");
                if (m_net.OutputNodes().size() == 0)
                    throw std::logic_error("There is no default output node specified in the network.");

                outputNodes = m_net.OutputNodes();
            }
            else
            {
                for (int i=0; i<outputNodeNames.size(); i++)
                    outputNodes.push_back(m_net.GetNodeFromName(outputNodeNames[i]));
            }

            //specify feature value nodes
            std::vector<ComputationNodePtr>& featureNodes = m_net.FeatureNodes();
            std::vector<ComputationNodePtr> & labelNodes = m_net.LabelNodes();
            std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
            for (size_t i=0; i<featureNodes.size(); i++)
            {
                inputMatrices[featureNodes[i]->NodeName()] = &featureNodes[i]->FunctionValues();
            }
            for (size_t i=0; i<labelNodes.size(); i++)
            {
                inputMatrices[labelNodes[i]->NodeName()] = &labelNodes[i]->FunctionValues();                
            }
			Matrix<ElemType> endOfFile =  Matrix<ElemType>(1,1);
			endOfFile(0,0)=0;

            //evaluate with minibatches
            dataReader.StartMinibatchLoop(mbSize, 0, numOutputSamples);
            dataReader.SetNbrSlicesEachRecurrentIter(1);

            size_t totalEpochSamples = 0;
            std::map<std::wstring, void *, nocase_compare> outputMatrices;

            while (dataReader.GetMinibatch(inputMatrices))
            {
                UpdateEvalTimeStamps(featureNodes);
                UpdateEvalTimeStamps(labelNodes);

                size_t actualMBSize = m_net.GetActualMBSize();
                m_net.SetActualMiniBatchSize(actualMBSize);
                m_net.SetActualNbrSlicesInEachRecIter(dataReader.NumberSlicesInEachRecurrentIter());
				dataReader.SetSentenceEndInBatch(m_net.m_sentenceEnd);

	            for (int i=0; i<outputNodes.size(); i++)
                {
                    m_net.Evaluate(outputNodes[i]);
					outputMatrices[outputNodes[i]->NodeName()] = (void *)(&outputNodes[i]->FunctionValues());
                }

                if (doUnitTest) 
                {
                    std::map<std::wstring, void *, nocase_compare> inputMatricesUnitTest;
                    for (auto iter = inputMatrices.begin(); iter!= inputMatrices.end(); iter++)
                    {
                        inputMatricesUnitTest[iter->first] = (void *)(iter->second);
                    }
                    dataWriter.SaveData(0, inputMatricesUnitTest, actualMBSize, actualMBSize, 0);
                }
                else 
                {
                    dataWriter.SaveData(0, outputMatrices, actualMBSize, actualMBSize, 0);
                }

                totalEpochSamples += actualMBSize;
            
                /// call DataEnd function in dataReader to do
                /// reader specific process if sentence ending is reached
                dataReader.DataEnd(endDataSentence);

            }           

            if (m_verbosity > 0)
                fprintf(stderr,"Total Samples Evaluated = %d\n", totalEpochSamples);

            //clean up
            
        }
		

        void WriteOutput(IDataReader<ElemType>& dataReader, size_t mbSize, std::wstring outputPath, const std::vector<std::wstring>& outputNodeNames, size_t numOutputSamples=requestDataSize)
        {
            msra::files::make_intermediate_dirs (outputPath);

            //specify output nodes and files
            std::vector<ComputationNodePtr> outputNodes;
            if (outputNodeNames.size() == 0)
            {
                fprintf (stderr, "OutputNodeNames are not specified, using the default outputnodes.\n");
                if (m_net.OutputNodes().size() == 0)
                    throw std::logic_error("There is no default output node specified in the network.");

                outputNodes = m_net.OutputNodes();
            }
            else
            {
                for (int i=0; i<outputNodeNames.size(); i++)
                    outputNodes.push_back(m_net.GetNodeFromName(outputNodeNames[i]));
            }

            std::vector<ofstream *> outputStreams;
            for (int i=0; i<outputNodes.size(); i++)
                outputStreams.push_back(new ofstream ((outputPath + L"." + outputNodes[i]->NodeName()).c_str()));

            //specify feature value nodes
            std::vector<ComputationNodePtr>& featureNodes = m_net.FeatureNodes();
            std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
            for (size_t i=0; i<featureNodes.size(); i++)
            {
                inputMatrices[featureNodes[i]->NodeName()] = &featureNodes[i]->FunctionValues();
            }
                        
            //evaluate with minibatches
            dataReader.StartMinibatchLoop(mbSize, 0, numOutputSamples);

            size_t totalEpochSamples = 0;
            size_t numMBsRun = 0;
            size_t tempArraySize = 0;
            ElemType* tempArray = nullptr;

            while (dataReader.GetMinibatch(inputMatrices))
            {
                UpdateEvalTimeStamps(featureNodes);

                size_t actualMBSize = m_net.GetActualMBSize();
                m_net.SetActualMiniBatchSize(actualMBSize);
				dataReader.SetSentenceEndInBatch(m_net.m_sentenceEnd);

                for (int i=0; i<outputNodes.size(); i++)
                {
                    m_net.Evaluate(outputNodes[i]);
					
                    Matrix<ElemType> & outputValues = outputNodes[i]->FunctionValues();
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
            
                fprintf(stderr,"Minibatch[%d]: ActualMBSize = %d\n", ++numMBsRun, actualMBSize);
            }           

            fprintf(stderr,"Totol Samples Evaluated = %d\n", totalEpochSamples);

            //clean up
            for (int i=0; i<outputStreams.size(); i++)
            {
                outputStreams[i]->close();
                delete outputStreams[i];
            }

            delete [] tempArray;
        }
    private:
        ComputationNetwork<ElemType>& m_net;
        int m_verbosity;
    };
    template class SimpleOutputWriter<float>; 
    template class SimpleOutputWriter<double>;

}}}