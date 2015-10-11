#pragma once

#include "ComputationNetwork.h"
#include "DataReader.h"
#include <vector>
#include <string>
#include <stdexcept>
#include "Basics.h"
#include "fileutil.h"
#include "commandArgUtil.h"
#include <Windows.h>
#include <WinBase.h>
#include <fstream>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    class RecurrentNetEvaluator : public SimpleEvaluator<ElemType>
    {

    public:

        RecurrentNetEvaluator (ComputationNetwork<ElemType>& net,  const size_t numMBsToShowResult=100) 
            : SimpleEvaluator<ElemType>(net, numMBsToShowResult)
        {
        }

        //returns error rate
//        ElemType Evaluate(IDataReader<ElemType>& dataReader, size_t mbSize,  ElemType &evalSetCrossEntropy, const wchar_t* output=nullptr, size_t testSize=requestDataSize)
        ElemType Evaluate(IDataReader<ElemType>& dataReader, ElemType &evalSetCrossEntropy, const wchar_t* output=nullptr)
        {

            std::vector<ComputationNodePtr> FeatureNodes = m_net.FeatureNodes();
            std::vector<ComputationNodePtr> labelNodes = m_net.LabelNodes();
            std::vector<ComputationNodePtr> evaluationNodes = m_net.EvaluationNodes();
            std::list<ComputationNodePtr> crossEntropyNodes = m_net.GetNodesWithType(L"CrossEntropyWithSoftmax");
            
            if (crossEntropyNodes.size()==0)
            {
                RuntimeError("No CrossEntropyWithSoftmax node found\n");
            }
            if (evaluationNodes.size()==0)
            {
                RuntimeError("No Evaluation node found\n");
            }
            if (crossEntropyNodes.size()==0)
            {
                RuntimeError("Evaluate() does not yet support reading multiple CrossEntropyWithSoftMax Nodes\n");
            }
            if (evaluationNodes.size() == 0)
            {
                RuntimeError("Evaluate() does not yet support reading multiple Evaluation Nodes\n");
            }

            std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
            for (size_t i=0; i<FeatureNodes.size(); i++)
            {
                inputMatrices[FeatureNodes[i]->NodeName()] = &FeatureNodes[i]->FunctionValues();
            }
            for (size_t i=0; i<labelNodes.size(); i++)
            {
                inputMatrices[labelNodes[i]->NodeName()] = &labelNodes[i]->FunctionValues();                
            }

//            dataReader.StartMinibatchLoop(mbSize, 0, testSize);

            ElemType epochEvalError = 0;
            ElemType epochCrossEntropy = 0;
            size_t totalEpochSamples = 0;
            ElemType prevEpochEvalError = 0;
            ElemType prevEpochCrossEntropy = 0;
            size_t prevTotalEpochSamples = 0;
            size_t prevStart = 1;
            size_t numSamples =  0;
            ElemType crossEntropy = 0;
            ElemType evalError = 0;
            
            ofstream outputStream;
            if (output)
            {
                outputStream.open(output);
            }

            size_t numMBsRun = 0;
            size_t actualMBSize = 0;

            GenerateOneSentence(FeatureNodes, labelNodes, 10);

//            while (dataReader.GetMinibatch(inputMatrices))
            while (true)
            {
                actualMBSize = labelNodes[0]->FunctionValues().GetNumCols();

                for (size_t i=0; i<FeatureNodes.size(); i++)
                {
                    FeatureNodes[i]->UpdateEvalTimeStamp();
                }
                for (size_t i=0; i<labelNodes.size(); i++)
                {
                    labelNodes[i]->UpdateEvalTimeStamp();
                }

                size_t npos = 0; 
                for (auto nodeIter = crossEntropyNodes.begin(); nodeIter != crossEntropyNodes.end() && npos < 100; nodeIter++)
                {
                    m_net.Evaluate(evaluationNodes[npos]);
                    ElemType mbEvalError = evaluationNodes[npos]->FunctionValues().Get00Element(); //criterionNode should be a scalar
                    epochEvalError += mbEvalError;
                
                    //std::list<ComputationNodePtr>::iterator iter = crossEntropyNodes.begin(); 
                    //ComputationNodePtr cnp = crossEntropyNodes.front();
                    ComputationNodePtr crossEntropyNode = (*nodeIter);
                    m_net.Evaluate(crossEntropyNode);
                    ElemType mbCrossEntropy = crossEntropyNode->FunctionValues().Get00Element(); // criterionNode should be a scalar
                    epochCrossEntropy += mbCrossEntropy;

                    totalEpochSamples += actualMBSize;
                }
                break; 
            }

            cout << "entropy = " << epochCrossEntropy << endl; 

            if (outputStream.is_open())
            {
                //TODO: add support to dump multiple outputs
                ComputationNodePtr outputNode = m_net.OutputNodes()[0];
                foreach_column(j, outputNode->FunctionValues())
                {
                    foreach_row(i,outputNode->FunctionValues())
                    {
                        outputStream<<outputNode->FunctionValues()(i,j)<<" ";
                    }
                    outputStream<<endl;
                }
            }

            numMBsRun++;


            // show final grouping of output
            numSamples =  totalEpochSamples-prevTotalEpochSamples;
            crossEntropy = epochCrossEntropy - prevEpochCrossEntropy;
            evalError = epochEvalError - prevEpochEvalError;            
            fprintf(stderr,"Minibatch[%lu-%lu]: Samples Evaluated = %lu    EvalErr Per Sample = %.8g    Loss Per Sample = %.8g\n", 
                prevStart, numMBsRun, numSamples, evalError/numSamples, crossEntropy/numSamples);

            //final statistics
            epochEvalError /= (ElemType)totalEpochSamples;
            epochCrossEntropy /= (ElemType)totalEpochSamples;
            fprintf(stderr,"Overall: Samples Evaluated = %lu   EvalErr Per Sample = %.8g   Loss Per Sample = %.8g\n", totalEpochSamples, epochEvalError,epochCrossEntropy);
            if (outputStream.is_open())
            {
                outputStream.close();
            }
            evalSetCrossEntropy = epochCrossEntropy;
            return epochEvalError;
        }
                
        ElemType Evaluate(IDataReader<ElemType>& dataReader, size_t mbSize, const wchar_t* output=nullptr, size_t testSize=requestDataSize)
        {
            ElemType tmpCrossEntropy;
            return Evaluate(dataReader,mbSize,tmpCrossEntropy,output,testSize);
        }

        bool GenerateOneSentence(           
            std::vector<ComputationNodePtr>& FeatureNodes,
            std::vector<ComputationNodePtr>& labelNodes,
            size_t nbrSamples
            )
        {
            for (size_t i = 0; i < nbrSamples; i++)
            {
                for (size_t d = 0; d < FeatureNodes[0]->FunctionValues().GetNumRows(); d++)
                {
                    FeatureNodes[i]->FunctionValues()(d,0) = (ElemType)rand();
                }
                for (size_t d = 0; d < labelNodes[0]->FunctionValues().GetNumRows(); d++)
                {
                    labelNodes[i]->FunctionValues()(d,0) = (ElemType)((d == i)?1:0);
                }
            }
            return true;
        }


    };

}}}