//
// <copyright file="ComputationNetworkHelper.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "ComputationNetwork.h"
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

    //utility class used by SGD, outputWriter and Evaluator
    template<class ElemType>
    class ComputationNetworkHelper
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr;

    protected:
        void UpdateEvalTimeStamps(const std::vector<ComputationNodePtr> & nodes)
        {
            for (size_t i=0; i<nodes.size(); i++)
            {
                nodes[i]->UpdateEvalTimeStamp();
            }
        }

        void SetDropoutRate(ComputationNetwork<ElemType>& net, const ComputationNodePtr criterionNode, const ElemType dropoutRate, ElemType & prevDropoutRate, ULONG & dropOutSeed)
        {
            if (dropoutRate != prevDropoutRate)
            {
                fprintf(stderr,"Switching dropout rate to %.8g.\n", dropoutRate);
                std::list<ComputationNodePtr> dropoutNodes = net.GetNodesWithType(DropoutNode<ElemType>::TypeName(), criterionNode);
                if (dropoutNodes.size() == 0)
                {
                    fprintf(stderr,"WARNING: there is no dropout node.\n");
                }
                else
                {
                    for (auto nodeIter=dropoutNodes.begin(); nodeIter != dropoutNodes.end(); nodeIter++)
                    {
                        DropoutNode<ElemType>* node = static_cast<DropoutNode<ElemType>*>(*nodeIter);
				        node->SetDropoutRate(dropoutRate);
				        node->SetRandomSeed(dropOutSeed++);
                    }
                }

                prevDropoutRate = dropoutRate;
            }
        }

        void SetMaxTempMemSizeForCNN(ComputationNetwork<ElemType>& net, const ComputationNodePtr criterionNode, const size_t maxTempMemSizeInSamples)
        {
            fprintf(stderr,"Set Max Temp Mem Size For Convolution Nodes to %lu samples.\n", maxTempMemSizeInSamples);
            std::list<ComputationNodePtr> convolutionNodes = net.GetNodesWithType(ConvolutionNode<ElemType>::TypeName(), criterionNode);
            if (convolutionNodes.size() == 0)
            {
                fprintf(stderr,"WARNING: there is no convolution node.\n");
            }
            else
            {
                for (auto nodeIter=convolutionNodes.begin(); nodeIter != convolutionNodes.end(); nodeIter++)
                {
                    ConvolutionNode<ElemType>* node = static_cast<ConvolutionNode<ElemType>*>(*nodeIter);
				    node->SetmMaxTempMemSizeInSamples(maxTempMemSizeInSamples);
                }
            }
        }
    };
}}}