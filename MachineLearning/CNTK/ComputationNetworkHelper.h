//
// <copyright file="ComputationNetworkHelper.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>

#include "Basics.h"
#include "fileutil.h"

#include "ComputationNetwork.h"
#include "NonlinearityNodes.h"  // TODO: move functions that depend on this to a .cpp file
#include "ConvolutionalNodes.h"
#include "DataReader.h"

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    //utility class used by SGD, outputWriter and Evaluator
    // TODO: make independent of ElemType
    // These can be static methods on ComputationNetwork
    template<class ElemType>
    class ComputationNetworkHelper
    {
        typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;

    protected:
        // TODO: make all static?
        static void UpdateEvalTimeStamps(const std::vector<ComputationNodeBasePtr> & nodes)
        {
            for (size_t i=0; i<nodes.size(); i++)
                nodes[i]->UpdateEvalTimeStamp();
        }

        // TODO: why is dropoutRate an ElemType and not a double?
        // TODO: just call twice, once for float and once for double
        static void SetDropoutRate(ComputationNetwork& net, const ComputationNodeBasePtr criterionNode, const ElemType dropoutRate, ElemType & prevDropoutRate, unsigned long & dropOutSeed)
        {
            if (dropoutRate != prevDropoutRate)
            {
                fprintf(stderr,"Switching dropout rate to %.8g.\n", dropoutRate);
                std::list<ComputationNodeBasePtr> dropoutNodes = net.GetNodesWithType(DropoutNode<ElemType>::TypeName(), criterionNode);
                if (dropoutNodes.size() == 0 && dropoutRate > 0)
                {
                    fprintf(stderr,"WARNING: there is no dropout node.\n");
                }
                else
                {
                    for (auto nodeIter=dropoutNodes.begin(); nodeIter != dropoutNodes.end(); nodeIter++)
                    {
                        auto node = dynamic_pointer_cast<DropoutNode<ElemType>>(*nodeIter);
                        node->SetDropoutRate(dropoutRate);
                        node->SetRandomSeed(dropOutSeed++);
                    }
                }

                prevDropoutRate = dropoutRate;
            }
        }

        static void SetMaxTempMemSizeForCNN(ComputationNetwork& net, const ComputationNodeBasePtr criterionNode, const size_t maxTempMemSizeInSamples)
        {
            fprintf(stderr,"Set Max Temp Mem Size For Convolution Nodes to %lu samples.\n", maxTempMemSizeInSamples);
            std::list<ComputationNodeBasePtr> convolutionNodes = net.GetNodesWithType(ConvolutionNode<ElemType>::TypeName(), criterionNode);
            if (convolutionNodes.size() == 0 && maxTempMemSizeInSamples != 0)
            {
                fprintf(stderr,"WARNING: there is no convolution node.\n");
            }
            else
            {
                for (auto nodeIter=convolutionNodes.begin(); nodeIter != convolutionNodes.end(); nodeIter++)
                {
                    auto node = dynamic_pointer_cast<ConvolutionNode<ElemType>>(*nodeIter);
                    node->SetmMaxTempMemSizeInSamples(maxTempMemSizeInSamples);
                }
            }
        }
    };
}}}
