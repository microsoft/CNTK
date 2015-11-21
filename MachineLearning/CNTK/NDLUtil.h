//
// <copyright file="NDLUtil.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "Basics.h"
#include "NetworkDescriptionLanguage.h"
#include "ComputationNetwork.h"
#include "SynchronousExecutionEngine.h"
#include <string>
#include "commandArgUtil.h"
#include <stdexcept>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    class NDLUtil
    {
        typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
    private:
        ComputationNetworkPtr m_net;

    public:
        NDLUtil(ComputationNetworkPtr net) : m_net(net)
        {
        }

        // FixupInputMinibatchSize - go through all the inputs and make sure they have a consistent minibatch size
        void FixupInputMinibatchSize()
        {
            m_net->FixupInputMinibatchSize();
        }

        // ProcessNDLConfig - Process the NDL script from a configuration string value
        // config - configuration string containing script
        void ProcessNDLConfig(const ConfigValue& config, bool fullValidate = false)
        {
            NDLScript<ElemType> script(config);
            ProcessNDLScript(&script, ndlPassAll, nullptr, fullValidate);
        }

        // ProcessNDLScript - Process the NDL script 
        // netNdl - netNDL structure
        // ndlPassUntil - complete processing through this pass, all passes if ndlPassAll
        // fullValidate - validate as a complete network? (false if this might be a snippet of a full network)
        void ProcessNDLScript(NetNdl<ElemType>* netNdl, NDLPass ndlPassUntil=ndlPassAll, bool fullValidate = false)
        {
            ProcessNDLScript(netNdl->ndl, ndlPassUntil, netNdl->lastNode, fullValidate);
        }

        // ProcessNDLScript - Process the NDL script 
        // script - NDL Script to process
        // ndlPassUntil - complete processing through this pass, all passes if ndlPassAll
        // skipThrough - [in/out] for iterative processing, a pointer to an array of NDLNode*, one for each pass
        //               the pointer will be updated to last node processed for that pass, can be NULL if all node processing is desired
        // fullValidate - validate as a complete network? (false if this might be a snippet of a full network)
        void ProcessNDLScript(NDLScript<ElemType>* script, NDLPass ndlPassUntil = ndlPassAll, NDLNode<ElemType>** skipThrough = nullptr, bool fullValidate = false, const std::wstring& dumpFileName = L"")
        {
            // if we don't have a script yet, don't bother
            if (script == nullptr)
                return;

            // set the Computational network in the script, so we can do name lookup in the model
            script->SetComputationNetwork(m_net);

            // loop through the different passes, processing as we go
            // skipThrough (when not null) is a pointer to the following structure in the NetNdl class:
            //     NDLNode<ElemType>* lastNode[ndlPassMax]; // last node we evaluated for each pass
            NDLNode<ElemType>* lastNode = nullptr;
            for (NDLPass ndlPass=ndlPassInitial;ndlPass <= ndlPassUntil;++ndlPass)
            {
                NDLNode<ElemType>* skipThroughNode = skipThrough?*skipThrough:nullptr;
                lastNode = ProcessPassNDLScript(script, ndlPass, skipThroughNode, fullValidate, dumpFileName);
                if (skipThrough)
                {
                    *skipThrough = lastNode;
                    skipThrough++;
                }
            }
        }

        // ProcessPassNDLScript - Process a pass of the NDL script 
        // script - NDL Script to process
        // ndlPass - complete processing for this pass, all passes if ndlPassAll
        // skipThrough - for iterative processing, skip through this node in the script (used for in-line MEL processing)
        // fullValidate - validate as a complete network? (false if this might be a snippet of a full network)
        // returns: last NDL node processed 
        NDLNode<ElemType>* ProcessPassNDLScript(NDLScript<ElemType>* script, NDLPass ndlPass, NDLNode<ElemType>* skipThrough = nullptr, bool fullValidate = false, const std::wstring& dumpFileName = L"")
        {
            if (ndlPass == ndlPassFinal)
            {
                // make sure to clear the caches so we pick up the new nodes
                m_net->ClearCaches();
                // validate the network
                //m_net->SetFakeMBLayoutForValidation();  // fake an MB layout to match the initial values of Input
                if (dumpFileName != L"")
                    m_net->DumpAllNodesToFile(false, dumpFileName, false);
                m_net->ValidateNetwork(!fullValidate);
            }
            SynchronousNodeEvaluator<ElemType> ndlEvaluator(m_net);
            NDLNode<ElemType>* lastNode = script->Evaluate(ndlEvaluator, L"", ndlPass, skipThrough);
            if (ndlPass == ndlPassResolve)
            {
                SetOutputNodes(script);
                FixupInputMinibatchSize();
            }
            return lastNode;
        }


        // CheckOutputNodes - check output nodes
        // symbolName - name of the computation nodes we are collecting
        // compNodes - array of computation nodes
        void CheckOutputNodes(NDLScript<ElemType>* script, std::string symbolName, std::vector<ComputationNodeBasePtr> & compNodes)
        {
            NDLNode<ElemType>* nodeArray = script->FindSymbol(symbolName);
            bool valid = m_net->FeatureNodes().size() > 0; // see if it's already valid
            if (!valid && nodeArray) //otherwise, see if we found a symbol
            {
                NDLType outputType = nodeArray->GetType();
                // accept either an array of nodes, or a single node
                valid = (outputType == ndlTypeArray || outputType == ndlTypeFunction || outputType == ndlTypeMacroCall);
            }
            if (!valid)
                RuntimeError("Invalid network node definition for '%s', nonexistant or wrong type", symbolName.c_str());
            if (nodeArray)
            {
                vector<NDLNode<ElemType>*> nodes;
                if (nodeArray->GetType() == ndlTypeArray)
                    nodes = nodeArray->GetParameters();
                else
                    nodes.push_back(nodeArray);

                for (size_t i=0; i<nodes.size(); i++)
                {
                    // get the computation node
                    auto cnNode = ComputationNode<ElemType>::FromVoidPtr(nodes[i]->GetEvalValue());

                    // if no evaluation value exists throw an error
                    if (cnNode == nullptr)
                    {
                        RuntimeError("Invalid node '%s' as an output node, nonexistant or wrong type", nodes[i]->GetName().c_str());
                    }

                    // see if it's already in the collection
                    bool found = false;
                    for (const auto & compNode : compNodes)
                    {
                        if (cnNode == compNode)
                        {
                            found = true;
                            break;
                        }
                    }

                    // add it if it's not already there
                    if (!found)
                        compNodes.push_back(cnNode);
                }
            }
        }

        // SetOutputNodes - Set the output nodes for the Computational Network
        // NOTE: seems to be specific to SynchronousExecutionEngine, should be in a derived class for that execution engine
        void SetOutputNodes(NDLScript<ElemType>* script)
        {
            // NOTE: all optional parameter nodes (i.e. tag=feature) have already been processed in ProcessOptionalParameters()

            // handle the alternate way of specifying nodes, the array of nodes method
            CheckOutputNodes(script, "FeatureNodes", m_net->FeatureNodes());
            CheckOutputNodes(script, "LabelNodes", m_net->LabelNodes());
            CheckOutputNodes(script, "CriteriaNodes", m_net->FinalCriterionNodes());
            CheckOutputNodes(script, "EvalNodes", m_net->EvaluationNodes());
            CheckOutputNodes(script, "OutputNodes", m_net->OutputNodes());
        }
    };

    template class NDLUtil<float>; 
    template class NDLUtil<double>;

}}}