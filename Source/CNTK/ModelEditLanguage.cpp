//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// NetworkDescriptionLanguage.cpp : Code used to interpret the Network Description Language.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ModelEditLanguage.h"
#include "ConvolutionalNodes.h"
#include "InputAndParamNodes.h"
#include <map>

namespace Microsoft { namespace MSR { namespace CNTK {

// EqualInsensitive - check to see if two nodes are equal up to the length of the first string (must be at least half as long as actual node name)
// TODO: Allowing partial matches seems misguided. We should discourage that, or just remove it.
// string1 - [in,out] string to compare, if comparision is equal insensitive but not sensitive, will replace with sensitive version
// string2 - second string to compare
// alternate - alternate naming of the string
// return - true if strings are equal insensitive and modifies string1 to sensitive version if different
bool EqualInsensitive(std::string& string1, const char* string2, const char* alternate /*=NULL*/)
{
    bool equal = !_strnicmp(string1.c_str(), string2, string1.size());

    // don't allow partial matches that are less than half the string
    if (equal && string1.size() < strlen(string2) / 2)
        equal = false;

    // if we have a (partial) match replace with the full name
    if (equal && strcmp(string1.c_str(), string2))
        string1 = string2;

    if (!equal && alternate != NULL)
    {
        equal = !_strnicmp(string1.c_str(), alternate, string1.size());

        // don't allow partial matches that are less than half the string
        if (equal && string1.size() < strlen(alternate) / 2)
            equal = false;

        // if we have a match of the alternate string replace with the full name
        if (equal)
            string1 = string2;
    }

    return equal;
}

// MELProperty - the properties for SetProperty
enum MELProperty
{
    melPropNull,
    melPropParameterUpdateRequired,
    melPropLearningRateMultiplier,
    melPropFeature,
    melPropLabel,
    melPropFinalCriterion,
    melPropEvaluation,
    melPropOutput,
    melPropRecurrent,
    melPropBatchNormMode
};

// SetProperty - Set the Property on the passed node
// nodeProp - node on which the property will be set/cleared
// propArray - Array which contains all nodes that are associated with a particular property
// set - true if property is to be added, false if property is deleted
template <typename ElemType>
void MELScript<ElemType>::SetProperty(ComputationNodeBasePtr nodeProp, vector<ComputationNodeBasePtr>& propArray, bool set)
{
    auto found = propArray.begin();
    for (; found != propArray.end() && *found != nodeProp; ++found)
        ; // loop until you find the node, or the end

    if (set && found == propArray.end())
    {
        propArray.push_back(nodeProp);
    }
    else if (!set && found != propArray.end())
    {
        propArray.erase(found);
    }
}

// ProcessNDLScript - Process the NDL script
// netNdl - netNDL structure
// ndlPassUntil - complete processing through this pass, all passes if ndlPassAll
// fullValidate - validate as a complete network? (false if this might be a snippet of a full network)
template <typename ElemType>
void MELScript<ElemType>::ProcessNDLScript(NetNdl<ElemType>* netNdl, NDLPass ndlPassUntil, bool fullValidate)
{
    NDLUtil<ElemType> ndlUtil(netNdl->cn);
    ndlUtil.ProcessNDLScript(netNdl, ndlPassUntil, fullValidate);
}

// CallFunction - call the MEL function
// name - name of the function to call
// params - parameters to the function
template <typename ElemType>
void MELScript<ElemType>::CallFunction(const std::string& p_name, const ConfigParamList& params)
{
    std::string name = p_name;
    if (EqualInsensitive(name, "CreateModel")) // create a blank model
    {
        size_t numFixedParams = 0, numOptionalParams = 0;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters: CreateModel(). newly created model always becomes the new default.");

        auto cn = make_shared<ComputationNetwork>(CPUDEVICE);
        OverrideModelNameAndSetDefaultModel(cn);
    }
    if (EqualInsensitive(name, "CreateModelWithName")) // create a blank model
    {
        size_t numFixedParams = 1, numOptionalParams = 0;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters: CreateModelWithName(modelName). newly created model always becomes the new default.");

        auto cn = make_shared<ComputationNetwork>(CPUDEVICE);
        OverrideModelNameAndSetDefaultModel(cn, params[0]);
    }
    else if (EqualInsensitive(name, "LoadModel"))
    {
        size_t numFixedParams = 1, numOptionalParams = 1;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters: LoadModel(modelFileName, [format=cntk]). newly loaded model always becomes the new default.");

        std::wstring modelFormat = GetOptionalModelFormat(params, numFixedParams);

        auto cn = make_shared<ComputationNetwork>(CPUDEVICE);
        cn->Load<ElemType>(params[0]);
        OverrideModelNameAndSetDefaultModel(cn);
    }
    else if (EqualInsensitive(name, "LoadModelWithName"))
    {
        size_t numFixedParams = 2, numOptionalParams = 1;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters: LoadModelWithName(modelName, modelFileName, [format=cntk]). newly loaded model always becomes the new default.");

        std::wstring modelFormat = GetOptionalModelFormat(params, numFixedParams);

        auto cn = make_shared<ComputationNetwork>(CPUDEVICE);
#if 1 // support for a specific kind of legacy format, for the sole purpose of allowing users to convert (=load & save) them
        if (modelFormat == L"cntk_legacy_no_tensorlib")
        {
            cn->Read<ElemType>(params[1]);
            for (auto node : cn->FeatureNodes())
                node->SetDims(TensorShape(node->GetSampleMatrixNumRows()), node->HasMBLayout()); // pre-tensorlib InputValues had incorrect tensor dimensions
            cn->CompileNetwork();
        }
        else
#endif
            cn->Load<ElemType>(params[1]);
        OverrideModelNameAndSetDefaultModel(cn, params[0]);
    }
    else if (EqualInsensitive(name, "LoadNDLSnippet"))
    {
        size_t numFixedParams = 2, numOptionalParams = 1;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters: LoadNDLSnippet(modelName, ndlsnippet).");

        string modelName = params[0];
        wstring ndlSnippetFileName = params[1];
        auto cn = make_shared<ComputationNetwork>(CPUDEVICE);
        NDLScript<ElemType> script;
        ConfigParameters ndlScript(script.ReadConfigFile(ndlSnippetFileName));

        // check for a section of the snippet file we wish to read
        std::string section = GetOptionalSnippetSection(params, numFixedParams);

        if (!section.empty())
        {
            if (!ndlScript.Exists(section))
            {
                RuntimeError("Section %s specified in optional parameter was not found in the %ls file\n", section.c_str(), ndlSnippetFileName.c_str());
            }
            ConfigValue ndlSnippet = ndlScript(section);
            EvaluateNDLSnippet(ndlSnippet, cn);
        }
        else
        {
            script.LoadConfigFile(ndlSnippetFileName);
        }

        OverrideModelNameAndSetDefaultModel(cn, modelName);
    }
    else if (EqualInsensitive(name, "SaveDefaultModel"))
    {
        size_t numFixedParams = 1, numOptionalParams = 1;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters: SaveDefaultModel(modelFileName, [format=cntk]).");

        std::wstring modelFormat = GetOptionalModelFormat(params, numFixedParams);

        std::wstring fileName = params[0];

        auto cn = m_netNdlDefault->cn;
        if (cn == NULL)
            RuntimeError("SaveDefaultModel can only be called after a default name exists (i.e., at least one model is loaded.)");

        // validate the network before we save it out
        ProcessNDLScript(m_netNdlDefault, ndlPassAll, true);
        cn->SaveEdited(fileName);
    }
    else if (EqualInsensitive(name, "SaveModel"))
    {
        size_t numFixedParams = 2, numOptionalParams = 1;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters: SaveModel(modelName, modelFileName, [format=cntk]).");

        std::wstring modelFormat = GetOptionalModelFormat(params, numFixedParams);

        std::string modelName = params[0];
        std::wstring fileName = params[1];

        NetNdl<ElemType>* netNdl = &m_mapNameToNetNdl[modelName];
        if (netNdl->cn == NULL)
            RuntimeError("SaveModel can only be called after a network has been setup, no active model named %s.", modelName.c_str());

        // validate and finish the second pass through NDL if any in-line NDL was defined
        ProcessNDLScript(netNdl, ndlPassAll, true);
        netNdl->cn->SaveEdited(fileName);
    }
    else if (EqualInsensitive(name, "SetDefaultModel"))
    {
        size_t numFixedParams = 1, numOptionalParams = 0;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters: SetDefaultModel(modelName)");

        SetExistingModelAsDefault(params[0]);
    }
    else if (EqualInsensitive(name, "UnloadModel"))
    {
        // UnloadModel takes a variable number of parameters, all expected to be model names
        for (int i = 0; i < params.size(); ++i)
        {
            string modelName = params[i];
            auto found = m_mapNameToNetNdl.find(modelName);
            if (found != m_mapNameToNetNdl.end())
            {
                found->second.Clear();
                // if this was the default model, clear it out
                if (&(found->second) == m_netNdlDefault)
                    m_netNdlDefault = nullptr;
                m_mapNameToNetNdl.erase(found);
            }
            else
                fprintf(stderr, "WARNING: model %s does not exist.", modelName.c_str());
        }
    }
    else if (EqualInsensitive(name, "DumpModel", "Dump"))
    {
        size_t numFixedParams = 2, numOptionalParams = 1;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters: DumpNetwork(modelName, fileName, [includeData=false|true])");

        bool includeData = GetOptionalIncludeDataValue(params, numFixedParams);

        std::string modelName = params[0];
        std::wstring fileName = params[1];

        auto found = m_mapNameToNetNdl.find(modelName);
        if (found == m_mapNameToNetNdl.end())
            RuntimeError("Model %s does not exist. Cannot dump non-existant model.", modelName.c_str());
        else
        {
            NetNdl<ElemType>* netNdl = &found->second;
            ProcessNDLScript(netNdl, ndlPassAll, true);
            found->second.cn->DumpAllNodesToFile(includeData, true, fileName);
        }
    }
    else if (EqualInsensitive(name, "DumpNode"))
    {
        size_t numFixedParams = 2, numOptionalParams = 1;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters: DumpNode(nodeName, fileName, [includeData=false|true])");

        bool includeData = GetOptionalIncludeDataValue(params, numFixedParams);

        std::wstring fileName = params[1];

        NetNdl<ElemType>* netNdl;
        vector<ComputationNodeBasePtr> nodes = FindSymbols(params[0], netNdl);
        ProcessNDLScript(netNdl, ndlPassAll);
        netNdl->cn->DumpNodeInfoToFile(nodes, includeData, true, fileName);
    }
    else if (EqualInsensitive(name, "CopyNode", "Copy"))
    {
        size_t numFixedParams = 2, numOptionalParams = 1;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters are: CopyNode(fromNode, toNode, [copy=all|value])");

        CopyNodeFlags copyFlags = GetOptionalCopyNodeFlags(params, numFixedParams);

        std::string from = params[0];
        std::string to = params[1];
        CopyNodes(from, to, copyFlags);
    }
    else if (EqualInsensitive(name, "CopySubTree"))
    {
        size_t numFixedParams = 3, numOptionalParams = 1;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters are: CopySubTree(fromNode, toNetwork, toNodeNamePrefix, [copy=all|value])");

        CopyNodeFlags copyFlags = GetOptionalCopyNodeFlags(params, numFixedParams);

        std::string from = params[0];
        std::string to = params[1];
        std::string prefix = params[2];
        CopySubTree(from, to, prefix, copyFlags);
    }
    else if (EqualInsensitive(name, "CopyNodeInputs", "CopyInputs"))
    {
        size_t numFixedParams = 2, numOptionalParams = 0;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters are: CopyNodeInputs(fromNode, toNode)");

        // get the nodes
        NetNdl<ElemType>* netNdlTo;
        NetNdl<ElemType>* netNdlFrom;
        vector<GenNameValue> names = GenerateNames(params[0], params[1], netNdlFrom, netNdlTo);

        if (netNdlFrom != netNdlTo)
            RuntimeError("CopyInputs requires two symbols from the same network, %s and %s belong to different networks", params[0].c_str(), params[1].c_str());

        ProcessNDLScript(netNdlFrom, ndlPassAll);
        for (GenNameValue name : names)
        {
            auto& node = name.first;
            std::wstring nodeName = node->NodeName();
            std::wstring toNodeName = name.second;

            netNdlTo->cn->CopyNode(*netNdlFrom->cn, nodeName, toNodeName, CopyNodeFlags::copyNodeChildren);
        }
    }
    else if (EqualInsensitive(name, "SetNodeInput", "SetInput"))
    {
        size_t numFixedParams = 3, numOptionalParams = 0;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters are: SetNodeInput(toNode, inputID(0-based), inputNodeName)");

        // get the nodes
        NetNdl<ElemType>* netNdlTo;
        NetNdl<ElemType>* netNdlFrom;
        vector<ComputationNodeBasePtr> nodeTo = FindSymbols(params[0], netNdlTo);
        vector<ComputationNodeBasePtr> nodeFrom = FindSymbols(params[2], netNdlFrom);
        int inputNum = params[1];

        if (netNdlTo != netNdlFrom)
            RuntimeError("SetNodeInput() requires two symbols from the same network, %s and %s belong to different networks", params[0].c_str(), params[2].c_str());

        if (nodeFrom.size() != 1)
            RuntimeError("SetNodeInput() must have a single value input, %s doesn't represent one item", params[0].c_str());
        if (nodeTo.size() < 1)
            RuntimeError("SetNodeInput() must have at least one target, %s doesn't represent any items", params[2].c_str());

        // process outstanding NDL scripts ensuring that the inputs have all been resolved
        ProcessNDLScript(netNdlFrom, ndlPassResolve);
        for (auto& node : nodeTo)
        {
            node->SetInput(inputNum, nodeFrom[0]);
        }
    }
    else if (EqualInsensitive(name, "SetNodeInputs", "SetInputs"))
    {
        if (params.size() > 4 || params.size() < 2)
            RuntimeError("Invalid number of parameters. Valid parameters are: SetNodeInputs(toNode, inputNodeName1, [inputNodeName2, inputNodeName3])");

        // get the nodes
        NetNdl<ElemType>* netNdlTo;
        vector<ComputationNodeBasePtr> nodeTo = FindSymbols(params[0], netNdlTo);
        if (nodeTo.size() != 1)
            RuntimeError("SetNodeInputs() must have exactly one target, %s doesn't represent any node.", params[0].c_str());

        vector<ComputationNodeBasePtr> inputNodes;
        inputNodes.resize(params.size() - 1);

        // process outstanding NDL scripts ensuring that the inputs have all been resolved
        ProcessNDLScript(netNdlTo, ndlPassResolve);

        for (int i = 1; i < params.size(); i++)
        {
            NetNdl<ElemType>* netNdlFrom;
            vector<ComputationNodeBasePtr> nodeFrom = FindSymbols(params[i], netNdlFrom);

            if (netNdlTo != netNdlFrom)
                RuntimeError("SetNodeInputs() requires all symbols from the same network, %s and %s belong to different networks", params[0].c_str(), params[i].c_str());

            if (nodeFrom.size() != 1)
                RuntimeError("SetNodeInputs() each input node should be translated to one node name. %s is translated to multiple node names.", params[i].c_str());

            inputNodes[i - 1] = nodeFrom[0];
        }

        if (inputNodes.size() == 1)
            nodeTo[0]->AttachInputs(inputNodes[0]);
        else if (inputNodes.size() == 2)
            nodeTo[0]->AttachInputs(inputNodes[0], inputNodes[1]);
        else if (inputNodes.size() == 3)
            nodeTo[0]->AttachInputs(inputNodes[0], inputNodes[1], inputNodes[2]);
        else
            RuntimeError("SetNodeInputs(): You specified more than 3 input nodes.");
    }
    else if (EqualInsensitive(name, "SetProperty"))
    {
        if (params.size() != 3)
            RuntimeError("Invalid number of parameters: Valid parameters are: SetProperty(toNode, propertyName, propertyValue)");

        std::string propName = params[1];
        MELProperty prop = melPropNull;
        if (EqualInsensitive(propName, "needGradient", "needsGradient") || EqualInsensitive(propName, "computeGradient"))
        {
            prop = melPropParameterUpdateRequired;  // for backward compatibility
        }
        else if (EqualInsensitive(propName, "learningRateMultiplier"))
        {
            prop = melPropLearningRateMultiplier;
        }
        else if (EqualInsensitive(propName, "feature"))
        {
            prop = melPropFeature;
        }
        else if (EqualInsensitive(propName, "label"))
        {
            prop = melPropLabel;
        }
        else if (EqualInsensitive(propName, "criterion") || /*legacy:*/EqualInsensitive(propName, "finalCriterion", "Criteria"))
        {
            prop = melPropFinalCriterion;
        }
        else if (EqualInsensitive(propName, "multiSeq", "reqMultiSeqHandling")) // legacy
        {
            fprintf(stderr, "WARNING: '%s' property is defunct and will be ignored.\n", propName.c_str());
        }
        else if (EqualInsensitive(propName, "evaluation", "eval")) // TODO: choose one
        {
            prop = melPropEvaluation;
        }
        else if (EqualInsensitive(propName, "batchNormEvalMode"))
        {
            prop = melPropBatchNormMode;
        }
        else if (EqualInsensitive(propName, "output"))
        {
            prop = melPropOutput;
        }
        else if (EqualInsensitive(propName, "recurrent"))
        {
            prop = melPropRecurrent;
        }
        else
        {
            RuntimeError("Invalid property, %s, is not supported", propName.c_str());
        }

        // get the nodes
        NetNdl<ElemType>* netNdl;
        vector<ComputationNodeBasePtr> nodes = FindSymbols(params[0], netNdl);

        // this probabably won't do anything, but make sure all NDL has been created
        ProcessNDLScript(netNdl, ndlPassInitial, false);

        auto cn = netNdl->cn;
        for (auto& node : nodes)
        {
            switch (prop)
            {
            case melPropParameterUpdateRequired:  // for backward compatibility
            {
                node->SetLearningRateMultiplier((bool)params[2] ? 1.0f : 0);
                break;
            }
            case melPropLearningRateMultiplier:
            {
                node->SetLearningRateMultiplier((float)params[2]);
                break;
            }
            case melPropFeature:
            {
                bool set = params[2];
                SetProperty(node, cn->FeatureNodes(), set);
                break;
            }
            case melPropLabel:
            {
                bool set = params[2];
                SetProperty(node, cn->LabelNodes(), set);
                break;
            }
            case melPropFinalCriterion:
            {
                bool set = params[2];
                SetProperty(node, cn->FinalCriterionNodes(), set);
                break;
            }
            case melPropEvaluation:
            {
                bool set = params[2];
                SetProperty(node, cn->EvaluationNodes(), set);
                break;
            }
            case melPropOutput:
            {
                bool set = params[2];
                SetProperty(node, cn->OutputNodes(), set);
                break;
            }
            case melPropRecurrent:
            {
                // what to do here?
                break;
            }
            case melPropBatchNormMode:
            {
                if (node->OperationName() != OperationNameOf(BatchNormalizationNode))
                {
                    RuntimeError("Invalid node type: node %ls (type:%ls) is not a %ls node; therefore cannot apply batchNormEvalMode on it.",
                                 node->NodeName().c_str(),
                                 node->OperationName().c_str(),
                                 OperationNameOf(BatchNormalizationNode).c_str());
                }
                bool property = params[2];
                auto pnode = dynamic_pointer_cast<BatchNormalizationNode<float>>(node);
                if (pnode)
                    pnode->SetEvalMode(property);
                else
                {
                    auto pnode2 = dynamic_pointer_cast<BatchNormalizationNode<double>>(node);
                    if (pnode2)
                        pnode2->SetEvalMode(property);
                    else
                    {
                        RuntimeError("Invalid node type: node name=%ls. We assume either BatchNormalizationNode<float> or BatchNormalizationNode<double>\n",
                                     node->NodeName().c_str());
                    }
                }
                break;
            }
            default:
            {
                RuntimeError("Invalid property, %s, is not supported", propName.c_str());
                break;
            }
            }
        }
    }
    else if (EqualInsensitive(name, "SetPropertyForSubTree"))
    {
        size_t numFixedParams = 3, numOptionalParams = 0;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters are: SetPropertyForSubTree(rootNodeName, propertyName, propertyValue)");

        std::string propName = params[1];
        MELProperty prop = melPropNull;

        if (EqualInsensitive(propName, "needGradient", "needsGradient") || EqualInsensitive(propName, "computeGradient"))
        {
            prop = melPropParameterUpdateRequired;  // for backward compatability
        }
        else if (EqualInsensitive(propName, "learningRateMultiplier"))
        {
            prop = melPropLearningRateMultiplier;
        }
        else if (EqualInsensitive(propName, "batchNormEvalMode"))
        {
            prop = melPropBatchNormMode;
        }
        else
        {
            RuntimeError("Invalid property, %s, is not supported", propName.c_str());
        }

        // get the nodes
        NetNdl<ElemType>* netNdl;
        vector<ComputationNodeBasePtr> nodes = FindSymbols(params[0], netNdl);

        // make sure all NDL links have been resolved
        ProcessNDLScript(netNdl, ndlPassResolve);

        for (auto& node : nodes)
        {
            switch (prop)
            {
            case melPropParameterUpdateRequired:  //for backward compatibility
            {
                float learningRateMultiplier = (bool)params[2] ? 1.0f : 0;
                netNdl->cn->SetLearnableNodesBelowLearningRateMultiplier(learningRateMultiplier, node);
                break;
            }
            case melPropLearningRateMultiplier:
            {
                float learningRateMultiplier = (float)params[2];
                netNdl->cn->SetLearnableNodesBelowLearningRateMultiplier(learningRateMultiplier, node);
                break;
            }
            case melPropBatchNormMode:
            {
                bool evalMode = params[2];
                netNdl->cn->SetBatchNormalizationNodesBelowEvalMode(evalMode, node);
                break;
            }
            default:
            {
                RuntimeError("Invalid property, %s, is not supported", propName.c_str());
                break;
            }
            }
        }
    }
    else if (EqualInsensitive(name, "RemoveNode", "Remove") || EqualInsensitive(name, "DeleteNode", "Delete"))
    {
        std::map<NetNdl<ElemType>*, bool> processed;
        // remove takes a variable number of parameters, all expected to be node names or wildcard patterns
        for (int i = 0; i < params.size(); ++i)
        {
            // get the nodes
            NetNdl<ElemType>* netNdl;
            vector<ComputationNodeBasePtr> nodes = FindSymbols(params[i], netNdl);

            // make sure all NDL has been processed in case we are removing some of them...
            // only process each network once, because validates will start failing after first delete
            if (processed.find(netNdl) == processed.end())
            {
                ProcessNDLScript(netNdl, ndlPassAll, false);
                processed[netNdl] = true;
            }

            if (nodes.size() < 1)
                RuntimeError("Delete must have at least one target, %s doesn't represent any items", params[i].c_str());
            for (const auto& node : nodes)
            {
                netNdl->cn->DeleteNode(node->NodeName());
            }
        }
    }
    else if (EqualInsensitive(name, "Rename"))
    {
        size_t numFixedParams = 2, numOptionalParams = 0;
        if (params.size() > numFixedParams + numOptionalParams || params.size() < numFixedParams)
            RuntimeError("Invalid number of parameters. Valid parameters are Rename(oldNodeName, newNodeName)");

        // get the nodes
        NetNdl<ElemType>* netNdlTo;
        NetNdl<ElemType>* netNdlFrom;
        vector<GenNameValue> nodeNames = GenerateNames(params[0], params[1], netNdlFrom, netNdlTo);

        if (netNdlFrom != netNdlTo)
            RuntimeError("CopyInputs requires two symbols from the same network, %s and %s belong to different networks", params[0].c_str(), params[1].c_str());

        // process everything in case these nodes may have tags on them
        ProcessNDLScript(netNdlFrom, ndlPassAll);

        // now we have the original nodeNames from the input symbol, generate the output nodeNames
        for (GenNameValue nodeName : nodeNames)
        {
            auto& node = nodeName.first;
            netNdlFrom->cn->RenameNode(node, nodeName.second);
        }
    }
    else if (EqualInsensitive(name, "ReviseParameter"))
    {
        typedef LearnableParameter<ElemType> LearnableParameterNode;
        if (params.size() != 2)
            RuntimeError("Invalid number of parameters: Valid parameters are: ReviseParameter(nodeName, nodeParametersInASCIIPathName)");
        std::string nodeName = params[0];
        std::string paramPath = params[1];

        NetNdl<ElemType>* netNdl;
        vector<ComputationNodeBasePtr> nodes = FindSymbols(params[0], netNdl);

        for (auto& pNodes : nodes)
        {
            if (pNodes->OperationName() != LearnableParameter<ElemType>::TypeName())
            {
                fprintf(stderr, "WARNING: you want to change the parameter of node (%ls), but it is not a learnable parameter (it is a %ls node). Skipping this node\n",
                        pNodes->NodeName().c_str(), pNodes->OperationName().c_str());
                continue;
            }
            shared_ptr<LearnableParameterNode> pParamNode = std::dynamic_pointer_cast<LearnableParameterNode>(pNodes);
            pParamNode->ReviseFromFile(msra::strfun::utf16(paramPath));
            fprintf(stderr, "Revise node %ls using parameter file %s\n", pNodes->NodeName().c_str(), paramPath.c_str());
        }
    }
    else
    {
        RuntimeError("Unknown Editor function %s", name.c_str());
    }
}

template class MELScript<float>;
template class MELScript<double>;
} } }
