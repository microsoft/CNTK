//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ScriptableObjects.h"

#include "ComputationNode.h"
#include "InputAndParamNodes.h"
#include "RecurrentNodes.h"
#include "NonlinearityNodes.h"
#include "LinearAlgebraNodes.h"
#include "ReshapingNodes.h"

#include "ComputationNetwork.h"
#include "ComputationNetworkBuilder.h"

#include <memory>
#include <deque>
#include <set>
#include <string>

#ifndef let
#define let const auto
#endif

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace Microsoft::MSR::ScriptableObjects;

// ===================================================================
// construction from config
// ===================================================================

// construct a ComputationNetwork from a ConfigRecord
ComputationNetwork::ComputationNetwork(const IConfigRecordPtr configp) :
    ComputationNetwork()
{
    let& config = *configp;

    DEVICEID_TYPE deviceId = (DEVICEID_TYPE)(int) config[L"deviceId"];

    deque<ComputationNodeBasePtr> workList;

    // process 'special nodes'
    ProcessSpecialNodes(config, workList);

    // flatten the set of all nodes
    // we collect all root ComputationNodes from the config record, and then expand into all their children by work-list processing
    // TODO: This currently only supports nodes of the same ElemType. We could allow conversion operators.
    for (let& id : config.GetMemberIds())
    {
        let& value = config[id];
        if (value.Is<ComputationNodeBase>())
            workList.push_back((const ComputationNodeBasePtr&) value);
    }

    // TODO: process "outputNodes" etc. arrays: Sync to node Tags, and make them all roots.

    // construct from roots
    ConstructFromRoots(deviceId, move(workList), map<ComputationNodeBasePtr, ComputationNodeBasePtr>()/*no mapping*/);
}

// process the special-nodes parameters
void ComputationNetwork::ProcessSpecialNodes(const ScriptableObjects::IConfigRecord& config, std::deque<ComputationNodeBasePtr>& workList)
{
    set<wstring> nodeGroupNames{ L"feature", L"label", L"criterion", L"evaluation", L"output" };

    for (let& id : config.GetMemberIds())
    {
        let pos = id.find(L"Nodes");
        if (pos == wstring::npos || pos != id.size() - 5)  // special node name = node-group name + L"Nodes"
            continue;
        let nodeGroup = id.substr(0, id.size() - 5);
        if (nodeGroupNames.find(nodeGroup) == nodeGroupNames.end())
            continue;

        let nodeSet = config[id];
        let nodes = ScriptableObjects::ConfigArray::FlattenedVectorFrom<ComputationNodeBasePtr>(nodeSet);
        for (let& node : nodes)
        {
            node->SetTag(nodeGroup);
            workList.push_back(node);
        }
    }
}

// construct a network from a list of roots (passed in 'workList')
// This will add to m_nameToNodeMap[] all roots and all nodes reachable from those roots.
// If 'replacements' is given, all root pointers as well as all input pointers of reachable nodes will be mapped. This is needed for model editing.
void ComputationNetwork::ConstructFromRoots(DEVICEID_TYPE deviceId, deque<ComputationNodeBasePtr>&& workList, const map<ComputationNodeBasePtr, ComputationNodeBasePtr>& replacements)
{
    SetDeviceId(deviceId);
    assert(this->GetTotalNumberOfNodes() == 0);

    // replace if requested
    // This happens for model editing.
    // workList operates on mapped nodes.
    size_t numRelinked = 0;
    for (auto& nodeRef : workList)
    {
        let iter = replacements.find(nodeRef);
        if (iter != replacements.end())
        {
            assert(nodeRef->GetEnvironmentPtr()); // must be in some network if mapped
            nodeRef = iter->second; // nodeRef is a reference, so this patches the workList in-place
            numRelinked++;
        }
    }

    // process work list
    // Also call LateAttachInputs() where needed.
    while (!workList.empty())
    {
        let node = workList.front();
        workList.pop_front();

        // add to set
        let wasAdded = AddNodeToNetIfNotYet(node, /*makeUniqueName=*/ true);
        if (!wasAdded) // node already there (above will fail if there is a different node with the same name)
            continue;

        // If node derives from ILateAttachingNode() then it has unresolved inputs. Resolve them now.
        // This may generate a whole new load of nodes, including nodes which in turn have late init.
        // Note: In case of editing, we may be adding a new node that references nodes from the old
        // network that must be mapped because their inputs have changed. Hence, it is important to
        // to the mapping *after* late attaching.
        if (node->GetNumInputs() == 0) // (if this function is called during model editing, we may already have our inputs)
        {
            let lateAttachingNode = dynamic_pointer_cast<ILateAttachingNode>(node);
            if (lateAttachingNode)
                lateAttachingNode->LateAttachInputs();
        }

        // add it to the respective node groups based on the tags
        for (auto tag : node->GetTags())
        {
#if 1       // we keep this for a while (we already verified that our samples no longer use this)
            // map legacy names
            if      (tag == L"criteria") tag = L"criterion";
            else if (tag == L"eval"    ) tag = L"evaluation";
#endif
            AddToNodeGroup(tag, node); // tag may be empty, or may have been set by array parameters
        }

        // traverse children: append them to the end of the work list
        // In case of model editing, map inputs.
        for (size_t i = 0; i < node->GetNumInputs(); i++)
        {
            auto input = node->Input(i);

            // replace input if needed
            let iter = replacements.find(input);
            if (iter != replacements.end())
            {
                assert(input->GetEnvironmentPtr()); // must be in some network if mapped
                input = iter->second;
                numRelinked++;
                node->SetInput(i, input);
            }

            workList.push_back(input); // (we could check whether c is in 'nodes' already here to optimize, but this way it is cleaner)
        }
    }
    if (numRelinked > 0)
        fprintf(stderr, "ConstructFromRoots: %d references were remapped.", (int)numRelinked);

    // perform all necessary post-processing
    CompileNetwork();
}

// ===================================================================
// behave like a config
// This allows to access nodes inside a network as if it was an IConfigRecord.
// This is meant to be used by whatever we will replace MEL.
// TODO: Is there more than nodes that we want to return? Node groups? deviceId?
// ===================================================================

// not in the cache yet: create it (or not if no such member)
void /*CustomConfigRecord::*/ ComputationNetwork::LazyCreateConfigMember(const wstring& id) const /*override*/
{
    auto iter = m_nameToNodeMap.find(id);
    if (iter == m_nameToNodeMap.end())
    {
        // workaround to allow to access members with '.' inside: change to _
        for (iter = m_nameToNodeMap.begin(); iter != m_nameToNodeMap.end(); iter++)
            if (msra::strfun::ReplaceAll<wstring>(iter->first, L".", L"_") == id)
                break;
        if (iter == m_nameToNodeMap.end())
            return; // no such node
    }
    const ComputationNodeBasePtr& node = iter->second;
    // TODO: What is the expressionPath?
    let& nodeName = node->NodeName();   // failFn lambda below holds a copy of the name for the error message. Let's not hold an unneccessary shared_ptr to the node, risking cycles & stuff.
    auto valuep = ConfigValuePtr(node, [nodeName](const std::wstring &) { LogicError("ComputationNetwork: Failed to retrieve node '%ls'.", nodeName.c_str()); }, node->NodeName());
    InsertConfigMember(id, move(valuep));
}

vector<wstring> /*IConfigRecord::*/ ComputationNetwork::GetMemberIds() const
{
    set<wstring> nodeNames;
    for (let& iter : m_nameToNodeMap)
    {
        const ComputationNodeBasePtr& node = iter.second;
        wstring nodeName = node->NodeName();
        if (nodeName.find_first_of(L"$") != nodeName.npos) // skip non-top-level names
            continue;
        // temp solution for composites: use _ instead of .
        nodeName = msra::strfun::ReplaceAll<wstring>(nodeName, L".", L"_");
        if (nodeName.find_first_of(L".[") != nodeName.npos) // skip composite names
            continue;
        nodeNames.insert(nodeName);
    }
    return vector<wstring>(nodeNames.begin(), nodeNames.end());
}

// ===================================================================
// ComputationNetworkFromFile
// scripting wrapper to construct ComputationNetwork from file (aka 'Load')
// ===================================================================

template<class ElemType>
class ComputationNetworkFromFile : public ComputationNetwork
{
public:
    ComputationNetworkFromFile(const IConfigRecordPtr configp) :
        ComputationNetwork()
    {
        let& config = *configp;

        DEVICEID_TYPE deviceId = (DEVICEID_TYPE)(int)config[L"deviceId"];
        SetDeviceId(deviceId);

        wstring pathName = config[L"pathName"];
        fprintf(stderr, "Load: Loading model file: %ls", pathName.c_str());
        Load<ElemType>(pathName); // note that for CNTK_MODEL_VERSION_7 and above, 'ElemType' is ignored
    }
};

ScriptableObjects::ConfigurableRuntimeTypeRegister::AddFloatDouble<ComputationNetworkFromFile<float>, ComputationNetworkFromFile<double>> registerComputationNetworkFromFile(L"ComputationNetworkFromFile");

// ===================================================================
// ComputationNetworkWithEdits
// scripting wrapper to construct by modifying an input network (aka 'Edit')
// ===================================================================

class ComputationNetworkWithEdits : public ComputationNetwork
{
    // helper to execute a BS function that maps a CompuationNode to a ComputationNode
    // The function may return:
    //  - its input --> no edit was made
    //  - an different existing node --> all nodes that use this input should use the returned node instead
    //  - a newly created node or sub-graph --> this node should replace the old one
    // In the latter two cases, the returned node may have inputs that are totally different
    // from the original node's.
    ComputationNodeBasePtr CallEditFunction(ComputationNodeBasePtr node, const ConfigLambda& editFn)
    {
        // wrap the argument in a ConfigValuePtr
        const wstring& nodeName = node->NodeName();
        const wstring& expressionName = nodeName;   // TODO: think this through
        auto valuep = ConfigValuePtr(static_pointer_cast<Object>(node), [nodeName](const std::wstring &) { LogicError("CallEditFunction: Failed to retrieve node '%ls'.", nodeName.c_str()); }, expressionName);
        vector<ConfigValuePtr> args{ valuep };
        // execute the lambda (this executes a function that is BS)
        ConfigValuePtr result = editFn.Apply(move(args), ConfigLambda::NamedParams(), expressionName);
        // cast the result back
        return result.AsPtr<ComputationNodeBase>();
    }

public:
    // constructor
    // This constructs a new model from an existing one by:
    //  - iterating over all nodes
    //  - trying a sequence of edit functions until one made an edit
    //    This is like pattern matching: The first edit function that matches will return an updated node.
    //  - assemble a new network that consists of the old network with edits applied
    // Note that the old model is not edited in-place; instead a new copy is made that shares
    // unchanged nodes with the original one.
    ComputationNetworkWithEdits(const IConfigRecordPtr configp) :
        ComputationNetwork()
    {
        // get config parameters
        let& config = *configp;
        let& net = config[L"inputModel"].AsRef<ComputationNetwork>();
        let editFunctions = ScriptableObjects::ConfigArray::FlattenedVectorFrom<ConfigLambda>(config[L"editFunctions"]);
        let additionalRoots = ScriptableObjects::ConfigArray::FlattenedVectorFrom<ComputationNodeBasePtr>(config[L"additionalRoots"]);

        // gather all the edits
        // This runs the edit functions over all nodes.
        map<ComputationNodeBasePtr, ComputationNodeBasePtr> replacements; // [orig, replacement] all return values from the Edit-function calls
        let allNodes = net.GetAllNodes();
        for (let& node : allNodes) // iterate over all nodes
        {
            for (let& editFn : editFunctions) // try all edit functions until one matched
            {
                let newNode = CallEditFunction(node, editFn);
                if (newNode != node) // true if the edit function provided a replacement (an "edit")
                {
                    replacements[node] = newNode; // remember the replaceent
                    break;                        // we only apply the first edit function & stop
                }
            }
        }
        fprintf(stderr, "Edit: %d nodes were edited.\n", (int)replacements.size());
#ifdef _DEBUG
        for (let& replacement : replacements)
            fprintf(stderr, "\t%ls = %ls() --> %ls = %ls()\n", replacement.first->NodeName().c_str(), replacement.first->OperationName().c_str(), replacement.second->NodeName().c_str(), replacement.second->OperationName().c_str());
#endif

        // also 'edit' all nodes that have updated *inputs*
        // All nodes that take inputs that have been edited must have their inputs updated.
        // Since we do not update the model in-place, we must also create replacements for these.
        // That is achieved by recursively including all parents of edits into the set of edits.
        let parents = net.CreateParentsMap();
        deque<ComputationNodeBasePtr> workList; // work list for recursion
        for (let& replacement : replacements)
            workList.push_back(replacement.first);
        while (!workList.empty())
        {
            let node = workList.front();
            workList.pop_front();
            // loop over the node's parents
            for (let& parent : parents.find(node)->second)
            {
                // "edit" (clone) the parent if not yet
                if (replacements.find(parent) != replacements.end())
                    continue; // already a known replacement
                // we must "edit" the parent since it depends on a replaced input
                replacements[parent] = parent->Duplicate();
                // and put this parent into the workList, so that we will gets its parent in turn, etc.
                workList.push_back(parent);
#if 0 //def _DEBUG
                fprintf(stderr, "\t%ls = %ls() --> relink %ls\n", parent->NodeName().c_str(), parent->OperationName().c_str(), replacements[parent]->NodeName().c_str());
#endif
            }
        }
        fprintf(stderr, "Edit: %d out of %d nodes were either edited or need to be relinked.\n", (int)replacements.size(), (int)net.GetTotalNumberOfNodes());
        // Now the keys of replacements[] define the set of all nodes that must be relinked.

        // replacements may point to nodes that are replacements themselves
        // This really can only happen if a replacement itself is an old node.
        for (auto& iter : replacements)
            while (replacements.find(iter.second) != replacements.end())
                iter.second = replacements.find(iter.second)->second;

        // Now we have three kinds of nodes:
        //  - unmodified nodes that will be shared with the old network
        //  - modified nodes (user edits and their parents)
        //  - original nodes that are no longer referenced
        // The new network will be constructed to have the same roots as the original.

        // determine all roots
        deque<ComputationNodeBasePtr> roots;
        // process 'special nodes'
        // BUGBUG: This does not allow to unset tags. If special nodes are listed, they should completely override existing tags for the same node.
        ProcessSpecialNodes(config, workList);
        // then the original network
        for (let& node : allNodes)
            if (parents.find(node)->second.empty()) // no parents: it's a root
                roots.push_back(node);
        // also add new roots
        for (let& node : additionalRoots)
            roots.push_back(node);
        fprintf(stderr, "Edit: %d roots to construct the network from.\n", (int)roots.size());
#ifdef _DEBUG
        for (let& node : roots)
            fprintf(stderr, "\t%ls = %ls()\n", node->NodeName().c_str(), node->OperationName().c_str());
#endif
        // The new network is now defined by roots.

        // now construct the new network
        DEVICEID_TYPE deviceId = (DEVICEID_TYPE)(int)config[L"deviceId"];
        ConstructFromRoots(deviceId, move(roots), replacements);
    }
};

ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<ComputationNetworkWithEdits> registerComputationNetworkWithEdits(L"ComputationNetworkWithEdits");

// ===================================================================
// CloneFunctionConfigLambda -- lambda to produce a clone of a network
//  - creates a BrainScript function that carbon-copies a subsection of an existing network
//  - the copy can be shallow or deep, where a deep copy gets its own copy of LearnableParameters
//     - a shallow copy (parameters="shared") is a copy of all nodes that depend on the specified input(s),
//       while all other nodes are shared from the original network section
//     - a deep copy (parameters="lernable" or "constant") also copies all reachable LearnableParameters and their dependents
//     - Input() nodes not listed as `inputNodes` are always shared
//  - the source network may be a different network, e.g. loaded with BS.Network.Load()
//  - a deep copy can be read-only (parameters="constant")
//     - Note: multiple uses of the lambda will not share read-only parameters. This is trickier to implement that one might expect.
//  - example use cases:
//     - adaptation (KL): a frozen read-only copy of the starting model is used as a KL-regularizer
//     - adaptation (DLR): an injected input transform is trained while the network is fixed
//     - image: lower layers of ImageNet networks serve as immutable feature extractors for another image task
//     - DSSM: applying the same network subsection to two inputs
// Usage:
//    f = CloneFunction (inputNodes, outputNodes, parameters="lernable" /*|"constant"|"shared"*/)
// Parameters:
//  - inputNodes:  single node or array of nodes that will become parameters of the function.
//                 Commonly, this list will include all Input()s that the outputNode(s) depend on.
//  - outputNodes: single node or dictionary of nodes that the function will emit
// Example:
//    # create a BS function by copying a piece of network
//    net = CloneFunction (network.features, network.logP)
//    # apply the copy to a new input
//    out = net (myFeatures)
//    # This will create a copy of the subsection from network.features to network.logP
//    # where all links to network.features get replaced by links to myFeatures.
// Example with multiple input and output nodes:
//    # create a BS function by copying a piece of network
//    # This specific example converts a network back into a BrainScript function.
//    # It passes two input nodes --> the BS function will have 2 inputs;
//    # and it passes a record of output nodes --> the BS function will return a record with the same member names
//    network = BS.Network.Load ("some.dnn")
//    net = CloneFunction ((network.features:network.labels), [ ce = network.ce ; errs = network.errs ])
//    # create a network from the BS function
//    features = Input (13)
//    labels = Input (42)
//    out = net (features, labels)
//    criterionNodes = (out.ce)
//    evaluationNodes = (out.errs)
// A specific example: Adapting a network, while using the original network as a regularizer (KLD)
//    # load network
//    network = BS.Network.Load ("some.dnn")
//    # create a trainable clone and a read-only reference clone
//    adaptNet = CloneFunction (network.features, [ z = network.z ], readOnly=false)
//    # create a read-only clone
//    refNet = CloneFunction (network.features, [ z = network.z ], readOnly=true)
//    # create the main network
//    features = Input (42)
//    labels = Input (9000)
//    z = adaptNet (features).z
//    zRef = refNet (features).z
//    # training criterion
//    refWeight = 0.9
//    kldLabels = labels * (1-refWeight) + Softmax (zRef) * refWeight  # interpolate with ref output
//    ce = CrossEntropyWithSoftmax (z, kldLabels)
//    errs = ErrorPrediction (z, labels)
//    criterionNodes = (ce)
//    evaluationNodes = (errs)
// ===================================================================

class CloneFunctionConfigLambda : public ConfigLambda
{
    // how we treat the parameters in the clone
    enum class ParameterTreatment
    {
        learnable, // parameters are copied and kept trainable
        constant,  // parameters are copied and made immutable (e.g. for use of this as a fixed feature extractor)
        shared     // parameters are shared with where they came from (e.g. for parallel identical paths through a network)
    };
public:
    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    // Executing this function from BrainScript merely sets up a lambda, but does not actually create any clone.
    // This is so that the function can be called multiple times in order to create multiple clones.
    CloneFunctionConfigLambda(const IConfigRecordPtr configp) :
        ConfigLambda(CreateParamNames(*configp), NamedParams(), [this](vector<ConfigValuePtr> &&args, NamedParams &&namedArgs, const std::wstring &exprName){ return this->DoClone(args, exprName); })
    {
        let& config = *configp;
        // input nodes
        inputNodes = GetInputNodes(config);
        // output nodes
        let outputNodesParam = config[L"outputNodes"];  // can be a node or a record
        if (outputNodesParam.Is<ComputationNodeBase>()) // scalar case: result is a single node
            outputNodes[L""] = outputNodesParam.AsPtr<ComputationNodeBase>(); // indicated by a "" node name in outputNodes[]
        else                                            // multi-valued case: result is a record of nodes
        {
            let& outputNodesRecord = outputNodesParam.AsRef<IConfigRecord>();
            for (let& nodeName : outputNodesRecord.GetMemberIds())
                outputNodes[nodeName] = outputNodesRecord[nodeName].AsPtr<ComputationNodeBase>();
            if (outputNodes.empty())
                InvalidArgument("CloneFunction: At least one output nodes must be specified.");
        }
        // treatment of parameters
        wstring parametersOption = config[L"parameters"];
        if      (parametersOption == L"learnable") parameterTreatment = ParameterTreatment::learnable;
        else if (parametersOption == L"constant")  parameterTreatment = ParameterTreatment::constant;
        else if (parametersOption == L"shared")    parameterTreatment = ParameterTreatment::shared;
        else InvalidArgument("CloneFunction: 'parameters' option must be 'learnable', 'constant', or 'shared'.");

        // determine which nodes must be cloned
        //  - intersection of:
        //     - all indirect inputs of the specified outputs
        //     - all dependents of leaves
        //  - where leaves are:
        //     - specified inputs
        //     - unless parameters="shared": all parameters the specified outputs depend on

        // determine all indirect inputs of the specified outputs
        vector<ComputationNodeBasePtr> roots;
        for (let& outputNodeKV : outputNodes)
            roots.push_back(outputNodeKV.second);
        let allInputs = ComputationNodeBase::EnumerateNodes(roots);

        // take the chance to validate inputNodes
        let allInputsSet = set<ComputationNodeBasePtr>(allInputs.begin(), allInputs.end());
        for (let& input : inputNodes)
            if (allInputsSet.find(input) == allInputsSet.end())
                InvalidArgument("CloneFunction: No specified output depends on the specified input %ls.", input->NodeDescription().c_str());
        // TODO: Is this really always an error? Are there valid cases where one would over-specify possible input nodes, even if they are not used/needed?

        // determine all leaves and their dependents
        dependentSet = set<ComputationNodeBasePtr>(inputNodes.begin(), inputNodes.end()); // start with the specified inputs
        // determine all leaves and their dependents
        for (let& node : allInputs)
        {
            // add parameters that are to be cloned to dependent set
            if (parameterTreatment != ParameterTreatment::shared && node->Is<IFreezable>())
                dependentSet.insert(node);
            // if at least one input is in the dependent set then this node is, too
            else
                for (let& input : node->GetInputs())
                    if (dependentSet.find(input) != dependentSet.end())
                        dependentSet.insert(node);
        }

#if 0
        for (let& node : dependentSet)
            fprintf(stderr, "CloneFunction: cloning %ls\n", node->NodeDescription().c_str());
#endif

        // ensure none of the specified inputs reference back into the cloned set
        // The function we extract must be separable.
        for (let& input : inputNodes)
            for (let& node : ComputationNodeBase::EnumerateNodes(vector<ComputationNodeBasePtr>{input})) // check all indirect inputs of each specified input
            {
                let iter = dependentSet.find(input);
                if (iter != dependentSet.end() && *iter != input)
                    InvalidArgument("CloneFunction: specified function input %ls recursively depends on %ls inside the function.", input->NodeDescription().c_str(), node->NodeDescription().c_str());
            }
    }

private:
    // get the input nodes from the config
    static vector<ComputationNodeBasePtr> GetInputNodes(const IConfigRecord& config)
    {
        return ScriptableObjects::ConfigArray::FlattenedVectorFrom<ComputationNodeBasePtr>(config[L"inputNodes"]);
    }
    // create an array of parameter names for all inputs
    // These names are never actually used, but required by the ConfigLambda constructor, and maybe useful for debugging.
    static vector<wstring> CreateParamNames(const IConfigRecord& config)
    {
        let inputNodes = GetInputNodes(config);
        vector<wstring> paramNames(inputNodes.size());
        for (size_t i = 0; i < paramNames.size(); i++)
            paramNames[i] = msra::strfun::wstrprintf(L"input_%d", (int)i);
        return paramNames;
    }

private:
    // -----------------------------------------------------------------------
    // the cloning operation itself
    // -----------------------------------------------------------------------

    // execute the lambda
    // This will clone all nodes that the outputNodes depend on, and rewire inputs matching inputNodes to inputArgs.
    ConfigValuePtr DoClone(const vector<ConfigValuePtr>& inputValues, const std::wstring& exprName)
    {
        // resolve the input arguments
        vector<ComputationNodeBasePtr> inputs;
        for (let& inputValue : inputValues)
            inputs.push_back(inputValue.ResolveValue());
        assert(inputValues.size() == inputNodes.size()); // (this should have been checked by BrainScript)

        // do some logging
        fprintf(stderr, "CloneFunction: ");
        for (size_t i = 0; i < inputs.size(); i++)
            fprintf(stderr, "%s%ls : %ls", i == 0 ? "(" : ", ", inputs[i]->NodeName().c_str(), inputs[i]->OperationName().c_str());
        fprintf(stderr, ") -> ");
        let singleOutput = outputNodes.size() == 1 && outputNodes.begin()->first.empty();
        if (singleOutput)
            fprintf(stderr, "%ls\n", outputNodes.begin()->second->NodeDescription().c_str());
        else
        {
            fprintf(stderr, "[\n");
            for (let& outputNodesKV : outputNodes)
                fprintf(stderr, "    %ls = %ls : %ls\n", outputNodesKV.first.c_str(), outputNodesKV.second->NodeName().c_str(), outputNodesKV.second->OperationName().c_str());
            fprintf(stderr, "]\n");
        }

        // clone everything in the dependent set
        //  - specified inputs get mapped to actual parameters
        //  - all others get duplicated
        // Note that at this point, the "shared" option has already been considered,
        // and is reflected in whether parameters are included or not in 'dependentSet'.
        map<ComputationNodeBasePtr, ComputationNodeBasePtr> clonedNodes;
        size_t numCloned = 0;
        for (size_t i = 0; i < inputNodes.size(); i++)
            clonedNodes[inputNodes[i]] = inputs[i];
        for (let& node : dependentSet)
        {
            // if already there then it's an input that we just mapped above
            if (clonedNodes.find(node) != clonedNodes.end())
                continue;
            // clone
            ComputationNodeBasePtr newNode;
            let newName = exprName + L"." + node->GetName();
            newNode = node->Duplicate(newName, CopyNodeFlags::copyNodeAll);
            // make it read-only if desired
            if (parameterTreatment == ParameterTreatment::constant && newNode->Is<IFreezable>())
                newNode->As<IFreezable>()->FreezeParameters();
            // and that's our cloned node
            clonedNodes[node] = newNode;
            numCloned++;
        }
#if 0
        for (let& nodeKV : clonedNodes)
            fprintf(stderr, "CloneFunction: cloning %ls -> %ls (%d -> %d)\n", nodeKV.first->NodeDescription().c_str(), nodeKV.second->NodeDescription().c_str(), (int)nodeKV.first->m_uniqueNumericId, (int)nodeKV.second->m_uniqueNumericId);
#endif

        // all cloned nodes' inputs must be redirected if they reference a node that has been cloned as well
        size_t numRelinks = 0; // (statistics: how many inputs have we relinked?)
        for (let& clonedNodesKV : clonedNodes)
        {
            let& node = clonedNodesKV.second;
            let& inputs = node->GetInputs();
            for (size_t i = 0; i < inputs.size(); i++)
            {
                fprintf(stderr, "%ls.inputs[%d] = %ls (%d)", node->NodeName().c_str(), (int)i, inputs[i]->NodeName().c_str(), (int)inputs[i]->m_uniqueNumericId);
                let iter = clonedNodes.find(inputs[i]);
                if (iter == clonedNodes.end())
                    continue;
                // input is also a cloned node: relink
                node->SetInput(i, iter->second);
                fprintf(stderr, " ==>  %ls (%d)\n", inputs[i]->NodeName().c_str(), (int)inputs[i]->m_uniqueNumericId);
                numRelinks++;
            }
        }

        fprintf(stderr, "CloneFunction: Cloned %d nodes and relinked %d inputs.\n", (int)numCloned, (int)numRelinks);

        // return the result
        //  - if outputNodes was specified as a single node, return a single node
        //  - if specified as a record, then return a record with the specified names

        if (singleOutput)
        {
            return NodeToConfigValuePtr(clonedNodes.find(outputNodes.begin()->second)->second);
        }
        else
        {
            auto record = make_shared<ConfigRecord>(nullptr, [](const std::wstring & msg){ RuntimeError("CloneFunction: %ls", msg.c_str()); });
            for (let& outputNodesKV : outputNodes)
                record->Add(outputNodesKV.first, [](const wstring&){}, move(NodeToConfigValuePtr(clonedNodes.find(outputNodesKV.second)->second)));
            auto valuep = ConfigValuePtr(record, [](const std::wstring &) { LogicError("CloneFunction: Unexpected failure."); }, exprName);
            return valuep;
        }
    }

    ConfigValuePtr NodeToConfigValuePtr(ComputationNodeBasePtr node)
    {
        assert(node);
        auto valuep = ConfigValuePtr(node, [](const std::wstring &) { LogicError("CloneFunction: Unexpected failure."); }, node->NodeName());
        return valuep;
    }

private:
    // parameters
    vector<ComputationNodeBasePtr> inputNodes;
    map<wstring, ComputationNodeBasePtr> outputNodes;
    ParameterTreatment parameterTreatment;
    // other
    set<ComputationNodeBasePtr> dependentSet;                                     // set of nodes that outputNodes depend on
};

ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<CloneFunctionConfigLambda> registerCloneFunctionConfigLambda(L"CloneFunctionConfigLambda");

}}}
