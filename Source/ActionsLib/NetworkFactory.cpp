//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// NetworkFactory.cpp -- CNTK network creation related functions
//

#include "stdafx.h"
#include "Actions.h"
#include "SimpleNetworkBuilder.h"
#include "NDLNetworkBuilder.h"
#include "ScriptableObjects.h"
#include "BrainScriptEvaluator.h"
#include "BrainScriptParser.h"

function<ComputationNetworkPtr(DEVICEID_TYPE)> GetCreateNetworkFn(const ScriptableObjects::IConfigRecord& config)
{
    // createNetwork() is a BrainScript lambda that creates the model
    // We create a C++ wrapper around it, which we then pass to Train().
    auto createNetworkConfigLambda = config[L"createNetwork"].AsPtr<ScriptableObjects::ConfigLambda>();
    return [createNetworkConfigLambda](DEVICEID_TYPE /*deviceId*/)
    {
        // execute the lambda
        vector<ScriptableObjects::ConfigValuePtr> args; // this lambda has no arguments
        ScriptableObjects::ConfigLambda::NamedParams namedArgs;
        let netValue = createNetworkConfigLambda->Apply(move(args), move(namedArgs), L"BuildNetworkFromDescription");
        // typecast the result to the desired type
        return netValue.AsPtr<ComputationNetwork>();
    };
}

function<ComputationNetworkPtr(DEVICEID_TYPE)> GetCreateNetworkFn(const ConfigParameters&)
{
    NOT_IMPLEMENTED;
} // old CNTK config does not support lambdas

template <class ConfigRecordType, typename ElemType>
bool TryGetNetworkFactory(const ConfigRecordType& config, function<ComputationNetworkPtr(DEVICEID_TYPE)>& createNetworkFn)
{
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);

    if (config.Exists(L"createNetwork"))
    {
        createNetworkFn = GetCreateNetworkFn(config); // (we need a separate function needed due to template code)
        return true;
    }
    else if (config.Exists(L"SimpleNetworkBuilder"))
    {
        const ConfigRecordType& simpleNetworkBuilderConfig(config(L"SimpleNetworkBuilder"));
        auto netBuilder = make_shared<SimpleNetworkBuilder<ElemType>>(simpleNetworkBuilderConfig); // parses the configuration and stores it in the SimpleNetworkBuilder object
        createNetworkFn = [netBuilder](DEVICEID_TYPE deviceId)
        {
            return shared_ptr<ComputationNetwork>(netBuilder->BuildNetworkFromDescription()); // this operates based on the configuration saved above
        };
        return true;
    }
    // legacy NDL
    else if (config.Exists(L"NDLNetworkBuilder"))
    {
        const ConfigRecordType& ndlNetworkBuilderConfig(config(L"NDLNetworkBuilder"));
        shared_ptr<NDLBuilder<ElemType>> netBuilder = make_shared<NDLBuilder<ElemType>>(ndlNetworkBuilderConfig);
        createNetworkFn = [netBuilder](DEVICEID_TYPE deviceId)
        {
            return shared_ptr<ComputationNetwork>(netBuilder->BuildNetworkFromDescription());
        };
        return true;
    }
    // legacy test mode for BrainScript. Will go away once we fully integrate with BS.
    else if (config.Exists(L"BrainScriptNetworkBuilder") || config.Exists(L"ExperimentalNetworkBuilder" /*legacy name*/))
    {
        // We interface with outer old CNTK config by taking the inner part, which we get as a string, as BrainScript.
        // We prepend a few standard definitions, and also definition of deviceId and precision, which all objects will pull out again when they are being constructed.
        // BUGBUG: We are not getting TextLocations right in this way! Do we need to inject location markers into the source? Moot once we fully switch to BS
        wstring sourceOfNetwork = config.Exists(L"BrainScriptNetworkBuilder") ? config(L"BrainScriptNetworkBuilder") : config(L"ExperimentalNetworkBuilder");
        if (sourceOfNetwork.find_first_of(L"([") != 0)
            InvalidArgument("BrainScript network description must be either a BS expression in ( ) or a config record in [ ]");

        // set the include paths to all paths that configs were read from; no additional configurable include paths are supported by BrainScriptNetworkBuilder
        auto includePaths = ConfigParameters::GetBrainScriptNetworkBuilderIncludePaths();

        // inject additional items into the source code
        // We support two ways of specifying the network in BrainScript:
        //  - BrainScriptNetworkBuilder = ( any BS expression that evaluates to a ComputationNetwork )
        //  - BrainScriptNetworkBuilder = [ constructor parameters for a ComputationNetwork ]
        if (sourceOfNetwork[0] == '[') // if [ ] form then we turn it into ComputationNetwork by constructing a ComputationNetwork from it
            sourceOfNetwork = L"new ComputationNetwork " + sourceOfNetwork;
        let sourceOfBS = msra::strfun::wstrprintf(L"include \'cntk.core.bs\'\n" // include our core lib. Note: Using lowercase here to match the Linux name of the CNTK exe.
            L"deviceId = %d\n"            // deviceId as passed in
            L"precision = '%ls'\n"        // 'float' or 'double'
            L"network = %ls",             // source code of expression that evaluates to a ComputationNetwork
            (int)deviceId, ElemTypeName<ElemType>(), sourceOfNetwork.c_str());
        let expr = BS::ParseConfigDictFromString(sourceOfBS, move(includePaths));

        // the rest is done in a lambda that is only evaluated when a virgin network is needed
        // Note that evaluating the BrainScript *is* instantiating the network, so the evaluate call must be inside the lambda.
        createNetworkFn = [expr](DEVICEID_TYPE /*deviceId*/)
        {
            // evaluate the parse tree, particularly the top-level field 'network'
            // Evaluating it will create the network.
            let object = EvaluateField(expr, L"network");                   // this comes back as a BS::Object
            let network = dynamic_pointer_cast<ComputationNetwork>(object); // cast it
            if (!network)
                LogicError("BuildNetworkFromDescription: ComputationNetwork not what it was meant to be");
            // success
            return network;
        };
        return true;
    }
    else
        return false;
}

template <class ConfigRecordType, typename ElemType>
function<ComputationNetworkPtr(DEVICEID_TYPE)> GetNetworkFactory(const ConfigRecordType& config)
{
    function<ComputationNetworkPtr(DEVICEID_TYPE)> createNetworkFn;
    bool gotIt = TryGetNetworkFactory<ConfigRecordType, ElemType>(config, createNetworkFn);
    if (!gotIt)
        RuntimeError("No network builder found in the config file. NDLNetworkBuilder, SimpleNetworkBuilder, or BrainScriptNetworkBuilder must be specified");
    else
        return createNetworkFn;
}

// helper to remove all existing Output nodes and replace them by a new given set
static void PatchOutputNodes(const ComputationNetworkPtr& net, const ConfigArray& outputNodeNames, vector<wstring>& outputNodeNamesVector)
{
    // clear out current list of outputNodes
    while (!net->OutputNodes().empty())
        net->RemoveFromNodeGroup(L"output", net->OutputNodes().back());
    // and insert the desired nodes instead
    for (wstring name : outputNodeNames)
    {
        if (!net->NodeNameExists(name))
        {
            fprintf(stderr, "PatchOutputNodes: No node named '%ls'; skipping\n", name.c_str());
            continue;
        }
        outputNodeNamesVector.push_back (name);
        let& node = net->GetNodeFromName(name);
        net->AddToNodeGroup(L"output", node);
    }
}

template <class ConfigRecordType, typename ElemType>
ComputationNetworkPtr GetModelFromConfig(const ConfigRecordType& config, vector<wstring>& outputNodeNamesVector)
{
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);

    ConfigArray outputNodeNames = config(L"outputNodeNames", ConfigArray(""));

    ComputationNetworkPtr net;

    // first try if a NetworkBuilder is present
    function<ComputationNetworkPtr(DEVICEID_TYPE)> createNetworkFn;
    bool gotIt = TryGetNetworkFactory<ConfigRecordType, ElemType>(config, createNetworkFn);
    if (gotIt)
    {
        // We have several ways to create a network.
        net = createNetworkFn(deviceId);
        if (outputNodeNames.size() > 0)
        {
            net->InvalidateCompiledNetwork();
            PatchOutputNodes(net, outputNodeNames, outputNodeNamesVector);
            net->CompileNetwork();
            // BUGBUG: This will generate double Validation output in the log
        }
    }
    else // no NetworkBuilder given: load from 'modelPath'
    {
        wstring modelPath = config(L"modelPath");

        // We don't use CreateFromFile() here since the user might specify OutputNodeNames in the config.
        // By not compiling the network before patching, we avoid double log output for validation.
        net = make_shared<ComputationNetwork>(deviceId);
        net->Read<ElemType>(modelPath);
        if (outputNodeNames.size() > 0)
            PatchOutputNodes(net, outputNodeNames, outputNodeNamesVector);
        net->CompileNetwork();
    }

    return net;
}

template function<ComputationNetworkPtr(DEVICEID_TYPE)> GetNetworkFactory<ScriptableObjects::IConfigRecord, float>(const ScriptableObjects::IConfigRecord& config);
template function<ComputationNetworkPtr(DEVICEID_TYPE)> GetNetworkFactory<ScriptableObjects::IConfigRecord, double>(const ScriptableObjects::IConfigRecord& config);
template function<ComputationNetworkPtr(DEVICEID_TYPE)> GetNetworkFactory<ConfigParameters, float>(const ConfigParameters& config);
template function<ComputationNetworkPtr(DEVICEID_TYPE)> GetNetworkFactory<ConfigParameters, double>(const ConfigParameters& config);
template ComputationNetworkPtr GetModelFromConfig<ConfigParameters, float>(const ConfigParameters& config, vector<wstring>& outputNodeNamesVector);
template ComputationNetworkPtr GetModelFromConfig<ConfigParameters, double>(const ConfigParameters& config, vector<wstring>& outputNodeNamesVector);
