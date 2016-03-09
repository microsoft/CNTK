//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// NetworkFactory.cpp -- CNTK network creation related functions
//

#include "stdafx.h"
#include "NetworkFactory.h"

template <> /*static*/ const wchar_t* ElemTypeName<float>()  {
    return L"float";
}
template <> /*static*/ const wchar_t* ElemTypeName<double>() {
    return L"double";
}

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
function<ComputationNetworkPtr(DEVICEID_TYPE)> GetNetworkFactory(const ConfigRecordType& config)
{
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);

    if (config.Exists(L"createNetwork"))
    {
        return GetCreateNetworkFn(config); // (we need a separate function needed due to template code)
    }
    else if (config.Exists(L"SimpleNetworkBuilder"))
    {
        const ConfigRecordType& simpleNetworkBuilderConfig(config(L"SimpleNetworkBuilder"));
        auto netBuilder = make_shared<SimpleNetworkBuilder<ElemType>>(simpleNetworkBuilderConfig); // parses the configuration and stores it in the SimpleNetworkBuilder object
        return [netBuilder](DEVICEID_TYPE deviceId)
        {
            return shared_ptr<ComputationNetwork>(netBuilder->BuildNetworkFromDescription()); // this operates based on the configuration saved above
        };
    }
    // legacy NDL
    else if (config.Exists(L"NDLNetworkBuilder"))
    {
        const ConfigRecordType& ndlNetworkBuilderConfig(config(L"NDLNetworkBuilder"));
        shared_ptr<NDLBuilder<ElemType>> netBuilder = make_shared<NDLBuilder<ElemType>>(ndlNetworkBuilderConfig);
        return [netBuilder](DEVICEID_TYPE deviceId)
        {
            return shared_ptr<ComputationNetwork>(netBuilder->BuildNetworkFromDescription());
        };
    }
    // legacy test mode for BrainScript. Will go away once we fully integrate with BS.
    else if (config.Exists(L"BrainScriptNetworkBuilder") || config.Exists(L"ExperimentalNetworkBuilder" /*legacy*/))
    {
        // We interface with outer old CNTK config by taking the inner part, which we get as a string, as BrainScript.
        // We prepend a few standard definitions, and also definition of deviceId and precision, which all objects will pull out again when they are being constructed.
        // BUGBUG: We are not getting TextLocations right in this way! Do we need to inject location markers into the source? Moot once we fully switch to BS
        wstring sourceCode = config.Exists(L"BrainScriptNetworkBuilder") ? config(L"BrainScriptNetworkBuilder") : config(L"ExperimentalNetworkBuilder");
        let expr = BS::ParseConfigDictFromString(standardFunctions + computationNodes + commonMacros + msra::strfun::wstrprintf(L"deviceId = %d ; precision = '%ls' ; network = new ComputationNetwork ", (int)deviceId, ElemTypeName<ElemType>()) // TODO: check if typeid needs postprocessing
                                                 + sourceCode,
                                                 vector<wstring>()); // source code has the form [ ... ]
        return [expr](DEVICEID_TYPE /*deviceId*/)
        {
            // evaluate the parse tree--specifically the top-level field 'network'--which will create the network
            let object = EvaluateField(expr, L"network");                   // this comes back as a BS::Object
            let network = dynamic_pointer_cast<ComputationNetwork>(object); // cast it
            // This should not really fail since we constructed the source code above such that this is the right type.
            // However, it is possible (though currently not meaningful) to locally declare a different 'precision' value.
            // In that case, the network might come back with a different element type. We need a runtime check for that.
            if (!network)
                RuntimeError("BuildNetworkFromDescription: network has the wrong element type (float vs. double)");
            // success
            return network;
        };
    }
    else
    {
        RuntimeError("No network builder found in the config file. NDLNetworkBuilder or SimpleNetworkBuilde must be specified");
    }
}

template function<ComputationNetworkPtr(DEVICEID_TYPE)> GetNetworkFactory<ScriptableObjects::IConfigRecord, float>(const ScriptableObjects::IConfigRecord& config);
template function<ComputationNetworkPtr(DEVICEID_TYPE)> GetNetworkFactory<ScriptableObjects::IConfigRecord, double>(const ScriptableObjects::IConfigRecord& config);
template function<ComputationNetworkPtr(DEVICEID_TYPE)> GetNetworkFactory<ConfigParameters, float>(const ConfigParameters& config);
template function<ComputationNetworkPtr(DEVICEID_TYPE)> GetNetworkFactory<ConfigParameters, double>(const ConfigParameters& config);