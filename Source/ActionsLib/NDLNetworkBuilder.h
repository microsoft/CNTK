//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once
#include "NetworkDescriptionLanguage.h"
#include "ComputationNetwork.h"
#include "ComputationNetworkBuilder.h"
#include "Basics.h"
#include <string>
#include "DataReader.h"
#include "Matrix.h"
#include "NDLUtil.h"
#include "ScriptableObjects.h"
#include "BestGpu.h"
#include <stdexcept>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace Microsoft::MSR;

// NDLNodeEvaluatorImpl
// Process the Network Description Language into a Computation Network useable
// by NDLBuilderImpl.
template <typename ElemType>
class NDLNodeEvaluatorImpl : public NDLNodeEvaluator<ElemType>
{
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;

public:
    // Constructor - create evaluator
    NDLNodeEvaluatorImpl(ComputationNetworkPtr cn)
        : m_net(cn)
    {
    }

    // Evaluate - evaluate a node and translate into underlying
    // node - node we are evaluating
    // baseName - base name for all symbols at this level
    // pass - NDLPass through the evaluation (0-initial, 1-resolve variables, 2-final)
    virtual void Evaluate(NDLNode<ElemType>* node, const wstring& baseName, const NDLPass pass);

#ifdef LATER
    // EvaluateDotName - Evaluate a dot name and resolve to target node
    // node - NDLNode of the script
    // nodeParam - NDLNode parameter we are evaluating
    // baseName - name of the base node
    // pass - which pass through the NDL nodes
    // returns: the node that is the evaluated parameter
    virtual NDLNode<ElemType>* EvaluateDotName(NDLNode<ElemType>* node, NDLNode<ElemType>* nodeParam, const std::wstring& baseNameP, const NDLPass pass)

    {
        if (pass > ndlPassInitial && evaluateNode)
        {
            std::string name = nodeParam->GetName();
            std::wstring wname = msra::strfun::utf16(name);
            if (nodeParam->GetType() == ndlTypeDotParameter)
            {
                // When we see a variable of the form "A.B" in a macro, we need to resolve it to an actual node, by first constructing it's
                // fully-qualified name. There are 2 possibilities:
                // 1) "A" was defined locally within the macro.  In this case, we must find the fully-qualified name of the node that this macro
                //    call is being assigned to (eg, "C" in the example "C=Macro(X)"), and concatenate it's name with "A.B" (eg, "C.A.B").
                // 2) "A" was passed in as a parameter to a macro.  In this case, we must find the fully-qualified name of the node that
                //    was passed in as "A", and replace the "A" and "A.B" with this name.

                // Consider the following example:
                // NdlBLob=[
                //      P=MacroCall1(...)
                //      C=MacroCall2(P)
                // ]
                // # MacroDefinition
                // MacroCall2(X)
                // {
                //      A=MacroCall3(...)
                //      D=Times(A.B,X.B)}
                // }
                //

                // In this example, in the call D=Times(A.B,X.B), we need to resolve A.B and X.B appropriately.
                // Specifically, "A.B" must be resolved to the fully qualified name "C.A.B", whereas "X.B" must be resolved to the fully qualified name "P.B".
                // We then use this fully-qualified name to look up this node in the model (using "m_net->GetNodeFromName").

                std::size_t firstDotPos = name.find_first_of(".");
                if (firstDotPos == std::string::npos)
                {
                    LogicError("nodeParam of type \"ndlTypeDotParameter\" doesn't have a dot in its name: %s", name.c_str());
                }

                std::string nameBeforeDot = name.substr(0, firstDotPos);
                std::string nameAfterDot = name.substr(firstDotPos + 1, name.size() - (firstDotPos + 1));

                // look up if "nameBeforeDot" was a parameter to the macro.
                NDLNode<ElemType>* resolvedParam = nodeParam->GetParentScript()->FindSymbol(nameBeforeDot);
                if (resolvedParam != nullptr && resolvedParam->GetType() == ndlTypeMacroCall)
                {
                    // if "nameBeforeDot" was a parameter to the macro, builds it's fully qualified name by
                    // replacing "nameBeforeDot" with the fully qualified name of the node passed in as the parameter.
                    NDLScript<ElemType>* parentScript = resolvedParam->GetParentScript();
                    baseName = parentScript->GetBaseName();
                    std::wstring resolvedParamName = msra::strfun::utf16(resolvedParam->GetName());
                    wname = baseName.empty() ? resolvedParamName + L"." + msra::strfun::utf16(nameAfterDot) : baseName + L"." + resolvedParamName + L"." + msra::strfun::utf16(nameAfterDot);
                }
                else if (!baseName.empty())
                {
                    // else, "nameBeforeDot" wasn't a parameter to the macro, so treat it as a local variable.
                    wname = baseName + L"." + wname;
                }
            }
            else if (!baseName.empty())
            {
                wname = baseName + L"." + wname;
            }

            // fully qualified names can be looked up in the model
            if (m_net->NodeNameExists(wname))
            {
                void* np = (void*)m_net->GetNodeFromName(wname);
                nodeParam->SetEvalValue(np);
            }
            // NOTE: there is a bug here, we allow an abbreviated node reference (i.e. L1.BFF) based on return values in NDL
            // when the actual full node reference that the computational network uses would be L1.BFF.FF.P, so that is what CN sees
            // can we do the normal find symbol here to allow abbreviated node references?

            // if we still didn't get a value, throw an error
            if (nodeParam->GetEvalValue() == nullptr)
            {
                LogicError("Dot name could not be resolved '%s': should have a node named '%ls' in computational network\n", nodeParam->GetName().c_str(), name.c_str());
            }
        }
        return nodeParam;
    }
#endif

    // EvaluateParameter - Evaluate a parameter of a call
    // node - NDLNode of the script
    // nodeParam - NDLNode parameter we are evaluating
    // baseName - name of the base node
    // pass - which pass through the NDL nodes
    // returns: the node that is the evaluated parameter
    virtual NDLNode<ElemType>* EvaluateParameter(NDLNode<ElemType>* node, NDLNode<ElemType>* nodeParam, const std::wstring& baseNameP, const NDLPass pass)
    {
        // get the parent script that includes the symbol table we are interested in
        NDLScript<ElemType>* script = node->GetParentScript();
        wstring baseName = baseNameP;
        if (script == NULL)
        {
            std::wstring name = baseName + L"." + msra::strfun::utf16(node->GetName());
            LogicError("no script for a parameter node in call to %ls\n", name.c_str());
        }

        // evaluate the parameter if we haven't yet, or if we are in the resolve pass (need to set the inputs)
        bool evaluateNode = nodeParam->GetEvalValue() == NULL || pass == ndlPassResolve;
        switch (nodeParam->GetType())
        {
            // if the node is a parameter then look it up in the symbol table
        case ndlTypeUndetermined: // an undetermined parameter needs to be looked up again in the symbol table
        case ndlTypeParameter:
        {
            // lookup the parameter
            NDLNode<ElemType>* nodeResolve = script->FindSymbol(nodeParam->GetName());

            // if we have resolved the name, no need to continue evaluation
            if (!(pass == ndlPassResolve && nodeResolve && nodeParam->GetEvalValue() == nullptr))
            {
                break;
            }
            if (pass > ndlPassInitial && evaluateNode && nodeResolve)
            {
                std::string name = nodeResolve->GetName();
                // we need to start from the parent script, because that is the namespace of the parameter being passed in
                NDLScript<ElemType>* parentScript = nodeResolve->GetParentScript();
                nodeResolve = parentScript->FindSymbol(name);

                // if we still didn't get a value
                if (nodeResolve == nullptr || nodeResolve->GetEvalValue() == nullptr)
                {
                    // check for the fully quantified name in the computation network
                    // this is needed for MEL processing, since CN nodes names can be used as parameters in MEL
                    std::wstring wname = msra::strfun::utf16(name);
                    if (m_net->NodeNameExists(wname))
                    {
                        void* np = (void*)m_net->GetNodeFromName(wname).get();
                        // if we don't have a resolve node, it's because the name didn't exist in NDL
                        if (!nodeResolve)
                            nodeResolve = nodeParam;
                        nodeResolve->SetEvalValue(np);
                    }
                    else
                    {
                        RuntimeError("Parameter name could not be resolved '%s'\n", name.c_str());
                    }
                }
            }
            nodeParam = nodeResolve;
            break;
        }
        case ndlTypeFunction:
            if (evaluateNode)
                Evaluate(nodeParam, baseName, pass);
            break;
        case ndlTypeMacroCall:
            if (evaluateNode)
                nodeParam->EvaluateMacro(*this, baseName, pass);
            break;
            // constants and variables are good as is
        case ndlTypeConstant:
        case ndlTypeVariable:
            break;
            // everything else is illegal as a parameter
        default:
        {
            std::wstring name = baseName + L"." + msra::strfun::utf16(node->GetName());
            RuntimeError("Invalid parameter (macro definitions and arrays not allowed), see call to %ls\n", name.c_str());
        }
        break;
        }
        return nodeParam;
    }

    // EvaluateParameters - Evaluate the parameters of a call
    // node - NDLNode we are evaluating paramters for
    // baseName - baseName for the current node
    // nodeParamStart - starting parameter that contains a node
    // nodeParamCount - ending parameter that contains a node
    // pass - NDL pass we are evaluating
    // returns: vector of eval pointers, which are ComputationNodePtr for CNEvaluator
    virtual std::vector<void*> EvaluateParameters(NDLNode<ElemType>* node, const wstring& baseName, int nodeParamStart, int nodeParamCount, const NDLPass pass)
    {
        std::vector<void*> inputs;
        std::vector<NDLNode<ElemType>*> parameter = node->GetParameters();
        ConfigArray paramString = node->GetParamString();

        if (parameter.size() < 1)
        {
            return inputs;
        }
        if (nodeParamStart + nodeParamCount > parameter.size())
            LogicError("EvaluateParmeters: nodeParamters specified that do not exist");
        size_t numChildren = nodeParamCount;
        for (size_t i = 0; i < numChildren; ++i)
        {
            int index = i + nodeParamStart;
            NDLNode<ElemType>* nodeParam = parameter[index];
            std::wstring paramS = paramString[index];

            // default base is same as current
            std::wstring baseSymbol = baseName;

            NDLNode<ElemType>* nodeResult = EvaluateParameter(node, nodeParam, baseSymbol, pass);
            // look for a prefix here and set baseName appropriately

            if (pass == ndlPassResolve)
            {
                void* np = nodeResult->GetEvalValue();
                assert(np != nullptr);
                inputs.push_back((void*)np);
            }
            else if (pass == ndlPassInitial) // for initial pass we are only interested in resolved nodes (to get constant values)
            {
                inputs.push_back((void*)nodeResult);
            }
            // NOTE: in final pass inputs are always NULL
        }

        // now return the vector
        return inputs;
    }

    // ProcessOptionalParameters - Process the optional parameters of a node
    virtual void ProcessOptionalParameters(NDLNode<ElemType>* node)
    {
        vector<NDLNode<ElemType>*> params = node->GetParameters(true); // get all the optional parameters only
        auto compNode = ComputationNode<ElemType>::FromVoidPtr(node->GetEvalValue());
        std::string empty;

        // loop through all the optional parameters processing them as necessary
        for (NDLNode<ElemType>* param : params)
        {
            // we only process the "tag" optional parameter for now
            if (!EqualCI(param->GetName(), "tag"))
                continue;

            std::string value = param->GetValue();

            // deal with legacy
            if      (EqualCI(value, "multiSeq")) continue;                       // ignored (no longer needed)
            else if (EqualCI(value, "criteria"))           value = "criterion";  // legacy (mis-spelled)
            else if (!_strnicmp(value.c_str(), "eval", 4)) value = "evaluation"; // only compare the first 4 characters. Yikes!!

            // map all to lowercase
            std::wstring lvalue = std::wstring(value.begin(), value.end());
            std::transform(lvalue.begin(), lvalue.end(), lvalue.begin(), ::tolower); // note: may crash for chars >127. Don't use those.

            // add to the respective node group
            m_net->AddToNodeGroup(lvalue, compNode);
        }
    }

    // FindSymbol - Search the nodes for a fully quantified symbol
    // symbol - name of the symbol fully quantified name with "dots"
    // returns - pointer to the matching EvalValue for that node, of NULL if not found
    virtual void* FindSymbol(const wstring& symbol)
    {
        if (m_net->NodeNameExists(symbol))
            return m_net->GetNodeFromName(symbol).get();
        return nullptr;
    }

    virtual ~NDLNodeEvaluatorImpl()
    {
    }

protected:
    TensorShape ProcessTensorShapeParameters(const NDLNode<ElemType>* node, const vector<void*>& params, size_t& i, bool isImage, const wstring& cnNodeType /*for error messages only*/);

private:
    ComputationNetworkPtr m_net;
    void operator=(const NDLNodeEvaluatorImpl&);
};

// NDLBuilderImpl
// TODO JC Refactor eligible methods and members into abstract base class.
template <typename ElemType>
class NDLBuilderImpl
{
public:
    NDLBuilderImpl(DEVICEID_TYPE deviceId, unsigned long randomSeedOffset = 0)
    {
        m_computationNetwork = make_shared<ComputationNetwork>(deviceId);
        m_computationNetwork->SetRandomSeedOffset(randomSeedOffset);
        m_nodeEvaluator = new NDLNodeEvaluatorImpl<ElemType>(m_computationNetwork);
    }

    NDLBuilderImpl(ComputationNetworkPtr computationNetwork)
    {
        m_computationNetwork = computationNetwork;
        m_nodeEvaluator = new NDLNodeEvaluatorImpl<ElemType>(m_computationNetwork);
    }

    virtual ~NDLBuilderImpl()
    {
        delete m_nodeEvaluator;
    }

    ComputationNetworkPtr GetComputationNetwork()
    {
        return m_computationNetwork;
    }

    NDLNodeEvaluator<ElemType>& GetNodeEvaluator()
    {
        return *m_nodeEvaluator;
    }

private:
    ComputationNetworkPtr m_computationNetwork;
    NDLNodeEvaluatorImpl<ElemType>* m_nodeEvaluator;

protected:
    // Copy constructor, should never be called.
    NDLBuilderImpl(const NDLBuilderImpl<ElemType>& /*deepCopyFrom*/)
    {
        LogicError("'NDLBuilderImpl(const NDLBuilderImpl<ElemType>& deepCopyFrom)' should never be called.");
    }

    // Assignment operator, should never be called.
    NDLBuilderImpl<ElemType>& operator=(const NDLBuilderImpl<ElemType>& /*deepCopyFrom*/)
    {
        LogicError("'NDLBuilderImpl<ElemType>& operator=(const NDLBuilderImpl<ElemType>& deepCopyFrom)' should never be called.");
    }
};

template <class ElemType>
class NDLBuilder
{
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;

    NDLScript<ElemType> m_script;
    const ConfigParameters* m_baseConfig; // NOTE: the lifetime of the parent MUST exist from the call to Init to the BuildNetworkFromDescription() call for stringize

public:
    NDLBuilder()
    {
        m_executionEngine = NULL;
        m_baseConfig = NULL;
    } // empty constructor, call Init immediately hereafter

    NDLBuilder(const ConfigParameters& config)
    {
        m_baseConfig = config.GetParent();
        Init(config);
    }
    NDLBuilder(const ScriptableObjects::IConfigRecord&)
    {
        NOT_IMPLEMENTED;
    }

    void Init(
        NDLBuilderImpl<ElemType>* executionEngine,
        const std::wstring& networkConfig,
        const std::string& configParams,
        const std::wstring& dumpFileName,
        DEVICEID_TYPE deviceId)
    {
        m_executionEngine = executionEngine;
        m_networkConfig = networkConfig;
        m_dumpFileName = dumpFileName;
        m_initialConfig = configParams;
        m_deviceId = deviceId;
        m_net = executionEngine->GetComputationNetwork();

        m_net->SetDeviceId(m_deviceId);
        if (m_deviceId < 0)
            fprintf(stderr, "NDLBuilder Using CPU\n");
        else
            fprintf(stderr, "NDLBuilder Using GPU %d\n", m_deviceId);
    }

    // Init - Builder Initialize for multiple data sets
    // config - [in] configuration parameters for the network builder
    virtual void Init(const ConfigParameters& config)
    {
        ConfigParameters newConfig;
        ConfigValue networkConfig = config("networkDescription", "");
        ConfigValue dumpFileName = config("dumpFileName", "");
        DEVICEID_TYPE deviceId = DeviceFromConfig(config);
        unsigned long randomSeedOffset = config("randomSeedOffset", "0");
        auto executionEngine = new NDLBuilderImpl<ElemType>(deviceId, randomSeedOffset);

        // If there was no network configuration file specified:
        //     See if the user specified the run and load sections in config, and if so, load them.
        //     Note that in this case a user MUST specify a "run" section, but the "load" sections are optional
        //     ("load" section is useful for specifying macros in config).
        // Otherwise:
        //     Allow a user to override the "load" and "run" variables specified in the "networkDescription" file.
        if (networkConfig.empty())
        {
            if (config.Exists("load"))
            {
                std::string name = config("load");
                if (!config.Exists(name))
                    RuntimeError("the configuration parameter 'load=%s' doesn't specify another section in this configuration file.\n"
                                 "No 'networkDescription' variable was defined if specifying a separate file was desired.\n ",
                                 name.c_str());

                newConfig.Insert(name, config(name));
                newConfig.Insert("load", name);
            }

            if (!config.Exists("run"))
                RuntimeError("In NDLNetworkBuilder section either a 'networkDescription=filename' or 'run=sectionName' must exist.");

            std::string name = config("run");
            if (!config.Exists(name))
                RuntimeError("the configuration parameter 'run=%s' doesn't specify another section in this configuration file.\n"
                             "No 'networkDescription' variable was defined if specifying a separate file was desired.\n ",
                             name.c_str());

            newConfig.Insert(name, config(name));
            newConfig.Insert("run", name);
        }
        else
        {
            std::string networkConfigString = networkConfig;
            if (networkConfigString.find_first_of("+") != std::string::npos)
                RuntimeError("\"+\" not allowed in \"networkDescription\" value.  Multiple files cannot be specified via \"networkDescription\" parameter. "
                             "In order to load multiple NDL files (eg, for loading several files of macros), use the \"ndlMacros\" parameter.");

            // find the "run" and "load" keys and add them
            if (config.Exists("run"))
            {
                ConfigValue sectionToRun = config("run");
                newConfig.Insert("run", sectionToRun);
            }

            if (config.Exists("load"))
            {
                ConfigValue sectionToLoad = config("load");
                newConfig.Insert("load", sectionToLoad);
            }
        }

        // look for default macros and load them (they load into a global area)
        if (config.Exists("ndlMacros"))
        {
            ConfigValue ndlMacrosPaths = config("ndlMacros");
            NDLScript<ElemType> ndlScript;
            if (!ndlMacrosPaths.empty())
            {
                // load the macro files 1 at a time, so that the sections specified by each file's
                // "load" parameter are in fact loaded (if they were all processed at once, the last file's "load"
                // parameter would override all the earlier ones, and those sections wouldn't get loaded).
                std::vector<std::string> filePathVec = msra::strfun::split(ndlMacrosPaths, "+");
                for (const auto& filePath : filePathVec)
                {
                    ndlScript.LoadConfigFileAndResolveVariables(msra::strfun::utf16(filePath), config);
                }
            }
        }

        Init(executionEngine, networkConfig, newConfig, dumpFileName, deviceId);
    }

    virtual ~NDLBuilder()
    {
        delete m_executionEngine;
    }

    ComputationNetworkPtr LoadNetworkFromConfig(const wstring& configFilePaths, bool forceLoad = true)
    {
        if (m_net->GetTotalNumberOfNodes() == 0 || forceLoad) // not built or force load
            LoadFromConfig(configFilePaths);
        else
            m_net->ResetEvalTimeStamps();
        return m_net;
    }

    // LoadFromConfig - Load a list of files and interpret as network definition file
    // filePaths - A "+" separated list of files to load (full path if needed)
    void LoadFromConfig(const std::wstring& filePaths)
    {
        m_net->ClearNetwork();

        // Process all the config files in order, and then add on the inital config,
        // which will override run and load if they exist
        std::string fileContents = m_script.ReadConfigFiles(filePaths);
        fileContents += m_initialConfig;

        // next replace any $variable$ occurances
        if (m_baseConfig)
        {
            if (m_baseConfig->Exists("NDLNetworkBuilder"))
            {
                fileContents = ((const ConfigParameters&) (*m_baseConfig)("NDLNetworkBuilder")).ResolveVariables(fileContents);
            }
            else
            {
                fileContents = m_baseConfig->ResolveVariables(fileContents);
            }
            m_baseConfig = NULL; // don't want this hanging around past parents going out of scope
        }
        m_script.FileParse(fileContents);

        NDLUtil<ElemType> ndlUtil(m_net);
        ndlUtil.ProcessNDLScript(&m_script, ndlPassAll, nullptr, true, m_dumpFileName);

        // perform necessary post-processing steps
        m_net->CompileNetwork();
    }

    // SetFromConfig - Set the NDL script from a configuration string value
    // config - configuration string containing script
    void SetFromConfig(ConfigValue config)
    {
        NDLUtil<ElemType> ndlUtil(m_net);
        ndlUtil.ProcessNDLConfig(config, true);

        // perform necessary post-processing steps
        m_net->CompileNetwork();
    }

    virtual ComputationNetworkPtr BuildNetworkFromDescription(ComputationNetwork* = nullptr)
    {
        if (m_net->GetTotalNumberOfNodes() < 1) // not built yet
            LoadNetworkFromConfig(m_networkConfig);
        else
            m_net->ResetEvalTimeStamps();
        return m_net;
    }

private:
    ComputationNetworkPtr m_net;
    NDLBuilderImpl<ElemType>* m_executionEngine;
    std::wstring m_networkConfig;
    std::wstring m_dumpFileName;
    std::string m_initialConfig;

    DEVICEID_TYPE m_deviceId;
};

template class NDLBuilder<float>;
template class NDLBuilder<double>;

} } }
