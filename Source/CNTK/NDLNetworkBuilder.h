//
// <copyright file="NDLNetworkBuilder.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once
#include "NetworkDescriptionLanguage.h"
#include "ComputationNetwork.h"
#include "IExecutionEngine.h"
#include "Basics.h"
#include <string>
#include "DataReader.h"
#include "Matrix.h"
#include "NDLUtil.h"
#include "ScriptableObjects.h"
#include <stdexcept>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace Microsoft::MSR;

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
        IExecutionEngine<ElemType>* executionEngine,
        const std::wstring& networkConfig,
        const std::string& configParams,
        const std::wstring& dumpFileName,
        DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
    {
        m_executionEngine = executionEngine;
        m_networkConfig = networkConfig;
        m_dumpFileName = dumpFileName;
        m_initialConfig = configParams;
        m_deviceId = deviceId;
        m_net = executionEngine->GetComputationNetwork();
        if (m_deviceId == AUTOPLACEMATRIX)
            m_deviceId = Matrix<ElemType>::GetBestGPUDeviceId();
        m_deviceId = EnforceOneGPUOnly(m_deviceId); // see EnforceOneGPUOnly() for comment on what this is

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
        auto executionEngine = new SynchronousExecutionEngine<ElemType>(deviceId, randomSeedOffset);

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
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
            LoadNetworkFromConfig(m_networkConfig);
        else
            m_net->ResetEvalTimeStamps();
        return m_net;
    }

private:
    ComputationNetworkPtr m_net;
    IExecutionEngine<ElemType>* m_executionEngine;
    std::wstring m_networkConfig;
    std::wstring m_dumpFileName;
    std::string m_initialConfig;

    DEVICEID_TYPE m_deviceId;
};

template class NDLBuilder<float>;
template class NDLBuilder<double>;
} } }
