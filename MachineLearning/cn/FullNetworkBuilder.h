//
// <copyright file="FullNetworkBuilder.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "ComputationNetwork.h"
#include "IComputationNetBuilder.h"
#include "basetypes.h"
#include <string>
#include "commandArgUtil.h"
#include "DataReader.h"
#include "Matrix.h"
#include <stdexcept>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    class FullNetworkBuilder : public IComputationNetBuilder<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr;

    public:
        FullNetworkBuilder() {} // empty constructor, call Init immediately hereafter
        FullNetworkBuilder(const ConfigParameters& config)
        {
            Init(config);
        }

        void Init(const std::wstring& networkConfig, short deviceId=AUTOPLACEMATRIX, const bool uniformInit = true, const ElemType initValueScale = 1.0f)
        {
            m_deviceId=deviceId;
            m_networkConfig=networkConfig;
            m_uniformInit=uniformInit;
            m_initValueScale=initValueScale;

            if (m_deviceId == AUTOPLACEMATRIX)
                m_deviceId = Matrix<ElemType>::GetBestGPUDeviceId();

            m_net.SetDeviceID(m_deviceId);
            if (m_deviceId < 0)
                fprintf(stderr,"FullNetworkBuilder Using CPU\n");
            else
                fprintf(stderr,"FullNetworkBuilder Using GPU %d\n", m_deviceId);
        }

        // Init - Builder Initialize for multiple data sets
        // config - [in] configuration parameters for the network builder
        virtual void Init(const ConfigParameters& config)
        {
            const std::wstring& networkConfig = config("networkDescriptionPath");
            short deviceId = config("deviceId", "1000");//AUTOPLACEMATRIX);
            bool uniformInit = config("uniformInit", "true");
            ElemType initValueScale=config("initValueScale", "1.0");

            Init(networkConfig, deviceId, uniformInit, initValueScale);
        }

        virtual ~FullNetworkBuilder()
        {}
        virtual ComputationNetwork<ElemType>& LoadNetworkFromFile(const wstring& modelFileName, bool forceLoad = true) 
        {
            if (m_net.GetTotalNumberOfNodes() == 0 || forceLoad) //not built or force load
                m_net.LoadFromFile(modelFileName);

            m_net.ResetEvalTimeStamp();
            return m_net;
        }

        ComputationNetwork<ElemType>& LoadNetworkFromConfig(const wstring& configFileName, bool forceLoad = true) 
        {
            if (m_net.GetTotalNumberOfNodes() == 0 || forceLoad) //not built or force load
                LoadFromConfig(configFileName);

            m_net.ResetEvalTimeStamp();
            return m_net;
        }

        void SaveNetworkToConfig(const wstring& configFileName) 
        {
            m_net.SaveToConfig(configFileName);
        }

        void LoadFromConfig(const std::wstring& fileName)
        {
            ConfigParameters configCN;
            m_net.ClearNet();
            configCN.LoadConfigFile(fileName);
            ULONG randomSeed = 1;

            std::map<std::wstring, ComputationNodePtr> origNameToNormNameMap;

            ConfigParameters configNodes = configCN("NodeList");
            for(auto iter = configNodes.begin(); iter != configNodes.end(); iter++)
            {
                std::wstring nodeName;
                nodeName = msra::strfun::utf16(iter->first);
                ConfigArray configNode = iter->second;
                std::wstring opName = configNode[0];

                if (opName == L"Input" || opName == InputValue<ElemType>::TypeName())
                {
                    size_t rows = configNode.size() >= 2? configNode[1] : 1;
                    size_t cols = configNode.size() >= 3? configNode[2] : 1;

                    ComputationNodePtr input = m_net.Input(rows, cols, nodeName);
                
                    if (configNode.size()>=4)
                    {
                        std::wstring cmd = configNode[3];
                        transform(cmd.begin(), cmd.end(), cmd.begin(),tolower); 

                        if (cmd == L"mvnorm") // do mean/var normalization
                        {
                            ComputationNodePtr meanPtr = m_net.Mean(input, nodeName + L"Mean");
                            ComputationNodePtr varPtr = m_net.InvStdDev(input, nodeName + L"InvStdDev");
                            ComputationNodePtr normPtr = m_net.PerDimMeanVarNormalization(input, meanPtr, varPtr, nodeName + L"MVNorm");

                            origNameToNormNameMap[nodeName] = normPtr;                            
                        }
                        else
                            throw new runtime_error("for InputValue nodes, only mvNorm is supported 4th argument\n");
                    }
                }
                else if (opName == L"Parameter" || opName == LearnableParameter<ElemType>::TypeName())
                {
                    size_t rows = configNode.size() >= 2? configNode[1] : 1;
                    size_t cols = configNode.size() >= 3? configNode[2] : 1;

                    bool needGradient = false;
                    bool init = false;
                    ConfigArray initData;

                    // look for optional parameters
                    for (int i = 3; i < configNode.size(); ++i)
                    {
                        ConfigParameters configParam = configNode[i];
                        if (configParam.Exists("needGradient"))
                            needGradient = true;
                        else if (configParam.Exists("init"))
                        {
                            init = true;
                            initData = configParam["init"];
                        }
                    }
                    ComputationNodePtr nodePtr = m_net.Parameter(rows, cols, nodeName);
                    nodePtr->NeedGradient() = needGradient;
                    if (init)
                    {
                        m_net.InitLearnableParameters(nodePtr, m_uniformInit,  randomSeed++, m_initValueScale);
                    }
                }
                else if (opName==L"Constant")
                {
                    size_t rows = configNode.size() >= 2? configNode[1] : 1;
                    size_t cols = configNode.size() >= 3? configNode[2] : 1;

                    bool init = false;
                    ElemType val = configNode.size() >= 4? configNode[3] : 0;

                    ComputationNodePtr nodePtr = m_net.Parameter(rows, cols, nodeName);
                    nodePtr->NeedGradient() = false;
                    nodePtr->FunctionValues().SetValue(val);
                }
                else
                {
                    m_net.CreateComputationNode(opName, nodeName);
                }
            }

            //now link up all the nodes
            ConfigParameters configRelation = configCN("Relation");
            for(ConfigParameters::iterator iter = configRelation.begin();
                iter != configRelation.end(); iter++)
            {
                std::wstring nodeName = msra::strfun::utf16(iter->first);
                ConfigArray configNode = iter->second;
                ComputationNodePtr nodePtr = m_net.GetNodeFromName(nodeName);
                vector<ComputationNodePtr> inputs;
                size_t numChildren = configNode.size();
                for (size_t i=0; i < numChildren;++i)
                {
                    ComputationNodePtr cnp;
                    if (origNameToNormNameMap.find(configNode[i])==origNameToNormNameMap.end())
                        cnp = m_net.GetNodeFromName(configNode[i]);
                    else
                        cnp = origNameToNormNameMap[configNode[i]];

                    inputs.push_back(cnp);
                }

                switch (numChildren)
                {
                case 1:
                    nodePtr->AttachInputs(inputs[0]);
                    break;
                case 2:
                    nodePtr->AttachInputs(inputs[0], inputs[1]);
                    break;
                case 3:
                    nodePtr->AttachInputs(inputs[0], inputs[1], inputs[2]);
                    break;
                default:
                    throw std::logic_error("Invalid number of children.");
                }
            }

            ConfigParameters configRoots = configCN("RootNodes");
            ConfigArray configNode = configRoots("FeatureNodes");
            for (size_t i=0; i<configNode.size(); i++)
            {
                std::wstring nodeName = configNode[i];
                m_net.FeatureNodes().push_back(m_net.GetNodeFromName(nodeName));
            }

            configNode = configRoots("LabelNodes");
            for (size_t i=0; i<configNode.size(); i++)
            {
                std::wstring nodeName = configNode[i];
                m_net.LabelNodes().push_back(m_net.GetNodeFromName(nodeName));
            }

            configNode = configRoots("CriteriaNodes");
            for (size_t i=0; i<configNode.size(); i++)
            {
                std::wstring nodeName = configNode[i];
                m_net.FinalCriterionNodes().push_back(m_net.GetNodeFromName(nodeName));
            }

            configNode = configRoots("EvalNodes");
            for (size_t i=0; i<configNode.size(); i++)
            {
                std::wstring nodeName = configNode[i];
                m_net.EvaluationNodes().push_back(m_net.GetNodeFromName(nodeName));
            }

            configNode = configRoots("OutputNodes");
            for (size_t i=0; i<configNode.size(); i++)
            {
                std::wstring nodeName = configNode[i];
                m_net.OutputNodes().push_back(m_net.GetNodeFromName(nodeName));
            }

            m_net.PrintComputationTree(m_net.FinalCriterionNodes()[0],true,false);
        }

        virtual ComputationNetwork<ElemType>& BuildNetworkFromDescription()
        {
            if (m_net.GetTotalNumberOfNodes() < 1) //not built yet
            {
                ULONG randomSeed = 1;

                LoadNetworkFromConfig(m_networkConfig);
            }

            m_net.ResetEvalTimeStamp();
            return m_net;
        }


    private:
        ComputationNetwork<ElemType> m_net;
        std::wstring m_networkConfig;
        bool m_uniformInit;
        ElemType m_initValueScale;

        short m_deviceId;
    };
    template class FullNetworkBuilder<float>; 
    template class FullNetworkBuilder<double>;
}}}