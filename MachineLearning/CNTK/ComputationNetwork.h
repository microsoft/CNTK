#pragma warning (disable: 4702) // this function is flagged but unclear why
//
// <copyright file="ComputationNetwork.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

//The basic idea of this implementation is learned from Brian Guenter <bguenter@microsoft.com>

#include <map>
#include <string>
#include <stdexcept>
#include <list>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <iostream>
#include <regex>
#include <chrono>

#include "File.h"
#include "Matrix.h"
#include "commandArgUtil.h" // for nocase_compare

#include "ComputationNode.h"
#include "InputAndParamNodes.h"
#include "LinearAlgebraNodes.h"
#include "NonlinearityNodes.h"
#include "ConvolutionalNodes.h"
#include "RecurrentNodes.h"
#include "DecoderNode.h"
#include "TrainingCriterionNodes.h"
#include "CompositeComputationNodes.h"
#include "EvaluationCriterionNodes.h"
#include "BrainScriptObjects.h"
#include "BrainScriptEvaluator.h"   // TODO: move (I)ConfigRecord to BrainScriptConfig that only has the config-related stuff (ConfigValuePtr and IConfigRecord, possibly need to do the same for Array and Lambda)

#include "MatrixPool.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
class ComputationNetwork : public BS::Object, public BS::HasToString, public BS::IConfigRecord
{
protected:
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
    typedef std::pair<ComputationNodePtr, ComputationNodePtr> ComputationArc;

    typedef struct stRecurrentInfo
    {
        std::vector<ComputationNodePtr> m_recurrentNodes;
        std::vector<ComputationNodePtr> m_recurrentNodesForForward;
        ComputationNodePtr m_sourceNode;
        int m_loopId;
        bool m_completedGradient;
        bool m_completedEvaluate;
        bool m_loopClosed;
        bool m_isForwardLoop; 

        void Reset()
        {
            m_completedGradient = false;
            m_completedEvaluate = false;
            m_loopClosed = false;
        }

        // TODO: why is this not a copy constructor or assignment operator?
        void Copy(const stRecurrentInfo& src)
        {
            m_recurrentNodes = src.m_recurrentNodes;
            m_recurrentNodesForForward = src.m_recurrentNodesForForward;
            m_sourceNode = src.m_sourceNode;
            m_loopId = src.m_loopId;
            m_completedGradient = src.m_completedGradient;
            m_completedEvaluate = src.m_completedEvaluate;
            m_loopClosed = src.m_loopClosed;
        }
    } RecurrentInfo;

public:

    // TODO: sort methods into functional groups; some methods are at random places

    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    ComputationNetwork(DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
                    : m_deviceId(deviceId), m_SentenceBoundary(CPUDEVICE)
    {
        m_randomSeedOffset = 0;
        m_actMiniBSize = 0;
        if (m_deviceId == AUTOPLACEMATRIX)
        {
            m_deviceId = Matrix<ElemType>::GetBestGPUDeviceId();
        }
        m_nbrSlicesInEachRecurrentIteration = 1;
    }

    virtual ~ComputationNetwork()
    {
        ClearNet();
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    static bool IsSmaller(const ComputationNodePtr lhs, const ComputationNodePtr rhs)
    {
        return lhs->GetVisitedOrder() < rhs->GetVisitedOrder();
    }

    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    void ClearNet()
    {
        for (auto groupIter : GetAllNodeGroups())
            (groupIter)->clear();

        m_recurrentInfo.clear();

        m_built.clear();

        m_cacheEvalOrders.clear();
        m_cacheGradientCalcOrders.clear();

        m_inputs.clear();
        m_learnableParameters.clear();

        //for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        //{
        //    delete nodeIter->second;
        //}
        m_nameToNodeMap.clear();    // will also deref and likely deallocate all nodes we hold in here
    }

    // -----------------------------------------------------------------------
    // diagnostics
    // -----------------------------------------------------------------------

    //if node name is not found, dump all nodes
    //otherwise dump just that node
    void DumpNodeInfoToFile(const std::wstring & nodeName, const bool printValues, const std::wstring outputFile)
    {
        if (NodeNameExist(nodeName))
        {
            ValidateNetwork(true); //some internal values in the nodes are computed during validation

            File fstream(outputFile,
                         FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

            const ComputationNodePtr nodePtr = GetNodeFromName(nodeName);
            nodePtr->DumpNodeInfo(printValues, fstream);
        }
        else  //node name is not found, dump all nodes
        {
            fprintf(stderr, "Warning: node name %ls does not exist in the network. dumping all nodes.\n",
                    nodeName.c_str());
            DumpAllNodesToFile(printValues, outputFile);
        }
    }

    //dump all nodes in the network to file
    void DumpAllNodesToFile(const bool printValues,
                            const std::wstring outputFile,
                            const bool validateBeforeDump = true)
    {
        if (validateBeforeDump) 
        {
            //some internal values in the nodes are computed during validation
            ValidateNetwork();
        }

        File fstream(outputFile,
                     FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodePtr nodePtr = nodeIter->second;
            nodePtr->DumpNodeInfo(printValues, fstream);
        }
    }

    void DumpNodeInfoToFile(const vector<ComputationNodePtr>& nodes,
                            const bool printValues,
                            const std::wstring outputFile)
    {
        ValidateNetwork(); //some internal values in the nodes are computed during validation

        File fstream(outputFile,
                     FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            ComputationNodePtr nodePtr = *nodeIter;
            nodePtr->DumpNodeInfo(printValues, fstream);
        }
    }

private:

    // -----------------------------------------------------------------------
    // topological plot [erw]
    // TODO: Can this be a separate class? Can it be moved to a CPP?
    // -----------------------------------------------------------------------

    class DotGraphConfigure
    {
    public:
        wstring m_LearnableParameterStyle;
        wstring m_featuresStyle;
        wstring m_CriteriaStyle;
        wstring m_nodesReqMultiSeqHandlingStyle;
        wstring m_labelsStyle;
        wstring m_normalNodeStyle;
        wstring m_PrecomputingNodeStyle;
        wstring m_pastValueNodeStyle;
        wstring m_futureValueNodeStyle;

        DotGraphConfigure()
        {
            m_LearnableParameterStyle = L"node [ shape = box     , color = gray , style = \"filled, rounded\"  ]; ";
            m_featuresStyle = L"node [ shape = ellipse , color = red  , fillcolor = white ]; ";
            m_CriteriaStyle = L"node [ shape = doublecircle , color =  red , fillcolor = white  ]; ";
            m_nodesReqMultiSeqHandlingStyle = L"node [ shape = doublecircle , color =  brown , fillcolor = white  ]; ";
            m_normalNodeStyle = L"node [ shape = ellipse, color = blue, fillcolor = white, style = solid ]; ";
            m_PrecomputingNodeStyle = L"node [ shape = box    , color = black, style = \"dashed, filled\",  fillcolor= limegreen ] ;";
            m_labelsStyle = L"node [ shape = diamond, color = brown, style = bold ] ;  ";
            m_pastValueNodeStyle = L"node [ shape = box3d  , color = lightgray, style = \"filled\" , fillcolor = white ] ";
            m_futureValueNodeStyle = L"node [ shape = box3d  , color = red, style = \"filled\" , fillcolor = white ] ";
        }
    };

    wstring FormSpecialNodes(wstring style, std::vector<ComputationNodePtr>& specialNodes)
    {
        if (specialNodes.empty())
        {
            return L"";
        }

        wstring str = style;

        for (auto x : specialNodes)
        {
            str = str + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        }
        return str + L"; \n";
    }
public:

    void DescribeNetworkUsingDot(std::list<ComputationArc>& arcs,
                                 std::wstring outFile,
                                 DotGraphConfigure dotcfg = DotGraphConfigure())
    {
        File fstream(outFile,
                     FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);
        wstring line;

        // get precompute node
        std::vector<ComputationNodePtr> PreComputedNodes;
        std::vector<ComputationNodePtr> allnodes = GetAllNodes();
        for (auto n : allnodes)
        {
            if (n->RequirePreCompute())
            {
                PreComputedNodes.push_back(n);
            }
        }

        // get PastValue node
        std::vector<ComputationNodePtr> pastValueNodes;
        for (auto n : allnodes)
        {
            if (n->OperationName() == PastValueNode<ElemType>::TypeName() || 
                n->OperationName() == L"Delay")
            {
                pastValueNodes.push_back(n);
            }
        }

        // get FuturetValue node
        std::vector<ComputationNodePtr> futureValueNodes;
        for (auto n : allnodes)
        {
            if (n->OperationName() == FutureValueNode<ElemType>::TypeName())
            {
                futureValueNodes.push_back(n);
            }
        }
        // get learnableParameters
        std::vector<ComputationNodePtr> learnableParameters;
        for (auto n : allnodes)
        {
            if (n->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                learnableParameters.push_back(n);
            }
        }

        fstream << "strict digraph {\n";
        fstream << "rankdir = BT ;  \n";

        //////////////////////////////////////////////////////////////////////////
        //	special nodes
        //////////////////////////////////////////////////////////////////////////
        fstream << L"// special nodes \n";

        // learnable parameters:
        fstream << FormSpecialNodes(dotcfg.m_LearnableParameterStyle, learnableParameters);
        // features
        fstream << FormSpecialNodes(dotcfg.m_featuresStyle, m_features);
        // labels
        fstream << FormSpecialNodes(dotcfg.m_labelsStyle, m_labels);
        // critera
        fstream << FormSpecialNodes(dotcfg.m_CriteriaStyle, m_finalCriteria);
        // nodes that requires multi sequence handling 
        fstream << FormSpecialNodes(dotcfg.m_nodesReqMultiSeqHandlingStyle, m_nodesReqMultiSeqHandling);            
        // pre-compute nodes
        fstream << FormSpecialNodes(dotcfg.m_PrecomputingNodeStyle, PreComputedNodes);
        // PastValue nodes
        fstream << FormSpecialNodes(dotcfg.m_pastValueNodeStyle, pastValueNodes);
        // FutureValue nodes
        fstream << FormSpecialNodes(dotcfg.m_futureValueNodeStyle, futureValueNodes);
        // normal nodes
        fstream << dotcfg.m_normalNodeStyle << L"\n";

        //////////////////////////////////////////////////////////////////////////
        //	add labels for each node
        //////////////////////////////////////////////////////////////////////////
        fstream << L"\n// add labels and operation name\n";
        for (auto x : allnodes)
        {
            line.clear();
            size_t nrows = x->FunctionValues().GetNumRows();
            size_t ncols = x->FunctionValues().GetNumCols();
            line = msra::strfun::wstrprintf(L" \"%ls\" [ label = \"%ls [%d,%d]\\n%ls\" ] ;\n",
                                            x->GetName().c_str(), x->GetName().c_str(), nrows, ncols,
                                            x->OperationName().c_str());
            fstream << line;
        }

        //////////////////////////////////////////////////////////////////////////
        //	sub-graph
        //////////////////////////////////////////////////////////////////////////
        // subgraph source
        fstream << L"subgraph {\n";
        fstream << L"\t\t rank=source ; ";
        line.clear();
        for (auto x : m_features)
        {
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        }
        fstream << line << L"\n}\n";

        // subgraph eval/output/criteria
        fstream << L"subgraph {\n";
        fstream << L"\t\t rank=sink ; ";
        line.clear();
        for (auto x : m_finalCriteria)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        for (auto x : m_nodesReqMultiSeqHandling)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        for (auto x : m_outputNodes)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        for (auto x : m_pairNodes)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        for (auto x : m_evalNodes)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());

        fstream << line << L"\n}\n";

        //////////////////////////////////////////////////////////////////////////
        //	specify arc connections
        //////////////////////////////////////////////////////////////////////////
        for (auto x = arcs.begin(); x != arcs.end(); x++)
        {
            ComputationNodePtr src = (*x).first;
            ComputationNodePtr des = (*x).second;

            std::wstring srcname = src->GetName();
            std::wstring desname = des->GetName();

            if (des->OperationName() == PastValueNode<ElemType>::TypeName() || des->OperationName() == L"Delay")
            {
                // special treament for arc with PastValue node as the children
                // create a dummy node
                ComputationNodePtr pastValueNode = des;
                wstring dummyName = des->GetName() + L".dummy";
                wstring out = msra::strfun::wstrprintf(L"node [ shape = box3d  , color = lightgray, style = \"filled\" , label = \"%ls\" ] ; \"%ls\"\n",
                                                       (pastValueNode->GetName() + L"\\n(PastValue)").c_str(),
                                                       dummyName.c_str());
                line = out;
                line += msra::strfun::wstrprintf(L"\"%ls\" -> \"%ls\" ; \n", dummyName.c_str(), srcname.c_str());
            }
            else if (des->OperationName() == FutureValueNode<ElemType>::TypeName())
            {
                // special treament for arc with FutureValue node as the children
                // create a dummy node
                ComputationNodePtr futureValueNode = des;
                wstring dummyName = des->GetName() + L".dummy";
                wstring out = msra::strfun::wstrprintf(L"node [ shape = box3d  , color = red, style = \"filled\" , label = \"%ls\" ] ; \"%ls\"\n",
                    (futureValueNode->GetName() + L"\\n(FutureValue)").c_str(),
                    dummyName.c_str());
                line = out;
                line += msra::strfun::wstrprintf(L"\"%ls\" -> \"%ls\" ; \n", dummyName.c_str(), srcname.c_str());
            }
            else
            {
                line = msra::strfun::wstrprintf(L"\"%ls\" -> \"%ls\" ; \n", desname.c_str(), srcname.c_str());
            }

            fstream << line;
        }
        fstream << L"\n}\n";

    }
    void PlotNetworkTopology(const std::wstring outputFile) //  [1/13/2015 erw] plot network topology using dot language
    {
        BuildAndValidateNetwork(m_evalNodes[0]);

        //////////////////////////////////////////////////////////////////////////
        //	step 1.		get all the arcs in the network
        //////////////////////////////////////////////////////////////////////////
        std::unordered_set<ComputationNodePtr> visited;
        std::list<ComputationArc> arcs;

        for (auto groupIter : GetAllNodeGroups())
        {
            // note: this will also loop over m_features and m_labels, which will do nothing since they have no inputs
            // TODO: test whether that is true
            const auto & group = *groupIter;
            for (size_t i = 0; i < group.size(); i++)
                group[i]->EnumerateArcs(visited, arcs);
        }

        //////////////////////////////////////////////////////////////////////////
        //	step 2.		output dot description
        //////////////////////////////////////////////////////////////////////////
        DescribeNetworkUsingDot(arcs, outputFile);
    }

    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    void SetDeviceID(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
    {
        m_deviceId = deviceId;
        if (m_deviceId == AUTOPLACEMATRIX)
            m_deviceId = Matrix<ElemType>::GetBestGPUDeviceId();
    }

    DEVICEID_TYPE GetDeviceID() { return m_deviceId; }

    unsigned long GetRandomSeedOffset() { return m_randomSeedOffset; }
    void SetRandomSeedOffset(unsigned long value) { m_randomSeedOffset = value; }

    // -----------------------------------------------------------------------
    // serialization
    // -----------------------------------------------------------------------

    void SaveToFile(const std::wstring& fileName, const FileOptions fileFormat = FileOptions::fileOptionsBinary) const
    {
       // Saving into temporary file and then renaming it to the requested fileName
       // This is a standard trick to avoid havign corrupted model files if process dies during writing
       wstring tmpFileName = fileName + L".tmp";
       SaveToFileImpl(tmpFileName, fileFormat);
       renameOrDie(tmpFileName, fileName);
    }

private:
    void SaveToFileImpl(const std::wstring& fileName, const FileOptions fileFormat) const
    {
        File fstream(fileName, fileFormat | FileOptions::fileOptionsWrite);
        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BCN");

        //model version
        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BVersion");
        fstream << (size_t) CURRENT_CNTK_MODEL_VERSION;
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EVersion");

        fstream << (size_t) m_nameToNodeMap.size();

        //put all node info first
        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BNodeList");
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodePtr nodePtr = nodeIter->second;
            nodePtr->SaveToFile(fstream);
        }

        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ENodeList");

        //put relationship
        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BRelation");
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodePtr nodePtr = nodeIter->second;
            fstream << nodePtr->NodeName() << nodePtr->ChildrenSize();
            for (size_t i = 0; i < nodePtr->ChildrenSize(); i++)
            {
                if (nodePtr->Inputs(i) == nullptr)
                    fprintf(stderr, "Warning: node %ls 's child is null, please check your ndl/mel file.\n", nodePtr->NodeName().c_str());
                else
                    fstream << nodePtr->Inputs(i)->NodeName();
                }
            }
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ERelation");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BRootNodes");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BFeatureNodes");
        fstream << m_features.size();
        for (size_t i = 0; i < m_features.size(); i++)
            fstream << m_features[i]->NodeName();
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EFeatureNodes");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BLabelNodes");
        fstream << m_labels.size();
        for (size_t i = 0; i < m_labels.size(); i++)
            fstream << m_labels[i]->NodeName();
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ELabelNodes");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BCriteriaNodes");
        fstream << m_finalCriteria.size();
        for (size_t i = 0; i < m_finalCriteria.size(); i++)
            fstream << m_finalCriteria[i]->NodeName();
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ECriteriaNodes");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BNodesReqMultiSeqHandling");
        fstream << m_nodesReqMultiSeqHandling.size();
        for (size_t i = 0; i<m_nodesReqMultiSeqHandling.size(); i++)
            fstream << m_nodesReqMultiSeqHandling[i]->NodeName();
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ENodesReqMultiSeqHandling");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BEvalNodes");
        fstream << m_evalNodes.size();
        for (size_t i = 0; i < m_evalNodes.size(); i++)
            fstream << m_evalNodes[i]->NodeName();
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EEvalNodes");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BOutputNodes");
        fstream << m_outputNodes.size();
        for (size_t i = 0; i < m_outputNodes.size(); i++)
        {
            fstream << m_outputNodes[i]->NodeName();
        }
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EOutputNodes");

        if (m_pairNodes.size() > 0)
        {
            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BPairNodes");

            fstream << m_pairNodes.size();
            for (size_t i = 0; i < m_pairNodes.size(); i++)
                fstream << m_pairNodes[i]->NodeName();
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EPairNodes");
        }

        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ERootNodes");

        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ECN");
       
        fstream.Flush();
    }

public:
    void LoadPersistableParametersFromFile(const std::wstring& fileName, const bool requireValidation = true,
                                           const FileOptions fileFormat = FileOptions::fileOptionsBinary)
    {
        File fstream(fileName, fileFormat | FileOptions::fileOptionsRead);

        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BCN");

        //model version
        size_t modelVersion = CNTK_MODEL_VERSION_1; //if version info is not there it is version 1
        if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BVersion"))
        {
            fstream >> modelVersion;
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EVersion");
        }

        size_t numNodes;
        fstream >> numNodes;

        //get all node info first
        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BNodeList");
        for (size_t i = 0; i < numNodes; i++)
        {
            std::wstring opName, nodeName;
            fstream >> opName >> nodeName;
            ComputationNodePtr nodePtr = GetNodeFromName(nodeName);
            // TODO: don't we have a load constructor? Then when to call which? Document the calling sequence
            nodePtr->LoadFromFile(fstream, modelVersion);
        }

        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ENodeList");

        size_t actualMBSize = GetActualMBSize();
        SetActualMiniBatchSize(actualMBSize);

        if (requireValidation)
        {
            ValidateNetwork();
        }
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    size_t GetActualMBSize()
    {
        size_t actualMBSize = 0;

        const auto & FeatureNodes = this->FeatureNodes();   // TODO: a getter; should be called GetFeatureNodes()
        for (auto nodeIter = FeatureNodes.begin(); nodeIter != FeatureNodes.end(); nodeIter++)
        {
            actualMBSize = max(actualMBSize, ((*nodeIter)->FunctionValues()).GetNumCols());
        }

        return actualMBSize;
    }

    // -----------------------------------------------------------------------
    // serialization
    // -----------------------------------------------------------------------

    virtual void LoadFromFile(const std::wstring& fileName, const FileOptions fileFormat = FileOptions::fileOptionsBinary,
                              const bool bAllowNoCriterionNode = false, ComputationNetwork<ElemType>* anotherNetwork = nullptr)
    {
        ClearNet();

        File fstream(fileName, fileFormat | FileOptions::fileOptionsRead);

        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BCN");

        //model version
        size_t modelVersion = CNTK_MODEL_VERSION_1; //if version info is not there it is version 1
        if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BVersion"))
        {
            fstream >> modelVersion;
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EVersion");
        }

        size_t numNodes;
        fstream >> numNodes;

        //get all node info first
        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BNodeList");
        for (size_t i = 0; i < numNodes; i++)
        {
            std::wstring opName, nodeName;
            fstream >> opName >> nodeName;

            CreateNodeFromFile(opName, nodeName, fstream, modelVersion);
        }
        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ENodeList");

        //put relationship
        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BRelation");
        for (size_t i = 0; i < numNodes; i++)
        {
            std::wstring nodeName;
            size_t numChildren;
            fstream >> nodeName >> numChildren;
            if (numChildren > 0)
            {
                std::vector<std::wstring> childrenNames;
                childrenNames.resize(numChildren);
                for (size_t j = 0; j < numChildren; j++)
                {
                    fstream >> childrenNames[j];
                }

                ComputationNodePtr nodePtr = GetNodeFromName(nodeName);
                std::vector<ComputationNodePtr> childrenNodes;
                childrenNodes.resize(numChildren);
                for (int j = 0; j < numChildren; j++)
                                childrenNodes[j] = GetNodeFromName(childrenNames[j], anotherNetwork);

                if (nodePtr->OperationName() == RowStackNode<ElemType>::TypeName()) {
                    //allow for variable input nodes
                    nodePtr->AttachInputs(childrenNodes);
                }
                else
                {
                    //fixed input nodes
                    switch (numChildren)
                    {
                        case 1:
                            nodePtr->AttachInputs(childrenNodes[0]);
                            break;

                        case 2:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1]);
                            break;
                        case 3:
                            nodePtr->AttachInputs(childrenNodes[0],childrenNodes[1],
                                                  childrenNodes[2]);
                            break;
                        case 4:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1],
                                                  childrenNodes[2], childrenNodes[3]);
                            break;
                        case 5:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1], childrenNodes[2],
                                                  childrenNodes[3], childrenNodes[4]);
                            break;
                        case 6:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1], childrenNodes[2],
                                                  childrenNodes[3], childrenNodes[4], childrenNodes[5]);
                            break;

                        default:
                            LogicError("Invalid number of children.");
                    }
                }
            }
        }

        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ERelation");

        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BRootNodes");
        {
            std::wstring nodeName;
            size_t num;

            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BFeatureNodes");
            fstream >> num;

            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                m_features.push_back(GetNodeFromName(nodeName));
            }

            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EFeatureNodes");

            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BLabelNodes");
            fstream >> num;
            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                m_labels.push_back(GetNodeFromName(nodeName));
            }

            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ELabelNodes");

            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BCriteriaNodes");
            fstream >> num;
            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                m_finalCriteria.push_back(GetNodeFromName(nodeName));
            }

            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ECriteriaNodes");

            if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BNodesReqMultiSeqHandling"))
            {
                fstream >> num;
                for (size_t i = 0; i<num; i++)
                {
                    fstream >> nodeName;
                    m_nodesReqMultiSeqHandling.push_back(GetNodeFromName(nodeName));
                }
                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ENodesReqMultiSeqHandling");
            }

            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BEvalNodes");
            fstream >> num;
            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                m_evalNodes.push_back(GetNodeFromName(nodeName));
            }
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EEvalNodes");

            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BOutputNodes");
            fstream >> num;
            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                m_outputNodes.push_back(GetNodeFromName(nodeName));
            }
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EOutputNodes");

            if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BPairNodes"))
            {
                fstream >> num;
                for (size_t i = 0; i < num; i++)
                {
                    fstream >> nodeName;
                    m_pairNodes.push_back(GetNodeFromName(nodeName));
                }
                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EPairNodes");
            }
        }

        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ERootNodes");

        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ECN");

        //some internal values in the nodes are computed during validation
        ValidateNetwork(false, bAllowNoCriterionNode);
    }

#pragma region Network Modification

    void SetLeanableNodesBelowNeedGradient(const bool needGradient, const ComputationNodePtr rootNode = nullptr)
    {
        //find nodes from all available nodes
        if (rootNode == nullptr)
        {
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodePtr node = nodeIter->second;
                if (node->OperationName() == LearnableParameter<ElemType>::TypeName())
                {
                    node->NeedGradient() = needGradient;
                }
            }
        }
        else
        {
            //for calculating a specific node
            std::list<ComputationNodePtr>& nodes = GetEvalOrder(rootNode);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodePtr node = (*nodeIter);
                if (node->OperationName() == LearnableParameter<ElemType>::TypeName())
                {
                    node->NeedGradient() = needGradient;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // TODO: describe what this function does
    //this is a temp solution since some nodes such as plus can be just aggregate of two scalar values 
    //in which case the packing info is not available (and not meaningful) for them
    size_t GetNumSamplesWithLabel(const size_t numAllSamples)
    {
        if (!m_SentenceBoundary.IsEmpty() &&
            !m_minibatchPackingFlag.size() == 0)
        {
            size_t numTimeSteps = m_SentenceBoundary.GetNumCols();
            size_t numSequences = m_SentenceBoundary.GetNumRows();

            if (m_minibatchPackingFlag.size() != numTimeSteps)
                LogicError("GetNumSamplesWithLabel(): m_minibatchPackingFlag should have one element for each timestep of all streams.Check feature reader. ");

            size_t numSamplesWithoutLabel = 0;

            for (size_t j = 0; j < numTimeSteps; j++)
            {
                if (m_minibatchPackingFlag[j] & MinibatchPackingFlag::NoLabel)
                {
                    for (int i = 0; i < numSequences; i++)
                    {
                        if ((int)(m_SentenceBoundary(i, j)) & NO_LABEL)
                            numSamplesWithoutLabel++;
                        }
                    }
                }

            return numTimeSteps*numSequences - numSamplesWithoutLabel;
        }
        else
            return numAllSamples;
        }

    // -----------------------------------------------------------------------
    // serialization
    // -----------------------------------------------------------------------

    // Read a matrix stored in text format from 'filePath' (whitespace-separated columns, newline-separated rows),
    // and return a flat array containing the contents of this file in column-major format.
    // filePath: path to file containing matrix in text format.
    // numRows/numCols: after this function is called, these parameters contain the number of rows/columns in the matrix.
    // returns: a flat array containing the contents of this file in column-major format
    // NOTE: caller is responsible for deleting the returned buffer once it is finished using it.
    // TODO: change to return a std::vector<ElemType>; solves the ownership issue
    // TODO: move this elsewhere, this is a general utility function that does not belong into the ComputationNetwork class
    static ElemType* LoadArrayFromTextFile(const std::string filePath, size_t& numRows, size_t& numCols)
    {
        size_t r = 0;
        size_t numColsInFirstRow = 0;

        // NOTE: Not using the Microsoft.MSR.CNTK.File API here because it
        // uses a buffer of fixed size, which doesn't allow very long rows.
        // See fileutil.cpp fgetline method (std::string fgetline (FILE * f) { fixed_vector<char> buf (1000000); ... })
        std::ifstream myfile(filePath);

        // load matrix into vector of vectors (since we don't know the size in advance).
        std::vector<std::vector<ElemType>> elements;
        if (myfile.is_open())
        {
            std::string line;
            while (std::getline(myfile, line))
            {
                // Break on empty line.  This allows there to be an empty line at the end of the file.
                if (line == "")
                    break;

                istringstream iss(line);
                ElemType element;
                int numElementsInRow = 0;
                elements.push_back(std::vector<ElemType>());
                while (iss >> element)
                {
                    elements[r].push_back(element);
                    numElementsInRow++;
                }

                if (r == 0)
                    numColsInFirstRow = numElementsInRow;
                else if (numElementsInRow != numColsInFirstRow)
                    RuntimeError("The rows in the provided file do not all have the same number of columns: " + filePath);

                r++;
            }
            myfile.close();
        }
        else
            RuntimeError("Unable to open file");

        numRows = r;
        numCols = numColsInFirstRow;

        ElemType* pArray = new ElemType[numRows * numCols];

        // Perform transpose when copying elements from vectors to ElemType[],
        // in order to store in column-major format.
        for (int i = 0; i < numCols; i++)
        {
            for (int j = 0; j < numRows; j++)
                pArray[i * numRows + j] = elements[j][i];
            }

        return pArray;
    }

    // TODO: why is this here? Move to LearnableParameter class?
    static void InitLearnableParametersFromFile(const ComputationNodePtr node,
                                         const std::wstring & initFromFilePath,
                                         DEVICEID_TYPE deviceId)    // TODO: why not just use node->m_deviceId?
    {
        size_t numRows = 0;
        size_t numCols = 0;
        ElemType *pArray = LoadArrayFromTextFile(msra::strfun::utf8(initFromFilePath), numRows, numCols); // TODO: change pathname to wstring
        node->FunctionValues().SetValue(numRows, numCols, pArray, matrixFlagNormal, deviceId);
        delete[] pArray;    // TODO: use std::vector to avoid mem leak on error
    }
    void InitLearnableParametersFromFile(const ComputationNodePtr node, const std::string & initFromFilePath)   // TODO: remove this method or change pathname to wstring
    {
        InitLearnableParametersFromFile(node, msra::strfun::utf16(initFromFilePath), this->GetDeviceID());
    }

    // -----------------------------------------------------------------------
    // node construction
    // -----------------------------------------------------------------------

    // TODO: move this into LearnableParameter directly; no value to keep it out
    static void InitLearnableParameters(const ComputationNodePtr node,
                                        const bool uniformInit,
                                        const unsigned long randomSeed,
                                        const ElemType initValueScale,
                                        unsigned long randomSeedOffset)
    {
        size_t inputSize = node->FunctionValues().GetNumCols();

        // the random seed offset is set via the "randomSeedOffset" parameter in config
        if (uniformInit)
        {
            ElemType randRange = 0.05f * initValueScale; //initValueScale/sqrt(inputSize);
            node->FunctionValues().SetUniformRandomValue(-randRange, randRange, randomSeedOffset + randomSeed);
        }
        else
        {
            ElemType randInitstd = 0.2f * initValueScale / sqrt(ElemType(inputSize));
            node->FunctionValues().SetGaussianRandomValue(0, randInitstd, randomSeedOffset + randomSeed);
        }
    }
    // non-static version needed because it access m_randomSeedOffset
    void InitLearnableParameters(const ComputationNodePtr node,
        const bool uniformInit,
        const unsigned long randomSeed,
        const ElemType initValueScale)
    {
        return InitLearnableParameters(node, uniformInit, randomSeed, initValueScale, GetRandomSeedOffset());
    }

    // -----------------------------------------------------------------------
    // network editing
    // -----------------------------------------------------------------------

    void DeleteNode(const std::wstring & nodeName)
    {
        //so that deleted node will not be referenced
        ClearCaches();

        ComputationNodePtr nodeToDelete = GetNodeFromName(nodeName);

        //first delete links, if this node is involved, the whole connection will be removed
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodePtr node = nodeIter->second;
            for (size_t i = 0; i < node->ChildrenSize(); i++)
            {
                ComputationNodePtr child = node->Inputs(i);

                //nodeToDelete is a child
                if (child == nodeToDelete)
                {
                    // this used to call DetatchInputs(), but it's better for MEL to retain other inputs
                    node->SetInput(i, NULL);
                    break;
                }
            }
        }

        //nodeToDelete is a parent
        nodeToDelete->DetachInputs();       // deref all its inputs; if we don't do that, we might end up with a mem leak due to a circular reference

        // unlink from all node-group sets
        for (auto groupIter : GetAllNodeGroups())
        {
            auto search = std::find(groupIter->begin(), groupIter->end(), nodeToDelete);
            if (search != groupIter->end())
                groupIter->erase(search);
        }

        // ? how to deal with m_recurrentInfo, when we delete a node.

        //delete the node itself
        m_nameToNodeMap.erase(nodeName);    // this will deref the node and possibly deallocate it
    }

    // RenameNode - Rename a node to another name
    // nodeNameOrig - original node name
    // nodeNameNew - new node name
    void RenameNode(const std::wstring& nodeNameOrig, const std::wstring& nodeNameNew)
    {
        //so that renamed node will not be referenced
        ClearCaches();

        ComputationNodePtr nodeToRename = GetNodeFromName(nodeNameOrig);

        auto iter = m_nameToNodeMap.find(nodeNameNew);
        if (iter != m_nameToNodeMap.end()) //found
            RuntimeError("RenameNode: Target name already exists.");

        //rename the node and update the mapping table
        nodeToRename->NodeName() = nodeNameNew;
        m_nameToNodeMap.erase(nodeNameOrig);
        m_nameToNodeMap[nodeNameNew] = nodeToRename;
    }

    // -----------------------------------------------------------------------
    // node construction
    // -----------------------------------------------------------------------

    // TODO: comment what this function does. Seems to either initialize LearnableParameters or precompute nodes.
    ComputationNodePtr SetNodeValue(const std::wstring & nodeName, const ElemType value)
    {
        ComputationNodePtr pNode = GetNodeFromName(nodeName);

        if (pNode->OperationName() == LearnableParameter<ElemType>::TypeName())
            pNode->FunctionValues().SetValue(value);
        else if (pNode->RequirePreCompute())
        {
            auto preComputedNode = static_pointer_cast<PreComputedNode<ElemType>>(pNode);
            pNode->FunctionValues().SetValue(value);    // TODO: comment: is this an expensive operation?
            preComputedNode->MarkComputed(true);
        }
        else
            LogicError("Only values of learnable parameters and precomputed nodes can be set.");

        return pNode;
    }

    // -----------------------------------------------------------------------
    // network editing
    // -----------------------------------------------------------------------

    ComputationNodePtr CopyNode(const ComputationNetwork<ElemType> & fromNet,
                                const std::wstring fromName,
                                std::wstring toName = L"",
                                const CopyNodeFlags flags = CopyNodeFlags::copyNodeAll)
    {
        if (toName == L"") {
            toName = fromName;
        }

        ComputationNodePtr pFromNode = fromNet.GetNodeFromName(fromName);
        ComputationNodePtr pToNode;

        // don't allow cross network child copy unless caller explicity handles children fixup
        if ((flags & CopyNodeFlags::copyNodeChildren) &&
            this != &fromNet && !(flags & CopyNodeFlags::copyNodeChildrenCrossNetwork))
        {
            LogicError("CopyNode: Copying node children across network is invalid.");
        }

        if (!NodeNameExist(toName))
        {
            pToNode = pFromNode->Duplicate(toName, flags);
            AddNodeToNet(pToNode);
        }
        else
        {
            //node already exists

            pToNode = GetNodeFromName(toName);

            //same node. no copy needed
            if (pFromNode == pToNode)
                LogicError("CopyNode: You are copying the node to the same network with same node name.");
            else
                pFromNode->CopyTo(pToNode, toName, flags);  // blast it over the existing node
            }
        return pToNode;
    }

    //only copy a complete independent tree
    //when node name exists
    void CopySubTree(const ComputationNetwork<ElemType> & fromNet,
                     const std::wstring fromName, std::wstring toNamePrefix = L"",
                     const CopyNodeFlags flags = copyNodeAll)
    {
        if (!(flags & CopyNodeFlags::copyNodeValue))
            LogicError("CopySubTree: you cannot copy a tree without copying the node values.");

        ComputationNodePtr fromRoot = fromNet.GetNodeFromName(fromName);

        std::list<ComputationNodePtr>& nodes = GetEvalOrder(fromRoot);
        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            ComputationNodePtr fromNode = *nodeIter;
            wstring fromNodeName = fromNode->NodeName();
            wstring toNodeName = toNamePrefix + fromNodeName;

            ComputationNodePtr toNode = CopyNode(fromNet, fromNodeName,
                                                 toNodeName,
                                                 CopyNodeFlags::copyNodeValue);

            if (flags & CopyNodeFlags::copyNodeChildren)
            {
                //copy the children structure but use the new nodes generated
                for (int i = 0; i < fromNode->ChildrenSize(); i++)
                    toNode->SetInput(i, GetNodeFromName(toNamePrefix + fromNode->Inputs(i)->NodeName()));
                }
            }
        }

    //you can only copy inputs from nodes in the same network
    void CopyInputs(const std::wstring fromName, std::wstring toName)
    {
        CopyNode(*this, fromName, toName, CopyNodeFlags::copyNodeChildren);
    }

#pragma endregion Network Modification

    // -----------------------------------------------------------------------
    // node creation
    // -----------------------------------------------------------------------

    // TODO: There is quite a bit of redundancy here
    //  - create/load by name
    //  - create by calling constructor directly
    //  - create node by type--one function per node; one could just use the constructor
    //  - create node and add to network--users could just add the node by themselves
    // We should
    //  - move node creation to a separate class, e.g. NodeFactory
    //    One goal would be that ComputationNetwork.h becomes agnostic of node types as much as possible, and does not have to pull in all node headers
    //  - choose one of the methods above (probably we need the by-name method separately, but tucked away in a CPP please)

    // create a new node of a type given as a string, with var args so that this can be used at multiple places
    // This function only creates nodes that accept (m_deviceId, nodeName).
    // TODO: Is this ever called with additional _Args? If not, simplify
    template<class... _Types>
    static ComputationNodePtr NewStandardNode(const std::wstring & nodeType, DEVICEID_TYPE deviceId, const wstring & name, _Types&&... _Args)
    {
        // please keep this table sorted
        if (nodeType == CRFNode<ElemType>::TypeName())	return New<CRFNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == ClassBasedCrossEntropyWithSoftmaxNode<ElemType>::TypeName()) return New<ClassBasedCrossEntropyWithSoftmaxNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == ColumnElementTimesNode<ElemType>::TypeName())  return New<ColumnElementTimesNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == CosDistanceNode<ElemType>::TypeName())	    return New<CosDistanceNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == CosDistanceWithNegativeSamplesNode<ElemType>::TypeName()) return New<CosDistanceWithNegativeSamplesNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == CosineNode<ElemType>::TypeName())	            return New<CosineNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == CrossEntropyNode<ElemType>::TypeName())	    return New<CrossEntropyNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == CrossEntropyWithSoftmaxNode<ElemType>::TypeName())	return New<CrossEntropyWithSoftmaxNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == DiagTimesNode<ElemType>::TypeName())	    return New<DiagTimesNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == DropoutNode<ElemType>::TypeName())	            return New<DropoutNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == DummyCriterionNode<ElemType>::TypeName())	    return New<DummyCriterionNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == ElementTimesNode<ElemType>::TypeName())	    return New<ElementTimesNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == ErrorPredictionNode<ElemType>::TypeName())	    return New<ErrorPredictionNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == ExpNode<ElemType>::TypeName())	            return New<ExpNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == FutureValueNode<ElemType>::TypeName())	    return New<FutureValueNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == GMMLogLikelihoodNode<ElemType>::TypeName())    return New<GMMLogLikelihoodNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == InvStdDevNode<ElemType>::TypeName())	    return New<InvStdDevNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == KhatriRaoProductNode<ElemType>::TypeName())    return New<KhatriRaoProductNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == LSTMNode<ElemType>::TypeName())	            return New<LSTMNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == LogNode<ElemType>::TypeName())	            return New<LogNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == LogSoftmaxNode<ElemType>::TypeName())	    return New<LogSoftmaxNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == LookupTableNode<ElemType>::TypeName())	    return New<LookupTableNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == MatrixL1RegNode<ElemType>::TypeName())	    return New<MatrixL1RegNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == MatrixL2RegNode<ElemType>::TypeName())	    return New<MatrixL2RegNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == MeanNode<ElemType>::TypeName())	            return New<MeanNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == MinusNode<ElemType>::TypeName())	            return New<MinusNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == NegateNode<ElemType>::TypeName())	            return New<NegateNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == NoiseContrastiveEstimationNode<ElemType>::TypeName()) return New<NoiseContrastiveEstimationNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == PairNetworkNode<ElemType>::TypeName())	    return New<PairNetworkNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == ParallelNode<ElemType>::TypeName())	    return New<ParallelNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == PastValueNode<ElemType>::TypeName() || nodeType == L"Delay") return New<PastValueNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == PerDimMeanVarDeNormalizationNode<ElemType>::TypeName() || nodeType == L"PerDimMeanVarDeNormalizationNode")	return New<PerDimMeanVarDeNormalizationNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == PerDimMeanVarNormalizationNode<ElemType>::TypeName() || nodeType == L"PerDimMeanVarNormalizationNode")	return New<PerDimMeanVarNormalizationNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == PlusNode<ElemType>::TypeName())	            return New<PlusNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == RectifiedLinearNode<ElemType>::TypeName())	    return New<RectifiedLinearNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == ReshapeNode<ElemType>::TypeName())	            return New<ReshapeNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == RowElementTimesNode<ElemType>::TypeName())	    return New<RowElementTimesNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == RowRepeatNode<ElemType>::TypeName())	    return New<RowRepeatNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == RowSliceNode<ElemType>::TypeName())	    return New<RowSliceNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == RowStackNode<ElemType>::TypeName())	    return New<RowStackNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == ScaleNode<ElemType>::TypeName())	            return New<ScaleNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == SequenceDecoderNode<ElemType>::TypeName())	    return New<SequenceDecoderNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == SigmoidNode<ElemType>::TypeName())	            return New<SigmoidNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == SoftmaxNode<ElemType>::TypeName())	            return New<SoftmaxNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == SquareErrorNode<ElemType>::TypeName())	    return New<SquareErrorNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == StrideTimesNode<ElemType>::TypeName())	    return New<StrideTimesNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == SumColumnElementsNode<ElemType>::TypeName())   return New<SumColumnElementsNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == SumElementsNode<ElemType>::TypeName())	    return New<SumElementsNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == TanhNode<ElemType>::TypeName())	            return New<TanhNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == TimeReverseNode<ElemType>::TypeName())	    return New<TimeReverseNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == TimesNode<ElemType>::TypeName())	            return New<TimesNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == TransposeNode<ElemType>::TypeName())	    return New<TransposeNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == TransposeTimesNode<ElemType>::TypeName())	    return New<TransposeTimesNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else return nullptr;
    }
    // create a new node of a type given as a string, with var args so that this can be used at multiple places
    // This function is used for loading, while the above is used for creating standard-type networks.
    template<class... _Types>
    static ComputationNodePtr NewNode(const std::wstring & nodeType, DEVICEID_TYPE deviceId, const wstring & name, _Types&&... _Args)
    {
        // TODO: Is this ever called with additional _Args? If not, simplify
        // try first those that accept the standard two constructor arguments
        auto newNode = NewStandardNode(nodeType, deviceId, name, forward<_Types>(_Args)...);
        if (newNode) return newNode;
        // check more types
        else if (nodeType == AveragePoolingNode<ElemType>::TypeName())	     return New<AveragePoolingNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == ConvolutionNode<ElemType>::TypeName())	     return New<ConvolutionNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == InputValue<ElemType>::SparseTypeName())	     return New<InputValue<ElemType>>(deviceId, name, forward<_Types>(_Args)..., true);
        else if (nodeType == InputValue<ElemType>::TypeName())	             return New<InputValue<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == LearnableParameter<ElemType>::TypeName())	     return New<LearnableParameter<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == MaxPoolingNode<ElemType>::TypeName())	     return New<MaxPoolingNode<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else if (nodeType == SparseLearnableParameter<ElemType>::TypeName()) return New<SparseLearnableParameter<ElemType>>(deviceId, name, forward<_Types>(_Args)...);
        else return nullptr;
    }

    // -----------------------------------------------------------------------
    // serialization
    // -----------------------------------------------------------------------

    ComputationNodePtr CreateNodeFromFile(const std::wstring& nodeType,
                                          const std::wstring & nodeName,
                                          File& fstream,
                                          size_t modelVersion)
        {
        auto newNode = NewNode(nodeType, m_deviceId, nodeName);
        if (!newNode)
        {
            fprintf(stderr, "Unknown ComputationNode type %ls (node name %ls)\n", nodeType.c_str(), nodeName.c_str());
            InvalidArgument("Invalid node type.");
        }
        newNode->LoadFromFile(fstream, modelVersion);
        return AddNodeToNet(newNode);
        }

    // -----------------------------------------------------------------------
    // node creation
    // -----------------------------------------------------------------------

    // The following functions create nodes and add them to the net, but don't attach inputs (some don't have inputs).
    // There are special versions for nodes with custom constructors, and a catch-all, CreateComputationNode(), for all others.
    // TODO: Do we really need these? Folks who want to use C++ can instead say net->AddNodeToNet(New<>(...)), which is not that different.
    // TODO: separate into nodes that have inputs and those that duplicate functions with input adding except just not adding inputs. Clear?

    ComputationNodePtr CreateLearnableParameter(const std::wstring & paramName, const size_t rows, const size_t cols)
    {
        return AddNodeToNet(New<LearnableParameter<ElemType>>(m_deviceId, paramName, rows, cols));
    }

    //sparse matrix size is optionally specified
    ComputationNodePtr CreateSparseLearnableParameter(const std::wstring & paramName, const size_t rows, const size_t cols, const size_t size = 0)
    {
        return AddNodeToNet(New<SparseLearnableParameter<ElemType>>(m_deviceId, paramName, rows, cols, size));
    }

    ComputationNodePtr CreateInputNode(const std::wstring & inputName, const size_t rows, const size_t cols)
    {
        return AddNodeToNet(New<InputValue<ElemType>>(m_deviceId, inputName, rows, cols));
    }

    ComputationNodePtr CreateSparseInputNode(const std::wstring & inputName, const size_t rows, const size_t cols)
    {
        return AddNodeToNet(New<InputValue<ElemType>>(m_deviceId, inputName, rows, cols, true));
    }

    ComputationNodePtr CreateInputNode(const std::wstring & inputName,
                                       const size_t imageWidth,
                                       const size_t imageHeight,
                                       const size_t imageChannels,
                                       const size_t numImages)
    {
        return AddNodeToNet(New<InputValue<ElemType>>(m_deviceId, inputName, imageWidth, imageHeight, imageChannels, numImages));
    }

    ComputationNodePtr CreateSparseInputNode(const std::wstring & inputName,
                                             const size_t imageWidth,
                                             const size_t imageHeight,
                                             const size_t imageChannels,
                                             const size_t numImages)
    {
        return AddNodeToNet(New<InputValue<ElemType>>(m_deviceId, inputName, imageWidth, imageHeight, imageChannels, numImages, true));
    }

    ComputationNodePtr CreatePairNetworkNode(const std::wstring & inputName, const size_t rows, const size_t cols)
                {
        return AddNodeToNet(New<PairNetworkNode<ElemType>>(m_deviceId, inputName, rows, cols));
                }

    ComputationNodePtr CreateConvolutionNode(const std::wstring & nodeName,
                    const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels,
                                             const size_t horizontalSubsample, const size_t verticalSubsample,
                                             const bool zeroPadding = false,
                                             const size_t maxTempMemSizeInSamples = 0)
    {
        return AddNodeToNet(New<ConvolutionNode<ElemType>>(m_deviceId, nodeName,
                                                           kernelWidth, kernelHeight,
                                                                 outputChannels,
                                                                 horizontalSubsample,
                                                                 verticalSubsample, zeroPadding,
                                                                 maxTempMemSizeInSamples));
    }

    ComputationNodePtr CreateMaxPoolingNode(const std::wstring & nodeName,
                                            const size_t windowWidth,
                                            const size_t windowHeight,
                                            const size_t horizontalSubsample,
                                            const size_t verticalSubsample)
    {
        return AddNodeToNet(New<MaxPoolingNode<ElemType>>(m_deviceId, nodeName,
                                                          windowWidth, windowHeight,
                                                                horizontalSubsample,
                                                          verticalSubsample));
    }

    ComputationNodePtr CreateAveragePoolingNode(const std::wstring & nodeName, const size_t windowWidth,
                                                const size_t windowHeight, const size_t horizontalSubsample,
                                                const size_t verticalSubsample)
    {
        return AddNodeToNet(New<AveragePoolingNode<ElemType>>(m_deviceId, nodeName,
                                                              windowWidth, windowHeight,
                                                                    horizontalSubsample,
                                                              verticalSubsample));
    }

    // this is the catch-all for all cases not covered as special cases above
    // Unlike the specialized ones above, this one creates nodes by type given as a string.
    ComputationNodePtr CreateComputationNode(const std::wstring & nodeType, const std::wstring & nodeName)
    {
        return AddNodeToNet(NewStandardNode(nodeType, m_deviceId, nodeName));
    }

    // TODO: These next three functions are wrappers around CreateXXXNode(). Remove these.

    ComputationNodePtr Parameter(const size_t rows, size_t cols, const std::wstring nodeName = L"") // TODO: remove
    {
        return CreateLearnableParameter(nodeName, rows, cols);
    }

    ComputationNodePtr Input(const size_t rows, const size_t cols, const std::wstring nodeName = L"")   // TODO: remove
    {
        return CreateInputNode(nodeName, rows, cols);
    }

    ComputationNodePtr Input(const size_t imageWidth, const size_t imageHeight,     // TODO: remove
                             const size_t imageChannels, const size_t numImages,
                             const std::wstring nodeName = L"")
    {
        return CreateInputNode(nodeName, imageWidth, imageHeight, imageChannels, numImages);
    }

    // -----------------------------------------------------------------------
    // node creation
    // -----------------------------------------------------------------------

    // The following functions create nodes and link them to the network and their inputs.
    // TODO: Do we need both this set and the one above that does not add inputs? Can they share more code?

    ComputationNodePtr PairNetwork(const ComputationNodePtr & a, const std::wstring nodeName = L"")
    {
        if (this->GetNodeFromName(a->NodeName(), nullptr, false) != nullptr)
        {
            fprintf(stderr, "PairNetwork: asked to pair a node with name %ls in another network. However, this network has already a node with the same name. Should avoid this case.\n", a->NodeName().c_str());
            RuntimeError("PairNetwork: asked to pair a node with name in another network. However, this network has already a node with the same name. Should avoid this case.\n");
        }
        return AddNodeToNetAndAttachInputs(New<PairNetworkNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr Convolution(const ComputationNodePtr weight,
                                   const ComputationNodePtr inputValues,
                                   const size_t kernelWidth,
                                   const size_t kernelHeight,
                                   const size_t outputChannels,
                                   const size_t horizontalSubsample,
                                   const size_t verticalSubsample,
                                   const bool zeroPadding = false,
                                   const std::wstring nodeName = L"",
                                   const size_t maxTempMemSizeInSamples = 0)
    {
        return AddNodeToNetAndAttachInputs(New<ConvolutionNode<ElemType>>(m_deviceId, nodeName,
                                                                          kernelWidth, kernelHeight,
                                                                          outputChannels,
                                                                          horizontalSubsample,
                                                                          verticalSubsample, zeroPadding,
                                                                          maxTempMemSizeInSamples),
                                           weight, inputValues);
    }

    ComputationNodePtr MaxPooling(const ComputationNodePtr inputValues,
                                  const size_t windowWidth,
                                  const size_t windowHeight,
                                  const size_t horizontalSubsample,
                                  const size_t verticalSubsample,
                                  const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<MaxPoolingNode<ElemType>>(m_deviceId, nodeName,
                                                                         windowWidth, windowHeight,
                                                                         horizontalSubsample,
                                                                         verticalSubsample),
                                           inputValues);
    }

    ComputationNodePtr AveragePooling(const ComputationNodePtr inputValues,
                                      const size_t windowWidth,
                                      const size_t windowHeight,
                                      const size_t horizontalSubsample,
                                      const size_t verticalSubsample,
                                      const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<AveragePoolingNode<ElemType>>(m_deviceId, nodeName,
                                                                             windowWidth, windowHeight,
                                                                             horizontalSubsample,
                                                                             verticalSubsample),
                                           inputValues);
    }

    ComputationNodePtr ErrorPrediction(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<ErrorPredictionNode<ElemType>>(m_deviceId, nodeName), a, b);
    }

    ComputationNodePtr PerDimMeanVarNormalization(const ComputationNodePtr feature, const ComputationNodePtr mean,
                                                  const ComputationNodePtr InvStdDev, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<PerDimMeanVarNormalizationNode<ElemType>>(m_deviceId, nodeName), feature, mean, InvStdDev);
    }

    ComputationNodePtr PerDimMeanVarDeNormalization(const ComputationNodePtr feature, const ComputationNodePtr mean,
                                                    const ComputationNodePtr InvStdDev, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<PerDimMeanVarDeNormalizationNode<ElemType>>(m_deviceId, nodeName), feature, mean, InvStdDev);
    }

    ComputationNodePtr SquareError(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<SquareErrorNode<ElemType>>(m_deviceId, nodeName), a, b);
    }


    ComputationNodePtr SequenceDecoder(const ComputationNodePtr label, const ComputationNodePtr prediction, const ComputationNodePtr pairscore, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<SequenceDecoderNode<ElemType>>(m_deviceId, nodeName), label, prediction, pairscore);
    }

    ComputationNodePtr CrossEntropyWithSoftmax(const ComputationNodePtr label, const ComputationNodePtr prediction, const std::wstring nodeName = L"")

    {
        return AddNodeToNetAndAttachInputs(New<CrossEntropyWithSoftmaxNode<ElemType>>(m_deviceId, nodeName), label, prediction);
    }

    ComputationNodePtr NoiseContrastiveEstimation(const ComputationNodePtr label, const ComputationNodePtr prediction,
                                                  const ComputationNodePtr input_weight,
                                                  const ComputationNodePtr input_bias, const std::wstring nodeName = L"",
                                                  NCEEvalMode mode = NCEEvalMode::None)
    {
        return AddNodeToNetAndAttachInputs(New<NoiseContrastiveEstimationNode<ElemType>>(m_deviceId, nodeName, mode), label, prediction, input_weight, input_bias);
    }

    ComputationNodePtr ClassCrossEntropyWithSoftmax(const ComputationNodePtr label, const ComputationNodePtr prediction,
                                                    const ComputationNodePtr input_weight,
                                                    const ComputationNodePtr cls_log_post_prob,
                                                    const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<ClassBasedCrossEntropyWithSoftmaxNode<ElemType>>(m_deviceId, nodeName), label, prediction, input_weight, cls_log_post_prob);
    }

    ComputationNodePtr CRF(const ComputationNodePtr label,
                           const ComputationNodePtr postDepScore,
                           const ComputationNodePtr transition_score,
                           const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<CRFNode<ElemType>>(m_deviceId, nodeName), label, postDepScore, transition_score);
    }

    ComputationNodePtr DummyCriterion(const ComputationNodePtr objectives, const ComputationNodePtr derivatives, const ComputationNodePtr prediction, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<DummyCriterionNode<ElemType>>(m_deviceId, nodeName), objectives, derivatives, prediction);
    }

    ComputationNodePtr LSTM(const ComputationNodePtr obs, 
                            const ComputationNodePtr inputGate, 
                            const ComputationNodePtr forgetGate, 
                            const ComputationNodePtr outputGate, 
                            const ComputationNodePtr memoryCellWgt, 
                            const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<LSTMNode<ElemType>>(m_deviceId, nodeName), obs, inputGate, forgetGate, outputGate, memoryCellWgt);
    }

    ComputationNodePtr CrossEntropy(const ComputationNodePtr label, const ComputationNodePtr prediction, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<CrossEntropyNode<ElemType>>(m_deviceId, nodeName), label, prediction);
    }

    ComputationNodePtr MatrixL1Reg(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<MatrixL1RegNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr MatrixL2Reg(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<MatrixL2RegNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr Mean(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<MeanNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr InvStdDev(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<InvStdDevNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr Negate(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<NegateNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr RectifiedLinear(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<RectifiedLinearNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr Sigmoid(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<SigmoidNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr Tanh(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<TanhNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr Exp(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<ExpNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr Log(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<LogNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr Cos(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<CosineNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr Softmax(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<SoftmaxNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr LogSoftmax(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<LogSoftmaxNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr Sum(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<SumElementsNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr Scale(const ComputationNodePtr scalar, const ComputationNodePtr matrix, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<ScaleNode<ElemType>>(m_deviceId, nodeName), scalar, matrix);
    }

    ComputationNodePtr Transpose(const ComputationNodePtr matrix, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<TransposeNode<ElemType>>(m_deviceId, nodeName), matrix);
    }

    ComputationNodePtr Times(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<TimesNode<ElemType>>(m_deviceId, nodeName), a, b);
    }

    ComputationNodePtr TransposeTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<TransposeTimesNode<ElemType>>(m_deviceId, nodeName), a, b);
    }

    ComputationNodePtr ElementTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<ElementTimesNode<ElemType>>(m_deviceId, nodeName), a, b);
    }

    ComputationNodePtr RowElementTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<RowElementTimesNode<ElemType>>(m_deviceId, nodeName), a, b);
    }

    ComputationNodePtr ColumnElementTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<ColumnElementTimesNode<ElemType>>(m_deviceId, nodeName), a, b);
    }

    ComputationNodePtr StrideTimes(const ComputationNodePtr a, const ComputationNodePtr b, const ComputationNodePtr c, const std::wstring nodeName = L"")
                {
        return AddNodeToNetAndAttachInputs(New<StrideTimesNode<ElemType>>(m_deviceId, nodeName), a, b, c);
                }

    ComputationNodePtr DiagTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<DiagTimesNode<ElemType>>(m_deviceId, nodeName), a, b);
    }

    ComputationNodePtr CosDistance(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<CosDistanceNode<ElemType>>(m_deviceId, nodeName), a, b);
    }

    ComputationNodePtr KhatriRaoProduct(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<KhatriRaoProductNode<ElemType>>(m_deviceId, nodeName), a, b);
    }

    ComputationNodePtr Plus(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<PlusNode<ElemType>>(m_deviceId, nodeName), a, b);
    }

    ComputationNodePtr Minus(const ComputationNodePtr a,
                             const ComputationNodePtr b,
                             const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<MinusNode<ElemType>>(m_deviceId, nodeName), a, b);
    }

    ComputationNodePtr Dropout(const ComputationNodePtr a, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<DropoutNode<ElemType>>(m_deviceId, nodeName), a);
    }

    ComputationNodePtr Reshape(const ComputationNodePtr a,
                               const size_t num_rows,
                               const size_t img_width,
                               const size_t img_height,
                               const size_t img_channels,
                               const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<ReshapeNode<ElemType>>(m_deviceId, nodeName, num_rows, img_width, img_height, img_channels), a);
    }

    ComputationNodePtr RowRepeat(const ComputationNodePtr a, const size_t num_repeat, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<RowRepeatNode<ElemType>>(m_deviceId, nodeName, num_repeat), a);
    }

    ComputationNodePtr PastValue(const ComputationNodePtr a, const float initHiddenActivity, const size_t row_size, const size_t col_size, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<PastValueNode<ElemType>>(m_deviceId, nodeName, initHiddenActivity, row_size, col_size), a);
    }

    ComputationNodePtr FutureValue(const ComputationNodePtr a, const float initHiddenActivity, const size_t row_size, const size_t col_size, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<FutureValueNode<ElemType>>(m_deviceId, nodeName, initHiddenActivity, row_size, col_size), a);
    }

    ComputationNodePtr Parallel(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<ParallelNode<ElemType>>(m_deviceId, nodeName), a, b);
    }

    ComputationNodePtr RowSlice(const ComputationNodePtr a, const size_t start_index, const size_t num_rows, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<RowSliceNode<ElemType>>(m_deviceId, nodeName, start_index, num_rows), a);
    }

    ComputationNodePtr RowStack(const std::vector<ComputationNodePtr> inputs, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<RowStackNode<ElemType>>(m_deviceId, nodeName), inputs);
    }

    ComputationNodePtr GMMLogLikelihood(const ComputationNodePtr unnormedPrior,
                                        const ComputationNodePtr mean,
                                        const ComputationNodePtr logStddev,
                                        const ComputationNodePtr feature,
                                        const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<GMMLogLikelihoodNode<ElemType>>(m_deviceId, nodeName), unnormedPrior, mean, logStddev, feature);
    }

    ComputationNodePtr TimeReverse(const ComputationNodePtr input, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<TimeReverseNode<ElemType>>(m_deviceId, nodeName), input);
    }

    ComputationNodePtr LookupTable(const ComputationNodePtr dictionary, const ComputationNodePtr input, const std::wstring nodeName = L"")
    {
        return AddNodeToNetAndAttachInputs(New<LookupTableNode<ElemType>>(m_deviceId, nodeName), dictionary, input);
    }

    // -----------------------------------------------------------------------
    // node access
    // -----------------------------------------------------------------------

    bool NodeNameExist(const std::wstring& name) const
    {
        auto iter = m_nameToNodeMap.find(name);
        return (iter != m_nameToNodeMap.end());
    }

    ComputationNodePtr GetNodeFromName(const std::wstring& name, ComputationNetwork<ElemType>* anotherNetwork = nullptr, bool bPanic = true) const
    {
        auto iter = m_nameToNodeMap.find(name);
        if (iter != m_nameToNodeMap.end())
        {
            //found
            return iter->second;
        }

                    if (anotherNetwork != nullptr)
                        return anotherNetwork->GetNodeFromName(name);

                    if (bPanic)
                        RuntimeError("GetNodeFromName: Node name %s does not exist.", name.c_str());
        else
                        return nullptr;
    }

    // GetNodesFromName - Get all the nodes from a name that may match a wildcard '*' pattern
    //   only patterns with a single '*' at the beginning, in the middle, or at the end are accepted
    // name - node name (with possible wildcard)
    // returns: vector of nodes that match the pattern, may return an empty vector for no match
    std::vector<ComputationNodePtr> GetNodesFromName(const std::wstring& name) const
    {
        std::vector<ComputationNodePtr> nodes;
        size_t found = name.find_first_of(L'*');
        if (found == std::wstring::npos)
        {
            if (NodeNameExist(name))
                nodes.push_back(GetNodeFromName(name));
            }
        else
        {
            std::wstring head = name.substr(0, found);
            std::wstring tail = name.substr(found + 1);
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                const wstring& nodeName = nodeIter->first;

                // if it matches on both ends (we only support A*B patterns it's a match
                bool headMatch = head.empty() || nodeName.find(head) == 0;
                bool tailMatch = tail.empty() || nodeName.rfind(tail) == nodeName.size() - tail.size();
                if (headMatch && tailMatch)
                    nodes.push_back(nodeIter->second);
                }
            }
        return nodes;
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    int FindInRecurrentLoop(const ComputationNodePtr startNode, vector<ComputationNodePtr>& recurrentNodes)
    {
        int iFound = -1;

        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            if (std::find((*iter).m_recurrentNodes.begin(), (*iter).m_recurrentNodes.end(), startNode) != (*iter).m_recurrentNodes.end())
            {
                iFound = (*iter).m_loopId;
                recurrentNodes = (*iter).m_recurrentNodesForForward;
                break;
            }
        }

        return iFound;
    }

    int FindInRecurrentLoop(const ComputationNodePtr startNode)
    {
        int iFound = -1;

        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            if (std::find((*iter).m_recurrentNodes.begin(), (*iter).m_recurrentNodes.end(), startNode) != (*iter).m_recurrentNodes.end())
            {
                iFound = (*iter).m_loopId;
                break;
            }
        }

        return iFound;
    }

    bool IsFuncValueOlderThanInputs(const std::vector<ComputationNodePtr>& recurrentNodes)
    {
        for (auto ptr = recurrentNodes.begin(); ptr != recurrentNodes.end(); ptr++)
        {
            if ((*ptr)->IsFuncValueOlderThanInputs() && 
                (*ptr)->OperationName() != PastValueNode<ElemType>::TypeName() &&
                (*ptr)->OperationName() != FutureValueNode<ElemType>::TypeName())
            {
                return true;
            }
        }
        return false;
    }

    void EvaluateLoop(std::list<ComputationNodePtr>& /*allNodes*/, const ComputationNodePtr startNode)
    {
        std::vector<ComputationNodePtr> recurrentNodes;
        int iLoopId = FindInRecurrentLoop(startNode, recurrentNodes);
        if (iLoopId != -1 && IsFuncValueOlderThanInputs(recurrentNodes) && m_recurrentInfo[iLoopId].m_completedEvaluate == false)
        {
            for (auto nodeIter = recurrentNodes.begin(); nodeIter != recurrentNodes.end(); nodeIter++)
                (*nodeIter)->SetFunctionAndGradientSize(m_actMiniBSize);

            int iMBSize = m_actMiniBSize / m_nbrSlicesInEachRecurrentIteration;

            if (m_recurrentInfo[iLoopId].m_isForwardLoop)
            {
                for (int timeIndex = 0; timeIndex < iMBSize; timeIndex ++)
                {
                    for (auto nodeIter = recurrentNodes.begin(); nodeIter != recurrentNodes.end(); nodeIter++)
                    {
                        (*nodeIter)->EvaluateThisNodeGivenInputs(timeIndex);
                        (*nodeIter)->UpdateEvalTimeStamp();
                    }
                } 
            }
            else
            {
                for (int timeIndex = iMBSize-1; timeIndex >= 0; timeIndex--)
                {
                    for (auto nodeIter = recurrentNodes.begin(); nodeIter != recurrentNodes.end(); nodeIter++)
                    {
                        (*nodeIter)->EvaluateThisNodeGivenInputs(timeIndex);
                        (*nodeIter)->UpdateEvalTimeStamp();
                    }
                }
            }

            m_recurrentInfo[iLoopId].m_completedEvaluate = true;
        }
    }

    bool IsTypicalCriterionNode(ComputationNodePtr nodePtr)
    {
        if (nodePtr->OperationName() == SquareErrorNode<ElemType>::TypeName() ||
            nodePtr->OperationName() == CrossEntropyWithSoftmaxNode<ElemType>::TypeName() ||
            nodePtr->OperationName() == CrossEntropyNode<ElemType>::TypeName() ||
            nodePtr->OperationName() == ClassBasedCrossEntropyWithSoftmaxNode<ElemType>::TypeName() ||
            nodePtr->OperationName() == ErrorPredictionNode<ElemType>::TypeName() ||               
            nodePtr->OperationName() == CRFNode<ElemType>::TypeName() ||
            nodePtr->OperationName() == DummyCriterionNode<ElemType>::TypeName())
            return true;

        return false;
    }

    void SetNodesReqMultiSeqHandling()
    {
        for (auto node : m_nodesReqMultiSeqHandling)
        {
            //SumElements node will generate a scalar value and so it should never require special handling
            //TransposeNode will change the size of columns and so it should also not included for special handling
            //their child node should instead
            if (node->OperationName() != SumElementsNode<ElemType>::TypeName() &&
                node->OperationName() != TransposeNode<ElemType>::TypeName() &&
                node->OperationName() != MeanNode<ElemType>::TypeName() &&
                node->OperationName() != InvStdDevNode<ElemType>::TypeName() 
                )
                node->SetReqMultiSeqHandlingTo(true);
        }

        //if a typical criterion node is used as the training criterion node we assume it requires multiseq handling 
        //this is for backward compatibility
        for (auto node : m_finalCriteria)
            if (IsTypicalCriterionNode(node))
                node->SetReqMultiSeqHandlingTo(true);

        for (auto node : m_evalNodes)
            if (IsTypicalCriterionNode(node))
                node->SetReqMultiSeqHandlingTo(true);
        }

    void Evaluate(const ComputationNodePtr rootNode)
    {
        BuildAndValidateNetwork(rootNode);

        std::list<ComputationNodePtr>& allNodes = GetEvalOrder(rootNode);

#ifdef DISPLAY_DEBUG
        for (auto nodeIter=allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
            fprintf (stderr, "Evaluate Node: %s\n",(msra::strfun::utf8 ((*nodeIter)->NodeName())).c_str());
#endif

        for (int i = 0; i < m_recurrentInfo.size(); i++)
            m_recurrentInfo[i].m_completedEvaluate = false;

        for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
        {
            (*nodeIter)->SetNbrSlicesInEachRecurrentIteration(m_nbrSlicesInEachRecurrentIteration);
            if ((*nodeIter)->ReqMultiSeqHandling())
                    (*nodeIter)->ResetBound(&m_SentenceBoundary, &m_minibatchPackingFlag);
        }

        for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
        {
            EvaluateLoop(allNodes, (*nodeIter));

            if ((*nodeIter)->IsFuncValueOlderThanInputs() && (FindInRecurrentLoop(*nodeIter) == -1))
            {
#ifdef DISPLAY_DEBUG
                fprintf (stderr, "Evaluate Node: %s\n",(msra::strfun::utf8 ((*nodeIter)->NodeName())).c_str());
#endif
#if DUMPOUTPUT
                fprintf(stderr,"Forward_%ls\n",(*nodeIter)->NodeName().c_str());
#endif
                // we manage time stamp here so that derived classes don't need to worry about it
                (*nodeIter)->EvaluateThisNodeGivenInputs(); 
                (*nodeIter)->UpdateEvalTimeStamp();
            }
        }
    }

    void SetActualMiniBatchSize(const size_t aSize, vector<ComputationNodePtr>* featNodes = nullptr)
    {
        m_actMiniBSize = (int) aSize;

        // assume that all nodes in recurrent loops need to be reset to aSize minibatch size, so need to reset the following
        for (int i = 0; i < m_recurrentInfo.size(); i++)
        {
            m_recurrentInfo[i].m_completedEvaluate = false;
            m_recurrentInfo[i].m_completedGradient = false;
        }

        for (int i = 0; i < m_recurrentInfo.size(); i++)
            for (auto nodeIter = m_recurrentInfo[i].m_recurrentNodes.begin(); nodeIter != m_recurrentInfo[i].m_recurrentNodes.end(); nodeIter++)
                (*nodeIter)->SetFunctionAndGradientSize(m_actMiniBSize);

        if (featNodes)
        {
            for (auto ptr = featNodes->begin(); ptr != featNodes->end(); ptr++)
            {
                size_t nr = (*ptr)->FunctionValues().GetNumRows();
                (*ptr)->FunctionValues().Resize(nr, aSize);
            }
        }
    }

    // GetMaxMBSize - Get the maximum minibatch size that will be seen in a training run
    // returns the result from SetActualMiniBatchSize(). Note GetActualMBSize() also exists but returns a value derived from the inputs dimensions
    size_t GetMaxMBSize() { return m_actMiniBSize; }

    void SetActualNbrSlicesInEachRecIter(const size_t aSize)
    {
        m_nbrSlicesInEachRecurrentIteration = aSize;
    }

    void ComputeGradientLoop(std::list<ComputationNodePtr>& /*allNodes*/, const ComputationNodePtr startNode)
    {
        std::vector<ComputationNodePtr> recurrentNodes;
        int iLoopId = FindInRecurrentLoop(startNode, recurrentNodes);
        if (iLoopId != -1)
        {
            if (m_recurrentInfo[iLoopId].m_completedGradient == false)
            {
                int mbSize = m_actMiniBSize / m_nbrSlicesInEachRecurrentIteration;
                if (m_recurrentInfo[iLoopId].m_isForwardLoop)
                {
                    for (int timeIndex = mbSize - 1; timeIndex >= 0; timeIndex--)
                    {
                        for (auto nodeIter = recurrentNodes.rbegin(); nodeIter != recurrentNodes.rend(); ++nodeIter)
                        {
                            (*nodeIter)->SetNbrSlicesInEachRecurrentIteration(m_nbrSlicesInEachRecurrentIteration); // TODO: move to FrameRange object
                            (*nodeIter)->ComputeGradientForChildren(timeIndex);
                        }
                    }
                }
                else
                {
                    for (int timeIndex = 0; timeIndex < mbSize; timeIndex++)
                    {
                        for (auto nodeIter = recurrentNodes.rbegin(); nodeIter != recurrentNodes.rend(); ++nodeIter)
                        {
                            (*nodeIter)->SetNbrSlicesInEachRecurrentIteration(m_nbrSlicesInEachRecurrentIteration);
                            (*nodeIter)->ComputeGradientForChildren(timeIndex);
                        }
                    }
                }

                m_recurrentInfo[iLoopId].m_completedGradient = true;
            }
        }
    }

    virtual void ComputeGradient(const ComputationNodePtr rootNode, 
                                 bool bResetToOne = true,  /// true if reset the gradient of rootnode to 1.0
                    const Matrix<ElemType>* rootGradientInitValue = nullptr,
                                 bool bClearGradient = true,
                                 bool resetTimeStampAfterComputation = false
                    )
    {
        if (bResetToOne && rootNode->FunctionValues().GetNumElements() != 1)
            RuntimeError("ComputeGradient: The root of the Gradient computation must evaluate to R1 value.");

        //run forward pass first
        Evaluate(rootNode);

                    if (bClearGradient)
        ClearGradientForAllNodes(rootNode);

        //run backward pass
        std::list<ComputationNodePtr>& allNodes = GetGradientCalcOrder(rootNode);
            
        if (bResetToOne)
        {
            rootNode->GradientValues().Resize(1, 1);
            rootNode->GradientValues().SetValue(1);
        }

        if (rootGradientInitValue != nullptr)
            rootNode->GradientValues().SetValue(*rootGradientInitValue);

        for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
        {
#ifdef DISPLAY_DEBUG
            fprintf(stderr, "Compute Gradient For Node: %s(%s) Against Children\n",
                        (msra::strfun::utf8 ((*nodeIter)->OperationName())).c_str(),
                        (msra::strfun::utf8 ((*nodeIter)->NodeName())).c_str());
#endif
            ComputeGradientLoop(allNodes, *nodeIter);

            (*nodeIter)->ComputeGradientForChildren();
        }

        //since we now allow sharing of the matrix for function value and gradient value. the function values are now destroyed
        //after gradient computation and need to be recomputed. This is indicated by the timestamp updated using this function
        //resetTimeStampAfterComputation is by default false because ComputeGradient in normal case is followed by new batch of input
        if (resetTimeStampAfterComputation)
            ResetEvalTimeStamp();
    }

    //for debugging purpose
    void PrintComputationTree(const ComputationNodePtr rootNode,
                              const bool forwardCompute,
                              const bool printMatrices = false)
    {
        std::list<ComputationNodePtr> nodes;
        if (forwardCompute)
        {
            fprintf(stderr, "\n\nPrinting Forward Computation Node Order ... \n");
            nodes = GetEvalOrder(rootNode);
        }
        else
        {
            fprintf(stderr, "\n\nPrinting Gradient Computation Node Order ... \n");
            nodes = GetGradientCalcOrder(rootNode);
        }

        if (nodes.size() == 0)
        {
            fprintf(stderr, "\n$$$$ EMPTY !!!!!\n");
            return;
        }

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            ComputationNodePtr node = (*nodeIter);
            node->PrintSelf(printMatrices);
        }
    }

    // -----------------------------------------------------------------------
    // network editing
    // -----------------------------------------------------------------------

    void RenameNode(const ComputationNodePtr node, const std::wstring newNodeName)
    {
        // TODO: check if new name exists
        m_nameToNodeMap.erase(node->NodeName());
        node->NodeName() = newNodeName;
        AddNodeToNet(node);
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    void ClearCaches()
    {
        m_built.clear();
        m_inputs.clear();
        m_learnableParameters.clear();
        ClearCalcOrderCaches();
    }

    void RebuildNetwork(const ComputationNodePtr rootNode)
    {
        ClearCaches();
        BuildAndValidateNetwork(rootNode);
    }

    // -----------------------------------------------------------------------
    // node-group access
    // -----------------------------------------------------------------------

    std::list<ComputationNodePtr> & InputNodes(const ComputationNodePtr rootNode, bool bNoBuild = false)
    {
        if (bNoBuild == false)
            BuildAndValidateNetwork(rootNode);
        return m_inputs[rootNode];
    }

    std::list<ComputationNodePtr> & LearnableNodes(const ComputationNodePtr rootNode)
    {
        BuildAndValidateNetwork(rootNode);
        return m_learnableParameters[rootNode];
    }

    inline std::vector<ComputationNodePtr> & FeatureNodes()        { return m_features; }
    inline std::vector<ComputationNodePtr> & LabelNodes()          { return m_labels; }
    inline std::vector<ComputationNodePtr> & FinalCriterionNodes() { return m_finalCriteria; }

    inline std::vector<ComputationNodePtr> & TrainCriterionNodesFrom(wstring criterionNodeName)
    {
        ComputationNodePtr node = this->GetNodeFromName(criterionNodeName);
        this->ValidateNetwork(node);
        if (node->FunctionValues().GetNumElements() != 1)
            InvalidArgument("the trainCriterionNodeName specified in the config file is not a valid training criterion node.");
        m_tmpTrainCriterion.clear();
        m_tmpTrainCriterion.push_back(node);
        return m_tmpTrainCriterion;
    }

    inline std::vector<ComputationNodePtr> & EvalCriterionNodesFrom(wstring criterionNodeName)
    {
        ComputationNodePtr node = this->GetNodeFromName(criterionNodeName);
        this->ValidateNetwork(node);
        if (node->FunctionValues().GetNumElements() != 1)
            InvalidArgument("the trainCriterionNodeName specified in the config file is not a valid training criterion node.");
        m_tmpEvalulationCriterion.clear();
        m_tmpEvalulationCriterion.push_back(node);
        return m_tmpEvalulationCriterion;
    }

    inline std::vector<ComputationNodePtr> & NodesReqMultiSeqHandling() { return m_nodesReqMultiSeqHandling; }
    inline std::vector<ComputationNodePtr> & EvaluationNodes()          { return m_evalNodes; }
    inline std::vector<ComputationNodePtr> & OutputNodes()              { return m_outputNodes; }
    inline std::vector<ComputationNodePtr> & PairNodes()                { return m_pairNodes; }

    inline std::vector<RecurrentInfo> & RecurrentNodes() { return m_recurrentInfo; }

    // -----------------------------------------------------------------------
    // node access
    // -----------------------------------------------------------------------

    size_t GetTotalNumberOfNodes() const { return m_nameToNodeMap.size(); }

    // TODO: could be a dup
    std::map<const std::wstring, ComputationNodePtr, nocase_compare> & GetNameToNodeMap()    // specially for ExperimentalNetworkBuilder; don't use this otherwise
    {
        return m_nameToNodeMap;
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    void ResetEvalTimeStamp()
    {
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            nodeIter->second->ResetEvalTimeStamp();
    }

    // -----------------------------------------------------------------------
    // network editing
    // -----------------------------------------------------------------------

    //change the node associated with nodeName to newNode; used in the KL-reg based adaptation to reduce feature copy
    //need to update all the mappings as well childrens
    void ChangeNode(wstring nodeName, ComputationNodePtr newNode)
    {
        ComputationNodePtr oldNode = GetNodeFromName(nodeName);
        if (oldNode->OperationName() != newNode->OperationName())
            InvalidArgument("newNode must have the same type as the old node.");

        //change children
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodePtr node = nodeIter->second;
            for (int i = 0; i < node->ChildrenSize(); i++)
                if (node->Inputs(i) == oldNode)
                    node->SetInput(i, newNode);
        }

        //change name map
        m_nameToNodeMap[nodeName] = newNode;
        for (int i = 0; i < oldNode->ChildrenSize(); i++)
            newNode->SetInput(i, oldNode->Inputs(i));

        //change other maps
        for (auto groupIter : GetAllNodeGroups())
        {
            auto & group = *groupIter;
            for (int i = 0; i < group.size(); i++)
                if (group[i] == oldNode)
                    group[i] = newNode;
        }
    }

    // replace the old node with the current node, assuming the old node is a leaf node
    // need to update those nodes who use oldNode as their child
    void ReplaceLeafNode(wstring oldNodeName, ComputationNodePtr newNode)
    {
        ComputationNodePtr oldNode = GetNodeFromName(oldNodeName);

        // change the input of those nodes whose child is oldNode
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodePtr node = nodeIter->second;
            for (int i = 0; i < node->ChildrenSize(); i++)
                if (node->Inputs(i) == oldNode)
                    node->SetInput(i, newNode);
        }
        m_nameToNodeMap[newNode->GetName()] = newNode;

        // now the old node becomes a orphan node , remove it
        DeleteNode(oldNodeName);
        //RemoveOrphanNode(oldNode);
    }

    void ReplaceFinalCriterionNode(wstring oldNodeName, ComputationNodePtr newNode)
    {
        // Checks if the node is a criterion node.
        int index = -1;
        for (int i = 0; i < m_finalCriteria.size(); ++i)
        {
            if (m_finalCriteria[i]->NodeName() == oldNodeName)
            {
                index = i;
                break;
            }
        }
        if (index == -1)
            RuntimeError("ReplaceFinalCriterionNode: the node to be replaced is not a criterion node.");

        // Replaces children.
        for (int i = 0; i < newNode->ChildrenSize(); ++i)
        {
            if (m_nameToNodeMap.find(newNode->Inputs(i)->NodeName()) == m_nameToNodeMap.end())
                RuntimeError("Child node does not exist.");
            newNode->SetInput(i, m_nameToNodeMap[newNode->Inputs(i)->NodeName()]);
        }

        // Addes it to criterion node list.
        m_finalCriteria[index] = newNode;
        m_nameToNodeMap[newNode->NodeName()] = newNode;
    }

    void AddFeatureNode(ComputationNodePtr featureNode)
    {
        wstring nodeName = featureNode->NodeName();
        if (NodeNameExist(nodeName))
            RuntimeError("AddFeatureNode: feature node already exists.");
        m_nameToNodeMap[nodeName] = featureNode;
        m_features.push_back(featureNode);
    }

    // We only remove the node, not delete it.
    void RemoveFeatureNode(ComputationNodePtr featureNode)
    {
        wstring nodeName = featureNode->NodeName();
        if (!NodeNameExist(nodeName))
            RuntimeError("RemoveFeatureNode: feature node does not exist.");

        ClearCaches();

        // Removes links.
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); ++nodeIter)
        {
            ComputationNodePtr node = nodeIter->second;
            for (size_t i = 0; i < node->ChildrenSize(); ++i)
            {
                ComputationNodePtr child = node->Inputs(i);
                if (child == featureNode)
                {
                    node->SetInput(i,NULL);
                    break;
                }
            }
        }

        // Removes from feature list.
        auto search = std::find(m_features.begin(), m_features.end(), featureNode);
        if (search != m_features.end())
            m_features.erase(search);

        m_nameToNodeMap.erase(nodeName);
    }

    // -----------------------------------------------------------------------
    // node access
    // -----------------------------------------------------------------------

    std::vector<ComputationNodePtr> GetAllNodes() const
    {
        std::vector<ComputationNodePtr> nodes;
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodePtr node = nodeIter->second;
            nodes.push_back(node);
        }
        return nodes;
    }

    std::list<ComputationNodePtr> GetNodesWithType(const wstring typeName, const ComputationNodePtr rootNode = nullptr)
    {
        std::list<ComputationNodePtr> nodesWithType;

        //find nodes from all available nodes
        if (rootNode == nullptr)
        {
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodePtr node = nodeIter->second;
                if (node->OperationName() == typeName)
                    nodesWithType.push_back(node);
            }
        }
        else
        {
            //for calculating a specific node
            std::list<ComputationNodePtr>& nodes = GetEvalOrder(rootNode);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodePtr node = (*nodeIter);
                if (node->OperationName() == typeName)
                    nodesWithType.push_back(node);
            }
        }

        return nodesWithType;
    }

    //return list of nodes that require precomputation and not precomputed yet.
    // TODO: name has a grammar error, fix
    std::list<ComputationNodePtr> GetNodesRequirePreComputation(const ComputationNodePtr rootNode = nullptr, bool checkComputed = true)
    {
        std::list<ComputationNodePtr> nodesRequirePreComputation;

        //find nodes from all available nodes
        if (rootNode == nullptr)
        {
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodePtr node = nodeIter->second;
                if (node->RequirePreCompute())
                {
                    auto preComputedNode = static_pointer_cast<PreComputedNode<ElemType>>(node);
                    if (!checkComputed || !preComputedNode->HasComputed())
                    {
                        nodesRequirePreComputation.push_back(node);
                    }
                }
            }
        }
        else //for calculating a specific node
        {
            std::list<ComputationNodePtr>& nodes = GetEvalOrder(rootNode);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodePtr node = *nodeIter;
                if (node->RequirePreCompute())
                {
                    auto preComputedNode = static_pointer_cast<PreComputedNode<ElemType>>(node);
                    if (!checkComputed || !preComputedNode->HasComputed())
                    {
                        nodesRequirePreComputation.push_back(node);
                    }
                }
            }
        }

        return nodesRequirePreComputation;
    }

    //return list of nodes that require precomputation and not precomputed yet.
    // TODO: name has grammar error, fix
    std::list<ComputationNodePtr> GetNodesRequireBatchMode(const ComputationNodePtr rootNode = nullptr, bool checkComputed = true)
    {
        std::list<ComputationNodePtr> nodesRequirePreComputation;

        if (rootNode == nullptr) //find nodes from all available nodes
        {
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodePtr node = nodeIter->second;
                if (node->RequireBatchMode())
                {
                    auto preComputedNode = static_pointer_cast<BatchModeNode<ElemType>>(node);
                    if (!checkComputed || !preComputedNode->HasComputed())
                        nodesRequirePreComputation.push_back(node);
                }
            }
        }
        else //for calculating a specific node
        {
            std::list<ComputationNodePtr>&  nodes = GetEvalOrder(rootNode);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodePtr node = (*nodeIter);
                if (node->RequireBatchMode())
                {
                    auto preComputedNode = static_pointer_cast<BatchModeNode<ElemType>>(node);
                    if (!checkComputed || !preComputedNode->HasComputed())
                        nodesRequirePreComputation.push_back(node);
                }
            }
        }

        return nodesRequirePreComputation;
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // Validate - Validate the network
    void ValidateNetwork(bool allowFragment = false, const bool bAllowNoCriterion = false)
    {
        // currently only validates nodes, we should validate everything we can
        if (FeatureNodes().size() == 0 && !allowFragment)
            RuntimeError("No Feature nodes specified");

        // first give criteria nodes as root node
        if (FinalCriterionNodes().size() > 0)
        {
            for (ComputationNodePtr & node : FinalCriterionNodes())
            {
                if (!allowFragment)
                    FormRecurrentLoops(node);
                PrintComputationTree(node, false);
                size_t actualMBSize = this->GetActualMBSize();
                this->SetActualMiniBatchSize(actualMBSize);
                ValidateNetwork(node);
            }
        }
        else if (bAllowNoCriterion == true)
        {
            // do nothing
        }
        else if (!allowFragment)
            RuntimeError("No Criterion nodes specified");

        // now output nodes
        if (OutputNodes().size() > 0)
        {
            for (ComputationNodePtr node : OutputNodes())
            {
                if (!allowFragment)
                    FormRecurrentLoops(node);
                ValidateNetwork(node);
            }
        }
        else if (!allowFragment)
            RuntimeError("No Output nodes specified");

        // now evaluation nodes
        if (EvaluationNodes().size() > 0)
        {
            for (ComputationNodePtr node : EvaluationNodes())
            {
                if (!allowFragment)
                    FormRecurrentLoops(node);
                ValidateNetwork(node);
            }
        }
    }

    void ValidateNetwork(const ComputationNodePtr rootNode)
    {
        fprintf(stderr, "\n\nValidating node %ls \n", rootNode->NodeName().c_str());

        std::list<ComputationNodePtr>& nodes = GetEvalOrder(rootNode);

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            (*nodeIter)->Validate();
        }

        fprintf(stderr, "\n\n");
    }

    void BuildAndValidateNetwork(const ComputationNodePtr rootNode)
    {
        const ComputationNodePtr key = rootNode;

        //not found
        if (m_built.find(key) == m_built.end())
        {
            m_built[key] = true;
            FormRecurrentLoops(rootNode);
            ValidateNetwork(rootNode);
            CollectInputAndLeanableParameters(rootNode);
            SetNodesReqMultiSeqHandling();
        }
    }

    //this function will need to be called before actual validation and execution to 
    //predetermine how to share matrices to reduce memory usage.
    //evalRootNodes do not need gradient computation
    //trainRootNodes need gradient computation
    void AllocateMatrices(std::vector<ComputationNodePtr>& evalRootNodes, std::vector<ComputationNodePtr>& trainRootNodes)
    {
        //allocate memory for forward computation
        fprintf(stderr, "\n\nAllocate matrices for forward computing\n");
        for (int i = 0; i < evalRootNodes.size(); i++)
            AllocateEvalMatrices(evalRootNodes[i]);

        for (int i = 0; i < trainRootNodes.size(); i++)
            AllocateEvalMatrices(trainRootNodes[i]);

        //allocate memory for backward computation
        //we intentionally separate it from above loop to make sure forward computing gets the right matrices
        for (int i = 0; i < trainRootNodes.size(); i++)
            AllocateGradientMatrices(trainRootNodes[i]);
    }

    void AllocateEvalMatrices(ComputationNodePtr rootNode)
    {
        FormRecurrentLoops(rootNode);

        std::list<ComputationNodePtr>& nodes = GetEvalOrder(rootNode);

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            (*nodeIter)->RequestEvalMatrices(m_matrixPool);
            (*nodeIter)->ReleaseMatricesAfterEval(m_matrixPool);
        }
    }

    void AllocateGradientMatrices(ComputationNodePtr rootNode)
    {
        //first, compute the number of parents for each node
        std::map<ComputationNodePtr, int> numParents;

        std::list<ComputationNodePtr>& nodes = GetEvalOrder(rootNode);

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            std::vector<ComputationNodePtr> children = (*nodeIter)->GetChildren();
            for (int i = 0; i < children.size(); i++)
                numParents[children[i]] ++;
        }

        //now, simulate the gradient computation order to determine how to allocate matrices
        std::list<ComputationNodePtr>& allNodes = GetGradientCalcOrder(rootNode);

        for (int i = 0; i < m_recurrentInfo.size(); i++)
            m_recurrentInfo[i].m_completedGradient = false;

        for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
        {
            std::vector<ComputationNodePtr> recurrentNodes;
            int iLoopId = FindInRecurrentLoop(*nodeIter, recurrentNodes);
            if (iLoopId != -1 && m_recurrentInfo[iLoopId].m_completedGradient == false)
            {
                for (auto nodeIterInLoop = recurrentNodes.rbegin(); nodeIterInLoop != recurrentNodes.rend(); ++nodeIterInLoop)
                    AllocateGradientMatricesForChildren(*nodeIterInLoop, numParents);
                m_recurrentInfo[iLoopId].m_completedGradient = true;
            }
            else
                AllocateGradientMatricesForChildren(*nodeIter, numParents);

            (*nodeIter)->ReleaseGradientMatrices(m_matrixPool);
        }
    }

    void AllocateGradientMatricesForChildren(ComputationNodePtr parentNode, std::map<ComputationNodePtr, int>& numParents)
    {
        std::vector<ComputationNodePtr> children = parentNode->GetChildren();
        for (int i = 0; i < children.size(); i++)
            children[i]->RequestGradientMatrices(m_matrixPool, numParents[children[i]]);
    }

    /**
    call unit test of each node
    this adds a verification of the correctness of node operations.
    */
    bool UnitTest(bool allowFragment = false)
    {
        vector<wstring> vErrors;
        // currently only validates nodes, we should validate everything we can
        if (FeatureNodes().size() == 0 && !allowFragment)
            RuntimeError("No Feature nodes specified");
        // first give criteria nodes as root node
        if (FinalCriterionNodes().size() > 0)
        {
            for (auto node : FinalCriterionNodes())
            {
                if (!allowFragment)
                    FormRecurrentLoops(node);
                size_t actualMBSize = this->GetActualMBSize();
                this->SetActualMiniBatchSize(actualMBSize);
                if (!UnitTest(node))
                    vErrors.push_back(node->NodeName().c_str());
            }
        }
        else if (!allowFragment)
            RuntimeError("No Criterion nodes specified");
        // now output nodes
        if (OutputNodes().size() > 0)
        {
            for (auto node : OutputNodes())
            if (!UnitTest(node))
                vErrors.push_back(node->NodeName().c_str());
        }
        else if (!allowFragment)
            RuntimeError("No Output nodes specified");
        // now evaluation nodes
        if (EvaluationNodes().size() > 0)
        {
            for (auto node : EvaluationNodes())
            if (!UnitTest(node))
                vErrors.push_back(node->NodeName().c_str());
        }
        return vErrors.empty();
    }

    bool UnitTest(const ComputationNodePtr rootNode)
    {
        fprintf(stderr, "\n\n Unit test node %ls \n", rootNode->NodeName().c_str());

        std::list<ComputationNodePtr>&  nodes = GetEvalOrder(rootNode);

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        if (!(*nodeIter)->UnitTest())
            return false;

        fprintf(stderr, "\n\n");

        return true;
    }

    // -----------------------------------------------------------------------
    // specialized operations
    // -----------------------------------------------------------------------

    // TODO: lift this into config language, move underlying code to math lib

    //========================================
    // This function performs SVD decomposition for different groups of learnable  parameters
    // we perform SVD decomposition such that
    //  A \approx B*C, where rank(B)=rank(C)=r < rank(A)
    // After SVD decomposition, the node A will become an intermediate node whose children are B,C ;
    // B and C are two learnable parameters
    //========================================
    void PerformSVDecomposition(const map<wstring, float>& SVDConfig)
    {
        vector<pair<vector<wstring>, float> > nodeGroups;
        wregex NameFilter;

        for (auto e : SVDConfig)
        {
            wstring regexStr = e.first;
            float keepRatio = e.second;
            vector<wstring> NamesInGroup;

            NameFilter.assign(regexStr);

            for (auto n = m_nameToNodeMap.begin(); n != m_nameToNodeMap.end();  n++)
            {
                if (!regexStr.empty() && !regex_match(n->first, NameFilter))
                {
                    // if regexStr is not empty and the the node node does not match with the regexStr
                    continue;
                }

                ComputationNodePtr ptr = n->second;
                if (ptr->OperationName() != LearnableParameter<ElemType>::TypeName())
                    continue;

                Matrix<ElemType> W = ptr->FunctionValues();
                if (W.GetNumCols() == 1 || W.GetNumRows() == 1)
                    continue;

                // still here ?
                NamesInGroup.push_back(n->first);
            }
            nodeGroups.push_back(make_pair(NamesInGroup, keepRatio));
        }

        size_t groupID = 0;
        for (auto& group : nodeGroups)
        {
            float keepratio = group.second;
            fprintf(stderr,
                    "--------------------------------------------------------------------------------------------\n");
            fprintf(stderr,
                    "ParameterSVD: start to process group %d with KeepRatio=%.2f\n",
                    (int) groupID++, keepratio);
            fprintf(stderr,
                    "--------------------------------------------------------------------------------------------\n");

            for (auto name : group.first)
            {
                if (m_nameToNodeMap.find(name) == m_nameToNodeMap.end())
                {
                    // could be deleted in the previous groups
                    continue;
                }

                ComputationNodePtr pNode = m_nameToNodeMap[name];
                //========================================
                // Step 1. do SVD decomposition
                //========================================
                Matrix<ElemType> A = pNode->FunctionValues();

                // it is a vector, no need to do it
                if (A.GetNumCols() == 1 || A.GetNumRows() == 1)
                    continue;

                size_t m = A.GetNumRows();
                size_t n = A.GetNumCols();

                Matrix<ElemType> S(-1), U(-1), VT(-1), W(-1);
                std::chrono::time_point < std::chrono::system_clock > stTime = std::chrono::system_clock::now();
                Matrix<ElemType>::SVD(A, S, U, VT, W);
                std::chrono::time_point < std::chrono::system_clock > enTime = std::chrono::system_clock::now();

                // A \in R^{mXn}
                // U \in R^{mXm}
                // VT \in R^{nXn}
                // S \in R^{min(m,n),1}
                // S is in descending order
                //
                ElemType totalenergy = 0.0f;
                for (size_t i = 0; i < S.GetNumRows(); i++)
                    totalenergy += S(i, 0);
                ElemType keepenergy = totalenergy * keepratio;
                ElemType runenergy = 0.0f;

                size_t r = 0;
                for (size_t indx = 0; indx < S.GetNumRows(); indx++)
                {
                    runenergy += S(indx, 0);
                    if (runenergy > keepenergy)
                    {
                        r = indx + 1;
                        break;
                    }
                }

                r = (r + 7) & (~7); //  to keep the number of rows/cols of resultant matrix a multipier of 8
                //  which can be helpful at runtime

                std::chrono::duration<double> elapsedtime = enTime - stTime;
                fprintf(stderr,
                        "Performing SVD for a %5d-by-%-5d matrix (node name: %-20ls) ---  computation time %5.2f secs ;  keep %4.1f%% energy ===> keep %5d svd values (reduce to %4.1f%% parameters) \n",
                        (int) m, (int) n, name.c_str(), elapsedtime.count(),
                        keepratio * 100, (int) r,
                        ((m + n) * r + 0.0f) / m / n * 100);

                // redU in R^ {mXr}
                Matrix<ElemType> redU = U.ColumnSlice(0, r);
                Matrix<ElemType> redVT(-1);

                // redVT in R^{rXn}
                redVT.Resize(r, n);
                redVT.AssignRowSliceValuesOf(VT, 0, r);

                Matrix<ElemType> redS(r, (size_t) 1);
                for (size_t i = 0; i < r; i++)
                {
                    ElemType sqrtsigma = (ElemType) sqrt((double) S(i, 0));
                    redS(i, 0) = sqrtsigma;
                }

                redU.RowElementMultiplyWith(redS.Transpose());
                redVT.ColumnElementMultiplyWith(redS);

                //========================================
                // Step 2. create two new Parameter nodes and one Times node
                //========================================
                wstring LeftChildName = name + L"-U";
                wstring rightChildName = name + L"-V";
                ComputationNodePtr pLeft = Parameter(m, r, LeftChildName);
                ComputationNodePtr pRight = Parameter(r, n, rightChildName);

                pLeft->FunctionValues() = redU;
                pRight->FunctionValues() = redVT;

                ComputationNodePtr pTimes = Times(pLeft, pRight, name + L"-SVD");

                //========================================
                // Step 3. remove old node
                //========================================
                ReplaceLeafNode(name, pTimes);
            }
        }
        RebuildNetwork(m_finalCriteria[0]);
    }

public:
    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    virtual void GetHistory(map<wstring, Matrix<ElemType>>& history, bool bLastTime = false)
    {
        //put all node info first
        Matrix<ElemType> hist;
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodePtr nodePtr = nodeIter->second;
            if (nodePtr->GetHistory(hist, bLastTime))
                history[nodeIter->first] = hist;
        }
    };

    void SetHistory(map<wstring, Matrix<ElemType>>& history)
    {
        //put all node info first
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodePtr nodePtr = nodeIter->second;
            if (history.find(nodeIter->first) != history.end())
            {
                nodePtr->SetHistory(history[nodeIter->first]);
            }
        }
    };

    Matrix<ElemType> & SentenceBoundary() { return m_SentenceBoundary; }

    vector<MinibatchPackingFlag> & MinibatchPackingFlags() { return m_minibatchPackingFlag; }

protected:
    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    // Copy constructor, should never be called.
#pragma warning (push)
#pragma warning (disable: 4702) // this function is flagged but unclear why
    ComputationNetwork(const ComputationNetwork<ElemType>& /*deepCopyFrom*/)
    {
        // TODO: can we just define it as private without implementation?
        LogicError("'ComputationNetwork(const ComputationNetwork<ElemType>& deepCopyFrom)' should never be called.");
    }
#pragma warning (pop)

    // Assignment operator, should never be called.
    ComputationNetwork<ElemType>& operator=(const ComputationNetwork<ElemType>& /*deepCopyFrom*/)
    {
        // TODO: can we just define it as private without implementation?
        LogicError("'ComputationNetwork<ElemType>& operator=(const ComputationNetwork<ElemType>& deepCopyFrom)' should never be called.");
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // The methods below determine evaluation order, which is tricky in presence of recurrent loops.
    // TODO: Can this be moved to a separate class, or at least a separate CPP?

    void ClearCalcOrderCaches()
    {
        for (typename std::map<const ComputationNodePtr, std::list<ComputationNodePtr>>::iterator it = m_cacheEvalOrders.begin(); it != m_cacheEvalOrders.end(); ++it)
            for (auto iter2 = m_cacheEvalOrders[it->first].begin(); iter2 != m_cacheEvalOrders[it->first].end(); iter2++)
                (*iter2)->clearCache();
        m_cacheEvalOrders.clear();
        m_cacheGradientCalcOrders.clear();
    }

    void MergeRecurrentLoops(const ComputationNodePtr /*rootNode*/)
    {
        /// merge loops if they have the same source node
        std::vector<RecurrentInfo> m_recurrentInfoTmp;
                    if (m_recurrentInfo.size() <= 1)
                        return; 

        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            if (m_recurrentInfoTmp.size() == 0)
            {
                RecurrentInfo rInfo;
                            rInfo.Copy(*iter); 
                m_recurrentInfoTmp.push_back(rInfo);
            }
            else
            {
                bool bFound = false;
                for (auto iter2 = m_recurrentInfoTmp.begin(); iter2 != m_recurrentInfoTmp.end(); iter2++)
                {
                    if ((*iter2).m_sourceNode == (*iter).m_sourceNode)
                    {
                        bFound = true;
                        break;
                    }
                }

                if (bFound == false)
                {
                    RecurrentInfo rInfo;
                                rInfo.Copy(*iter);
                    m_recurrentInfoTmp.push_back(rInfo);
                }
                else
                    continue;
            }
        }

        // no need to sort the vector of recurrent loops, because they are pushed and later used as FIFO
        m_recurrentInfo.clear();
        for (auto iter = m_recurrentInfoTmp.begin(); iter != m_recurrentInfoTmp.end(); iter++)
            m_recurrentInfo.push_back(*iter);

        // for debug purposes
        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            fprintf(stderr, " nodes in the recurrent loops : \n");
            for (auto itr = (*iter).m_recurrentNodes.begin(); itr != (*iter).m_recurrentNodes.end(); itr++)
                fprintf(stderr, "%ls\t", (*itr)->NodeName().c_str());
        }
    }

    // get the strong connected component from the graph
    void getStrongSCC(const ComputationNodePtr rootNode)    // TODO: method names start uppercase
    {
                    /// notice that this graph including graphs from a parent networks if two or more networks are connected via pairnetwork node
        std::unordered_set<ComputationNodePtr> visited;
        std::list<ComputationNodePtr> sccStack;
        size_t index = 0;
        size_t loopId = 0;
        if (rootNode->isVisisted() == false)
            strongSCC(rootNode, sccStack, index, loopId);
    }

    void strongSCC(ComputationNodePtr cur,      // TODO: method names start uppercase
                   std::list<ComputationNodePtr>& sccStack,
                   size_t& index, size_t& loopId)
    {
        cur->SetIndex(index);
        cur->Setlowlink(index);
        index++;

        cur->SetVisited(true);
        sccStack.push_back(cur);
        cur->SetInStack(true);

        if (cur->OperationName() != L"PairNetwork")
        {
            // pairnetwork is the socket from other network, so ignore its children, which are in the other networks
            for (int i = 0; i < cur->ChildrenSize(); i++)
            {
                if (cur->Inputs(i)->isVisisted() == false)
                {
                    strongSCC(cur->Inputs(i), sccStack, index, loopId);
                    cur->Setlowlink(min(cur->Getlowlink(), cur->Inputs(i)->Getlowlink()));
                }
                else if (cur->Inputs(i)->isInStack())
                {
                    cur->Setlowlink(min(cur->Getlowlink(), cur->Inputs(i)->Getlowlink()));
                }
            }
        }

        if (cur->Getlowlink() == cur->GetIndex())
        {
            RecurrentInfo rInfo;
            rInfo.m_loopId = loopId;
            rInfo.m_sourceNode = cur;
            size_t sccSize = 0;
            for (;;)
            {
                ComputationNodePtr w = sccStack.back();
                sccStack.pop_back();
                w->SetInStack(false);
                rInfo.m_recurrentNodes.push_back(w);
                sccSize++;
                if (w == cur)
                    break;
            }
            rInfo.Reset();
            if (sccSize > 1)
            {
                loopId++;
                m_recurrentInfo.push_back(rInfo);
            }
        }
    }

    void getLoopForwordOrder(std::unordered_set<ComputationNodePtr>& visited,   // TODO: method name
                             std::unordered_set<ComputationNodePtr>& recStack,
                             std::list<ComputationNodePtr>& nodesStack,
                             ComputationNodePtr cur)
    {
        if (visited.find(cur) == visited.end())
        {
            visited.insert(cur);
            recStack.insert(cur);

            if (cur->OperationName() != PastValueNode<ElemType>::TypeName() && 
                cur->OperationName() != FutureValueNode<ElemType>::TypeName())
            {
                for (size_t i = 0; i < cur->ChildrenSize(); i++)
                    if (cur->Inputs(i)->LoopId() == cur->LoopId())
                        getLoopForwordOrder(visited, recStack, nodesStack, cur->Inputs(i));
            }
            recStack.erase(cur);
            nodesStack.push_back(cur);
        }
        else
        {
            if (!(recStack.find(cur) == recStack.end()))
                LogicError("There is infinite Loop which cannot be unrolled!!");
        }
    }
            
    //must be called before ValidateNetwork
    void FormRecurrentLoops(const ComputationNodePtr rootNode)
    {
        std::vector<ComputationNodePtr> sourceLoopNodes;

                    getStrongSCC(rootNode);
        std::list<ComputationNodePtr>& nodes = GetEvalOrder(rootNode, sourceLoopNodes);
        std::list<ComputationNodePtr> nodesForGrad;

                    MergeRecurrentLoops(rootNode);

        /// debug purpose
        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            fprintf(stderr, " nodes in the recurrent loops : \n");
            size_t max_visitedOrderInLoop = 0;
            for (auto itr = (*iter).m_recurrentNodes.begin(); itr != (*iter).m_recurrentNodes.end(); itr++)
            {
                fprintf(stderr, "%ls\t", (*itr)->NodeName().c_str());
                if (max_visitedOrderInLoop < (*itr)->GetVisitedOrder())
                    max_visitedOrderInLoop = (*itr)->GetVisitedOrder();
            }
            for (auto itr = (*iter).m_recurrentNodes.begin(); itr != (*iter).m_recurrentNodes.end(); itr++)
                (*itr)->SetVisitedOrder(max_visitedOrderInLoop);
        }

        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            // sort the recurrent nodes in their ascending name, which is the same as visiting nodes in G^R
            if ((*iter).m_recurrentNodes.size() > 1)
            {
                /// it is done in the mergerecurrentloops function, but just keep the code
                std::sort((*iter).m_recurrentNodes.begin(),
                          (*iter).m_recurrentNodes.end(),
                          (*iter).m_recurrentNodes[0]->IsSmaller);

                for (auto nodeRecIter = (*iter).m_recurrentNodes.begin(); nodeRecIter != (*iter).m_recurrentNodes.end(); nodeRecIter++)
                {
                    (*nodeRecIter)->SetLoop(true);
                    (*nodeRecIter)->SetLoopId((*iter).m_loopId);
                }
            }
        }

        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            // sort the recurrent nodes in their ascending name, which is the same as visiting nodes in G^R
            (*iter).m_recurrentNodesForForward.clear();
            if ((*iter).m_recurrentNodes.size() > 1)
            {
                std::list<ComputationNodePtr> result;
                std::unordered_set<ComputationNodePtr> visited;
                std::unordered_set<ComputationNodePtr> recStack;

                for (size_t j = 0; j < (*iter).m_recurrentNodes.size(); j++)
                {
                    ComputationNodePtr nodeRecIter = (*iter).m_recurrentNodes[j];
                    for (size_t i = 0; i < nodeRecIter->ChildrenSize(); i++)
                    {
                        if (nodeRecIter->Inputs(i)->LoopId() == nodeRecIter->LoopId() && 
                            nodeRecIter->OperationName() != PastValueNode<ElemType>::TypeName() &&
                            nodeRecIter->OperationName() != FutureValueNode<ElemType>::TypeName())
                        {
                            nodeRecIter->Inputs(i)->SetIndexInLoop(nodeRecIter->Inputs(i)->GetIndexInLoop() + 1);
                        }
                    }
                }

                //for (auto nodeRecIter = startNodes.begin(); nodeRecIter != startNodes.end(); nodeRecIter++)

                for (size_t i = 0; i < (*iter).m_recurrentNodes.size(); i++)
                {
                    ComputationNodePtr nodeRecIter = (*iter).m_recurrentNodes[i];
                    if (visited.find(nodeRecIter) == visited.end() && nodeRecIter->GetIndexInLoop() == 0)
                        getLoopForwordOrder(visited, recStack, result, nodeRecIter);
                }

                for (size_t i = 0; i < (*iter).m_recurrentNodes.size(); i++)
                {
                    (*iter).m_recurrentNodesForForward.push_back(result.front());
                    result.pop_front();
                }

                (*iter).m_recurrentNodes = (*iter).m_recurrentNodesForForward;
            }
        }

        if (m_recurrentInfo.size() > 0)
        {
            std::map<int, std::list<ComputationNodePtr>> recurrentNodes;
            std::list<ComputationNodePtr> noRecurrentNodes;

            noRecurrentNodes = rootNode->ReshuffleNodes(recurrentNodes);

            nodes.sort(IsSmaller);

            ReorderLoops(nodes, recurrentNodes, noRecurrentNodes);

            m_cacheEvalOrders[rootNode] = nodes;
            nodesForGrad = nodes;
            nodesForGrad.reverse();
            m_cacheGradientCalcOrders[rootNode] = nodesForGrad;

#ifdef DISPLAY_DEBUG
            fprintf(stderr, "Reordered nodes\n");
            for (auto itr = nodes.begin(); itr != nodes.end(); itr++)
            {
                fprintf (stderr, "%ls\n", (*itr)->NodeName().c_str() );
            }
#endif
        }
        
        DetermineLoopTypes();
        
        for (auto iter = nodes.begin(); iter != nodes.end(); iter++)
            (*iter)->clearCache();
    }

    void DetermineLoopTypes()
    {
        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            bool hasPastValueNode = false;
            bool hasFutureValueNode = false;

            RecurrentInfo* recurrentInfo = &(*iter);

            if (recurrentInfo->m_recurrentNodes.size() > 0)
            {
                for (size_t j = 0; j < recurrentInfo->m_recurrentNodes.size(); j++)
                {
                    ComputationNodePtr nodeRecIter = recurrentInfo->m_recurrentNodes[j];

                    if (nodeRecIter->OperationName() == PastValueNode<ElemType>::TypeName())
                    {
                        hasPastValueNode = true;
                    }
                    else if (nodeRecIter->OperationName() == FutureValueNode<ElemType>::TypeName())
                    {
                        hasFutureValueNode = true;
                    }
                }

                if (hasPastValueNode && hasFutureValueNode)
                {
                    RuntimeError("It is not allowed to have both PastValue and FutureValue nodes in the same loop.");
                }
                else if (!hasPastValueNode && !hasFutureValueNode)
                {
                    RuntimeError("There is neither PastValue nor FutureValue nodes in the loop.");
                }
                else if (hasPastValueNode)
                {
                    recurrentInfo->m_isForwardLoop = true;
                }
                else
                {
                    recurrentInfo->m_isForwardLoop = false;
                }
            }
        }
    }

    void ReorderLoops(std::list<ComputationNodePtr>& nodes,
                      const std::map<int, std::list<ComputationNodePtr>>& /*recurrentNodes*/,
                      const std::list<ComputationNodePtr> & /*noRecurrentNodes*/)
    {
        std::list<ComputationNodePtr> newList;

        std::list<ComputationNodePtr> vTmp;
        std::list<ComputationNodePtr> vRecurrentTmp;
        //int  prevId = -1;
        vector<bool> accessed;
        accessed.assign(m_recurrentInfo.size(), false);
        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            int iId = FindInRecurrentLoop(*nodeIter);
            if (iId >= 0)
            {

                if (!accessed[iId])
                {
                    newList.insert(newList.end(),
                                   m_recurrentInfo[iId].m_recurrentNodes.begin(),
                                   m_recurrentInfo[iId].m_recurrentNodes.end());
                    accessed[iId] = true;
                }
            }
            else
            {
                newList.push_back(*nodeIter);
            }
        }

        if (vRecurrentTmp.size() > 0)
        {
            newList.insert(newList.end(), vRecurrentTmp.begin(), vRecurrentTmp.end());
            vRecurrentTmp.clear();
        }

        if (vTmp.size() > 0)
        {
            newList.insert(newList.end(), vTmp.begin(), vTmp.end());
            vTmp.clear();
        }

        nodes = newList;
    }

    void CollectInputAndLeanableParameters(const ComputationNodePtr rootNode)
    {
        //not found
        if (m_inputs.find(rootNode) == m_inputs.end())
        {
            std::list<ComputationNodePtr> inputs;

            std::list<ComputationNodePtr>& nodes = GetEvalOrder(rootNode);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end();
                    nodeIter++)
            {
                ComputationNodePtr node = (*nodeIter);
                if (node->OperationName() == InputValue<ElemType>::TypeName() /*L"InputValue"*/ ||
                    node->OperationName() == InputValue<ElemType>::SparseTypeName())
                {
                    inputs.push_back(node);
                }
            }
            m_inputs[rootNode] = inputs;
        }

        //not found
        if (m_learnableParameters.find(rootNode) == m_learnableParameters.end())
        {
            std::list<std::wstring> learnableParameterNames;
            std::list<ComputationNodePtr> learnableParameters;

            std::list<ComputationNodePtr>& nodes = GetEvalOrder(rootNode);
            ;
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodePtr node = (*nodeIter);
                if ((node->OperationName() == LearnableParameter<ElemType>::TypeName() && node->NeedGradient()) ||
                    (node->OperationName() == SparseLearnableParameter<ElemType>::TypeName() && node->NeedGradient()))
                {
                    learnableParameterNames.push_back(node->NodeName());
                }
            }

            //we need to sort it so that we get consistent order when load it from saved file
            learnableParameterNames.sort();
            for (auto nodeNameIter = learnableParameterNames.begin(); nodeNameIter != learnableParameterNames.end(); nodeNameIter++)
            {
                learnableParameters.push_back(GetNodeFromName((*nodeNameIter)));
            }

            m_learnableParameters[rootNode] = learnableParameters;
        }
    }

    // -----------------------------------------------------------------------
    // node creation
    // -----------------------------------------------------------------------

    // TODO: move these close to where they are used

    // add a node to m_nameToNodeMap[], which is our node holder
    // Duplicate node names are rejected.
    ComputationNodePtr AddNodeToNet(const ComputationNodePtr nodePtr)
    {
        //found
        // TODO: use .insert() and test result.second == false means not inserted since already exists
        if (m_nameToNodeMap.find(nodePtr->NodeName()) != m_nameToNodeMap.end())
            RuntimeError("Duplicated computation node name.");

        m_nameToNodeMap[nodePtr->NodeName()] = nodePtr;
        return nodePtr; // allows e.g. return AddNodeToNet(New...);
    }

    template<class... _Types>
    ComputationNodePtr AddNodeToNetAndAttachInputs(const ComputationNodePtr nodePtr, _Types&&... _Args)
    {
        nodePtr->AttachInputs(std::forward<_Types>(_Args)...);
        AddNodeToNet(nodePtr);
        return nodePtr; // allows e.g. return AddNodeToNetAndAttachInputs(New..., inputs);
    }

public:

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    void ClearGradientForAllNodes(const ComputationNodePtr rootNode)
    {
        std::list<ComputationNodePtr>& allNodes = GetGradientCalcOrder(rootNode);

        for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
            (*nodeIter)->ClearGradientForChildren(m_actMiniBSize);

        //for (auto nodeIter = m_recurrentInfo.begin(); nodeIter != m_recurrentInfo.end(); nodeIter++)
        //    (*nodeIter).m_completedGradient = false;

        for (int i = 0; i < m_recurrentInfo.size(); i++)
            m_recurrentInfo[i].m_completedGradient = false;
    }

    std::list<ComputationNodePtr>& GetEvalOrder(const ComputationNodePtr rootNode)
    {
        if (!rootNode)
            LogicError("rootNode is pointing to a nullptr.");

        return GetCalcOrder(rootNode, m_cacheEvalOrders, true);
    }

    std::list<ComputationNodePtr>& GetEvalOrder(const ComputationNodePtr rootNode,
                                                std::vector<ComputationNodePtr>& recurrentNodes)
    {
        if (!rootNode)
            LogicError("rootNode is pointing to a nullptr.");

        return GetCalcOrder(rootNode, m_cacheEvalOrders, true, recurrentNodes);
    }

    std::list<ComputationNodePtr>& GetGradientCalcOrder(const ComputationNodePtr rootNode)
    {
        if (!rootNode)
            LogicError("rootNode is pointing to a nullptr.");

        return GetCalcOrder(rootNode, m_cacheGradientCalcOrders, false);
    }

protected:

    std::list<ComputationNodePtr>& GetCalcOrder(const ComputationNodePtr rootNode,
                                                std::map<const ComputationNodePtr, std::list<ComputationNodePtr>>& orderMap,
                                                const bool forwardCompute)
    {
        const ComputationNodePtr key = rootNode;

        //not found
        if (orderMap.find(key) == orderMap.end())
            orderMap[key] = rootNode->EnumerateNodes(forwardCompute);

        return orderMap[key];
    }

    std::list<ComputationNodePtr>& GetCalcOrder(const ComputationNodePtr rootNode,
                                                std::map<const ComputationNodePtr, std::list<ComputationNodePtr>>& orderMap,
                                                const bool forwardCompute,
                                                std::vector<ComputationNodePtr> & rootRecurrentNodes)
    {
        const ComputationNodePtr key = rootNode;
        std::list<ComputationNodePtr> listNodes;

        //not found
        if (orderMap.find(key) == orderMap.end())
        {
            rootRecurrentNodes.clear();
            listNodes = rootNode->EnumerateNodes(forwardCompute, rootRecurrentNodes);

            orderMap[key] = listNodes;

        }
        return orderMap[key];
    }

public:

    // -----------------------------------------------------------------------
    // BS integration
    // -----------------------------------------------------------------------

    // create a somewhat readable representation, aimed at diagnostics/debugging
    wstring /*HasToString::*/ToString() const
    {
        wstring args;
        for (auto & iter : m_nameToNodeMap)
        {
            const auto node = iter.second;
            if (!args.empty())
                args.append(L"\n");
            args.append(node->ToString());
        }
        return TypeId<decltype(*this)>() + L" " + NestString(args, L'[', true, ']');
    }

    // pretending to be a ConfigRecord. TODO: implement this when we actually need it (when we get to MEL)
    const BS::ConfigValuePtr & /*IConfigRecord::*/operator()(const wstring & id, wstring message) const   // e.g. confRec(L"message", helpString)
    {
        id; message; RuntimeError("unknown class parameter");    // (for now)
    }
    const BS::ConfigValuePtr * /*IConfigRecord::*/Find(const wstring & id) const         // returns nullptr if not found
    {
        id; return nullptr; // (for now)
    }
    vector<wstring> /*IConfigRecord::*/GetMemberIds() const
    {
        return vector<wstring>();
    }

protected:

    // -----------------------------------------------------------------------
    // data members
    // -----------------------------------------------------------------------

    // TODO: move basic accessors in here?

    DEVICEID_TYPE m_deviceId;           // TODO: is this shared by all nodes?
    unsigned long m_randomSeedOffset;

    // node groups
    std::vector<ComputationNodePtr> m_features;
    std::vector<ComputationNodePtr> m_labels;
    std::vector<ComputationNodePtr> m_finalCriteria;
    std::vector<ComputationNodePtr> m_evalNodes;
    std::vector<ComputationNodePtr> m_outputNodes;
    std::vector<ComputationNodePtr> m_pairNodes; /// nodes for the children network to pair
    std::vector<ComputationNodePtr> m_nodesReqMultiSeqHandling;
    vector<std::vector<ComputationNodePtr>*> GetAllNodeGroups()    // get all groups to allow to iterate over all of them ...continue
    {
        return vector<std::vector<ComputationNodePtr>*> { &m_features, &m_labels, &m_finalCriteria, &m_evalNodes, &m_outputNodes, &m_pairNodes, &m_nodesReqMultiSeqHandling };
    }

    std::vector<RecurrentInfo> m_recurrentInfo;

    /** temporary space
    */
    std::vector<ComputationNodePtr> m_tmpTrainCriterion; /// array saving tempary query terms
    std::vector<ComputationNodePtr> m_tmpEvalulationCriterion; /// array saving tempary query terms

    //used for sentence boundary information passed from reader to reset RNN state 
    Matrix<ElemType> m_SentenceBoundary; // this matrix is always in CPU memory
    // specify how the minibatch is packed for each sample
    vector<MinibatchPackingFlag> m_minibatchPackingFlag;

    int m_actMiniBSize;
    size_t m_nbrSlicesInEachRecurrentIteration;

    std::map<const ComputationNodePtr, bool> m_built;
    std::map<const std::wstring, ComputationNodePtr, nocase_compare> m_nameToNodeMap;   // this is the main container that holds this networks' nodes

    std::map<const ComputationNodePtr, std::list<ComputationNodePtr>> m_cacheEvalOrders;
    std::map<const ComputationNodePtr, std::list<ComputationNodePtr>> m_cacheGradientCalcOrders;

    std::map<const ComputationNodePtr, std::list<ComputationNodePtr>> m_inputs;
    std::map<const ComputationNodePtr, std::list<ComputationNodePtr>> m_learnableParameters;

    MatrixPool<ElemType> m_matrixPool;
};

template class ComputationNetwork<float>;
template class ComputationNetwork<double>;

}}}
