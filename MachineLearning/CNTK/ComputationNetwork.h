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

namespace Microsoft { namespace MSR { namespace CNTK {
    template<class ElemType>
    class ComputationNetwork
    {
    protected:
        typedef ComputationNode<ElemType>* ComputationNodePtr;
		typedef std::pair<ComputationNodePtr, ComputationNodePtr> ComputationArc;
        typedef struct stRecurrentInfo{
            std::vector<ComputationNodePtr> m_recurrentNodes;
            std::vector<ComputationNodePtr> m_recurrentNodesForForward;
            ComputationNodePtr    m_sourceNode;
            int m_loopId;
            bool m_completedGradient;
            bool m_completedEvaluate;
            bool m_loopClosed;

            void Reset()
            {
                m_completedGradient = false; 
                m_completedEvaluate = false;
                m_loopClosed = false; 
            }
        } RecurrentInfo;

    public:
        ComputationNetwork(DEVICEID_TYPE deviceId = AUTOPLACEMATRIX) : m_deviceId(deviceId), mSentenceBoundary(deviceId), mExistsBeginOrNoLabels(deviceId)
        {
            m_randomSeedOffset = 0;
            m_actMiniBSize = 0;
            if (m_deviceId == AUTOPLACEMATRIX)
                m_deviceId = Matrix<ElemType>::GetBestGPUDeviceId();
            m_nbrSlicesInEachRecurrentIteration = 1; 
        }

        virtual ~ComputationNetwork()
        {
            ClearNet();
        }

        static bool IsSmaller(const ComputationNodePtr lhs, const ComputationNodePtr rhs) 
        { 
            return lhs->GetVisitedOrder() < rhs->GetVisitedOrder();
        }

        void ClearNet()
        {
            m_features.clear();
            m_labels.clear();
            m_finalCriteria.clear();
            m_nodesReqMultiSeqHandling.clear();
            m_evalNodes.clear();
            m_outputNodes.clear();
            m_recurrentInfo.clear();

            m_built.clear();

            m_cacheEvalOrders.clear();
            m_cacheGradientCalcOrders.clear();

            m_inputs.clear();
            m_learnableParameters.clear();

            for (auto nodeIter=m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
                delete nodeIter->second;      
            m_nameToNodeMap.clear();
        }

        //if node name is not found, dump all nodes
        //otherwise dump just that node
        void DumpNodeInfoToFile(const std::wstring nodeName, const bool printValues, const std::wstring outputFile)
        {
            if (NodeNameExist(nodeName))
            {
                ValidateNetwork(true);  //some internal values in the nodes are computed during validation

                File fstream(outputFile, FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

                const ComputationNodePtr nodePtr = GetNodeFromName(nodeName);
                nodePtr->DumpNodeInfo(printValues, fstream);
            }
            else  //node name is not found, dump all nodes
            {
                fprintf (stderr, "Warning: node name %ls does not exist in the network. dumping all nodes.\n", nodeName.c_str());
                DumpAllNodesToFile(printValues, outputFile);
            }
        }

        //dump all nodes in the network to file
        void DumpAllNodesToFile(const bool printValues, const std::wstring outputFile, const bool validateBeforeDump = true)
        {
            if (validateBeforeDump)
                ValidateNetwork();  //some internal values in the nodes are computed during validation

            File fstream(outputFile, FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

            for (auto nodeIter=m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodePtr nodePtr = nodeIter->second;
                nodePtr->DumpNodeInfo(printValues, fstream);
            }
        }

        void DumpNodeInfoToFile(const vector<ComputationNode<ElemType>*>& nodes, const bool printValues, const std::wstring outputFile)
        {
            ValidateNetwork();  //some internal values in the nodes are computed during validation

            File fstream(outputFile, FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

            for (auto nodeIter=nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodePtr nodePtr = *nodeIter;
                nodePtr->DumpNodeInfo(printValues, fstream);
            }
        }


private:	// [erw] added for Toplological Plot only
		class DotGraphConfigure
		{
		public: 
			wstring m_LearnableParameterStyle ; 
			wstring m_featuresStyle; 
			wstring m_CriteriaStyle;
            wstring m_nodesReqMultiSeqHandlingStyle;
			wstring m_labelsStyle; 
			wstring m_normalNodeStyle; 
			wstring m_PrecomputingNodeStyle;
			wstring m_DelayNodeStyle;

			DotGraphConfigure()
			{
				m_LearnableParameterStyle	= L"node [ shape = box     , color = gray , style = \"filled, rounded\"  ]; "; 
				m_featuresStyle				= L"node [ shape = ellipse , color = red  , fillcolor = white ]; "; 
				m_CriteriaStyle				= L"node [ shape = doublecircle , color =  red , fillcolor = white  ]; ";
                m_nodesReqMultiSeqHandlingStyle = L"node [ shape = doublecircle , color =  brown , fillcolor = white  ]; ";
                m_normalNodeStyle = L"node [ shape = ellipse, color = blue, fillcolor = white, style = solid ]; ";
				m_PrecomputingNodeStyle		= L"node [ shape = box    , color = black, style = \"dashed, filled\",  fillcolor= limegreen ] ;";
				m_labelsStyle				= L"node [ shape = diamond, color = brown, style = bold ] ;  ";
				m_DelayNodeStyle			= L"node [ shape = box3d  , color = lightgray, style = \"filled\" , fillcolor = white ] ";
			}
		};
		wstring FormSpecialNodes(wstring style, std::vector<ComputationNodePtr>& specialNodes)
		{
            if (specialNodes.empty())
            {
                return L"";
            }
			wstring str = style; 
			for (auto x : specialNodes){
				str = str + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
			}
			return str + L"; \n";
		}
public:


		void DescribeNetworkUsingDot(std::list<ComputationArc>& arcs, std::wstring outFile, DotGraphConfigure dotcfg = DotGraphConfigure())
		{
			File fstream(outFile, FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);
			wstring line;

			// get precompute node 
			std::vector<ComputationNodePtr>	PreComputedNodes;
			std::vector<ComputationNodePtr>	allnodes = GetAllNodes();
			for (auto n : allnodes)
			{
				if (n->RequirePreCompute())
				{
					PreComputedNodes.push_back(n);
				}
			}
			// get delay node 
			std::vector<ComputationNodePtr> DelayNodes; 
			for (auto n : allnodes)
			{
				if (n->OperationName() == L"Delay")
				{
					DelayNodes.push_back(n);
				}
			}
			// get learnableParameters 
			std::vector<ComputationNodePtr> learnableParameters; 
			for (auto n : allnodes)
			{
				if (n->OperationName() == L"LearnableParameter")
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
			// delay nodes 
			fstream << FormSpecialNodes(dotcfg.m_DelayNodeStyle, DelayNodes);
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
                    x->GetName().c_str(), x->GetName().c_str(), nrows, ncols,  x->OperationName().c_str());
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
			{
				line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
			}
            for (auto x : m_nodesReqMultiSeqHandling)
            {
                line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
            }
            for (auto x : m_outputNodes)
			{
				line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
			}
			for (auto x : m_evalNodes)
			{
				line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
			}
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
				

				if (des->OperationName() == L"Delay")
				{
					// special treament for arc with Delay node as the children 
					// create a dummy node 
					ComputationNodePtr delayedNode = des;
					wstring dummyName = des->GetName() + L".dummy";
                    wstring out = msra::strfun::wstrprintf(L"node [ shape = box3d  , color = lightgray, style = \"filled\" , label = \"%ls\" ] ; \"%ls\"\n",
						(delayedNode->GetName() + L"\\n(delayed)").c_str(), dummyName.c_str());
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
			std::unordered_set<ComputationNodePtr>	visited; 
			std::list<ComputationArc>		arcs;

			for (size_t i = 0; i < m_finalCriteria.size(); i++)
			{
				m_finalCriteria[i]->EnumerateArcs(visited, arcs);
			}
            for (size_t i = 0; i < m_nodesReqMultiSeqHandling.size(); i++)
            {
                m_nodesReqMultiSeqHandling[i]->EnumerateArcs(visited, arcs);
            }
            for (size_t i = 0; i < m_outputNodes.size(); i++)
			{
				m_outputNodes[i]->EnumerateArcs(visited, arcs); 
			}
			for (size_t i = 0; i < m_evalNodes.size(); i++)
			{
				m_evalNodes[i]->EnumerateArcs(visited, arcs);
			}

			//////////////////////////////////////////////////////////////////////////
			//	step 2.		output dot description
			//////////////////////////////////////////////////////////////////////////
			DescribeNetworkUsingDot(arcs, outputFile);

		}

        void SetDeviceID(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX)
        {
            m_deviceId = deviceId;  
            if (m_deviceId == AUTOPLACEMATRIX)
                m_deviceId = Matrix<ElemType>::GetBestGPUDeviceId();
        }

        DEVICEID_TYPE GetDeviceID() {return m_deviceId;}
        unsigned long GetRandomSeedOffset() {return m_randomSeedOffset;}
        void SetRandomSeedOffset(unsigned long value) {m_randomSeedOffset = value;}

        void SaveToFile(const std::wstring& fileName, const FileOptions fileFormat = FileOptions::fileOptionsBinary) const
        {
            File fstream(fileName, fileFormat | FileOptions::fileOptionsWrite);
            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BCN");

            //model version
            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BVersion");
            fstream << (size_t)CURRENT_CNTK_MODEL_VERSION;
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EVersion");

            fstream << (size_t)m_nameToNodeMap.size();

            //put all node info first
            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BNodeList");
            for (auto nodeIter=m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodePtr nodePtr = nodeIter->second;
                nodePtr->SaveToFile(fstream); 
            }
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ENodeList");

            //put relationship
            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BRelation");
            for (auto nodeIter=m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodePtr nodePtr = nodeIter->second;
                fstream << nodePtr->NodeName() << nodePtr->ChildrenSize();
                for (size_t i=0; i<nodePtr->ChildrenSize(); i++)
                {
					if (nodePtr->Inputs(i) == nullptr)
					{
						fprintf(stderr, "Warning: node %ls 's child is null, please check your ndl/mel file.\n", nodePtr->NodeName().c_str());
					}
					else
					{
						fstream << nodePtr->Inputs(i)->NodeName();
					}
                }
            }
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ERelation");

            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BRootNodes");

            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BFeatureNodes");
            fstream << m_features.size();
            for (size_t i=0; i<m_features.size(); i++)
            {
                fstream << m_features[i]->NodeName();
            }
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EFeatureNodes");

            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BLabelNodes");
            fstream << m_labels.size();
            for (size_t i=0; i<m_labels.size(); i++)
            {
                fstream << m_labels[i]->NodeName();
            }
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ELabelNodes");

            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BCriteriaNodes");
            fstream << m_finalCriteria.size();
            for (size_t i=0; i<m_finalCriteria.size(); i++)
            {
                fstream << m_finalCriteria[i]->NodeName();
            }
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ECriteriaNodes");

            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BNodesReqMultiSeqHandling");
            fstream << m_nodesReqMultiSeqHandling.size();
            for (size_t i = 0; i<m_nodesReqMultiSeqHandling.size(); i++)
            {
                fstream << m_nodesReqMultiSeqHandling[i]->NodeName();
            }
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ENodesReqMultiSeqHandling");

            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BEvalNodes");
            fstream << m_evalNodes.size();
            for (size_t i=0; i<m_evalNodes.size(); i++)
            {
                fstream << m_evalNodes[i]->NodeName();
            }
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EEvalNodes");

            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BOutputNodes");
            fstream << m_outputNodes.size();
            for (size_t i=0; i<m_outputNodes.size(); i++)
            {
                fstream << m_outputNodes[i]->NodeName();
            }
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EOutputNodes");

            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ERootNodes");

            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ECN");
        }

        void LoadPersistableParametersFromFile(const std::wstring& fileName, const bool requireValidation = true, const FileOptions fileFormat = FileOptions::fileOptionsBinary)
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
            for (size_t i=0; i<numNodes; i++)
            {
                std::wstring opName, nodeName;
                fstream >> opName >> nodeName;
                ComputationNodePtr nodePtr = GetNodeFromName(nodeName);
                nodePtr->LoadFromFile(fstream, modelVersion, m_deviceId);
            }
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ENodeList");

            size_t actualMBSize = GetActualMBSize();
            SetActualMiniBatchSize(actualMBSize);
            
            if (requireValidation)
                ValidateNetwork();
        }


        size_t GetActualMBSize()
        {
            size_t actualMBSize = 0;

            std::vector<ComputationNodePtr> featureNodes = FeatureNodes();
            for (auto nodeIter=featureNodes.begin(); nodeIter != featureNodes.end(); nodeIter++)
            {
                actualMBSize = max(actualMBSize, ((*nodeIter)->FunctionValues()).GetNumCols());
            }

            return actualMBSize;
        }

        virtual void LoadFromFile(const std::wstring& fileName, const FileOptions fileFormat = FileOptions::fileOptionsBinary, 
            const bool bAllowNoCriterionNode = false, ComputationNetwork<ElemType>* anotherNetwork=nullptr)
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
            for (size_t i=0; i<numNodes; i++)
            {
                std::wstring opName, nodeName;
                fstream >> opName >> nodeName;
                
                CreateNodeFromFile(opName, nodeName, fstream, modelVersion);
            }
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ENodeList");

            //put relationship
            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BRelation");
            for (size_t i=0; i<numNodes; i++)
            {
                std::wstring nodeName;
                size_t numChildren;
                fstream >> nodeName >> numChildren;
                if (numChildren > 0)
                {
                    std::vector<std::wstring> childrenNames;
                    childrenNames.resize(numChildren);
                    for (size_t j=0; j<numChildren; j++)
                    {
                        fstream >> childrenNames[j];
                    }

                    ComputationNodePtr nodePtr = GetNodeFromName(nodeName);
                    std::vector<ComputationNodePtr> childrenNodes;
                    childrenNodes.resize(numChildren);
                    for (int j = 0; j < numChildren; j++)
                        childrenNodes[j] = GetNodeFromName(childrenNames[j], anotherNetwork);

                    if (nodePtr->OperationName() == RowStackNode<ElemType>::TypeName()) //allow for variable input nodes
                        nodePtr->AttachInputs(childrenNodes);
                    else //fixed input nodes
                    {
                        switch (numChildren)
                        {
                        case 1:
                            nodePtr->AttachInputs(childrenNodes[0]);
                            break;
                        case 2:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1]);
                            break;
                        case 3:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1], childrenNodes[2]);
                            break;
                        case 4:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1], childrenNodes[2], childrenNodes[3]);
                            break;
                        case 5:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1], childrenNodes[2], childrenNodes[3], childrenNodes[4]);
                            break;
                        case 6:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1], childrenNodes[2], childrenNodes[3], childrenNodes[4], childrenNodes[5]);
                            break;
                        default:
                            throw std::logic_error("Invalid number of children.");
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
                for (size_t i=0; i<num; i++)
                {
                    fstream >> nodeName;
                    m_features.push_back(GetNodeFromName(nodeName));
                }
                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EFeatureNodes");

                fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BLabelNodes");
                fstream >> num;
                for (size_t i=0; i<num; i++)
                {
                    fstream >> nodeName;
                    m_labels.push_back(GetNodeFromName(nodeName));
                }
                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ELabelNodes");

                fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BCriteriaNodes");
                fstream >> num;
                for (size_t i=0; i<num; i++)
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
                for (size_t i=0; i<num; i++)
                {
                    fstream >> nodeName;
                    m_evalNodes.push_back(GetNodeFromName(nodeName));
                }
                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EEvalNodes");

                fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BOutputNodes");
                fstream >> num;
                for (size_t i=0; i<num; i++)
                {
                    fstream >> nodeName;
                    m_outputNodes.push_back(GetNodeFromName(nodeName));
                }
                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EOutputNodes");

            }

            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ERootNodes");

            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ECN");
            

            ValidateNetwork(false, bAllowNoCriterionNode);  //some internal values in the nodes are computed during validation

        }

#pragma region Network Modification

        void SetLeanableNodesBelowNeedGradient(const bool needGradient, const ComputationNodePtr rootNode = nullptr)
        {
            if (rootNode == nullptr) //find nodes from all available nodes
            {
                for (auto nodeIter=m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
                {
                    ComputationNodePtr node = nodeIter->second;
                    if (node->OperationName() == LearnableParameter<ElemType>::TypeName())
                    {
                        node->NeedGradient() = needGradient;
                    }
                }
            }
            else //for calculating a specific node
            {
                std::list<ComputationNodePtr>&  nodes = GetEvalOrder(rootNode);
                for (auto nodeIter=nodes.begin(); nodeIter != nodes.end(); nodeIter++)
                {
                    ComputationNodePtr node =  (*nodeIter);
                    if (node->OperationName() == LearnableParameter<ElemType>::TypeName())
                    {
                        node->NeedGradient() = needGradient;
                    }
                }
            }
        }

        // Read a matrix stored in text format from 'filePath' (whitespace-separated columns, newline-separated rows),
        // and return a flat array containing the contents of this file in column-major format.
        // filePath: path to file containing matrix in text format.
        // numRows/numCols: after this function is called, these parameters contain the number of rows/columns in the matrix.
        // returns: a flat array containing the contents of this file in column-major format
        // NOTE: caller is responsible for deleting the returned buffer once it is finished using it.
        ElemType* LoadArrayFromTextFile(const std::string filePath, size_t& numRows, size_t& numCols)
        {
            size_t r = 0;
            size_t numColsInFirstRow = 0;

            // NOTE: Not using the Microsoft.MSR.CNTK.File API here because it uses a buffer of fixed size, which doesn't allow very long rows.
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
                    while(iss >> element)
                    {
                        elements[r].push_back(element);
                        numElementsInRow++;
                    }
                
                    if (r == 0)
                        numColsInFirstRow = numElementsInRow;
                    else if (numElementsInRow != numColsInFirstRow)
                        throw std::runtime_error("The rows in the provided file do not all have the same number of columns: " + filePath);

                    r++;
                }
                myfile.close();
            }
            else
                throw std::runtime_error("Unable to open file");

            numRows = r;
            numCols = numColsInFirstRow;

            ElemType* pArray = new ElemType[numRows * numCols];

            // Perform transpose when copying elements from vectors to ElemType[],
            // in order to store in column-major format.
            for(int i = 0; i < numCols; i++)
                for(int j = 0; j < numRows; j++)
                    pArray[i * numRows + j] = elements[j][i];

            return pArray;
        }

        void InitLearnableParametersFromFile(const ComputationNodePtr node, const std::string initFromFilePath)
        {
            size_t numRows = 0;
            size_t numCols = 0;
            ElemType *pArray = LoadArrayFromTextFile(initFromFilePath, numRows, numCols);
            node->FunctionValues().SetValue(numRows, numCols, pArray, matrixFlagNormal, this->GetDeviceID());
            delete[] pArray;
        }
        
        void InitLearnableParameters(const ComputationNodePtr node, const bool uniformInit,  const unsigned long randomSeed, const ElemType initValueScale)
        {
            size_t inputSize = node->FunctionValues().GetNumCols();

            // the random seed offset is set via the "randomSeedOffset" parameter in config
            if (uniformInit)
            {
                ElemType randRange = 0.05f * initValueScale; //initValueScale/sqrt(inputSize);
                node->FunctionValues().SetUniformRandomValue(-randRange, randRange, GetRandomSeedOffset() + randomSeed);
            }
            else
            {
                ElemType randInitstd = 0.2f * initValueScale/sqrt(ElemType(inputSize));
                node->FunctionValues().SetGaussianRandomValue(0,randInitstd, GetRandomSeedOffset() + randomSeed);
            }
        }

        void DeleteNode(const std::wstring nodeName)
        {
            ClearCaches(); //so that deleted node will not be referenced

            ComputationNodePtr nodeToDelete = GetNodeFromName(nodeName);

            //first delete links, if this node is involved, the whole connection will be removed
            for (auto nodeIter=m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodePtr node = nodeIter->second;
                for (size_t i=0; i<node->ChildrenSize(); i++)
                {
                    ComputationNodePtr child = node->Inputs(i);
                    if (child == nodeToDelete)  //nodeToDelete is a child
                    {
                        // this used to call DetatchInputs(), but it's better for MEL to retain other inputs
                        node->SetInput(i,NULL);
                        break;
                    }
                }
            }

            nodeToDelete->DetachInputs(); //nodeToDelete is a parent
			auto search = std::find(m_labels.begin(), m_labels.end(), nodeToDelete);
			if (search != m_labels.end())
			{
				m_labels.erase(search);
			}
			search = std::find(m_features.begin(), m_features.end(), nodeToDelete);
			if (search != m_features.end())
			{
				m_features.erase(search);
			}
			search = std::find(m_finalCriteria.begin(), m_finalCriteria.end(), nodeToDelete);
			if (search != m_finalCriteria.end())
			{
				m_finalCriteria.erase(search);
			}
            search = std::find(m_nodesReqMultiSeqHandling.begin(), m_nodesReqMultiSeqHandling.end(), nodeToDelete);
            if (search != m_nodesReqMultiSeqHandling.end())
            {
                m_nodesReqMultiSeqHandling.erase(search);
            }
            search = std::find(m_evalNodes.begin(), m_evalNodes.end(), nodeToDelete);
			if (search != m_evalNodes.end())
			{
				m_evalNodes.erase(search);
			}

			search = std::find(m_outputNodes.begin(), m_outputNodes.end(), nodeToDelete);
			if (search != m_outputNodes.end())
			{
				m_outputNodes.erase(search);
			}

			// ? how to deal with m_recurrentInfo, when we delete a node.

            //delete the node itself
            m_nameToNodeMap.erase(nodeName);
            delete nodeToDelete;
        }


        // RenameNode - Rename a node to another name
        // nodeNameOrig - original node name
        // nodeNameNew - new node name
        void RenameNode(const std::wstring& nodeNameOrig, const std::wstring& nodeNameNew)
        {
            ClearCaches(); //so that renamed node will not be referenced

            ComputationNodePtr nodeToRename = GetNodeFromName(nodeNameOrig);

            auto iter = m_nameToNodeMap.find(nodeNameNew);
            if (iter != m_nameToNodeMap.end()) //found
                throw std::runtime_error("RenameNode: Target name already exists.");

            //rename the node and update the mapping table
            nodeToRename->NodeName() = nodeNameNew;
            m_nameToNodeMap.erase(nodeNameOrig);
            m_nameToNodeMap[nodeNameNew] = nodeToRename;

        }


        ComputationNodePtr SetNodeValue(const std::wstring nodeName, const ElemType value)
        {
            ComputationNodePtr pNode = GetNodeFromName(nodeName);

            if (pNode->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                pNode->FunctionValues().SetValue(value);
            }
            else if (pNode->RequirePreCompute())
            {
                PreComputedNode<ElemType> * preComputedNode = static_cast<PreComputedNode<ElemType> *> (pNode);
                pNode->FunctionValues().SetValue(value);
                preComputedNode->MarkComputed(true);
            }
            else
            {
                throw std::logic_error("Only values of learnable parameters and precomputed nodes can be set.");
            }

            return pNode;
        }

        ComputationNodePtr CopyNode(const ComputationNetwork<ElemType> & fromNet, const std::wstring fromName, std::wstring toName = L"", const CopyNodeFlags flags=CopyNodeFlags::copyNodeAll)
        {
            if (toName == L"")
                toName = fromName;

            ComputationNodePtr pFromNode = fromNet.GetNodeFromName(fromName);
            ComputationNodePtr pToNode = nullptr;

            // don't allow cross network child copy unless caller explicity handles children fixup
            if ((flags & CopyNodeFlags::copyNodeChildren) && this != &fromNet &&
                !(flags & CopyNodeFlags::copyNodeChildrenCrossNetwork))
                throw std::logic_error("CopyNode: Copy node children across network is invalid.");

            if (!NodeNameExist(toName))
            {
                pToNode = pFromNode->Duplicate(toName, flags);
                AddNodeToNet(pToNode);
            }
            else //node already exists
            {
                pToNode = GetNodeFromName(toName);
                if (pFromNode == pToNode)  //same node. no copy needed
                    throw std::logic_error("CopyNode: You are copying the node to the same network with same node name.");
                else
                {
                    pFromNode->CopyTo(pToNode, toName, flags);
                }
            }
            return pToNode;
        }

        //only copy a complete independent tree
        //when node name exists 
        void CopySubTree(const ComputationNetwork<ElemType> & fromNet, const std::wstring fromName, std::wstring toNamePrefix = L"", const CopyNodeFlags flags=copyNodeAll)
        {
            if (!(flags & CopyNodeFlags::copyNodeValue))
                throw std::logic_error("CopySubTree: you cannot copy a tree without copying the node values.");

            ComputationNodePtr fromRoot = fromNet.GetNodeFromName(fromName);

            std::list<ComputationNodePtr>&  nodes = GetEvalOrder(fromRoot);
            for (auto nodeIter=nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodePtr fromNode =  (*nodeIter);
                wstring fromNodeName = fromNode->NodeName();
                wstring toNodeName = toNamePrefix + fromNodeName;
                
                ComputationNodePtr toNode = CopyNode(fromNet, fromNodeName, toNodeName, CopyNodeFlags::copyNodeValue);

                if (flags & CopyNodeFlags::copyNodeChildren) //copy the children structure but use the new nodes generated
                {
                    for (int i=0; i<fromNode->ChildrenSize(); i++)
                    {
                        toNode->SetInput(i, GetNodeFromName(toNamePrefix + fromNode->Inputs(i)->NodeName()));
                    }                     
                }
            }
        }

        //you can only copy inputs from nodes in the same network
        void CopyInputs(const std::wstring fromName, std::wstring toName)
        {
            CopyNode(*this, fromName, toName, CopyNodeFlags::copyNodeChildren);
        }
       
#pragma endregion Network Modification

        ComputationNode<ElemType>* CreateNodeFromFile(const std::wstring nodeType, const std::wstring nodeName, File & fstream, size_t modelVersion)
        {            
            ComputationNode<ElemType>* newNode = nullptr;

            if (nodeType == LearnableParameter<ElemType>::TypeName())
                newNode = new LearnableParameter<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == ConstParameter<ElemType>::TypeName())
                newNode = new ConstParameter<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == InputValue<ElemType>::TypeName())
                newNode = new InputValue<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == SparseLearnableParameter<ElemType>::TypeName())
                newNode = new SparseLearnableParameter<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == SparseInputValue<ElemType>::TypeName())
                newNode = new SparseInputValue<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == ConvolutionNode<ElemType>::TypeName())
                newNode = new ConvolutionNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == MaxPoolingNode<ElemType>::TypeName())
                newNode = new MaxPoolingNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == AveragePoolingNode<ElemType>::TypeName())
                newNode = new AveragePoolingNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == NegateNode<ElemType>::TypeName())
                newNode = new NegateNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == RectifiedLinearNode<ElemType>::TypeName())
                newNode = new RectifiedLinearNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == SigmoidNode<ElemType>::TypeName())
                newNode = new SigmoidNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == TanhNode<ElemType>::TypeName())
                newNode = new TanhNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == ExpNode<ElemType>::TypeName())
                newNode = new ExpNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == LogNode<ElemType>::TypeName())
                newNode = new LogNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == CosineNode<ElemType>::TypeName())
                newNode = new CosineNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == SoftmaxNode<ElemType>::TypeName())
                newNode = new SoftmaxNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == LogSoftmaxNode<ElemType>::TypeName())
                newNode = new LogSoftmaxNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == SumElementsNode<ElemType>::TypeName())
                newNode = new SumElementsNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == ScaleNode<ElemType>::TypeName())
                newNode = new ScaleNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == TransposeNode<ElemType>::TypeName())
                newNode = new TransposeNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == TimesNode<ElemType>::TypeName())
                newNode = new TimesNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == StrideTimesNode<ElemType>::TypeName())
                newNode = new StrideTimesNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == ElementTimesNode<ElemType>::TypeName())
                newNode = new ElementTimesNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == DiagTimesNode<ElemType>::TypeName())
                newNode = new DiagTimesNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == CosDistanceNode<ElemType>::TypeName())
                newNode = new CosDistanceNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == KhatriRaoProductNode<ElemType>::TypeName())
                newNode = new KhatriRaoProductNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == PlusNode<ElemType>::TypeName())
                newNode = new PlusNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == MinusNode<ElemType>::TypeName())
                newNode = new MinusNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == SquareErrorNode<ElemType>::TypeName())
                newNode = new SquareErrorNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == CrossEntropyWithSoftmaxNode<ElemType>::TypeName())
                newNode = new CrossEntropyWithSoftmaxNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == ClassBasedCrossEntropyWithSoftmaxNode<ElemType>::TypeName())
                newNode = new ClassBasedCrossEntropyWithSoftmaxNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName); 
            else if (nodeType == NoiseContrastiveEstimationNode<ElemType>::TypeName())
                newNode = new NoiseContrastiveEstimationNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == CRFNode<ElemType>::TypeName())
                newNode = new CRFNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == LSTMNode<ElemType>::TypeName())
                newNode = new LSTMNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == CrossEntropyNode<ElemType>::TypeName())
                newNode = new CrossEntropyNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == MatrixL1RegNode<ElemType>::TypeName())
                newNode = new MatrixL1RegNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == MatrixL2RegNode<ElemType>::TypeName())
                newNode = new MatrixL2RegNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == PerDimMeanVarNormalizationNode<ElemType>::TypeName() || nodeType==L"PerDimMeanVarNormalizationNode") // mseltzer - hack b/c this changed (Dong?) and old models didn't load...
                newNode = new PerDimMeanVarNormalizationNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);            
            else if (nodeType == PerDimMeanVarDeNormalizationNode<ElemType>::TypeName() || nodeType==L"PerDimMeanVarDeNormalizationNode") // mseltzer - hack b/c this changed (Dong?) and old models didn't load...
                newNode = new PerDimMeanVarDeNormalizationNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == ErrorPredictionNode<ElemType>::TypeName())
                newNode = new ErrorPredictionNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);    
            else if (nodeType == DropoutNode<ElemType>::TypeName())
                newNode = new DropoutNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == MeanNode<ElemType>::TypeName())
                newNode = new MeanNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == InvStdDevNode<ElemType>::TypeName())
                newNode = new InvStdDevNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == DelayNode<ElemType>::TypeName())
                newNode = new DelayNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == LookupTableNode<ElemType>::TypeName())
                newNode = new LookupTableNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == RowSliceNode<ElemType>::TypeName())
                newNode = new RowSliceNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == RowStackNode<ElemType>::TypeName())
                newNode = new RowStackNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == GMMLogLikelihoodNode<ElemType>::TypeName())
                newNode = new GMMLogLikelihoodNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == SequenceDecoderNode<ElemType>::TypeName())
                newNode = new SequenceDecoderNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
			else if (nodeType == CosDistanceWithNegativeSamplesNode<ElemType>::TypeName())
				newNode = new CosDistanceWithNegativeSamplesNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == TimeReverseNode<ElemType>::TypeName())
                newNode = new TimeReverseNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == ParallelNode<ElemType>::TypeName())
                newNode = new ParallelNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else if (nodeType == PairNetworkNode<ElemType>::TypeName())
                newNode = new PairNetworkNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else
            {
                fprintf(stderr, "Error creating new ComputationNode of type %ls, with name %ls\n", nodeType.c_str(), nodeName.c_str());
                throw std::invalid_argument("Invalid node type.");
            }
            
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CreateConstParameter(const std::wstring paramName, const size_t rows, const size_t cols)
        {
            ComputationNodePtr newNode(new ConstParameter<ElemType>(rows, cols, m_deviceId, paramName));
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CreateLearnableParameter(const std::wstring paramName, const size_t rows, const size_t cols)
        {
            ComputationNodePtr newNode(new LearnableParameter<ElemType>(rows, cols, m_deviceId, paramName));
            AddNodeToNet(newNode);
            return newNode;
        }

        //sparse matrix size is optionally specified
        ComputationNodePtr CreateSparseLearnableParameter(const std::wstring paramName, const size_t rows, const size_t cols, const size_t size = 0)
        {
            ComputationNodePtr newNode(new SparseLearnableParameter<ElemType>(rows, cols, size, m_deviceId, paramName));
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CreateInputNode(const std::wstring inputName, const size_t rows, const size_t cols)
        {
            ComputationNodePtr newNode(new InputValue<ElemType>(rows, cols, m_deviceId, inputName));
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr PairNetwork(const ComputationNodePtr & a, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new PairNetworkNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CreateSparseInputNode(const std::wstring inputName, const size_t rows, const size_t cols)
        {
            ComputationNodePtr newNode(new SparseInputValue<ElemType>(rows, cols, m_deviceId, inputName));
            AddNodeToNet(newNode);
            return newNode;
        }


        ComputationNodePtr CreateInputNode(const std::wstring inputName, const size_t imageWidth, const size_t imageHeight, const size_t imageChannels, const size_t numImages)
        {
            ComputationNodePtr newNode(new InputValue<ElemType>(imageWidth, imageHeight, imageChannels, numImages, m_deviceId, inputName));
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CreateSparseInputNode(const std::wstring inputName, const size_t imageWidth, const size_t imageHeight, const size_t imageChannels, const size_t numImages)
        {
            ComputationNodePtr newNode(new SparseInputValue<ElemType>(imageWidth, imageHeight, imageChannels, numImages, m_deviceId, inputName));
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CreatePairNetworkNode(const std::wstring inputName, const size_t rows, const size_t cols)
        {
            ComputationNodePtr newNode(new PairNetworkNode<ElemType>(rows, cols, m_deviceId, inputName));
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CreateConvolutionNode(const std::wstring nodeName,
                        const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels, 
                        const size_t horizontalSubsample, const size_t verticalSubsample, 
                        const bool zeroPadding = false, const size_t maxTempMemSizeInSamples = 0)
        {
            ComputationNodePtr newNode(new ConvolutionNode<ElemType>(
                        kernelWidth, kernelHeight, outputChannels, 
                        horizontalSubsample, verticalSubsample, 
                        zeroPadding, m_deviceId, nodeName, maxTempMemSizeInSamples));
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CreateMaxPoolingNode(const std::wstring nodeName, 
                        const size_t windowWidth, const size_t windowHeight, 
                        const size_t horizontalSubsample, const size_t verticalSubsample)
        {
            ComputationNodePtr newNode(new MaxPoolingNode<ElemType>(
                        windowWidth, windowHeight, horizontalSubsample, verticalSubsample, m_deviceId, nodeName));
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CreateAveragePoolingNode(const std::wstring nodeName, 
                        const size_t windowWidth, const size_t windowHeight, 
                        const size_t horizontalSubsample, const size_t verticalSubsample)
        {
            ComputationNodePtr newNode(new AveragePoolingNode<ElemType>(
                        windowWidth, windowHeight, horizontalSubsample, verticalSubsample, m_deviceId, nodeName));
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CreateComputationNode(const std::wstring nodeType, const std::wstring nodeName) 
        {
            ComputationNode<ElemType>* newNode;

            if (nodeType == NegateNode<ElemType>::TypeName())
                newNode = new NegateNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == RectifiedLinearNode<ElemType>::TypeName())
                newNode = new RectifiedLinearNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == SigmoidNode<ElemType>::TypeName())
                newNode = new SigmoidNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == TanhNode<ElemType>::TypeName())
                newNode = new TanhNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == ExpNode<ElemType>::TypeName())
                newNode = new ExpNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == LogNode<ElemType>::TypeName())
                newNode = new LogNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == CosineNode<ElemType>::TypeName())
                newNode = new CosineNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == SoftmaxNode<ElemType>::TypeName())
                newNode = new SoftmaxNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == LogSoftmaxNode<ElemType>::TypeName())
                newNode = new LogSoftmaxNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == SumElementsNode<ElemType>::TypeName())
                newNode = new SumElementsNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == ScaleNode<ElemType>::TypeName())
                newNode = new ScaleNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == TransposeNode<ElemType>::TypeName())
                newNode = new TransposeNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == TimesNode<ElemType>::TypeName())
                newNode = new TimesNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == StrideTimesNode<ElemType>::TypeName())
                newNode = new StrideTimesNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == ElementTimesNode<ElemType>::TypeName())
                newNode = new ElementTimesNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == DiagTimesNode<ElemType>::TypeName())
                newNode = new DiagTimesNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == CosDistanceNode<ElemType>::TypeName())
                newNode = new CosDistanceNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == KhatriRaoProductNode<ElemType>::TypeName())
                newNode = new KhatriRaoProductNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == PlusNode<ElemType>::TypeName())
                newNode = new PlusNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == MinusNode<ElemType>::TypeName())
                newNode = new MinusNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == SquareErrorNode<ElemType>::TypeName())
                newNode = new SquareErrorNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == CrossEntropyWithSoftmaxNode<ElemType>::TypeName())
                newNode = new CrossEntropyWithSoftmaxNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == CrossEntropyNode<ElemType>::TypeName())
                newNode = new CrossEntropyNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == ClassBasedCrossEntropyWithSoftmaxNode<ElemType>::TypeName())
                newNode = new ClassBasedCrossEntropyWithSoftmaxNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == CRFNode<ElemType>::TypeName())
                newNode = new CRFNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == LSTMNode<ElemType>::TypeName())
                newNode = new LSTMNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == MatrixL1RegNode<ElemType>::TypeName())
                newNode = new MatrixL1RegNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == MatrixL2RegNode<ElemType>::TypeName())
                newNode = new MatrixL2RegNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == PerDimMeanVarNormalizationNode<ElemType>::TypeName())
                newNode = new PerDimMeanVarNormalizationNode<ElemType>(m_deviceId, nodeName);        
            else if (nodeType == PerDimMeanVarDeNormalizationNode<ElemType>::TypeName())
                newNode = new PerDimMeanVarDeNormalizationNode<ElemType>(m_deviceId, nodeName);        
            else if (nodeType == ErrorPredictionNode<ElemType>::TypeName())
                newNode = new ErrorPredictionNode<ElemType>(m_deviceId, nodeName);    
            else if (nodeType == DropoutNode<ElemType>::TypeName())
                newNode = new DropoutNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == MeanNode<ElemType>::TypeName())
                newNode = new MeanNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == InvStdDevNode<ElemType>::TypeName())
                newNode = new InvStdDevNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == DelayNode<ElemType>::TypeName())
                newNode = new DelayNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == LookupTableNode<ElemType>::TypeName())
                newNode = new LookupTableNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == GMMLogLikelihoodNode<ElemType>::TypeName())
                newNode = new GMMLogLikelihoodNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == SequenceDecoderNode<ElemType>::TypeName())
                newNode = new SequenceDecoderNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == TimeReverseNode<ElemType>::TypeName())
                newNode = new TimeReverseNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == CosDistanceWithNegativeSamplesNode<ElemType>::TypeName())
				newNode = new CosDistanceWithNegativeSamplesNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == ParallelNode<ElemType>::TypeName())
                newNode = new ParallelNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == RowStackNode<ElemType>::TypeName())
                newNode = new RowStackNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == PairNetworkNode<ElemType>::TypeName())
                newNode = new PairNetworkNode<ElemType>(m_deviceId, nodeName);
            else
            {
                fprintf(stderr, "Error creating new ComputationNode of type %ls, with name %ls\n", nodeType.c_str(), nodeName.c_str());
                throw std::invalid_argument("Invalid node type.");
            }
            
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Parameter(const size_t rows, size_t cols, const std::wstring nodeName = L"")
        {
            return CreateLearnableParameter(nodeName, rows, cols);
        }

        ComputationNodePtr Input(const size_t rows, const size_t cols, const std::wstring nodeName = L"")
        {
            return CreateInputNode(nodeName, rows, cols);
        }

        ComputationNodePtr Input(const size_t imageWidth, const size_t imageHeight, const size_t imageChannels, const size_t numImages, const std::wstring nodeName = L"")
        {
            return CreateInputNode(nodeName, imageWidth, imageHeight, imageChannels, numImages);
        }

        ComputationNodePtr Convolution(const ComputationNodePtr weight, const ComputationNodePtr inputValues, 
                        const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels, 
                        const size_t horizontalSubsample, const size_t verticalSubsample, 
                        const bool zeroPadding = false, const std::wstring nodeName = L"", const size_t maxTempMemSizeInSamples = 0)
        {
            ComputationNodePtr newNode(new ConvolutionNode<ElemType>(
                        kernelWidth, kernelHeight, outputChannels, 
                        horizontalSubsample, verticalSubsample, 
                        zeroPadding, m_deviceId, nodeName, maxTempMemSizeInSamples));
            newNode->AttachInputs(weight, inputValues);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr MaxPooling(const ComputationNodePtr inputValues, 
                        const size_t windowWidth, const size_t windowHeight, 
                        const size_t horizontalSubsample, const size_t verticalSubsample, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new MaxPoolingNode<ElemType>(
                        windowWidth, windowHeight, horizontalSubsample, verticalSubsample, m_deviceId, nodeName));
            newNode->AttachInputs(inputValues);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr AveragePooling(const ComputationNodePtr inputValues, 
                        const size_t windowWidth, const size_t windowHeight, 
                        const size_t horizontalSubsample, const size_t verticalSubsample, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new AveragePoolingNode<ElemType>(
                        windowWidth, windowHeight, horizontalSubsample, verticalSubsample, m_deviceId, nodeName));
            newNode->AttachInputs(inputValues);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr ErrorPrediction (const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new ErrorPredictionNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a, b);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr PerDimMeanVarNormalization (const ComputationNodePtr feature, const ComputationNodePtr mean, const ComputationNodePtr InvStdDev, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new PerDimMeanVarNormalizationNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(feature, mean, InvStdDev);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr PerDimMeanVarDeNormalization (const ComputationNodePtr feature, const ComputationNodePtr mean, const ComputationNodePtr InvStdDev, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new PerDimMeanVarDeNormalizationNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(feature, mean, InvStdDev);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr SquareError (const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new SquareErrorNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a, b);
            AddNodeToNet(newNode);
            return newNode;
        }


        ComputationNodePtr SequenceDecoder(const ComputationNodePtr label, const ComputationNodePtr prediction, const ComputationNodePtr pairscore, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new SequenceDecoderNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(label, prediction, pairscore);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CrossEntropyWithSoftmax (const ComputationNodePtr label, const ComputationNodePtr prediction, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new CrossEntropyWithSoftmaxNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(label, prediction);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr NoiseContrastiveEstimation(const ComputationNodePtr label, const ComputationNodePtr prediction,
            const ComputationNodePtr input_weight, const ComputationNodePtr input_bias, const std::wstring nodeName = L"", NCEEvalMode mode = NCEEvalMode::None)
        {
            ComputationNodePtr newNode(new NoiseContrastiveEstimationNode<ElemType>(m_deviceId, nodeName, mode));
            newNode->AttachInputs(label, prediction, input_weight, input_bias);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr ClassCrossEntropyWithSoftmax(const ComputationNodePtr label, const ComputationNodePtr prediction,
            const ComputationNodePtr input_weight, const ComputationNodePtr cls_log_post_prob, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new ClassBasedCrossEntropyWithSoftmaxNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(label, prediction, input_weight, cls_log_post_prob);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CRF(const ComputationNodePtr label, const ComputationNodePtr postDepScore,
            const ComputationNodePtr transition_score, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new CRFNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(label, postDepScore, transition_score);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr LSTM(const ComputationNodePtr obs, const ComputationNodePtr inputGate, const ComputationNodePtr forgetGate, const ComputationNodePtr outputGate, const ComputationNodePtr memoryCellWgt, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new LSTMNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(obs, inputGate, forgetGate, outputGate, memoryCellWgt);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CrossEntropy(const ComputationNodePtr label, const ComputationNodePtr prediction, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new CrossEntropyNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(label, prediction);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr MatrixL1Reg (const ComputationNodePtr a, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new MatrixL1RegNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr MatrixL2Reg (const ComputationNodePtr a, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new MatrixL2RegNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Mean(const ComputationNodePtr a, const std::wstring nodeName = L"") 
        {
            ComputationNodePtr newNode(new MeanNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr InvStdDev(const ComputationNodePtr a, const std::wstring nodeName = L"") 
        {
            ComputationNodePtr newNode(new InvStdDevNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Negate(const ComputationNodePtr a, const std::wstring nodeName = L"") 
        {
            ComputationNodePtr newNode(new NegateNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr RectifiedLinear(const ComputationNodePtr a, const std::wstring nodeName = L"") 
        {
            ComputationNodePtr newNode(new RectifiedLinearNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Sigmoid(const ComputationNodePtr a, const std::wstring nodeName = L"") 
        {
            ComputationNodePtr newNode(new SigmoidNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Tanh(const ComputationNodePtr a, const std::wstring nodeName = L"") 
        {
            ComputationNodePtr newNode(new TanhNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Exp(const ComputationNodePtr a, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new ExpNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Log(const ComputationNodePtr a, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new LogNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Cos(const ComputationNodePtr a, const std::wstring nodeName = L"") 
        {
            ComputationNodePtr newNode(new CosineNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Softmax(const ComputationNodePtr a, const std::wstring nodeName = L"") 
        {
            ComputationNodePtr newNode(new SoftmaxNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr LogSoftmax(const ComputationNodePtr a, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new LogSoftmaxNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Sum(const ComputationNodePtr a, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new SumElementsNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }


        ComputationNodePtr Scale(const ComputationNodePtr scalar, const ComputationNodePtr matrix, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new ScaleNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(scalar, matrix);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Transpose(const ComputationNodePtr matrix, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new TransposeNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(matrix);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Times(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new TimesNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a, b);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr StrideTimes(const ComputationNodePtr a, const ComputationNodePtr b, const ComputationNodePtr c, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new StrideTimesNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a, b, c);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr ElementTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new ElementTimesNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a, b);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr DiagTimes (const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new DiagTimesNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a, b);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CosDistance (const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new CosDistanceNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a, b);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr KhatriRaoProduct (const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new KhatriRaoProductNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a, b);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Plus (const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new PlusNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a, b);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Minus (const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new MinusNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a, b);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Dropout (const ComputationNodePtr a, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new DropoutNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Delay (const ComputationNodePtr a, const float initHiddenActivity, const size_t row_size, const size_t col_size, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new DelayNode<ElemType>(m_deviceId, initHiddenActivity, row_size, col_size, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);

            return newNode;
        }

        ComputationNodePtr Parallel(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new ParallelNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a, b);
            AddNodeToNet(newNode);

            return newNode;
        }

        ComputationNodePtr RowSlice(const ComputationNodePtr a, const size_t start_index, const size_t num_rows, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new RowSliceNode<ElemType>(m_deviceId, start_index, num_rows, nodeName));
            newNode->AttachInputs(a);
            AddNodeToNet(newNode);

            return newNode;
        }

        ComputationNodePtr RowStack(const std::vector<ComputationNodePtr> inputs, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new RowStackNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(inputs);
            AddNodeToNet(newNode);

            return newNode;
        }

        ComputationNodePtr GMMLogLikelihood(const ComputationNodePtr unnormedPrior, const ComputationNodePtr mean, const ComputationNodePtr logStddev, const ComputationNodePtr feature, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new GMMLogLikelihoodNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(unnormedPrior, mean, logStddev, feature);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr TimeReverse(const ComputationNodePtr input, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new TimeReverseNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(input);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr LookupTable(const ComputationNodePtr dictionary, const ComputationNodePtr input, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new LookupTableNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(dictionary, input);
            AddNodeToNet(newNode);
            return newNode;
        }

        bool NodeNameExist(const std::wstring& name) const
        {
            auto iter = m_nameToNodeMap.find(name);
            return (iter != m_nameToNodeMap.end());
        }

        ComputationNodePtr GetNodeFromName(const std::wstring& name, ComputationNetwork<ElemType>* anotherNetwork = nullptr)  const
        {
            auto iter = m_nameToNodeMap.find(name);
            if (iter != m_nameToNodeMap.end()) //found
                return iter->second;
            if (anotherNetwork != nullptr)
                return anotherNetwork->GetNodeFromName(name);
            
            RuntimeError("GetNodeFromName: Node name %s does not exist.", name.c_str());
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
                std::wstring tail = name.substr(found+1);
                for (auto nodeIter=m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
                {
                    const wstring& nodeName = nodeIter->first;

                    // if it matches on both ends (we only support A*B patterns it's a match
                    bool headMatch = head.empty() || nodeName.find(head) == 0;
                    bool tailMatch = tail.empty() || nodeName.rfind(tail) == nodeName.size()-tail.size();
                    if (headMatch && tailMatch)
                    {
                        nodes.push_back(nodeIter->second);
                    }
                }
            }
            return nodes;
        }

        int FindInRecurrentLoop(const ComputationNodePtr startNode, std::vector<ComputationNodePtr>& recurrentNodes, bool isForwardComputing=false)
        {
            int iFound = -1;

            for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
            {
                if (std::find((*iter).m_recurrentNodes.begin(), (*iter).m_recurrentNodes.end(), startNode) != (*iter).m_recurrentNodes.end())
                {
                    iFound = (*iter).m_loopId;
                    if (isForwardComputing)
                    {
                        recurrentNodes = (*iter).m_recurrentNodesForForward;
                    }
                    else
                    {
                        recurrentNodes = (*iter).m_recurrentNodesForForward;
                    }
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
            for ( auto ptr = recurrentNodes.begin(); ptr != recurrentNodes.end(); ptr++ )
            {
                if ((*ptr)->IsFuncValueOlderThanInputs() && (*ptr)->OperationName() != L"Delay") 
                    return true;
            }
            return false; 
        }

        void EvaluateLoop(std::list<ComputationNodePtr>& /*allNodes*/, const ComputationNodePtr startNode)
        {
            bool bLoopCompleted = true;
            std::vector<ComputationNodePtr> recurrentNodes;
            int iLoopId = FindInRecurrentLoop(startNode, recurrentNodes, true);
            if (iLoopId != -1 && IsFuncValueOlderThanInputs(recurrentNodes) && m_recurrentInfo[iLoopId].m_completedEvaluate == false)
            {

                for (auto nodeIter=recurrentNodes.begin(); nodeIter != recurrentNodes.end(); nodeIter++)
                {
                    (*nodeIter)->SetFunctionAndGradientSize(m_actMiniBSize);
                }

                size_t iCnt = 0; 
                size_t iMBSize = m_actMiniBSize / m_nbrSlicesInEachRecurrentIteration;
                do {
                    bLoopCompleted = true;
                    for (auto nodeIter=recurrentNodes.begin(); nodeIter != recurrentNodes.end(); nodeIter++)
                    {
                        (*nodeIter)->EvaluateThisNodeGivenInputs(iCnt);

                        (*nodeIter)->UpdateEvalTimeStamp();

                    }

                    iCnt ++;
                } while (iCnt < iMBSize);

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
                nodePtr->OperationName() == CRFNode<ElemType>::TypeName())
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
            {
                if (IsTypicalCriterionNode(node))
                    node->SetReqMultiSeqHandlingTo(true);
            }

            for (auto node : m_evalNodes)
            {
                if (IsTypicalCriterionNode(node))
                    node->SetReqMultiSeqHandlingTo(true);
            }
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
            {
                m_recurrentInfo[i].m_completedEvaluate = false;
            }

            for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
            {
                (*nodeIter)->SetNbrSlicesInEachRecurrentIteration(m_nbrSlicesInEachRecurrentIteration);
                if ((*nodeIter)->ReqMultiSeqHandling())
                {
                    (*nodeIter)->ResetBound(&mSentenceBoundary, &mExistsBeginOrNoLabels);
                }
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
                    (*nodeIter)->EvaluateThisNodeGivenInputs(); // we manage time stamp here so that derived classes don't need to worry about it
                    (*nodeIter)->UpdateEvalTimeStamp();
                }
            }
        }

        void SetActualMiniBatchSize(const size_t aSize)
        {
            m_actMiniBSize = (int)aSize;

            // assume that all nodes in recurrent loops need to be reset to aSize minibatch size, so need to reset the following
            for (int i=0; i < m_recurrentInfo.size(); i++)
            {
                m_recurrentInfo[i].m_completedEvaluate = false;
                m_recurrentInfo[i].m_completedGradient = false;
            }

            for (int i=0; i < m_recurrentInfo.size(); i++)
            {
                for (auto nodeIter=m_recurrentInfo[i].m_recurrentNodes.begin(); nodeIter != m_recurrentInfo[i].m_recurrentNodes.end(); nodeIter++)
                {
                    (*nodeIter)->SetFunctionAndGradientSize(m_actMiniBSize);
                }
            }
        }

        // GetMaxMBSize - Get the maximum minibatch size that will be seen in a training run
        // returns the result from SetActualMiniBatchSize(). Note GetActualMBSize() also exists but returns a value derived from the inputs dimensions
        size_t GetMaxMBSize()
        {
            return m_actMiniBSize;
        }


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
                    int iCol = m_actMiniBSize/m_nbrSlicesInEachRecurrentIteration -1;
                    do{
                        for (auto nodeIter=recurrentNodes.rbegin(); nodeIter != recurrentNodes.rend(); ++nodeIter)
                        {
                            (*nodeIter)->SetNbrSlicesInEachRecurrentIteration(m_nbrSlicesInEachRecurrentIteration);
                            (*nodeIter)->ComputeGradientForChildren(iCol);
                        }

                        iCol --;
                    }while (iCol >= 0);
                }

                m_recurrentInfo[iLoopId].m_completedGradient = true;
            }
        }

        virtual void ComputeGradient(const ComputationNodePtr rootNode, 
            bool bResetToOne = true,  /// true if reset the gradient of rootnode to 1.0
            const Matrix<ElemType>* rootGradientInitValue = nullptr
            )
        {
            if (bResetToOne && rootNode->FunctionValues().GetNumElements() != 1)
                throw std::runtime_error("ComputeGradient: The root of the Gradient computation must evaluate to R1 value.");

            //run forward pass first
            Evaluate(rootNode);

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
                fprintf (stderr, "Compute Gradient For Node: %s(%s) Against Children\n",
                    (msra::strfun::utf8 ((*nodeIter)->OperationName())).c_str(),
                    (msra::strfun::utf8 ((*nodeIter)->NodeName())).c_str());
#endif
                ComputeGradientLoop(allNodes, *nodeIter);

                (*nodeIter)->ComputeGradientForChildren();
            }
        }

        //for debugging purpose
        void PrintComputationTree(const ComputationNodePtr rootNode, const bool forwardCompute, const bool printMatrices = false)
        {
            std::list<ComputationNodePtr> nodes;
            if (forwardCompute)
            {
                fprintf (stderr, "\n\nPrinting Forward Computation Node Order ... \n");
                nodes = GetEvalOrder(rootNode);
            }
            else
            {
                fprintf (stderr, "\n\nPrinting Gradient Computation Node Order ... \n");
                nodes = GetGradientCalcOrder(rootNode);
            }

            if (nodes.size() == 0)
            {
                fprintf (stderr, "\n$$$$ EMPTY !!!!!\n");
                return;
            }

            for (auto nodeIter=nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodePtr node = (*nodeIter);

                node->PrintSelf(printMatrices);
            }
        }

        void RenameNode(const ComputationNodePtr node, const std::wstring newNodeName)
        {
            m_nameToNodeMap.erase(node->NodeName());
            node->NodeName() = newNodeName;
            AddNodeToNet(node);
        }

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

        std::list<ComputationNodePtr>& InputNodes(const ComputationNodePtr rootNode)
        {
            BuildAndValidateNetwork(rootNode);
            return m_inputs[rootNode];
        }

        std::list<ComputationNodePtr>& LearnableNodes(const ComputationNodePtr rootNode)
        {
            BuildAndValidateNetwork(rootNode);
            return m_learnableParameters[rootNode];
        }

        inline std::vector<ComputationNodePtr>& FeatureNodes() {return m_features;}

        inline std::vector<ComputationNodePtr>& LabelNodes() {return m_labels;}

        inline std::vector<ComputationNodePtr>& FinalCriterionNodes() {return m_finalCriteria;}

        inline std::vector<ComputationNodePtr>& NodesReqMultiSeqHandling() { return m_nodesReqMultiSeqHandling; }

        inline std::vector<ComputationNodePtr>& EvaluationNodes() { return m_evalNodes; }

        inline std::vector<ComputationNodePtr>& OutputNodes() {return m_outputNodes;}

        inline std::vector<RecurrentInfo>& RecurrentNodes() {return m_recurrentInfo;}

        size_t GetTotalNumberOfNodes() const { return m_nameToNodeMap.size();}

        void ResetEvalTimeStamp()
        {
            for (auto nodeIter=m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
                nodeIter->second->ResetEvalTimeStamp();      
        }

        //change the node associated with nodeName to newNode; used in the KL-reg based adaptation to reduce feature copy
        //need to update all the mappings as well childrens
        void ChangeNode(wstring nodeName, ComputationNodePtr newNode)
        {
            ComputationNodePtr oldNode = GetNodeFromName(nodeName);
            if (oldNode->OperationName() != newNode->OperationName())
                throw invalid_argument("newNode must have the same type as the old node.");

            //change children
            for (auto nodeIter=m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodePtr node = nodeIter->second;
                for (int i=0; i<node->ChildrenSize(); i++)
                {
                    if (node->Inputs(i) == oldNode)
                        node->SetInput(i, newNode);
                }                
            }

            //change name map
            m_nameToNodeMap[nodeName] = newNode;
            for (int i=0; i<oldNode->ChildrenSize(); i++)
            {
                newNode->SetInput(i, oldNode->Inputs(i));
            }     

            //change other maps
            for (int i=0; i<m_features.size(); i++)
            {
                if (m_features[i] == oldNode)
                    m_features[i] = newNode;
            }
            for (int i=0; i<m_labels.size(); i++)
            {
                if (m_labels[i] == oldNode)
                    m_labels[i] = newNode;
            }
            for (int i=0; i<m_finalCriteria.size(); i++)
            {
                if (m_finalCriteria[i] == oldNode)
                    m_finalCriteria[i] = newNode;
            }            
            for (int i = 0; i<m_nodesReqMultiSeqHandling.size(); i++)
            {
                if (m_nodesReqMultiSeqHandling[i] == oldNode)
                    m_nodesReqMultiSeqHandling[i] = newNode;
            }
            for (int i = 0; i<m_evalNodes.size(); i++)
            {
                if (m_evalNodes[i] == oldNode)
                    m_evalNodes[i] = newNode;
            } 
            for (int i=0; i<m_outputNodes.size(); i++)
            {
                if (m_outputNodes[i] == oldNode)
                    m_outputNodes[i] = newNode;
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
                {
                    if (node->Inputs(i) == oldNode)
                        node->SetInput(i, newNode);
                }
            }
            m_nameToNodeMap[newNode->GetName()] = newNode;
            // now the old node becomes a orphan node , remove it 
            DeleteNode(oldNodeName);
            //RemoveOrphanNode(oldNode);
        }
        std::vector<ComputationNodePtr> GetAllNodes() const
        {
            std::vector<ComputationNodePtr> nodes;
            for (auto nodeIter=m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodePtr node = nodeIter->second;
                nodes.push_back(node);
            }
            return nodes;
        }

        std::list<ComputationNodePtr> GetNodesWithType(const wstring typeName, const ComputationNodePtr rootNode = nullptr)
        {
            std::list<ComputationNodePtr> nodesWithType;

            if (rootNode == nullptr) //find nodes from all available nodes
            {
                for (auto nodeIter=m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
                {
                    ComputationNodePtr node = nodeIter->second;
                    if (node->OperationName() == typeName)
                        nodesWithType.push_back(node);
                }
            }
            else //for calculating a specific node
            {
                std::list<ComputationNodePtr>&  nodes = GetEvalOrder(rootNode);
                for (auto nodeIter=nodes.begin(); nodeIter != nodes.end(); nodeIter++)
                {
                    ComputationNodePtr node = (*nodeIter);
                    if (node->OperationName() == typeName)
                        nodesWithType.push_back(node);
                }
            }

            return nodesWithType;
        }

        //return list of nodes that require precomputation and not precomputed yet.
        std::list<ComputationNodePtr> GetNodesRequirePreComputation(const ComputationNodePtr rootNode = nullptr, bool checkComputed=true)
        {
            std::list<ComputationNodePtr> nodesRequirePreComputation;

            if (rootNode == nullptr) //find nodes from all available nodes
            {
                for (auto nodeIter=m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
                {
                    ComputationNodePtr node = nodeIter->second;
                    if (node->RequirePreCompute())
                    {
                        PreComputedNode<ElemType> * preComputedNode = static_cast<PreComputedNode<ElemType> *> (node);
                        if (!checkComputed || !preComputedNode->HasComputed())
                            nodesRequirePreComputation.push_back(node);
                    }
                }
            }
            else //for calculating a specific node
            {
                std::list<ComputationNodePtr>&  nodes = GetEvalOrder(rootNode);
                for (auto nodeIter=nodes.begin(); nodeIter != nodes.end(); nodeIter++)
                {
                    ComputationNodePtr node = (*nodeIter);
                    if (node->RequirePreCompute())
                    {
                        PreComputedNode<ElemType> * preComputedNode = static_cast<PreComputedNode<ElemType> *> (node);
                        if (!checkComputed || !preComputedNode->HasComputed())
                            nodesRequirePreComputation.push_back(node);
                    }
                }
            }

            return nodesRequirePreComputation;
        }

        //return list of nodes that require precomputation and not precomputed yet.
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
                        BatchModeNode<ElemType> * preComputedNode = static_cast<BatchModeNode<ElemType> *> (node);
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
                        BatchModeNode<ElemType> * preComputedNode = static_cast<BatchModeNode<ElemType> *> (node);
                        if (!checkComputed || !preComputedNode->HasComputed())
                            nodesRequirePreComputation.push_back(node);
                    }
                }
            }

            return nodesRequirePreComputation;
        }

        // Validate - Validate the network 
        void ValidateNetwork(bool allowFragment=false, const bool bAllowNoCriterion = false)
        {
            // currently only validates nodes, we should validate everything we can
            if (FeatureNodes().size() == 0 && !allowFragment)
            {
                throw std::runtime_error("No Feature nodes specified");
            }
            // first give criteria nodes as root node
            if (FinalCriterionNodes().size() > 0)
            {
                for (ComputationNodePtr node : FinalCriterionNodes())
                {
                    if(!allowFragment) FormRecurentLoops(node);
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
            {
                throw std::runtime_error("No Criterion nodes specified");
            }
            // now output nodes
            if (OutputNodes().size() > 0)
            {
				for (ComputationNodePtr node : OutputNodes())
				{
					if (!allowFragment) FormRecurentLoops(node);
					ValidateNetwork(node);
				}
            }
            else if (!allowFragment)
            {
                throw std::runtime_error("No Output nodes specified");
            }
            // now evaluation nodes
            if (EvaluationNodes().size() > 0)
            {
				for (ComputationNodePtr node : EvaluationNodes())
				{
					if (!allowFragment) FormRecurentLoops(node);
					ValidateNetwork(node);
				}
            }
        }

        void ValidateNetwork(const ComputationNodePtr rootNode)
        {
            fprintf(stderr, "\n\nValidating node %ls \n", rootNode->NodeName().c_str());

            std::list<ComputationNodePtr>&  nodes = GetEvalOrder(rootNode);
            
            for (auto nodeIter=nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                (*nodeIter)->Validate();
            }

            fprintf(stderr, "\n\n");
        }

        void BuildAndValidateNetwork(const ComputationNodePtr rootNode)
        {
            const ComputationNodePtr key = rootNode;
            if (m_built.find(key) == m_built.end()) //not found
            {
                m_built[key] = true;
                FormRecurentLoops(rootNode);
                ValidateNetwork(rootNode);
                CollectInputAndLeanableParameters(rootNode);
                SetNodesReqMultiSeqHandling();
            }
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
            {
                throw std::runtime_error("No Feature nodes specified");
            }
            // first give criteria nodes as root node
            if (FinalCriterionNodes().size() > 0)
            {
                for (auto node : FinalCriterionNodes())
                {
                    if (!allowFragment) FormRecurentLoops(node);
                    size_t actualMBSize = this->GetActualMBSize();
                    this->SetActualMiniBatchSize(actualMBSize);
                    if (UnitTest(node) == false)
                        vErrors.push_back(node->NodeName().c_str());
                }
            }
            else if (!allowFragment)
            {
                throw std::runtime_error("No Criterion nodes specified");
            }
            // now output nodes
            if (OutputNodes().size() > 0)
            {
                for (auto node : OutputNodes())
                    if (UnitTest(node) == false)
                        vErrors.push_back(node->NodeName().c_str());
            }
            else if (!allowFragment)
            {
                throw std::runtime_error("No Output nodes specified");
            }
            // now evaluation nodes
            if (EvaluationNodes().size() > 0)
            {
                for (auto node : EvaluationNodes())
                    if (UnitTest(node) == false)
                        vErrors.push_back(node->NodeName().c_str());
            }
            if (vErrors.size() > 0)
                return false; 
            return true;
        }

        bool UnitTest(const ComputationNodePtr rootNode)
        {
            fprintf(stderr, "\n\n Unit test node %ws \n", rootNode->NodeName().c_str());

            std::list<ComputationNodePtr>&  nodes = GetEvalOrder(rootNode);

            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                if ((*nodeIter)->UnitTest() == false)
                    return false;
            }

            fprintf(stderr, "\n\n");

            return true;
        }

        //========================================
        // This function performs SVD decomposition for different groups of learnable  parameters 
        // we perform SVD decomposition such that 
        //  A \approx B*C, where rank(B)=rank(C)=r < rank(A)
        // After SVD decomposition, the node A will become an intermediate node whose children are B,C ; 
        // B and C are two learnable parameters 
        //========================================
        void PerformSVDecomposition(const map<wstring, float>& SVDConfig)   
        {
            vector<pair<vector<wstring>,float> > nodeGroups; 
            wregex NameFilter; 

            for (auto e : SVDConfig)
            {
                wstring regexStr = e.first; 
                float   keepRatio = e.second; 
                vector<wstring>     NamesInGroup;

                NameFilter.assign(regexStr);

                for (auto n = m_nameToNodeMap.begin(); n != m_nameToNodeMap.end(); n++)
                {
                    if (!regexStr.empty() && !regex_match(n->first, NameFilter))
                        continue;           // if regexStr is not empty and the the node node does not match with the regexStr
                    ComputationNodePtr ptr = n->second; 
                    if (ptr->OperationName() != LearnableParameter<ElemType>::TypeName())
                        continue;
                    Matrix<ElemType>  W = ptr->FunctionValues();
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
                fprintf(stderr, "--------------------------------------------------------------------------------------------\n");
                fprintf(stderr, "ParameterSVD: start to process group %d with KeepRatio=%.2f\n", (int) groupID++, keepratio);
                fprintf(stderr, "--------------------------------------------------------------------------------------------\n");
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
                    if (A.GetNumCols() == 1 || A.GetNumRows() == 1)       // it is a vector, no need to do it 
                        continue;
                    size_t m = A.GetNumRows();
                    size_t n = A.GetNumCols();

                    Matrix<ElemType> S(-1), U(-1), VT(-1), W(-1);
                    std::chrono::time_point<std::chrono::system_clock> stTime = std::chrono::system_clock::now();
                    Matrix<ElemType>::SVD(A, S, U, VT, W);
                    std::chrono::time_point<std::chrono::system_clock> enTime = std::chrono::system_clock::now();
                    // A \in R^{mXn} 
                    // U \in R^{mXm} 
                    // VT \in R^{nXn} 
                    // S \in R^{min(m,n),1}  
                    // S is in descending order 
                    // 
                    ElemType totalenergy = 0.0f;
                    for (size_t i = 0; i < S.GetNumRows(); i++)
                    {
                        totalenergy += S(i, 0);
                    }
                    ElemType keepenergy = totalenergy*keepratio;
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

                    r = (r + 7) & (~7);             //  to keep the number of rows/cols of resultant matrix a multipier of 8 
                    //  which can be helpful at runtime 

                    std::chrono::duration<double>  elapsedtime = enTime - stTime;
                    fprintf(stderr, "Performing SVD for a %5d-by-%-5d matrix (node name: %-20ls) ---  computation time %5.2f secs ;  keep %4.1f%% energy ===> keep %5d svd values (reduce to %4.1f%% parameters) \n",
                        (int)m, (int)n, name.c_str(), elapsedtime.count(), keepratio * 100, (int)r, ((m + n)*r + 0.0f) / m / n * 100);


                    Matrix<ElemType> redU = U.ColumnSlice(0, r);        // redU in R^ {mXr}
                    Matrix<ElemType> redVT(-1); redVT.Resize(r, n);         // redVT in R^{rXn}
                    redVT.AssignRowSliceValuesOf(VT, 0, r);

                    Matrix<ElemType>    redS(r, (size_t)1);
                    for (size_t i = 0; i < r; i++)
                    {
                        ElemType sqrtsigma = (ElemType)sqrt((double)S(i, 0));
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
        virtual void GetHistory(map<wstring, Matrix<ElemType>>& history, bool bLastTime = false)
        {
            //put all node info first
            Matrix<ElemType> hist;
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodePtr nodePtr = nodeIter->second;
                if (nodePtr->GetHistory(hist, bLastTime))
                {
                    history[nodeIter->first] = hist;
                }
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
    protected:
        // Copy constructor, should never be called.
#pragma warning (push)
#pragma warning (disable: 4702) // this function is flagged but unclear why
        ComputationNetwork(const ComputationNetwork<ElemType>& /*deepCopyFrom*/)
        {   
            //assert(false);
            throw std::logic_error("'ComputationNetwork(const ComputationNetwork<ElemType>& deepCopyFrom)' should never be called.");
        } 
#pragma warning (pop)

        // Assignment operator, should never be called.
        ComputationNetwork<ElemType>& operator=(const ComputationNetwork<ElemType>& /*deepCopyFrom*/)
        {            
            throw std::logic_error("'ComputationNetwork<ElemType>& operator=(const ComputationNetwork<ElemType>& deepCopyFrom)' should never be called.");
        } 

        void ClearCalcOrderCaches()
        {
			for (typename std::map<const ComputationNodePtr, std::list<ComputationNodePtr>>::iterator it = m_cacheEvalOrders.begin(); it != m_cacheEvalOrders.end(); ++it)
			{
				for (auto iter2 = m_cacheEvalOrders[it->first].begin(); iter2 != m_cacheEvalOrders[it->first].end(); iter2++)
				{
					(*iter2)->clearCache();
				}
			}
            m_cacheEvalOrders.clear();
            m_cacheGradientCalcOrders.clear();
        } 

        void MergeRecurrentLoops(const ComputationNodePtr /*rootNode*/)
        {
            /// merge loops if they have the same source node
            std::vector<RecurrentInfo>      m_recurrentInfoTmp;
            int iLoopId = 0; 
            for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
            {
                if (m_recurrentInfoTmp.size() == 0)
                {
                    RecurrentInfo rInfo;
                    rInfo.m_recurrentNodes = (*iter).m_recurrentNodes; 
                    rInfo.m_sourceNode = (*iter).m_sourceNode;
                    rInfo.m_loopId = iLoopId++;
                    rInfo.Reset();
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
                        rInfo.m_recurrentNodes = (*iter).m_recurrentNodes; 
                        rInfo.m_sourceNode = (*iter).m_sourceNode;
                        rInfo.m_loopId = iLoopId++;
                        rInfo.Reset();
                        m_recurrentInfoTmp.push_back(rInfo);
                    }
                    else 
                    {
                        for (auto iter2 = m_recurrentInfoTmp.begin(); iter2 != m_recurrentInfoTmp.end(); iter2++)
                        {
                            if ((*iter2).m_sourceNode == (*iter).m_sourceNode)
                            {
                                for (auto iter3 = (*iter).m_recurrentNodes.begin(); iter3 != (*iter).m_recurrentNodes.end(); iter3++)
                                {
                                    (*iter2).m_recurrentNodes.push_back(*iter3);
                                }
                            }
                        }
                    }
                }
            }

            for (auto iter = m_recurrentInfoTmp.begin(); iter != m_recurrentInfoTmp.end(); iter++)
            {
                // sort the recurrent nodes in their ascending name, which is the same as visiting nodes in G^R
                if ((*iter).m_recurrentNodes.size() > 1)
                {
                    std::sort((*iter).m_recurrentNodes.begin(), (*iter).m_recurrentNodes.end(), (*iter).m_recurrentNodes[0]->IsSmaller);
                }
            }

            /// debug purpose 
            for (auto iter = m_recurrentInfoTmp.begin(); iter != m_recurrentInfoTmp.end(); iter++)
            {
                fprintf(stderr, " nodes in the recurrent loops : \n"); 
                for (auto itr = (*iter).m_recurrentNodes.begin(); itr != (*iter).m_recurrentNodes.end(); itr++)
                {
                    fprintf (stderr, "%ls\t", (*itr)->NodeName().c_str() ); 
                }
            }

            m_recurrentInfo.clear();
            for (auto iter = m_recurrentInfoTmp.begin(); iter != m_recurrentInfoTmp.end(); iter++)
            {
                RecurrentInfo rInfo;
                rInfo.m_recurrentNodes.clear();
                rInfo.m_sourceNode = (*iter).m_sourceNode;
                rInfo.m_loopId = (*iter).m_loopId;
                rInfo.Reset(); 

                ComputationNodePtr lastOne = nullptr ;
                for (auto itr = (*iter).m_recurrentNodes.begin(); itr != (*iter).m_recurrentNodes.end(); itr++)
                {
                    if (lastOne != nullptr && lastOne->NodeName() == (*itr)->NodeName())
                        continue;
                    rInfo.m_recurrentNodes.push_back(*itr);
                    lastOne = *itr;
                }

                m_recurrentInfo.push_back(rInfo);
            }

            /// debug purpose 
            for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
            {
                fprintf(stderr, " nodes in the recurrent loops : \n"); 
                for (auto itr = (*iter).m_recurrentNodes.begin(); itr != (*iter).m_recurrentNodes.end(); itr++)
                {
                    fprintf (stderr, "%ls\t", (*itr)->NodeName().c_str() ); 
                }
            }
        }

        // get the strong connected component from the graph
        void getStrongSCC (const ComputationNodePtr rootNode)
        {
            std::unordered_set<ComputationNodePtr> visited;
            std::list<ComputationNodePtr> sccStack;
            size_t index = 0;
            size_t loopId = 0;
            if(rootNode->isVisisted()==false)
                strongSCC(rootNode, sccStack, index, loopId);
            
        }

        void strongSCC (ComputationNodePtr cur, std::list<ComputationNodePtr>& sccStack, size_t& index, size_t& loopId)
        {
            cur->SetIndex(index);
            cur->Setlowlink(index);
            index++;

            cur->SetVisited(true);
            sccStack.push_back(cur);
            cur->SetInStack(true);

            for (int i = 0; i < cur->ChildrenSize(); i++)
            {
                if (cur->Inputs(i)->isVisisted() == false)
                {
                    strongSCC(cur->Inputs(i),sccStack, index, loopId);
                    cur->Setlowlink(min(cur->Getlowlink(), cur->Inputs(i)->Getlowlink()));
                } else if (cur->Inputs(i)->isInStack())
                {
                    cur->Setlowlink(min(cur->Getlowlink(),cur->Inputs(i)->Getlowlink())); 
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
                    {
                        break;
                    }
                }
                rInfo.Reset(); 
                if (sccSize>1)
                {
                    loopId++;
                    m_recurrentInfo.push_back(rInfo);
                }
            }
        }
        
        void getLoopForwordOrder(std::unordered_set<ComputationNodePtr>& visited,std::unordered_set<ComputationNodePtr>& recStack, std::list<ComputationNodePtr>& nodesStack, ComputationNodePtr cur)
        {
            if (visited.find(cur) == visited.end())
            {
                visited.insert(cur);
                recStack.insert(cur);

                if (cur->OperationName() != L"Delay")
                {
                    for (size_t i = 0; i < cur->ChildrenSize() ; i++)
                    {
                        if (cur->Inputs(i)->LoopId()==cur->LoopId())
                        {
                            getLoopForwordOrder(visited, recStack, nodesStack, cur->Inputs(i));
                        }
                    }
                }
                recStack.erase(cur);
                nodesStack.push_back(cur);
            } else
            {
                if (!(recStack.find(cur) == recStack.end()))
                {
                     throw std::logic_error("There is infinite Loop which cannot be unrolled!!");
                }

            }
        }
        //must be called before ValidateNetwork
        void FormRecurentLoops(const ComputationNodePtr rootNode)
        {
            std::vector<ComputationNodePtr> sourceLoopNodes; 

            getStrongSCC(rootNode);
            std::list<ComputationNodePtr>&  nodes = GetEvalOrder(rootNode, sourceLoopNodes);
			std::list<ComputationNodePtr> nodesForGrad;

            /// debug purpose 
            for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
            {
                fprintf(stderr, " nodes in the recurrent loops : \n"); 
                size_t max_visitedOrderInLoop = 0;
                for (auto itr = (*iter).m_recurrentNodes.begin(); itr != (*iter).m_recurrentNodes.end(); itr++)
                {
                    fprintf (stderr, "%ls\t", (*itr)->NodeName().c_str() ); 
                    if (max_visitedOrderInLoop < (*itr)->GetVisitedOrder())
                    {
                        max_visitedOrderInLoop = (*itr)->GetVisitedOrder();
                    }
                }
                for (auto itr = (*iter).m_recurrentNodes.begin(); itr != (*iter).m_recurrentNodes.end(); itr++)
                {
                    (*itr)->SetVisitedOrder(max_visitedOrderInLoop);
                    
                }
            }


            for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
            {
                // sort the recurrent nodes in their ascending name, which is the same as visiting nodes in G^R
                if ((*iter).m_recurrentNodes.size() > 1)
                {
                    /// it is done in the mergerecurrentloops function, but just keep the code
                    std::sort((*iter).m_recurrentNodes.begin(), (*iter).m_recurrentNodes.end(), (*iter).m_recurrentNodes[0]->IsSmaller);
                    for (auto nodeRecIter = (*iter).m_recurrentNodes.begin(); nodeRecIter != (*iter).m_recurrentNodes.end(); 
                        nodeRecIter++)
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
                    
                    for (size_t j = 0 ; j < (*iter).m_recurrentNodes.size(); j++)
                    {
                        ComputationNodePtr nodeRecIter = (*iter).m_recurrentNodes[j];
                        for (size_t i = 0; i < nodeRecIter->ChildrenSize() ; i++)
                        {
                            if ((nodeRecIter->Inputs(i)->LoopId() == nodeRecIter->LoopId()) && (nodeRecIter->OperationName() != L"Delay"))
                            {
                                nodeRecIter->Inputs(i)->SetIndexInLoop(nodeRecIter->Inputs(i)->GetIndexInLoop()+1);
                            }
                        }
                    }
                
                    //for (auto nodeRecIter = startNodes.begin(); nodeRecIter != startNodes.end(); nodeRecIter++)
                        
                    for (size_t i = 0 ; i < (*iter).m_recurrentNodes.size(); i++)
                    {
                        ComputationNodePtr nodeRecIter = (*iter).m_recurrentNodes[i];
                        if (visited.find(nodeRecIter) == visited.end() && nodeRecIter->GetIndexInLoop() == 0)
                            getLoopForwordOrder(visited,recStack, result,nodeRecIter);
                    }
                    for(size_t i = 0; i < (*iter).m_recurrentNodes.size(); i++)
                    {
                        (*iter).m_recurrentNodesForForward.push_back(result.front());
                        result.pop_front();
                    }
                    
					(*iter).m_recurrentNodes = (*iter).m_recurrentNodesForForward;
                }
            }



            if (m_recurrentInfo.size() > 0)
            {
                std::map<int , std::list<ComputationNodePtr>> recurrentNodes;
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
        }

        void ReorderLoops(std::list<ComputationNodePtr>&  nodes, 
            const std::map<int , std::list<ComputationNodePtr>>& /*recurrentNodes*/,
            const std::list<ComputationNodePtr> & /*noRecurrentNodes*/)
        {
            std::list<ComputationNodePtr> newList;

            std::list<ComputationNodePtr> vTmp;
            std::list<ComputationNodePtr> vRecurrentTmp;
            //int  prevId = -1;
			vector<bool> accessed;
			accessed.assign(m_recurrentInfo.size(),false);
            for (auto nodeIter=nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                int iId = FindInRecurrentLoop(*nodeIter);
                if (iId >= 0)
                {
					
					if (! accessed[iId])
					{
						newList.insert(newList.end(), m_recurrentInfo[iId].m_recurrentNodes.begin(), m_recurrentInfo[iId].m_recurrentNodes.end());
						accessed[iId] = true;
					}

                    /*if (prevId != iId && vRecurrentTmp.size() > 0)
                    {
                        newList.insert(newList.end(), vRecurrentTmp.begin(), vRecurrentTmp.end());
                        vRecurrentTmp.clear();
                    }

                    if (vTmp.size() > 0)
                    {
                        newList.insert(newList.end(), vTmp.begin(), vTmp.end());
                        vTmp.clear();
                    }

                    vRecurrentTmp.push_back(*nodeIter);

                    prevId = iId;*/
                }
                else
                {
                    //vTmp.push_back(*nodeIter);
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
            if (m_inputs.find(rootNode) == m_inputs.end()) //not found
            {
                std::list<ComputationNodePtr> inputs;

                std::list<ComputationNodePtr>&  nodes = GetEvalOrder(rootNode);
                for (auto nodeIter=nodes.begin(); nodeIter != nodes.end(); nodeIter++)
                {
                    ComputationNodePtr node = (*nodeIter);
                    if (node->OperationName() == InputValue<ElemType>::TypeName() /*L"InputValue"*/ || 
                        node->OperationName() == SparseInputValue<ElemType>::TypeName())
                        inputs.push_back(node);
                }
                m_inputs[rootNode] = inputs;
            }

            if (m_learnableParameters.find(rootNode) == m_learnableParameters.end()) //not found
            {
                std::list<std::wstring> learnableParameterNames;
                std::list<ComputationNodePtr> learnableParameters;

                std::list<ComputationNodePtr>&  nodes = GetEvalOrder(rootNode);;
                for (auto nodeIter=nodes.begin(); nodeIter != nodes.end(); nodeIter++)
                {
                    ComputationNodePtr node = (*nodeIter);
                    if ((node->OperationName() == LearnableParameter<ElemType>::TypeName() && node->NeedGradient())  ||
                        (node->OperationName() == SparseLearnableParameter<ElemType>::TypeName() && node->NeedGradient()))
                        learnableParameterNames.push_back(node->NodeName());
                }

                //we need to sort it so that we get consistent order when load it from saved file
                learnableParameterNames.sort();
                for (auto nodeNameIter=learnableParameterNames.begin(); nodeNameIter != learnableParameterNames.end(); nodeNameIter++)
                {
                    learnableParameters.push_back(GetNodeFromName((*nodeNameIter)));
                }

                m_learnableParameters[rootNode] = learnableParameters;
            }
        }

        void AddNodeToNet(const ComputationNodePtr nodePtr)
        {
            if (m_nameToNodeMap.find(nodePtr->NodeName()) != m_nameToNodeMap.end()) //found
                throw std::runtime_error("Duplicated computation node name.");

            m_nameToNodeMap[nodePtr->NodeName()] = nodePtr;
        }

public: // public so PTask can use eval/gradient order, and pre-compute matrix sizes
        void ClearGradientForAllNodes(const ComputationNodePtr rootNode) 
        {
            try{
                std::list<ComputationNodePtr>& allNodes = GetGradientCalcOrder(rootNode);

                for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
                {
                    (*nodeIter)->ClearGradientForChildren(m_actMiniBSize);
                }
                for (auto nodeIter = m_recurrentInfo.begin(); nodeIter != m_recurrentInfo.end(); nodeIter++)
                {
                    (*nodeIter).m_completedGradient = false;
                }
                for (int i = 0; i < m_recurrentInfo.size(); i++)
                {
                    m_recurrentInfo[i].m_completedGradient = false;
                }
            }
            catch (...)
            {
                fprintf(stderr, "Error in ClearGradientForAllNodes");
                throw;
            }
        }

        std::list<ComputationNodePtr>& GetEvalOrder(const ComputationNodePtr rootNode)
        {
            if (!rootNode) 
                throw std::logic_error("rootNode is pointing to a nullptr.");

            return GetCalcOrder(rootNode, m_cacheEvalOrders, true);
        }

        std::list<ComputationNodePtr>& GetEvalOrder(const ComputationNodePtr rootNode, std::vector<ComputationNodePtr>& recurrentNodes)
        {
            if (!rootNode) 
                throw std::logic_error("rootNode is pointing to a nullptr.");

            return GetCalcOrder(rootNode, m_cacheEvalOrders, true, recurrentNodes);
        }

        std::list<ComputationNodePtr>& GetGradientCalcOrder(const ComputationNodePtr rootNode)
        {
            if (!rootNode) 
                throw std::logic_error("rootNode is pointing to a nullptr.");

            return GetCalcOrder(rootNode, m_cacheGradientCalcOrders, false);
        }

        /**
        The below is used for sentence boundary information passed from reader. 
        This information can be used to reset RNN state or disregard gradient proprogation such as those
        used in reinforcement learning
        */
        Matrix<ElemType> mSentenceBoundary;  /// sentence boudary information. passed from reader
        Matrix<ElemType> mExistsBeginOrNoLabels;  /// whether there is a sentence begining or no_label at a time. note that one time can include many sentences. 
protected:
        std::list<ComputationNodePtr>& GetCalcOrder(const ComputationNodePtr rootNode, std::map<const ComputationNodePtr, std::list<ComputationNodePtr>>& orderMap, const bool forwardCompute) 
        {
            const ComputationNodePtr key = rootNode;
            if (orderMap.find(key) == orderMap.end()) //not found
            {
                orderMap[key] = rootNode->EnumerateNodes(forwardCompute);
            }
            return orderMap[key];
        }

        std::list<ComputationNodePtr>& GetCalcOrder(const ComputationNodePtr rootNode, std::map<const ComputationNodePtr, std::list<ComputationNodePtr>>& orderMap, const bool forwardCompute,
            std::vector<ComputationNodePtr>  & rootRecurrentNodes) 
        {
            const ComputationNodePtr key = rootNode;
            std::list<ComputationNodePtr> listNodes; 
            if (orderMap.find(key) == orderMap.end()) //not found
            {
                rootRecurrentNodes.clear();
                listNodes = rootNode->EnumerateNodes(forwardCompute, rootRecurrentNodes);

                orderMap[key] = listNodes;

            }
            return orderMap[key];
        }

        DEVICEID_TYPE m_deviceId;
        unsigned long m_randomSeedOffset;

        std::vector<ComputationNodePtr> m_features;
        std::vector<ComputationNodePtr> m_labels;
        std::vector<ComputationNodePtr> m_finalCriteria;
        std::vector<ComputationNodePtr> m_evalNodes;
        std::vector<ComputationNodePtr> m_outputNodes;
        std::vector<ComputationNodePtr> m_nodesReqMultiSeqHandling;
        std::vector<RecurrentInfo>      m_recurrentInfo;

        int m_actMiniBSize; 
        size_t m_nbrSlicesInEachRecurrentIteration; 


        std::map<const ComputationNodePtr, bool> m_built;
        std::map<const std::wstring, ComputationNodePtr, nocase_compare> m_nameToNodeMap;

        std::map<const ComputationNodePtr, std::list<ComputationNodePtr>> m_cacheEvalOrders;
        std::map<const ComputationNodePtr, std::list<ComputationNodePtr>> m_cacheGradientCalcOrders;

        std::map<const ComputationNodePtr, std::list<ComputationNodePtr>> m_inputs;
        std::map<const ComputationNodePtr, std::list<ComputationNodePtr>> m_learnableParameters;
    };

    template class ComputationNetwork<float>; 
    template class ComputationNetwork<double>;

}}}
