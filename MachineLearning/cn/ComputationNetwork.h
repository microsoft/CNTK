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
#include "ComputationNode.h"
#include "TrainingCriterionNode.h"
#include "CompositeComputationNode.h"
#include "EvaluationCriterionNode.h"
#include "File.h"
#include "Matrix.h"
#include "commandArgUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    template<class ElemType>
    class ComputationNetwork
    {
    protected:
        typedef ComputationNode<ElemType>* ComputationNodePtr;
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
        ComputationNetwork(DEVICEID_TYPE deviceId=AUTOPLACEMATRIX) : m_deviceId(deviceId)
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

        void SetDeviceID(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX)
        {
            m_deviceId = deviceId;  
            if (m_deviceId == AUTOPLACEMATRIX)
                m_deviceId = Matrix<ElemType>::GetBestGPUDeviceId();
        }

        DEVICEID_TYPE GetDeviceID() {return m_deviceId;}
        unsigned long GetRandomSeedOffset() {return m_randomSeedOffset;}
        void SetRandomSeedOffset(unsigned long value) {m_randomSeedOffset = value;}

        void SaveToFile(const std::wstring& fileName, const FileOptions fileFormat = FileOptions::fileOptionsBinary)
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
                    fstream << nodePtr->Inputs(i)->NodeName();
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

        virtual void LoadFromFile(const std::wstring& fileName, const FileOptions fileFormat = FileOptions::fileOptionsBinary)
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
                    ComputationNodePtr childNodePtr0, childNodePtr1, childNodePtr2, childNodePtr3, childNodePtr4;
                    switch (numChildren)
                    {
                    case 1:
                        childNodePtr0 = GetNodeFromName(childrenNames[0]);
                        nodePtr->AttachInputs(childNodePtr0);
                        break;
                    case 2:
                        childNodePtr0 = GetNodeFromName(childrenNames[0]);
                        childNodePtr1 = GetNodeFromName(childrenNames[1]);
                        nodePtr->AttachInputs(childNodePtr0, childNodePtr1);
                        break;
                    case 3:
                        childNodePtr0 = GetNodeFromName(childrenNames[0]);
                        childNodePtr1 = GetNodeFromName(childrenNames[1]);
                        childNodePtr2 = GetNodeFromName(childrenNames[2]);
                        nodePtr->AttachInputs(childNodePtr0, childNodePtr1, childNodePtr2);
                        break;
                    case 4:
                        childNodePtr0 = GetNodeFromName(childrenNames[0]);
                        childNodePtr1 = GetNodeFromName(childrenNames[1]);
                        childNodePtr2 = GetNodeFromName(childrenNames[2]);
                        childNodePtr3 = GetNodeFromName(childrenNames[3]);
                        nodePtr->AttachInputs(childNodePtr0, childNodePtr1, childNodePtr2, childNodePtr3);
                        break;
                    case 5:
                        childNodePtr0 = GetNodeFromName(childrenNames[0]);
                        childNodePtr1 = GetNodeFromName(childrenNames[1]);
                        childNodePtr2 = GetNodeFromName(childrenNames[2]);
                        childNodePtr3 = GetNodeFromName(childrenNames[3]);
                        childNodePtr4 = GetNodeFromName(childrenNames[4]);
                        nodePtr->AttachInputs(childNodePtr0, childNodePtr1, childNodePtr2, childNodePtr3, childNodePtr4);
                        break;
                    default:
                        throw std::logic_error("Invalid number of children.");
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
            

            ValidateNetwork();  //some internal values in the nodes are computed during validation
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
            else if (nodeType == TimesNode<ElemType>::TypeName())
                newNode = new TimesNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
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
            else if (nodeType == GMMLogLikelihoodNode<ElemType>::TypeName())
                newNode = new GMMLogLikelihoodNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
			else if (nodeType == CosDistanceWithNegativeSamplesNode<ElemType>::TypeName())
				newNode = new CosDistanceWithNegativeSamplesNode<ElemType>(fstream, modelVersion, m_deviceId, nodeName);
            else
            {
                fprintf(stderr, "Error creating new ComputationNode of type %ls, with name %ls\n", nodeType.c_str(), nodeName.c_str());
                throw std::invalid_argument("Invalid node type.");
            }
            
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
            else if (nodeType == TimesNode<ElemType>::TypeName())
                newNode = new TimesNode<ElemType>(m_deviceId, nodeName);
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
			else if (nodeType == CosDistanceWithNegativeSamplesNode<ElemType>::TypeName())
				newNode = new CosDistanceWithNegativeSamplesNode<ElemType>(m_deviceId, nodeName);
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

        ComputationNodePtr CrossEntropyWithSoftmax (const ComputationNodePtr label, const ComputationNodePtr prediction, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new CrossEntropyWithSoftmaxNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(label, prediction);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr ClassCrossEntropyWithSoftmax (const ComputationNodePtr label, const ComputationNodePtr prediction, const ComputationNodePtr matrix, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new ClassBasedCrossEntropyWithSoftmaxNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(label, prediction, matrix);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr CrossEntropy (const ComputationNodePtr label, const ComputationNodePtr prediction, const std::wstring nodeName = L"")
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


        ComputationNodePtr Scale (const ComputationNodePtr scalar, const ComputationNodePtr matrix, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new ScaleNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(scalar, matrix);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr Times (const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new TimesNode<ElemType>(m_deviceId, nodeName));
            newNode->AttachInputs(a, b);
            AddNodeToNet(newNode);
            return newNode;
        }

        ComputationNodePtr ElementTimes (const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"")
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

        ComputationNodePtr RowSlice(const ComputationNodePtr a, const size_t start_index, const size_t num_rows, const std::wstring nodeName = L"")
        {
            ComputationNodePtr newNode(new RowSliceNode<ElemType>(m_deviceId, start_index, num_rows, nodeName));
            newNode->AttachInputs(a);
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
        
        ComputationNodePtr LookupTable (const ComputationNodePtr dictionary, const ComputationNodePtr input, const std::wstring nodeName = L"")
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

        ComputationNodePtr GetNodeFromName(const std::wstring& name)  const
        {
            auto iter = m_nameToNodeMap.find(name);
            if (iter != m_nameToNodeMap.end()) //found
                return iter->second;
            else  //should never try to get a node from nonexisting name
                throw std::runtime_error("GetNodeFromName: Node name does not exist.");
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
                do{
                    bLoopCompleted = true;
                    for (auto nodeIter=recurrentNodes.begin(); nodeIter != recurrentNodes.end(); nodeIter++)
                    {
                        (*nodeIter)->EvaluateThisNode(iCnt); 

                        (*nodeIter)->UpdateEvalTimeStamp();

                    }

                    iCnt ++;
                }while (iCnt < iMBSize);

                m_recurrentInfo[iLoopId].m_completedEvaluate = true;
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

            for (int i=0; i < m_recurrentInfo.size(); i++)
            {
                m_recurrentInfo[i].m_completedEvaluate = false;
            }

            for (auto nodeIter=allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
            {
                (*nodeIter)->SetNbrSlicesInEachRecurrentIteration(m_nbrSlicesInEachRecurrentIteration);
                if ((*nodeIter)->OperationName() == L"Delay")
                {
                    for (size_t i = 0; i < m_nbrSlicesInEachRecurrentIteration; i++)
                    {
                        (*nodeIter)->ResetBound(i, m_sentenceEnd[i]);
                    }
                    if (m_sentenceEnd[0] <= m_actMiniBSize)
                    {
                        (*nodeIter)->Reset();
                    } else
                    {
                        (*nodeIter)->NotReset();
                    }
                }
            }

            for (auto nodeIter=allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
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
                    (*nodeIter)->EvaluateThisNode(); // we manage time stamp here so that derived classes don't need to worry about it
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
            m_sentenceEnd.assign(aSize, m_actMiniBSize/aSize);
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

        virtual void ComputeGradient(const ComputationNodePtr rootNode)
        {
            if (rootNode->FunctionValues().GetNumElements() != 1)
                throw std::runtime_error("ComputeGradient: The root of the Gradient computation must evaluate to R1 value.");

            //run forward pass first
            Evaluate(rootNode);

            ClearGradientForAllNodes(rootNode);

            //run backward pass
            std::list<ComputationNodePtr>& allNodes = GetGradientCalcOrder(rootNode);
            rootNode->GradientValues().Resize(1,1);
            rootNode->GradientValues().SetValue(1);

            for (auto nodeIter=allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
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

        inline std::vector<ComputationNodePtr>& EvaluationNodes() {return m_evalNodes;}

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
            for (int i=0; i<m_evalNodes.size(); i++)
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

        // Validate - Validate the network 
        void ValidateNetwork(bool allowFragment=false)
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
            }
        }

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
			for (std::map<const ComputationNodePtr, std::list<ComputationNodePtr>>::iterator it = m_cacheEvalOrders.begin(); it != m_cacheEvalOrders.end(); ++it)
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
            std::list<ComputationNodePtr>& allNodes = GetGradientCalcOrder(rootNode);

            for (auto nodeIter=allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
            {
                (*nodeIter)->ClearGradientForChildren(m_actMiniBSize);
            }
            for (auto nodeIter=m_recurrentInfo.begin(); nodeIter != m_recurrentInfo.end(); nodeIter++)
            {
                (*nodeIter).m_completedGradient = false;
            }
            for (int i=0; i < m_recurrentInfo.size(); i++)
            {
                m_recurrentInfo[i].m_completedGradient = false;
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
        vector<size_t> m_sentenceEnd;
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
