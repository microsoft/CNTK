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
#include "BrainScriptObjects.h"
#include "BrainScriptEvaluator.h"   // TODO: move (I)ConfigRecord to BrainScriptConfig that only has the config-related stuff (ConfigValuePtr and IConfigRecord, possibly need to do the same for Array and Lambda)

//#include "MatrixPool.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class ComputationNetwork : public BS::Object, public BS::HasToString, public BS::IConfigRecord
{
protected:
    typedef std::pair<ComputationNodeBasePtr, ComputationNodeBasePtr> ComputationArc;

    typedef struct stRecurrentInfo
    {
        std::vector<ComputationNodeBasePtr> m_recurrentNodes;
        std::vector<ComputationNodeBasePtr> m_recurrentNodesForForward;
        ComputationNodeBasePtr m_sourceNode;
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
        if (m_deviceId == AUTOPLACEMATRIX)  // TODO: code dup with SetDeviceId()
            m_deviceId = Matrix<float>::GetBestGPUDeviceId();
        m_nbrSlicesInEachRecurrentIteration = 1;
    }

    virtual ~ComputationNetwork()
    {
        ClearNet();
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    static bool IsSmaller(const ComputationNodeBasePtr lhs, const ComputationNodeBasePtr rhs)
    {
        return lhs->GetVisitedOrder() < rhs->GetVisitedOrder();
    }

    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    void ClearNet();

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

            const ComputationNodeBasePtr nodePtr = GetNodeFromName(nodeName);
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
            ComputationNodeBasePtr nodePtr = nodeIter->second;
            nodePtr->DumpNodeInfo(printValues, fstream);
        }
    }

    void DumpNodeInfoToFile(const vector<ComputationNodeBasePtr>& nodes,
                            const bool printValues,
                            const std::wstring outputFile)
    {
        ValidateNetwork(); //some internal values in the nodes are computed during validation

        File fstream(outputFile,
                     FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            ComputationNodeBasePtr nodePtr = *nodeIter;
            nodePtr->DumpNodeInfo(printValues, fstream);
        }
    }

    // -----------------------------------------------------------------------
    // topological plot [erw]
    // TODO: Can this be a separate class? Can it be moved to a CPP?
    // -----------------------------------------------------------------------

private:
    wstring FormSpecialNodes(wstring style, std::vector<ComputationNodeBasePtr>& specialNodes);
public:
    void DescribeNetworkUsingDot(std::list<ComputationArc>& arcs, std::wstring outFile);
    void PlotNetworkTopology(const std::wstring outputFile); //  [1/13/2015 erw] plot network topology using dot language

    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    void SetDeviceID(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
    {
        m_deviceId = deviceId;
        if (m_deviceId == AUTOPLACEMATRIX)
            m_deviceId = Matrix<float>::GetBestGPUDeviceId();
    }

    DEVICEID_TYPE GetDeviceID() { return m_deviceId; }

    unsigned long GetRandomSeedOffset() { return m_randomSeedOffset; }
    void SetRandomSeedOffset(unsigned long value) { m_randomSeedOffset = value; }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    size_t GetActualMBSize()
    {
        size_t actualMBSize = 0;

        const auto & featureNodes = this->FeatureNodes();   // TODO: a getter; should be called GetFeatureNodes()
        for (auto nodeIter = featureNodes.begin(); nodeIter != featureNodes.end(); nodeIter++)
            actualMBSize = max(actualMBSize, (*nodeIter)->GetNumCols());

        return actualMBSize;
    }

    // -----------------------------------------------------------------------
    // serialization
    // -----------------------------------------------------------------------

    // TODO: how does the file distinguish float vs double nodes?
    void SaveToFile(const std::wstring& fileName, const FileOptions fileFormat = FileOptions::fileOptionsBinary) const;
private:
    void SaveToFileImpl(const std::wstring& fileName, const FileOptions fileFormat) const;
public:

    void LoadPersistableParametersFromFile(const std::wstring& fileName, const bool requireValidation = true,
                                           const FileOptions fileFormat = FileOptions::fileOptionsBinary);
    template<typename ElemType>
    void LoadFromFile(const std::wstring& fileName, const FileOptions fileFormat = FileOptions::fileOptionsBinary,
                      const bool bAllowNoCriterionNode = false, ComputationNetwork* anotherNetwork = nullptr);

#pragma region Network Modification

    void SetLearnableNodesBelowNeedGradient(const bool needGradient, const ComputationNodeBasePtr rootNode = nullptr);

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
    template<class ElemType>
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
    template<class ElemType>
    static void InitLearnableParametersFromFile(const shared_ptr<ComputationNode<ElemType>> node,
                                                const std::wstring & initFromFilePath,
                                                DEVICEID_TYPE deviceId)    // TODO: why not just use node->m_deviceId?
    {
        size_t numRows = 0;
        size_t numCols = 0;
        ElemType *pArray = LoadArrayFromTextFile<ElemType>(msra::strfun::utf8(initFromFilePath), numRows, numCols); // TODO: change pathname to wstring
        node->FunctionValues().SetValue(numRows, numCols, pArray, matrixFlagNormal, deviceId);
        delete[] pArray;    // TODO: use std::vector to avoid mem leak on error
    }
    template<class ElemType>
    void InitLearnableParametersFromFile(const shared_ptr<ComputationNode<ElemType>> node, const std::string & initFromFilePath)   // TODO: remove this method or change pathname to wstring
    {
        InitLearnableParametersFromFile(node, msra::strfun::utf16(initFromFilePath), this->GetDeviceID());
    }

    // -----------------------------------------------------------------------
    // node construction
    // -----------------------------------------------------------------------

    // non-static version needed because it accesses m_randomSeedOffset
    // Excessively used by SimpleNetworkBuilder, but always after CreateLearnableParameter(), so we should really absorb it there
    template<typename ElemType>
    void InitLearnableParameters(const ComputationNodeBasePtr node,
                                 const bool uniformInit,
                                 const unsigned long randomSeed,
                                 const ElemType initValueScale,
                                 bool initOnCPUOnly = false);

    // -----------------------------------------------------------------------
    // network editing
    // -----------------------------------------------------------------------

    void DeleteNode(const std::wstring & nodeName)
    {
        //so that deleted node will not be referenced
        ClearCaches();

        ComputationNodeBasePtr nodeToDelete = GetNodeFromName(nodeName);

        //first delete links, if this node is involved, the whole connection will be removed
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodeBasePtr node = nodeIter->second;
            for (size_t i = 0; i < node->ChildrenSize(); i++)
            {
                ComputationNodeBasePtr child = node->GetChildren()[i];

                //nodeToDelete is a child
                if (child == nodeToDelete)
                {
                    // this used to call DetatchInputs(), but it's better for MEL to retain other inputs
                    node->SetInput(i, nullptr);
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

        ComputationNodeBasePtr nodeToRename = GetNodeFromName(nodeNameOrig);

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

    template<typename N>
    static shared_ptr<N> AsNodePtr(const ComputationNodeBasePtr & inode)
    {
        return dynamic_pointer_cast<N>(inode);
    }
    template<typename N>
    static bool IsNodePtr(const ComputationNodeBasePtr & inode)
    {
        return AsNodePtr<N>(inode) != nullptr;
    }

    // TODO: comment what this function does. Seems to either initialize LearnableParameters or precompute nodes.
    ComputationNodeBasePtr SetNodeValue(const std::wstring & nodeName, const double value);

    // -----------------------------------------------------------------------
    // network editing
    // -----------------------------------------------------------------------

    ComputationNodeBasePtr CopyNode(const ComputationNetwork & fromNet,
                                const std::wstring fromName,
                                std::wstring toName = L"",
                                const CopyNodeFlags flags = CopyNodeFlags::copyNodeAll)
    {
        if (toName == L"") {
            toName = fromName;
        }

        ComputationNodeBasePtr pFromNode = fromNet.GetNodeFromName(fromName);
        ComputationNodeBasePtr pToNode;

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
    void CopySubTree(const ComputationNetwork & fromNet,
                     const std::wstring fromName, std::wstring toNamePrefix = L"",
                     const CopyNodeFlags flags = copyNodeAll)
    {
        if (!(flags & CopyNodeFlags::copyNodeValue))
            LogicError("CopySubTree: you cannot copy a tree without copying the node values.");

        ComputationNodeBasePtr fromRoot = fromNet.GetNodeFromName(fromName);

        std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(fromRoot);
        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            ComputationNodeBasePtr fromNode = *nodeIter;
            wstring fromNodeName = fromNode->NodeName();
            wstring toNodeName = toNamePrefix + fromNodeName;

            ComputationNodeBasePtr toNode = CopyNode(fromNet, fromNodeName,
                                                  toNodeName,
                                                  CopyNodeFlags::copyNodeValue);

            if (flags & CopyNodeFlags::copyNodeChildren)
            {
                //copy the children structure but use the new nodes generated
                for (int i = 0; i < fromNode->ChildrenSize(); i++)
                    toNode->SetInput(i, GetNodeFromName(toNamePrefix + fromNode->GetChildren()[i]->NodeName()));
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
    // node access
    // -----------------------------------------------------------------------

    bool NodeNameExist(const std::wstring& name) const
    {
        auto iter = m_nameToNodeMap.find(name);
        return (iter != m_nameToNodeMap.end());
    }

    ComputationNodeBasePtr GetNodeFromName(const std::wstring& name, ComputationNetwork* anotherNetwork = nullptr, bool bPanic = true) const
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
    std::vector<ComputationNodeBasePtr> GetNodesFromName(const std::wstring& name) const
    {
        std::vector<ComputationNodeBasePtr> nodes;
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

    int FindInRecurrentLoop(const ComputationNodeBasePtr startNode, vector<ComputationNodeBasePtr>& recurrentNodes)
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

    int FindInRecurrentLoop(const ComputationNodeBasePtr startNode)
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

    bool IsFuncValueOlderThanInputs(const std::vector<ComputationNodeBasePtr>& recurrentNodes);

    void EvaluateLoop(std::list<ComputationNodeBasePtr>& /*allNodes*/, const ComputationNodeBasePtr startNode)
    {
        std::vector<ComputationNodeBasePtr> recurrentNodes;
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

    bool IsTypicalCriterionNode(ComputationNodeBasePtr nodePtr);

    void SetNodesReqMultiSeqHandling();

    void Evaluate(const ComputationNodeBasePtr rootNode)
    {
        BuildAndValidateNetwork(rootNode);

        std::list<ComputationNodeBasePtr>& allNodes = GetEvalOrder(rootNode);

#ifdef DISPLAY_DEBUG
        for (auto nodeIter=allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
            fprintf (stderr, "Evaluate Node: %s\n",(msra::strfun::utf8 ((*nodeIter)->NodeName())).c_str());
#endif

        for (int i = 0; i < m_recurrentInfo.size(); i++)
            m_recurrentInfo[i].m_completedEvaluate = false;

        for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
        {
            // TODO: nbrSlices set once to the same value for all nodes each evaluation--is it ever changed later?
            (*nodeIter)->SetNbrSlicesInEachRecurrentIteration(m_nbrSlicesInEachRecurrentIteration);
            if ((*nodeIter)->ReqMultiSeqHandling())
                    (*nodeIter)->ResetBound(&m_SentenceBoundary, &m_minibatchPackingFlag);
        }

        for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
        {
            // TODO: is this the frame-by-frame evaluation? Why is there no comment here??
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
                // TODO: is this the whole-batch evaluation?
                (*nodeIter)->EvaluateThisNodeGivenInputs(); 
                (*nodeIter)->UpdateEvalTimeStamp();
            }
        }
    }

    void SetActualMiniBatchSize(const size_t aSize, vector<ComputationNodeBasePtr>* featNodes = nullptr)
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
                size_t nr = (*ptr)->GetNumRows();
                (*ptr)->Resize(nr, aSize);
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

    void ComputeGradientLoop(std::list<ComputationNodeBasePtr>& /*allNodes*/, const ComputationNodeBasePtr startNode)
    {
        std::vector<ComputationNodeBasePtr> recurrentNodes;
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

    template<typename ElemType>
    void ComputeGradient(const ComputationNodeBasePtr rootNode, 
                         bool bResetToOne = true,  /// true if reset the gradient of rootnode to 1.0
                         const Matrix<ElemType>* rootGradientInitValue = nullptr,
                         bool bClearGradient = true,
                         bool resetTimeStampAfterComputation = false
                    )
    {
        if (bResetToOne && (rootNode->GetNumRows() != 1 || rootNode->GetNumCols() != 1))
            RuntimeError("ComputeGradient: The root of the Gradient computation must evaluate to R1 value.");

        //run forward pass first
        Evaluate(rootNode);

        if (bClearGradient)
            ClearGradientForAllNodes(rootNode);

        //run backward pass
        std::list<ComputationNodeBasePtr>& allNodes = GetGradientCalcOrder(rootNode);
            
        // TODO: do a runtime check for float vs. double. Also use the Is/AsPtr macros
        if (bResetToOne)
        {
            dynamic_pointer_cast<ComputationNode<ElemType>>(rootNode)->GradientValues().Resize(1, 1);   // TODO: make this a function of ComputationNode; but first need to get rid of Matrix<ElemType> here, or make it a local template parameter
            dynamic_pointer_cast<ComputationNode<ElemType>>(rootNode)->GradientValues().SetValue(1);
        }

        if (rootGradientInitValue != nullptr)
            dynamic_pointer_cast<ComputationNode<ElemType>>(rootNode)->GradientValues().SetValue(*rootGradientInitValue);

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
    void PrintComputationTree(const ComputationNodeBasePtr rootNode,
                              const bool forwardCompute,
                              const bool printMatrices = false)
    {
        std::list<ComputationNodeBasePtr> nodes;
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
            ComputationNodeBasePtr node = (*nodeIter);
            node->PrintSelf(printMatrices);
        }
    }

    // -----------------------------------------------------------------------
    // network editing
    // -----------------------------------------------------------------------

    void RenameNode(const ComputationNodeBasePtr node, const std::wstring newNodeName)
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

    void RebuildNetwork(const ComputationNodeBasePtr rootNode)
    {
        ClearCaches();
        BuildAndValidateNetwork(rootNode);
    }

    // -----------------------------------------------------------------------
    // node-group access
    // -----------------------------------------------------------------------

    std::list<ComputationNodeBasePtr> & InputNodes(const ComputationNodeBasePtr rootNode, bool bNoBuild = false)
    {
        if (bNoBuild == false)
            BuildAndValidateNetwork(rootNode);
        return m_inputs[rootNode];
    }

    std::list<ComputationNodeBasePtr> & LearnableNodes(const ComputationNodeBasePtr rootNode)
    {
        BuildAndValidateNetwork(rootNode);
        return m_learnableParameters[rootNode];
    }

    inline std::vector<ComputationNodeBasePtr> & FeatureNodes()        { return m_features; }
    inline std::vector<ComputationNodeBasePtr> & LabelNodes()          { return m_labels; }
    inline std::vector<ComputationNodeBasePtr> & FinalCriterionNodes() { return m_finalCriteria; }

    inline std::vector<ComputationNodeBasePtr> & TrainCriterionNodesFrom(wstring criterionNodeName)
    {
        ComputationNodeBasePtr node = this->GetNodeFromName(criterionNodeName);
        this->ValidateNetwork(node);
        if (node->GetNumRows() != 1 || node->GetNumCols() != 1)
            InvalidArgument("the trainCriterionNodeName specified in the config file is not a valid training criterion node.");
        m_tmpTrainCriterion.clear();
        m_tmpTrainCriterion.push_back(node);
        return m_tmpTrainCriterion;
    }

    inline std::vector<ComputationNodeBasePtr> & EvalCriterionNodesFrom(wstring criterionNodeName)
    {
        ComputationNodeBasePtr node = this->GetNodeFromName(criterionNodeName);
        this->ValidateNetwork(node);
        if (node->GetNumRows() != 1 || node->GetNumCols() != 1)
            InvalidArgument("the trainCriterionNodeName specified in the config file is not a valid training criterion node.");
        m_tmpEvalulationCriterion.clear();
        m_tmpEvalulationCriterion.push_back(node);
        return m_tmpEvalulationCriterion;
    }

    inline std::vector<ComputationNodeBasePtr> & NodesReqMultiSeqHandling() { return m_nodesReqMultiSeqHandling; }
    inline std::vector<ComputationNodeBasePtr> & EvaluationNodes()          { return m_evalNodes; }
    inline std::vector<ComputationNodeBasePtr> & OutputNodes()              { return m_outputNodes; }
    inline std::vector<ComputationNodeBasePtr> & PairNodes()                { return m_pairNodes; }

    inline std::vector<RecurrentInfo> & RecurrentNodes() { return m_recurrentInfo; }

    // -----------------------------------------------------------------------
    // node access
    // -----------------------------------------------------------------------

    size_t GetTotalNumberOfNodes() const { return m_nameToNodeMap.size(); }

    // TODO: could be a dup
    std::map<const std::wstring, ComputationNodeBasePtr, nocase_compare> & GetNameToNodeMap()    // specially for ExperimentalNetworkBuilder; don't use this otherwise
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
    void ChangeNode(wstring nodeName, ComputationNodeBasePtr newNode)
    {
        ComputationNodeBasePtr oldNode = GetNodeFromName(nodeName);
        if (oldNode->OperationName() != newNode->OperationName())
            InvalidArgument("newNode must have the same type as the old node.");

        //change children
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodeBasePtr node = nodeIter->second;
            for (int i = 0; i < node->ChildrenSize(); i++)
                if (node->GetChildren()[i] == oldNode)
                    node->SetInput(i, newNode);
        }

        //change name map
        m_nameToNodeMap[nodeName] = newNode;
        for (int i = 0; i < oldNode->ChildrenSize(); i++)
            newNode->SetInput(i, oldNode->GetChildren()[i]);

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
    void ReplaceLeafNode(wstring oldNodeName, ComputationNodeBasePtr newNode)
    {
        ComputationNodeBasePtr oldNode = GetNodeFromName(oldNodeName);

        // change the input of those nodes whose child is oldNode
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodeBasePtr node = nodeIter->second;
            for (int i = 0; i < node->ChildrenSize(); i++)
                if (node->GetChildren()[i] == oldNode)
                    node->SetInput(i, newNode);
        }
        m_nameToNodeMap[newNode->GetName()] = newNode;

        // now the old node becomes a orphan node , remove it
        DeleteNode(oldNodeName);
        //RemoveOrphanNode(oldNode);
    }

    void ReplaceFinalCriterionNode(wstring oldNodeName, ComputationNodeBasePtr newNode)
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
            if (m_nameToNodeMap.find(newNode->GetChildren()[i]->NodeName()) == m_nameToNodeMap.end())
                RuntimeError("Child node does not exist.");
            newNode->SetInput(i, m_nameToNodeMap[newNode->GetChildren()[i]->NodeName()]);
        }

        // Addes it to criterion node list.
        m_finalCriteria[index] = newNode;
        m_nameToNodeMap[newNode->NodeName()] = newNode;
    }

    void AddFeatureNode(ComputationNodeBasePtr featureNode)
    {
        wstring nodeName = featureNode->NodeName();
        if (NodeNameExist(nodeName))
            RuntimeError("AddFeatureNode: feature node already exists.");
        m_nameToNodeMap[nodeName] = featureNode;
        m_features.push_back(featureNode);
    }

    // We only remove the node, not delete it.
    void RemoveFeatureNode(ComputationNodeBasePtr featureNode)
    {
        wstring nodeName = featureNode->NodeName();
        if (!NodeNameExist(nodeName))
            RuntimeError("RemoveFeatureNode: feature node does not exist.");

        ClearCaches();

        // Removes links.
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); ++nodeIter)
        {
            ComputationNodeBasePtr node = nodeIter->second;
            for (size_t i = 0; i < node->ChildrenSize(); ++i)
            {
                ComputationNodeBasePtr child = node->GetChildren()[i];
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

    std::vector<ComputationNodeBasePtr> GetAllNodes() const
    {
        std::vector<ComputationNodeBasePtr> nodes;
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodeBasePtr node = nodeIter->second;
            nodes.push_back(node);
        }
        return nodes;
    }

    std::list<ComputationNodeBasePtr> GetNodesWithType(const wstring typeName, const ComputationNodeBasePtr rootNode = nullptr)
    {
        std::list<ComputationNodeBasePtr> nodesWithType;

        //find nodes from all available nodes
        if (rootNode == nullptr)
        {
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = nodeIter->second;
                if (node->OperationName() == typeName)
                    nodesWithType.push_back(node);
            }
        }
        else
        {
            //for calculating a specific node
            std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = (*nodeIter);
                if (node->OperationName() == typeName)
                    nodesWithType.push_back(node);
            }
        }

        return nodesWithType;
    }

private:
    template<class N> void GetNodesRequiringX(std::list<ComputationNodeBasePtr> & nodesRequirePreComputation, const ComputationNodeBasePtr rootNode, bool checkComputed);
public:
    //return list of nodes that require precomputation and not precomputed yet.
    // TODO: name has a grammar error, fix
    std::list<ComputationNodeBasePtr> GetNodesRequiringPreComputation(const ComputationNodeBasePtr rootNode = nullptr, bool checkComputed = true);
    //return list of nodes that require precomputation and not precomputed yet.
    // TODO: name has grammar error, fix
    std::list<ComputationNodeBasePtr> GetNodesRequiringBatchMode(const ComputationNodeBasePtr rootNode = nullptr, bool checkComputed = true);

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
            for (ComputationNodeBasePtr & node : FinalCriterionNodes())
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
            for (ComputationNodeBasePtr node : OutputNodes())
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
            for (ComputationNodeBasePtr node : EvaluationNodes())
            {
                if (!allowFragment)
                    FormRecurrentLoops(node);
                ValidateNetwork(node);
            }
        }
    }

    void ValidateNetwork(const ComputationNodeBasePtr rootNode)
    {
        fprintf(stderr, "\n\nValidating node %ls \n", rootNode->NodeName().c_str());

        std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode);

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            (*nodeIter)->Validate();
        }

        fprintf(stderr, "\n\n");
    }

    void BuildAndValidateNetwork(const ComputationNodeBasePtr rootNode)
    {
        const ComputationNodeBasePtr key = rootNode;

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
    void AllocateMatrices(std::vector<ComputationNodeBasePtr>& evalRootNodes, std::vector<ComputationNodeBasePtr>& trainRootNodes)
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

    void AllocateEvalMatrices(ComputationNodeBasePtr rootNode)
    {
        FormRecurrentLoops(rootNode);

        std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode);

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            (*nodeIter)->RequestEvalMatrices(m_matrixPool);
            (*nodeIter)->ReleaseMatricesAfterEval(m_matrixPool);
        }
    }

    void AllocateGradientMatrices(ComputationNodeBasePtr rootNode)
    {
        //first, compute the number of parents for each node
        std::map<ComputationNodeBasePtr, int> numParents;

        std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode);

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            std::vector<ComputationNodeBasePtr> children = (*nodeIter)->GetChildren();
            for (int i = 0; i < children.size(); i++)
                numParents[children[i]] ++;
        }

        //now, simulate the gradient computation order to determine how to allocate matrices
        std::list<ComputationNodeBasePtr>& allNodes = GetGradientCalcOrder(rootNode);

        for (int i = 0; i < m_recurrentInfo.size(); i++)
            m_recurrentInfo[i].m_completedGradient = false;

        for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
        {
            std::vector<ComputationNodeBasePtr> recurrentNodes;
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

    void AllocateGradientMatricesForChildren(ComputationNodeBasePtr parentNode, std::map<ComputationNodeBasePtr, int>& numParents)
    {
        std::vector<ComputationNodeBasePtr> children = parentNode->GetChildren();
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

    bool UnitTest(const ComputationNodeBasePtr rootNode)
    {
        fprintf(stderr, "\n\n Unit test node %ls \n", rootNode->NodeName().c_str());

        std::list<ComputationNodeBasePtr>&  nodes = GetEvalOrder(rootNode);

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
    // BUGBUG: this only currently works for one ElemType, not both
    template<typename ElemType>
    void PerformSVDecomposition(const map<wstring, float>& SVDConfig);

public:
    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // TODO: make these templated on <ElemType> locally
    template<typename ElemType>
    void GetHistory(map<wstring, Matrix<ElemType>>& history, bool bLastTime = false)
    {
        //put all node info first
        Matrix<ElemType> hist;
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            shared_ptr<ComputationNode<ElemType>> nodePtr = dynamic_pointer_cast<ComputationNode<ElemType>>(nodeIter->second);
            if (nodePtr && nodePtr->GetHistory(hist, bLastTime))
                history[nodeIter->first] = hist;
        }
    };

    template<typename ElemType>
    void SetHistory(map<wstring, Matrix<ElemType>>& history)
    {
        //put all node info first
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            shared_ptr<ComputationNode<ElemType>> nodePtr = dynamic_pointer_cast<ComputationNode<ElemType>>(nodeIter->second);
            if (nodePtr && history.find(nodeIter->first) != history.end())
                nodePtr->SetHistory(history[nodeIter->first]);
        }
    };

    Matrix<float> & SentenceBoundary() { return m_SentenceBoundary; }

    vector<MinibatchPackingFlag> & MinibatchPackingFlags() { return m_minibatchPackingFlag; }

protected:
    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    // Copy constructor, should never be called.
#pragma warning (push)
#pragma warning (disable: 4702) // this function is flagged but unclear why
    ComputationNetwork(const ComputationNetwork& /*deepCopyFrom*/)
    {
        // TODO: can we just define it as private without implementation?
        LogicError("'ComputationNetwork(const ComputationNetwork& deepCopyFrom)' should never be called.");
    }
#pragma warning (pop)

    // Assignment operator, should never be called.
    ComputationNetwork& operator=(const ComputationNetwork& /*deepCopyFrom*/)
    {
        // TODO: can we just define it as private without implementation?
        LogicError("'ComputationNetwork& operator=(const ComputationNetwork& deepCopyFrom)' should never be called.");
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // The methods below determine evaluation order, which is tricky in presence of recurrent loops.
    // TODO: Can this be moved to a separate class, or at least a separate CPP?

    void ClearCalcOrderCaches();
    void MergeRecurrentLoops(const ComputationNodeBasePtr /*rootNode*/);
    // get the strong connected component from the graph
    void getStrongSCC(const ComputationNodeBasePtr rootNode);    // TODO: method names start uppercase
    void strongSCC(ComputationNodeBasePtr cur, std::list<ComputationNodeBasePtr>& sccStack, size_t& index, size_t& loopId);     // TODO: method names start uppercase
    void getLoopForwordOrder(std::unordered_set<ComputationNodeBasePtr>& visited, std::unordered_set<ComputationNodeBasePtr>& recStack, std::list<ComputationNodeBasePtr>& nodesStack, ComputationNodeBasePtr cur);   // TODO: method name
    //must be called before ValidateNetwork
    void FormRecurrentLoops(const ComputationNodeBasePtr rootNode);
    void DetermineLoopTypes();
    void ReorderLoops(std::list<ComputationNodeBasePtr>& nodes, const std::map<int, std::list<ComputationNodeBasePtr>>& /*recurrentNodes*/, const std::list<ComputationNodeBasePtr> & /*noRecurrentNodes*/);
    void CollectInputAndLeanableParameters(const ComputationNodeBasePtr rootNode);

    // -----------------------------------------------------------------------
    // node creation
    // -----------------------------------------------------------------------

public:

    // TODO: move these close to where they are used

    // add a node to m_nameToNodeMap[], which is our node holder
    // Duplicate node names are rejected.
    ComputationNodeBasePtr AddNodeToNet(const ComputationNodeBasePtr nodePtr)
    {
        //found
        // TODO: use .insert() and test result.second == false means not inserted since already exists
        if (m_nameToNodeMap.find(nodePtr->NodeName()) != m_nameToNodeMap.end())
            RuntimeError("Duplicated computation node name.");

        m_nameToNodeMap[nodePtr->NodeName()] = nodePtr;
        return nodePtr; // allows e.g. return AddNodeToNet(New...);
    }
    // TODO: not very nice--need to fix way more outside to get this right
    template<class N>
    shared_ptr<N> AddNodeToNetWithElemType(const shared_ptr<N> nodePtr)
    {
        return dynamic_pointer_cast<N>(AddNodeToNet(nodePtr));
    }

    template<class N, class... _Types>
    shared_ptr<N> AddNodeToNetAndAttachInputs(const shared_ptr<N> nodePtr, _Types&&... _Args)
    {
        nodePtr->AttachInputs(std::forward<_Types>(_Args)...);
        return AddNodeToNetWithElemType(nodePtr);
        //return nodePtr; // allows e.g. return AddNodeToNetAndAttachInputs(New..., inputs);
    }

public:

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    void ClearGradientForAllNodes(const ComputationNodeBasePtr rootNode)
    {
        std::list<ComputationNodeBasePtr>& allNodes = GetGradientCalcOrder(rootNode);

        for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
            (*nodeIter)->ClearGradientForChildren(m_actMiniBSize);

        //for (auto nodeIter = m_recurrentInfo.begin(); nodeIter != m_recurrentInfo.end(); nodeIter++)
        //    (*nodeIter).m_completedGradient = false;

        for (int i = 0; i < m_recurrentInfo.size(); i++)
            m_recurrentInfo[i].m_completedGradient = false;
    }

    std::list<ComputationNodeBasePtr>& GetEvalOrder(const ComputationNodeBasePtr rootNode)
    {
        if (!rootNode)
            LogicError("rootNode is pointing to a nullptr.");

        return GetCalcOrder(rootNode, m_cacheEvalOrders, true);
    }

    std::list<ComputationNodeBasePtr>& GetEvalOrder(const ComputationNodeBasePtr rootNode,
                                                    std::vector<ComputationNodeBasePtr>& recurrentNodes)
    {
        if (!rootNode)
            LogicError("rootNode is pointing to a nullptr.");

        return GetCalcOrder(rootNode, m_cacheEvalOrders, true, recurrentNodes);
    }

    std::list<ComputationNodeBasePtr>& GetGradientCalcOrder(const ComputationNodeBasePtr rootNode)
    {
        if (!rootNode)
            LogicError("rootNode is pointing to a nullptr.");

        return GetCalcOrder(rootNode, m_cacheGradientCalcOrders, false);
    }

protected:

    static std::list<ComputationNodeBasePtr>& GetCalcOrder(const ComputationNodeBasePtr rootNode,
                                                           std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>>& orderMap,
                                                           const bool forwardCompute)
    {
        const ComputationNodeBasePtr key = rootNode;

        //not found
        if (orderMap.find(key) == orderMap.end())
            orderMap[key] = rootNode->EnumerateNodes(forwardCompute);

        return orderMap[key];
    }

    static std::list<ComputationNodeBasePtr>& GetCalcOrder(const ComputationNodeBasePtr rootNode,
                                                           std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>>& orderMap,
                                                           const bool forwardCompute,
                                                           std::vector<ComputationNodeBasePtr> & rootRecurrentNodes)
    {
        const ComputationNodeBasePtr key = rootNode;
        std::list<ComputationNodeBasePtr> listNodes;

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

    // FixupInputMinibatchSize - go through all the inputs and make sure they have a consistent minibatch size (after creation)
    void FixupInputMinibatchSize();

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
    std::vector<ComputationNodeBasePtr> m_features;
    std::vector<ComputationNodeBasePtr> m_labels;
    std::vector<ComputationNodeBasePtr> m_finalCriteria;
    std::vector<ComputationNodeBasePtr> m_evalNodes;
    std::vector<ComputationNodeBasePtr> m_outputNodes;
    std::vector<ComputationNodeBasePtr> m_pairNodes; /// nodes for the children network to pair
    std::vector<ComputationNodeBasePtr> m_nodesReqMultiSeqHandling;
    vector<std::vector<ComputationNodeBasePtr>*> GetAllNodeGroups()    // get all groups to allow to iterate over all of them ...continue
    {
        return vector<std::vector<ComputationNodeBasePtr>*> { &m_features, &m_labels, &m_finalCriteria, &m_evalNodes, &m_outputNodes, &m_pairNodes, &m_nodesReqMultiSeqHandling };
    }

    std::vector<RecurrentInfo> m_recurrentInfo;

    /** temporary space
    */
    std::vector<ComputationNodeBasePtr> m_tmpTrainCriterion; /// array saving tempary query terms
    std::vector<ComputationNodeBasePtr> m_tmpEvalulationCriterion; /// array saving tempary query terms

    //used for sentence boundary information passed from reader to reset RNN state 
    Matrix<float> m_SentenceBoundary; // this matrix is always in CPU memory  --TODO: should rather be a matrix of some int
    // specify how the minibatch is packed for each sample
    vector<MinibatchPackingFlag> m_minibatchPackingFlag;

    int m_actMiniBSize;
    size_t m_nbrSlicesInEachRecurrentIteration;

    std::map<const ComputationNodeBasePtr, bool> m_built;
    std::map<const std::wstring, ComputationNodeBasePtr, nocase_compare> m_nameToNodeMap;   // this is the main container that holds this networks' nodes

    std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>> m_cacheEvalOrders;
    std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>> m_cacheGradientCalcOrders;

    std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>> m_inputs;
    std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>> m_learnableParameters;

    MatrixPool m_matrixPool;
};

}}}
