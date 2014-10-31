//
// <copyright file="ComputationNode.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include <Rpc.h>
#pragma comment(lib, "Rpcrt4.lib")

#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <algorithm>
#include <WinBase.h>
#include <assert.h>
#include "basetypes.h"

#include <Matrix.h>
#include "PTaskGraphBuilder.h"

//#define RNN_DEBUG 1
#define DEFAULT_HIDDEN_ACTIVITY 0.1

#ifndef NOT_IMPLEMENTED
#define NOT_IMPLEMENTED throw std::exception("Not implemented")
#endif

#pragma warning (disable: 4267)

//version number to control how to read and write 
#define CNTK_MODEL_VERSION_1 1
#define CURRENT_CNTK_MODEL_VERSION 1

namespace Microsoft { namespace MSR { namespace CNTK {

    enum CopyNodeFlags
    {
        copyNodeNull = 0, // invalid value
        copyNodeValue=1, // copy everything but the children links
        copyNodeChildren=2, // only copy over children links
        copyNodeAll=3, // copy everything
        copyNodeChildrenCrossNetwork=4, // allow a cross network child copy
    };

#pragma region base computation class

    template<class ElemType>
    class ComputationNode //Abstract Class that cannot be instantiated
    {
    protected:
        //std containers such as list and map does not support class reference so we need to use pointer
        typedef ComputationNode<ElemType>* ComputationNodePtr;
        int     m_loopId;
        size_t  m_samplesInRecurrentStep; 

        /// the order in reverse graph. 
        int     m_visitedOrder;  
		int m_index;
		int m_lowlink;
		bool m_visited;
		bool m_inStack;
		int m_indexInLoop;
		vector<size_t> m_SentenceEnd;
    public:
        ComputationNode(short deviceId): m_functionValues(deviceId), m_gradientValues(deviceId) 
        {
            m_deviceId = deviceId;
            m_loopId = -1;
            m_samplesInRecurrentStep = 1;
            m_visitedOrder = -1;
			m_index = -1;
			m_lowlink = -1;
			m_indexInLoop = 0;
			m_visited = false;
			m_inStack = false;
        }

        virtual ~ComputationNode()
        {
#ifdef DISPLAY_DEBUG
            fprintf (stderr, "Called Destructor NodeName: %s\n",(msra::strfun::utf8 (NodeName())).c_str());
#endif
        }

        virtual const std::wstring OperationName() const = 0;
        virtual void SaveToFile(File& fstream) const
        {
            fstream << OperationName() << NodeName();
        }

        virtual void LoadFromFile(File& /*fstream*/, const size_t /*modelVersion*/, const short deviceId = AUTOPLACEMATRIX)
        {
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        virtual void ComputeInputPartial(const size_t inputIndex) = 0;
        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/) 
        {
            NOT_IMPLEMENTED;
        }
        
        virtual void EvaluateThisNode() = 0;
        // evaluate only at time index timeIdxInSeq
        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/) 
        {
            NOT_IMPLEMENTED;
        }
        virtual void Validate() = 0;
        
        virtual void Reset() {}
        virtual void NotReset() {}

        virtual void AttachInputs(const ComputationNodePtr /*singleInput*/) 
        {
            throw std::logic_error("This operation does not support single input.");
        }

        virtual void AttachInputs(const ComputationNodePtr /*leftInput*/, const ComputationNodePtr /*rightInput*/) 
        {
            throw std::logic_error("This operation does not support two inputs.");
        }

        virtual void AttachInputs(const ComputationNodePtr /*leftInput*/, const ComputationNodePtr /*middleInput*/, const ComputationNodePtr /*rightInput*/) 
        {
            throw std::logic_error("This operation does not support three inputs.");
        }

        virtual void AttachInputs(const ComputationNodePtr /*firstInput*/, const ComputationNodePtr /*secondInput*/, const ComputationNodePtr /*thirdInput*/, const ComputationNodePtr /*fourthInput*/)
        {
            throw std::logic_error("This operation does not support four inputs.");
        }

        virtual void AttachInputs(const ComputationNodePtr /*firstInput*/, const ComputationNodePtr /*secondInput*/, const ComputationNodePtr /*thirdInput*/, 
            const ComputationNodePtr /*fourthInput*/, const ComputationNodePtr /*fifthInput*/)
        {
            throw std::logic_error("This operation does not support five inputs.");
        }

        virtual void DetachInputs()
        {
            m_children.resize(0);
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_functionValues.GetDeviceId() != deviceId)
                {
                    bool fEmpty = m_functionValues.GetNumElements() == 0;
                    m_functionValues.TransferFromDeviceToDevice(m_functionValues.GetDeviceId(), deviceId,true, fEmpty);
                }

                if (m_gradientValues.GetDeviceId() != deviceId)
                {
                    bool fEmpty = m_gradientValues.GetNumElements() == 0;
                    m_gradientValues.TransferFromDeviceToDevice(m_gradientValues.GetDeviceId(), deviceId,true, fEmpty);
                }
            }
        }

        //making them virtual so that nodes that only copy values from it's children (e.g., dropout) can be efficient in evaluation
        virtual const Matrix<ElemType>& FunctionValues() const {return m_functionValues;}
        virtual Matrix<ElemType>& FunctionValues() { return m_functionValues;}

        //return true if the node's value should be computed before the normal training. e.g., mean and invStd of input features.
        virtual bool RequirePreCompute() const { return false;}

        virtual void DumpNodeInfo(const bool /*printValues*/, File& fstream) const
        {
            WCHAR str[4096];
            wsprintf(str, L"\n%ws=%ws", NodeName().c_str(), OperationName().c_str());           
            fstream << wstring(str);

            if (!IsLeaf())
            {
                fstream << wstring(L"(");
                for (size_t i=0; i<ChildrenSize(); i++)
                {
                    if (i > 0)
                        fstream << wstring(L",");
                    wsprintf(str, L"%ws", Inputs(i)?Inputs(i)->NodeName().c_str():L"NULL");
                    fstream << wstring(str);
                }
                fstream << wstring(L")");
            }
        }

        virtual void SetFunctionAndGradientSize(const int numSamples) 
        {
            size_t numRows = m_functionValues.GetNumRows();
            if (numRows > 0 && numSamples > 0)
            {
                m_functionValues.Resize(numRows, numSamples); 
                m_gradientValues.Resize(numRows, numSamples); 
            }
        }
		void ResetBound(size_t indexInBatch, size_t frameNum)
		{
			m_SentenceEnd[indexInBatch] = frameNum;
		}
        void SetLoopId(const int id)
        {
            m_loopId = id;
        }
        void SetVisitedOrder(const int id)
        {
            m_visitedOrder = id;
        }
		void SetIndex (const size_t ind)
		{
			m_index = ind;
		}

		void Setlowlink (const size_t lowlink)
		{
			m_lowlink = lowlink;
		}

		void SetVisited (const bool visited)
		{
			m_visited = visited;
		}

		void SetInStack (const bool instack)
		{
			m_inStack = instack;
		}

		void SetIndexInLoop (const size_t index)
		{
			m_indexInLoop = index;
		}

		size_t GetIndex ()
		{
			return m_index;
		}

        size_t GetVisitedOrder()
        {
            return m_visitedOrder;
        }

        size_t Getlowlink ()
		{
			return m_lowlink;
		}

		size_t GetIndexInLoop ()
		{
			return m_indexInLoop;
		}
		bool isVisisted()
		{
			return m_visited;
		}

		bool isInStack()
		{
			return m_inStack;
		}
        int LoopId()
        {
            return m_loopId;
        }

        void SetNbrSlicesInEachRecurrentIteration(size_t bsz)
        {
            m_samplesInRecurrentStep = bsz;
			m_SentenceEnd.resize(bsz);
        }

        LONG64 UpdateEvalTimeStamp()
        {
            m_evalTimeStamp = InterlockedIncrement64(&s_timeStampCounter);
            return m_evalTimeStamp;
        }

        void ResetEvalTimeStamp()
        {
            m_evalTimeStamp = s_timeStampCounter;
        }

        //for debugging purpose
        virtual void PrintSelf(bool printMatrices = false) const
        {
            fprintf(stderr, "\n%ws[%lu, %lu] = %ws", NodeName().c_str(), FunctionValues().GetNumRows(),  FunctionValues().GetNumCols(), OperationName().c_str());           

            if (!IsLeaf())
            {
                fprintf(stderr, "(");           
                for (size_t i=0; i<ChildrenSize(); i++)
                {
                    if (i > 0)
                        fprintf(stderr, ", ");           
                    fprintf(stderr, "%ws[%lu, %lu]", Inputs(i)?Inputs(i)->NodeName().c_str():L"NULL", Inputs(i)->FunctionValues().GetNumRows(), Inputs(i)->FunctionValues().GetNumCols());
                }
                fprintf(stderr, ")");           
            }

            if (printMatrices)
            {
                fprintf (stderr, "\n    $$$$ Function Values\n");
                FunctionValues().Print("FunctionValue");

                fprintf (stderr, "\n    $$$$ Gradient Values\n");
                GradientValues().Print("GradientValue");
            }
        }

        const std::wstring& NodeName() const { return m_nodeName;}
        std::wstring& NodeName() { return m_nodeName;}
        
        const std::wstring& DerivativeName() const {return L"D_" + m_nodeName;}

        const Matrix<ElemType>& GradientValues() const {return m_gradientValues;}
        Matrix<ElemType>& GradientValues() {return m_gradientValues;}

        bool IsLeaf() const {return m_children.size() == 0;}
        bool& NeedGradient() {return m_needGradient;}
        const bool& NeedGradient() const {return m_needGradient; }

        void InitRecurrentNode() 
        {
            SetLoop(0);
        }

        bool HasLoop() const { return m_hasloop ; }
        void SetLoop(const bool bl)
        {
            m_hasloop = bl; 
        }

        virtual ComputationNodePtr FindChildInASet(const std::list<ComputationNodePtr>& loop) const
        {
            for (int i = 0; i < this->m_children.size(); i++)
            {
                if (std::find(loop.begin(), loop.end(), this->m_children[i]) != loop.end())
                {
                    return this->m_children[i];
                }
            }
            return NULL;
        }

        virtual void CopyImageSizeFromInputs()
        {
            if (!IsLeaf())
                CopyImageSizeFromInput(0); //copy from child 0 by default.
        }

        bool IsChildAnImage(const size_t index) const
        {
            if (index > ChildrenSize())
                throw invalid_argument("IsChildAnImage: out of index.");

            return (Inputs(index)->m_outputWidth != 1 || Inputs(index)->m_outputChannels != 1);
        }

        const size_t ChildrenSize() const {return m_children.size();}

        inline const ComputationNodePtr Inputs(const size_t childIndex) const 
        {
#ifdef DEBUG  // profile shows this is range check very expensive in release mode, skip it  
            if (childIndex >= m_children.size())
                throw std::invalid_argument ("childIndex is out of range.");
#endif
            return m_children[childIndex];
        }

        inline ComputationNodePtr Inputs(const size_t childIndex)
        {
#ifdef DEBUG // profile shows this is range check very expensive in release mode, skip it  
            if (childIndex >= m_children.size())
                throw std::invalid_argument ("childIndex is out of range.");
#endif
            return m_children[childIndex];
        }

        void SetInput(const size_t childIndex, const ComputationNodePtr node)
        {
            //require first nodes specified before the second to avoid null nodes condition.
           if (childIndex > m_children.size())
               throw invalid_argument("SetInput: You must specify the input for children with index less than this one first.");

           // expand the inputs to exist up to the desired index
           while (childIndex >= m_children.size())
           {
               m_children.push_back(NULL);
           }

           // set the input value
            m_children[childIndex] = node;
        }

        void ComputeGradientForChildren()
        {

            /// batch is done only for feed-forward nodes
            if (HasLoop()) 
                return;

            for (size_t i=0; i<m_children.size(); i++)
            {
                ComputationNodePtr child = m_children[i];
                if (child->NeedGradient())
                {
#ifdef DISPLAY_DEBUG
                    fprintf (stderr, "    [%lu]: %s(%s)\n", i, 
                        (msra::strfun::utf8 (child->OperationName())).c_str(),
                        (msra::strfun::utf8 (child->NodeName())).c_str());
#endif              
#if DUMPOUTPUT
                    fprintf(stderr,"Backprop%d_%ws\n",i,NodeName().c_str());
#endif
                    ComputeInputPartial(i); //this computes partial wrt to the child and sums the gradient value in the child
                }
#ifdef DISPLAY_DEBUG
                else fprintf (stderr, "    [%lu]: %s(%s) (no gradient needed so don't compute for)\n", i, 
                        (msra::strfun::utf8 (child->OperationName())).c_str(),
                        (msra::strfun::utf8 (child->NodeName())).c_str());
#endif              
            }
            
        }

        void ComputeGradientForChildren(const size_t timeIdxInSeq)
        {

            for (size_t i=0; i<m_children.size(); i++)
            {
                ComputationNodePtr child = m_children[i];
                if (child->NeedGradient())
                {
#ifdef DISPLAY_DEBUG
                    fprintf (stderr, "    [%lu]: %s(%s)\n", i, 
                        (msra::strfun::utf8 (child->OperationName())).c_str(),
                        (msra::strfun::utf8 (child->NodeName())).c_str());
#endif              
                    ComputeInputPartial(i, timeIdxInSeq); //this computes partial wrt to the child and sums the gradient value in the child
                }
#ifdef DISPLAY_DEBUG
                else fprintf (stderr, "    [%lu]: %s(%s) (no gradient needed so don't compute for)\n", i, 
                        (msra::strfun::utf8 (child->OperationName())).c_str(),
                        (msra::strfun::utf8 (child->NodeName())).c_str());
#endif              
            }
        }

        static bool IsSmaller(const ComputationNodePtr lhs, const ComputationNodePtr rhs) 
        { 
            return lhs->m_visitedOrder < rhs->m_visitedOrder;
        }

        bool IsEqualTo (const ComputationNodePtr other) const //this will be used to determine whehter two nodes are the same
        {
            if (OperationName() != other->OperationName() || m_children.size() != other->m_children.size())
                return false;

            if (NodeName() == other->NodeName())  //assume names are unique in the system
                return true;

            if (IsLeaf() && other->IsLeaf())  //since names are not equal otherwise will return above
                return false;

            for (size_t i=0; i<m_children.size(); i++)
            {
                if (!(Inputs(i) == other->Inputs(i)))
                    return false;
            }

            return true;
        }
        
        std::list<ComputationNodePtr> EnumerateNodes(const bool forwardComputation, std::vector<ComputationNodePtr>& rootOfLoop)
        {
            std::list<ComputationNodePtr> result;

            if (forwardComputation)
            {
                std::unordered_set<ComputationNodePtr> visited;
                EnumerateNodesForEval(visited, result, rootOfLoop,false);
            }
            else
            {
                result = EnumerateNodesForGradient();
            }
           
            return result;          
        }

        std::list<ComputationNodePtr> ReshuffleNodes(std::map<int, std::list<ComputationNodePtr>> recurrentResult)
        {
            std::list<ComputationNodePtr> noRecurrentResult;
            std::unordered_set<ComputationNodePtr> visited;

            ReshuffleNodesForEvalWithRecurrentLoops(visited, recurrentResult, noRecurrentResult);
           
            return noRecurrentResult;          
        }



        std::list<ComputationNodePtr> EnumerateNodes(const bool forwardComputation)
        {
            std::list<ComputationNodePtr> result;

            if (forwardComputation)
            {
                std::unordered_set<ComputationNodePtr> visited;
                EnumerateNodesForEval(visited, result);
            }
            else
            {
                result = EnumerateNodesForGradient();
            }
           
            return result;          
        }

        bool IsFuncValueOlderThanInputs() const
        {
              for (size_t i=0; i<ChildrenSize(); i++)
              {
                  //the second condition is used when the time stamp change from positive to negative
                  if (Inputs(i)->m_evalTimeStamp >= m_evalTimeStamp || Inputs(i)->m_evalTimeStamp + 1e10 < m_evalTimeStamp) 
                      return true;
              }

              return false;
        }

        void ClearGradientForChildren(const int /*iActMiniBatchSize*/)
        {
            for (size_t i=0; i<m_children.size(); i++)
            {
                ComputationNodePtr child = m_children[i];
                if (child->NeedGradient())
                {
                    if(child->GradientValues().GetMatrixType() == DENSE) 
                    {
                        child->GradientValues().Resize(child->FunctionValues().GetNumRows(), child->FunctionValues().GetNumCols());
                        child->GradientValues().SetValue(0); 
                    }
                    else
                    {
                        child->GradientValues().Reset();
                    }
                }
            }
        }


        // NOTE: we should reimplement this to be thread-safe and use a larger than requested initialized memory block
        // we can then just wrap that memory block in a matrix of the correct dimensions since it will be const no one can change it
        // should only need one memory block per device
        static const Matrix<ElemType>& ConstOnes(const size_t rows, const size_t cols, const int deviceId)
        {
            if (s_constOnes.find(rows) == s_constOnes.end() ||
                s_constOnes[rows].find(cols) == s_constOnes[rows].end()) //not found
            {
                Matrix<ElemType>* matrix = new Matrix<ElemType>(rows, cols, (short)deviceId);
                matrix->SetValue(ElemType(1.000));
                s_constOnes[rows][cols] = matrix;
            }

            Matrix<ElemType>* m = s_constOnes[rows][cols];
            m->TransferFromDeviceToDevice(m->GetDeviceId(), deviceId);

            return *m;
        }

    protected:
        void CopyImageSizeFromInput(const size_t index, const bool outputSameAsInput = true)
        {
            if (index >= ChildrenSize())
                throw invalid_argument("CopyImageSizeFromInput: output index");
        
            ComputationNodePtr child = m_children[index];
            if (child != nullptr)
            {
                m_inputWidth = child->m_outputWidth;
                m_inputHeight = child->m_outputHeight;
                m_inputChannels = child->m_outputChannels;
            }

            if (outputSameAsInput)
            {
                m_outputWidth = m_inputWidth;
                m_outputHeight = m_inputHeight;
                m_outputChannels = m_inputChannels;
            }
        }

        virtual void PrintSelfBeforeValidation(bool allowNulls=false) const
        {
            fprintf(stderr, "\nValidating --> %ws = %ws", NodeName().c_str(), OperationName().c_str());           

            if (!IsLeaf())
            {
                fprintf(stderr, "(");           
                for (size_t i=0; i<ChildrenSize(); i++)
                {
                    ComputationNodePtr child = Inputs(i);
                    if (i > 0)
                        fprintf(stderr, ", ");           

                    if (child == nullptr)
                    {
                        if (allowNulls)
                        {
                            fprintf(stderr, "NULL");
                            continue;
                        }
                        throw runtime_error("One of the children is missing.");
                    }


                    if (IsChildAnImage(i))  //image
                        fprintf(stderr, "%ws[%lu {W=%lu, H=%lu, C=%lu}, %lu]", child->NodeName().c_str(), child->FunctionValues().GetNumRows(), 
                            child->m_outputWidth, child->m_outputHeight, child->m_outputChannels, child->FunctionValues().GetNumCols());
                    else
                        fprintf(stderr, "%ws[%lu, %lu]", child->NodeName().c_str(), child->FunctionValues().GetNumRows(), child->FunctionValues().GetNumCols());

                }
                fprintf(stderr, ")");           
            }
        }

        //to be called by derived classed if that class needs to print node values
        void PrintNodeValuesToFile(const bool printValues, File& fstream) const
        {
            if (printValues)
            {
                fstream << wstring(L"\n");
                const Matrix<ElemType>&  m = FunctionValues();
                for (size_t i=0; i < m.GetNumRows(); i++)
                {
                    for (size_t j=0; j < m.GetNumCols(); j++)
                    {
                        fstream << m(i,j);
                    }
                    fstream << wstring(L"\n");
                }
                fstream << wstring(L"####################################################################");
            }
       }

        std::list<ComputationNodePtr> EnumerateNodesForGradient() 
        {
            std::list<ComputationNodePtr>  nodes = this->EnumerateNodes(true);  //get forward computation order first

            nodes.sort(IsSmaller); 
            nodes.reverse();
            
            return nodes;
        }



        std::wstring CreateUniqNodeName() const
        {
#ifdef USE_GUID_AS_NAME
            UUID uuid;
            ZeroMemory(&uuid, sizeof(UUID));
            std::wstring name;

            UuidCreate(&uuid);
            WCHAR* szUuid = nullptr;
            if (UuidToStringW(&uuid, (RPC_WSTR*)&szUuid) != RPC_S_OK)
                throw std::runtime_error("Failed to craete unique node name.");
            else
            {
              name = szUuid;
              RpcStringFreeW((RPC_WSTR*)&szUuid);
            }
#else
            LONG64 id = InterlockedIncrement64(&s_timeStampCounter);
            msra::strfun::wstrprintf name(L"%s%d", L"AutoName", id);
#endif

            return name;
        }

        bool ChildrenNeedGradient()  const //this is only valid when called in the forward computation order.
        {
            for (int i=0; i<m_children.size(); i++)         
            {
                if (m_children[i] == nullptr)
                    continue;
                if (m_children[i]->m_needGradient) 
                    return true;
            }
            return false;
        }





        void EnumerateNodesForEval(std::unordered_set<ComputationNodePtr>& visited, std::list<ComputationNodePtr>& result,
            std::vector<ComputationNodePtr>& sourceRecurrentNodePtr, const bool bFromDelayNode) 
        {
            if (visited.find(this) == visited.end())  //not visited
            {   
                visited.insert(this);   // have visited tagged here to avoid infinite loop over children, children's children, etc

                for (int i=0; i<m_children.size(); i++)
                {
                    if (m_children[i] == nullptr)
                        continue;
                    m_children[i]->EnumerateNodesForEval(visited, result, sourceRecurrentNodePtr, this->OperationName() == L"Delay");
                }
                
                //children first for function evaluation
                if (!IsLeaf())
                {
                    if (ChildrenNeedGradient())  //only nodes that require gradient calculation is included in gradient calculation
                        m_needGradient = true;
                    else
                        m_needGradient = false;
                }
                
                result.push_back(ComputationNodePtr(this));  //we put this in the list even if it's leaf since we need to use it to determine learnable params 
                this->m_visitedOrder = result.size();
            }
            else
            {
                if (!IsLeaf() && bFromDelayNode)
                    sourceRecurrentNodePtr.push_back(this) ;
            }
        }

        void ReshuffleNodesForEvalWithRecurrentLoops(std::unordered_set<ComputationNodePtr>& visited, std::map<int, std::list<ComputationNodePtr>>& recurrentResult, 
            std::list<ComputationNodePtr>& noRecurrentResult) 
        {
            if (visited.find(this) == visited.end())  //not visited
            {   
                visited.insert(this);   // have visited tagged here to avoid infinite loop over children, children's children, etc

                for (int i=0; i<m_children.size(); i++)
                {
                    m_children[i]->ReshuffleNodesForEvalWithRecurrentLoops(visited, recurrentResult, noRecurrentResult);
                }
                
                //children first for function evaluation
                if (!IsLeaf())
                {
                    if (ChildrenNeedGradient())  //only nodes that require gradient calculation is included in gradient calculation
                        m_needGradient = true;
                    else
                        m_needGradient = false;
                }
                
                if (LoopId() >= 0)
                {
                    recurrentResult[LoopId()].push_back(ComputationNodePtr(this));
                }
                else
                {
                    noRecurrentResult.push_back(ComputationNodePtr(this));  //we put this in the list even if it's leaf since we need to use it to determine learnable params 
                }
            }
        }

        void EnumerateNodesForEval(std::unordered_set<ComputationNodePtr>& visited, std::list<ComputationNodePtr>& result) 
        {
            if (visited.find(this) == visited.end())  //not visited
            {   
                visited.insert(this);   // have visited tagged here to avoid infinite loop over children, children's children, etc

                for (int i=0; i<m_children.size(); i++)
                {
                    m_children[i]->EnumerateNodesForEval(visited, result);
                }
                
                //children first for function evaluation
                if (!IsLeaf())
                {
                    if (ChildrenNeedGradient())  //only nodes that require gradient calculation is included in gradient calculation
                        m_needGradient = true;
                    else
                        m_needGradient = false;
                }
                
                result.push_back(ComputationNodePtr(this));  //we put this in the list even if it's leaf since we need to use it to determine learnable params 
            }
        }


    public:
        virtual void CopyTo(const ComputationNodePtr node, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            if (OperationName() != node->OperationName())
                throw std::runtime_error("Cannot copy from one node type to another node type");
            if (flags & CopyNodeFlags::copyNodeChildren)
            {
                node->m_children = m_children;
            }
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_deviceId = m_deviceId;
                node->m_needGradient = m_needGradient;
                node->m_nodeName = newName;
                node->m_evalTimeStamp = m_evalTimeStamp;

                node->m_hasloop = m_hasloop; 

                node->m_inputWidth = m_inputWidth;
                node->m_inputHeight = m_inputHeight;
                node->m_inputChannels = m_inputChannels;

                node->m_outputWidth = m_outputWidth;
                node->m_outputHeight = m_outputHeight;
                node->m_outputChannels = m_outputChannels;

                node->m_functionValues = m_functionValues; 
                node->m_gradientValues = m_gradientValues;
            }
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const = 0;
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType /*taskType*/, size_t /*inputIndex=0*/) const
        {
            assert(false);
            NOT_IMPLEMENTED;
            //return NULL;
        }
    protected:

        short m_deviceId; //CPU=-1, >=0 GPU
        bool m_needGradient;  //only used for leaf, i.e., learnable parameters, etc.

        size_t m_inputWidth, m_inputHeight, m_inputChannels;  //how to interprerate each column in the input as an image
        size_t m_outputWidth, m_outputHeight, m_outputChannels;  //how to interprerate each column in the output as an image

        std::vector<ComputationNodePtr> m_children;

        std::wstring m_nodeName;
        Matrix<ElemType> m_functionValues, m_gradientValues;

        static LONG64 s_timeStampCounter;
        LONG64 m_evalTimeStamp; //this is used to reduce unnecessary recomputation when a different node in the model is reevaluated

        static std::map<size_t, std::map<size_t, Matrix<ElemType>*>> s_constOnes;

    private:
        /// for loop nodes
        bool m_hasloop; 
    };

#pragma endregion base computation class

#pragma region derived operations

    //used to represent weight Matrix<ElemType> and biases
    template<class ElemType>
    class LearnableParameter : public ComputationNode<ElemType>
    {
    public:
        LearnableParameter(size_t rows, size_t cols, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)
        {
            //intentionally comment out so that we may support automatic dimention inference
            //if (rows * cols == 0) 
            //    throw std::logic_error("This LearnableParameter dimension is 0.");

            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_needGradient = true;
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_functionValues.Resize(rows, cols);

            m_outputWidth = 1;
            m_outputHeight = rows;
            m_outputChannels = 1;

            InitRecurrentNode();
        }

        LearnableParameter(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual void SaveToFile(File& fstream) const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << NeedGradient();
            fstream << FunctionValues().GetNumRows() << FunctionValues().GetNumCols(); 
            fstream << FunctionValues();
        }
        
        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const short deviceId = AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            size_t rows, cols;
            fstream >> m_needGradient;
            fstream >> rows >> cols;

            //intentionally comment out to support automatic dimention inference
            //if (rows * cols == 0) 
            //    throw std::logic_error("This LearnableParameter dimension is 0.");

            m_functionValues.Resize(rows, cols);
            fstream >> m_functionValues;

            m_outputWidth = 1;
            m_outputHeight = rows;
            m_outputChannels = 1;
        }


        virtual const std::wstring OperationName() const {return TypeName();}
        virtual void ComputeInputPartial(const size_t /*inputIndex*/) {}
        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/) {}
        virtual void EvaluateThisNode()  {}
        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/) {}
        virtual void Validate() 
        {
            PrintSelfBeforeValidation();
        }

        static const std::wstring TypeName() {return L"LearnableParameter";} 

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            ComputationNode<ElemType>::DumpNodeInfo(printValues, fstream);

            WCHAR str[4096];
            wsprintf(str, L"[%lu,%lu]  ", FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            fstream << wstring(str);
            wsprintf(str, L"NeedGradient=%ws", NeedGradient()? L"true" : L"false");
            fstream << wstring(str);

            PrintNodeValuesToFile(printValues, fstream);
        }

        // copy constructor
        LearnableParameter(const LearnableParameter<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new LearnableParameter<ElemType>(this, name, flags);
            return node;
        }

        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const;
    };

    template<class ElemType>
    class SparseLearnableParameter : public LearnableParameter<ElemType>
    {
    public:
        SparseLearnableParameter (size_t rows, size_t cols, const size_t size, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") 
            : LearnableParameter<ElemType>(rows, cols, deviceId, name)
        {
            m_gradientValues.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol);
            m_gradientValues.Resize(rows, cols, size);
        }

        SparseLearnableParameter (File& fstream, const size_t modelVersion, const short deviceId = AUTOPLACEMATRIX, const std::wstring name = L"") 
            : LearnableParameter<ElemType>(fstream, modelVersion, deviceId, name)
        {
            m_gradientValues.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol);
            m_gradientValues.Resize(FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const short deviceId = AUTOPLACEMATRIX)
        {
            LearnableParameter<ElemType>::LoadFromFile(fstream,   modelVersion, deviceId);
            m_gradientValues.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol);
            m_gradientValues.Resize(FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"SparseLearnableParameter";} 

        // copy constructor
        SparseLearnableParameter (const SparseLearnableParameter <ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) 
            : LearnableParameter( node, newName, flags)
        {
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new SparseLearnableParameter<ElemType>(this, name, flags);
            return node;
        }
    };

    template class SparseLearnableParameter<float>; 
    template class SparseLearnableParameter<double>;

    template<class ElemType>
    class InputValue : public ComputationNode<ElemType>
    {
    public:
        InputValue(size_t rows, size_t cols, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId) 
        {
            if (rows * cols == 0) 
                throw std::logic_error("This InputValue dimension is 0.");

            m_outputWidth = 1;
            m_outputHeight = rows;
            m_outputChannels = 1;

            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_functionValues.Resize(rows, cols);
            m_needGradient = false;
            InitRecurrentNode();
        }
        
        InputValue(size_t imageWidth, size_t imageHeight, size_t imageChannels, size_t numImages, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId) 
        {
            size_t rows = imageWidth * imageHeight * imageChannels;
            size_t cols = numImages;

            if (rows * cols == 0) 
                throw std::logic_error("This InputValue dimension is 0.");

            m_outputWidth = imageWidth;
            m_outputHeight = imageHeight;
            m_outputChannels = imageChannels;

            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_functionValues.Resize(rows, cols);
            m_needGradient = false;
            InitRecurrentNode();
        }        

        InputValue(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual void SaveToFile(File& fstream) const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << FunctionValues().GetNumRows() << FunctionValues().GetNumCols(); 
            fstream << m_outputWidth << m_outputHeight << m_outputChannels; 
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const short deviceId = AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            size_t rows, cols;
            fstream >> rows >> cols;
            if (rows * cols == 0) 
                throw std::logic_error("This InputValue dimension is 0.");

            fstream >> m_outputWidth >> m_outputHeight >> m_outputChannels; 

            m_functionValues.Resize(rows, cols);
            m_needGradient = false;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"InputValue";} 

        virtual void EvaluateThisNode()  {} 
        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/) {}
        
        virtual void ComputeInputPartial(const size_t /*inputIndex*/) {}
        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/) {}

        virtual void Validate() 
        {
            PrintSelfBeforeValidation();
            //CopyImageSizeFromInputs(); //not necessary since InputValue are leafs. put it here for consistent
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            ComputationNode<ElemType>::DumpNodeInfo(printValues, fstream);

            WCHAR str[4096];
            wsprintf(str, L"[%lu,%lu]", FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            fstream << wstring(str);        
        }

        // copy constructor
        InputValue(const InputValue<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new InputValue<ElemType>(this, name, flags);
            return node;
        }

        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType /*taskType*/, size_t inputIndex=0) const
        {
            inputIndex;
            return nullptr;
        }
    };

    template class InputValue<float>; 
    template class InputValue<double>;

    template<class ElemType>
    class SparseInputValue : public ComputationNode<ElemType>
    {
    public:
        SparseInputValue (size_t rows, size_t cols, size_t size, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId) 
        {
            if (rows * cols == 0) 
                throw std::logic_error("This InputValue dimension is 0.");

            m_outputWidth = 1;
            m_outputHeight = rows;
            m_outputChannels = 1;

            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_functionValues.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSC);
            m_functionValues.Resize(rows, cols, size);
            m_needGradient = false;
            InitRecurrentNode();
        }
        
        SparseInputValue (size_t imageWidth, size_t imageHeight, size_t imageChannels, size_t numImages, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") 
            : ComputationNode<ElemType>(deviceId) 
        {
            size_t rows = imageWidth * imageHeight * imageChannels;
            size_t cols = numImages;

            if (rows * cols == 0) 
                throw std::logic_error("This InputValue dimension is 0.");

            m_outputWidth = imageWidth;
            m_outputHeight = imageHeight;
            m_outputChannels = imageChannels;

            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_functionValues.SwitchToMatrixType(MatrixType::SPARSE);
            m_functionValues.Resize(rows, cols);
            m_needGradient = false;
            InitRecurrentNode();
        }        

        SparseInputValue (File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual void SaveToFile(File& fstream) const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << FunctionValues().GetNumRows() << FunctionValues().GetNumCols(); 
            fstream << FunctionValues().GetAllocatedSize();
            fstream << m_outputWidth << m_outputHeight << m_outputChannels; 
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const short deviceId = AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            size_t rows, cols;
            fstream >> rows >> cols;
            if (rows * cols == 0) 
                throw std::logic_error("This InputValue dimension is 0.");

            size_t size; //sparse matrix size
            fstream >> size;

            fstream >> m_outputWidth >> m_outputHeight >> m_outputChannels; 
                        
            m_functionValues.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSC);
            m_functionValues.Resize(rows, cols, size);
            m_needGradient = false;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"SparseInputValue";} 

        virtual void EvaluateThisNode()  {} 
        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/) {}
        
        virtual void ComputeInputPartial(const size_t /*inputIndex*/) {}
        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/) {}

        virtual void Validate() 
        {
            PrintSelfBeforeValidation();
            //CopyImageSizeFromInputs(); //not necessary since InputValue are leafs. put it here for consistent
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            ComputationNode<ElemType>::DumpNodeInfo(printValues, fstream);

            WCHAR str[4096];
            wsprintf(str, L"[%lu,%lu]", FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            fstream << wstring(str);        
        }

        // copy constructor
        SparseInputValue (const SparseInputValue <ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new SparseInputValue<ElemType>(this, name, flags);
            return node;
        }

        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType /*taskType*/, size_t inputIndex=0) const
        {
            inputIndex;
            return nullptr;
        }
    };


    template class SparseInputValue<float>; 
    template class SparseInputValue<double>;

    template<class ElemType>
    class NegateNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


    public:
        NegateNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        NegateNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        NegateNode(const NegateNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new NegateNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Negate";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Negate operation only has one input.");
            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Negate operation only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(sliceInputGrad, sliceOutputGrad);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& childGradientValues, const Matrix<ElemType>& gradientValues)
        {
            childGradientValues -= gradientValues;
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // forward - EvalutateThisNode() if set otherwise ComputeInputPartial()
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            if (taskType == taskEvaluate)
            {
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
            }
            else if (taskType == taskComputeInputPartial)
            {
                descriptor->GradientParam(0, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->SetFunction((FARPROC)ComputeInputPartialS);
            }
            else
            {
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input)  
        {
            functionValues.AssignDifferenceOf(0, input);
#if NANCHECK
            functionValues.HasNan("Negate");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("Negate operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Negate operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            
            CopyImageSizeFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }
    };

    template class NegateNode<float>; 
    template class NegateNode<double>;

    template<class ElemType>
    class RectifiedLinearNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


    public:
        RectifiedLinearNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode(deviceId), m_gradientOfRectifiedLinear(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        RectifiedLinearNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_gradientOfRectifiedLinear(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("RectifiedLinear only has one input.");
            ComputeInputPartialS(m_gradientOfRectifiedLinear, Inputs(0)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("RectifiedLinear only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradientOfRectifiedLinear, sliceInputValue, sliceInputGrad, sliceOutputGrad);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientOfRectifiedLinear, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
        {
            gradientOfRectifiedLinear.AssignLinearRectifierDerivativeOf(inputFunctionValues);
#if DUMPOUTPUT
            inputGradientValues.Print("RecitifiedLinearNode-Partial-in");
#endif
            inputGradientValues.AddElementProductOf(gradientValues, gradientOfRectifiedLinear); 
#if DUMPOUTPUT
            inputGradientValues.Print("RecitifiedLinearNode-Partial-out");
#endif
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - type of task we are making a descriptor for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            case taskComputeInputPartial:
                descriptor->MatrixParam(m_gradientOfRectifiedLinear, "GradientOfRectifiedLinear", paramOptionsInput | paramOptionsTemporary);
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->GradientParam(0, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->SetFunction((FARPROC)ComputeInputPartialS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.AssignTruncateBottomOf(inputFunctionValues, 0);
#if NANCHECK
            functionValues.HasNan("RectifiedLinear");
#endif
#if DUMPOUTPUT
            functionValues.Print("RectifiedLinearNode");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("RectifiedLinear operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("RectifiedLinear operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_gradientOfRectifiedLinear.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_gradientOfRectifiedLinear.GetDeviceId() != deviceId)
                    m_gradientOfRectifiedLinear.TransferFromDeviceToDevice(m_gradientOfRectifiedLinear.GetDeviceId(), deviceId);
            }
        }

        static const std::wstring TypeName() {return L"RectifiedLinear";} 

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            RectifiedLinearNode<ElemType>* node = (RectifiedLinearNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_gradientOfRectifiedLinear = m_gradientOfRectifiedLinear;
            }
        }

        // copy constructor
        RectifiedLinearNode(const RectifiedLinearNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), m_gradientOfRectifiedLinear(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new RectifiedLinearNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_gradientOfRectifiedLinear;
    };

    template class RectifiedLinearNode<float>; 
    template class RectifiedLinearNode<double>;

    template<class ElemType>
    class SigmoidNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 

    public:
        SigmoidNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode(deviceId), m_gradientOfSigmoid(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        SigmoidNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_gradientOfSigmoid(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Sigmoid";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Sigmoid only has one input.");
            ComputeInputPartialS(m_gradientOfSigmoid, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());  
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Sigmoid only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradientOfSigmoid, sliceInputGrad, sliceOutputGrad, sliceOutputValue);  
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientOfSigmoid, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)  
        {
            gradientOfSigmoid.AssignSigmoidDerivativeOf(functionValues);

            inputGradientValues.AddElementProductOf(gradientValues, gradientOfSigmoid);
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            //case taskComputeInputPartialRecurrent:
            //    descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
            //    descriptor->MatrixParam(m_gradientOfSigmoid, "GradientOfSigmoid", paramOptionsInput | paramOptionsTemporary);
            //    descriptor->GradientParam(0, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
            //    descriptor->GradientParam();
            //    descriptor->FunctionParam(-1, paramOptionsInput);
            //    descriptor->Param(paramTypeLongLong, "nbrSlicesInEachIter");
            //    descriptor->SetFunction((FARPROC)ComputeInputPartialSR);
                //break;
            case taskComputeInputPartial:
                descriptor->MatrixParam(m_gradientOfSigmoid, "GradientOfSigmoid", paramOptionsInput | paramOptionsTemporary);
                descriptor->GradientParam(0, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->FunctionParam(-1, paramOptionsInput);
                descriptor->SetFunction((FARPROC)ComputeInputPartialS);
                break;
            //case taskEvaluateRecurrent:
            //    descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
            //    descriptor->FunctionParam();
            //    descriptor->FunctionParam(0, paramOptionsInput);
            //    descriptor->Param(paramTypeLongLong, "nbrSlicesInEachIter");
            //    descriptor->SetFunction((FARPROC)EvaluateThisNodeSR);
                //break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.AssignSigmoidOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Sigmoid");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("Sigmoid operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Sigmoid operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_gradientOfSigmoid.Resize(FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_gradientOfSigmoid.GetDeviceId() != deviceId)
                    m_gradientOfSigmoid.TransferFromDeviceToDevice(m_gradientOfSigmoid.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            SigmoidNode<ElemType>* node = (SigmoidNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_gradientOfSigmoid = m_gradientOfSigmoid;
            }
        }

        // copy constructor
        SigmoidNode(const SigmoidNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), m_gradientOfSigmoid(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new SigmoidNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_gradientOfSigmoid;
    };

    template class SigmoidNode<float>; 
    template class SigmoidNode<double>;

    template<class ElemType>
    class TanhNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 

    public:
        TanhNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode(deviceId), m_gradientOfTanh(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        TanhNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_gradientOfTanh(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Tanh";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Tanh only has one input.");
            ComputeInputPartialS(m_gradientOfTanh, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Tanh only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradientOfTanh, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientOfTanh, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)  
        {
            gradientOfTanh.AssignElementProductOf(functionValues, functionValues); // v .* v
            gradientOfTanh.AssignDifferenceOf(1, gradientOfTanh); // 1-v^2

            inputGradientValues.AddElementProductOf(gradientValues, gradientOfTanh); // += d .* ((1-v) .* v))
        }


        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            //case taskComputeInputPartialRecurrent:
            //    recurrant = true;
            //    descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
            //    descriptor->MatrixParam(m_gradientOfTanh, "GradientOfTanh", paramOptionsInput | paramOptionsTemporary);
            //    descriptor->GradientParam(0, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
            //    descriptor->GradientParam();
            //    descriptor->FunctionParam(-1, paramOptionsInput);
            //    descriptor->Param(paramTypeLongLong, "nbrSlicesInEachIter");
            //    descriptor->SetFunction((FARPROC)ComputeInputPartialSR);
                //break;
            case taskComputeInputPartial:
                descriptor->MatrixParam(m_gradientOfTanh, "GradientOfTanh", paramOptionsInput | paramOptionsTemporary);
                descriptor->GradientParam(0, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->FunctionParam(-1, paramOptionsInput);
                descriptor->SetFunction((FARPROC)ComputeInputPartialS);
                break;
            //case taskEvaluateRecurrent:
            //    descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
            //    descriptor->FunctionParam();
            //    descriptor->FunctionParam(0, paramOptionsInput);
            //    descriptor->Param(paramTypeLongLong, "nbrSlicesInEachIter");
            //    descriptor->SetFunction((FARPROC)EvaluateThisNodeSR);
            //    break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.AssignTanhOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Tanh");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("Tanh operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Tanh operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_gradientOfTanh.Resize(FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_gradientOfTanh.GetDeviceId() != deviceId)
                    m_gradientOfTanh.TransferFromDeviceToDevice(m_gradientOfTanh.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            TanhNode<ElemType>* node = (TanhNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_gradientOfTanh = m_gradientOfTanh;
            }
        }

        // copy constructor
        TanhNode(const TanhNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), m_gradientOfTanh(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new TanhNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_gradientOfTanh;
    };

    template class TanhNode<float>; 
    template class TanhNode<double>;

	template<class ElemType>
	class LogNode : public ComputationNode<ElemType>
	{
		typedef ComputationNode<ElemType>* ComputationNodePtr;


	public:
		LogNode(const short deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
			: ComputationNode(deviceId), m_gradientOfLog(deviceId)
		{
			m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
			m_deviceId = deviceId;
			MoveMatricesToDevice(deviceId);
			InitRecurrentNode();
		}

		LogNode(File& fstream, const size_t modelVersion, const short deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
			: ComputationNode(deviceId), m_gradientOfLog(deviceId)
		{
			m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
			LoadFromFile(fstream, modelVersion, deviceId);
		}

		virtual const std::wstring OperationName() const { return TypeName(); }
		static const std::wstring TypeName() { return L"Log"; }


		virtual void ComputeInputPartial(const size_t inputIndex)
		{
			if (inputIndex != 0)
				throw std::invalid_argument("Log only has one input.");
			ComputeInputPartialS(m_gradientOfLog, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), GradientValues());
		}

		virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
		{
			if (inputIndex != 0)
				throw std::invalid_argument("Log only has one input.");

			Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
			Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

			Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

			ComputeInputPartialS(m_gradientOfLog, sliceInputGrad, sliceInputValue, sliceOutputGrad);
		}

		static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientOfLog, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& gradientValues)
		{
			gradientOfLog.AssignElementInverseOf(inputFunctionValues); // 1/x (x is input to log(x))

			inputGradientValues.AddElementProductOf(gradientValues, gradientOfLog);
		}

		// GetTaskDescriptor - Get a task descriptor for this node
		// taskType - task type we are generating a task for
		virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex = 0) const
		{
			TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
			switch (taskType)
			{
			case taskComputeInputPartial:
				descriptor->MatrixParam(m_gradientOfLog, "GradientOfLog", paramOptionsInput | paramOptionsTemporary);
				descriptor->GradientParam(0, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
				descriptor->FunctionParam(0, paramOptionsInput);
				descriptor->GradientParam();
				descriptor->SetFunction((FARPROC)ComputeInputPartialS);
				break;
			case taskEvaluate:
				descriptor->FunctionParam();
				descriptor->FunctionParam(0, paramOptionsInput);
				descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
				break;
			default:
				assert(false);
				throw std::logic_error("Unsupported task requested");
			}
			return descriptor;
		}

		virtual void EvaluateThisNode()
		{
			EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues());
		}

		virtual void EvaluateThisNode(const size_t timeIdxInSeq)
		{
			Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
			Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

			EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
		}

		static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)
		{
			functionValues.AssignLogOf(inputFunctionValues);
#if NANCHECK
			functionValues.HasNan("Log");
#endif
		}

		virtual void Validate()
		{
			PrintSelfBeforeValidation();

			if (m_children.size() != 1)
				throw std::logic_error("Log operation should have one input.");

			if (Inputs(0)->FunctionValues().GetNumElements() == 0)
				throw std::logic_error("Log operation: the input node has 0 element.");

			FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
			m_gradientOfLog.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
			CopyImageSizeFromInputs();
		}

		virtual void AttachInputs(const ComputationNodePtr singleInput)
		{
			m_children.resize(1);
			m_children[0] = singleInput;
		}

		virtual void MoveMatricesToDevice(const short deviceId)
		{
			ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

			if (deviceId != AUTOPLACEMATRIX)
			{
				if (m_gradientOfLog.GetDeviceId() != deviceId)
					m_gradientOfLog.TransferFromDeviceToDevice(m_gradientOfLog.GetDeviceId(), deviceId);
			}
		}

		virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
		{
			ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
			LogNode<ElemType>* node = (LogNode<ElemType>*) nodeP;

			if (flags & CopyNodeFlags::copyNodeValue)
			{
				node->m_gradientOfLog = m_gradientOfLog;
			}
		}

		// copy constructor
		LogNode(const LogNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
			: ComputationNode(node->m_deviceId), m_gradientOfLog(node->m_deviceId)
		{
			node->CopyTo(this, newName, flags);
		}

		virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
		{
			const std::wstring& name = (newName == L"") ? NodeName() : newName;

			ComputationNodePtr node = new LogNode<ElemType>(this, name, flags);
			return node;
		}

	private:
		Matrix<ElemType> m_gradientOfLog;
	};

	template class LogNode<float>;
	template class LogNode<double>;


	template<class ElemType>
	class ExpNode : public ComputationNode<ElemType>
	{
		typedef ComputationNode<ElemType>* ComputationNodePtr;


	public:
		ExpNode(const short deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
			: ComputationNode(deviceId), m_gradientOfExp(deviceId)
		{
			m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
			m_deviceId = deviceId;
			MoveMatricesToDevice(deviceId);
			InitRecurrentNode();
		}

		ExpNode(File& fstream, const size_t modelVersion, const short deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
			: ComputationNode(deviceId), m_gradientOfExp(deviceId)
		{
			m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
			LoadFromFile(fstream, modelVersion, deviceId);
		}

		virtual const std::wstring OperationName() const { return TypeName(); }
		static const std::wstring TypeName() { return L"Exp"; }


		virtual void ComputeInputPartial(const size_t inputIndex)
		{
			if (inputIndex != 0)
				throw std::invalid_argument("Exp only has one input.");
			ComputeInputPartialS(m_gradientOfExp, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), GradientValues());
		}

		virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
		{
			if (inputIndex != 0)
				throw std::invalid_argument("Exp only has one input.");

			Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
			Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

			Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

			ComputeInputPartialS(m_gradientOfExp, sliceInputGrad, sliceInputValue, sliceOutputGrad);
		}

		static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientOfExp, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& gradientValues)
		{
			gradientOfExp.AssignExpOf(inputFunctionValues); // Exp(x) is its own partial

			inputGradientValues.AddElementProductOf(gradientValues, gradientOfExp);
		}

		// GetTaskDescriptor - Get a task descriptor for this node
		// taskType - task type we are generating a task for
		virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex = 0) const
		{
			TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
			switch (taskType)
			{
			case taskComputeInputPartial:
				descriptor->MatrixParam(m_gradientOfExp, "GradientOfExp", paramOptionsInput | paramOptionsTemporary);
				descriptor->GradientParam(0, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
				descriptor->FunctionParam(0, paramOptionsInput);
				descriptor->GradientParam();
				descriptor->SetFunction((FARPROC)ComputeInputPartialS);
				break;
			case taskEvaluate:
				descriptor->FunctionParam();
				descriptor->FunctionParam(0, paramOptionsInput);
				descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
				break;
			default:
				assert(false);
				throw std::logic_error("Unsupported task requested");
			}
			return descriptor;
		}

		virtual void EvaluateThisNode()
		{
			EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues());
		}

		virtual void EvaluateThisNode(const size_t timeIdxInSeq)
		{
			Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
			Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

			EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
		}

		static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)
		{
			functionValues.AssignExpOf(inputFunctionValues);
#if NANCHECK
			functionValues.HasNan("Exp");
#endif
		}

		virtual void Validate()
		{
			PrintSelfBeforeValidation();

			if (m_children.size() != 1)
				throw std::logic_error("Exp operation should have one input.");

			if (Inputs(0)->FunctionValues().GetNumElements() == 0)
				throw std::logic_error("Exp operation: the input node has 0 element.");

			FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
			m_gradientOfExp.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
			CopyImageSizeFromInputs();
		}

		virtual void AttachInputs(const ComputationNodePtr singleInput)
		{
			m_children.resize(1);
			m_children[0] = singleInput;
		}

		virtual void MoveMatricesToDevice(const short deviceId)
		{
			ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

			if (deviceId != AUTOPLACEMATRIX)
			{
				if (m_gradientOfExp.GetDeviceId() != deviceId)
					m_gradientOfExp.TransferFromDeviceToDevice(m_gradientOfExp.GetDeviceId(), deviceId);
			}
		}

		virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
		{
			ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
			ExpNode<ElemType>* node = (ExpNode<ElemType>*) nodeP;

			if (flags & CopyNodeFlags::copyNodeValue)
			{
				node->m_gradientOfExp = m_gradientOfExp;
			}
		}

		// copy constructor
		ExpNode(const ExpNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
			: ComputationNode(node->m_deviceId), m_gradientOfExp(node->m_deviceId)
		{
			node->CopyTo(this, newName, flags);
		}

		virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
		{
			const std::wstring& name = (newName == L"") ? NodeName() : newName;

			ComputationNodePtr node = new ExpNode<ElemType>(this, name, flags);
			return node;
		}

	private:
		Matrix<ElemType> m_gradientOfExp;
	};

	template class ExpNode<float>;
	template class ExpNode<double>;


	template<class ElemType>
    class CosineNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


    public:
        CosineNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode(deviceId), m_gradientOfCosine(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        CosineNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_gradientOfCosine(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Cosine";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Cosine only has one input.");
            ComputeInputPartialS(m_gradientOfCosine, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Cosine only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradientOfCosine, sliceInputGrad, sliceInputValue, sliceOutputGrad);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientOfCosine, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& gradientValues)  
        {
            gradientOfCosine.AssignNegativeSineOf(inputFunctionValues); // -sin(x) (x is input to Cosine(x))
            inputGradientValues.AddElementProductOf(gradientValues, gradientOfCosine);
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                descriptor->MatrixParam(m_gradientOfCosine, "GradientOfCosine", paramOptionsInput | paramOptionsTemporary);
                descriptor->GradientParam(0, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->GradientParam();
                descriptor->SetFunction((FARPROC)ComputeInputPartialS);
                break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.AssignCosineOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Cosine");
#endif
        }


        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("Cosine operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Cosine operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_gradientOfCosine.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_gradientOfCosine.GetDeviceId() != deviceId)
                    m_gradientOfCosine.TransferFromDeviceToDevice(m_gradientOfCosine.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            CosineNode<ElemType>* node = (CosineNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_gradientOfCosine = m_gradientOfCosine;
            }
        }

        // copy constructor
        CosineNode(const CosineNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), m_gradientOfCosine(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new CosineNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_gradientOfCosine;
    };

    template class CosineNode<float>; 
    template class CosineNode<double>;


    //we assume it's  column-wise by default
    //the derivative will increase the Matrix<ElemType> size to the power of column size and should not be used.
    template<class ElemType>
    class SoftmaxNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


    public:
        SoftmaxNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"", const bool isColWise = true)
            : ComputationNode(deviceId), m_gradientDotValue(deviceId), m_diff(deviceId), m_isColWise(isColWise)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        SoftmaxNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_gradientDotValue(deviceId), m_diff(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Softmax";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Softmax only has one input.");
            ComputeInputPartialS(m_gradientDotValue, m_diff, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());
        }

		virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
		{
			if (inputIndex != 0)
				throw std::invalid_argument("Softmax only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradientDotValue, m_diff, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
		}

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientDotValue, Matrix<ElemType>& diff, Matrix<ElemType>& inputGradientValues, 
            const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)  
		{
			gradientDotValue.AssignInnerProductOf(gradientValues, functionValues, true);
			diff.AssignDifferenceOf(gradientValues, gradientDotValue);

            inputGradientValues.AddElementProductOf(diff, functionValues);
		}

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                descriptor->MatrixParam(m_gradientDotValue, "GradientDotValue", paramOptionsInput | paramOptionsTemporary);
                descriptor->MatrixParam(m_diff, "Diff", paramOptionsInput | paramOptionsTemporary);
                descriptor->GradientParam(0, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->FunctionParam(-1, paramOptionsInput);
                descriptor->SetFunction((FARPROC)ComputeInputPartialS);
                break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues());
        }

		virtual void EvaluateThisNode(const size_t timeIdxInSeq)
		{
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
		}

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.AssignLogSoftmaxOf(inputFunctionValues, true);
			functionValues.InplaceExp();
#if NANCHECK
            functionValues.HasNan("SoftMax");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("SoftmaxNode operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("SoftmaxNode operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
            m_gradientDotValue.Resize(GradientValues().GetNumRows(), GradientValues().GetNumCols());
            m_diff.Resize(GradientValues().GetNumRows(), GradientValues().GetNumCols());
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_gradientDotValue.GetDeviceId() != deviceId)
                    m_gradientDotValue.TransferFromDeviceToDevice(m_gradientDotValue.GetDeviceId(), deviceId);
                if (m_diff.GetDeviceId() != deviceId)
                    m_diff.TransferFromDeviceToDevice(m_diff.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            SoftmaxNode<ElemType>* node = (SoftmaxNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_gradientDotValue = m_gradientDotValue;
                node->m_diff = m_diff;
            }
        }

        // copy constructor
        SoftmaxNode(const SoftmaxNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), m_gradientDotValue(node->m_deviceId), m_diff(node->m_deviceId) 
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new SoftmaxNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        bool m_isColWise;
        Matrix<ElemType> m_gradientDotValue;
        Matrix<ElemType> m_diff;
    };

    template class SoftmaxNode<float>; 
    template class SoftmaxNode<double>;

    template<class ElemType>
    class SumElementsNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


    public:
        SumElementsNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        SumElementsNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        SumElementsNode(const SumElementsNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new SumElementsNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"SumElements";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("SumElements only has one input.");
            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("SumElements only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(sliceInputGrad, sliceOutputGrad);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            inputGradientValues += gradientValues; //here the assumption is that gradientValues are 1x1 matrix
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                descriptor->GradientParam(0, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->SetFunction((FARPROC)ComputeInputPartialS);
                break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.AssignSumOfElements(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("SumElements");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("SumElements operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("SumElements operation: the input node has 0 element.");

            FunctionValues().Resize(1, 1);
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, false);

            m_outputWidth = 1;
            m_outputHeight = 1;        
            m_outputChannels = 1;
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }
    };

    template class SumElementsNode<float>; 
    template class SumElementsNode<double>;

    //this node is used to extract part of the input by rows as the output
    //it has to be continuous segments of rows since each column is treated as one sample
    template<class ElemType>
    class RowSliceNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 

    public:
        RowSliceNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId), m_startIndex(0), m_numRows (0) 
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        RowSliceNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        RowSliceNode(const RowSliceNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }
        
		RowSliceNode(const short deviceId, size_t start_index, size_t num_rows, const std::wstring name = L"") : ComputationNode(deviceId)  
        {
			m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
			m_startIndex = start_index;
			m_numRows = num_rows;
			

            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
		}

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new RowSliceNode<ElemType>(this, name, flags);
            return node;
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            RowSliceNode<ElemType>* node = (RowSliceNode<ElemType>*) nodeP;

            node->m_startIndex = m_startIndex;
            node->m_numRows = m_numRows;
        }

        virtual void SaveToFile(File& fstream) const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << m_startIndex << m_numRows;
        }
        
        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const short deviceId = AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            fstream >> m_startIndex >> m_numRows;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"RowSlice";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("RowSlice only has one input.");

            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues(), m_startIndex, m_numRows);
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("RowSlice only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(sliceInputGrad, sliceOutputGrad, m_startIndex, m_numRows);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const size_t startIndex, const size_t numRows)  
        {
            inputGradientValues.AddToRowSliceValuesOf(gradientValues, startIndex, numRows); 
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                descriptor->GradientParam(0, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->SetFunction((FARPROC)ComputeInputPartialS);
                break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues(), m_startIndex, m_numRows);
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue, m_startIndex, m_numRows);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues, const size_t startIndex, const size_t numRows)  
        {
            functionValues.AssignRowSliceValuesOf(inputFunctionValues, startIndex, numRows);
#if NANCHECK
            functionValues.HasNan("RowSlice");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("RowSlice operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("RowSlice operation: the input node has 0 element.");

            if (Inputs(0)->FunctionValues().GetNumRows() < m_startIndex + m_numRows)
                throw std::logic_error("RowSlice operation: m_startIndex + m_numRows exceeds number of rows in the input.");

            FunctionValues().Resize(m_numRows, Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, true);
            m_outputHeight = m_numRows;        

            //WARNING: this node will destroy the image size information from the child
            if (m_inputWidth * m_inputChannels != 1)
                fprintf(stderr, "WARNING: RowSlice operation cannot inherit image size information from its child. Image size info is lost.\n");
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

    private:
        size_t m_startIndex, m_numRows;
    };

    template class RowSliceNode<float>; 
    template class RowSliceNode<double>;

    template<class ElemType>
    class ScaleNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 

    public:
        ScaleNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        ScaleNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        ScaleNode(const ScaleNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new ScaleNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Scale";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("ScaleNode operation only takes two inputs.");

            //left Node must be a scalar
            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
            }
            else
            {
                ComputeInputPartialRight(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues());
            }
        }

		virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("ScaleNode operation only takes two inputs.");

            //left Node must be a scalar
            if (inputIndex == 0)  //left derivative
            {
                Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialLeft(sliceInput1Value, Inputs(0)->GradientValues(), sliceOutputGrad);
            }
            else
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialRight(Inputs(0)->FunctionValues(), sliceInput1Grad, sliceOutputGrad);
            }
        }

        static void WINAPI ComputeInputPartialLeft(const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            inputGradientValues += Matrix<ElemType>::InnerProductOfMatrices(gradientValues, inputFunctionValues);
        }

        static void WINAPI ComputeInputPartialRight(const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            Matrix<ElemType>::ScaleAndAdd(inputFunctionValues.Get00Element(), gradientValues, inputGradientValues);
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                descriptor->FunctionParam(1-inputIndex, paramOptionsInput);
                descriptor->GradientParam(inputIndex, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->SetFunction(inputIndex?(FARPROC)ComputeInputPartialRight:(FARPROC)ComputeInputPartialLeft);
                break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, const Matrix<ElemType>& input1)  
        {
            functionValues.AssignProductOf(input0.Get00Element(), input1);
#if NANCHECK
            functionValues.HasNan("Scale");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("Scale operation requires two inputs.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Scale operation: one of the operands has 0 element.");

            if (Inputs(0)->FunctionValues().GetNumRows() != 1 || Inputs(0)->FunctionValues().GetNumCols() != 1)
                throw std::logic_error("The left value of ScaleNode must be a scalar value.");

            FunctionValues().Resize(Inputs(1)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());
            //left Node must be a scalar
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(1); 
        }

        virtual void AttachInputs(const ComputationNodePtr scalarValue, const ComputationNodePtr Value) 
        {
            m_children.resize(2);
            m_children[0] = scalarValue;
            m_children[1] = Value;
        }
    };

    template class ScaleNode<float>; 
    template class ScaleNode<double>;

    template<class ElemType>
    class TimesNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 

    public:
        TimesNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        TimesNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        TimesNode(const TimesNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new TimesNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Times";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("Times operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
            }
            else  //right derivative
            {
                ComputeInputPartialRight(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues());
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("Times operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialLeft(sliceInput1Value, Inputs(0)->GradientValues(), sliceOutputGrad);
            }
            else  //right derivative
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialRight(Inputs(0)->FunctionValues(), sliceInput1Grad, sliceOutputGrad);
            }
        }

        static void WINAPI ComputeInputPartialLeft(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
            {
#if DUMPOUTPUT
            gradientValues.Print("Gradient-in");
            inputGradientValues.Print("child Gradient-in/out");
            inputFunctionValues.Print("child Function values");
#endif

                Matrix<ElemType>::MultiplyAndAdd(gradientValues, false, inputFunctionValues, true, inputGradientValues);
#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
        }

        static void WINAPI ComputeInputPartialRight(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
            {
#if DUMPOUTPUT
            gradientValues.Print("Gradient-in");
            inputGradientValues.Print("child Gradient-in/out");
            inputFunctionValues.Print("child Function values");
#endif
                Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, true, gradientValues, false, inputGradientValues);
#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            //case taskComputeInputPartialRecurrent:
            //    descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
            //    descriptor->FunctionParam(1-inputIndex, paramOptionsInput);
            //    descriptor->GradientParam(inputIndex, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
            //    descriptor->GradientParam();
            //    descriptor->SetFunction((inputIndex?(FARPROC)ComputeInputPartialRightR:(FARPROC)ComputeInputPartialLeftR));
            //    break;
            case taskComputeInputPartial:
                descriptor->FunctionParam(1-inputIndex, paramOptionsInput);
                descriptor->GradientParam(inputIndex, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->SetFunction((inputIndex?(FARPROC)ComputeInputPartialRight:(FARPROC)ComputeInputPartialLeft));
                break;
            //case taskEvaluateRecurrent:
            //    descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
            //    descriptor->FunctionParam();
            //    descriptor->FunctionParam(0, paramOptionsInput);
            //    descriptor->FunctionParam(1, paramOptionsInput);
            //    descriptor->Param(paramTypeSizet, "mNbrSlicesInEachRecurrentIter", paramOptionsInput);
            //    break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, const Matrix<ElemType>& input1)  
        {
#if DUMPOUTPUT
            input0.Print("TimesNode - Input0");
#endif
            functionValues.AssignProductOf(input0, false, input1, false);
#if NANCHECK
            functionValues.HasNan("Times");
#endif
#if DUMPOUTPUT
            functionValues.Print("TimesNode");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("Times operation requires two inputs.");

            //support automatic dimention inference for learnable parameters
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if ((rows0 == 0 || cols1 == 0 ) && this->LoopId() < 0)
                throw logic_error("Times operation: Inputs(0)->FunctionValues().GetNumRows() and Inputs(1)->FunctionValues().GetNumCols() should not be 0 since it cannot be automatically inferred");

            if ((Inputs(0)->OperationName() == LearnableParameter<ElemType>::TypeName() && cols0 == 0 && rows1 != 0) && this->LoopId() < 0)
                Inputs(0)->FunctionValues().Resize(rows0, rows1);

            if (Inputs(1)->OperationName() == LearnableParameter<ElemType>::TypeName() && cols0 != 0 && rows1 == 0)
                Inputs(1)->FunctionValues().Resize(cols0, cols1);

            if ((Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)&& this->LoopId() < 0)
                throw std::logic_error("Times operation: One of the operants has 0 elements.");

            //cols0 and rows1 may have been changed so don't use them in the following check
            if ((Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumCols()) && this->LoopId() < 0)
            {
                throw std::logic_error("The Matrix dimension in the Times operation does not match.");
            }
            FunctionValues().Resize(rows0, cols1);
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()  
        {
            CopyImageSizeFromInput(1, false); //the second one is the input since it's column wize

            //after multiplication the structure is lost
            m_outputWidth = 1;
            m_outputHeight = Inputs(0)->FunctionValues().GetNumRows();
            m_outputChannels =  1;
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }
    };

    template class TimesNode<float>; 
    template class TimesNode<double>;

    template<class ElemType>
    class ElementTimesNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 

    public:
        ElementTimesNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        ElementTimesNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        ElementTimesNode(const ElementTimesNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new ElementTimesNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"ElementTimes";} 

        virtual void ComputeInputPartial(const size_t inputIndex)  
        {
            if (inputIndex > 1)
                throw std::invalid_argument("ElementTimes operation only takes two inputs.");

            ComputeInputPartialS(Inputs(1-inputIndex)->FunctionValues(), Inputs(inputIndex)->GradientValues(), GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)  
        {
            if (inputIndex > 1)
                throw std::invalid_argument("ElementTimes operation only takes two inputs.");

            Matrix<ElemType> sliceInput0Grad = Inputs(inputIndex)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInput1Value = Inputs(1-inputIndex)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(sliceInput1Value, sliceInput0Grad, sliceOutputGrad);
        }

        // depending on inputIndex, all the input variables change meaning
        // inputIndex == 0 (left) -  inputGradientValues[0], inputFunctionValues[1]
        // inputIndex == 1 (right) - inputGradientValues[1], inputFunctionValues[0]
        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            inputGradientValues.AddElementProductOf(gradientValues, inputFunctionValues);
#if NANCHECK
            inputGradientValues.HasNan("ElementTimes");
#endif
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            //case taskComputeInputPartialRecurrent:
            //    descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
            //    descriptor->FunctionParam(1-inputIndex, paramOptionsInput);
            //    descriptor->GradientParam(inputIndex, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
            //    descriptor->GradientParam();
            //    descriptor->SetFunction((FARPROC)ComputeInputPartialSR);
            //    break;
            case taskComputeInputPartial:
                descriptor->FunctionParam(1-inputIndex, paramOptionsInput);
                descriptor->GradientParam(inputIndex, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->SetFunction((FARPROC)ComputeInputPartialS);
                break;
            //case taskEvaluateRecurrent:
            //    descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
            //    descriptor->FunctionParam();
            //    descriptor->FunctionParam(0, paramOptionsInput);
            //    descriptor->FunctionParam(1, paramOptionsInput);
            //    descriptor->SetFunction((FARPROC)EvaluateThisNodeSR);
            //    break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, const Matrix<ElemType>& input1)  
        {
            functionValues.AssignElementProductOf(input0, input1);
#if NANCHECK
            functionValues.HasNan("ElementTimes");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("ElementTimes operation requires two inputs.");

            size_t index = 0;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            index = 1;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("ElementTimes operation: one of the operants has 0 element.");

            if (Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumRows() ||
                Inputs(1)->FunctionValues().GetNumCols() != Inputs(0)->FunctionValues().GetNumCols())
                throw std::logic_error("The Matrix<ElemType> dimension in the ElementTimes operation does not match.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()
        {
            if (IsChildAnImage(0))  //when conflict, give priority to child 0
                CopyImageSizeFromInput(0);
            else
                CopyImageSizeFromInput(1);
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }
    };

    template class ElementTimesNode<float>; 
    template class ElementTimesNode<double>;

    template<class ElemType>
    class PlusNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


    public:
        PlusNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        PlusNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        PlusNode(const PlusNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new PlusNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Plus";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("Plus operation only takes two inputs.");
            ComputationNodePtr child = Inputs(inputIndex);
            ComputeInputPartialS(FunctionValues(), GradientValues(), child->FunctionValues(), child->GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("Plus operation only takes two inputs.");

            //only the one with more columns can be sliced, if both have same columns both are sliced
            size_t cols0 = Inputs(inputIndex)->FunctionValues().GetNumCols(), cols1=Inputs(1-inputIndex)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (cols0 >= cols1)
            {
                Matrix<ElemType> sliceInput0Grad = Inputs(inputIndex)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput0Value = Inputs(inputIndex)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialS(sliceOutputValue, sliceOutputGrad, sliceInput0Value, sliceInput0Grad);
            }
            else 
            {
                ComputeInputPartialS(sliceOutputValue, sliceOutputGrad, Inputs(inputIndex)->FunctionValues(), Inputs(inputIndex)->GradientValues());
            }
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& functionValues, Matrix<ElemType>& gradientValues, Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues)
        {
#if DUMPOUTPUT

            functionValues.Print("PlusNode");
#endif

            size_t rowsc = inputFunctionValues.GetNumRows(), colsc = inputFunctionValues.GetNumCols();
            size_t rowsp = functionValues.GetNumRows(), colsp = functionValues.GetNumCols();
#if DUMPOUTPUT
            fprintf(stderr, "input dimensions %lld x %lld,  this node dimensions %lld x %lld\n", rowsc, colsc, rowsp, colsp);
            gradientValues.Print("Gradient-in");
            inputGradientValues.Print("child Gradient-in/out");
#endif

            if (colsc == colsp && rowsc == rowsp)
                inputGradientValues += gradientValues;
            else if (colsc == 1 && rowsc == 1)
                inputGradientValues += gradientValues.SumOfElements();
            else if (colsc == 1 && colsp != 1)
            {
                size_t colspExpand = rowsp*colsp/rowsc;
                gradientValues.Reshape(rowsc, colspExpand);
                Matrix<ElemType>::MultiplyAndAdd(gradientValues, false, ConstOnes(colspExpand, 1, functionValues.GetDeviceId()), false, inputGradientValues);
                gradientValues.Reshape(rowsp, colsp);
            }
            else if (rowsc == 1 && rowsp != 1)
                Matrix<ElemType>::MultiplyAndAdd(ConstOnes(1, rowsp,functionValues.GetDeviceId()), false, gradientValues, false, inputGradientValues);
            else
                throw std::runtime_error("Plus partial: unexpected condition.");
#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
                }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            //case taskComputeInputPartialRecurrent:
            //    descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
            //    descriptor->FunctionParam(-1, paramOptionsInput); // only used to check dimensions
            //    descriptor->GradientParam();
            //    descriptor->FunctionParam(inputIndex, paramOptionsInput); // only used to check dimensions
            //    descriptor->GradientParam(inputIndex, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
            //    descriptor->SetFunction((FARPROC)ComputeInputPartialSR);
            //    break;
            case taskComputeInputPartial:
                descriptor->FunctionParam(-1, paramOptionsInput); // only used to check dimensions
                descriptor->GradientParam();
                descriptor->FunctionParam(inputIndex, paramOptionsInput); // only used to check dimensions
                descriptor->GradientParam(inputIndex, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->SetFunction((FARPROC)ComputeInputPartialS);
                break;
            //case taskEvaluateRecurrent:
            //    descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
            //    descriptor->FunctionParam();
            //    descriptor->FunctionParam(0, paramOptionsInput);
            //    descriptor->FunctionParam(1, paramOptionsInput);
            //    descriptor->SetFunction((FARPROC)EvaluateThisNodeSR);
            //    break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }       


        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            size_t cols0 = Inputs(0)->FunctionValues().GetNumCols(), cols1=Inputs(1)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            //only the one with more columns can be sliced, if both have same columns both are sliced
            if (cols0 == cols1)
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value);
            }
            else if (cols0 > cols1)
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, Inputs(1)->FunctionValues());
            }
            else //cols0 < cols1)
            {
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value);
            }
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, Matrix<ElemType>& inputFunctionValues0, Matrix<ElemType>& inputFunctionValues1)  
        {
            size_t rows0 = inputFunctionValues0.GetNumRows(), cols0 = inputFunctionValues0.GetNumCols();
            size_t rows1 = inputFunctionValues1.GetNumRows(), cols1 = inputFunctionValues1.GetNumCols();
            functionValues.Resize(max(rows0, rows1), max(cols0,cols1));

            if ((rows0 == rows1 && cols0 == cols1) || ((rows0 == 1 || rows1 == 1) && cols0 == cols1))
            {
                functionValues.AssignSumOf(inputFunctionValues0, inputFunctionValues1);
            }
            else if (cols0 == 1 && rows1 % rows0 == 0)  //one is col vec with divisable rows, including scalar
            {
                inputFunctionValues1.Reshape(rows0, rows1 * cols1 / rows0);
                functionValues.AssignSumOf(inputFunctionValues0, inputFunctionValues1);
                inputFunctionValues1.Reshape(rows1, cols1);
                functionValues.Reshape(max(rows0, rows1), max(cols0,cols1));
            }
            else if (cols1 == 1 && rows0 % rows1 == 0)  //one is col vec with divisable rows, including scalar
            {
                inputFunctionValues0.Reshape(rows1, rows0 * cols0 / rows1);
                functionValues.AssignSumOf(inputFunctionValues0, inputFunctionValues1);
                inputFunctionValues0.Reshape(rows0, cols0);
                functionValues.Reshape(max(rows0, rows1), max(cols0,cols1));
            }       

#if NANCHECK
            m_functionValues.HasNan("Plus");
#endif
#if DUMPOUTPUT
            functionValues.Print("PlusNode");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("Plus operation requires two inputs.");

            //if dimention not specified we assume two operants' dimentions should be the same
            size_t index = 0;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            index = 1;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            if ((Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0) && this->LoopId() < 0)
                throw std::logic_error("Plus operation: one of the operants has 0 element.");

            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if ((!(rows0 == rows1 && cols0 == cols1) &&  //match size
                !((rows0 == 1 || rows1 == 1) && cols0 == cols1) && //one is row vec
                !((cols0 == 1 && rows1 % rows0 == 0) || (cols1 == 1 && rows0 % rows1 == 0)))&& this->LoopId() < 0) //one is col vec with divisable rows, including scalar
            {
                throw std::logic_error("The Matrix dimension in the Plus operation does not match.");
            }       

            FunctionValues().Resize(max(rows0, rows1), max(cols0,cols1) );
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs() //based on the matrix with larger size
        {
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if (rows0 > rows1 || cols0 > cols1) //child 0 is larger
                CopyImageSizeFromInput(0);
            else if (rows0 < rows1 || cols0 < cols1) //child 1 is larger
                CopyImageSizeFromInput(1);
            else //same size
            {
                if (IsChildAnImage(0))  //when conflict, give priority to child 0
                    CopyImageSizeFromInput(0);
                else
                    CopyImageSizeFromInput(1);
            }
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }
    };

    template class PlusNode<float>; 
    template class PlusNode<double>;

    template<class ElemType>
    class MinusNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


    public:
        MinusNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        MinusNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        MinusNode(const MinusNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new MinusNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Minus";}

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("Minus operation only takes two inputs.");

            // prepare a matrix of ones as needed
            ComputationNodePtr child = Inputs(inputIndex);
            size_t rowsc = child->FunctionValues().GetNumRows(), colsc = child->FunctionValues().GetNumCols();
            size_t rowsp = FunctionValues().GetNumRows(), colsp = FunctionValues().GetNumCols();

            Matrix<ElemType> ones = Matrix<ElemType>();
            if (colsc == 1 && colsp != 1)
            {
                size_t colspExpand = rowsp*colsp/rowsc;
                ones = ConstOnes(colspExpand, 1,FunctionValues().GetDeviceId());
            }
            else if (rowsc == 1 && rowsp != 1)
            {
                ones = ConstOnes(1, rowsp,FunctionValues().GetDeviceId());
            }

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(child->FunctionValues(), child->GradientValues(), FunctionValues(), GradientValues(), ones); 
            }
            else  //right derivative
        {
                ComputeInputPartialRight(child->FunctionValues(), child->GradientValues(), FunctionValues(), GradientValues(), ones); 
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq) 
        {
            //only the one with more columns can be sliced, if both have same columns both are sliced
            size_t cols0 = Inputs(inputIndex)->FunctionValues().GetNumCols(), cols1=Inputs(1-inputIndex)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInput0Grad = Inputs(inputIndex)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput0Value = Inputs(inputIndex)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> ones = Matrix<ElemType>();

            size_t rowsc = Inputs(inputIndex)->FunctionValues().GetNumRows(), rowsp = FunctionValues().GetNumRows();
            size_t colsp = FunctionValues().GetNumCols();

            if (cols0 >= cols1) //indicates cols0 == functionValue.cols
            {

                if (rowsc == 1 && rowsp != 1)
        {
                    ones = ConstOnes(1, rowsp, FunctionValues().GetDeviceId());
                }
    
                if (inputIndex == 0)  //left derivative
            {
                    ComputeInputPartialLeft(sliceInput0Value, sliceInput0Grad, sliceOutputValue, sliceOutputGrad, ones); 
            }
                else  //right derivativeAzqz
            {
                    ComputeInputPartialRight(sliceInput0Value, sliceInput0Grad, sliceOutputValue, sliceOutputGrad, ones); 
            }
            }
            else // cols0 < cols1 -> cols0=1
            {
                if (cols0 == 1 && colsp != 1)
                {
                    size_t colspExpand = rowsp*colsp/rowsc;
                    ones = ConstOnes(colspExpand, 1,FunctionValues().GetDeviceId());
                }

                if (inputIndex == 0)  //left derivative
            {
                    ComputeInputPartialLeft(sliceInput0Value, sliceInput0Grad, sliceOutputValue, sliceOutputGrad, ones); 
                }
                else  //right derivative
                {
                    ComputeInputPartialRight(sliceInput0Value, sliceInput0Grad, sliceOutputValue, sliceOutputGrad, ones); 
                }
            }
        }

        static void WINAPI ComputeInputPartialLeft(Matrix<ElemType>& childFunctionValues, Matrix<ElemType>& childGradientValues, const Matrix<ElemType>& functionValues, /*const*/ Matrix<ElemType>& gradientValues, /*const*/ Matrix<ElemType>& ones)
            {
            ComputeInputPartialS(0, childFunctionValues, childGradientValues, functionValues, gradientValues, ones);
        }

        static void WINAPI ComputeInputPartialRight(Matrix<ElemType>& childFunctionValues, Matrix<ElemType>& childGradientValues, const Matrix<ElemType>& functionValues, /*const*/ Matrix<ElemType>& gradientValues, /*const*/ Matrix<ElemType>& ones)  
                {
            ComputeInputPartialS(1, childFunctionValues, childGradientValues, functionValues, gradientValues, ones);
        }

        static void WINAPI ComputeInputPartialS(const size_t inputIndex, Matrix<ElemType>& childFunctionValues, Matrix<ElemType>& childGradientValues, const Matrix<ElemType>& functionValues, /*const*/ Matrix<ElemType>& gradientValues, /*const*/ Matrix<ElemType>& ones)
        {
            ElemType weight = ElemType(inputIndex == 0? 1:-1);
            size_t rowsc = childFunctionValues.GetNumRows(), colsc = childFunctionValues.GetNumCols();
            size_t rowsp = functionValues.GetNumRows(), colsp = functionValues.GetNumCols();

            if (colsc == 1 && colsp != 1)
            {
                size_t colspExpand = rowsp*colsp/rowsc;
                ones.Resize(colspExpand, 1);
                }
            else if (rowsc == 1 && rowsp != 1)
            {
                ones.Resize(1, rowsp);
            }

            if (colsc == colsp && rowsc == rowsp)
            {
                if (inputIndex == 0)
                    childGradientValues += gradientValues;
                else
                    childGradientValues -= gradientValues;
            }
            else if (colsc == 1 && rowsc == 1)
                {
                if (inputIndex == 0)
                    childGradientValues += gradientValues.SumOfElements();
                else
                    childGradientValues -= gradientValues.SumOfElements();
                }
            else if (colsc == 1 && colsp != 1)
            {
                size_t colspExpand = rowsp*colsp/rowsc;
                gradientValues.Reshape(rowsc, colspExpand);
                Matrix<ElemType>::MultiplyAndWeightedAdd(weight, gradientValues, false, ones, false, 1, childGradientValues);
                gradientValues.Reshape(rowsp, colsp);
            }
            else if (rowsc == 1 && rowsp != 1)
                Matrix<ElemType>::MultiplyAndWeightedAdd(weight, ones, false, gradientValues, false, 1, childGradientValues);
            else
                throw std::runtime_error("Minus partial: unexpected condition.");
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                {
                descriptor->FunctionParam(inputIndex, paramOptionsInput);
                descriptor->GradientParam(inputIndex, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->FunctionParam(-1, paramOptionsInput);
                descriptor->GradientParam();

                // use dimensions of functionValues, will always be big enough
                ParamData<ElemType>* param = descriptor->MatrixParam(m_functionValues, "ones", paramOptionsInput | paramOptionsTemporary | paramOptionsInitialize);
                ElemType val(1.0);
                param->SetInitialize(val);

                descriptor->SetFunction(inputIndex?(FARPROC)ComputeInputPartialRight:(FARPROC)ComputeInputPartialLeft);
                break;
                }
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());  
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            size_t cols0 = Inputs(0)->FunctionValues().GetNumCols(), cols1=Inputs(1)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            //only the one with more columns can be sliced, if both have same columns both are sliced
            if (cols0 == cols1)
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value);
            }
            else if (cols0 > cols1)
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, Inputs(1)->FunctionValues());
            }
            else //cols0 < cols1)
            {
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value);
            }
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, Matrix<ElemType>& in0, Matrix<ElemType>& in1)  
        {
            size_t rows0 = in0.GetNumRows(), cols0 = in0.GetNumCols();
            size_t rows1 = in1.GetNumRows(), cols1 = in1.GetNumCols();
            functionValues.Resize(max(rows0, rows1), max(cols0,cols1));

            if ((rows0 == rows1 && cols0 == cols1) || ((rows0 == 1 || rows1 == 1) && cols0 == cols1))
            {
                functionValues.AssignDifferenceOf(in0, in1);
            }
            else if (cols0 == 1 && rows1 % rows0 == 0)  //one is col vec with divisable rows, including scalar
            {
                in1.Reshape(rows0, rows1 * cols1 / rows0);
                functionValues.AssignDifferenceOf(in0, in1);
                in1.Reshape(rows1, cols1);
                functionValues.Reshape(max(rows0, rows1), max(cols0,cols1));
            }
            else if (cols1 == 1 && rows0 % rows1 == 0)  //one is col vec with divisable rows, including scalar
            {
                in0.Reshape(rows1, rows0 * cols0 / rows1);
                functionValues.AssignDifferenceOf(in0, in1);
                in0.Reshape(rows0, cols0);
                functionValues.Reshape(max(rows0, rows1), max(cols0,cols1));
            }      
#if NANCHECK
            functionValues.HasNan("Minus");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("Minus operation requires two inputs.");

            //if dimention is missing make the two operatants to have same size
            size_t index = 0;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            index = 1;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Minus operation: one of the operants has 0 element.");

            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if (!(rows0 == rows1 && cols0 == cols1) &&  //match size
                !((rows0 == 1 || rows1 == 1) && cols0 == cols1) && //one is row vec
                !((cols0 == 1 && rows1 % rows0 == 0) || (cols1 == 1 && rows0 % rows1 == 0)))  //one is col vec with divisable rows, including scalar
            {
                throw std::logic_error("The Matrix dimension in the Minus operation does not match.");
            }       

            FunctionValues().Resize(max(rows0, rows1), max(cols0,cols1) );
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs() //based on the matrix with larger size
        {
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if (rows0 > rows1 || cols0 > cols1) //child 0 is larger
                CopyImageSizeFromInput(0);
            else if (rows0 < rows1 || cols0 < cols1) //child 1 is larger
                CopyImageSizeFromInput(1);
            else //same size
            {
                if (IsChildAnImage(0))  //when conflict, give priority to child 0
                    CopyImageSizeFromInput(0);
                else
                    CopyImageSizeFromInput(1);
            }
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }
    };

    template class MinusNode<float>; 
    template class MinusNode<double>;

    //The first matrix should be a vector regpresting the diagonal of a square matrix in the DiagTimes operation
    template<class ElemType>
    class DiagTimesNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


    public:
        DiagTimesNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode(deviceId), m_innerproduct(deviceId), m_rightGradient(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        DiagTimesNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_innerproduct(deviceId), m_rightGradient(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"DiagTimes";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("DiagTimes operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(m_innerproduct, Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
            }
            else  //right derivative
            {
                ComputeInputPartialRight(m_rightGradient, Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues());
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("DiagTimes operation only takes two inputs.");

            //left parameter (diag matix cannot be sliced)
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (inputIndex == 0)  //left derivative
            {
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                ComputeInputPartialLeft(m_innerproduct, sliceInput1Value, Inputs(0)->GradientValues(), sliceOutputGrad);
            }
            else  //right derivative
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                ComputeInputPartialRight(m_rightGradient, Inputs(0)->FunctionValues(), sliceInput1Grad, sliceOutputGrad);
            }
        }

        static void WINAPI ComputeInputPartialLeft(Matrix<ElemType>& temp, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            temp.AssignInnerProductOf(gradientValues, inputFunctionValues, false);
            inputGradientValues += temp;
        }

        static void WINAPI ComputeInputPartialRight(Matrix<ElemType>& temp, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            temp.SetValue(gradientValues);
            temp.ColumnElementMultiplyWith(inputFunctionValues);
            inputGradientValues += temp;
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            //case taskComputeInputPartialRecurrent:
            //    descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
            //    descriptor->FunctionParam(1-inputIndex, paramOptionsInput);
            //    descriptor->GradientParam(inputIndex, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
            //    descriptor->GradientParam();
            //    descriptor->Param(paramTypeLongLong, "nbrSlicesInEachIter");
            //    descriptor->SetFunction((inputIndex?(FARPROC)ComputeInputPartialRight:(FARPROC)ComputeInputPartialLeft));
            //    break;
            case taskComputeInputPartial:
                descriptor->FunctionParam(1-inputIndex, paramOptionsInput);
                descriptor->GradientParam(inputIndex,paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->SetFunction((inputIndex?(FARPROC)ComputeInputPartialRight:(FARPROC)ComputeInputPartialLeft));
                break;
            //case taskEvaluateRecurrent:
            //    descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
            //    descriptor->FunctionParam();
            //    descriptor->FunctionParam(0, paramOptionsInput);
            //    descriptor->FunctionParam(1, paramOptionsInput);
            //    descriptor->Param(paramTypeLongLong, "nbrSlicesInEachIter");
            //    descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
            //    break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }


        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues()); 
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value); 
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues0, const Matrix<ElemType>& inputFunctionValues1)  
        {
            functionValues.SetValue(inputFunctionValues1);
            functionValues.ColumnElementMultiplyWith(inputFunctionValues0);
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("DiagTimes operation requires two inputs.");

            //if dimention not specified we assume two operants' dimentions should match
            if (Inputs(0)->OperationName() == LearnableParameter<ElemType>::TypeName() && Inputs(0)->FunctionValues().GetNumRows() == 0 && Inputs(1)->FunctionValues().GetNumRows() != 0)
            {
                Inputs(0)->FunctionValues().Resize(Inputs(1)->FunctionValues().GetNumRows(), 1);
            }

            if (Inputs(1)->OperationName() == LearnableParameter<ElemType>::TypeName() && Inputs(0)->FunctionValues().GetNumRows() != 0 && Inputs(1)->FunctionValues().GetNumRows() == 0)
            {
                Inputs(1)->FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());
            }

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("DiagTimes operation: one of the operants has 0 element.");

            if (Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumRows())
                throw std::logic_error("The Matrix dimension in the DiagTimes operation does not match.");

            if (1 != Inputs(0)->FunctionValues().GetNumCols())
                throw std::logic_error("The first matrix should be a vector regpresting the diagonal of a square matrix in the DiagTimes operation.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());
            m_innerproduct.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());
            m_rightGradient.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());

            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs() //this is element wise scaling, so based on child 1
        {
            CopyImageSizeFromInput(1);
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_innerproduct.GetDeviceId() != deviceId)
                    m_innerproduct.TransferFromDeviceToDevice(m_innerproduct.GetDeviceId(), deviceId);
                if (m_rightGradient.GetDeviceId() != deviceId)
                    m_rightGradient.TransferFromDeviceToDevice(m_rightGradient.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            DiagTimesNode<ElemType>* node = (DiagTimesNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_innerproduct = m_innerproduct;
                node->m_rightGradient = m_rightGradient;
            }
        }

        // copy constructor
        DiagTimesNode(const DiagTimesNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), m_innerproduct(node->m_deviceId), m_rightGradient(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new DiagTimesNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_innerproduct;
        Matrix<ElemType> m_rightGradient;
    };

    template class DiagTimesNode<float>; 
    template class DiagTimesNode<double>;

    //The first matrix should be a vector regpresting the diagonal of a square matrix in the DiagTimes operation
    template<class ElemType>
    class CosDistanceNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


    public:
        CosDistanceNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode(deviceId), m_invNorm0(deviceId), m_invNorm1(deviceId), m_leftTerm(deviceId), m_rightTerm(deviceId), m_temp(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        CosDistanceNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_invNorm0(deviceId), m_invNorm1(deviceId), m_leftTerm(deviceId), m_rightTerm(deviceId), m_temp(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"CosDistance";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("CosDistance operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(m_invNorm0, m_invNorm1, FunctionValues(), m_temp, m_rightTerm, m_leftTerm, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(inputIndex)->GradientValues());  
            }
            else  //right derivative
            {
                ComputeInputPartialRight(m_invNorm0, m_invNorm1, FunctionValues(), m_temp, m_rightTerm, m_leftTerm, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(inputIndex)->GradientValues());  
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq) 
        {
            if (inputIndex > 1)
                throw std::invalid_argument("CosDistance operation only takes two inputs.");

            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInputGrad = Inputs(inputIndex)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(m_invNorm0, m_invNorm1, sliceOutputValue, m_temp, m_rightTerm, m_leftTerm, sliceInput0Value, sliceInput1Value, sliceInputGrad);  
            }
            else  //right derivative
            {
                ComputeInputPartialRight(m_invNorm0, m_invNorm1, sliceOutputValue, m_temp, m_rightTerm, m_leftTerm, sliceInput0Value, sliceInput1Value, sliceInputGrad);  
            }
        }

        static void WINAPI ComputeInputPartialLeft(const Matrix<ElemType>& invNorm0, const Matrix<ElemType>& invNorm1, const Matrix<ElemType>& functionValues, 
            Matrix<ElemType>& temp, Matrix<ElemType>& rightTerm, Matrix<ElemType>& leftTerm, // the temporary variables
            const Matrix<ElemType>& in0, const Matrix<ElemType>& in1, 
            Matrix<ElemType>& inputGradientValues)
        {
            ComputeInputPartialS(0, invNorm0, invNorm1, functionValues, temp, rightTerm, leftTerm, in0, in1, inputGradientValues);  
        }

        static void WINAPI ComputeInputPartialRight(const Matrix<ElemType>& invNorm0, const Matrix<ElemType>& invNorm1, const Matrix<ElemType>& functionValues, 
            Matrix<ElemType>& temp, Matrix<ElemType>& rightTerm, Matrix<ElemType>& leftTerm, // the temporary variables
            const Matrix<ElemType>& in0, const Matrix<ElemType>& in1, 
            Matrix<ElemType>& inputGradientValues)  
        {
            ComputeInputPartialS(1, invNorm0, invNorm1, functionValues, temp, rightTerm, leftTerm, in0, in1, inputGradientValues);  
        }

        // functionValues, invNorm0, invNorm1 - output from the EvaluateNode() method
        // temp, rightTerm, leftTerm - temporary matrices
        // in0, in1 - input functionValues from other nodes
        // inputGradientValues(x) - gradients to update, where x matches inputIndex
        static void WINAPI ComputeInputPartialS(const size_t inputIndex, const Matrix<ElemType>& invNorm0, const Matrix<ElemType>& invNorm1, const Matrix<ElemType>& functionValues, 
            Matrix<ElemType>& temp, Matrix<ElemType>& rightTerm, Matrix<ElemType>& leftTerm, // the temporary variables
            const Matrix<ElemType>& in0, const Matrix<ElemType>& in1, 
            Matrix<ElemType>& inputGradientValues)  
        {
            if (inputIndex == 0)  //left derivative
            {
                temp.AssignElementProductOf(invNorm0, invNorm0);
            }
            else  //right derivative
            {
                temp.AssignElementProductOf(invNorm1, invNorm1);
            }

            temp.ElementMultiplyWith(functionValues);
            rightTerm.SetValue(inputIndex?in1:in0);
            rightTerm.RowElementMultiplyWith(temp);

            temp.AssignElementProductOf(invNorm0, invNorm1);
            leftTerm.SetValue(inputIndex?in0:in1);
            leftTerm.RowElementMultiplyWith(temp);

            Matrix<ElemType>::AddScaledDifference(1, leftTerm, rightTerm, inputGradientValues);
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                descriptor->MatrixParam(m_invNorm0, "invNorm0", paramOptionsInput);
                descriptor->MatrixParam(m_invNorm1, "invNorm1", paramOptionsInput);
                descriptor->FunctionParam(-1, paramOptionsInput);
                descriptor->MatrixParam(m_temp, "temp", paramOptionsInput | paramOptionsTemporary);
                descriptor->MatrixParam(m_rightTerm, "rightTerm", paramOptionsInput | paramOptionsTemporary);
                descriptor->MatrixParam(m_leftTerm, "leftTerm", paramOptionsInput | paramOptionsTemporary);
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->GradientParam(inputIndex,paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->SetFunction(inputIndex?(FARPROC)ComputeInputPartialRight:(FARPROC)ComputeInputPartialLeft);
                break;
            case taskEvaluate:
                descriptor->MatrixParam(m_invNorm0, "invNorm0", paramOptionsOutput);
                descriptor->MatrixParam(m_invNorm1, "invNorm1", paramOptionsOutput);
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_invNorm0, m_invNorm1, FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());  
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(m_invNorm0, m_invNorm1, sliceOutputValue, sliceInput0Value, sliceInput1Value);  
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& invNorm0, Matrix<ElemType>& invNorm1, 
            Matrix<ElemType>& functionValues, Matrix<ElemType>& in0, Matrix<ElemType>& in1)  
        {
            invNorm0.AssignVectorNorm2Of(in0, true); // seems to modify input (in0)
            invNorm0.AssignElementInverseOf(invNorm0);

            invNorm1.AssignVectorNorm2Of(in1, true); // seems to modify the input (in1)
            invNorm1.AssignElementInverseOf(invNorm1);

            functionValues.AssignInnerProductOf(in0, in1, true);
            functionValues.ElementMultiplyWith(invNorm0);
            functionValues.ElementMultiplyWith(invNorm1);
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("CosDistance operation requires two inputs.");

            //if dimention is missing make the two operatants to have same size
            size_t index = 0;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            index = 1;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("CosDistance operation: one of the operants has 0 element.");

            if (Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumRows() || 
                Inputs(1)->FunctionValues().GetNumCols() != Inputs(0)->FunctionValues().GetNumCols())
                throw std::logic_error("The Matrix dimension in the CosDistance operation does not match.");

            FunctionValues().Resize(1, Inputs(1)->FunctionValues().GetNumCols());
            size_t rowsp = FunctionValues().GetNumRows(), colsp = FunctionValues().GetNumCols();
            m_invNorm0.Resize(rowsp, colsp);
            m_invNorm1.Resize(rowsp, colsp);
            m_leftTerm.Resize(rowsp, colsp);
            m_rightTerm.Resize(rowsp, colsp);
            m_temp.Resize(rowsp, colsp);

            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs() 
        {
            CopyImageSizeFromInput(0, false);

            m_outputChannels = 1;
            m_outputWidth = 1;
            m_outputHeight = 1;        
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_invNorm0.GetDeviceId() != deviceId)
                    m_invNorm0.TransferFromDeviceToDevice(m_invNorm0.GetDeviceId(), deviceId);
                if (m_invNorm1.GetDeviceId() != deviceId)
                    m_invNorm1.TransferFromDeviceToDevice(m_invNorm1.GetDeviceId(), deviceId);
                if (m_leftTerm.GetDeviceId() != deviceId)
                    m_leftTerm.TransferFromDeviceToDevice(m_leftTerm.GetDeviceId(), deviceId);
                if (m_rightTerm.GetDeviceId() != deviceId)
                    m_rightTerm.TransferFromDeviceToDevice(m_rightTerm.GetDeviceId(), deviceId);
                if (m_temp.GetDeviceId() != deviceId)
                    m_temp.TransferFromDeviceToDevice(m_temp.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            CosDistanceNode<ElemType>* node = (CosDistanceNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_invNorm0 = m_invNorm0;
                node->m_invNorm1 = m_invNorm1;
                node->m_leftTerm = m_leftTerm;
                node->m_rightTerm = m_rightTerm;
                node->m_temp = m_temp;
            }
        }

        // copy constructor
        CosDistanceNode(const CosDistanceNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), m_invNorm0(node->m_deviceId), m_invNorm1(node->m_deviceId), m_leftTerm(node->m_deviceId), m_rightTerm(node->m_deviceId), m_temp(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new CosDistanceNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        // invNorm nodes tranfer data between EvaluateThisNode and ComputeInputPartial
        Matrix<ElemType> m_invNorm0;
        Matrix<ElemType> m_invNorm1;
        // the rest are temporaries, values don't need to be maintained
        Matrix<ElemType> m_leftTerm;
        Matrix<ElemType> m_rightTerm;
        Matrix<ElemType> m_temp;
    };

    template class CosDistanceNode<float>; 
    template class CosDistanceNode<double>;

    template<class ElemType>
    class DropoutNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


    public:

        DropoutNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode(deviceId), m_maskOfDropout(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_dropoutRate = 0;
            m_randomSeed = (ULONG) InterlockedIncrement64(&s_timeStampCounter);;
            InitRecurrentNode();
        }

        DropoutNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_maskOfDropout(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_dropoutRate = 0;  //dropout is consisered as a training parameter and thus not reinitialized if loadfromfile
            m_randomSeed = (ULONG) InterlockedIncrement64(&s_timeStampCounter);

            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("Dropout operation only takes one input.");
            ComputeInputPartialS(m_dropoutRate, Inputs(0)->GradientValues(), m_maskOfDropout, GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("Dropout operation only takes one input.");

            Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceMask = Matrix<ElemType>();
            if(m_dropoutRate > 0)
            {
                sliceMask = m_maskOfDropout.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            }

            ComputeInputPartialS(m_dropoutRate, sliceInput0Grad, sliceMask, sliceOutputGrad);
        }

        static void WINAPI ComputeInputPartialS(const ElemType dropoutRate, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& maskOfDropout, const Matrix<ElemType>& gradientValues)
        {
            if (dropoutRate > 0)
            {
                inputGradientValues.AddElementProductOf(gradientValues, maskOfDropout);
            }
            else
            {   
                inputGradientValues += gradientValues;
            }
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                descriptor->Param(paramTypeLong, "dropoutRate");
                descriptor->GradientParam(inputIndex,paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->MatrixParam(m_maskOfDropout, "maskOfDropout", paramOptionsInput | paramOptionsTemporary);
                descriptor->GradientParam();
                descriptor->SetFunction((FARPROC)ComputeInputPartialS);
                break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_dropoutRate, m_randomSeed, FunctionValues(), m_maskOfDropout, Inputs(0)->FunctionValues());
        }
        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceMask = Matrix<ElemType>();
            if(m_dropoutRate > 0)
            {
                m_maskOfDropout.Resize(m_functionValues.GetNumRows(), m_functionValues.GetNumCols());
                sliceMask = m_maskOfDropout.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            }

            EvaluateThisNodeS(m_dropoutRate, m_randomSeed, sliceOutputValue, sliceMask, sliceInput0Value);
        }

        static void WINAPI EvaluateThisNodeS(const ElemType dropoutRate, ULONG& randomSeed, Matrix<ElemType>& functionValues, Matrix<ElemType>& maskOfDropout, const Matrix<ElemType>& inputFunctionValues)
        {
            if(dropoutRate > 0)
            {
                maskOfDropout.Resize(inputFunctionValues.GetNumRows(), inputFunctionValues.GetNumCols());

                maskOfDropout.SetUniformRandomMask(dropoutRate, ElemType(1.0) / (ElemType(1) - dropoutRate), randomSeed);
                randomSeed += 1073807359;  //1073807359 is a very large prime number to avoid collision with other dropout nodes

                functionValues.AssignElementProductOf(maskOfDropout, inputFunctionValues);
#if NANCHECK
                functionValues.HasNan("DropOut");
#endif
            }
            else
            {
                //remove this line since we can get same effect by overwritting the FunctionValues functions without copying the values
                //functionValues = inputFunctionValues;
            }
        }

        virtual const Matrix<ElemType>& FunctionValues() const 
        {
            if(m_dropoutRate > 0)
                 return m_functionValues;
            else
                return Inputs(0)->FunctionValues();
        }

        virtual Matrix<ElemType>& FunctionValues() 
        {
            if(m_dropoutRate > 0)
                 return m_functionValues;
            else
                return Inputs(0)->FunctionValues();
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("Dropout operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Dropout operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_maskOfDropout.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr inputNode) 
        {
            m_children.resize(1);
            m_children[0] = inputNode;
        }

        void SetDropoutRate(const ElemType val)
        {
            if (val < 0 || val >= 1) 
                throw std::logic_error("DropoutRate must be >= 0 and < 1.");
            m_dropoutRate = val;
        }

        void SetRandomSeed(const ULONG val)
        {
            m_randomSeed = (ULONG) val;
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_maskOfDropout.GetDeviceId() != deviceId)
                    m_maskOfDropout.TransferFromDeviceToDevice(m_maskOfDropout.GetDeviceId(), deviceId, true);
            }
        }

        static const std::wstring TypeName() {return L"Dropout";} 

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            DropoutNode<ElemType>* node = (DropoutNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_dropoutRate = m_dropoutRate;
                node->m_randomSeed = m_randomSeed;
                node->m_maskOfDropout = m_maskOfDropout;
            }
        }

        // copy constructor
        DropoutNode(const DropoutNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), m_maskOfDropout(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new DropoutNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        ElemType m_dropoutRate;
        ULONG m_randomSeed;

        Matrix<ElemType> m_maskOfDropout;
    };

    template class DropoutNode<float>; 
    template class DropoutNode<double>;

    template<class ElemType>
    class KhatriRaoProductNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 

    public:
        KhatriRaoProductNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        KhatriRaoProductNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        KhatriRaoProductNode(const KhatriRaoProductNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new KhatriRaoProductNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"KhatriRaoProduct";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("KhatriRaoProduct operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues()); 
            }
            else  //right derivative
            {
                ComputeInputPartialRight(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues()); 
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq) 
        {
            if (inputIndex > 1)
                throw std::invalid_argument("KhatriRaoProduct operation only takes two inputs.");

            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (inputIndex == 0)  //left derivative
            {
                Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialLeft(sliceInput1Value, sliceInput0Grad, sliceOutputGrad); 
            }
            else  //right derivative
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialRight(sliceInput0Value, sliceInput1Grad, sliceOutputGrad); 
            }
        }

        static void WINAPI ComputeInputPartialLeft(Matrix<ElemType>& childFunctionValues, Matrix<ElemType>& childGradientValues, const Matrix<ElemType>& gradientValues)
        {
            childGradientValues.AddColumnReshapeProductOf(gradientValues, childFunctionValues, false);
        }

        static void WINAPI ComputeInputPartialRight(Matrix<ElemType>& childFunctionValues, Matrix<ElemType>& childGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            childGradientValues.AddColumnReshapeProductOf(gradientValues, childFunctionValues, true);
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                descriptor->FunctionParam(1-inputIndex, paramOptionsInput);
                descriptor->GradientParam(inputIndex,paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->SetFunction(inputIndex?(FARPROC)ComputeInputPartialRight:(FARPROC)ComputeInputPartialLeft);
                break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());  
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value); 
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, Matrix<ElemType>& in0, Matrix<ElemType>& in1)  
        {
            functionValues.AssignKhatriRaoProductOf(in0,in1);
#if NANCHECK
            m_functionValues.HasNan("KhatriRaoProduct");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("KhatriRaoProduct operation requires two inputs.");

            //support automatic dimention inference for learnable parameters
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if (rows0 == 0 || rows1 == 0)
                throw logic_error("KhatriRaoProduct operation: The number of rows in the input should not be 0.");

            if (Inputs(0)->OperationName() == LearnableParameter<ElemType>::TypeName() && cols0 == 0 && cols1 != 0)
                Inputs(0)->FunctionValues().Resize(rows0, cols1);

            if (Inputs(1)->OperationName() == LearnableParameter<ElemType>::TypeName() && cols0 != 0 && cols1 == 0)
                Inputs(1)->FunctionValues().Resize(rows1, cols0);

            //cols may be changed before this line and so cannot use cached cols values below
            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("KhatriRaoProduct operation: One of the operants has 0 elements.");

            if (Inputs(1)->FunctionValues().GetNumCols() != Inputs(0)->FunctionValues().GetNumCols())
            {
                throw std::logic_error("The Matrices should have same number of columns.");
            }

            FunctionValues().Resize(rows0 * rows1, Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()  
        {
            //since it's symmetrical any one of the input may be the true input. 
            //since we dont' use the input image size info in the operation, the input part doesn't matter.
            CopyImageSizeFromInput(1, false); 

            //after KhatriRaoProduct the structure is lost
            m_outputWidth = 1;
            m_outputHeight = m_functionValues.GetNumRows();
            m_outputChannels =  1;
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }
    };

    template class KhatriRaoProductNode<float>; 
    template class KhatriRaoProductNode<double>;

    //originally designed to extract word embedding representation from bag-of-word. 
    //takes two inputs, input0 is weight matrix and input1 is the bag-of-word representation of the inputs
    template<class ElemType>
    class LookupTableNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr;

    public:
        LookupTableNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        LookupTableNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"LookupTable";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("LookupTable operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
            }
            else  //right derivative
        {
                ComputeInputPartialRight(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues());
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("LookupTable operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
        {
                Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialLeft(sliceInput1Value, Inputs(0)->GradientValues(), sliceOutputGrad);
            }
            else  //right derivative
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialRight(Inputs(0)->FunctionValues(), sliceInput1Grad, sliceOutputGrad);
            }
        }

        static void WINAPI ComputeInputPartialLeft(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues)  
        {
            size_t rows1 =inputFunctionValues.GetNumRows(), cols1 = inputFunctionValues.GetNumCols();
            size_t rowsp = gradientValues.GetNumRows(), colsp = gradientValues.GetNumCols();
            int wordsInEachSample = rows1 / inputGradientValues.GetNumCols();

            inputFunctionValues.Reshape(rows1 / wordsInEachSample, cols1 * wordsInEachSample);
            gradientValues.Reshape(rowsp / wordsInEachSample, colsp * wordsInEachSample);

            Matrix<ElemType>::MultiplyAndAdd(gradientValues, false, inputFunctionValues, true, inputGradientValues);

            inputFunctionValues.Reshape(rows1, cols1);
            gradientValues.Reshape(rowsp, colsp);
        }

        static void WINAPI ComputeInputPartialRight(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues)  
            {
            size_t rows1 =inputGradientValues.GetNumRows(), cols1 = inputGradientValues.GetNumCols();
            size_t rowsp = gradientValues.GetNumRows(), colsp = gradientValues.GetNumCols();
            int wordsInEachSample = rows1 / inputFunctionValues.GetNumCols();

            inputGradientValues.Reshape(rows1 / wordsInEachSample, cols1 * wordsInEachSample);
            gradientValues.Reshape(rowsp / wordsInEachSample, colsp * wordsInEachSample);

            Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, true, gradientValues, false, inputGradientValues);

            inputGradientValues.Reshape(rows1, cols1);
            gradientValues.Reshape(rowsp, colsp);
        }

        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value);
        }

        //input0 is the weight (each column is an embedding of one word), input 1 contains m_bnrLooked words in each column (sample)
        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, Matrix<ElemType>& input1)  
        {
            size_t rows1 =input1.GetNumRows(), cols1 = input1.GetNumCols();
            int wordsInEachSample = rows1 / input0.GetNumCols();

            input1.Reshape(rows1 / wordsInEachSample, cols1 * wordsInEachSample);

            functionValues.AssignProductOf(input0, false, input1, false);

            input1.Reshape(rows1, cols1);
            size_t rows = functionValues.GetNumRows();
            functionValues.Reshape(rows * wordsInEachSample, cols1);
        }
            
        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (Inputs(1)->FunctionValues().GetNumRows() % Inputs(0)->FunctionValues().GetNumCols() != 0)
                throw invalid_argument("Mismatched dimention. rows in input1 must be multiples of cols in input0.");

            int wordsInEachSample = Inputs(1)->FunctionValues().GetNumRows() / Inputs(0)->FunctionValues().GetNumCols();
          
            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows() * wordsInEachSample, Inputs(1)->FunctionValues().GetNumCols());

            CopyImageSizeFromInputs(); 
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
        {
            //case taskComputeInputPartialRecurrent:
            //    descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
            //    descriptor->FunctionParam(1-inputIndex, paramOptionsInput);
            //    descriptor->GradientParam(inputIndex, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
            //    descriptor->GradientParam();
            //    descriptor->SetFunction((inputIndex?(FARPROC)ComputeInputPartialRightR:(FARPROC)ComputeInputPartialLeftR));
            //    break;
            case taskComputeInputPartial:
                descriptor->FunctionParam(1-inputIndex, paramOptionsInput);
                descriptor->GradientParam(inputIndex,paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->SetFunction((inputIndex?(FARPROC)ComputeInputPartialRight:(FARPROC)ComputeInputPartialLeft));
                break;
            //case taskEvaluateRecurrent:
            //    descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
            //    descriptor->FunctionParam();
            //    descriptor->FunctionParam(0, paramOptionsInput);
            //    descriptor->FunctionParam(1, paramOptionsInput);
            //    descriptor->Param(paramTypeSizet, "mNbrSlicesInEachRecurrentIter", paramOptionsInput);
            //    break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
        }
            return descriptor;
        }
        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
            ComputationNodePtr node = new LookupTableNode<ElemType>(this, name, flags);
            return node;
        }

        LookupTableNode(const LookupTableNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }
    };

    template class LookupTableNode<float>;
    template class LookupTableNode<double>;

    template<class ElemType>
    class DelayNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


        ElemType  m_default_activity; 

    public:

        DelayNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode(deviceId), m_pastActivity(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_default_activity = (ElemType)DEFAULT_HIDDEN_ACTIVITY;
            m_delay = 1;
            m_functionValues.Resize(1,1);
            m_pastActivity.Resize(1,1);
            Reset();
            InitRecurrentNode();
        }
                
        DelayNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_pastActivity(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);

            m_default_activity = (ElemType)DEFAULT_HIDDEN_ACTIVITY;
            m_delay = 1;
            m_functionValues.Resize(1,1);
            m_pastActivity.Resize(1,1);
            Reset();

            LoadFromFile(fstream, modelVersion, deviceId);
        }

        void SaveToFile(File& fstream) const
        {
            fstream << OperationName() << NodeName();
            fstream << m_delay; 
            fstream << FunctionValues().GetNumRows() << FunctionValues().GetNumCols(); 
        }

        void LoadFromFile(File& fstream, const size_t /*modelVersion*/, const short deviceId = AUTOPLACEMATRIX)
        {
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();

            fstream >> m_delay;

            size_t iRow, timeIdxInSeq;
            fstream >> iRow >> timeIdxInSeq;
            FunctionValues().Resize(iRow,timeIdxInSeq);
            m_pastActivity.Resize(iRow, timeIdxInSeq);
        }

        DelayNode(const short deviceId, ElemType initHiddenActivity, size_t row_size, size_t col_size, const std::wstring name = L"") : ComputationNode(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_default_activity = initHiddenActivity;
            m_delay = 1;

            m_functionValues.Resize(row_size, col_size);
            m_functionValues.SetValue(m_default_activity);

            m_pastActivity.Resize(row_size, col_size);
            m_pastActivity.SetValue(m_default_activity);

            m_gradientValues.Resize(row_size, col_size);
            m_gradientValues.SetValue(0.0f);


            Reset();
            InitRecurrentNode();

        }

        virtual const std::wstring OperationName() const {return TypeName();}

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("Delay operation only takes one input.");

            size_t nbrSamples = GradientValues().GetNumCols() / m_samplesInRecurrentStep; 
            for (int i = nbrSamples - 1; i >= 0; i--)
            {
                ComputeInputPartialSR(i, m_delay, Inputs(0)->GradientValues(), GradientValues(), m_samplesInRecurrentStep);
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("Delay operation only takes one input.");
            assert(m_functionValues.GetNumRows() == GradientValues().GetNumRows()); // original used m_functionValues.GetNumRows() for loop dimension
            //if (m_samplesInRecurrentStep == 1)
			//{
			ComputeInputPartialSR(timeIdxInSeq, m_delay, Inputs(0)->GradientValues(), GradientValues(), m_samplesInRecurrentStep);
			/*}else
			{
				for (size_t i = 0 ; i < m_samplesInRecurrentStep; i++)
				{
					ComputeInputPartialSRP(timeIdxInSeq, m_delay, Inputs(0)->GradientValues(), GradientValues(), i, m_samplesInRecurrentStep);
				}
			}*/
        }

        static void WINAPI ComputeInputPartialSR(int timeIdxInSeq, int delay,  
            Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const size_t mNbr)
        {
            assert(timeIdxInSeq >= 0);
            if ((timeIdxInSeq - delay) >= 0 && (timeIdxInSeq - delay) * mNbr <= inputGradientValues.GetNumCols())
            {
                Matrix<ElemType> to = inputGradientValues.ColumnSlice((timeIdxInSeq - delay)*mNbr, mNbr);
                Matrix<ElemType> frm= gradientValues.ColumnSlice(timeIdxInSeq * mNbr, mNbr);
                to += frm; 
            }
        }

		static void WINAPI ComputeInputPartialSRP(int timeIdxInSeq, int delay,  
            Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const size_t indexInBatch, const size_t mNbr)
        {
            assert(timeIdxInSeq >= 0);
			if ((timeIdxInSeq - delay) >= 0 && (timeIdxInSeq - delay) * mNbr <= inputGradientValues.GetNumCols())
            {
                Matrix<ElemType> to = inputGradientValues.ColumnSlice((timeIdxInSeq - delay)*mNbr + indexInBatch, 1);
                Matrix<ElemType> frm= gradientValues.ColumnSlice(timeIdxInSeq * mNbr + indexInBatch, 1);
                to += frm; 
            }
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartialRecurrent:
                descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
                descriptor->Param(paramTypeInteger, "delay", paramOptionsInput | paramOptionsOutput);
                descriptor->GradientParam(inputIndex,paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->Param(paramTypeLongLong, "nbrSlicesInEachIter");
                descriptor->SetFunction((FARPROC)ComputeInputPartialSR);
                break;
            case taskEvaluateRecurrent:
                descriptor->Param(paramTypeInteger, "RecurrantIterator", paramOptionsInput | paramOptionsRecurrantIterator);
                descriptor->Param(paramTypeInteger, "delay", paramOptionsInput | paramOptionsOutput);
                descriptor->Param(paramTypeBool, "reset", paramOptionsInput | paramOptionsOutput);
                descriptor->Param(sizeof(ElemType)==4?paramTypeSingle:paramTypeDouble, "default_activity", paramOptionsInput);
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->Param(paramTypeLongLong, "nbrSlicesInEachIter");
                descriptor->SetFunction((FARPROC)EvaluateThisNodeSR);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            ASSERT(m_delay > 0);
            size_t blogSize = Inputs(0)->FunctionValues().GetNumCols();

            for (size_t i = 0; i < blogSize / m_samplesInRecurrentStep; i++)
                EvaluateThisNodeSR(i, m_delay, m_Reset, m_default_activity, m_functionValues, m_pastActivity, Inputs(0)->FunctionValues(), m_samplesInRecurrentStep);
            /// reset past activity
            m_pastActivity = Inputs(0)->FunctionValues();
        }

        void Reset()
        {
            m_Reset = true;
        }


        void NotReset()
        {
            m_Reset = false;
        }

		
		

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            /// reset past activity as it reached to the begining of a minibatch
            /// the node pointed hasn't yet updated, so it is the past activity 
            if (timeIdxInSeq == 0)
                m_pastActivity = Inputs(0)->FunctionValues();
			
			if (m_samplesInRecurrentStep == 1)
			{
                bool reset = (m_SentenceEnd[0] <= timeIdxInSeq);
                EvaluateThisNodeSR(timeIdxInSeq, m_delay, reset, m_default_activity, m_functionValues, m_pastActivity, Inputs(0)->FunctionValues(), m_samplesInRecurrentStep);
			} else
			{
				for (size_t i = 0 ; i < m_samplesInRecurrentStep; i++)
				{
					bool reset = (m_SentenceEnd[i] <= timeIdxInSeq);
					EvaluateThisNodeSRP(timeIdxInSeq, m_delay, reset, m_default_activity, m_functionValues, m_pastActivity, Inputs(0)->FunctionValues(), i, m_samplesInRecurrentStep);
				}
			}

        }

        static void WINAPI EvaluateThisNodeSR(const size_t timeIdxInSeq, const int delay, const bool reset, const ElemType default_activity, Matrix<ElemType>& functionValues, const Matrix<ElemType>& pastActivity, const Matrix<ElemType>& inputFunctionValues, const size_t mNbr)
        {
            ASSERT(delay > 0);

            if (functionValues.GetNumRows() != inputFunctionValues.GetNumRows() ||
                functionValues.GetNumCols() != inputFunctionValues.GetNumCols())
                functionValues.Resize(inputFunctionValues.GetNumRows(),
                    inputFunctionValues.GetNumCols());

            int iPastIndex = (int) (timeIdxInSeq - delay) * mNbr;
            int d = iPastIndex; 
            if (d < 0)
                d = (int)functionValues.Mod((float)iPastIndex, (float)pastActivity.GetNumCols());  
           /// this can point to the past activity of the previous mninibatch

            Matrix<ElemType> out = functionValues.ColumnSlice(timeIdxInSeq * mNbr, mNbr);
            Matrix<ElemType> inp((short)functionValues.GetDeviceId()) ;

            if (iPastIndex < 0 && reset)
                out.SetValue(default_activity);
            else
            {
                if (iPastIndex < 0)
                    inp = pastActivity.ColumnSlice(d, mNbr);
                else
                    inp = inputFunctionValues.ColumnSlice(d, mNbr);
                out.SetValue(inp);
            }
        }

        static void WINAPI EvaluateThisNodeSRP(const size_t timeIdxInSeq, const int delay, const bool reset, const ElemType default_activity, Matrix<ElemType>& functionValues, const Matrix<ElemType>& pastActivity, const Matrix<ElemType>& inputFunctionValues, const size_t indexInBatch, const size_t mNbr)
        {
            ASSERT(delay > 0);

            if (functionValues.GetNumRows() != inputFunctionValues.GetNumRows() ||
                functionValues.GetNumCols() != inputFunctionValues.GetNumCols())
                functionValues.Resize(inputFunctionValues.GetNumRows(),
                    inputFunctionValues.GetNumCols());

            int iPastIndex = (int) (timeIdxInSeq - delay) * mNbr;
            int d = iPastIndex; 
            if (d < 0)
                d = (int)functionValues.Mod((float)iPastIndex, (float)pastActivity.GetNumCols());  
            /// this can point to the past activity of the previous mninibatch

			Matrix<ElemType> out = functionValues.ColumnSlice(timeIdxInSeq * mNbr+indexInBatch, 1);
            Matrix<ElemType> inp((short)functionValues.GetDeviceId()) ;

            if (iPastIndex < 0 && reset)
                out.SetValue(default_activity);
            else
            {
                if (iPastIndex < 0)
					inp = pastActivity.ColumnSlice(d+indexInBatch, 1);
                else
                    inp = inputFunctionValues.ColumnSlice(d+indexInBatch, 1);
                out.SetValue(inp);
            }
        }



        virtual const Matrix<ElemType>& FunctionValues() const 
        {
            return m_functionValues;
        }

        virtual Matrix<ElemType>& FunctionValues() 
        {
            return m_functionValues;
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation(true/*allowNulls*/);

            if (m_children.size() != 1) 
                throw std::logic_error("Delay operation should have one input.");

			if (!(Inputs(0) == nullptr))
			{
				size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();

				if (rows0 > 0 && cols0 > 0) FunctionValues().Resize(rows0, cols0);
			}
            CopyImageSizeFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr inputNode) 
        {
            m_children.resize(1);
            m_children[0] = inputNode;
        }

        void SetDelay(const int val)
        {
            if (val <= 0)
                throw std::logic_error("Delay must be > 0.");
            m_delay = val;
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_functionValues.GetDeviceId() != deviceId)
                    m_functionValues.TransferFromDeviceToDevice(m_functionValues.GetDeviceId(), deviceId);
                if (m_pastActivity.GetDeviceId() != deviceId)
                    m_pastActivity.TransferFromDeviceToDevice(m_pastActivity.GetDeviceId(), deviceId, true);
            }
        }

        static const std::wstring TypeName() {return L"Delay";} 

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            DelayNode<ElemType>* node = (DelayNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_delay = m_delay;
                node->m_Reset = m_Reset;
                node->m_default_activity = m_default_activity;
                node->m_pastActivity = m_pastActivity;
            }
        }

        // copy constructor
        DelayNode(const DelayNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), m_pastActivity(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new DelayNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_pastActivity;  /// saves the past activity this delay node points to
        int      m_delay;    /// steps for delay 
        bool     m_Reset; 

    };

    template class DelayNode<float>; 
    template class DelayNode<double>;


#pragma endregion derived operations

}}}