//
// <copyright file="EvaluationCriterionNode.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include "ComputationNode.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    //note: to save computation the gradient may be scaled by an constant. 

    template<class ElemType>
    class ErrorPredictionNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 

    public:
        ErrorPredictionNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") 
            : ComputationNode(deviceId), m_maxIndexes0(deviceId), m_maxIndexes1(deviceId), m_maxValues(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        ErrorPredictionNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_maxIndexes0(deviceId), m_maxIndexes1(deviceId), m_maxValues(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }
                
        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"ErrorPrediction";} 

        void Reset()
        {
        }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/)  //scaled by 2*number of elements in the Matrix<ElemType>
        {
            throw std::logic_error("ErrorPrediction is used for evaluation only.");
        }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/)
        {
            throw std::logic_error("ErrorPrediction is used for evaluation only.");
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->MatrixParam(m_maxIndexes0, "maxIndexes0", paramOptionsInput | paramOptionsTemporary);
                descriptor->MatrixParam(m_maxIndexes1, "maxIndexes1", paramOptionsInput | paramOptionsTemporary);
                descriptor->MatrixParam(m_maxValues, "maxValues", paramOptionsInput | paramOptionsTemporary);
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
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_maxIndexes0, m_maxIndexes1, m_maxValues);
        }

        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/)
        {
            throw std::logic_error("ErrorPrediction node should never be in a loop.");
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues0, const Matrix<ElemType>& inputFunctionValues1, Matrix<ElemType>& maxIndexes0, Matrix<ElemType>& maxIndexes1, Matrix<ElemType>& maxValues)  
        {
            inputFunctionValues0.VectorMax(maxIndexes0, maxValues, true);
            inputFunctionValues1.VectorMax(maxIndexes1, maxValues, true);
            functionValues.AssignNumOfDiff(maxIndexes0, maxIndexes1);
        #if NANCHECK
            functionValues.HasNan("ErrorPrediction");
        #endif
#if DUMPOUTPUT
            functionValues.Print("ErrorPredictionNode");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("ErrorPrediction operation requires two inputs.");

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
                m_maxIndexes0.Resize(1,cols);
                m_maxIndexes1.Resize(1,cols);
                m_maxValues.Resize(1,cols);
            }

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("ErrorPrediction operation: one of the operants has 0 element.");

            if (((!(Inputs(0)->FunctionValues().GetNumRows() == Inputs(1)->FunctionValues().GetNumRows()  &&  //match size
                Inputs(0)->FunctionValues().GetNumCols() == Inputs(1)->FunctionValues().GetNumCols()) )) && Inputs(0)->LoopId() < 0)
            {
                throw std::logic_error("The Matrix dimension in the ErrorPrediction operation does not match.");
            }       

            FunctionValues().Resize(1,1);
            CopyImageSizeFromInputs(); 

            // resize the temporaries to their proper size
            size_t cols = Inputs(0)->FunctionValues().GetNumCols();
            m_maxIndexes0.Resize(1,cols);
            m_maxIndexes1.Resize(1,cols);
            m_maxValues.Resize(1,cols);
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
                if (m_maxIndexes0.GetDeviceId() != deviceId)
                    m_maxIndexes0.TransferFromDeviceToDevice(m_maxIndexes0.GetDeviceId(), deviceId,true);

                if (m_maxIndexes1.GetDeviceId() != deviceId)
                    m_maxIndexes1.TransferFromDeviceToDevice(m_maxIndexes1.GetDeviceId(), deviceId,true);

                if (m_maxValues.GetDeviceId() != deviceId)
                    m_maxValues.TransferFromDeviceToDevice(m_maxValues.GetDeviceId(), deviceId,true);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            ErrorPredictionNode<ElemType>* node = (ErrorPredictionNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_maxIndexes0 = m_maxIndexes0;
                node->m_maxIndexes1 = m_maxIndexes1;
                node->m_maxValues = m_maxValues;
            }
        }

        // copy constructor
        ErrorPredictionNode(const ErrorPredictionNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) 
            : ComputationNode(node->m_deviceId), m_maxIndexes0(node->m_deviceId), m_maxIndexes1(node->m_deviceId), m_maxValues(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new ErrorPredictionNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_maxIndexes0, m_maxIndexes1;
        Matrix<ElemType> m_maxValues;
    };

    template class ErrorPredictionNode<float>; 
    template class ErrorPredictionNode<double>;

}}}