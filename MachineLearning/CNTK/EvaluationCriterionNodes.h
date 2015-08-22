//
// <copyright file="EvaluationCriterionNodes.h" company="Microsoft">
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
    class ErrorPredictionNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        void Construct(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") 
        {
            m_maxIndexes0 = Matrix<ElemType>(deviceId), m_maxIndexes1 = Matrix<ElemType>(deviceId), m_maxValues = Matrix<ElemType>(deviceId);
            ComputationNode<ElemType>::Construct(deviceId, name);
            // further initializations
            MoveMatricesToDevice(deviceId); // TODO: does more than constructor
        }

        void Construct(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
        {
            m_maxIndexes0 = Matrix<ElemType>(deviceId), m_maxIndexes1 = Matrix<ElemType>(deviceId), m_maxValues = Matrix<ElemType>(deviceId);
            ComputationNode<ElemType>::Construct(deviceId, name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() {return L"ErrorPrediction";} 

        void Reset()
        {
        }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/)  //scaled by 2*number of elements in the Matrix<ElemType>
        {
            throw std::logic_error("ErrorPrediction is used for evaluation only.");
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_maxIndexes0, m_maxIndexes1, m_maxValues, shared_from_this());
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues0, const Matrix<ElemType>& inputFunctionValues1, Matrix<ElemType>& maxIndexes0, Matrix<ElemType>& maxIndexes1, Matrix<ElemType>& maxValues, ComputationNodePtr curNode)
        {
            inputFunctionValues0.VectorMax(maxIndexes0, maxValues, true);
            inputFunctionValues1.VectorMax(maxIndexes1, maxValues, true);
            curNode->MaskToZeroWhenLabelAndFeatureMissing(maxIndexes0); //we are fine since it will only be called with full minibatch
            curNode->MaskToZeroWhenLabelAndFeatureMissing(maxIndexes1);
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

            if (Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements())
                throw std::logic_error("ErrorPrediction operation: one of the operants has 0 element.");

            if (((!(Inputs(0)->FunctionValues().GetNumRows() == Inputs(1)->FunctionValues().GetNumRows()  &&  //match size
                Inputs(0)->FunctionValues().GetNumCols() == Inputs(1)->FunctionValues().GetNumCols()) )) && Inputs(0)->LoopId() < 0)
            {
                throw std::logic_error("The Matrix dimension in the ErrorPrediction operation does not match.");
            }       

            FunctionValues().Resize(1,1);
            InferImageDimsFromInputs(); 

            // resize the temporaries to their proper size
            size_t cols = Inputs(0)->FunctionValues().GetNumCols();
            m_maxIndexes0.Resize(1,cols);
            m_maxIndexes1.Resize(1,cols);
            m_maxValues.Resize(1,cols);
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

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

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);
            m_maxIndexes0.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId, true);
            m_maxIndexes1.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId, true);
            m_maxValues.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId, true);
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            auto node = dynamic_pointer_cast<ErrorPredictionNode<ElemType>>(nodeP);

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_maxIndexes0 = m_maxIndexes0;
                node->m_maxIndexes1 = m_maxIndexes1;
                node->m_maxValues = m_maxValues;
            }
        }

        // copy constructor
        void Construct(const ErrorPredictionNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
        {
            m_maxIndexes0 = Matrix<ElemType>(node->m_deviceId), m_maxIndexes1 = Matrix<ElemType>(node->m_deviceId), m_maxValues = Matrix<ElemType>(node->m_deviceId);
            ComputationNode<ElemType>::Construct(node->m_deviceId, newName);
            node->CopyTo(shared_from_this(), newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
            return New<ErrorPredictionNode<ElemType>>(this, name, flags);
        }

    protected:
        virtual bool UseCustomizedMultiSeqHandling() { return true; }

    private:
        Matrix<ElemType> m_maxIndexes0, m_maxIndexes1;
        Matrix<ElemType> m_maxValues;
    };

    template class ErrorPredictionNode<float>; 
    template class ErrorPredictionNode<double>;

}}}
