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

    // -----------------------------------------------------------------------
    // ErrorPredictionNode (label, prediction)    --TODO: is that correct?
    // -----------------------------------------------------------------------

    template<class ElemType>
    class ErrorPredictionNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>
    {
        typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"ErrorPrediction"; }
    public:
        ErrorPredictionNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        void Reset()        // TODO: what is this??
        {
        }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/)  //scaled by 2*number of elements in the Matrix<ElemType>
        {
            LogicError("ErrorPrediction is used for evaluation only.");
        }

        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), *m_maxIndexes0, *m_maxIndexes1, *m_maxValues, m_topK, shared_from_this());
        }

        void EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues0, const Matrix<ElemType>& inputFunctionValues1, Matrix<ElemType>& maxIndexes0, Matrix<ElemType>& maxIndexes1, Matrix<ElemType>& maxValues, int topK, ComputationNodePtr curNode)
        {
            inputFunctionValues0.VectorMax(maxIndexes0, maxValues, true);
            inputFunctionValues1.VectorMax(maxIndexes1, maxValues, true, topK);
            curNode->MaskMissingColumnsToZero(maxIndexes0, Inputs(0)->GetMBLayout());   // we are fine since it will only be called with full minibatch
            curNode->MaskMissingColumnsToZero(maxIndexes1, Inputs(1)->GetMBLayout());
            functionValues.AssignNumOfDiff(maxIndexes0, maxIndexes1, topK > 1);
        #if NANCHECK
            functionValues.HasNan("ErrorPrediction");
        #endif
#if DUMPOUTPUT
            functionValues.Print("ErrorPredictionNode");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            size_t index = 0;
            {
                size_t rows = Inputs(index)->GetNumRows() == 0? Inputs(1-index)->GetNumRows() : Inputs(index)->GetNumRows();
                size_t cols = Inputs(index)->GetNumCols() == 0? Inputs(1-index)->GetNumCols() : Inputs(index)->GetNumCols();
                ValidateInferChildDims(index, rows, cols);
            }

            index = 1;
            {
                size_t rows = Inputs(index)->GetNumRows() == 0? Inputs(1-index)->GetNumRows() : Inputs(index)->GetNumRows();
                size_t cols = Inputs(index)->GetNumCols() == 0? Inputs(1-index)->GetNumCols() : Inputs(index)->GetNumCols();
                ValidateInferChildDims(index, rows, cols);
            }

            m_topK = 1;
            if (m_children.size() == 3)
            {
                if (Inputs(2)->FunctionValues().GetNumRows() != 1 || Inputs(2)->FunctionValues().GetNumCols() != 1)
                    throw std::logic_error("TopK in ErrorPredictionNode must be a scalar value.");
                m_topK = static_cast<int>(Inputs(2)->FunctionValues().Get00Element());
            }
            //if (Inputs(0)->GetNumRows() == 0 || Inputs(1)->GetNumRows() == 0)
            //    LogicError("ErrorPrediction operation: one of the operands has 0 elements.");

            if (isFinalValidationPass)
            {
                if (!(Inputs(0)->GetNumRows() == Inputs(1)->GetNumRows() && Inputs(0)->GetNumCols() == Inputs(1)->GetNumCols()))
                {
                    LogicError("The Matrix dimension in the ErrorPrediction operation does not match.");
                }       

                if (((!(Inputs(0)->FunctionValues().GetNumRows() == Inputs(1)->FunctionValues().GetNumRows() &&  //match size
                    Inputs(0)->FunctionValues().GetNumCols() == Inputs(1)->FunctionValues().GetNumCols()))) && Inputs(0)->GetLoopId() < 0)
                {
                    LogicError("The Matrix dimension in the ErrorPrediction operation does not match.");
                }
            }
            Resize(1,1);
            m_pMBLayout = nullptr;    // this node does not hold mini-batch data
            InferImageDimsFromInputs(); 

            // resize the temporaries to their proper size
            size_t cols = Inputs(0)->GetNumCols();
            m_maxIndexes0->Resize(m_topK, cols);
            m_maxIndexes1->Resize(m_topK, cols);
            m_maxValues->Resize(m_topK, cols);
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            m_outputImageLayout = ImageLayout();
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            Base::MoveMatricesToDevice(deviceId);
            m_maxIndexes0->TransferToDeviceIfNotThereAndNotAutoPlace(deviceId, true);
            m_maxIndexes1->TransferToDeviceIfNotThereAndNotAutoPlace(deviceId, true);
            m_maxValues->TransferToDeviceIfNotThereAndNotAutoPlace(deviceId, true);
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<ErrorPredictionNode<ElemType>>(nodeP);
                *node->m_maxIndexes0 = *m_maxIndexes0;
                *node->m_maxIndexes1 = *m_maxIndexes1;
                *node->m_maxValues = *m_maxValues;
            }
        }
        //request matrices needed to do node function value evaluation
        virtual void RequestMatricesBeforeEval(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeEval(matrixPool);
            RequestMatrixFromPool(m_maxIndexes0, matrixPool);
            RequestMatrixFromPool(m_maxIndexes1, matrixPool);
            RequestMatrixFromPool(m_maxValues, matrixPool);
        }

        //release temp matrices that are only used by forward computation
        //don't release matrices that need to be used in the gradient computation
        virtual void ReleaseMatricesAfterEval(MatrixPool& matrixPool)
        {
            Base::ReleaseMatricesAfterEval(matrixPool);
            ReleaseMatrixToPool(m_maxIndexes0, matrixPool);
            ReleaseMatrixToPool(m_maxIndexes1, matrixPool);
            ReleaseMatrixToPool(m_maxValues, matrixPool);
        }
protected:
        virtual bool NodeDoesItsOwnCustomizedMissingColumnsMasking() { return true; }

    private:
        shared_ptr<Matrix<ElemType>> m_maxIndexes0, m_maxIndexes1;
        shared_ptr<Matrix<ElemType>> m_maxValues;
        int m_topK;
    };

    template class ErrorPredictionNode<float>; 
    template class ErrorPredictionNode<double>;

}}}
