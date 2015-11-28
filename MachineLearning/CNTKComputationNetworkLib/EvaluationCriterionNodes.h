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
    // ErrorPredictionNode (label, prediction)   or ErrorPredictionNode (prediction, label)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class ErrorPredictionNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>
    {
        typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"ErrorPrediction"; }
    public:
        DeclareConstructorFromConfig(ErrorPredictionNode);
        ErrorPredictionNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void ComputeInputPartialNonLooping(size_t /*inputIndex*/) override
        {
            LogicError("%ls operation is used for evaluation only.", OperationName().c_str());
        }

        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            FrameRange frameRange(Inputs(0)->GetMBLayout());
            Inputs(0)->ValueSlice(frameRange).VectorMax(*m_maxIndexes0, *m_maxValues, true);
            Inputs(1)->ValueSlice(frameRange).VectorMax(*m_maxIndexes1, *m_maxValues, true, m_topK);
            MaskMissingColumnsToZero(*m_maxIndexes0, Inputs(0)->GetMBLayout(), frameRange);
            MaskMissingColumnsToZero(*m_maxIndexes1, Inputs(1)->GetMBLayout(), frameRange);
            FunctionValues().AssignNumOfDiff(*m_maxIndexes0, *m_maxIndexes1, m_topK > 1);
        #if NANCHECK
            FunctionValues().HasNan("ErrorPrediction");
        #endif
#if DUMPOUTPUT
            FunctionValues().Print("ErrorPredictionNode");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
#if 1
            ValidateBinaryReduce(isFinalValidationPass);
#else
            Base::Validate(isFinalValidationPass);

            ValidateInferBinaryChildrenDims();

            if (isFinalValidationPass)
            {
                if (!(
                      Inputs(0)->GetNumRows() == Inputs(1)->GetNumRows() &&  //match size
                      (Inputs(0)->HasMBLayout() || Inputs(0)->GetNumCols() == Inputs(1)->GetNumCols())
                     ))
                {
                    LogicError("The Matrix dimension in the ErrorPrediction operation does not match.");
                }
            }
            Resize(1,1);
            m_pMBLayout = nullptr;    // this node does not hold mini-batch data
            InferImageDimsFromInputs(); 
#endif

            m_topK = 1;
            // TODO: Make topK a constructor parameter
            if (m_inputs.size() == 3)
            {
                if (Inputs(2)->GetNumRows() != 1 || Inputs(2)->GetNumCols() != 1)
                    throw std::logic_error("TopK in ErrorPredictionNode must be a scalar value.");
                m_topK = static_cast<int>(Inputs(2)->Get00Element());
            }

            // resize the temporaries to their proper size
            size_t cols = Inputs(0)->GetNumCols();
            m_maxIndexes0->Resize(m_topK, cols);
            m_maxIndexes1->Resize(m_topK, cols);
            m_maxValues->Resize(m_topK, cols);
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            m_imageLayout = ImageLayout();
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
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

    private:
        shared_ptr<Matrix<ElemType>> m_maxIndexes0, m_maxIndexes1;
        shared_ptr<Matrix<ElemType>> m_maxValues;
        int m_topK;
    };

    template class ErrorPredictionNode<float>; 
    template class ErrorPredictionNode<double>;

}}}
