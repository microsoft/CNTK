//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once
#include "LinearAlgebraNodes.h"
#ifdef USE_MKLDNN
#include "TimesEngine.h"
#endif
namespace Microsoft
{
namespace MSR
{
namespace CNTK
{
#ifdef USE_MKLDNN

// -----------------------------------------------------------------------
// TimesTransposeNode (A, B')
// Right operand and output can have MB layout, while left operand cannot.
// This differs from TimesNode in that A is transposed, where A must be a
// rank-1 or rank-2 tensor.
// A common use of transposition is trace(X'X) where X is a matrix of samples.
// This can be more efficiently implemented as ReduceSum (ElementTimes (X, X))
// -----------------------------------------------------------------------

template <class ElemType>
class TimesTransposeNode : public ComputationNode<ElemType>, public NumInputs<2>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;

    static const std::wstring TypeName()
    {
        return L"TimesTranspose";
    }

private:
    static const int NumInputs = 2;
    shared_ptr<Matrix<ElemType>> m_tempMklDnnIndices[NumInputs];
    std::unique_ptr<TimesEngine<ElemType>> m_timesEng;

protected:
    // if the left argument of the matrix product (A) has a time axis, it can only be applied sample by sample
    // where each sample is treated as a separate matrix object (as a consequence, it then also applies to B and the
    // result as well)
    TensorView<ElemType> OneSampleTensorFor(int inputIndex /*-1 for output*/, bool gradient /*instead of value*/, const FrameRange& fr)
    {
        auto input = inputIndex < 0 ? this : Input(inputIndex).get();
        auto data = gradient ? input->GradientPtr() : input->ValuePtr();
        size_t rank = input->GetSampleLayout().GetRank();

        if (!InputRef(0).HasMBLayout()) // left input is no MB data: run normally
            return input->DataTensorFor(data, rank, fr);
        auto tensorShape = input->GetTensorSliceFor(rank, fr);
        return TensorView<ElemType>(data, tensorShape);
    }

public:
    void RequestReduceSequenceAxisMatricesIfNeeded(MatrixPool& matrixPool)
    {
        for (int i = 0; i < NumInputs; i++)
        {
            RequestMatrixFromPool(m_tempMklDnnIndices[i], matrixPool, 0, false, true);
        }
    }

    void ReleaseReduceSequenceAxisMatricesIfNeeded(MatrixPool& matrixPool)
    {
        for (int i = 0; i < NumInputs; i++)
        {
            ReleaseMatrixToPool(m_tempMklDnnIndices[i], matrixPool);
        }
    }

    void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestReduceSequenceAxisMatricesIfNeeded(matrixPool);
    }

    void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterForwardProp(matrixPool);
        ReleaseReduceSequenceAxisMatricesIfNeeded(matrixPool);
    }

    void RequestMatricesBeforeBackprop(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestReduceSequenceAxisMatricesIfNeeded(matrixPool);
    }

    void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseReduceSequenceAxisMatricesIfNeeded(matrixPool);
    }
    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        // TensorView::DoMatrixProductOf() will reduce each tensor object into a 2D tensor (or fail if it cannot)
        // and recreate actual Matrix objects (in case of sparse, they must be identical to the original tensor storage
        // object). Transposition is applied after flattening into 2D, but only allowed if the input sample is 2D
        // anyway.
        auto input0 = OneSampleTensorFor(0, /*gradient=*/false, fr.AllowBroadcast());
        auto input1 = OneSampleTensorFor(1, /*gradient=*/false, fr.AllowBroadcast());
        auto output = OneSampleTensorFor(-1, /*gradient=*/false, fr);
        if (m_timesEng != nullptr)
        {
            m_timesEng->Forward(input0.GetSOB(), input1.GetSOB(), output.GetSOB());
        }
        else
        {
            output.AssignMatrixProductOf(false /*transC*/, input0, false /*transA*/, input1, true /*transB*/, 1.0f, nullptr);
        }
    }
    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        bool overwriteInputGradient = (Input(inputIndex)->IsGradientInitializedBy(this));

        if (inputIndex == 1) // right derivative
        {
            auto input0 = OneSampleTensorFor(0, /*gradient=*/false, fr.AllowBroadcast());
            auto input1Gradient = OneSampleTensorFor(1, /*gradient=*/true, fr.AllowBroadcast());
            auto outputGradient = OneSampleTensorFor(-1, /*gradient=*/true, fr);
            if (m_timesEng != nullptr)
            {
                m_timesEng->BackwardWeight(input0.GetSOB(), outputGradient.GetSOB(), input1Gradient.GetSOB(),
                                           !overwriteInputGradient, *m_tempMklDnnIndices[inputIndex]);
            }
            else
            {
                if (overwriteInputGradient)
                    input1Gradient.AssignMatrixProductOf(true /*transC*/, input0, false /*transA*/, outputGradient,
                                                         false /*transB*/);
                else
                    input1Gradient.AddMatrixProductOf(true /*transC*/, input0, false /*transA*/, outputGradient,
                                                      false /*transB*/);
            }
        }
        else if (inputIndex == 0) // left derivative
        {
            auto input0Gradient = OneSampleTensorFor(0, /*gradient=*/true, fr.AllowBroadcast());
            auto input1 = OneSampleTensorFor(1, /*gradient=*/false, fr.AllowBroadcast());
            auto outputGradient = OneSampleTensorFor(-1, /*gradient=*/true, fr);
            if (m_timesEng)
            {
                m_timesEng->BackwardData(outputGradient.GetSOB(), input1.GetSOB(), input0Gradient.GetSOB(),
                                         !overwriteInputGradient, *m_tempMklDnnIndices[inputIndex]);
            }
            else
            {
                if (overwriteInputGradient)
                    input0Gradient.AssignMatrixProductOf(false /*transC*/, outputGradient, false /*transA*/, input1,
                                                         false /*transB*/);
                else
                    input0Gradient.AddMatrixProductOf(false /*transC*/, outputGradient, false /*transA*/, input1,
                                                      false /*transB*/);
            }
        }
    }

    DeclareConstructorFromConfigWithNumInputs(TimesTransposeNode);
    TimesTransposeNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name) {}
    virtual ParentGradientOptimization ImplementsGradientOptimization(const ComputationNodeBase*) const override
    {
        return ParentGradientOptimization::Overwrite;
    }
    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);
        // get tensor shapes, matrix is matrixFormatColMajor by default
        // Data Batch * NUM_INPUT
        auto dimsALayout = Input(0)->GetSampleLayout();
        auto dimsA = dimsALayout.GetDims();

        // Weight NUM_HINDDEN * NUM_input
        auto dimsB = Input(1)->GetSampleLayout().GetDims();
        string dimsAstring = string(Input(0)->GetSampleLayout()); // for error messages
        string dimsBstring = string(Input(1)->GetSampleLayout());

        auto dimsC = dimsB;
        // Output
        if (dimsC.size() > 1)
        {
            dimsC.resize(1);
            dimsC[0] = dimsB[dimsB.size() - 1]; // NUM_HIDDEN, (Batch size assign later)
            dimsB.resize(2);
            dimsB[0] = dimsALayout.GetNumElements();
            dimsB[1] = dimsC[0];
        }
        SetDims(TensorShape(dimsC), HasMBLayout());

        // update if LearnableParameter
        Input(1)->ValidateInferInputDimsFrom(TensorShape(dimsB));
        if (m_timesEng == nullptr && m_deviceId == CPUDEVICE)
        {
            m_timesEng = TimesEngine<ElemType>::Create(m_deviceId, dimsA, dimsB, TimesKind::All);
        }
    }
};

template class TimesTransposeNode<float>;
template class TimesTransposeNode<double>;
template class TimesTransposeNode<half>;
#endif

} // namespace CNTK
} // namespace MSR
} // namespace Microsoft
