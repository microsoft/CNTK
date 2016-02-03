//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "ConvolutionEngine.h"

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
/// SquareErrorNode (left, right)
// -----------------------------------------------------------------------

//note: to save computation the gradient may be scaled by an constant.

template <class ElemType>
class SquareErrorNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<2>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"SquareError";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(SquareErrorNode);
    SquareErrorNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        FrameRange fr(Input(0)->GetMBLayout());
        auto gradient = Input(inputIndex)->GradientFor(fr);
        Matrix<ElemType>::Multiply1x1AndWeightedAdd(inputIndex == 0 ? 1.0f : -1.0f, Gradient() /*1x1*/, *m_leftMinusRight, 1.0f, gradient);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override
    {
        return false;
    }

    virtual void UpdateFunctionMBSize() override
    {
        m_leftMinusRight->Resize(Input(0)->Value());
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(Input(0)->GetMBLayout());
        m_leftMinusRight->AssignDifferenceOf(Input(0)->ValueFor(fr), Input(1)->ValueFor(fr));
        MaskMissingColumnsToZero(*m_leftMinusRight, Input(0)->GetMBLayout(), fr); // we are fine since it will only be called with full minibatch.
        ElemType v = m_leftMinusRight->FrobeniusNorm();
        Value().VerifySize(1, 1);
        Value().SetValue(v * v / 2);
#if NANCHECK
        Value().HasNan("SquareError");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateBinaryReduce(isFinalValidationPass);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<SquareErrorNode<ElemType>>(nodeP);
            *node->m_leftMinusRight = *m_leftMinusRight;
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_leftMinusRight, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_leftMinusRight, matrixPool);
    }

private:
    shared_ptr<Matrix<ElemType>> m_leftMinusRight;
};

template class SquareErrorNode<float>;
template class SquareErrorNode<double>;

// -----------------------------------------------------------------------
// CrossEntropyWithSoftmaxNode (labels, prediction)
// calculates: -sum(left_i * log(softmax_i(right)))
// -----------------------------------------------------------------------

template <class ElemType>
class CrossEntropyWithSoftmaxNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<2>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"CrossEntropyWithSoftmax";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(CrossEntropyWithSoftmaxNode);
    CrossEntropyWithSoftmaxNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        FrameRange fr(Input(0)->GetMBLayout());
        // left input is scalar
        if (inputIndex == 0) // left derivative
        {
#if DUMPOUTPUT
            *m_logSoftmaxOfRight.Print("CrossEntropyWithSoftmax Partial-logSoftmaxOfRight");
            Gradient().Print("CrossEntropyWithSoftmax Partial-gradientValues");
            Input(0)->GradientFor(fr).Print("CrossEntropyWithSoftmaxNode Partial-Left-in");
#endif

            auto gradient = Input(0)->GradientFor(fr);
            Matrix<ElemType>::Multiply1x1AndWeightedAdd(-1.0f, Gradient() /*1x1*/, *m_logSoftmaxOfRight, 1.0f, gradient);
#if DUMPOUTPUT
            Input(0)->GradientFor(fr).Print("CrossEntropyWithSoftmaxNode Partial-Left-out");
#endif
        }

        else if (inputIndex == 1) // right derivative
        {
#if DUMPOUTPUT
            *m_softmaxOfRight.Print("CrossEntropyWithSoftmax Partial-softmaxOfRight");
            Input(0)->ValueFor(fr).Print("CrossEntropyWithSoftmax Partial-inputFunctionValues");
            Gradient().Print("CrossEntropyWithSoftmax Partial-gradientValues");
            Input(1)->GradientFor(fr).Print("CrossEntropyWithSoftmaxNode Partial-Right-in");
#endif

            auto gradient = Input(1)->GradientFor(fr);
            Matrix<ElemType>::AddScaledDifference(Gradient(), *m_softmaxOfRight, Input(0)->ValueFor(fr), gradient);
#if DUMPOUTPUT
            Input(1)->GradientFor(fr).Print("CrossEntropyWithSoftmaxNode Partial-Right");
#endif
#ifdef _DEBUG
            Input(1)->InvalidateMissingGradientColumns(fr); // TODO: This should not be necessary.
#endif
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    virtual void UpdateFunctionMBSize() override
    {
        m_logSoftmaxOfRight->Resize(Input(1)->Value());
        m_softmaxOfRight->Resize(*m_logSoftmaxOfRight);
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override // -sum(left_i * log(softmax_i(right)))
    {
        FrameRange fr(Input(0)->GetMBLayout());
        // first compute the softmax (column-wise)
        // Note that we need both log and non-log for gradient computation.
        m_logSoftmaxOfRight->AssignLogSoftmaxOf(Input(1)->ValueFor(fr), true);
        m_softmaxOfRight->SetValue(*m_logSoftmaxOfRight);
        m_softmaxOfRight->InplaceExp();
        // flatten all gaps to zero, such that gaps will contribute zero to the sum
        MaskMissingColumnsToZero(*m_logSoftmaxOfRight, Input(1)->GetMBLayout(), fr);
        // reduce over all frames
        Value().AssignInnerProductOfMatrices(Input(0)->MaskedValueFor(fr), *m_logSoftmaxOfRight);
        Value() *= -1;
#if NANCHECK
        Value().HasNan("CrossEntropyWithSoftmax");
#endif
#if DUMPOUTPUT
        Value().Print("CrossEntropyWithSoftmaxNode");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateBinaryReduce(isFinalValidationPass);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<CrossEntropyWithSoftmaxNode<ElemType>>(nodeP);
            *node->m_logSoftmaxOfRight = *m_logSoftmaxOfRight;
            *node->m_softmaxOfRight = *m_softmaxOfRight;
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_logSoftmaxOfRight, matrixPool);
        RequestMatrixFromPool(m_softmaxOfRight, matrixPool);
    }

protected:
    shared_ptr<Matrix<ElemType>> m_logSoftmaxOfRight;
    shared_ptr<Matrix<ElemType>> m_softmaxOfRight;
};

template class CrossEntropyWithSoftmaxNode<float>;
template class CrossEntropyWithSoftmaxNode<double>;

// -----------------------------------------------------------------------
/// CrossEntropyNode (labels, prediction)
// -----------------------------------------------------------------------

// calculates: -sum(left_i * log(right_i))
// assume softmax is already done
// You probably want to use CrossEntropyWithSoftMaxNode instead, it is more efficient in most cases.
template <class ElemType>
class CrossEntropyNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<2>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"CrossEntropy";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(CrossEntropyNode);
    CrossEntropyNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        FrameRange fr(Input(0)->GetMBLayout());
        // left Node must be a scalar
        if (inputIndex == 0) // left derivative
        {
            BackpropToLeft(*m_logOfRight, Input(0)->GradientFor(fr), Gradient());
        }
        else
        {
            BackpropToRight(*m_leftDivRight, Input(0)->ValueFor(fr), Input(1)->ValueFor(fr), Input(1)->GradientFor(fr), Gradient());
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    /*TODO: merge with call site*/ void BackpropToLeft(const Matrix<ElemType>& logOfRight, Matrix<ElemType> inputGradientValues,
                                                       const Matrix<ElemType>& gradientValues)
    {
        Matrix<ElemType>::Multiply1x1AndWeightedAdd(-1.0f, gradientValues /*1x1*/, logOfRight, 1.0f, inputGradientValues);
    }

    /*TODO: merge with call site*/ void BackpropToRight(Matrix<ElemType>& leftDivRight,
                                                        const Matrix<ElemType> inputFunctionValues0, const Matrix<ElemType> inputFunctionValues1,
                                                        Matrix<ElemType> inputGradientValues, const Matrix<ElemType>& gradientValues)
    {
        FrameRange fr(Input(0)->GetMBLayout());
        leftDivRight.AssignElementDivisionOf(inputFunctionValues0, inputFunctionValues1);
        MaskMissingColumnsToZero(leftDivRight, Input(0)->GetMBLayout(), fr);
        Matrix<ElemType>::Multiply1x1AndWeightedAdd(-1.0f, gradientValues /*1x1*/, leftDivRight, 1.0f, inputGradientValues);
    }

    virtual void UpdateFunctionMBSize() override
    {
        m_logOfRight->Resize(Input(1)->Value());
        m_leftDivRight->Resize(Input(1)->Value());
    }

    // -sum(left_i * log(right_i))
    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(Input(0)->GetMBLayout());
        m_logOfRight->SetValue(Input(1)->ValueFor(fr));
        m_logOfRight->InplaceLog();
        MaskMissingColumnsToZero(*m_logOfRight, Input(1)->GetMBLayout(), fr);
        Value().AssignInnerProductOfMatrices(Input(0)->MaskedValueFor(fr), *m_logOfRight);
        Value() *= -1;
#if NANCHECK
        functionValues.HasNan("CrossEntropy");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateBinaryReduce(isFinalValidationPass);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<CrossEntropyNode<ElemType>>(nodeP);
            *node->m_logOfRight = *m_logOfRight;
            *node->m_leftDivRight = *m_leftDivRight;
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_logOfRight, matrixPool);
    }

    // request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_leftDivRight, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_logOfRight, matrixPool);
        ReleaseMatrixToPool(m_leftDivRight, matrixPool);
    }

private:
    // matrix value passed from evaluate to computePartial
    shared_ptr<Matrix<ElemType>> m_logOfRight;
    // temporary
    shared_ptr<Matrix<ElemType>> m_leftDivRight;
};

template class CrossEntropyNode<float>;
template class CrossEntropyNode<double>;

// -----------------------------------------------------------------------
// MatrixL1RegNode (input)
// TODO: share most code with MatrixL2RegNode
// -----------------------------------------------------------------------

template <class ElemType>
class MatrixL1RegNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<1>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"MatrixL1Reg";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(MatrixL1RegNode);
    MatrixL1RegNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override // scale by number of cols (or samples)
    {
        FrameRange fr(Input(0)->GetMBLayout());
        assert(inputIndex == 0);
        inputIndex;
        BackpropToS(*m_gradientOfL1Norm, Input(0)->GradientFor(fr), Gradient(), Input(0)->ValueFor(fr));
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    /*TODO: merge with call site*/ void BackpropToS(Matrix<ElemType>& gradientOfL1Norm,
                                                    Matrix<ElemType> inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& inputFunctionValues)
    {
        gradientOfL1Norm.AssignSignOf(inputFunctionValues);
        Matrix<ElemType>::Multiply1x1AndWeightedAdd(+1.0f, gradientValues /*1x1*/, gradientOfL1Norm, 1.0f, inputGradientValues);
    }

    virtual void UpdateFunctionMBSize() override
    {
        m_gradientOfL1Norm->Resize(Input(0)->Value());
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(Input(0)->GetMBLayout());
        Value().VerifySize(1, 1);
        Value().SetValue(Input(0)->MaskedValueFor(fr).MatrixNorm1());
#if NANCHECK
        Value().HasNan("MatrixL1Reg");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateUnaryReduce(isFinalValidationPass);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<MatrixL1RegNode<ElemType>>(nodeP);
            *node->m_gradientOfL1Norm = *m_gradientOfL1Norm;
        }
    }

    // request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_gradientOfL1Norm, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_gradientOfL1Norm, matrixPool);
    }

private:
    shared_ptr<Matrix<ElemType>> m_gradientOfL1Norm; // temporary
};

template class MatrixL1RegNode<float>;
template class MatrixL1RegNode<double>;

// -----------------------------------------------------------------------
// MatrixL2RegNode (input)
// TODO: share most code with MatrixL1RegNode
// -----------------------------------------------------------------------

template <class ElemType>
class MatrixL2RegNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<1>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"MatrixL2Reg";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(MatrixL2RegNode);
    MatrixL2RegNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override // scale by number of cols (or samples)
    {
        FrameRange fr(Input(0)->GetMBLayout());
        assert(inputIndex == 0);
        inputIndex;
        BackpropToS(Input(0)->GradientFor(fr), Gradient(), Input(0)->ValueFor(fr), Value());
    }

    /*TODO: merge with call site*/ void BackpropToS(Matrix<ElemType> inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& functionValues)
    {
        ElemType v = gradientValues.Get00Element() / (functionValues.Get00Element() + EPS_IN_INVERSE); // TODO: GPU inefficiency
        inputGradientValues.AddWithScaleOf(v, inputFunctionValues);
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(Input(0)->GetMBLayout());
        Value().VerifySize(1, 1);
        Value().SetValue(Input(0)->MaskedValueFor(fr).FrobeniusNorm());
#if NANCHECK
        Value().HasNan("MatrixL2Reg");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateUnaryReduce(isFinalValidationPass);
    }
};

template class MatrixL2RegNode<float>;
template class MatrixL2RegNode<double>;

// -----------------------------------------------------------------------
// NoiseContrastiveEstimationNode (labels, input, inputWeights, biasWeights)
//  -labels: label in dense matrix in [4 x T]
//           the first row is the word index, the second row is the class index, the third row is the first word index of the class
//           the last row is the first word index of the next class
//  - input: hidden layer activity to the node in [hdsize x T]. for a simple rnn, this is the hidden layer activty
//  - inputWeights: weight matrix in [hdsize x vocab_size], for speed-up, as per word matrix can be simply obtained as column slice
//  - biasWeights: clsprob in dense matrix in [nbr_cls x T]. this is the output from logsoftmax node for the log-posterior probabilty of class given observations
// */
// BUGBUG: This node has not been converted to memshare conventions.
// -----------------------------------------------------------------------

enum NCEEvalMode
{
    Softmax = 0,
    Unnormalized = 1,
    None = 2
};
template <class ElemType>
class NoiseContrastiveEstimationNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<4>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"NCEBasedCrossEntropyWithSoftmax";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(NoiseContrastiveEstimationNode);
    NoiseContrastiveEstimationNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
          m_logSoftmax(deviceId),
          m_softMax(deviceId),
          m_grdToSoftMaxInput(deviceId),
          m_ncePrediction(deviceId),
          m_evalMode(NCEEvalMode::None)
    {
    }
    NoiseContrastiveEstimationNode(DEVICEID_TYPE deviceId, const wstring& name, NCEEvalMode xm_evalMode)
        : Base(deviceId, name),
          m_logSoftmax(deviceId),
          m_softMax(deviceId),
          m_grdToSoftMaxInput(deviceId),
          m_ncePrediction(deviceId),
          m_evalMode(xm_evalMode)
    {
    }
    // ^^ TODO: we can merge these two

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_evalMode;
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_evalMode;
        if (m_evalMode > NCEEvalMode::None)
        {
            m_evalMode = NCEEvalMode::None;
            fstream.SetPosition(fstream.GetPosition() - sizeof(m_evalMode));
        }
    }

    void SetEvalMode(NCEEvalMode& xevMode)
    {
        m_evalMode = xevMode;
    }
    NCEEvalMode& EvalMode()
    {
        return m_evalMode;
    } // TODO: really? Return a reference to a local? TODO: change to const? and call it GetEvalMode()

    /**
        compute gradients to input observations, the weights to the observations, and the class log posterior probabilities
        */
    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        FrameRange fr(Input(0)->GetMBLayout());
        m_needRecomputeGradientToSoftmaxInput = false;
        // gradient computation@yinggongzhao
        // inputIndex should be 2 this time
        if (m_evalMode != NCEEvalMode::None)
            LogicError("BackpropTo should only be called in training mode");
        if (inputIndex == 0)
            InvalidArgument("ComputeInput partial should not be called for label");
        //                                                                              samples+probs                   hidden                  embedding
        // Input(inputIndex)->GradientFor(fr).AssignNCEDerivative(m_ncePrediction, Input(0)->ValueFor(fr), Input(1)->ValueFor(fr), Input(2)->Value(), inputIndex);
        if (inputIndex >= 2)
            Input(inputIndex)->Gradient().AssignNCEDerivative(m_ncePrediction, Input(0)->ValueFor(fr), Input(1)->ValueFor(fr), Input(2)->ValueAsMatrix(), inputIndex);
        else
            Input(inputIndex)->GradientFor(fr).AssignNCEDerivative(m_ncePrediction, Input(0)->ValueFor(fr), Input(1)->ValueFor(fr), Input(2)->ValueAsMatrix(), inputIndex);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    virtual void UpdateFunctionMBSize() override
    {
        // TODO (this does not really break it since for full matrices, class Matrix will resize by itself)
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override // -sum(left_i * log(softmax_i(right)))
    {
        FrameRange fr(Input(0)->GetMBLayout());
        if (Input(0)->HasMBLayout() && Input(0)->GetMBLayout()->HasGaps())
            LogicError("%ls %ls operation does not handle multiple parallel sequences with gaps correctly. Contact fseide@microsoft.com if you have a need and a test case.", NodeName().c_str(), OperationName().c_str());

        int positive = 0, negative = 0;
        if (Input(0)->GetSampleLayout().GetNumElements() == 1)
        {
            for (int i = 0; i < Input(0)->Value().GetNumCols(); i++) // BUGBUG: Loops must be over frames, not columns. Columns may contain gaps.
            {
                if (Input(0)->Value()(0, i) > 0)
                    positive++;
                else if (Input(0)->Value()(0, i) < 0)
                    negative++;
            }
            assert(positive * negative == 0);
        }
        if (m_evalMode == NCEEvalMode::Softmax || (Input(0)->GetSampleLayout().GetNumElements() == 1 && positive > 0))
        {
            // evaluation uses softmax
            m_logSoftmax.AssignProductOf(Input(1)->Value(), true, Input(2)->ValueAsMatrix(), false);
            m_logSoftmax += Input(3)->Value();
            m_logSoftmax.InplaceLogSoftmax(false);
            MaskMissingColumnsToZero(m_logSoftmax, Input(1)->GetMBLayout(), fr); // TODO: is this the right way to neutralize gaps?
            Value().AssignSoftmaxSum(Input(0)->Value(), m_logSoftmax);
        }
        else if (m_evalMode == NCEEvalMode::Unnormalized || (Input(0)->GetSampleLayout().GetNumElements() == 1 && negative > 0))
        {
            // TODO: are we treating gaps correctly here?
            Value().AssignNceUnnormalizedEval(Input(0)->Value(), Input(1)->Value(), Input(2)->ValueAsMatrix(), Input(3)->Value());
        }
        else
        {
            // TODO: are we treating gaps correctly here?
            // training criterion uses NCE
            // likelihood                                         samples+probs                        hidden                       embedding            bias
            Value().AssignNoiseContrastiveEstimation(Input(0)->Value(), Input(1)->Value(), Input(2)->ValueAsMatrix(), Input(3)->Value(), m_ncePrediction);
        }
        m_needRecomputeGradientToSoftmaxInput = true;
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr; // this node does not hold mini-batch data

        if (isFinalValidationPass)
        {
            if (Input(1)->GetSampleMatrixNumRows() != Input(2)->GetAsMatrixNumRows())
                LogicError("The Matrix dimension for observation and weight in the NoiseContrastiveEstimationNode operation does not match.");
            if (!Input(0)->HasMBLayout() || !Input(1)->HasMBLayout() || Input(2)->HasMBLayout() || !Input(3)->HasMBLayout())
                LogicError("%ls %ls operation requires inputs 0, 1, and 3 to be a minibatch, and input 2 to be a matrix.", NodeName().c_str(), OperationName().c_str());
        }

        SetDims(TensorShape(1), false);
    }

protected:
    Matrix<ElemType> m_logSoftmax;
    Matrix<ElemType> m_softMax;
    Matrix<ElemType> m_ncePrediction;

    // gradient of cross entropy with respect to the input of softmax
    // a 1 row by \sum_t m_nbrWordsInEachTime[t] vector
    // one slice of size m_nbrWordsInEachTime[t] saves the input to softmax for word y_t
    Matrix<ElemType> m_grdToSoftMaxInput;
    bool m_needRecomputeGradientToSoftmaxInput;

    size_t m_nbrNoise;
    size_t m_totalNbrWords;

private:
    NCEEvalMode m_evalMode;
};
template class NoiseContrastiveEstimationNode<float>;
template class NoiseContrastiveEstimationNode<double>;

// -----------------------------------------------------------------------
// ClassBasedCrossEntropyWithSoftmaxNode (labeldata(.,t), inputdata(.,t), embeddingMatrix, clsProbBeforeSoftmaxData(.,t))
//  - Input(0) [4 x T] label in dense matrix in
//              (0,t) the first row is the word index
//              (1,t) the second row is the class index
//              (2,t) the third row is the first word index of the class
//              (3,t) the last row is the first word index of the next class
//  - Input(1) [hdsize x T] hidden layer activation to the node in. for a simple rnn, this is the hidden layer activty
//  - Input(2) [hdsize x vocab_size] weight matrix in, for speed-up, as per word matrix can be simply obtained as column slice
//  - Input(3) [nbr_cls x T] clsprob in dense matrix in. This input, if applied softmax on, is the posterior probabilty of class given observations
// -----------------------------------------------------------------------

// calculates: -sum(left_i * log(softmax_i(right))) for class given history and for word given history
// need to provide class probabilty from external node
template <class ElemType>
class ClassBasedCrossEntropyWithSoftmaxNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<4>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"ClassBasedCrossEntropyWithSoftmax";
    }

    // our inputs
    static const size_t LABELDATA = 0;
    static const size_t INPUTDATA = 1;
    static const size_t EMBEDDINGMATRIX = 2;
    static const size_t CLASSPROBINDATA = 3;

public:
    DeclareConstructorFromConfigWithNumInputs(ClassBasedCrossEntropyWithSoftmaxNode);
    ClassBasedCrossEntropyWithSoftmaxNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
          m_logSoftmax(deviceId),
          m_softMax(deviceId),
          m_grdToSoftMaxInput(deviceId),
          m_clsLogSoftmax(deviceId),
          m_clsSoftmax(deviceId)
    {
    }

    // compute gradients to input observations, the weights to the observations, and the class log posterior probabilites
    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        // this should never be called for input[0], which is controlled through the needGradient flag
        if (inputIndex != 1 && inputIndex != 2 && inputIndex != 3)
            InvalidArgument("ClassCrossEntropyWithSoftmaxNode criterion only takes with respect to input, weight to the input and class log posterior probability.");

        ComputeSoftMaxPartial();

        const size_t nT = Input(LABELDATA)->GetNumTimeSteps();
        const size_t nS = Input(LABELDATA)->GetNumParallelSequences();
        size_t sz = 0; // iterate over the packed concatenated class-conditioned prob vectors
        for (size_t s = 0; s < nS; s++)
            for (size_t t = 0; t < nT; t++)
            {
                FrameRange fr = FrameRange(Input(LABELDATA)->GetMBLayout(), t).Sequence(s);

                if (Input(LABELDATA)->GetMBLayout()->IsGap(fr)) // skip gaps
                    continue;

                Matrix<ElemType> lbl_t = Input(LABELDATA)->ValueFor(fr);
                size_t c_t = (size_t) lbl_t(1, 0);
                size_t lft_bnd = (size_t) lbl_t(2, 0); // index of first word belonging to current word token's class
                size_t rgt_bnd = (size_t) lbl_t(3, 0); // and end of that range
                size_t nbr_wrd = (rgt_bnd - lft_bnd);  // number of words in the class

                // compute prb - 1 and prb
                Matrix<ElemType> weightForClass = Input(EMBEDDINGMATRIX)->ValueAsMatrix().ColumnSlice(lft_bnd, nbr_wrd);
                Matrix<ElemType> obs = Input(INPUTDATA)->ValueFor(fr); // hidden activation vector for current word token
                Matrix<ElemType> grd_to_soft_max_input = m_grdToSoftMaxInput.ColumnSlice(sz, nbr_wrd);
                Matrix<ElemType> grd_to_cls_prob = DataWithMBLayoutFor(m_clsLogSoftmax, fr, Input(CLASSPROBINDATA)->GetMBLayout());

                switch (inputIndex)
                {
                case 1:
                {
                    // gradient to input
                    Matrix<ElemType> grd_t = Input(INPUTDATA)->GradientFor(fr);
                    Matrix<ElemType>::MultiplyAndAdd(weightForClass, false, grd_to_soft_max_input, true, grd_t);
                    break;
                }
                case 2:
                {
                    // gradient to input weight
                    Matrix<ElemType> grd_to_wgt_t = Input(EMBEDDINGMATRIX)->GradientAsMatrix().ColumnSlice(lft_bnd, nbr_wrd);
                    Matrix<ElemType>::MultiplyAndAdd(obs, false, grd_to_soft_max_input, false, grd_to_wgt_t);
                    break;
                }
                case 3:
                {
                    Matrix<ElemType> grd_t = Input(CLASSPROBINDATA)->GradientFor(fr);
                    grd_t.SetValue(DataWithMBLayoutFor(m_clsSoftmax, fr, Input(CLASSPROBINDATA)->GetMBLayout()));
                    ComputeCEPartialToSoftmaxInputs(grd_t, Gradient(), c_t);
                    break;
                }
                }

                sz += nbr_wrd;
            }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

private:
    void ComputeCEPartialToSoftmaxInputs(Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues, size_t y_t)
    {
        Matrix<ElemType>::MinusOneAt(inputGradientValues, y_t);
        Matrix<ElemType>::Scale(gradientValues, inputGradientValues);
    }

    // gradient of cross entropy w.r.t. to input to softmax
    void ComputeSoftMaxPartial()
    {
        if (m_needRecomputeGradientToSoftmaxInput)
        {
            m_grdToSoftMaxInput.Resize(1, m_totalNbrWords); // buffer that contains a concatenation of class-conditional values

            const size_t nT = Input(LABELDATA)->GetNumTimeSteps();
            const size_t nS = Input(LABELDATA)->GetNumParallelSequences();
            size_t sz = 0; // iterate over the packed concatenated class-conditioned prob vectors
            for (size_t s = 0; s < nS; s++)
                for (size_t t = 0; t < nT; t++)
                {
                    FrameRange fr = FrameRange(Input(LABELDATA)->GetMBLayout(), t).Sequence(s);

                    // if (Input(LABELDATA)->GetMBLayout()->IsGap(s, t))  // skip gaps
                    if (Input(LABELDATA)->GetMBLayout()->IsGap(fr)) // skip gaps
                        continue;

                    Matrix<ElemType> lbl_t = Input(LABELDATA)->ValueFor(fr);
                    size_t y_t = (size_t) lbl_t(0, 0);     // word index
                    size_t lft_bnd = (size_t) lbl_t(2, 0); // index of first word belonging to current word token's class
                    size_t rgt_bnd = (size_t) lbl_t(3, 0); // and end of that range
                    size_t nbr_wrd = (rgt_bnd - lft_bnd);  // number of words in the class

                    Matrix<ElemType> softMax = m_softMax.ColumnSlice(sz, nbr_wrd);

                    size_t idx_in_class = y_t - lft_bnd;
                    ComputeCEPartialToSoftmaxInputs(softMax, Gradient(), idx_in_class);

                    m_grdToSoftMaxInput.ColumnSlice(sz, nbr_wrd).SetValue(softMax);

                    sz += nbr_wrd;
                }

            m_needRecomputeGradientToSoftmaxInput = false;
        }
    }

public:
    virtual void UpdateFunctionMBSize() override
    {
        // TODO: Resize temp matrices here (not doing so does not really fail since for full matrices, class Matrix will resize by itself)
    }

    // -sum(left_i * log(softmax_i(right)))
    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        if (Input(LABELDATA)->Value().GetDeviceId() != CPUDEVICE)
            LogicError("ClassBasedCrossEntropyWithSoftmax (ForwardPropNonLooping()): The label matrix is not using CPU device. This will make computation slow, even though the label data is probably saved on GPU. Because of the external loop over time with explicit class id retrieved from the label matrix, the computation will be very slow if the label matrix is saved on GPU. However, this is only a constraint for label matrix and other matrices such as data are suggested to reside on GPU. ");
        // TODO: Get the label matrix into location=Both state.

        auto& functionValues = Value();

        const size_t hdSize = Input(INPUTDATA)->GetSampleMatrixNumRows(); // hdSize
        assert(m_nbrCls == Input(CLASSPROBINDATA)->GetSampleMatrixNumRows());

        // compute the class posteriors
        m_clsLogSoftmax = Input(CLASSPROBINDATA)->Value();
        m_clsLogSoftmax.InplaceLogSoftmax(true);   // log
        m_clsSoftmax.AssignExpOf(m_clsLogSoftmax); // non-log

        // create a large workspace to contain all class-conditioned probs concatenated
        // 'sz' is the offset into that vector. We will iterate over these vectors at a few places. Always use this same boilerplate code.
        // TODO: should we pull this iteration into an iterator, to reduce the code dup?
        const size_t nT = Input(LABELDATA)->GetNumTimeSteps();
        const size_t nS = Input(LABELDATA)->GetNumParallelSequences();
        size_t sz = 0;
        for (size_t s = 0; s < nS; s++)
            for (size_t t = 0; t < nT; t++)
            {
                FrameRange fr = FrameRange(Input(LABELDATA)->GetMBLayout(), t).Sequence(s);
                if (Input(LABELDATA)->GetMBLayout()->IsGap(fr)) // skip gaps
                    continue;

                const Matrix<ElemType>& lbl_t = Input(LABELDATA)->ValueFor(fr);
                size_t lft_bnd = (size_t) lbl_t(2, 0);
                size_t rgt_bnd = (size_t) lbl_t(3, 0);
                size_t nbr_wrd = (rgt_bnd - lft_bnd); // number of words in the class
                if (nbr_wrd == 0)
                    LogicError("ClassBasedCrossEntropyWithSoftmax (ForwardPropNonLooping()): Encountered a class of size 0. This sample seems to lack an NoInput flag.");

                sz += nbr_wrd;
            }
        m_totalNbrWords = sz; // total size of concatenated vector

        // buffer to hold the concatenated class-conditioned prob vectors
        m_softMax.Resize(1, sz);
        m_logSoftmax.Resize(1, sz);

        // accumulate objective
        functionValues.SetValue(0);
        sz = 0; // iterate over the packed concatenated class-conditioned prob vectors
        for (size_t s = 0; s < nS; s++)
            for (size_t t = 0; t < nT; t++)
            {
                FrameRange fr = FrameRange(Input(LABELDATA)->GetMBLayout(), t).Sequence(s);
                if (Input(LABELDATA)->GetMBLayout()->IsGap(fr)) // skip gaps
                    continue;

                const Matrix<ElemType>& lbl_t = Input(LABELDATA)->ValueFor(fr);
                size_t y_t = (size_t) lbl_t(0, 0);     // current word token index
                size_t c_t = (size_t) lbl_t(1, 0);     // current word token's class index
                size_t lft_bnd = (size_t) lbl_t(2, 0); // index of first word belonging to current word token's class
                size_t rgt_bnd = (size_t) lbl_t(3, 0); // and end of that range
                size_t nbr_wrd = (rgt_bnd - lft_bnd);  // number of words in the class

                // now get views of various arrays that correspond to the index range of words belonging to this class

                // get hidden vectors for the words in this class
                Matrix<ElemType> weightForClass = Input(EMBEDDINGMATRIX)->ValueAsMatrix().ColumnSlice(lft_bnd, nbr_wrd); // [hdSize x nbr_wrd]

                // buffer to hold the class-conditional distribution
                Matrix<ElemType> softMax_t = m_softMax.ColumnSlice(sz, nbr_wrd); // TODO: declare these outside of the loop to avoid the malloc
                Matrix<ElemType> logSoftMax_t = m_logSoftmax.ColumnSlice(sz, nbr_wrd);

                Matrix<ElemType> obs = Input(INPUTDATA)->ValueFor(fr); // hidden activation vector for current word token

                // multiply hidden activation with weight matrix (the slice of the weight matrix for the range of class members)
                // TODO: can we use 'true' here instead? Above transposition hack won't work with row slices. 'obs' not used elsewhere
                obs.Reshape(1, hdSize);                                                                                // transpose it (make it a column vector)
                logSoftMax_t.AssignProductOf(obs /*(1 x hdSize)*/, false, weightForClass /*hdSize x nbr_wrd*/, false); // -> 1 x nbr_word

                // log softmax(W x_t)
                logSoftMax_t.InplaceLogSoftmax(false);

                // and non-log version
                softMax_t.SetValue(logSoftMax_t);
                softMax_t.InplaceExp();
                // we now have a column vector of class-conditional probabilities over the class members

                // add  the word's class-conditional log posterior
                if (y_t < lft_bnd || y_t >= rgt_bnd)
                    LogicError("ClassBasedCrossEntropyWithSoftmax (ForwardPropNonLooping()): Word index out of bounds of class-member index range (word not a class member).");
                size_t idx_in_class = y_t - lft_bnd;
                Matrix<ElemType>::AddElementToElement(logSoftMax_t, 0, idx_in_class, functionValues, 0, 0); // (1x1)

                // add the class log posterior probability
                Matrix<ElemType>::AddElementToElement(m_clsLogSoftmax, c_t, t, functionValues, 0, 0); // (1x1)

                sz += nbr_wrd;
            }

        functionValues *= (-1);

#if NANCHECK
        functionValues.HasNan("ClassBasedCrossEntropyWithSoftmax");
#endif
        m_needRecomputeGradientToSoftmaxInput = true;
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr; // this node does not hold mini-batch data

        if (isFinalValidationPass)
        {
            if (Input(LABELDATA)->GetSampleMatrixNumRows() != 4) // label data needs to have 4 rows
                LogicError("The label data in the ClassBasedCrossEntropyWithSoftmax operation must have 4 rows.");
            if (Input(INPUTDATA)->GetSampleMatrixNumRows() != Input(EMBEDDINGMATRIX)->GetAsMatrixNumRows()) // input and matrix can be timed
                LogicError("The matrix dimension for observation and weight in the ClassBasedCrossEntropyWithSoftmax operation does not match.");
            if (Input(LABELDATA)->GetMBLayout() != Input(INPUTDATA)->GetMBLayout() || Input(LABELDATA)->GetMBLayout() != Input(CLASSPROBINDATA)->GetMBLayout())
                InvalidArgument("%ls %ls operation requires that the layouts of inputs 0 (label), 1 (hidden activation), and 3 (log softmax) match.", NodeName().c_str(), OperationName().c_str());
        }

        SetDims(TensorShape(1), false);

        m_nbrCls = Input(CLASSPROBINDATA)->GetSampleMatrixNumRows();
    }

protected:
    Matrix<ElemType> m_logSoftmax;
    Matrix<ElemType> m_softMax;

    Matrix<ElemType> m_clsLogSoftmax;
    Matrix<ElemType> m_clsSoftmax;

    // gradient of cross entropy with respect to the input of softmax
    // a 1 row by \sum_t m_nbrWordsInEachTime[t] vector
    // one slice of size m_nbrWordsInEachTime[t] saves the input to softmax for word y_t
    Matrix<ElemType> m_grdToSoftMaxInput;
    bool m_needRecomputeGradientToSoftmaxInput;

    size_t m_nbrCls;
    size_t m_totalNbrWords;
};

template class ClassBasedCrossEntropyWithSoftmaxNode<float>;
template class ClassBasedCrossEntropyWithSoftmaxNode<double>;

#ifdef COMING_SOON

// -----------------------------------------------------------------------
// CRFNode (labels, position_dependent_scores, transition_scores)
//  - labels: output label vector of [0:T-1]
//  - position_dependent_scores [0:T-1]: score from position dependent node,
//    in the R-CRF case, it is the RNN output score before softmax
//  - transition scores: square transition matrix,  --TODO: log?
//    in the R-CRF case, it is the transition probability between labels
// BUGBUG: This node cannot operate with truncated BPTT, but does not detect it. It also does not handle gaps or test boundary flags.
// -----------------------------------------------------------------------

/**
        CRF training criterion
        It uses forward-backward algorithm within a minibatch to compute statistics for sequence level optimization
        This node can serve a base class for other sequence level optimization

        Developed by Kaisheng Yao
        This node is for replicating results of the following work
        K. Yao, B. Peng, G. Zweig, D. Yu, X. Li and F. Gao, "Recurrent Conditional Random Fields", NIPS Deep Learning Workshop 2014
        K. Yao, B. Peng, G. Zweig, D. Yu, X. Li and F. Gao, "Recurrent Conditional Random Fields for Language Understanding", ICASSP 2014
        http://research.microsoft.com/pubs/210167/rcrf_v9.pdf

        The forward-backward algorithm follows the derivation in
        http://jmlr.org/papers/volume12/collobert11a/collobert11a.pdf

    */
template <class ElemType>
class CRFNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<3>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"CRF";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(CRFNode);
    CRFNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
          mAlpha(deviceId),
          mBeta(deviceId),
          mPostProb(deviceId)
    {
    }

    // compute posterior probability of label y at position t
    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(Input(0)->GetMBLayout());
        size_t nrow = Input(0)->Value().GetNumRows();
        size_t ncol = Input(0)->Value().GetNumCols();

        mAlpha.Resize(nrow, ncol);
        mBeta.Resize(nrow, ncol);
        mPostProb.Resize(nrow, ncol);

        Value().SetValue(0.0);
        Matrix<ElemType> funcVal = Value(); // TODO: This just creates a 1x1 matrix set to 0.

        size_t nS = Input(0)->GetNumParallelSequences();
        if (nS != 1)
            LogicError("CRFNode: >1 parallel sequences are curently not implemented correctly.");
        for (size_t i = 0; i < nS; i++) // process parallel sequences one by one  --BUGBUG: We should loop over individual sequences.
        {
            FrameRange sequenceRange = fr.Sequence(i); // FrameRange to select one sequence
            // BUGBUG: This ^^ is neither supported nor correct, since this code does not handle gaps or start/end flags.
            ForwardPropS(
                DataWithMBLayoutFor(mPostProb, sequenceRange, Input(0)->GetMBLayout()),
                DataWithMBLayoutFor(mAlpha, sequenceRange, Input(0)->GetMBLayout()),
                DataWithMBLayoutFor(mBeta, sequenceRange, Input(0)->GetMBLayout()),
                funcVal,
                Input(0)->ValueFor(sequenceRange),
                Input(1)->ValueFor(sequenceRange),
                Input(2)->ValueAsMatrix(), mStartLbl,
                mEndLbl);

            Value() += funcVal; // aggregate over sequences
        }
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override // scaled by 2*number of colmns (samples) in the Matrix<ElemType>
    {
        FrameRange fr(Input(0)->GetMBLayout());
        // inputIndex 0 should not get us here, it should be prevented by the needGradient flag of input[0]
        if (inputIndex != 1 && inputIndex != 2)
            InvalidArgument("CRFNode only takes with respect to input and weight.");

        if (inputIndex == 1)
        {
            auto gradient = Input(1)->GradientFor(fr);
            Matrix<ElemType>::AddScaledDifference(Gradient(), mPostProb, Input(0)->ValueFor(fr), gradient);
        }
        else if (inputIndex == 2)
        {
            assert(Input(inputIndex)->GradientFor(fr).GetNumElements() > 0);
            size_t nS = Input(0)->GetNumParallelSequences();
            for (size_t i = 0; i < nS; i++) // process all sequences one by one
            {
                FrameRange sequenceRange = fr.Sequence(i); // FrameRange to select one sequence
                auto& gradient = Input(2)->GradientAsMatrix();
                TransGrdCompute(Input(0)->ValueFor(sequenceRange),
                                DataWithMBLayoutFor(mAlpha, sequenceRange, Input(0)->GetMBLayout()),
                                DataWithMBLayoutFor(mBeta, sequenceRange, Input(0)->GetMBLayout()),
                                Input(2)->ValueAsMatrix(),
                                gradient,
                                mStartLbl, 1);
            }
        }
        else
            return;
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    // compute forward backward algorithm
    /*TODO: merge with call site*/ void ForwardPropS(Matrix<ElemType> postprob, Matrix<ElemType> alpha, Matrix<ElemType> beta, Matrix<ElemType>& functionValues, const Matrix<ElemType>& lbls, const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, int& firstLbl, int& lastLbl, const int iStep = 1)
    {
        // to-do, each slice is for one sentence
        // to-do, number of slices correspond to number of frames
        // this implementation only supports one sentence per minibatch

        int nObs = lbls.GetNumCols();

        // change to other values so can support multiple sentences in each minibatch
        assert(iStep == 1);
        ForwardCompute(alpha, lbls, pos_scores, pair_scores);
        BackwardCompute(alpha, beta, functionValues, lbls, pos_scores, pair_scores, iStep);
        PostProbCompute(postprob, alpha, beta);

        firstLbl = -1;
        for (int ik = 0; ik < lbls.GetNumRows(); ik++)
            if (lbls(ik, 0) != 0)
            {
                firstLbl = ik;
                break;
            }

        lastLbl = -1;
        for (int ik = 0; ik < lbls.GetNumRows(); ik++)
            if (lbls(ik, nObs - 1) != 0)
            {
                lastLbl = ik;
                break;
            }

        functionValues.AssignInnerProductOfMatrices(lbls, pos_scores);

        Matrix<ElemType> a = alpha.ColumnSlice(nObs - 1, 1);
        ElemType fAlpha;
        fAlpha = a.LogAddSumOfElements();

        // transition score
        ElemType tscore = 0;
        for (int t = 0; t < nObs - 1; t++)
        {
            int i = -1;
            for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, t) != 0)
                {
                    i = ik;
                    break;
                }
            int j = -1;
            for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, t + 1) != 0)
                {
                    j = ik;
                    break;
                }
            tscore += pair_scores(j, i);
        }
        tscore += functionValues.Get00Element(); // correct path score
        tscore -= fAlpha;                        // reduced by the scores from all paths
        functionValues.SetValue(tscore);

        functionValues *= (-1);
    }

    // compute forward backward algorithm
    static void ForwardCompute(Matrix<ElemType>& alpha,
                               const Matrix<ElemType>& lbls,
                               const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores)
    {
        // to-do, shift more than 1 to support muliple sentences per minibatch
        int iNumPos = lbls.GetNumCols();
        int iNumLab = lbls.GetNumRows();

        int firstLbl = -1;
        for (int ik = 0; ik < lbls.GetNumRows(); ik++)
            if (lbls(ik, 0) != 0)
            {
                firstLbl = ik;
                break;
            }

        // need to have
        alpha.Resize(iNumLab, iNumPos);

        for (int t = 0; t < iNumPos; t++)
        {
            for (int k = 0; k < iNumLab; k++)
            {
                ElemType fTmp = (ElemType) LZERO;
                for (int j = 0; j < iNumLab; j++)
                {
                    ElemType fAlpha = (j == firstLbl) ? (ElemType) 0.0 : (ElemType) LZERO;
                    if (t > 0)
                        fAlpha = alpha(j, t - 1);
                    fTmp = alpha.LogAdd(fTmp, fAlpha + pair_scores(k, j));
                }
                fTmp += pos_scores(k, t); // include position dependent score
                alpha(k, t) = fTmp;
            }
        }
    }

    // compute backward algorithm
    static void BackwardCompute(const Matrix<ElemType>& alpha, Matrix<ElemType>& beta,
                                Matrix<ElemType>& functionValues, const Matrix<ElemType>& lbls,
                                const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, const int shift = 1)
    {
        assert(shift == 1);

        alpha.RCRFBackwardCompute(alpha, beta, functionValues, lbls, pos_scores, pair_scores, shift);
    }

    static void TransGrdCompute(const Matrix<ElemType>& lbls,
                                const Matrix<ElemType>& alpha,
                                const Matrix<ElemType>& beta,
                                const Matrix<ElemType>& pair_scores,
                                Matrix<ElemType>& grd,
                                const int startLbl,
                                const int shift = 1)
    {
        assert(shift == 1);

        alpha.RCRFTransGrdCompute(lbls,
                                  alpha,
                                  beta,
                                  pair_scores,
                                  grd,
                                  startLbl, shift);
    }

    // compute forward backward algorithm
    static void PostProbCompute(Matrix<ElemType>& postprob, const Matrix<ElemType>& alpha, const Matrix<ElemType>& beta)
    {
        int iNumPos = alpha.GetNumCols();
        int iNumLab = alpha.GetNumRows();

        postprob.Resize(iNumLab, iNumPos);
        postprob.SetValue(beta);
        postprob.InplaceExp();
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr; // this node does not hold mini-batch data

        if (isFinalValidationPass)
            if (!(Input(1)->GetSampleMatrixNumRows() == Input(2)->GetAsMatrixNumRows() && // position dependent and pair scores have same number of labels
                  Input(0)->GetSampleMatrixNumRows() == Input(1)->GetSampleMatrixNumRows() &&
                  Input(0)->HasMBLayout() && Input(0)->GetMBLayout() == Input(1)->GetMBLayout() &&
                  // Input(0)->GetNumCols() == Input(1)->GetNumCols() && // position dependent and pair scores have the same observation numbers
                  Input(2)->GetAsMatrixNumCols() == Input(2)->GetAsMatrixNumRows()))
            {
                LogicError("The Matrix dimension in the CRFNode operation does not match.");
            }

        SetDims(TensorShape(1), false);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<CRFNode<ElemType>>(nodeP);
            node->mAlpha = mAlpha;
            node->mBeta = mBeta;
            node->mPostProb = mPostProb;

            node->mStartLbl = mStartLbl;
            node->mEndLbl = mEndLbl;
        }
    }

private:
    Matrix<ElemType> mAlpha; // TODO: m_Alpha etc.
    Matrix<ElemType> mBeta;
    Matrix<ElemType> mPostProb;
    int mStartLbl;
    int mEndLbl;
};

#endif

// -----------------------------------------------------------------------
// LogisticNode (labels, prediction, weight)
// calculates: -sum(left * log(right) + (1-left)*log(1-right)) (optionally * weight)
// -----------------------------------------------------------------------

template <class ElemType>
class LogisticNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"Logistic";
    }

public:
    DeclareConstructorFromConfig(LogisticNode);
    LogisticNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        FrameRange fr(Input(0)->GetMBLayout());
        if (inputIndex != 1)
            InvalidArgument("%ls %ls operation cannot compute the gradient for its first inpute.", NodeName().c_str(), OperationName().c_str());

        // BackpropToRight(m_temp, Input(0)->Value(), Input(2)->Value(), Input(inputIndex)->Gradient(), Gradient(), m_classZeroLabels, m_result);
        // Create vector with 1 for class 1, and -1 for class 0
        m_temp->AssignDifferenceOf(Input(0)->ValueFor(fr), *m_classZeroLabels); // TODO: need a slice for m_classZeroLabels?

        // Multiply the vector by the Input(2)->Value()
        if (m_inputs.size() == 3)                                            // with weight
            m_temp->AssignElementProductOf(*m_temp, Input(2)->ValueFor(fr)); // TODO: is Input(2) minibatch data? Confirm

        // divide class by p (class 1) or (1-p) (class 0)
        m_temp->AssignElementDivisionOf(*m_temp, *m_result); // TODO: this is in-place--does this function allow that?

        auto gradient = Input(inputIndex)->GradientFor(fr);
        Matrix<ElemType>::Multiply1x1AndWeightedAdd(-1.0f, Gradient() /*1x1*/, *m_temp, 1.0f, gradient);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    virtual void UpdateFunctionMBSize() override
    {
        m_classZeroLabels->Resize(Input(0)->Value());
        m_result->Resize(Input(0)->Value());
        m_temp->Resize(Input(0)->Value());
    }

    // -sum(left * log(right) + (1-left)*log(1-right)) (optionally * weight)
    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(Input(0)->GetMBLayout());

        const Matrix<ElemType>& classOneLabels = Input(0)->ValueFor(fr);
        const Matrix<ElemType>& classOneProbabilities = Input(1)->ValueFor(fr);
        Matrix<ElemType>& classZeroLabels = *m_classZeroLabels;

        Matrix<ElemType> ones = ConstOnes(classOneLabels.GetNumRows(), classOneLabels.GetNumCols(), classOneLabels.GetDeviceId());

        // compute the indices for the class 0 indices
        classZeroLabels.AssignDifferenceOf(ones, classOneLabels);

        /* We're computing result = weight*(y*p + (1-y)*(1-p) = 2*y*p + (1-y) - p) */

        /* First compute result = y*p */
        m_result->AssignElementProductOf(classOneLabels, classOneProbabilities);

        // TODO: verify that all these operations on m_result really can do in-place (or use different methods instead)
        /* Now compute result = 2*y*p */
        m_result->AssignProductOf((ElemType) 2.0, *m_result);

        /* Now compute result = 2*y*p + (1-y) */
        m_result->AssignSumOf(*m_result, classZeroLabels);

        /* Finally compute result = 2*y*p + (1-y) - p */
        m_result->AssignDifferenceOf(*m_result, classOneProbabilities);

        // compute the log, resulting in y*log(p) + (1-y)*log(1-p)
        m_temp->AssignLogOf(*m_result);

        // The error is the negative of the sum of the result
        if (m_inputs.size() == 2)
            Value().AssignSumOfElements(*m_temp);
        else
            Value().AssignInnerProductOf(Input(2)->ValueFor(fr), *m_temp, false);
        Value() *= (-1);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        if (m_inputs.size() != 2 && m_inputs.size() != 3)
            InvalidArgument("%ls %ls operation requires two or three inputs.", NodeName().c_str(), OperationName().c_str());

        ValidateBinaryReduce(isFinalValidationPass);

        /* Note that this is the same as ValidateInferBinaryInputDims, but done for the 3rd child if it exists */
        if (m_inputs.size() == 3)
        {
            auto weights = Input(2);
            auto other = Input(1);
            // borrow any unset dimension on one input from the other input
            weights->ValidateInferInputDimsFrom(other->GetSampleLayout());

            if (isFinalValidationPass &&
                !(Input(0)->GetSampleMatrixNumRows() == Input(2)->GetSampleMatrixNumRows() &&
                  (Input(0)->GetMBLayout() == Input(2)->GetMBLayout() || !Input(0)->HasMBLayout() || !Input(2)->HasMBLayout())))
            {
                LogicError("The Matrix dimensions of the second argument weights the %ls %ls operation do not match.", NodeName().c_str(), OperationName().c_str());
            }
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_classZeroLabels, matrixPool);
        RequestMatrixFromPool(m_result, matrixPool);
        RequestMatrixFromPool(m_temp, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_classZeroLabels, matrixPool);
        ReleaseMatrixToPool(m_result, matrixPool);
        ReleaseMatrixToPool(m_temp, matrixPool);
    }

    virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<LogisticNode<ElemType>>(nodeP);
            *node->m_classZeroLabels = *m_classZeroLabels;
            *node->m_result = *m_result;
            *node->m_temp = *m_temp;
        }
    }

private:
    shared_ptr<Matrix<ElemType>> m_classZeroLabels;
    shared_ptr<Matrix<ElemType>> m_result;
    shared_ptr<Matrix<ElemType>> m_temp;
};

template class LogisticNode<float>;
template class LogisticNode<double>;

// -----------------------------------------------------------------------
// DropoutNode (input) -- perform drop-out
// Output is scaled such that no post-scaling is necessary.
// -----------------------------------------------------------------------

template <class ElemType>
class DropoutNode : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"Dropout";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(DropoutNode);
    DropoutNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
          m_dropoutRate(0)
    {
        m_randomSeed = (unsigned long) CreateUniqId();
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Grad = Input(0)->GradientFor(fr);
        Matrix<ElemType> sliceOutputGrad = GradientFor(fr);

        if (m_dropoutRate > 0)
            sliceInput0Grad.AddElementProductOf(sliceOutputGrad, DataFor(*m_maskOfDropout, fr));
        else
            sliceInput0Grad += sliceOutputGrad;
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The DropoutNode does not require its output value for computing
        // the gradients of its input nodes
        return false;
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
    {
        // The DropoutNode does not require any of it's input's values for computing
        // the gradients of its input nodes
        UNREFERENCED_PARAMETER(childIndex);
        return false;
    }

    virtual void UpdateFunctionMBSize() override
    {
        Base::UpdateFunctionMBSize();
        // resize temporaries to their proper size
        if (m_dropoutRate > 0)
            m_maskOfDropout->Resize(Input(0)->Value());
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        if (m_dropoutRate > 0)
        {
            // determine drop-out mask for this minibatch
            auto sliceMask = DataFor(*m_maskOfDropout, fr);
            sliceMask.SetUniformRandomMask((ElemType) m_dropoutRate, (ElemType)(1.0 / (1.0 - m_dropoutRate)) /*pre-scaled*/, m_randomSeed);
            m_randomSeed += 1073807359; // 1073807359 is a very large prime number to avoid collision with other dropout nodes
            // apply dropout mask
            sliceOutputValue.AssignElementProductOf(sliceMask, sliceInput0Value);
        }
        else
        {
            sliceOutputValue.SetValue(sliceInput0Value);
        }
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateUnaryMap(isFinalValidationPass);
    }

    // special methods for this node type which ComputationNetwork knows about and calls to pass parameters
    void SetDropoutRate(const double val)
    {
        if (val < 0 || val >= 1)
            LogicError("DropoutRate must be >= 0 and < 1.");
        m_dropoutRate = val;
    }

    void SetRandomSeed(const unsigned long val)
    {
        m_randomSeed = (unsigned long) val;
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<DropoutNode<ElemType>>(nodeP);
            node->m_dropoutRate = m_dropoutRate;
            node->m_randomSeed = m_randomSeed;
            node->m_maskOfDropout = m_maskOfDropout;
        }
    }
    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_maskOfDropout, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_maskOfDropout, matrixPool);
    }

private:
    double m_dropoutRate;
    unsigned long m_randomSeed;

    shared_ptr<Matrix<ElemType>> m_maskOfDropout;
};

template class DropoutNode<float>;
template class DropoutNode<double>;

// -----------------------------------------------------------------------
// BatchNormalizationNode (...)  --TODO: document inputs
// -----------------------------------------------------------------------

// Implements batch normalization technique as described in:
// Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift [S. Ioffe, C. Szegedy]
// http://arxiv.org/abs/1502.03167
// REVIEW alexeyk: is this a right header for this node? It's not really limited to only convolutional nodes.
template <class ElemType>
class BatchNormalizationNode : public ComputationNode<ElemType>, public NumInputs<5>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"BatchNormalization";
    }

public:
    BatchNormalizationNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name), m_eval(false), m_spatial(false), m_expAvgFactor(0), m_mbCount(0), m_imageLayoutKind(ImageLayoutKind::CHW)
    {
    }
    BatchNormalizationNode(DEVICEID_TYPE deviceId, const wstring& name, bool eval, bool spatial, double expAvgFactor, ImageLayoutKind imageLayoutKind)
        : Base(deviceId, name), m_eval(eval), m_spatial(spatial), m_expAvgFactor(expAvgFactor), m_imageLayoutKind(imageLayoutKind), m_mbCount(0)
    {
    }
    BatchNormalizationNode(const ScriptableObjects::IConfigRecordPtr configp)
        : BatchNormalizationNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"eval"), configp->Get(L"spatial"), configp->Get(L"expAvgFactor"),
                                 ImageLayoutKindFrom(configp->Get(L"imageLayout")))
    {
        AttachInputs(configp, this->GetExpectedNumInputs());
    }

    void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_version.VerWrittenCur() << m_version.VerReadableCur();

        fstream << m_eval;
        fstream << m_spatial;
        fstream << m_expAvgFactor;
        fstream << (int32_t) m_imageLayoutKind;
        fstream << m_mbCount;
    }

    void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);

        // Read and check version.
        // REVIEW alexeyk: extract version checking so it can be re-used in other places.
        // BUGBUG: We must serialize m_inputLayout.
        int32_t verWritten;
        int32_t verReadable;
        fstream >> verWritten >> verReadable;

        if (verReadable > verWritten)
            RuntimeError("Corrupt model file.");
        if (verWritten < m_version.VerWeCanReadBack())
            RuntimeError("Model is too old.");
        if (verReadable > m_version.VerWrittenCur())
            RuntimeError("Model is too new.");

        fstream >> m_eval;
        fstream >> m_spatial;
        fstream >> m_expAvgFactor;
        if (verWritten >= 0x00010002)
        {
            fstream >> m_imageLayoutKind;
            fstream >> m_mbCount;
        }
    }

    void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<BatchNormalizationNode<ElemType>>(nodeP);
            assert(node != nullptr);

            node->m_eval = m_eval;
            node->m_spatial = m_spatial;
            node->m_expAvgFactor = m_expAvgFactor;
        }
    }

    void BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        if (m_eval)
            LogicError("BatchNormalization does not compute derivatives in inference mode.");

        if (inputIndex == 0) // derivative with respect to the input.
        {
            auto sliceOutputGrad = GradientFor(fr);
            auto sliceInputValue = Input(0)->ValueFor(fr);
            const Matrix<ElemType>& scale = Input(1)->Value();
            const Matrix<ElemType>& bias = Input(2)->Value();

            size_t batchSize = sliceInputValue.GetNumCols();
            m_inT->setN(batchSize);
            assert(m_convEng != nullptr);

            auto sliceInputGrad = Input(0)->GradientFor(fr);
            m_dScale->Resize(scale);
            m_dBias->Resize(bias);
            // Compute all derivatives in one step. Save derivatives with respect to scale and bias in temp matrices.
            m_convEng->BackwardNormalizeBatch(*m_inT, sliceInputValue, sliceOutputGrad, sliceInputGrad, *m_scaleBiasT, scale, m_spatial,
                                              *m_saveMean, *m_saveInvStdDev, *m_dScale, *m_dBias);
        }
        else if (inputIndex == 1) // derivative with respect to the scale
        {
            // Derivative with respect to the scale was precomputed during input derivative computation.
            Matrix<ElemType>& grad = Input(1)->Gradient();
            grad.SetValue(grad.GetNumRows(), grad.GetNumCols(), grad.GetDeviceId(), m_dScale->BufferPointer());
        }
        else if (inputIndex == 2) // derivative with respect to the bias
        {
            // Derivative with respect to the bias was precomputed during input derivative computation.
            Matrix<ElemType>& grad = Input(2)->Gradient();
            grad.SetValue(grad.GetNumRows(), grad.GetNumCols(), grad.GetDeviceId(), m_dBias->BufferPointer());
        }
        // No derivatives with respect to running mean and InvStdDev.
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The BatchNormalizationNode does not require its output value for computing
        // the gradients of its input nodes
        return false;
    }

    void ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInputValue = Input(0)->ValueFor(fr);

        const Matrix<ElemType>& scale = Input(1)->Value();
        const Matrix<ElemType>& bias = Input(2)->Value();
        Matrix<ElemType>& runMean = Input(3)->Value();
        Matrix<ElemType>& runInvStdDev = Input(4)->Value();
        assert(scale.GetNumRows() == bias.GetNumRows());
        assert(scale.GetNumCols() == bias.GetNumCols());
        assert(runMean.GetNumRows() == scale.GetNumRows());
        assert(runMean.GetNumCols() == scale.GetNumCols());
        assert(runMean.GetNumRows() == runInvStdDev.GetNumRows());
        assert(runMean.GetNumCols() == runInvStdDev.GetNumCols());

        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        size_t batchSize = sliceInputValue.GetNumCols();
        m_inT->setN(batchSize);
        assert(m_convEng != nullptr);
#if NANCHECK
        sliceInputValue.HasNan("BatchNormalization-input");
#endif
        if (m_eval)
            m_convEng->NormalizeBatchInference(*m_inT, sliceInputValue, *m_scaleBiasT, scale, bias, m_spatial, runMean, runInvStdDev, sliceOutputValue);
        else
        {
            // REVIEW alexeyk: hack, use m_expAvgFactor <= 0 to compute CMA.
            double expAvgFactor = (m_expAvgFactor > 0) ? m_expAvgFactor : (1.0 / (1.0 + m_mbCount));

            if (m_saveMean->GetNumElements() != runMean.GetNumElements())
                m_saveMean->Resize(runMean.GetNumRows(), runMean.GetNumCols());
            if (m_saveInvStdDev->GetNumElements() != runMean.GetNumElements())
                m_saveInvStdDev->Resize(runMean.GetNumRows(), runMean.GetNumCols());

            m_convEng->NormalizeBatch(*m_inT, sliceInputValue, *m_scaleBiasT, scale, bias, m_spatial, expAvgFactor, runMean, runInvStdDev,
                                      sliceOutputValue, *m_saveMean, *m_saveInvStdDev);

            m_mbCount++;
        }
#if NANCHECK
        sliceOutputValue.HasNan("BatchNormalization-output");
        runMean.HasNan("BatchNormalization-runMean");
        runInvStdDev.HasNan("BatchNormalization-runInvStdDev");
        m_saveMean->HasNan("BatchNormalization-saveMean");
        m_saveInvStdDev->HasNan("BatchNormalization-saveInvStdDev");
#endif
    }

    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        SetDims(Input(0));

        if (isFinalValidationPass)
        {
            auto shape = GetSampleLayout();

            if (m_factory == nullptr)
                m_factory = ConvolutionEngineFactory<ElemType>::Create(m_deviceId, ConvolutionEngineFactory<ElemType>::EngineType::Auto, m_imageLayoutKind);
            if (m_convEng == nullptr)
                m_convEng = m_factory->CreateConvEngine(m_deviceId, 0);
            if (m_spatial)
            {
                auto dims = ImageDimensions(shape, m_imageLayoutKind);
                if (m_inT == nullptr)
                    m_inT = m_factory->CreateTensor(dims.m_width, dims.m_height, dims.m_numChannels, 1);
                if (m_scaleBiasT == nullptr)
                    m_scaleBiasT = m_factory->CreateTensor(1, 1, dims.m_numChannels, 1);
            }
            else
            {
                if (m_inT == nullptr)
                    m_inT = m_factory->CreateTensor(shape.GetNumElements(), 1, 1, 1);
                if (m_scaleBiasT == nullptr)
                    m_scaleBiasT = m_factory->CreateTensor(shape.GetNumElements(), 1, 1, 1);
            }
        }
    }

    void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        if (!m_eval)
        {
            RequestMatrixFromPool(m_saveMean, matrixPool);
            RequestMatrixFromPool(m_saveInvStdDev, matrixPool);
        }
    }

    void RequestMatricesBeforeBackprop(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        if (!m_eval)
        {
            RequestMatrixFromPool(m_dScale, matrixPool);
            RequestMatrixFromPool(m_dBias, matrixPool);
        }
    }

    void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        if (!m_eval)
        {
            ReleaseMatrixToPool(m_saveMean, matrixPool);
            ReleaseMatrixToPool(m_saveInvStdDev, matrixPool);
            ReleaseMatrixToPool(m_dScale, matrixPool);
            ReleaseMatrixToPool(m_dBias, matrixPool);
        }
    }

    void SetEvalMode(bool bnEvalMode)
    {
        m_eval = bnEvalMode;
    }

private:
    struct VersionInfo
    {
        // int32_t VerWrittenCur() const     { return 0x00010001; } // Initial
        int32_t VerWrittenCur() const
        {
            return 0x00010002;
        } // Added m_imageLayoutKind and m_mbCount
        int32_t VerReadableCur() const
        {
            return 0x00010002;
        }
        int32_t VerWeCanReadBack() const
        {
            return 0x00010001;
        }
    };
    VersionInfo m_version;

private:
    // Determines whether to use training or inference(evaluation) mode.
    bool m_eval;
    // Determines whether to use per-activation (used after non-convolutional layers like fully connected)
    // or spatial (used after convolutional layers).
    bool m_spatial;
    // Smoothing factor.
    double m_expAvgFactor;
    // Layout (e.g. CHW).
    ImageLayoutKind m_imageLayoutKind;
    // Minibatch count, used to compute cumulative moving average.
    size_t m_mbCount;

    // Stores pre-computed on forward pass mean values that are used in gradient computation.
    shared_ptr<Matrix<ElemType>> m_saveMean;
    // Stores pre-computed on forward pass InvStdDev values that are used in gradient computation.
    shared_ptr<Matrix<ElemType>> m_saveInvStdDev;
    // Stores scale derivatives
    shared_ptr<Matrix<ElemType>> m_dScale;
    // Stores bias derivatives.
    shared_ptr<Matrix<ElemType>> m_dBias;

    std::unique_ptr<ConvolutionEngineFactory<ElemType>> m_factory;
    std::unique_ptr<ConvolutionEngine<ElemType>> m_convEng;
    std::unique_ptr<ConvolutionTensor4D> m_inT;
    std::unique_ptr<ConvolutionTensor4D> m_scaleBiasT;
};

template class BatchNormalizationNode<float>;
template class BatchNormalizationNode<double>;

} } }
