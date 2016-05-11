//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Basics.h"
#include "ComputationNode.h"
#include "Matrix.h"
#include "TensorView.h"
#include "RNNNodes.h"

#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <algorithm>
#include <utility>
#include <assert.h>


namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// RNNNode
// -----------------------------------------------------------------------

template<class ElemType>
RNNNode<ElemType>::RNNNode(DEVICEID_TYPE deviceId, const wstring& name) 
    : Base(deviceId, name), m_numLayers(7), m_numHidden(123)
{
}

// This constructor helps with BrainScript integration
template<class ElemType>
RNNNode<ElemType>::RNNNode(const ScriptableObjects::IConfigRecordPtr configp)
    : Base(configp->Get(L"deviceId"), L"<placeholder>"), m_numHidden(configp->Get(L"numHidden")), m_numLayers(configp->Get(L"numLayers")),
    m_BackwardDataCalledYet(false)
{
    AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
}

template<class ElemType>
void RNNNode<ElemType>::Save(File& fstream) const
{
    Base::Save(fstream);
    // todo: save RNN topology
    fstream << m_numHidden;
    fstream << m_numLayers;
}

template<class ElemType>
void RNNNode<ElemType>::Load(File& fstream, size_t modelVersion)
{
    Base::Load(fstream, modelVersion);
    // load RNN topology
    fstream >> m_numHidden;
    fstream >> m_numLayers;
}

template<class ElemType>
TensorView<ElemType> RNNNode<ElemType>::TensorHelper(int inputIndex/*-1 for output*/, bool gradient/*instead of value*/, const FrameRange& fr)
{
    auto input = inputIndex < 0 ? this : Input(inputIndex).get();
    return gradient ? input->GradientTensorFor(SIZE_MAX, fr) : input->ValueTensorFor(SIZE_MAX, fr);
}

template<class ElemType>
void RNNNode<ElemType>::TransposeHelper(const MatrixBasePtr matX, const TensorShape &shapeX, MatrixBasePtr matY, TensorShape &shapeY)
{
    shapeY = shapeX;
    shapeY.SwapDimsInPlace(1, 2);

    TensorView<ElemType> Y(matY, TensorShape(shapeY.GetDims()));
    TensorView<ElemType> X(matX, shapeY);
    Y.AssignCopyOf(X);
    shapeY = Y.GetShape();
};

template<class ElemType>
void RNNNode<ElemType>::ForwardProp(const FrameRange& fr)
{
    // The parameters are stored in a column matrix
    Matrix<ElemType>& paramW = Input(1)->Value();

    TensorView<ElemType> outputY = ValueTensorFor(SIZE_MAX, fr);

    // ComputationNode derived classes are guaranteed to have a MBLayout
    if (!this->HasMBLayout())
    {
        LogicError("RNNNode must operate on minibatches");
    }

    // For windowed LSTM, CNTK is providing data with the second dimension being time-like and the third dimension
    // being minibatch index. CuDnn expects the second dimension to be minibatch index, and the third dimension
    // to be time-like. This sequence of operations creates a transposed copy of the data in m_transposedInput
    // and shapeXT

    m_transposedInput->Resize(Input(0)->Value());
    TransposeHelper(Input(0)->ValuePtr(), Input(0)->GetTensorSliceFor(SIZE_MAX, fr), m_transposedInput, shapeXT);

    // Similarly, we will eventually need to transpose the output. Generate the necessary shape here, and do
    // the transposition after RNNForward() returns.

    // ensure enough storage.
    m_transposedOutput->Resize(this->Value());

    // create the necessary shape.
    shapeYT = TensorShape(this->GetTensorSliceFor(SIZE_MAX, fr));
    // this swap results in a shape with swapped dimensions, but also swapped strides
    shapeYT.SwapDimsInPlace(1, 2);
    // this copy is necessary so that the strides are dense.
    shapeYT = TensorShape(shapeYT.GetDims());

    m_transposedOutput->RNNForward(*m_transposedInput, shapeXT, paramW, shapeYT, m_numLayers, m_numHidden, *m_reserve, *m_workspace);

    // No one uses shapeY, but it is necessary
    TensorShape shapeY;
    TransposeHelper(m_transposedOutput, TensorShape(shapeYT.GetDims()), this->ValuePtr(), shapeY);
    m_BackwardDataCalledYet = false;
}

template<class ElemType>
void RNNNode<ElemType>::BackpropTo(const size_t inputIndex, const FrameRange& fr)
{
    // ensure BackwardData is the first method called, as required by CuDnn API
    if (!m_BackwardDataCalledYet)
    {
        Matrix<ElemType>& paramW = Input(1)->Value();

        // To obey the data layout constraints of CuDnn, we take the derivative we're given,
        // and transpose it before feeding to the interface.
        m_transposedDOutput->Resize(this->Gradient());
        TransposeHelper(this->GradientPtr(), this->GetTensorSliceFor(SIZE_MAX, fr), m_transposedDOutput, shapeYT);

        // Ensure enough space for the result
        m_transposedDInput->Resize(Input(1)->Gradient());

        // Do the work
        m_transposedOutput->RNNBackwardData(*m_transposedDOutput, shapeYT, paramW, *m_transposedDInput, shapeXT, *m_reserve, *m_workspace);
        m_BackwardDataCalledYet = true;
    }
    if (inputIndex == 1) // parameters
    {
        Matrix<ElemType>& paramDW = Input(1)->Gradient();
        m_transposedOutput->RNNBackwardWeights(*m_transposedInput, shapeXT, *m_transposedOutput, shapeYT, paramDW, *m_reserve, *m_workspace);
    }
    else if (inputIndex == 0) // data
    {
        // all of the work was done above, where RNNBackwardData is called. Now, just place a transposed result.
        TensorShape tmp;
        TransposeHelper(m_transposedDInput, shapeXT, Input(0)->GradientPtr(), tmp);
    }
}

template<class ElemType>
void RNNNode<ElemType>::Validate(bool isFinalValidationPass)
{
    // N.B.: I need both of these lines.
    Base::Validate(isFinalValidationPass);
    InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

    // get tensor shapes
    auto dimsA = Input(1)->GetSampleLayout().GetDims(); // data
    auto dimsB = Input(0)->GetSampleLayout().GetDims(); // parameters

    // validate and infer
    if (isFinalValidationPass || (dimsA.size() > 0 && dimsB.size() > 0)) // only if we got at least some input dimensions to work with or need to wrap up
    {
        // now determine result dimensions
        auto dimsC = dimsB;
        // output dims - bugbug: this is hard-coded for bidirectional models
        dimsC[0] = 2 * m_numHidden;

        // N.B. - this is the magical call, the reason for the function
        // dimensions would be outputRank * numSamples * minibatch * time.
        // This call establishes outputRank * numSamples, the rest will be filled in
        // dynamically though the MBLayout.
        SetDims(TensorShape(dimsC), HasMBLayout());

    }
};

template class RNNNode<float>;
template class RNNNode<double>;

} } }
