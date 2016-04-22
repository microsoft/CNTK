//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

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
	BackwardDataCalledYet(false)
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


	m_transposedInput->Resize(Input(0)->Value());
	TransposeHelper(Input(0)->ValuePtr(), Input(0)->GetTensorSliceFor(SIZE_MAX, fr), m_transposedInput, shapeXT);

	m_transposedOutput->Resize(this->Value());
	shapeYT = TensorShape(this->GetTensorSliceFor(SIZE_MAX, fr));
	shapeYT.SwapDimsInPlace(1, 2);
	shapeYT = TensorShape(shapeYT.GetDims());

	m_transposedOutput->RNNForward(*m_transposedInput, shapeXT, paramW, shapeYT, m_numLayers, m_numHidden);

	TensorShape shapeY;
	TransposeHelper(m_transposedOutput, TensorShape(shapeYT.GetDims()), this->ValuePtr(), shapeY);
	BackwardDataCalledYet = false;
}

template<class ElemType>
void RNNNode<ElemType>::BackpropTo(const size_t inputIndex, const FrameRange& fr)
{
	// ensure BackwardData is the first method called
	if (!BackwardDataCalledYet)
	{
		Matrix<ElemType>& paramW = Input(1)->Value();

		m_transposedDOutput->Resize(this->Gradient());
		TransposeHelper(this->GradientPtr(), this->GetTensorSliceFor(SIZE_MAX, fr), m_transposedDOutput, shapeYT);

		m_transposedDInput->Resize(Input(1)->Gradient());
		m_transposedOutput->RNNBackwardData(*m_transposedDOutput, shapeYT, paramW, *m_transposedDInput, shapeXT);
		BackwardDataCalledYet = true;
	}
	if (inputIndex == 1) // parameters
	{
		Matrix<ElemType>& paramDW = Input(1)->Gradient();
		m_transposedOutput->RNNBackwardWeights(*m_transposedInput, shapeXT, *m_transposedOutput, shapeYT, paramDW);
	}
	else if (inputIndex == 0) // data
	{
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
	auto dimsA = Input(1)->GetSampleLayout().GetDims();
	auto dimsB = Input(0)->GetSampleLayout().GetDims();
	string dimsAstring = string(Input(1)->GetSampleLayout()); // for error messages
	string dimsBstring = string(Input(0)->GetSampleLayout());

	// validate and infer
	if (isFinalValidationPass || (dimsA.size() > 0 && dimsB.size() > 0)) // only if we got at least some input dimensions to work with or need to wrap up
	{
		// now determine result dimensions
		// bugbug - could want to squash output dims, need to reduce?
		auto dimsC = dimsB;
		//dimsC.resize(m_outputRank);    // output dims
		dimsC[0] = 2 * m_numHidden;

		/// N.B. - this is the magical call, the reason for the function
		/// dimensions would be outputRank * numSamples * minibatch * time
		SetDims(TensorShape(dimsC), HasMBLayout());

		// update dimensions of A
		// update if LearnableParameter
		// Input(0)->ValidateInferInputDimsFrom(TensorShape(dimsA));
	}
};

template class RNNNode<float>;
template class RNNNode<double>;

} } }