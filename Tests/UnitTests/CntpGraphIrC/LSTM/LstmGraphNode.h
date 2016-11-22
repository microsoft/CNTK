#pragma once
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// EvalMultithreads.cpp : Sample application shows how to evaluate a model in multiple threading environment. 
//
#include "CNTKLibrary.h"

class LstmGraphNode : public CNTK::UserDefinedFunctionHandler
{
public:

	LstmGraphNode(
		std::vector<CNTK::Variable>& inputs,
		CNTK::Dictionary&& functionConfig,
		const std::wstring& name,
		const std::wstring& uid);

	virtual ~LstmGraphNode();

	virtual /*BackPropStatePtr*/void ForwardFloat(
		std::vector<float>& out,
		const std::vector<float>& left,
		const std::vector<float>& right
	) override;

	virtual void Backward(
		////const BackPropStatePtr& /*state*/,
		////const std::unordered_map<Variable, ValuePtr>& /*rootGradientValues*/,
		////std::unordered_map<Variable, ValuePtr>& /*backPropagatedGradientValuesForInputs*/
	) override;

private:
	std::vector<CNTK::Variable> _inputs;
	CNTK::Dictionary _functionConfig;
	const std::wstring _name;
	const std::wstring _uid;
};


CNTK::FunctionPtr LstmGraphNodeFactory(
	const std::wstring& op,
	std::vector<CNTK::Variable>& inputs,
	CNTK::Dictionary&& functionConfig,
	const std::wstring& functionName,
	const std::wstring& uid);
