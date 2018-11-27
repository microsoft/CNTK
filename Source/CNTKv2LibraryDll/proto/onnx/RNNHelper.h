#pragma once
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// originally from CNTK\Tests\EndToEndTests\CNTKv2Library\Common\Common.h
// 

#pragma once
#include "stdafx.h"
#include "CNTKLibrary.h"

#include <algorithm>
#include <functional>

using namespace std;

namespace onnxruntime
{
    class Node;
}

// once an input is wrapped with to-batch/sequence, it shall not get wrapped again
typedef std::unordered_map<CNTK::Variable, CNTK::FunctionPtr> VariableToFunctionPtr;

const std::string LSTMInputBiasNameHint = "_bias_";
const std::string LSTMInputInitialHNameHint = "_initial_h_";
const std::string LSTMInputInitialCNameHint = "_initial_c_";
const std::string LSTMInputPeepholeNameHint = "_peephole_";

const std::string GRUInputInitialHNameHint = "_initial_h_";

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#attributes-18
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#attributes-27
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#attributes-39

enum
{
    LSTMInputIndexX = 0,
    LSTMInputIndexW = 1,
    LSTMInputIndexH = 2,
    LSTMInputIndexB = 3,
    LSTMInputIndexSequenceLens = 4,
    LSTMInputIndexinitial_h = 5,
    LSTMInputIndexinitial_c = 6,
    LSTMInputIndexP = 7
};

enum
{
    LSTMActivationFIndex = 0,
    LSTMActivationGIndex = 1,
    LSTMActivationHIndex = 2,
    LSTMActivationCount = 3
};

enum {
    LSTMPeepholeCountCiIndex = 0,
    LSTMPeepholeCountCoIndex = 1,
    LSTMPeepholeCountCfIndex = 2,
    LSTMPeepholeCount = 3
};

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#inputs-3---8
// size of weight/bias matrix is a multiple of hidden size
enum
{
    LSTMWeightDimensionHiddenMultiplier = 4,
    LSTMBiasDimensionHiddenMultiplier = 8
};

typedef enum {
    Forward,
    Backward,
} RNNDirection;

enum
{
    CNTKLSTMBiasIndex = 0,
    CNTKLSTMWeightIndex = 1,
    CNTKLSTMHiddenWeightIndex = 2
};

enum
{
    CNTKLSTMOutputYhIndex = 0,
    CNTKLSTMOutputChIndex = 1
};

enum
{
    GRUActivationFIndex = 0,
    GRUActivationGIndex = 1,
    GRUActivationCount = 2
};

enum
{
    GRUInputIndexX = 0,
    GRUInputIndexW = 1,
    GRUInputIndexR = 2,
    GRUInputIndexB = 3,
    GRUInputIndexSequenceLens = 4,
    GRUInitialH = 5,
};

enum
{
    RNNInputIndexX = 0,
    RNNInputIndexW = 1,
    RNNInputIndexR = 2,
    RNNInputIndexB = 3,
    RNNInputIndexSequenceLens = 4,
    RNNInitialH = 5,
};

enum
{
    CNTKRNNOutputYhIndex = 0
};

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#inputs-3---6
// size of weight/bias matrix is a multiple of hidden size
enum
{
    GRUWeightDimensionHiddenMultiplier = 3,
    GRUBiasDimensionHiddenMultiplier = 6
};

enum
{
    CNTKGRUZRWeightMultiplier = 2
};
enum
{
    CNTKGRUBiasIndex = 1,
    CNTKGRUWeightIndex = 2,
    CNTKGRUHiddenWeightZRIndex = 3,
    CNTKGRUHiddenWeightHIndex = 4,
    CNTKGRUPastOrFutureIndex = 5,
    CNTKGRUInputIndex = 6,
    CNTKGRUInputCount = 7
};

enum
{
    CNTKRNNWeightIndex = 0,
    CNTKRNNHweightIndex = 1,
    CNTKRNNBiasIndex = 2,
    CNTKRNNDelayIndex = 3,
    CNTKRNNInputIndex = 4,
    CNTKRNNInputCount = 5
};

enum
{
    RNNBiasMultiplier = 2
};

const string RNNDirectionBidirection = "bidirectional";
const string RNNDirectionReverse = "reverse";
const string RNNDirectionForward = "forward";

CNTK::FunctionPtr CreateLSTM(const onnxruntime::Node *node, const std::vector<CNTK::Variable> &inputs, const std::string &direction,
    const std::vector<std::string> &activations, const std::vector<float> &activation_alpha, const std::vector<float> &activation_beta,
    VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr);

CNTK::FunctionPtr CreateGRU(const onnxruntime::Node *node, const std::vector<CNTK::Variable> &inputs, const std::string &direction,
    const std::vector<string> &activations, const std::vector<float> &activation_alpha, const std::vector<float> &activation_beta,
    VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr);

CNTK::FunctionPtr CreateRNN(const onnxruntime::Node *node, const std::vector<CNTK::Variable> &inputs, const std::string &direction,
    const std::vector<string> &activations, const std::vector<float> &activation_alpha, const std::vector<float> &activation_beta,
    VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr);

void TraceLSTMPathes(const CNTK::FunctionPtr& src, string &f_activation, string &g_activation, string &h_activation,
    RNNDirection &direction, CNTK::Variable &initStateH, CNTK::Variable &initStateC, CNTK::Variable &peepholeCi, CNTK::Variable &peepholeCo, CNTK::Variable &peepholeCf,
    double &stabilizer_dh, double &stabilizer_dc, double &stabilizer_c);

void TraceGRUPathes(const CNTK::FunctionPtr& src, string &f_activation, string &g_activation,
    RNNDirection &direction, CNTK::Variable &initStateH);

void TraceRNNPathes(const CNTK::FunctionPtr& src, string &activation,
    RNNDirection &direction, CNTK::Variable &initStateH);

std::string MapActivationNameONNXToCNTK(const std::string &onnxOp);
std::string MapActivationNameCNTKToONNX(const std::string &cntkOp);

std::vector<CNTK::FunctionPtr> GetRNNBlocksFromSingleOrBidirectionalRNN(const CNTK::FunctionPtr src, const std::string &RNNStepOpName);

CNTK::Variable ToBatchAndSequence(CNTK::Variable input, VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr);

CNTK::FunctionPtr UnwrapRNNOps(CNTK::FunctionPtr rnnFunction, int numDirections);
CNTK::FunctionPtr UnpackBatchAndSequence(CNTK::FunctionPtr rnnFunction, bool doTranspose = true);