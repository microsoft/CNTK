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
#include "Utils.h"

#include "proto/onnx/core/model.h"

#include <algorithm>
#include "CNTKLibrary.h"
#include <functional>

using namespace CNTK;
using namespace ONNXIR;

const std::string LSTMInputBiasNameHint = "_bias_";
const std::string LSTMInputInitialHNameHint = "_initial_h_";
const std::string LSTMInputInitialCNameHint = "_initial_c_";
const std::string LSTMInputPeepholeNameHint = "_peephole_";

const std::string GRUInputInitialHNameHint = "_initial_h_";

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#attributes-18
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#attributes-27
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#attributes-39
// CNTK RNN ops always output sequence. 
// ONNX requires to set the output_sequence attribute to 1 to output sequence. 
enum
{
    RNNOutputSequence = 1
};

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


const string RNNDirectionBidirection = "bidirectional";
const string RNNDirectionReverse = "reverse";
const string RNNDirectionForward = "forward";

FunctionPtr CreateLSTM(const ONNXIR::Node *node, const std::vector<Variable> &inputs, const std::string &direction,
    const std::vector<std::string> &activations, const std::vector<float> &activation_alpha, const std::vector<float> &activation_beta);

FunctionPtr CreateGRU(const ONNXIR::Node *node, const std::vector<Variable> &inputs, const std::string &direction,
    const std::vector<string> &activations, const std::vector<float> &activation_alpha, const std::vector<float> &activation_beta);

void TraceLSTMPathes(const FunctionPtr& src, string &f_activation, string &g_activation, string &h_activation,
    RNNDirection &direction, Variable &initStateH, Variable &initStateC, Variable &peepholeCi, Variable &peepholeCo, Variable &peepholeCf,
    double &stabilizer_dh, double &stabilizer_dc, double &stabilizer_c);

void TraceGRUPathes(const FunctionPtr& src, string &f_activation, string &g_activation,
    RNNDirection &direction, Variable &initStateH);

std::string MapActivationNameONNXToCNTK(const std::string &onnxOp);
std::string MapActivationNameCNTKToONNX(const std::string &cntkOp);