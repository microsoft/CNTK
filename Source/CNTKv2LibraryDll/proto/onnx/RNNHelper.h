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

typedef enum {
    Forward,
    Backward,
} LSTMDirection;

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
const string LSTMDirectionBidirection = "bidirectional";
const string LSTMDirectionReverse = "reverse";
const string LSTMDirectionForward = "forward";

FunctionPtr CreateLSTM(const ONNXIR::Node *node, const std::vector<Variable> &inputs, const std::string &direction,
    const std::vector<std::string> &activations, const std::vector<float> &activation_alpha, const std::vector<float> &activation_beta);
