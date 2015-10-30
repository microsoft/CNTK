//
// <copyright file="ComputationNetwork.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ComputationNetwork.h"
#include "LinearAlgebraNodes.h"
#include "NonlinearityNodes.h"
#include "ConvolutionalNodes.h"
#include "RecurrentNodes.h"
#include "ReshapingNodes.h"
#include "TrainingCriterionNodes.h"
#include "CompositeComputationNodes.h"
#include "EvaluationCriterionNodes.h"
#include <string>
#include <set>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

}}}
