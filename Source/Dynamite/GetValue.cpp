//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "GetValue.h"
#include "CNTKLibrary.h"
#include "Variable.h"
#include "PrimitiveOpType.h"
#include "PrimitiveFunction.h"
#include "CommonMatrix.h"

#include <unordered_map>

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#pragma warning (disable: 4456) // until I fixed the shdowing

#define let const auto
#define fail_if(cond, err) (!!(cond) ? (LogicError(__FUNCTION__ ": " err),0) : 0)
#define BreakPoint fprintf(stderr, "") // use this inside a conditional to be able to set a breakpoint in Release code

using namespace std;

namespace CNTK
{
} // namespace

// this will become Variable::Value()
// Computes lazily the value of a node. Does nothing if called again.
//CNTK::NDArrayViewPtr GetValue(const CNTK::Variable& v)
//{
//#if 0
//    // naive version for comparison purposes
//    return v.Value();
//#else
//    auto autoBatcher = CNTK::Memoize();
//    return autoBatcher.GetValue(v);
//#endif
//}
//
//// Perform backprop.
//// CNTK grad() allows to pass multiple roots. Does that ever make sense in this context?
//void Backward(const CNTK::Variable& root, std::unordered_map<CNTK::Parameter, CNTK::NDArrayViewPtr>& gradients)
//{
//    auto autoBatcher = CNTK::Memoize(); // has some internal state
//    autoBatcher.Backward(root, gradients);
//}
