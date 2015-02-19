//
// <copyright file="ComputationNode.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ComputationNode.h"
#include "SimpleEvaluator.h"
#include "IComputationNetBuilder.h"
#include "SGD.h"

namespace Microsoft {
    namespace MSR {
        namespace CNTK {

    typedef Matrix<float> FloatMatrix;
    typedef Matrix<double> DoubleMatrix;

    template<> int64_t ComputationNode<float>::s_timeStampCounter = 0;
    template<> int64_t ComputationNode<double>::s_timeStampCounter = 0;

    template<> std::map<size_t, std::map<size_t, FloatMatrix*>> ComputationNode<float>::s_constOnes;
    template<> std::map<size_t, std::map<size_t, DoubleMatrix*>> ComputationNode<double>::s_constOnes;

    template class LearnableParameter<float>;
    template class LearnableParameter<double>;
}}}
