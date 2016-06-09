//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Basics.h"
#include "InputAndParamNodes.h"
#include "File.h"        // for LoadMatrixFromTextFile()
#include "TensorShape.h" // for SmallVector<>

#include <string>
namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType, class QuantizedType>
void LearnableParameterQuantized<ElemType, QuantizedType>::InitRandom(const bool uniformInit, const unsigned long randomSeed, const ElemType initValueScale, bool initOnCPUOnly) /*override*/
{
    LogicError("This operation is not supported");
}

template <class ElemType, class QuantizedType>
void LearnableParameterQuantized<ElemType, QuantizedType>::InitFromFile(const std::wstring& initFromFilePath) /*override*/
{
    LogicError("To be implemented");
}

template <class ElemType, class QuantizedType>
void LearnableParameterQuantized<ElemType, QuantizedType>::Save(File& fstream) const /*override*/
{
    LogicError("To be implemented");
}

template <class ElemType, class QuantizedType>
void LearnableParameterQuantized<ElemType, QuantizedType>::Load(File& fstream, size_t modelVersion) /*override*/
{
    LogicError("To be implemented");
}

template <class ElemType, class QuantizedType>
void LearnableParameterQuantized<ElemType, QuantizedType>::Validate(bool isFinalValidationPass) /*override*/
{
    LogicError("To be implemented");
}

template class LearnableParameterQuantized<float, short>;
template class LearnableParameterQuantized<double, short>;


}}}