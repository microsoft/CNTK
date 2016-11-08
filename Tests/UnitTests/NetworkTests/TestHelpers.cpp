//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "TestHelpers.h"

using namespace Microsoft::MSR::CNTK;
using namespace Microsoft::MSR::CNTK::Test;

template <class ElemType>
bool Microsoft::MSR::CNTK::Test::AreEqual(const ElemType* a, const ElemType* b, const size_t count, const float threshold)
{
    for (size_t i = 0; i < count; i++)
    {
        if (abs(a[i] - b[i]) >= threshold)
            return false;
    }
    return true;
}

template bool Microsoft::MSR::CNTK::Test::AreEqual<float>(const float* a, const float* b, const size_t count,
                                                          const float threshold);

template bool Microsoft::MSR::CNTK::Test::AreEqual<double>(const double* a, const double* b, const size_t count,
                                                           const float threshold);

template <class ElemType>
/*static*/ const std::wstring DummyNodeTest<ElemType>::TypeName()
{
    return L"DummyTest";
}

template <class ElemType>
DummyNodeTest<ElemType>::DummyNodeTest(DEVICEID_TYPE deviceId, size_t minibatchSize,
                                       SmallVector<size_t> sampleDimensions, std::vector<ElemType>& data)
    : Base(deviceId, L"Dummy")
{
    // Set given shape and allocate matrices.
    SetMinibatch(minibatchSize, sampleDimensions, data);
}

template <class ElemType>
DummyNodeTest<ElemType>::DummyNodeTest(DEVICEID_TYPE deviceId, const wstring& name) : Base(deviceId, name)
{
}

template <class ElemType>
Matrix<ElemType>& DummyNodeTest<ElemType>::GetGradient()
{
    return this->Gradient();
}

template <class ElemType>
void DummyNodeTest<ElemType>::SetMinibatch(size_t minibatchSize, SmallVector<size_t> sampleDimensions,
                                               std::vector<ElemType>& data)
{
    // Set given shape and allocate matrices.
    TensorShape shape(sampleDimensions);
    if (shape.GetNumElements() * minibatchSize != data.size())
        LogicError("Data size is incompatible with specified dimensions.");
    MBLayoutPtr mbLayout = make_shared<MBLayout>();
    mbLayout->InitAsFrameMode(minibatchSize);
    this->LinkToMBLayout(mbLayout);
    this->SetDims(shape, true);
    this->CreateValueMatrixIfNull();
    this->Value().Resize(shape.GetNumElements(), minibatchSize);
    this->CreateGradientMatrixIfNull();
    this->Gradient().Resize(shape.GetNumElements(), minibatchSize);
    this->Value().SetValue(minibatchSize, shape.GetNumElements(), this->m_deviceId, data.data());
}

template class DummyNodeTest<float>;
template class DummyNodeTest<double>;