//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"

#include "../../../Source/ComputationNetworkLib/LinearAlgebraNodes.h"
#include "TestHelpers.h"
#include <memory>

using namespace Microsoft::MSR::CNTK;
using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

// We perform test on CPU.
const DEVICEID_TYPE c_deviceId = CPUDEVICE;

static const float c_epsilonFloatE4 = 0.0001f;

// Extends epoch accumulator node to provide access to protected members.
template <class ElemType>
class EpochAccumulatorNodeTest : public EpochAccumulatorNode<ElemType>
{
public:
    EpochAccumulatorNodeTest(DEVICEID_TYPE deviceId)
        : EpochAccumulatorNode<ElemType>(deviceId, L"EpochAccumulatorNodeTest")
    {
    }

    void ForwardPass(bool isEpochStart = false)
    {
        this->EnsureMatricesAreAllocated();

        if (isEpochStart)
            this->OnEpochStart(); // Resets accumulator state.

        FrameRange fr;
        this->BeginForwardProp();
        this->ForwardProp(fr);
        this->EndForwardProp();
    }

    bool IsOutputEqualTo(const std::vector<ElemType>& output) const
    {
        return AreEqual(output.data(), this->Value().Data(), output.size(), c_epsilonFloatE4);
    }

private:
    void EnsureMatricesAreAllocated()
    {
        this->CreateValueMatrixIfNull();
        this->CreateGradientMatrixIfNull();
        if (this->m_accumulator == nullptr)
        {
            // This is done in RequestMatricesBeforeForwardProp, but here we don't have matrix pool available.
            this->CreateMatrixIfNull(this->m_accumulator);
            this->m_accumulator->Resize(1, this->GetSampleLayout().GetNumElements());
            this->Reset();
        }
    }
};

template <class ElemType>
void EpochAccumulatorNodeForwardTestImpl()
{
    // Test that single forward propagation works.
    auto accumulatorNode = make_shared<EpochAccumulatorNodeTest<ElemType>>(c_deviceId);

    const size_t minibatchSize = 4;
    const SmallVector<size_t> sampleDimensions{4};
    std::vector<ElemType> inputValues{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    auto input = make_shared<DummyNodeTest<ElemType>>(c_deviceId, minibatchSize, sampleDimensions, inputValues);
    accumulatorNode->AttachInputs({input});
    accumulatorNode->Validate(true);

    accumulatorNode->ForwardPass();

    const std::vector<ElemType> expectedOutput{6, 7, 8, 9};
    BOOST_REQUIRE_MESSAGE(accumulatorNode->IsOutputEqualTo(expectedOutput), "Accumulator output is invalid");
}

template <class ElemType>
void EpochAccumulatorNodeMultipleForwardTestImpl()
{
    // Test that single forward propagation works.
    auto accumulatorNode = make_shared<EpochAccumulatorNodeTest<ElemType>>(c_deviceId);
    auto input = make_shared<DummyNodeTest<ElemType>>(c_deviceId, L"Input");
    accumulatorNode->AttachInputs({input});

    const SmallVector<size_t> sampleDimensions{4};

    {
        // Do first forward pass.
        const size_t minibatchSize = 4;
        std::vector<ElemType> minibatchData{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        input->SetMinibatch(minibatchSize, sampleDimensions, minibatchData);
        accumulatorNode->Validate(true);

        accumulatorNode->ForwardPass();

        const std::vector<ElemType> expectedOutput{6, 7, 8, 9};
        BOOST_REQUIRE_MESSAGE(accumulatorNode->IsOutputEqualTo(expectedOutput), "Accumulator output is invalid");
    }

    {
        // Do second forward pass.
        const size_t minibatchSize = 2;
        std::vector<ElemType> minibatchData{16, 17, 18, 19, 20, 21, 22, 23};
        input->SetMinibatch(minibatchSize, sampleDimensions, minibatchData);

        accumulatorNode->ForwardPass();

        const std::vector<ElemType> expectedOutput{10, 11, 12, 13};
        BOOST_REQUIRE_MESSAGE(accumulatorNode->IsOutputEqualTo(expectedOutput), "Accumulator output is invalid");
    }
}

template <class ElemType>
void EpochAccumulatorNodeMultipleForwardWithEpochResetTestImpl()
{
    // Test that multiple forward propagations with resetting accumulator work.
    auto accumulatorNode = make_shared<EpochAccumulatorNodeTest<ElemType>>(c_deviceId);
    auto input = make_shared<DummyNodeTest<ElemType>>(c_deviceId, L"Input");
    accumulatorNode->AttachInputs({input});

    const bool c_isEpochStart = true;
    const SmallVector<size_t> sampleDimensions{4};

    {
        // Do first forward pass.
        const size_t minibatchSize = 4;
        std::vector<ElemType> minibatchData{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        input->SetMinibatch(minibatchSize, sampleDimensions, minibatchData);
        accumulatorNode->Validate(true);

        accumulatorNode->ForwardPass(c_isEpochStart); // Reset accumulator state before forward pass.

        const std::vector<ElemType> expectedOutput{6, 7, 8, 9};
        BOOST_REQUIRE_MESSAGE(accumulatorNode->IsOutputEqualTo(expectedOutput), "Accumulator output is invalid");
    }

    {
        // Do second forward pass.
        const size_t minibatchSize = 2;
        std::vector<ElemType> minibatchData{16, 17, 18, 19, 20, 21, 22, 23};
        input->SetMinibatch(minibatchSize, sampleDimensions, minibatchData);

        accumulatorNode->ForwardPass(c_isEpochStart); // Reset accumulator state before forward pass.

        const std::vector<ElemType> expectedOutput{18, 19, 20, 21};
        BOOST_REQUIRE_MESSAGE(accumulatorNode->IsOutputEqualTo(expectedOutput), "Accumulator output is invalid");
    }
}

BOOST_AUTO_TEST_SUITE(EpochAccumulatorNodeTestSuite)

BOOST_AUTO_TEST_CASE(EpochAccumulatorNodeForwardTest)
{
    EpochAccumulatorNodeForwardTestImpl<float>();
    EpochAccumulatorNodeForwardTestImpl<double>();
}

BOOST_AUTO_TEST_CASE(EpochAccumulatorNodeMultipleForwardTest)
{
    EpochAccumulatorNodeMultipleForwardTestImpl<float>();
    EpochAccumulatorNodeMultipleForwardTestImpl<double>();
}

BOOST_AUTO_TEST_CASE(EpochAccumulatorNodeMultipleForwardWithEpochResetTest)
{
    EpochAccumulatorNodeMultipleForwardWithEpochResetTestImpl<float>();
    EpochAccumulatorNodeMultipleForwardWithEpochResetTestImpl<double>();
}

BOOST_AUTO_TEST_SUITE_END()
} } } }