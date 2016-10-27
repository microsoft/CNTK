#include "stdafx.h"

#include <memory>
#include "../../../Source/ComputationNetworkLib/ReshapingNodes.h"

using namespace Microsoft::MSR::CNTK;
using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

// We perform test on CPU since there is nothing device specific oin crop node.
const DEVICEID_TYPE c_deviceId = CPUDEVICE;

// Helper dummy node to be used as input to crop node.
template <class ElemType>
class DummyNodeTest : public ComputationNode<ElemType>
{
public:
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"DummyTest"; }

    DummyNodeTest(SmallVector<size_t> shapeVec) : Base(c_deviceId, L"Dummy")
    {
        // Set given shape and allocate matrices.
        TensorShape shape(shapeVec);
        this->SetDims(shape, false);
        this->CreateValueMatrixIfNull();
        this->Value().Resize(1, shape.GetNumElements());
        this->CreateGradientMatrixIfNull();
        this->Gradient().Resize(1, shape.GetNumElements());
    }
    DummyNodeTest(DEVICEID_TYPE deviceId, const wstring& name) : Base(deviceId, name) {}

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& /*fr*/) override {}

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange& /*fr*/) override {}

    Matrix<ElemType>& GetGradient() { return this->Gradient(); }
};

// Extends crop node to provide acces to protected members.
template <class ElemType>
class CropNodeTest : public CropNode<ElemType>
{
public:
    CropNodeTest() : CropNode<ElemType>(0, L"CropNodeTest"){}

    int OffsetX() { return this->m_xOffset; }
    int OffsetY() { return this->m_yOffset; }
    SmallVector<size_t> GetOutputDims() { return this->GetSampleLayout().GetDims(); }
    void AllocMatrices()
    {
        this->CreateValueMatrixIfNull();
        this->CreateGradientMatrixIfNull();
        this->Value().Resize(1, this->GetSampleLayout().GetNumElements());
        this->Gradient().Resize(1, this->GetSampleLayout().GetNumElements());
    }
    Matrix<ElemType>& GetGradient()
    {
        return this->Gradient();
    }
};

template<class ElemType>
void CropNodeValidateTestImpl()
{
    {
        // Test that validation fails if cropping cannot be done in x direction.
        auto cropNode = make_shared<CropNode<ElemType>>(6, 3, c_deviceId, L"CropNode");
        auto cropNodeTest = make_shared<CropNodeTest<ElemType>>();
        cropNode->CopyTo(cropNodeTest, cropNodeTest->GetName(), CopyNodeFlags::copyNodeValue);

        // 6 + 5 > 10 (offset + crop > input) -> cropping not possible in x direction.
        SmallVector<size_t> firstInputDims = { 10, 10 };
        SmallVector<size_t> secondInputDims = { 5, 5 };
        auto firstInput = make_shared<DummyNodeTest<ElemType>>(firstInputDims);
        auto secondInput = make_shared<DummyNodeTest<ElemType>>(secondInputDims);
        vector<ComputationNodeBasePtr> inputs = { firstInput, secondInput };
        cropNodeTest->AttachInputs(inputs);
        BOOST_REQUIRE_EXCEPTION(
            cropNodeTest->Validate(true),
            std::runtime_error,
            [](std::runtime_error const& ex) { return string("Input is small to be cropped along x dimension in crop node.") == ex.what(); }
        );
    }
    {
        // Test that validation fails if cropping cannot be done in y direction.
        auto cropNode = make_shared<CropNode<ElemType>>(3, 7, c_deviceId, L"CropNode");
        auto cropNodeTest = make_shared<CropNodeTest<ElemType>>();
        cropNode->CopyTo(cropNodeTest, cropNodeTest->GetName(), CopyNodeFlags::copyNodeValue);

        // 7 + 5 > 10 (offset + crop > input) -> cropping not possible in y direction.
        SmallVector<size_t> firstInputDims = { 10, 10 };
        SmallVector<size_t> secondInputDims = { 5, 5 };
        auto firstInput = make_shared<DummyNodeTest<ElemType>>(firstInputDims);
        auto secondInput = make_shared<DummyNodeTest<ElemType>>(secondInputDims);
        vector<ComputationNodeBasePtr> inputs = { firstInput, secondInput };
        cropNodeTest->AttachInputs(inputs);
        BOOST_REQUIRE_EXCEPTION(
            cropNodeTest->Validate(true),
            std::runtime_error,
            [](std::runtime_error const& ex) { return string("Input is small to be cropped along y dimension in crop node.") == ex.what(); }
        );
    }

    {
        // Test that crop node output is same size as second input after validation.
        auto cropNode = make_shared<CropNode<ElemType>>(3, 3, c_deviceId, L"CropNode");
        auto cropNodeTest = make_shared<CropNodeTest<ElemType>>();
        cropNode->CopyTo(cropNodeTest, cropNodeTest->GetName(), CopyNodeFlags::copyNodeValue);

        SmallVector<size_t> firstInputDims = { 10, 10 };
        SmallVector<size_t> secondInputDims = { 5, 5 };
        auto firstInput = make_shared<DummyNodeTest<ElemType>>(firstInputDims);
        auto secondInput = make_shared<DummyNodeTest<ElemType>>(secondInputDims);
        vector<ComputationNodeBasePtr> inputs = { firstInput, secondInput };
        cropNodeTest->AttachInputs(inputs);
        cropNodeTest->Validate(true);
        SmallVector<size_t> outputDims = cropNodeTest->GetOutputDims();

        BOOST_REQUIRE_MESSAGE(outputDims == secondInputDims, "Crop node output differs from its second input");
    }
}

template<class ElemType>
void CropNodeForwardTestImpl()
{
    // Test that input is correctly cropped.
    auto cropNode = make_shared<CropNode<ElemType>>(1, 1, c_deviceId, L"CropNode");
    auto cropNodeTest = make_shared<CropNodeTest<ElemType>>();
    cropNode->CopyTo(cropNodeTest, cropNodeTest->GetName(), CopyNodeFlags::copyNodeValue);

    SmallVector<size_t> firstInputDims = { 4, 4 };
    SmallVector<size_t> secondInputDims = { 2, 2 };
    auto firstInput = make_shared<DummyNodeTest<ElemType>>(firstInputDims);
    auto secondInput = make_shared<DummyNodeTest<ElemType>>(secondInputDims);

    Matrix<ElemType>& input = firstInput->Value();
    ElemType inputVals[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    input.SetValue(4, 4, c_deviceId, inputVals);

    vector<ComputationNodeBasePtr> inputs = { firstInput, secondInput };
    cropNodeTest->AttachInputs(inputs);
    cropNodeTest->Validate(true);
    cropNodeTest->AllocMatrices();

    FrameRange fr;
    cropNodeTest->ForwardProp(fr);
    ElemType* outputData = cropNodeTest->Value().Data();
    BOOST_REQUIRE_MESSAGE(outputData[0] == inputVals[5], "Cropping output is invalid");
    BOOST_REQUIRE_MESSAGE(outputData[1] == inputVals[6], "Cropping output is invalid");
    BOOST_REQUIRE_MESSAGE(outputData[2] == inputVals[9], "Cropping output is invalid");
    BOOST_REQUIRE_MESSAGE(outputData[3] == inputVals[10], "Cropping output is invalid");
}

template<class ElemType>
void CropNodeBackwardTestImpl()
{
    // Test that gradients are correctly propagated.
    auto cropNode = make_shared<CropNode<ElemType>>(1, 1, c_deviceId, L"CropNode");
    auto cropNodeTest = make_shared<CropNodeTest<ElemType>>();
    cropNode->CopyTo(cropNodeTest, cropNodeTest->GetName(), CopyNodeFlags::copyNodeValue);

    SmallVector<size_t> firstInputDims = { 4, 4 };
    SmallVector<size_t> secondInputDims = { 2, 2 };
    auto firstInput = make_shared<DummyNodeTest<ElemType>>(firstInputDims);
    auto secondInput = make_shared<DummyNodeTest<ElemType>>(secondInputDims);

    vector<ComputationNodeBasePtr> inputs = { firstInput, secondInput };
    cropNodeTest->AttachInputs(inputs);
    cropNodeTest->Validate(true);
    cropNodeTest->AllocMatrices();
    Matrix<ElemType>& outputGrad = cropNodeTest->GetGradient();
    ElemType outputGradVals[4] = { 0, 1, 2, 3 };
    outputGrad.SetValue(2, 2, c_deviceId, outputGradVals);

    FrameRange fr;
    cropNodeTest->BackpropTo(0, fr);
    ElemType* input0GradVals = firstInput->GetGradient().Data();
    BOOST_REQUIRE_MESSAGE(input0GradVals[0] == 0, "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[1] == 0, "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[2] == 0, "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[3] == 0, "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[4] == 0, "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[5] == outputGradVals[0], "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[6] == outputGradVals[1], "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[7] ==0, "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[8] == 0, "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[9] == outputGradVals[2], "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[10] == outputGradVals[3], "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[11] == 0, "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[12] == 0, "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[13] == 0, "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[14] == 0, "Cropping gradient is invalid");
    BOOST_REQUIRE_MESSAGE(input0GradVals[15] == 0, "Cropping gradient is invalid");

    // Test that gradients are not propagated to second input.
    ElemType secondInputGradValue = 10;
    secondInput->GetGradient().SetValue(secondInputGradValue);
    cropNodeTest->BackpropTo(1, fr);
    ElemType* input1GradVals = secondInput->GetGradient().Data();
    for (int i = 0; i < 4; i++)
    {
        BOOST_REQUIRE_MESSAGE(input1GradVals[i] == secondInputGradValue, "Cropping output is invalid");
    }
}

BOOST_AUTO_TEST_SUITE(CropNodeTestSuite)

BOOST_AUTO_TEST_CASE(CropNodeValidateTest)
{
    CropNodeValidateTestImpl<float>();
    CropNodeValidateTestImpl<double>();
}

BOOST_AUTO_TEST_CASE(CropNodeForwardTest)
{
    CropNodeForwardTestImpl<float>();
    CropNodeForwardTestImpl<double>();
}

BOOST_AUTO_TEST_CASE(CropNodeBackwardTest)
{
    CropNodeBackwardTestImpl<float>();
    CropNodeBackwardTestImpl<double>();
}

BOOST_AUTO_TEST_SUITE_END()

} } } }