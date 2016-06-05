//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "EvalTestHelper.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

// Used for retrieving the model appropriate for the element type (float / double)
template<typename ElemType>
using GetEvalProc = void(*)(IEvaluateModelExtended<ElemType>**);

typedef std::pair<std::wstring, std::vector<float>*> Variable;
typedef std::map<std::wstring, std::vector<float>*> Variables;

BOOST_FIXTURE_TEST_SUITE(EvalTestSuite, EvalFixture)

IEvaluateModelExtended<float>* SetupNetworkAndGetLayouts(std::string modelDefinition, VariableSchema& inputLayouts, VariableSchema& outputLayouts)
{
    // Load the eval library
    auto hModule = LoadLibrary(L"evaldll.dll");
    if (hModule == nullptr)
    {
        auto err = GetLastError();
        throw std::exception((boost::format("Cannot load evaldll.dll: 0x%08lx") % err).str().c_str());
    }

    // Get the factory method to the evaluation engine
    std::string func = "GetEvalExtendedF";
    auto procAddress = GetProcAddress(hModule, func.c_str());
    auto getEvalProc = (GetEvalProc<float>)procAddress;

    // Native model evaluation instance
    IEvaluateModelExtended<float> *eval;
    getEvalProc(&eval);

    try
    {
        eval->CreateNetwork(modelDefinition);
    }
    catch (std::exception& ex)
    {
        fprintf(stderr, ex.what());
        throw;
    }
    fflush(stderr);

    // Get the model's layers dimensions
    outputLayouts = eval->GetOutputSchema();

    for (auto vl : outputLayouts)
    {
        fprintf(stderr, "Output dimension: %d\n", vl.m_numElements);
        fprintf(stderr, "Output name: %ls\n", vl.m_name.c_str());
    }

    eval->StartForwardEvaluation({outputLayouts[0].m_name});
    inputLayouts = eval->GetInputSchema();

    return eval;
}

BOOST_AUTO_TEST_CASE(EvalConstantPlusTest)
{
    // Setup model for adding two constants (1 + 2)
    std::string modelDefinition =
        "deviceId = -1 \n"
        "precision = \"float\" \n"
        "traceLevel = 1 \n"
        "run=NDLNetworkBuilder \n"
        "NDLNetworkBuilder=[ \n"
        "v1 = Constant(1) \n"
        "v2 = Constant(2) \n"
        "ol = Plus(v1, v2, tag=\"output\") \n"
        "FeatureNodes = (v1) \n"
        "] \n";
    
    VariableSchema inputLayouts;
    VariableSchema outputLayouts;
    IEvaluateModelExtended<float> *eval;
    eval = SetupNetworkAndGetLayouts(modelDefinition, inputLayouts, outputLayouts);

    // Allocate the output values layer
    std::vector<VariableBuffer<float>> outputBuffer(1);

    // Allocate the input values layer (empty)

    std::vector<VariableBuffer<float>> inputBuffer;

    // We can call the evaluate method and get back the results...
    eval->ForwardPass(inputBuffer, outputBuffer);

    std::vector<float> expected{ 3 /* 1 + 2 */ };
    auto buf = outputBuffer[0].m_buffer;
    BOOST_CHECK_EQUAL_COLLECTIONS(buf.begin(), buf.end(), expected.begin(), expected.end());

    eval->Destroy();
}

BOOST_AUTO_TEST_CASE(EvalScalarTimesTest)
{
    std::string modelDefinition =
        "deviceId = -1 \n"
        "precision = \"float\" \n"
        "traceLevel = 1 \n"
        "run=NDLNetworkBuilder \n"
        "NDLNetworkBuilder=[ \n"
        "i1 = Input(1) \n"
        "o1 = Times(Constant(3), i1, tag=\"output\") \n"
        "FeatureNodes = (i1) \n"
        "] \n";

    VariableSchema inputLayouts;
    VariableSchema outputLayouts;
    IEvaluateModelExtended<float> *eval;
    eval = SetupNetworkAndGetLayouts(modelDefinition, inputLayouts, outputLayouts);

    // Allocate the output values layer
    std::vector<VariableBuffer<float>> outputBuffer(1);

    // Allocate the input values layer
    std::vector<VariableBuffer<float>> inputBuffer(1);
    inputBuffer[0].m_buffer = { 2 };
    
    // We can call the evaluate method and get back the results...
    eval->ForwardPass(inputBuffer, outputBuffer);

    std::vector<float> expected{ 6 };
    auto buf = outputBuffer[0].m_buffer;
    BOOST_CHECK_EQUAL_COLLECTIONS(buf.begin(), buf.end(), expected.begin(), expected.end());

    eval->Destroy();
}

BOOST_AUTO_TEST_CASE(EvalScalarTimesDualOutputTest)
{
    std::string modelDefinition =
        "deviceId = -1 \n"
        "precision = \"float\" \n"
        "traceLevel = 1 \n"
        "run=NDLNetworkBuilder \n"
        "NDLNetworkBuilder=[ \n"
        "i1 = Input(1) \n"
        "i2 = Input(1) \n"
        "o1 = Times(Constant(3), i1, tag=\"output\") \n"
        "o2 = Times(Constant(5), i1, tag=\"output\") \n"
        "FeatureNodes = (i1) \n"
        "] \n";

    VariableSchema inputLayouts;
    VariableSchema outputLayouts;
    IEvaluateModelExtended<float> *eval;
    eval = SetupNetworkAndGetLayouts(modelDefinition, inputLayouts, outputLayouts);

    // Allocate the output values layer
    std::vector<VariableBuffer<float>> outputBuffer(1);

    // Allocate the input values layer
    std::vector<VariableBuffer<float>> inputBuffer(1);
    inputBuffer[0].m_buffer = { 2 };

    // We can call the evaluate method and get back the results...
    // TODO: Indicate to ForwardPass that we want output o2 only
    eval->ForwardPass(inputBuffer, outputBuffer);

    std::vector<float> expected{ 6 };
    auto buf = outputBuffer[0].m_buffer;
    BOOST_CHECK_EQUAL_COLLECTIONS(buf.begin(), buf.end(), expected.begin(), expected.end());

    eval->Destroy();
}

BOOST_AUTO_TEST_CASE(EvalDenseTimesTest)
{
    std::string modelDefinition =
        "deviceId = -1 \n"
        "precision = \"float\" \n"
        "traceLevel = 1 \n"
        "run=NDLNetworkBuilder \n"
        "NDLNetworkBuilder=[ \n"
        "i1 = Input(4) \n"
        "o1 = Times(Constant(2, rows=1, cols=4), i1, tag=\"output\") \n"
        "FeatureNodes = (i1) \n"
        "] \n";

    VariableSchema inputLayouts;
    VariableSchema outputLayouts;
    IEvaluateModelExtended<float> *eval;
    eval = SetupNetworkAndGetLayouts(modelDefinition, inputLayouts, outputLayouts);

    // Allocate the output values layer
    std::vector<VariableBuffer<float>> outputBuffer(1);

    // Number of inputs must adhere to the schema
    std::vector<VariableBuffer<float>> inputBuffer1(0);
    BOOST_REQUIRE_THROW(eval->ForwardPass(inputBuffer1, outputBuffer), std::exception); // Not enough inputs

    // Number of elements in the input must adhere to the schema
    std::vector<VariableBuffer<float>> inputBuffer(1);
    inputBuffer[0].m_buffer = { 1, 2, 3 };
    BOOST_REQUIRE_THROW(eval->ForwardPass(inputBuffer, outputBuffer), std::exception); // Not enough elements in the sample

    // Output values and shape must be correct.
    inputBuffer[0].m_buffer = { 1, 2, 3, 4 };
    eval->ForwardPass(inputBuffer, outputBuffer);

    std::vector<float> expected{ 20 };
    auto buf = outputBuffer[0].m_buffer;
    BOOST_CHECK_EQUAL_COLLECTIONS(buf.begin(), buf.end(), expected.begin(), expected.end());

    eval->Destroy();
}

BOOST_AUTO_TEST_CASE(EvalSparseTimesTest)
{
    std::string modelDefinition =
        "deviceId = -1 \n"
        "precision = \"float\" \n"
        "traceLevel = 1 \n"
        "run=NDLNetworkBuilder \n"
        "NDLNetworkBuilder=[ \n"
        "i1 = SparseInput(3) \n"
        "o1 = Times(Constant(2, rows=1, cols=3), i1, tag=\"output\") \n"
        "FeatureNodes = (i1) \n"
        "] \n";

    VariableSchema inputLayouts;
    VariableSchema outputLayouts;
    IEvaluateModelExtended<float> *eval;
    eval = SetupNetworkAndGetLayouts(modelDefinition, inputLayouts, outputLayouts);

    // Allocate the output values layer
    std::vector<VariableBuffer<float>> outputBuffer(1);

    // Allocate the input values layer
    std::vector<VariableBuffer<float>> inputBuffer(1);
    inputBuffer[0].m_buffer = {1, 2, 3, 5, 6};
    inputBuffer[0].m_indices = {0, 2, 2, 1, 2};

    inputBuffer[0].m_colIndices = {};
    BOOST_REQUIRE_THROW(eval->ForwardPass(inputBuffer, outputBuffer), std::exception); // Empty input

    inputBuffer[0].m_colIndices = { 0 };
    BOOST_REQUIRE_THROW(eval->ForwardPass(inputBuffer, outputBuffer), std::exception); // Empty input

    inputBuffer[0].m_colIndices = { 1, 0 };
    BOOST_REQUIRE_THROW(eval->ForwardPass(inputBuffer, outputBuffer), std::exception); // Illegal: First entry must be 0

    inputBuffer[0].m_colIndices = { 0, 2, 2, 4 };
    BOOST_REQUIRE_THROW(eval->ForwardPass(inputBuffer, outputBuffer), std::exception); // Illegal: Last entry must be indices.size()

    inputBuffer[0].m_colIndices = { 0, 2, 2, 5 };

    // We can call the evaluate method and get back the results...
    eval->ForwardPass(inputBuffer, outputBuffer);
    
    // [2,2,2] * [1,2,3]^T etc.
    std::vector<float> expected{ 6, 0, 28 };
    auto buf = outputBuffer[0].m_buffer;
    BOOST_CHECK_EQUAL_COLLECTIONS(buf.begin(), buf.end(), expected.begin(), expected.end());
    eval->Destroy();
}

BOOST_AUTO_TEST_SUITE_END()
}}}}