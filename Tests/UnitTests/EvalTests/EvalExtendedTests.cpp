//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "EvalTestHelper.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

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
    // Native model evaluation instance
    IEvaluateModelExtended<float> *eval;

    GetEvalExtendedF(&eval);

    try
    {
        eval->CreateNetwork(modelDefinition);
    }
    catch (std::exception& ex)
    {
        fprintf(stderr, "%s\n", ex.what());
        throw;
    }
    fflush(stderr);

    // Get the model's layers dimensions
    outputLayouts = eval->GetOutputSchema();

    for (auto vl : outputLayouts)
    {        
        fprintf(stderr, "Output dimension: %" PRIu64 "\n", vl.m_numElements);
        fprintf(stderr, "Output name: %ls\n", vl.m_name.c_str());
    }

    eval->StartForwardEvaluation({outputLayouts[0].m_name});
    inputLayouts = eval->GetInputSchema();
    outputLayouts = eval->GetOutputSchema();

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
    Values<float> outputBuffer = outputLayouts.CreateBuffers<float>({ 1 });

    // Allocate the input values layer (empty)
    Values<float> inputBuffer(0);

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
    Values<float> outputBuffer(0);

    // Allocate the input values layer
    Values<float> inputBuffer(1);
    inputBuffer[0].m_buffer = { 2 };
    
    // We can call the evaluate method and get back the results...
    BOOST_REQUIRE_THROW(eval->ForwardPass(inputBuffer, outputBuffer), std::exception); // Output not initialized

    outputBuffer = outputLayouts.CreateBuffers<float>({ 1 });
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
    auto outputBuffer = outputLayouts.CreateBuffers<float>({ 1 });

    // Allocate the input values layer
    Values<float> inputBuffer(1);
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
    Values<float> outputBuffer = outputLayouts.CreateBuffers<float>({ 1 });

    // Number of inputs must adhere to the schema
    Values<float> inputBuffer1(0);
    BOOST_REQUIRE_THROW(eval->ForwardPass(inputBuffer1, outputBuffer), std::exception); // Not enough inputs

    // Number of elements in the input must adhere to the schema
    Values<float> inputBuffer(1);
    inputBuffer[0].m_buffer = { 1, 2, 3 };
    BOOST_REQUIRE_THROW(eval->ForwardPass(inputBuffer, outputBuffer), std::exception); // Not enough elements in the sample

    // Output values and shape must be correct.
    inputBuffer[0].m_buffer = { 1, 2, 3, 4 };
    eval->ForwardPass(inputBuffer, outputBuffer);

    std::vector<float> expected{ 20 };
    auto buf = outputBuffer[0].m_buffer;
    BOOST_CHECK_EQUAL_COLLECTIONS(buf.begin(), buf.end(), expected.begin(), expected.end());

    // Do the same via ValueRefs
    ValueRefs<float> inputRefs(1);
    inputRefs[0].m_buffer.InitFrom(inputBuffer[0].m_buffer);
    inputRefs[0].m_colIndices.InitFrom(inputBuffer[0].m_colIndices);
    inputRefs[0].m_indices.InitFrom(inputBuffer[0].m_indices);
    ValueRefs<float> outputRefs(1);
    std::vector<float> output(1);
    outputRefs[0].m_buffer.InitFrom(output);
    eval->ForwardPass(inputRefs, outputRefs);
    BOOST_CHECK_EQUAL_COLLECTIONS(output.begin(), output.end(), expected.begin(), expected.end());

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
    Values<float> outputBuffer = outputLayouts.CreateBuffers<float>({ 3 });

    // Allocate the input values layer
    Values<float> inputBuffer(1);
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

    // Do the same via ValueRefs
    ValueRefs<float> inputRefs(1);
    inputRefs[0].m_buffer.InitFrom(inputBuffer[0].m_buffer);
    inputRefs[0].m_colIndices.InitFrom(inputBuffer[0].m_colIndices);
    inputRefs[0].m_indices.InitFrom(inputBuffer[0].m_indices);
    ValueRefs<float> outputRefs(1);
    std::vector<float> output(3);
    outputRefs[0].m_buffer.InitFrom(output);
    eval->ForwardPass(inputRefs, outputRefs);
    BOOST_CHECK_EQUAL_COLLECTIONS(output.begin(), output.end(), expected.begin(), expected.end());

    outputBuffer = outputLayouts.CreateBuffers<float>({ 1 });
    BOOST_REQUIRE_THROW(eval->ForwardPass(inputBuffer, outputBuffer), std::exception); // Not enough capacity in output.

    eval->Destroy();
}

BOOST_AUTO_TEST_CASE(EvalRNNTest)
{
    std::string modelDefinition =
        "deviceId = -1 \n"
        "precision = \"float\" \n"
        "traceLevel = 1 \n"
        "run=NDLNetworkBuilder \n"
        "NDLNetworkBuilder = [ \n"
        "LSTMComponent(inputDim, outputDim, cellDim, inputx, cellDimX2, cellDimX3, cellDimX4) = [ \n"
        "   wx = Parameter(cellDimX4, 0, init = \"uniform\", initValueScale = 1); \n"
        "   b = Parameter(cellDimX4, 1, init = \"fixedValue\", value = 0.0);\n"
        "   Wh = Parameter(cellDimX4, 0, init = \"uniform\", initValueScale = 1);\n"
        
        "   Wci = Parameter(cellDim, init = \"uniform\", initValueScale = 1);\n"
        "   Wcf = Parameter(cellDim, init = \"uniform\", initValueScale = 1);\n"
        "    Wco = Parameter(cellDim, init = \"uniform\", initValueScale = 1);\n"

        "   dh = PastValue(outputDim, output, timeStep = 1);\n"
        "   dc = PastValue(cellDim, ct, timeStep = 1);\n"

        "   wxx = Times(wx, inputx);\n"
        "   wxxpb = Plus(wxx, b);\n"

        "   whh = Times(wh, dh);\n"

        "   wxxpbpwhh = Plus(wxxpb, whh)\n"

        "       G1 = RowSlice(0, cellDim, wxxpbpwhh)\n"
        "       G2 = RowSlice(cellDim, cellDim, wxxpbpwhh)\n"
        "       G3 = RowSlice(cellDimX2, cellDim, wxxpbpwhh);\n"
        "   G4 = RowSlice(cellDimX3, cellDim, wxxpbpwhh);\n"

        "   Wcidc = DiagTimes(Wci, dc);\n"
        "   it = Sigmoid(Plus(G1, Wcidc));\n"

        "   bit = ElementTimes(it, Tanh(G2));\n"

        "   Wcfdc = DiagTimes(Wcf, dc);\n"
        "   ft = Sigmoid(Plus(G3, Wcfdc));\n"

        "   bft = ElementTimes(ft, dc);\n"

        "   ct = Plus(bft, bit);\n"

        "   Wcoct = DiagTimes(Wco, ct);\n"
        "   ot = Sigmoid(Plus(G4, Wcoct));\n"

        "   mt = ElementTimes(ot, Tanh(ct));\n"

        "   Wmr = Parameter(outputDim, cellDim, init = \"uniform\", initValueScale = 1);\n"
        "   output = Times(Wmr, mt); \n"
        "]\n"

        "i1 = Input(4) \n"
            "o1 = LSTMComponent(4, 4, 1, i1, 2, 3, 4) \n"
            "FeatureNodes = (i1) \n"
            "outputNodes = (o1) \n"
         "] \n";

    VariableSchema inputLayouts;
    VariableSchema outputLayouts;
    IEvaluateModelExtended<float> *eval;
    size_t featDim = 4;
    size_t labelDim = 4;
    eval = SetupNetworkAndGetLayouts(modelDefinition, inputLayouts, outputLayouts);

    // Allocate the output values layer
    Values<float> outputBuffer = outputLayouts.CreateBuffers<float>({ 1 });

    Values<float> inputBuffer(1);
    for (size_t i = 0; i < featDim;i++)
        inputBuffer[0].m_buffer.push_back((float)i);

    // the first pass with reset
    eval->ForwardPass(inputBuffer, outputBuffer);

    // the result is different on GCC. The root cause is in the model initialization (default_random_engine class), which is platform specific.
#ifdef _WIN32
    std::vector<int> expected = { -67, 135, -58, 178 };
#else
    std::vector<int> expected = { 0, 0, 0, 0 };
#endif

    int scaler = 100000;
    std::vector<int> result;
    for (size_t i = 0; i < labelDim; i++)
        result.push_back((int)(outputBuffer[0].m_buffer[i] * scaler));
   
    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), expected.begin(), expected.end());

    // the second pass with reset
    eval->ForwardPass(inputBuffer, outputBuffer);

    for (size_t i = 0; i < labelDim; i++)
        result[i] = (int)(outputBuffer[0].m_buffer[i] * scaler);

    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), expected.begin(), expected.end());

    // another pass with reset
    eval->ForwardPass(inputBuffer, outputBuffer, true);

    for (size_t i = 0; i < labelDim; i++)
        result[i] = (int)(outputBuffer[0].m_buffer[i] * scaler);

    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), expected.begin(), expected.end());

    // pass w/o reset
    eval->ForwardPass(inputBuffer, outputBuffer, false);
    for (size_t i = 0; i < labelDim; i++)
        result[i] = (int)(outputBuffer[0].m_buffer[i] * scaler);

#ifdef _WIN32
    expected = { -63, 126, -54, 166 };
#else
    expected = { 109, -63, -7, -55 };
#endif

    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), expected.begin(), expected.end());

    // another pass w/o reset
    eval->ForwardPass(inputBuffer, outputBuffer, false);
    for (size_t i = 0; i < labelDim; i++)
        result[i] = (int)(outputBuffer[0].m_buffer[i] * scaler);

#ifdef _WIN32
    expected = { -61, 122, -52, 161 };
#else
    expected = { 158, -91, -10, -79 };
#endif

    
    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), expected.begin(), expected.end());

    eval->Destroy();
}

BOOST_AUTO_TEST_SUITE_END()
}}}}