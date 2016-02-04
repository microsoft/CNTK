//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "ScriptableObjects.h"
#include "Matrix.h"
#include "File.h" // for LoadMatrixFromTextFile()

#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <algorithm>
#include <assert.h>

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// LearnableParameter (/*no input*/)
// represents weight matrices and biases
// TODO: add -Node to the class name
// -----------------------------------------------------------------------

template <class ElemType>
class LearnableParameter : public ComputationNode<ElemType>, public NumInputs<0>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"LearnableParameter";
    }

public:
    LearnableParameter(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
        m_parameterUpdateRequired = true;
        this->m_valueSharable = false;
    }
    LearnableParameter(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& shape)
        : Base(deviceId, name)
    {
        m_parameterUpdateRequired = true;
        CreateMatrixIfNull(m_value);
        m_valueSharable = false;
        SetDims(shape, false);
        UpdateFunctionValuesSize(); // this allocates the matrix
        Value().SetValue(0);
    }
    LearnableParameter(DEVICEID_TYPE deviceId, const wstring& name, size_t rows, size_t cols)
        : LearnableParameter(deviceId, name, TensorShape(rows, cols))
    {
    }
    LearnableParameter(const ScriptableObjects::IConfigRecordPtr configp)
        : LearnableParameter(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"shape"))
    {
        // TODO: Change dimensions to take a generic tensor instead. That will be a (minor) breaking change that will require fix-ups when converting from NDL to BrainScript.
        AttachInputs(configp, this->GetExpectedNumInputs());
        // parameters[rows, [cols=1]] plus other optional parameters (needGradient=[true|false], init=[uniform|gaussian|fixedvalue], initValueScale=[1|float], value=[0|float])
        // TODO: "needGradient" should be renamed to better match m_parameterUpdateRequired
        SetParameterUpdateRequired(configp->Get(L"needGradient"));
        wstring initString = configp->Get(L"init");
        if (initString == L"fixedValue")
            Value().SetValue((ElemType) configp->Get(L"value"));
        else if (initString == L"uniform" || initString == L"gaussian")
        {
            // TODO: add these options also to old NDL
            static unsigned long randomSeed = 1;
            int forcedRandomSeed = configp->Get(L"randomSeed"); // forcing a specific random seed is useful for testing to get repeatable initialization independent of evaluation order
            InitRandom((initString == L"uniform"), forcedRandomSeed < 0 ? randomSeed++ : (unsigned long) forcedRandomSeed, configp->Get(L"initValueScale"), configp->Get(L"initOnCPUOnly"));
        }
        else if (initString == L"fromFile")
        {
            wstring initFromFilePath = configp->Get(L"initFromFilePath");
            if (initFromFilePath.empty())
                RuntimeError("initFromFilePath must be set when using \"fromFile\" initialization method");
            InitFromFile(initFromFilePath);
        }
        else
            RuntimeError("init must be one of the values of [ uniform | gaussian | fixedValue | fromFile ]");
    }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_parameterUpdateRequired;
        fstream << (size_t) 0 /*#rows in a legacy file format*/ << (size_t) 0 /*#cols in a legacy file format*/;
        m_sampleLayout.Save(fstream);
        fstream << Value();
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);

        size_t rows, cols;
        fstream >> m_parameterUpdateRequired;
        fstream >> rows >> cols;

        TensorShape sampleLayout;
        if (rows != 0) // legacy file format
            sampleLayout = TensorShape(rows, cols);
        else
        {
            sampleLayout.Load(fstream, /*acceptLegacyFormat=*/true);
            if (cols > 1) // in some legacy format, last tensor dimension was split off as an explicit column dimension
                sampleLayout.AppendInPlace(sampleLayout.GetRank(), cols);
        }
        LoadValue(fstream);
        SetDims(sampleLayout, false); // note: call this after LoadValue() since LoadValue() overwrites m_sampleLayout
        VerifyDataSize(Value());      // sanity check
    }

    // initialize with random numbers
    void InitRandom(const bool uniformInit,
                    const unsigned long randomSeed,
                    const ElemType initValueScale,
                    bool initOnCPUOnly) // if true then always init on CPU, making initialization consistent across both (for testing)
    {
        // fprintf(stderr, "%d x %d: %d  %ls\n", (int)GetNumRows(), (int)GetNumCols(), (int)randomSeed, NodeName().c_str());

        // the random seed offset is set via the "randomSeedOffset" parameter in config
        if (initOnCPUOnly)
            Value().TransferToDeviceIfNotThereAndNotAutoPlace(CPUDEVICE, true);
#if 1 // this more complex version is needed to repro test cases generated with an older version
        auto& value = GetSampleLayout().GetRank() > 2 ? Value() : ValueAsMatrix();
#else
        auto& value = Value();
#endif
        if (uniformInit)
        {
            // TODO: move these hidden extra factors out from here and into NDL, and make them visible in BS
            ElemType randRange = 0.05f * initValueScale;
            value.SetUniformRandomValue(-randRange, randRange, randomSeed);
        }
        else
        {
            size_t inputSize = GetAsMatrixNumCols();
            ElemType randInitstd = 0.2f * initValueScale / sqrt(ElemType(inputSize));
            value.SetGaussianRandomValue(0, randInitstd, randomSeed);
        }
        if (initOnCPUOnly)
            Value().TransferToDeviceIfNotThereAndNotAutoPlace(m_deviceId, true);
    }

    // initialize by reading a matrix from a text file
    void InitFromFile(const std::wstring& initFromFilePath)
    {
        size_t numRows = 0;
        size_t numCols = 0;
        auto array = File::LoadMatrixFromTextFile<ElemType>(msra::strfun::utf8(initFromFilePath), numRows, numCols); // TODO: change pathname to wstring
        Value().SetValue(numRows, numCols, m_deviceId, array.data(), matrixFlagNormal);
    }

    // TODO: share code with InitFromFile()
    void ReviseFromFile(const std::wstring& reviseFromFilePath)
    {
        size_t numRows = 0;
        size_t numCols = 0;
        auto array = File::LoadMatrixFromTextFile<ElemType>(msra::strfun::utf8(reviseFromFilePath), numRows, numCols); // TODO: change pathname to wstring
        size_t nRows = m_value->GetNumRows();
        size_t nCols = m_value->GetNumCols();

        if (numRows != nRows || numCols != nCols)
        {
            RuntimeError("Error in ReviseFromFile for node %ls using file %ls:  original size (%d x %d) vs current size (%d x %d)",
                         m_nodeName.c_str(), reviseFromFilePath.c_str(), (int) nRows, (int) nCols, (int) numRows, (int) numCols);
        }

        Value().SetValue(numRows, numCols, m_deviceId, array.data(), matrixFlagNormal);
    }

    // computation functions don't do anything for parameter nodes
    virtual void UpdateFunctionMBSize() override
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange&) override
    {
    }
    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange&) override
    {
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr; // this node does not hold mini-batch data
    }

    virtual void DumpNodeInfo(const bool printValues, File& fstream) const override
    {
        Base::DumpNodeInfo(printValues, fstream);

        char str[4096];
        sprintf(str, "[%lu,%lu]  ", GetAsMatrixNumRows(), GetAsMatrixNumCols());
        fstream << string(str);
        sprintf(str, "NeedGradient=%s", m_parameterUpdateRequired ? "true" : "false"); // TODO: update NDL to accept a better matching name as well
        fstream << string(str);

        PrintNodeValuesToFile(printValues, fstream);
    }
};

// -----------------------------------------------------------------------
// InputValueBase (/*no input*/)
// Base class for InputValue and SparseInputValue (typically fed by a DataReader)
// this covers four types: (regular vs. image) x (non-sparse vs. sparse)
// TODO: add -Node to the class names
// -----------------------------------------------------------------------

template <class ElemType>
class InputValueBase : public ComputationNode<ElemType>, public NumInputs<0>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembers;

    void Init(const TensorShape& sampleLayout, bool isSparse)
    {
        m_isSparse = isSparse;
        CreateMatrixIfNull(m_value);
        if (isSparse)
            ConvertToSparseMatrix();

        SetDims(sampleLayout, HasMBLayout()); // also called when reloading a file. Then we have an MBLayout, otherwise not yet
        UpdateFunctionValuesSize();           // we must allocate the matrix so that the readers get objects with valid row dimensions (some readers expect that)
        m_parameterUpdateRequired = false;
        this->m_valueSharable = false;
    }

protected:
    InputValueBase(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& sampleLayout, bool isSparse)
        : Base(deviceId, name)
    {
        Init(sampleLayout, isSparse);
    }
    InputValueBase(DEVICEID_TYPE deviceId, const wstring& name, size_t rows, bool isSparse)
        : InputValueBase(deviceId, name, TensorShape(rows), isSparse)
    {
    }
    InputValueBase(DEVICEID_TYPE deviceId, const wstring& name, bool isSparse)
        : InputValueBase(deviceId, name, TensorShape(), isSparse)
    {
    }
    InputValueBase(const ScriptableObjects::IConfigRecordPtr configp, bool isSparse)
        : Base(configp->Get(L"deviceId"), L"<placeholder>")
    {
        AttachInputs(configp, this->GetExpectedNumInputs());
        bool isImage = configp->Get(L"isImage");
        if (!isImage)
            Init(configp->Get(L"shape"), isSparse);
        else
            Init(ImageDimensions::AsTensorShape(configp->Get(L"imageWidth"), configp->Get(L"imageHeight"), configp->Get(L"imageChannels"), ImageLayoutKindFrom(configp->Get(L"imageLayout"))), isSparse);
    }

public:
    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        size_t rowsDummy = 0; // compat with old file format
        size_t colsDummy = 0;
        fstream << rowsDummy << colsDummy;
        m_sampleLayout.Save(fstream);
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);

        size_t rows, colsDummy;
        fstream >> rows >> colsDummy;
        TensorShape sampleLayout;
        sampleLayout.Load(fstream, /*acceptLegacyFormat=*/true);
        // some older files may have inconsistent tensor information
        if (rows != 0 /*old file*/ && rows != sampleLayout.GetNumElements() /*even older file*/)
        {
            fprintf(stderr, "WARNING: %ls InputValue has inconsistent serialized sample layout %s vs. number of rows %d. Resetting sample layout to vector.\n",
                    NodeName().c_str(), string(sampleLayout).c_str(), (int) rows);
            sampleLayout = TensorShape(rows);
        }
        Init(sampleLayout, m_isSparse);
    }

    // InputValue must not resize its inputs because that might destroy it. It should already have the correct size.
    virtual void UpdateFunctionMBSize() override
    {
        // don't touch our values
        // But take the opportunity for an additional check. Why not.
        if (Value().GetNumRows() != GetSampleLayout().GetNumElements())
            LogicError("UpdateFunctionMBSize: m_value not matching m_sampleLayout");
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange&) override
    {
    }
    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange&)
    {
        LogicError("InputValueBase::BackpropTo() should never be called.");
    }

    virtual void DumpNodeInfo(const bool printValues, File& fstream) const override
    {
        Base::DumpNodeInfo(printValues, fstream);
        fstream << "[" << string(GetSampleLayout()) << "]";
    }

private:
    bool m_isSparse = false;
    void ConvertToSparseMatrix()
    {
        m_value->SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSC, false);
    }
};

// -----------------------------------------------------------------------
// InputValue (/*no input*/)
// an input value (typically fed by a DataReader)
// this covers two types: (regular vs. image)
// TODO: There is still debate whether an InputValue without layout makes sense.
// -----------------------------------------------------------------------

template <class ElemType>
class InputValue : public InputValueBase<ElemType>
{
    typedef InputValueBase<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"InputValue";
    }

public:
    InputValue(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name, false)
    {
    }
    InputValue(DEVICEID_TYPE deviceId, const wstring& name, size_t rows)
        : Base(deviceId, name, rows, false)
    {
    }
    InputValue(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& sampleLayout)
        : Base(deviceId, name, sampleLayout, false)
    {
    }
    InputValue(const ScriptableObjects::IConfigRecordPtr configp)
        : Base(configp, false)
    {
    }
};


// -----------------------------------------------------------------------
// SparseInputValue (/*no input*/)
// a sparse input value (typically fed by a DataReader)
// this covers two types: (regular vs. image)
// -----------------------------------------------------------------------

template <class ElemType>
class SparseInputValue : public InputValueBase<ElemType>
{
    typedef InputValueBase<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"SparseInputValue";
    }

public:
    SparseInputValue(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name, true)
    {
    }
    SparseInputValue(DEVICEID_TYPE deviceId, const wstring& name, size_t rows)
        : Base(deviceId, name, rows, true)
    {
    }
    SparseInputValue(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& imageLayout)
        : Base(deviceId, name, imageLayout, true)
    {
    }
    SparseInputValue(const ScriptableObjects::IConfigRecordPtr configp)
        : Base(configp, true)
    {
    }
};


// -----------------------------------------------------------------------
// LookupTableNode (embedding matrix, bag-of-word representation of the inputs)
// implements an embedding, assuming a specific representation of the input data
// -----------------------------------------------------------------------

template <class ElemType>
class LookupTableNode : public ComputationNode<ElemType>, public NumInputs<2>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"LookupTable";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(LookupTableNode);
    LookupTableNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& t) override
    {
        if (inputIndex == 0) // left derivative (embedding matrix)
        {
            // This is a reduction operation, hence we need to mask out gaps.
            Matrix<ElemType> sliceInput1Value = Input(1)->MaskedValueFor(t);
            Matrix<ElemType> sliceOutputGrad = MaskedGradientFor(t);

            BackpropToLeft(sliceInput1Value, Input(0)->GradientAsMatrix(), sliceOutputGrad);
        }
        else if (inputIndex == 1) // right derivative (input)
        {
            Matrix<ElemType> sliceInput1Grad = Input(1)->GradientFor(t);
            Matrix<ElemType> sliceOutputGrad = GradientFor(t);

            BackpropToRight(Input(0)->ValueAsMatrix(), sliceInput1Grad, sliceOutputGrad);
        }
    }

    /*TODO: merge with call site*/ void BackpropToLeft(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues)
    {
        size_t rows1 = inputFunctionValues.GetNumRows(), cols1 = inputFunctionValues.GetNumCols();
        size_t rowsp = gradientValues.GetNumRows(), colsp = gradientValues.GetNumCols();
        int wordsInEachSample = rows1 / inputGradientValues.GetNumCols();

        inputFunctionValues.Reshape(rows1 / wordsInEachSample, cols1 * wordsInEachSample);
        gradientValues.Reshape(rowsp / wordsInEachSample, colsp * wordsInEachSample);

        Matrix<ElemType>::MultiplyAndAdd(gradientValues, false, inputFunctionValues, true, inputGradientValues);

        inputFunctionValues.Reshape(rows1, cols1);
        gradientValues.Reshape(rowsp, colsp);
    }

    /*TODO: merge with call site*/ void BackpropToRight(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues)
    {
        size_t rows1 = inputGradientValues.GetNumRows(), cols1 = inputGradientValues.GetNumCols();
        size_t rowsp = gradientValues.GetNumRows(), colsp = gradientValues.GetNumCols();
        int wordsInEachSample = rows1 / inputFunctionValues.GetNumCols();

        inputGradientValues.Reshape(rows1 / wordsInEachSample, cols1 * wordsInEachSample);
        gradientValues.Reshape(rowsp / wordsInEachSample, colsp * wordsInEachSample);

        Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, true, gradientValues, false, inputGradientValues);

        inputGradientValues.Reshape(rows1, cols1);
        gradientValues.Reshape(rowsp, colsp);
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& t) override
    {
        // input0 is the weight (each column is an embedding of one word), input 1 contains m_bnrLooked words in each column (sample)
        Matrix<ElemType> functionValues = ValueFor(t);
        const Matrix<ElemType>& input0 = Input(0)->ValueAsMatrix();
        Matrix<ElemType> input1 = Input(1)->ValueFor(t);

        size_t rows1 = input1.GetNumRows(), cols1 = input1.GetNumCols();
        size_t cols0 = input0.GetNumCols();

        int wordsInEachSample = rows1 / cols0;

        if (cols0 * wordsInEachSample != rows1)
            LogicError("LookupTableNode: rows of input 1 is not a multiple of cols of input 0. This usually happens when the feature dimension is not specified as that in the network definition of look-up-table dimension size.");

        auto input1Reshaped = input1.Reshaped(rows1 / wordsInEachSample, cols1 * wordsInEachSample);

        auto functionValuesReshaped = functionValues.Reshaped(input0.GetNumRows(), input1Reshaped.GetNumCols());
        functionValuesReshaped.AssignProductOf(input0, false, input1Reshaped, false);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        if (isFinalValidationPass && !HasMBLayout())
            InvalidArgument("%ls %ls operation can only operate on minibatches.", NodeName().c_str(), OperationName().c_str());
        if (isFinalValidationPass && Input(1)->GetSampleMatrixNumRows() % Input(0)->GetAsMatrixNumCols() != 0)
            InvalidArgument("Mismatched dimension. Rows in input1 must be multiples of cols in input0.");

        size_t wordsInEachSample = Input(1)->GetSampleMatrixNumRows() / Input(0)->GetAsMatrixNumCols() /*note: can never be 0*/;

        // TODO: Should this add a tensor dimension?
        SetDims(TensorShape(Input(0)->GetAsMatrixNumRows() * wordsInEachSample), true);
    }

    bool UnitTest()
    {
        try
        {
            size_t nInput = 2;
            size_t nHidden = 3;
            size_t nOutput = 3;

            Input(0)->SetDims1(nInput, nHidden);
            Input(0)->UpdateFunctionValuesSize();
            Input(0)->Value().SetValue(1.0);
            Input(1)->Value().TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true);
            Input(1)->Value().SwitchToMatrixType(DENSE, matrixFormatDense, false);
            Input(1)->SetDims1(nHidden, nOutput);
            Input(1)->UpdateFunctionValuesSize();
            Input(1)->Value().SetValue(0.0);
            Input(1)->Value().SetValue(0, 0, 1.0);
            Input(1)->Value().SetValue(1, 1, 2.0);
            Input(1)->Value().TransferFromDeviceToDevice(CPUDEVICE, m_deviceId, true);
            Input(1)->Value().SwitchToMatrixType(SPARSE, matrixFormatSparseCSC, true);
            SetDims1(nInput, nOutput);
            UpdateFunctionValuesSize();

            ForwardProp(FrameRange(m_pMBLayout));

            // check with expected values
            Value().TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true);
            if (!ISCLOSE(Value()(0, 0), 1.0, EPSILON) ||
                !ISCLOSE(Value()(0, 1), 2.0, EPSILON) ||
                !ISCLOSE(Value()(1, 1), 2.0, EPSILON))
                throw("LSTMNode forward computation error");

            Value().TransferToDeviceIfNotThere(m_deviceId, true);

            Gradient().Resize(nInput, nOutput);
            Gradient().SetValue(1.0);
            for (size_t i = 0; i < 2; i++)
            {
                Input(i)->Gradient().Resize(Input(i)->Value().GetNumRows(), Input(i)->Value().GetNumCols());
                Input(i)->Gradient().SetValue(0);
            }
            for (size_t i = 0; i < 2; i++)
                BackpropTo(i, FrameRange(m_pMBLayout));

            // check with expected values
            if (!ISCLOSE(Input(1)->Gradient()(0, 0), 2, EPSILON)    // bi
                || !ISCLOSE(Input(1)->Gradient()(0, 1), 2, EPSILON) // Wxi
                || !ISCLOSE(Input(1)->Gradient()(1, 0), 2, EPSILON) // Whi
                || !ISCLOSE(Input(1)->Gradient()(2, 1), 2, EPSILON) // Wci
                )
                throw("LSTMNode gradient error on input gates");

            for (size_t i = 0; i < 2; i++)
                Input(i)->Gradient().TransferToDeviceIfNotThere(m_deviceId, true);
        }
        catch (...)
        {
            fprintf(stderr, "LookupTableNode unit test is not passed!");
            return false;
        }

        fprintf(stderr, "LookupTableNode unit test passed!\n");
        return true;
    }
};

} } }
