//
// <copyright file="InputAndParamNodes.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "ScriptableObjects.h"
#include "Matrix.h"
#include "File.h"   // for LoadMatrixFromTextFile()
#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <algorithm>
#include <assert.h>
#include <atomic>
#include <sstream>
#include <iostream>

namespace Microsoft { namespace MSR { namespace CNTK {

    // -----------------------------------------------------------------------
    // LearnableParameter (/*no input*/)
    // represents weight matrices and biases
    // -----------------------------------------------------------------------

    template<class ElemType>
    class LearnableParameter : public ComputationNode<ElemType>, public NumInputs<0>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"LearnableParameter"; }
    public:
        LearnableParameter(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        {
            m_parameterUpdateRequired = true;
            m_sampleLayout = ImageLayoutWHC(1, SIZE_MAX, 1);
        }
        LearnableParameter(DEVICEID_TYPE deviceId, const wstring & name, size_t rows, size_t cols) :
            Base(deviceId, name)
        {
            m_parameterUpdateRequired = true;
            m_sampleLayout = ImageLayoutWHC(1, rows, 1);
            // TODO: Is ^^ this a wise choice? These are often weight matrices, where rows, not columns, are multiplied with input vectors.
            CreateMatrixIfNull(m_output);
            SetDims(rows, cols);
            UpdateFunctionValuesSize();   // this allocates the matrix
            Output().SetValue(0);
        }
        LearnableParameter(const ScriptableObjects::IConfigRecordPtr configp) :
            LearnableParameter(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"rows"), configp->Get(L"cols"))
        {
            AttachInputs(configp, this->GetExpectedNumInputs());
            // parameters[rows, [cols=1]] plus other optional parameters (needGradient=[true|false], init=[uniform|gaussian|fixedvalue], initValueScale=[1|float], value=[0|float])
            // TODO: "needGradient" should be renamed to better match m_parameterUpdateRequired
            SetParameterUpdateRequired(configp->Get(L"needGradient"));
            wstring initString = configp->Get(L"init");
            if (initString == L"fixedValue")
                Output().SetValue((ElemType)configp->Get(L"value"));
            else if (initString == L"uniform" || initString == L"gaussian")
            {
                // TODO: add these options also to old NDL
                static unsigned long randomSeed = 1;
                int forcedRandomSeed = configp->Get(L"randomSeed");   // forcing a specific random seed is useful for testing to get repeatable initialization independent of evaluation order
                InitRandom((initString == L"uniform"), forcedRandomSeed < 0 ? randomSeed++ : (unsigned long)forcedRandomSeed, configp->Get(L"initValueScale"), configp->Get(L"initOnCPUOnly"));
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
            fstream << GetNumRows() << GetNumCols(); 
            fstream << Output();
        }

        virtual void Load(File& fstream, size_t modelVersion) override
        {
            Base::Load(fstream, modelVersion);

            size_t rows, cols;
            fstream >> m_parameterUpdateRequired;
            fstream >> rows >> cols;

            SetDims(rows, cols);
            LoadFunctionValues(fstream);

            m_sampleLayout = ImageLayoutWHC(1, rows, 1);
        }

        // initialize with random numbers
        void InitRandom(const bool uniformInit,
                        const unsigned long randomSeed,
                        const ElemType initValueScale,
                        bool initOnCPUOnly) // if true then always init on CPU, making initialization consistent across both (for testing)
        {
            size_t inputSize = GetNumCols();

            // the random seed offset is set via the "randomSeedOffset" parameter in config
            if (initOnCPUOnly)
                m_output->TransferToDeviceIfNotThereAndNotAutoPlace(CPUDEVICE, true);
            if (uniformInit)
            {
                ElemType randRange = 0.05f * initValueScale; //initValueScale/sqrt(inputSize);
                Output().SetUniformRandomValue(-randRange, randRange, randomSeed);
            }
            else
            {
                ElemType randInitstd = 0.2f * initValueScale / sqrt(ElemType(inputSize));
                Output().SetGaussianRandomValue(0, randInitstd, randomSeed);
            }
            if (initOnCPUOnly)
                m_output->TransferToDeviceIfNotThereAndNotAutoPlace(m_deviceId, true);
        }

        // initialize by reading a matrix from a text file
        void InitFromFile(const std::wstring & initFromFilePath)
        {
            size_t numRows = 0;
            size_t numCols = 0;
            auto array = File::LoadMatrixFromTextFile<ElemType>(msra::strfun::utf8(initFromFilePath), numRows, numCols); // TODO: change pathname to wstring
            Output().SetValue(numRows, numCols, m_deviceId, array.data(), matrixFlagNormal);
        }

        void ReviseFromFile(const std::wstring & reviseFromFilePath)
        {
            size_t numRows = 0; 
            size_t numCols = 0; 
            auto array = File::LoadMatrixFromTextFile<ElemType>(msra::strfun::utf8(reviseFromFilePath), numRows, numCols); // TODO: change pathname to wstring
            size_t nRows = m_output->GetNumRows(); 
            size_t nCols = m_output->GetNumCols(); 

            if (numRows != nRows || numCols != nCols)
            {
                RuntimeError("Error in ReviseFromFile for node %ls using file %ls:  original size (%d x %d) vs current size (%d x %d)",
                    m_nodeName.c_str(), reviseFromFilePath.c_str(), (int)nRows, (int)nCols, (int)numRows, (int)numCols);
            }

            Output().SetValue(numRows, numCols, m_deviceId, array.data(), matrixFlagNormal);
            
        }

        // computation functions don't do anything for parameter nodes
        virtual void UpdateFunctionMBSize() override { }

        virtual void /*ComputationNode::*/BackpropTo(const size_t /*inputIndex*/, const FrameRange &) override { }
        virtual void /*ComputationNode::*/ForwardProp(const FrameRange &) override { }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);
            m_pMBLayout = nullptr;    // this node does not hold mini-batch data
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const override
        {
            Base::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "[%lu,%lu]  ", GetNumRows(), GetNumCols());
            fstream << string(str);
            sprintf(str, "NeedGradient=%s", m_parameterUpdateRequired ? "true" : "false");  // TODO: update NDL to accept a better matching name as well
            fstream << string(str);

            PrintNodeValuesToFile(printValues, fstream);
        }
    };

    // -----------------------------------------------------------------------
    // SparseLearnableParameter (/*no input*/)
    // -----------------------------------------------------------------------

    //WARNING: Don't use SparseLearnableParameter yet since the current version assumes the parameter is dense instead of sparse
    //WARNING: After the right implementation is put here we need to turn it on in NetworkDescriptionLangauge.cpp
    template<class ElemType>
    class SparseLearnableParameter : public LearnableParameter<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"SparseLearnableParameter"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(SparseLearnableParameter);
        SparseLearnableParameter(DEVICEID_TYPE deviceId, const wstring & name) :
            LearnableParameter<ElemType>(deviceId, name)
        {
            CreateMatrixIfNull(m_gradientValues);
            m_gradientValues->SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol, false);
        }
        SparseLearnableParameter(DEVICEID_TYPE deviceId, const wstring & name, size_t rows, size_t cols, size_t size) :
            LearnableParameter<ElemType>(deviceId, name, rows, cols)
        {
            CreateMatrixIfNull(m_gradientValues);
            m_gradientValues->SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol, false);
            m_gradientValues->Resize(rows, cols, size);
        }

        virtual void Load(File& fstream, size_t modelVersion) override
        {
            LearnableParameter<ElemType>::Load(fstream, modelVersion);
            CreateMatrixIfNull(m_gradientValues);
            m_gradientValues->Resize(GetNumRows(), GetNumCols());
        }
    };

    template class SparseLearnableParameter<float>; 
    template class SparseLearnableParameter<double>;

    // -----------------------------------------------------------------------
    // InputValueBase (/*no input*/)
    // Base class for InputValue and SparseInputValue (typically fed by a DataReader)
    // this covers four types: (regular vs. image) x (non-sparse vs. sparse)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class InputValueBase : public ComputationNode<ElemType>, public NumInputs<0>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;

        void Init(size_t rows, size_t cols, bool isSparse)
        {
            m_isSparse = isSparse;
            CreateMatrixIfNull(m_output);
            if (isSparse)
                ConvertToSparseMatrix();

            SetDims(rows, cols);
            UpdateFunctionValuesSize();     // we must allocate the matrix so that the readers get objects with valid row dimensions (some readers expect that)
            m_parameterUpdateRequired = false;
        }
    protected:
        InputValueBase(DEVICEID_TYPE deviceId, const wstring & name, bool isSparse) :
            Base(deviceId, name)
        {
            m_sampleLayout.Invalidate();
            Init(0, 0, isSparse);
        }
        InputValueBase(DEVICEID_TYPE deviceId, const wstring & name, size_t rows, size_t cols, bool isSparse) :
            Base(deviceId, name)
        {
            if (rows * cols == 0)
                LogicError("This InputValue dimension is 0.");

            m_sampleLayout = ImageLayoutVector(rows);
            Init(rows, cols, isSparse);
        }
        InputValueBase(DEVICEID_TYPE deviceId, const wstring & name, const TensorShape & imageLayout, size_t numImages, bool isSparse) :
            Base(deviceId, name)
        {
            size_t rows = imageLayout.GetNumElements();
            size_t cols = numImages;

            if (rows * cols == 0)
                LogicError("This InputValue dimension is 0.");

            m_sampleLayout = imageLayout;

            Init(rows, cols, isSparse);
        }
        InputValueBase(const ScriptableObjects::IConfigRecordPtr configp, bool isSparse) :
            Base(configp->Get(L"deviceId"), L"<placeholder>")
        {
            AttachInputs(configp, this->GetExpectedNumInputs());
            bool isImage  = configp->Get(L"isImage");
            if (!isImage)
            {
                size_t rows = configp->Get(L"rows");
                size_t cols = configp->Get(L"cols");
                m_sampleLayout = ImageLayoutVector(rows);    // no tensor, just a vector
                Init(rows, cols, isSparse);
            }
            else
            {
                m_sampleLayout = ImageLayoutWHC(configp->Get(L"imageWidth"), configp->Get(L"imageHeight"), configp->Get(L"imageChannels"));
                size_t rows = m_sampleLayout.GetNumElements();
                size_t cols = configp->Get(L"numImages");         // this is actually the MB size
                Init(rows, cols, isSparse);
            }
        }
    public:

        virtual void Save(File& fstream) const override
        {
            Base::Save(fstream);
            size_t rows = GetNumRows();                     // using explicitly typed variables to be 100% symmetrical to Load()
            size_t cols = m_pMBLayout ? 0 : GetNumCols();   // if this Input depends on MB size, we write it as having 0 dimensions
            fstream << rows << cols;
            m_sampleLayout.Save(fstream);
        }

        virtual void Load(File& fstream, size_t modelVersion) override
        {
            Base::Load(fstream, modelVersion);

            size_t rows, cols;
            fstream >> rows >> cols;
            if (m_pMBLayout)    // some older files retained the #columns when saving, which is meaningless
                cols = 0;
            m_sampleLayout.Load(fstream);
            Init(rows, cols, m_isSparse);
        }

        // InputValue must not resize its inputs because that might destroy it. It should already have the correct size.
        virtual void UpdateFunctionMBSize() override
        {
            if (!m_pMBLayout)               // if no layout, this node contains parameters independent of MB size, don't resize
                VerifyDims(GetNumRows(), m_pMBLayout->GetNumCols());
        }

        virtual void /*ComputationNode::*/ForwardProp(const FrameRange &) override { }
        virtual void /*ComputationNode::*/BackpropTo(const size_t /*inputIndex*/, const FrameRange &) { }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const override
        {
            Base::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "[%lu,%lu]", GetNumRows(), GetNumCols());
            fstream << string(str);         // TODO: string(.) necessary?
        }
    private:
        bool m_isSparse = false;
        void ConvertToSparseMatrix()
        {
            m_output->SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSC, false);
        }
    };

    // -----------------------------------------------------------------------
    // InputValue (/*no input*/)
    // an input value (typically fed by a DataReader)
    // this covers two types: (regular vs. image)
    // TODO: There is still debate whether an InputValue without layout makes sense.
    // -----------------------------------------------------------------------

    template<class ElemType>
    class InputValue : public InputValueBase<ElemType>
    {
        typedef InputValueBase<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"InputValue"; }
    public:
        InputValue(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name, false)
        { }
        InputValue(DEVICEID_TYPE deviceId, const wstring & name, size_t rows, size_t cols) :
            Base(deviceId, name, rows, cols, false)
        { }
        InputValue(DEVICEID_TYPE deviceId, const wstring & name, const TensorShape & imageLayout, size_t numImages) :
            Base(deviceId, name, imageLayout, numImages, false)
        { }
        InputValue(const ScriptableObjects::IConfigRecordPtr configp) :
            Base(configp, false)
        { }
    };

    template class InputValue<float>;
    template class InputValue<double>;

    // -----------------------------------------------------------------------
    // SparseInputValue (/*no input*/)
    // a sparse input value (typically fed by a DataReader)
    // this covers two types: (regular vs. image)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class SparseInputValue : public InputValueBase<ElemType>
    {
        typedef InputValueBase<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"SparseInputValue"; }
    public:
        SparseInputValue(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name, true)
        { }
        SparseInputValue(DEVICEID_TYPE deviceId, const wstring & name, size_t rows, size_t cols) :
            Base(deviceId, name, rows, cols, true)
        { }
        SparseInputValue(DEVICEID_TYPE deviceId, const wstring & name, const TensorShape & imageLayout, size_t numImages) :
            Base(deviceId, name, imageLayout, numImages, true)
        { }
        SparseInputValue(const ScriptableObjects::IConfigRecordPtr configp) :
            Base(configp, true)
        { }
    };

    template class SparseInputValue<float>;
    template class SparseInputValue<double>;

    // -----------------------------------------------------------------------
    // LookupTableNode (embedding matrix, bag-of-word representation of the inputs)
    // implements an embedding, assuming a specific representation of the input data
    // -----------------------------------------------------------------------

    template<class ElemType>
    class LookupTableNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"LookupTable"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(LookupTableNode);
        LookupTableNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/BackpropTo(const size_t inputIndex, const FrameRange & t) override
        {
            if (inputIndex == 0)        // left derivative (embedding matrix)
            {
                // This is a reduction operation, hence we need to mask out gaps.
                Matrix<ElemType> sliceInput1Value = Input(1)->MaskedValueSlice(t);
                Matrix<ElemType> sliceOutputGrad = MaskedGradientSlice(t);

                BackpropToLeft(sliceInput1Value, Input(0)->GradientValues(), sliceOutputGrad);
            }
            else if (inputIndex == 1)   // right derivative (input)
            {
                Matrix<ElemType> sliceInput1Grad = Input(1)->GradientFor(t);
                Matrix<ElemType> sliceOutputGrad = GradientFor(t);

                BackpropToRight(Input(0)->Output(), sliceInput1Grad, sliceOutputGrad);
            }
        }

        /*TODO: merge with call site*/void BackpropToLeft(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues)  
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

        /*TODO: merge with call site*/void BackpropToRight(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues)  
        {
            size_t rows1 =inputGradientValues.GetNumRows(), cols1 = inputGradientValues.GetNumCols();
            size_t rowsp = gradientValues.GetNumRows(), colsp = gradientValues.GetNumCols();
            int wordsInEachSample = rows1 / inputFunctionValues.GetNumCols();

            inputGradientValues.Reshape(rows1 / wordsInEachSample, cols1 * wordsInEachSample);
            gradientValues.Reshape(rowsp / wordsInEachSample, colsp * wordsInEachSample);

            Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, true, gradientValues, false, inputGradientValues);

            inputGradientValues.Reshape(rows1, cols1);
            gradientValues.Reshape(rowsp, colsp);
        }

        virtual void /*ComputationNode::*/ForwardProp(const FrameRange & t) override
        {
            // input0 is the weight (each column is an embedding of one word), input 1 contains m_bnrLooked words in each column (sample)
            Matrix<ElemType> functionValues = OutputFor(t);
            const Matrix<ElemType>&  input0 = Input(0)->Output();
            Matrix<ElemType>         input1 = Input(1)->OutputFor(t);

            size_t rows1 = input1.GetNumRows(), cols1 = input1.GetNumCols();
            size_t cols0 = input0.GetNumCols();

            if (rows1 % cols0 != 0)
                LogicError("LookupTableNode: rows of input 1 and cols of input 0 are not modular. e.g., rows1 = 0.9 cols and this is not allowed. Check feature reader and network definition. This usually happens when the feature dimension is not specified as that in the network definition of look-up-table dimension size.");

            int wordsInEachSample = rows1 / cols0;

            auto input1Reshaped = input1.Reshaped(rows1 / wordsInEachSample, cols1 * wordsInEachSample);

            auto functionValuesReshaped = functionValues.Reshaped(input0.GetNumRows(), input1Reshaped.GetNumCols());
            functionValuesReshaped.AssignProductOf(input0, false, input1Reshaped, false);
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            if (isFinalValidationPass && Input(1)->GetNumRows() % Input(0)->GetNumCols() != 0)
                InvalidArgument("Mismatched dimension. Rows in input1 must be multiples of cols in input0.");

            int wordsInEachSample = Input(1)->GetNumRows() / Input(0)->GetNumCols();

            SetDims(Input(0)->GetNumRows() * wordsInEachSample, Input(1)->GetNumCols());

            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs(); 
        }

        bool UnitTest()
        {
            try
            {
                size_t nInput = 2;
                size_t nHidden = 3;
                size_t nOutput = 3;

                Input(0)->SetDims(nInput, nHidden);
                Input(0)->UpdateFunctionValuesSize();
                Input(0)->Output().SetValue(1.0);
                Input(1)->Output().TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true);
                Input(1)->Output().SwitchToMatrixType(DENSE, matrixFormatDense, false);
                Input(1)->SetDims(nHidden, nOutput);
                Input(1)->UpdateFunctionValuesSize();
                Input(1)->Output().SetValue(0.0);
                Input(1)->Output().SetValue(0, 0, 1.0);
                Input(1)->Output().SetValue(1, 1, 2.0);
                Input(1)->Output().TransferFromDeviceToDevice(CPUDEVICE, m_deviceId, true);
                Input(1)->Output().SwitchToMatrixType(SPARSE, matrixFormatSparseCSC, true);
                SetDims(nInput, nOutput);
                UpdateFunctionValuesSize();

                ForwardProp(FrameRange(m_pMBLayout));

                /// check with expected values
                Output().TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true);
                if (!ISCLOSE(Output()(0, 0), 1.0, EPSILON) ||
                    !ISCLOSE(Output()(0, 1), 2.0, EPSILON) ||
                    !ISCLOSE(Output()(1, 1), 2.0, EPSILON) )
                    throw("LSTMNode forward computation error");

                Output().TransferToDeviceIfNotThere( m_deviceId, true);

                GradientValues().Resize(nInput, nOutput);
                GradientValues().SetValue(1.0);
                for (size_t i = 0; i < 2; i++)
                {
                    Input(i)->GradientValues().Resize(Input(i)->GetNumRows(), Input(i)->GetNumCols());
                    Input(i)->GradientValues().SetValue(0);
                }
                for (size_t i = 0; i < 2; i++)
                    BackpropTo(i, FrameRange(m_pMBLayout));

                // check with expected values
                if (!ISCLOSE(Input(1)->GradientValues()(0, 0), 2, EPSILON) /// bi
                    || !ISCLOSE(Input(1)->GradientValues()(0, 1), 2, EPSILON)  // Wxi
                    || !ISCLOSE(Input(1)->GradientValues()(1, 0), 2, EPSILON)  // Whi
                    || !ISCLOSE(Input(1)->GradientValues()(2, 1), 2, EPSILON)  // Wci
                    )
                    throw("LSTMNode gradient error on input gates");

                for (size_t i = 0; i < 2; i++)
                    Input(i)->GradientValues().TransferToDeviceIfNotThere(m_deviceId, true);
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

    template class LookupTableNode<float>;
    template class LookupTableNode<double>;

    // -----------------------------------------------------------------------
    // PairNetworkNode (input)
    // -----------------------------------------------------------------------

    /**
    pair this node to a node in another network
    this node provide an interface from this network. The next layer network then can use this interface to know which node to connect to.
    */
    template<class ElemType>
    class PairNetworkNode : public ComputationNode<ElemType>, public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"PairNetwork"; }

        void Init(size_t row_size, size_t col_size)
        {
            CreateMatrixIfNull(m_output);
            SetDims(row_size, col_size);
            UpdateFunctionValuesSize();
        }
    public:
        DeclareConstructorFromConfigWithNumInputs(PairNetworkNode);
        PairNetworkNode(DEVICEID_TYPE deviceId, const wstring & name, size_t row_size = 1, size_t col_size = 1) :
            Base(deviceId, name)
        {
            Init(row_size, col_size);
            CreateMatrixIfNull(m_gradientValues);
            m_gradientValues->Resize(row_size, col_size);
            m_gradientValues->SetValue(0.0f);
        }

        virtual void Load(File& fstream, size_t modelVersion) override
        {
            Init(1, 1); // TODO: this looks wrong; should the dimension not come from the loaded model data?
            Base::Load(fstream, modelVersion);
        }

        /// to-do: need to change to the new way of resetting state
        void BackpropToMap(const size_t inputIndex)
        {
            if (inputIndex > 0)
                InvalidArgument("PairNetwork operation only takes one input.");

            Matrix<ElemType>::ScaleAndAdd(1.0, GradientValues(), Input(inputIndex)->GradientValues());
        }

        virtual void /*ComputationNode::*/BackpropTo(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { BackpropToMap(inputIndex); return; } // TODO: remove these one by one
            assert(m_output->GetNumRows() == GradientValues().GetNumRows()); // original used m_output->GetNumRows() for loop dimension
            assert(m_pMBLayout);

            Matrix<ElemType> mTmp = Input(inputIndex)->GradientFor(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType>::ScaleAndAdd(1.0, GradientFor(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout)), mTmp);
        }

        virtual void /*ComputationNode::*/ForwardProp(const FrameRange & frameRange) override
        {
            Matrix<ElemType> mTmp = OutputFor(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            mTmp.SetValue(Input(0)->OutputFor(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout)));
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            size_t rows0 = Input(0)->GetNumRows(), cols0 = Input(0)->GetNumCols();
            if (rows0 > 0 && cols0 > 0) // TODO: is this check needed?
                SetDims(Input(0));

            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }
    };

    template class PairNetworkNode<float>;
    template class PairNetworkNode<double>;

}}}
