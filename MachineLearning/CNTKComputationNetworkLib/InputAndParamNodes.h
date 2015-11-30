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
            m_imageLayout = ImageLayoutWHC(1, SIZE_MAX, 1);
        }
        LearnableParameter(DEVICEID_TYPE deviceId, const wstring & name, size_t rows, size_t cols) :
            Base(deviceId, name)
        {
            m_parameterUpdateRequired = true;
            m_imageLayout = ImageLayoutWHC(1, rows, 1);
            // TODO: Is ^^ this a wise choice? These are often weight matrices, where rows, not columns, are multiplied with input vectors.
            CreateMatrixIfNull(m_functionValues);
            SetDims(rows, cols);
            UpdateFunctionValuesSize();   // this allocates the matrix
            FunctionValues().SetValue(0);
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
                FunctionValues().SetValue((ElemType)configp->Get(L"value"));
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

        virtual void SaveToFile(File& fstream) const override
        {
            Base::SaveToFile(fstream);
            fstream << m_parameterUpdateRequired;
            fstream << GetNumRows() << GetNumCols(); 
            fstream << FunctionValues();
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);

            size_t rows, cols;
            fstream >> m_parameterUpdateRequired;
            fstream >> rows >> cols;

            SetDims(rows, cols);
            LoadFunctionValues(fstream);

            m_imageLayout = ImageLayoutWHC(1, rows, 1);
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
                m_functionValues->TransferToDeviceIfNotThereAndNotAutoPlace(CPUDEVICE, true);
            if (uniformInit)
            {
                ElemType randRange = 0.05f * initValueScale; //initValueScale/sqrt(inputSize);
                FunctionValues().SetUniformRandomValue(-randRange, randRange, randomSeed);
            }
            else
            {
                ElemType randInitstd = 0.2f * initValueScale / sqrt(ElemType(inputSize));
                FunctionValues().SetGaussianRandomValue(0, randInitstd, randomSeed);
            }
            if (initOnCPUOnly)
                m_functionValues->TransferToDeviceIfNotThereAndNotAutoPlace(m_deviceId, true);
        }

        // initialize by reading a matrix from a text file
        void InitFromFile(const std::wstring & initFromFilePath)
        {
            size_t numRows = 0;
            size_t numCols = 0;
            auto array = File::LoadMatrixFromTextFile<ElemType>(msra::strfun::utf8(initFromFilePath), numRows, numCols); // TODO: change pathname to wstring
            FunctionValues().SetValue(numRows, numCols, m_deviceId, array.data(), matrixFlagNormal);
        }

        void ReviseFromFile(const std::wstring & reviseFromFilePath)
        {
            size_t numRows = 0; 
            size_t numCols = 0; 
            auto array = File::LoadMatrixFromTextFile<ElemType>(msra::strfun::utf8(reviseFromFilePath), numRows, numCols); // TODO: change pathname to wstring
            size_t nRows = m_functionValues->GetNumRows(); 
            size_t nCols = m_functionValues->GetNumCols(); 

            if (numRows != nRows || numCols != nCols)
            {
                RuntimeError("Error in ReviseFromFile for node %ls using file %ls:  original size (%d x %d) vs current size (%d x %d)",
                    m_nodeName.c_str(), reviseFromFilePath.c_str(), (int)nRows, (int)nCols, (int)numRows, (int)numCols);
            }

            FunctionValues().SetValue(numRows, numCols, m_deviceId, array.data(), matrixFlagNormal);
            
        }

        // computation functions don't do anything for parameter nodes
        virtual void UpdateFunctionMBSize() override { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange &) override { }
        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange &) override { }

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

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            LearnableParameter<ElemType>::LoadFromFile(fstream, modelVersion);
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
            CreateMatrixIfNull(m_functionValues);
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
            m_imageLayout.Invalidate();
            Init(0, 0, isSparse);
        }
        InputValueBase(DEVICEID_TYPE deviceId, const wstring & name, size_t rows, size_t cols, bool isSparse) :
            Base(deviceId, name)
        {
            if (rows * cols == 0)
                LogicError("This InputValue dimension is 0.");

            m_imageLayout = ImageLayoutVector(rows);
            Init(rows, cols, isSparse);
        }
        InputValueBase(DEVICEID_TYPE deviceId, const wstring & name, const ImageLayout & imageLayout, size_t numImages, bool isSparse) :
            Base(deviceId, name)
        {
            size_t rows = imageLayout.GetNumElements();
            size_t cols = numImages;

            if (rows * cols == 0)
                LogicError("This InputValue dimension is 0.");

            m_imageLayout = imageLayout;

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
                m_imageLayout = ImageLayoutVector(rows);    // no tensor, just a vector
                Init(rows, cols, isSparse);
            }
            else
            {
                m_imageLayout = ImageLayoutWHC(configp->Get(L"imageWidth"), configp->Get(L"imageHeight"), configp->Get(L"imageChannels"));
                size_t rows = m_imageLayout.GetNumElements();
                size_t cols = configp->Get(L"numImages");         // this is actually the MB size
                Init(rows, cols, isSparse);
            }
        }
    public:

        virtual void SaveToFile(File& fstream) const override
        {
            Base::SaveToFile(fstream);
            size_t rows = GetNumRows();                     // using explicitly typed variables to be 100% symmetrical to LoadFromFile()
            size_t cols = m_pMBLayout ? 0 : GetNumCols();   // if this Input depends on MB size, we write it as having 0 dimensions
            fstream << rows << cols;
            m_imageLayout.SaveToFile(fstream);
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);

            size_t rows, cols;
            fstream >> rows >> cols;
            if (m_pMBLayout)    // some older files retained the #columns when saving, which is meaningless
                cols = 0;
            m_imageLayout.LoadFromFile(fstream);
            Init(rows, cols, m_isSparse);
        }

        // InputValue must not resize its inputs because that might destroy it. It should already have the correct size.
        virtual void UpdateFunctionMBSize() override
        {
            if (!m_pMBLayout)               // if no layout, this node contains parameters independent of MB size, don't resize
                VerifyDims(GetNumRows(), m_pMBLayout->GetNumCols());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange &) override { }
        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange &) { }

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
            m_functionValues->SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSC, false);
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
        InputValue(DEVICEID_TYPE deviceId, const wstring & name, const ImageLayout & imageLayout, size_t numImages) :
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
        SparseInputValue(DEVICEID_TYPE deviceId, const wstring & name, const ImageLayout & imageLayout, size_t numImages) :
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

        //void ComputeInputPartialMap(const size_t inputIndex)
        //{
        //    if (inputIndex > 1)
        //        InvalidArgument("LookupTable operation only takes two inputs.");
        //
        //    //DEVICEID_TYPE input1DeviceId = Inputs(1)->FunctionValues().GetDeviceId();
        //    //DEVICEID_TYPE input0DeviceId = Inputs(0)->FunctionValues().GetDeviceId();
        //    //Inputs(1)->FunctionValues().TransferFromDeviceToDevice(input1DeviceId, input0DeviceId);
        //
        //    if (inputIndex == 0)  //left derivative
        //    {
        //        ComputeInputPartialLeft(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
        //    }
        //    else  //right derivative
        //    {
        //        ComputeInputPartialRight(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues());
        //    }
        //    //Inputs(1)->FunctionValues().TransferFromDeviceToDevice(input0DeviceId, input1DeviceId);
        //}

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & t) override
        {
            //if (t.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            if (inputIndex == 0)        // left derivative (embedding matrix)
            {
                // This is a reduction operation, hence we need to mask out gaps.
                Matrix<ElemType> sliceInput1Value = Inputs(1)->MaskedValueSlice(t);
                Matrix<ElemType> sliceOutputGrad = MaskedGradientSlice(t);

                ComputeInputPartialLeft(sliceInput1Value, Inputs(0)->GradientValues(), sliceOutputGrad);
            }
            else if (inputIndex == 1)   // right derivative (input)
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientSlice(t);
                Matrix<ElemType> sliceOutputGrad = GradientSlice(t);

                ComputeInputPartialRight(Inputs(0)->FunctionValues(), sliceInput1Grad, sliceOutputGrad);
            }
        }

        /*TODO: merge with call site*/void ComputeInputPartialLeft(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues)  
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

        /*TODO: merge with call site*/void ComputeInputPartialRight(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues)  
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

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & t) override
        {
            // input0 is the weight (each column is an embedding of one word), input 1 contains m_bnrLooked words in each column (sample)
            Matrix<ElemType> functionValues = ValueSlice(t);
            const Matrix<ElemType>&  input0 = Inputs(0)->FunctionValues();
            Matrix<ElemType>         input1 = Inputs(1)->ValueSlice(t);

            size_t rows1 = input1.GetNumRows(), cols1 = input1.GetNumCols();
            size_t cols0 = input0.GetNumCols();

            if (rows1 % cols0 != 0)
                LogicError("LookupTableNode: rows of input 1 and cols of input 0 are not modular. e.g., rows1 = 0.9 cols and this is not allowed. Check feature reader and network definition. This usually happens when the feature dimension is not specified as that in the network definition of look-up-table dimension size.");

            int wordsInEachSample = rows1 / cols0;

            auto input1Reshaped = input1.Reshaped(rows1 / wordsInEachSample, cols1 * wordsInEachSample);

            //DEVICEID_TYPE input1DeviceId = input1.GetDeviceId();
            //DEVICEID_TYPE input0DeviceId = input0.GetDeviceId();
            //input1.TransferFromDeviceToDevice(input1DeviceId, input0DeviceId);

            auto functionValuesReshaped = functionValues.Reshaped(input0.GetNumRows(), input1Reshaped.GetNumCols());
            functionValuesReshaped.AssignProductOf(input0, false, input1Reshaped, false);
            //size_t rows = functionValues.GetNumRows();
            //functionValues.Reshape(rows * wordsInEachSample, cols1);

            //input1.TransferFromDeviceToDevice(input0DeviceId, input1DeviceId);

            //input1.Reshape(rows1, cols1);
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            if (isFinalValidationPass && Inputs(1)->GetNumRows() % Inputs(0)->GetNumCols() != 0)
                InvalidArgument("Mismatched dimension. Rows in input1 must be multiples of cols in input0.");

            int wordsInEachSample = Inputs(1)->GetNumRows() / Inputs(0)->GetNumCols();

            SetDims(Inputs(0)->GetNumRows() * wordsInEachSample, Inputs(1)->GetNumCols());

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

                Inputs(0)->SetDims(nInput, nHidden);
                Inputs(0)->UpdateFunctionValuesSize();
                Inputs(0)->FunctionValues().SetValue(1.0);
                Inputs(1)->FunctionValues().TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true);
                Inputs(1)->FunctionValues().SwitchToMatrixType(DENSE, matrixFormatDense, false);
                Inputs(1)->SetDims(nHidden, nOutput);
                Inputs(1)->UpdateFunctionValuesSize();
                Inputs(1)->FunctionValues().SetValue(0.0);
                Inputs(1)->FunctionValues().SetValue(0, 0, 1.0);
                Inputs(1)->FunctionValues().SetValue(1, 1, 2.0);
                Inputs(1)->FunctionValues().TransferFromDeviceToDevice(CPUDEVICE, m_deviceId, true);
                Inputs(1)->FunctionValues().SwitchToMatrixType(SPARSE, matrixFormatSparseCSC, true);
                SetDims(nInput, nOutput);
                UpdateFunctionValuesSize();

                EvaluateThisNode(FrameRange(m_pMBLayout));

                /// check with expected values
                FunctionValues().TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true);
                if (!ISCLOSE(FunctionValues()(0, 0), 1.0, EPSILON) ||
                    !ISCLOSE(FunctionValues()(0, 1), 2.0, EPSILON) ||
                    !ISCLOSE(FunctionValues()(1, 1), 2.0, EPSILON) )
                    throw("LSTMNode forward computation error");

                FunctionValues().TransferToDeviceIfNotThere( m_deviceId, true);

                GradientValues().Resize(nInput, nOutput);
                GradientValues().SetValue(1.0);
                for (size_t i = 0; i < 2; i++)
                {
                    Inputs(i)->GradientValues().Resize(Inputs(i)->GetNumRows(), Inputs(i)->GetNumCols());
                    Inputs(i)->GradientValues().SetValue(0);
                }
                for (size_t i = 0; i < 2; i++)
                    ComputeInputPartial(i, FrameRange(m_pMBLayout));

                // check with expected values
                if (!ISCLOSE(Inputs(1)->GradientValues()(0, 0), 2, EPSILON) /// bi
                    || !ISCLOSE(Inputs(1)->GradientValues()(0, 1), 2, EPSILON)  // Wxi
                    || !ISCLOSE(Inputs(1)->GradientValues()(1, 0), 2, EPSILON)  // Whi
                    || !ISCLOSE(Inputs(1)->GradientValues()(2, 1), 2, EPSILON)  // Wci
                    )
                    throw("LSTMNode gradient error on input gates");

                for (size_t i = 0; i < 2; i++)
                    Inputs(i)->GradientValues().TransferToDeviceIfNotThere(m_deviceId, true);
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
            CreateMatrixIfNull(m_functionValues);
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

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Init(1, 1); // TODO: this looks wrong; should the dimension not come from the loaded model data?
            Base::LoadFromFile(fstream, modelVersion);
        }

        /// to-do: need to change to the new way of resetting state
        void ComputeInputPartialMap(const size_t inputIndex)
        {
            if (inputIndex > 0)
                InvalidArgument("PairNetwork operation only takes one input.");

            Matrix<ElemType>::ScaleAndAdd(1.0, GradientValues(), Inputs(inputIndex)->GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            assert(m_functionValues->GetNumRows() == GradientValues().GetNumRows()); // original used m_functionValues->GetNumRows() for loop dimension
            assert(m_pMBLayout);

            Matrix<ElemType> mTmp = Inputs(inputIndex)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType>::ScaleAndAdd(1.0, GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout)), mTmp);
        }

        void EvaluateThisNodeMap()    // TODO: This is a stop-gap; in most cases, we should just be able to delete this (but need to review one by one)
        {
            m_functionValues->SetValue(Inputs(0)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            //if (frameRange.IsAllFrames()) { EvaluateThisNodeMap(); return; }
            Matrix<ElemType> mTmp = ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            mTmp.SetValue(Inputs(0)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout)));
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            size_t rows0 = Inputs(0)->GetNumRows(), cols0 = Inputs(0)->GetNumCols();
            if (rows0 > 0 && cols0 > 0) // TODO: is this check needed?
                SetDims(Inputs(0));

            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }

#if 0   // folded into base function, to avoid virtual; that base function already knows about some node types anyway
        virtual void EnumerateNodesForEval(std::unordered_set<ComputationNodePtr>& visited, std::list<ComputationNodePtr>& result,
                                           std::vector<ComputationNodePtr>& sourceRecurrentNodePtr, const bool bFromDelayNode)
        {
            if (visited.find(shared_from_this()) == visited.end())  //not visited
            {
                visited.insert(shared_from_this());   // have visited tagged here to avoid infinite loop over children, children's children, etc

                //children first for function evaluation
                if (!IsLeaf())
                    m_parameterUpdateRequired = ChildrenNeedGradient();  //only nodes that require gradient calculation is included in gradient calculation

                result.push_back(shared_from_this());  //we put this in the list even if it's leaf since we need to use it to determine learnable params 
                this->m_visitedOrder = result.size();
            }
            else
            {
                if (!IsLeaf() && bFromDelayNode)
                    sourceRecurrentNodePtr.push_back(shared_from_this());
            }
        }
#endif
    };

    template class PairNetworkNode<float>;
    template class PairNetworkNode<double>;

}}}
