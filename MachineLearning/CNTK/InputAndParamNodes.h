//
// <copyright file="InputAndParamNodes.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

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

#include "Basics.h"
#include "Matrix.h"
#include "ComputationNode.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    //used to represent weight Matrix<ElemType> and biases
    template<class ElemType>
    class LearnableParameter : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        LearnableParameter(size_t rows, size_t cols, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            //intentionally comment out so that we may support automatic dimention inference
            //if (rows * cols == 0) 
            //    throw std::logic_error("This LearnableParameter dimension is 0.");

            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_needGradient = true;
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_functionValues.Resize(rows, cols);

            m_outputWidth = 1;
            m_outputHeight = rows;
            m_outputChannels = 1;

            InitRecurrentNode();
        }

        LearnableParameter(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual void SaveToFile(File& fstream) const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << NeedGradient();
            fstream << FunctionValues().GetNumRows() << FunctionValues().GetNumCols(); 
            fstream << FunctionValues();
        }
        
        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            size_t rows, cols;
            fstream >> m_needGradient;
            fstream >> rows >> cols;

            //intentionally comment out to support automatic dimention inference
            //if (rows * cols == 0) 
            //    throw std::logic_error("This LearnableParameter dimension is 0.");

            m_functionValues.Resize(rows, cols);
            fstream >> m_functionValues;

            m_outputWidth = 1;
            m_outputHeight = rows;
            m_outputChannels = 1;
        }


        virtual const std::wstring OperationName() const {return TypeName();}
        virtual void ComputeInputPartial(const size_t /*inputIndex*/) {}
        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/) {}
        virtual void EvaluateThisNode()  {}
        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/) {}
        virtual void Validate() 
        {
            PrintSelfBeforeValidation();
        }

        static const std::wstring TypeName() {return L"LearnableParameter";} 

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            ComputationNode<ElemType>::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "[%lu,%lu]  ", FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            fstream << string(str);
            sprintf(str, "NeedGradient=%s", NeedGradient()? "true" : "false");
            fstream << string(str);

            PrintNodeValuesToFile(printValues, fstream);
        }

        // copy constructor
        LearnableParameter(const LearnableParameter<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new LearnableParameter<ElemType>(this, name, flags);
            return node;
        }
    };

    //WARNING: Don't use SparseLearnableParameter yet since the current version assumes the parameter is dense instead of sparse
    //WARNING: After the right implementation is put here we need to turn it on in NetworkDescriptionLangauge.cpp
    template<class ElemType>
    class SparseLearnableParameter : public LearnableParameter<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        SparseLearnableParameter(size_t rows, size_t cols, const size_t size, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : LearnableParameter<ElemType>(rows, cols, deviceId, name)
        {
            m_gradientValues.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol, false);
            m_gradientValues.Resize(rows, cols, size);
        }

        SparseLearnableParameter (File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"") 
            : LearnableParameter<ElemType>(fstream, modelVersion, deviceId, name)
        {
            m_gradientValues.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol, false);
            m_gradientValues.Resize(FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
        {
            LearnableParameter<ElemType>::LoadFromFile(fstream,   modelVersion, deviceId);
            m_gradientValues.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol, false);
            m_gradientValues.Resize(FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"SparseLearnableParameter";} 

        // copy constructor
        SparseLearnableParameter (const SparseLearnableParameter <ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) 
            : LearnableParameter<ElemType>( node, newName, flags)
        {
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new SparseLearnableParameter<ElemType>(this, name, flags);
            return node;
        }
    };

    template class SparseLearnableParameter<float>; 
    template class SparseLearnableParameter<double>;

    //used to represent weight Matrix<ElemType> and biases
    template<class ElemType>
    class ConstParameter : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        ConstParameter(size_t rows, size_t cols, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            //intentionally comment out so that we may support automatic dimention inference
            //if (rows * cols == 0) 
            //    throw std::logic_error("This ConstParameter dimension is 0.");

            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            m_needGradient = false;
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_functionValues.Resize(rows, cols);

            m_outputWidth = 1;
            m_outputHeight = rows;
            m_outputChannels = 1;

            InitRecurrentNode();
        }

        ConstParameter(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual void SaveToFile(File& fstream) const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << NeedGradient();
            fstream << FunctionValues().GetNumRows() << FunctionValues().GetNumCols();
            fstream << FunctionValues();
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            size_t rows, cols;
            fstream >> m_needGradient;
            fstream >> rows >> cols;

            //intentionally comment out to support automatic dimention inference
            //if (rows * cols == 0) 
            //    throw std::logic_error("This ConstParameter dimension is 0.");

            m_functionValues.Resize(rows, cols);
            fstream >> m_functionValues;

            m_outputWidth = 1;
            m_outputHeight = rows;
            m_outputChannels = 1;
        }


        virtual const std::wstring OperationName() const { return TypeName(); }
        virtual void ComputeInputPartial(const size_t /*inputIndex*/) {}
        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/) {}
        virtual void EvaluateThisNode()  {}
        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/) {}
        virtual void Validate()
        {
            PrintSelfBeforeValidation();
        }

        static const std::wstring TypeName() { return L"ConstParameter"; }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            ComputationNode<ElemType>::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "[%lu,%lu]  ", FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            fstream << string(str);
            sprintf(str, "NeedGradient=%s", NeedGradient() ? "true" : "false");
            fstream << string(str);

            PrintNodeValuesToFile(printValues, fstream);
        }

        // copy constructor
        ConstParameter(const ConstParameter<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;

            ComputationNodePtr node = new ConstParameter<ElemType>(this, name, flags);
            return node;
        }
    };

    template class ConstParameter<float>;
    template class ConstParameter<double>;

    template<class ElemType>
    class InputValue : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        InputValue(size_t rows, size_t cols, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId) 
        {
            if (rows * cols == 0) 
                throw std::logic_error("This InputValue dimension is 0.");

            m_outputWidth = 1;
            m_outputHeight = rows;
            m_outputChannels = 1;

            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_functionValues.Resize(rows, cols);
            m_needGradient = false;
            InitRecurrentNode();
        }
        
        InputValue(size_t imageWidth, size_t imageHeight, size_t imageChannels, size_t numImages, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId) 
        {
            size_t rows = imageWidth * imageHeight * imageChannels;
            size_t cols = numImages;

            if (rows * cols == 0) 
                throw std::logic_error("This InputValue dimension is 0.");

            m_outputWidth = imageWidth;
            m_outputHeight = imageHeight;
            m_outputChannels = imageChannels;

            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_functionValues.Resize(rows, cols);
            m_needGradient = false;
            InitRecurrentNode();
        }        

        InputValue(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual void SaveToFile(File& fstream) const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << FunctionValues().GetNumRows() << FunctionValues().GetNumCols(); 
            fstream << m_outputWidth << m_outputHeight << m_outputChannels; 
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            size_t rows, cols;
            fstream >> rows >> cols;
            if (rows * cols == 0) 
                throw std::logic_error("This InputValue dimension is 0.");

            fstream >> m_outputWidth >> m_outputHeight >> m_outputChannels; 

            m_functionValues.Resize(rows, cols);
            m_needGradient = false;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"InputValue";} 

        virtual void EvaluateThisNode()  {} 
        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/) {}
        
        virtual void ComputeInputPartial(const size_t /*inputIndex*/) {}
        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/) {}

        virtual void Validate() 
        {
            PrintSelfBeforeValidation();
            //CopyImageSizeFromInputs(); //not necessary since InputValue are leafs. put it here for consistent
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            ComputationNode<ElemType>::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "[%lu,%lu]", FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            fstream << string(str);        
        }

        // copy constructor
        InputValue(const InputValue<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new InputValue<ElemType>(this, name, flags);
            return node;
        }

    };

    template class InputValue<float>; 
    template class InputValue<double>;

    template<class ElemType>
    class SparseInputValue : public InputValue<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        SparseInputValue (size_t rows, size_t cols, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : InputValue<ElemType>(rows, cols, deviceId, name) 
        {
            ConvertToSparseMatrix();
        }
        
        SparseInputValue (size_t imageWidth, size_t imageHeight, size_t imageChannels, size_t numImages, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") 
            : InputValue<ElemType>(imageWidth, imageHeight, imageChannels, numImages, deviceId, name)
        {
                ConvertToSparseMatrix();
        }

        SparseInputValue (File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : InputValue<ElemType>(fstream, modelVersion, deviceId, name)
        {
            ConvertToSparseMatrix();
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
        {
            InputValue<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);
            ConvertToSparseMatrix();
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"SparseInputValue";} 

        // copy constructor
        SparseInputValue (const SparseInputValue <ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : InputValue<ElemType>(node, newName, flags)
        {
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new SparseInputValue<ElemType>(this, name, flags);
            return node;
        }

    private:
        void ConvertToSparseMatrix()
        {
            size_t rows = m_functionValues.GetNumRows();
            size_t cols = m_functionValues.GetNumCols();
            m_functionValues.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSC, false);
            m_functionValues.Resize(rows, cols); //SwitchToMatrixType does not reserve information right now.
        }

    };


    template class SparseInputValue<float>; 
    template class SparseInputValue<double>;

    //originally designed to extract word embedding representation from bag-of-word. 
    //takes two inputs, input0 is weight matrix and input1 is the bag-of-word representation of the inputs
    template<class ElemType>
    class LookupTableNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        LookupTableNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        LookupTableNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"LookupTable";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("LookupTable operation only takes two inputs.");

            DEVICEID_TYPE input1DeviceId = Inputs(1)->FunctionValues().GetDeviceId();
            DEVICEID_TYPE input0DeviceId = Inputs(0)->FunctionValues().GetDeviceId();
            Inputs(1)->FunctionValues().TransferFromDeviceToDevice(input1DeviceId, input0DeviceId);

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
            }
            else  //right derivative
            {
                ComputeInputPartialRight(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues());
            }
            Inputs(1)->FunctionValues().TransferFromDeviceToDevice(input0DeviceId, input1DeviceId);
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("LookupTable operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
        {
                Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialLeft(sliceInput1Value, Inputs(0)->GradientValues(), sliceOutputGrad);
            }
            else  //right derivative
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialRight(Inputs(0)->FunctionValues(), sliceInput1Grad, sliceOutputGrad);
            }
        }

        static void WINAPI ComputeInputPartialLeft(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues)  
        {
            size_t rows1 =inputFunctionValues.GetNumRows(), cols1 = inputFunctionValues.GetNumCols();
            size_t rowsp = gradientValues.GetNumRows(), colsp = gradientValues.GetNumCols();
            int wordsInEachSample = rows1 / inputGradientValues.GetNumCols();

            inputFunctionValues.Reshape(rows1 / wordsInEachSample, cols1 * wordsInEachSample);
            gradientValues.Reshape(rowsp / wordsInEachSample, colsp * wordsInEachSample);

            Matrix<ElemType>::MultiplyAndAdd(gradientValues, false, inputFunctionValues, true, inputGradientValues);

            inputFunctionValues.Reshape(rows1, cols1);
            gradientValues.Reshape(rowsp, colsp);
        }

        static void WINAPI ComputeInputPartialRight(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues)  
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

        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
#ifdef DEBUG_DECODER
            fprintf(stderr, "LookupTableNode node %ls: Input[0]=%.8e Input[1]=%.8e output = %.8e\n", this->NodeName().c_str(), Inputs(0)->FunctionValues().FrobeniusNorm(), Inputs(1)->FunctionValues().FrobeniusNorm(), FunctionValues().FrobeniusNorm());
#endif
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value);
        }

        //input0 is the weight (each column is an embedding of one word), input 1 contains m_bnrLooked words in each column (sample)
        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, Matrix<ElemType>& input1)  
        {
            size_t rows1 = input1.GetNumRows(), cols1 = input1.GetNumCols();
            size_t cols0 = input0.GetNumCols();

            if (rows1 % cols0 != 0)
                LogicError("LookupTableNode: rows of input 1 and cols of input 0 are not modular. e.g., rows1 = 0.9 cols and this is not allowed. Check feature reader and network definition. This usually happens when the feature dimension is not specified as that in the network definition of look-up-table dimension size. ");

            int wordsInEachSample = rows1 / cols0;

            input1.Reshape(rows1 / wordsInEachSample, cols1 * wordsInEachSample);

            DEVICEID_TYPE input1DeviceId = input1.GetDeviceId();
            DEVICEID_TYPE input0DeviceId = input0.GetDeviceId();
            input1.TransferFromDeviceToDevice(input1DeviceId, input0DeviceId);

            functionValues.AssignProductOf(input0, false, input1, false);

            input1.TransferFromDeviceToDevice(input0DeviceId, input1DeviceId);

            input1.Reshape(rows1, cols1);
            size_t rows = functionValues.GetNumRows();
            functionValues.Reshape(rows * wordsInEachSample, cols1);
        }
            
        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (Inputs(1)->FunctionValues().GetNumRows() % Inputs(0)->FunctionValues().GetNumCols() != 0)
                throw invalid_argument("Mismatched dimention. rows in input1 must be multiples of cols in input0.");

            int wordsInEachSample = Inputs(1)->FunctionValues().GetNumRows() / Inputs(0)->FunctionValues().GetNumCols();
          
            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows() * wordsInEachSample, Inputs(1)->FunctionValues().GetNumCols());

            CopyImageSizeFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
            ComputationNodePtr node = new LookupTableNode<ElemType>(this, name, flags);
            return node;
        }

        LookupTableNode(const LookupTableNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }
        
        bool UnitTest()
        {
            try{
                size_t nInput = 2;
                size_t nHidden = 3;
                size_t nOutput = 3;

                Inputs(0)->FunctionValues().Resize(nInput, nHidden);
                Inputs(0)->FunctionValues().SetValue(1.0);
                Inputs(1)->FunctionValues().TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true);
                Inputs(1)->FunctionValues().SwitchToMatrixType(DENSE, matrixFormatDense, false);
                Inputs(1)->FunctionValues().Resize(nHidden, nOutput);
                Inputs(1)->FunctionValues().SetValue(0.0);
                Inputs(1)->FunctionValues().SetValue(0, 0, 1.0);
                Inputs(1)->FunctionValues().SetValue(1, 1, 2.0);
                Inputs(1)->FunctionValues().TransferFromDeviceToDevice(CPUDEVICE, m_deviceId, true);
                Inputs(1)->FunctionValues().SwitchToMatrixType(SPARSE, matrixFormatSparseCSC, true);
                FunctionValues().Resize(nInput, nOutput);

                EvaluateThisNode();

                /// check with expected values
                FunctionValues().TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true);
                if (!ISCLOSE(FunctionValues()(0, 0), 1.0, EPSILON) ||
                    !ISCLOSE(FunctionValues()(0, 1), 2.0, EPSILON) ||
                    !ISCLOSE(FunctionValues()(1, 1), 2.0, EPSILON) )
                    throw("LSTMNode forward computation error");

                if (FunctionValues().GetDeviceId() != m_deviceId)
                    FunctionValues().TransferFromDeviceToDevice(FunctionValues().GetDeviceId(), m_deviceId, true);

                GradientValues().Resize(nInput, nOutput);
                GradientValues().SetValue(1.0);
                for (size_t i = 0; i < 2; i++)
                {
                    Inputs(i)->GradientValues().Resize(Inputs(i)->FunctionValues().GetNumRows(), Inputs(i)->FunctionValues().GetNumCols());
                    Inputs(i)->GradientValues().SetValue(0);
                }
                for (size_t i = 0; i < 2; i++)
                    ComputeInputPartial(i);

                /// check with expected values
                if (!ISCLOSE(Inputs(1)->GradientValues()(0, 0), 2, EPSILON) /// bi
                    || !ISCLOSE(Inputs(1)->GradientValues()(0, 1), 2, EPSILON)  // Wxi
                    || !ISCLOSE(Inputs(1)->GradientValues()(1, 0), 2, EPSILON)  // Whi
                    || !ISCLOSE(Inputs(1)->GradientValues()(2, 1), 2, EPSILON)  // Wci
                    )
                    throw("LSTMNode gradient error on input gates");

                for (size_t i = 0; i < 2; i++)
                {
                    if (Inputs(i)->GradientValues().GetDeviceId() != m_deviceId)
                        Inputs(i)->GradientValues().TransferFromDeviceToDevice(Inputs(i)->GradientValues().GetDeviceId(), m_deviceId, true);
                }

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

    /**
    pair this node to a node in another network
    */
    template<class ElemType>
    class PairNetworkNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        PairNetworkNode(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_reqMultiSeqHandling = true;
            m_functionValues.Resize(1, 1);
            InitRecurrentNode();
        }

        PairNetworkNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);

            m_functionValues.Resize(1, 1);
            m_reqMultiSeqHandling = true;

            LoadFromFile(fstream, modelVersion, deviceId);
        }

        PairNetworkNode(const DEVICEID_TYPE deviceId, size_t row_size, size_t col_size, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_reqMultiSeqHandling = true;

            m_functionValues.Resize(row_size, col_size);

            m_gradientValues.Resize(row_size, col_size);
            m_gradientValues.SetValue(0.0f);

            InitRecurrentNode();
        }

        virtual const std::wstring OperationName() const { return TypeName(); }

        /// to-do: need to change to the new way of resetting state
        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("PairNetwork operation only takes one input.");

            Matrix<ElemType>::ScaleAndAdd(1.0, GradientValues(), Inputs(inputIndex)->GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("Delay operation only takes one input.");
            assert(m_functionValues.GetNumRows() == GradientValues().GetNumRows()); // original used m_functionValues.GetNumRows() for loop dimension
            assert(m_sentenceSeg != nullptr);
            assert(m_existsSentenceBeginOrNoLabels != nullptr);

            Matrix<ElemType> mTmp = Inputs(inputIndex)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType>::ScaleAndAdd(1.0, GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep), 
                mTmp);
        }

        virtual void EvaluateThisNode()
        {
            m_functionValues.SetValue(Inputs(0)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            Matrix<ElemType> mTmp = FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            mTmp.SetValue(Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep));
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation(true);

            if (m_children.size() != 1)
                throw std::logic_error("PairNetwork operation should have one input.");

            if (!(Inputs(0) == nullptr))
            {
                size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();

                if (rows0 > 0 && cols0 > 0) FunctionValues().Resize(rows0, cols0);
            }
            CopyImageSizeFromInputs();
        }

        virtual void AttachInputs(const ComputationNodePtr inputNode)
        {
            m_children.resize(1);
            m_children[0] = inputNode;
        }

        void EnumerateNodesForEval(std::unordered_set<ComputationNodePtr>& visited, std::list<ComputationNodePtr>& result,
            std::vector<ComputationNodePtr>& sourceRecurrentNodePtr, const bool bFromDelayNode)
        {
            if (visited.find(this) == visited.end())  //not visited
            {
                visited.insert(this);   // have visited tagged here to avoid infinite loop over children, children's children, etc

                //children first for function evaluation
                if (!IsLeaf())
                {
                    if (ChildrenNeedGradient())  //only nodes that require gradient calculation is included in gradient calculation
                        m_needGradient = true;
                    else
                        m_needGradient = false;
                }

                result.push_back(ComputationNodePtr(this));  //we put this in the list even if it's leaf since we need to use it to determine learnable params 
                this->m_visitedOrder = result.size();
            }
            else
            {
                if (!IsLeaf() && bFromDelayNode)
                    sourceRecurrentNodePtr.push_back(this);
            }
        }

        static const std::wstring TypeName() { return L"PairNetwork"; }

        // copy constructor
        PairNetworkNode(const PairNetworkNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;

            ComputationNodePtr node = new PairNetworkNode<ElemType>(this, name, flags);
            return node;
        }

    protected:
        virtual bool UseCustomizedMultiSeqHandling() { return true; }

    };

    template class PairNetworkNode<float>;
    template class PairNetworkNode<double>;

}}}
