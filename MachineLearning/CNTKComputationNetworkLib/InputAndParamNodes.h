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
#include "File.h"   // for LoadMatrixFromTextFile()
#include "ComputationNode.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    //used to represent weight Matrix<ElemType> and biases
    template<class ElemType>
    class LearnableParameter : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        LearnableParameter(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        {
            m_needGradient = true;
            m_outputWidth = 1;
            m_outputHeight = SIZE_MAX;
            m_outputChannels = 1;
        }
        LearnableParameter(DEVICEID_TYPE deviceId, const wstring & name, size_t rows, size_t cols) :
            ComputationNode<ElemType>(deviceId, name)
        {
            m_needGradient = true;
            m_outputWidth = 1;
            m_outputHeight = rows;
            m_outputChannels = 1;
            m_functionValues.Resize(rows, cols);
        }

        virtual void SaveToFile(File& fstream) const
        {
            Base::SaveToFile(fstream);
            fstream << m_needGradient;
            fstream << FunctionValues().GetNumRows() << FunctionValues().GetNumCols(); 
            fstream << FunctionValues();
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion)
        {
            Base::LoadFromFile(fstream, modelVersion);

            size_t rows, cols;
            fstream >> m_needGradient;
            fstream >> rows >> cols;

            //intentionally comment out to support automatic dimension inference
            //if (rows * cols == 0) 
            //    LogicError("This LearnableParameter dimension is 0.");

            m_functionValues.Resize(rows, cols);
            fstream >> m_functionValues;

            m_outputWidth = 1;
            m_outputHeight = rows;
            m_outputChannels = 1;
        }

        // initialize with random numbers
        void InitRandom(const bool uniformInit,
                        const unsigned long randomSeed,
                        const ElemType initValueScale,
                        bool initOnCPUOnly) // if true then always init on CPU, making initialization consistent across both (for testing)
        {
            size_t inputSize = FunctionValues().GetNumCols();

            // the random seed offset is set via the "randomSeedOffset" parameter in config
            if (initOnCPUOnly)
                m_functionValues.TransferToDeviceIfNotThereAndNotAutoPlace(CPUDEVICE, true);
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
                m_functionValues.TransferToDeviceIfNotThereAndNotAutoPlace(m_deviceId, true);
        }

        // initialize by reading a matrix from a text file
        void InitFromFile(const std::wstring & initFromFilePath)
        {
            size_t numRows = 0;
            size_t numCols = 0;
            auto array = File::LoadMatrixFromTextFile<ElemType>(msra::strfun::utf8(initFromFilePath), numRows, numCols); // TODO: change pathname to wstring
            FunctionValues().SetValue(numRows, numCols, array.data(), matrixFlagNormal, m_deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}

        virtual void ComputeInputPartial(const size_t /*inputIndex*/) {}
        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange &) {}
        virtual void EvaluateThisNode() {}
        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange &) {}

        static const std::wstring TypeName() {return L"LearnableParameter";} 

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            Base::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "[%lu,%lu]  ", FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            fstream << string(str);
            sprintf(str, "NeedGradient=%s", NeedGradient()? "true" : "false");
            fstream << string(str);

            PrintNodeValuesToFile(printValues, fstream);
        }
    };

    //WARNING: Don't use SparseLearnableParameter yet since the current version assumes the parameter is dense instead of sparse
    //WARNING: After the right implementation is put here we need to turn it on in NetworkDescriptionLangauge.cpp
    template<class ElemType>
    class SparseLearnableParameter : public LearnableParameter<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        SparseLearnableParameter(DEVICEID_TYPE deviceId, const wstring & name) :
            LearnableParameter<ElemType>(deviceId, name)
        {
            m_gradientValues.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol, false);
        }
        SparseLearnableParameter(DEVICEID_TYPE deviceId, const wstring & name, size_t rows, size_t cols, size_t size) :
            LearnableParameter<ElemType>(deviceId, name, rows, cols)
        {
            m_gradientValues.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol, false);
            m_gradientValues.Resize(rows, cols, size);
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion)
        {
            LearnableParameter<ElemType>::LoadFromFile(fstream, modelVersion);
            m_gradientValues.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol, false);       // TODO: needed? Constructor already sets this
            m_gradientValues.Resize(FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
        }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"SparseLearnableParameter"; }
    };

    template class SparseLearnableParameter<float>; 
    template class SparseLearnableParameter<double>;

    template<class ElemType>
    class InputValue : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
        void Init(size_t rows, size_t cols, bool isSparse)
        {
            m_isSparse = isSparse;
            if (isSparse)
                ConvertToSparseMatrix();

            m_functionValues.Resize(rows, cols);
            m_needGradient = false;
        }
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        InputValue(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        {
            m_outputWidth = SIZE_MAX;
            m_outputHeight = SIZE_MAX;
            m_outputChannels = SIZE_MAX;
            Init(0, 0, false);
        }
        InputValue(DEVICEID_TYPE deviceId, const wstring & name, bool isSparse) :
            ComputationNode<ElemType>(deviceId, name)
        {
            m_outputWidth = SIZE_MAX;
            m_outputHeight = SIZE_MAX;
            m_outputChannels = SIZE_MAX;
            Init(0, 0, isSparse);
        }
        // ^^ TODO: merge the two above with optional arg
        InputValue(DEVICEID_TYPE deviceId, const wstring & name, size_t rows, size_t cols, bool isSparse = false) :
            ComputationNode<ElemType>(deviceId, name)
        {
            if (rows * cols == 0)
                LogicError("This InputValue dimension is 0.");

            m_outputWidth = 1;
            m_outputHeight = rows;
            m_outputChannels = 1;

            Init(rows, cols, isSparse);
        }
        InputValue(DEVICEID_TYPE deviceId, const wstring & name, size_t imageWidth, size_t imageHeight, size_t imageChannels, size_t numImages, bool isSparse = false) :
            ComputationNode<ElemType>(deviceId, name)
        {
            size_t rows = imageWidth * imageHeight * imageChannels;
            size_t cols = numImages;

            if (rows * cols == 0)
                LogicError("This InputValue dimension is 0.");

            m_outputWidth = imageWidth;
            m_outputHeight = imageHeight;
            m_outputChannels = imageChannels;

            Init(rows, cols, isSparse);
        }

        virtual void SaveToFile(File& fstream) const
        {
            Base::SaveToFile(fstream);
            fstream << FunctionValues().GetNumRows() << FunctionValues().GetNumCols(); 
            fstream << m_outputWidth << m_outputHeight << m_outputChannels; 
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion)
        {
            Base::LoadFromFile(fstream, modelVersion);

            size_t rows, cols;
            fstream >> rows >> cols;
            if (rows * cols == 0) 
                LogicError("This InputValue dimension is 0.");

            fstream >> m_outputWidth >> m_outputHeight >> m_outputChannels; 

            if (m_isSparse)
                ConvertToSparseMatrix();

            m_functionValues.Resize(rows, cols);
            m_needGradient = false;
        }

        // TODO: This is bad. We should either serialize m_isSparse or define an explicit node type; this special-casing will cause grief
        virtual const std::wstring OperationName() const { return m_isSparse ? SparseTypeName() : TypeName(); }

        static const std::wstring TypeName() {return L"InputValue";} 
        static const std::wstring SparseTypeName() {return L"SparseInputValue";}    // special case used by old NDL

        virtual void EvaluateThisNode()  {} 
        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange &) {}

        virtual void ComputeInputPartial(const size_t /*inputIndex*/) {}
        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange &) {}

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            Base::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "[%lu,%lu]", FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            fstream << string(str);        
        }
    private:
        bool m_isSparse = false;
        void ConvertToSparseMatrix()
        {
            size_t rows = m_functionValues.GetNumRows();
            size_t cols = m_functionValues.GetNumCols();
            m_functionValues.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSC, false);
            m_functionValues.Resize(rows, cols); //SwitchToMatrixType does not reserve information right now.
        }
    };

    template class InputValue<float>; 
    template class InputValue<double>;

    //originally designed to extract word embedding representation from bag-of-word. 
    //takes two inputs, input0 is weight matrix and input1 is the bag-of-word representation of the inputs
    template<class ElemType>
    class LookupTableNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        LookupTableNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"LookupTable";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("LookupTable operation only takes two inputs.");

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

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex > 1)
                InvalidArgument("LookupTable operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialLeft(sliceInput1Value, Inputs(0)->GradientValues(), sliceOutputGrad);
            }
            else  //right derivative
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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
            
        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (Inputs(1)->FunctionValues().GetNumRows() % Inputs(0)->FunctionValues().GetNumCols() != 0)
                throw invalid_argument("Mismatched dimention. rows in input1 must be multiples of cols in input0.");

            int wordsInEachSample = Inputs(1)->FunctionValues().GetNumRows() / Inputs(0)->FunctionValues().GetNumCols();
          
            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows() * wordsInEachSample, Inputs(1)->FunctionValues().GetNumCols());

            InferImageDimsFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }

        bool UnitTest()
        {
            try
            {
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

                FunctionValues().TransferToDeviceIfNotThere( m_deviceId, true);

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

    /**
    pair this node to a node in another network
    this node provide an interface from this network. The next layer network then can use this interface to know which node to connect to.
    */
    template<class ElemType>
    class PairNetworkNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
        void Init(size_t row_size, size_t col_size)
        {
            m_reqMultiSeqHandling = true;
            m_functionValues.Resize(row_size, col_size);
        }
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        PairNetworkNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        {
            Init(1, 1); // TODO: do we not need to resize m_gradientValues?
        }
        PairNetworkNode(DEVICEID_TYPE deviceId, const wstring & name, size_t row_size, size_t col_size) :
            ComputationNode<ElemType>(deviceId, name)
        {
            Init(row_size, col_size);
            m_gradientValues.Resize(row_size, col_size);
            m_gradientValues.SetValue(0.0f);
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion)
        {
            Init(1, 1); // TODO: this looks wrong; should the dimension not come from the loaded model data?
            Base::LoadFromFile(fstream, modelVersion);
        }

        virtual const std::wstring OperationName() const { return TypeName(); }

        /// to-do: need to change to the new way of resetting state
        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                InvalidArgument("PairNetwork operation only takes one input.");

            Matrix<ElemType>::ScaleAndAdd(1.0, GradientValues(), Inputs(inputIndex)->GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex > 0)
                InvalidArgument("Delay operation only takes one input.");
            assert(m_functionValues.GetNumRows() == GradientValues().GetNumRows()); // original used m_functionValues.GetNumRows() for loop dimension
            assert(m_sentenceSeg != nullptr);

            Matrix<ElemType> mTmp = Inputs(inputIndex)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType>::ScaleAndAdd(1.0, GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep), 
                mTmp);
        }

        virtual void EvaluateThisNode()
        {
            m_functionValues.SetValue(Inputs(0)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            Matrix<ElemType> mTmp = FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            mTmp.SetValue(Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep));
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 1)
                LogicError("PairNetwork operation should have one input.");

            if (!(Inputs(0) == nullptr))
            {
                size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();

                if (rows0 > 0 && cols0 > 0) FunctionValues().Resize(rows0, cols0);
            }
            InferImageDimsFromInputs();
        }

        virtual void AttachInputs(const ComputationNodePtr inputNode)
        {
            m_children.resize(1);
            m_children[0] = inputNode;
        }

        virtual void EnumerateNodesForEval(std::unordered_set<ComputationNodePtr>& visited, std::list<ComputationNodePtr>& result,
            std::vector<ComputationNodePtr>& sourceRecurrentNodePtr, const bool bFromDelayNode)
        {
            if (visited.find(shared_from_this()) == visited.end())  //not visited
            {
                visited.insert(shared_from_this());   // have visited tagged here to avoid infinite loop over children, children's children, etc

                //children first for function evaluation
                if (!IsLeaf())
                {
                    if (ChildrenNeedGradient())  //only nodes that require gradient calculation is included in gradient calculation
                        m_needGradient = true;
                    else
                        m_needGradient = false;
                }

                result.push_back(shared_from_this());  //we put this in the list even if it's leaf since we need to use it to determine learnable params 
                this->m_visitedOrder = result.size();
            }
            else
            {
                if (!IsLeaf() && bFromDelayNode)
                    sourceRecurrentNodePtr.push_back(shared_from_this());
            }
        }

        static const std::wstring TypeName() { return L"PairNetwork"; }
protected:
        virtual bool UseCustomizedMultiSeqHandling() { return true; }

    };

    template class PairNetworkNode<float>;
    template class PairNetworkNode<double>;

}}}
