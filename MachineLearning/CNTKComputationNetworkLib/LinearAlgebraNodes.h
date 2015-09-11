//
// <copyright file="LinearAlgebraNodes.h" company="Microsoft">
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

    template<class ElemType>
    class NegateNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        NegateNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Negate";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                InvalidArgument("Negate operation only has one input.");
            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex != 0)
                InvalidArgument("Negate operation only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(sliceInputGrad, sliceOutputGrad);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& childGradientValues, const Matrix<ElemType>& gradientValues)
        {
            childGradientValues -= gradientValues;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) 
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input)  
        {
            functionValues.AssignDifferenceOf(0, input);
#if NANCHECK
            functionValues.HasNan("Negate");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 1) 
                throw std::logic_error("Negate operation should have one input.");

            if (Inputs(0)->FunctionValues().HasNoElements())
                throw std::logic_error("Negate operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            
            InferImageDimsFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }
    };

    template class NegateNode<float>; 
    template class NegateNode<double>;

    template<class ElemType>
    class SumElementsNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        SumElementsNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"SumElements";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                InvalidArgument("SumElements only has one input.");
            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex != 0)
                InvalidArgument("SumElements only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(sliceInputGrad, sliceOutputGrad);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            inputGradientValues += gradientValues; //here the assumption is that gradientValues are 1x1 matrix
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.AssignSumOfElements(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("SumElements");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 1) 
                throw std::logic_error("SumElements operation should have one input.");

            if (Inputs(0)->FunctionValues().HasNoElements())
                throw std::logic_error("SumElements operation: the input node has 0 element.");

            FunctionValues().Resize(1, 1);
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            m_outputWidth = 1;
            m_outputHeight = 1;        
            m_outputChannels = 1;
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }
    };

    template class SumElementsNode<float>; 
    template class SumElementsNode<double>;

    template<class ElemType>
    class SumColumnElementsNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        SumColumnElementsNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_sumValue(deviceId)
        { }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"SumColumnElements"; }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                InvalidArgument("SumColumnElements only has one input.");
            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex != 0)
                InvalidArgument("SumColumnElements only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(sliceInputGrad, sliceOutputGrad);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
        {
            inputGradientValues += gradientValues; //here the assumption is that gradientValues is a row vector
        }

        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)
        {
            Matrix<ElemType>::VectorSum(inputFunctionValues, functionValues, true);
#if NANCHECK
            functionValues.HasNan("SumColumnElements");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 1)
                throw std::logic_error("SumColumnElements operation should have one input.");

            if (Inputs(0)->FunctionValues().HasNoElements())
                throw std::logic_error("SumColumnElements operation: the input node has 0 element.");

            FunctionValues().Resize(1, Inputs(0)->FunctionValues().GetNumCols());
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            m_outputWidth = 1;
            m_outputHeight = 1;
            m_outputChannels = 1;
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput)
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<SumColumnElementsNode<ElemType>>(nodeP);
                node->m_sumValue = m_sumValue;
            }
        }

    private:
        Matrix<ElemType> m_sumValue;
    };

    template class SumColumnElementsNode<float>;
    template class SumColumnElementsNode<double>;

    //this node is used to extract part of the input by rows as the output
    //it has to be continuous segments of rows since each column is treated as one sample
    template<class ElemType>
    class RowSliceNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        //RowSliceNode(DEVICEID_TYPE deviceId, const wstring & name) :
        //    ComputationNode<ElemType>(deviceId, name),
        //    m_startIndex(0),
        //    m_numRows(0)
        //{ }
        RowSliceNode(DEVICEID_TYPE deviceId, const wstring & name, size_t startIndex = 0, size_t numRows = 0) :
            ComputationNode<ElemType>(deviceId, name),
            m_startIndex(startIndex),
            m_numRows(numRows)
        { }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            auto node = dynamic_pointer_cast<RowSliceNode<ElemType>>(nodeP);

            node->m_startIndex = m_startIndex;
            node->m_numRows = m_numRows;
        }

        virtual void SaveToFile(File& fstream) const
        {
            Base::SaveToFile(fstream);
            fstream << m_startIndex << m_numRows;
        }
        
        virtual void LoadFromFile(File& fstream, size_t modelVersion)
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_startIndex >> m_numRows;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"RowSlice";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                InvalidArgument("RowSlice only has one input.");
            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues(), m_startIndex, m_numRows);
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex != 0)
                InvalidArgument("RowSlice only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(sliceInputGrad, sliceOutputGrad, m_startIndex, m_numRows);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const size_t startIndex, const size_t numRows)  
        {
            inputGradientValues.AddToRowSliceValuesOf(gradientValues, startIndex, numRows); 
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues(), m_startIndex, m_numRows);
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue, m_startIndex, m_numRows);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues, const size_t startIndex, const size_t numRows)  
        {
            functionValues.AssignRowSliceValuesOf(inputFunctionValues, startIndex, numRows);
#if NANCHECK
            functionValues.HasNan("RowSlice");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 1) 
                throw std::logic_error("RowSlice operation should have one input.");

            if (Inputs(0)->FunctionValues().HasNoElements())
                throw std::logic_error("RowSlice operation: the input node has 0 element.");

            if (Inputs(0)->FunctionValues().GetNumRows() < m_startIndex + m_numRows)
                throw std::logic_error("RowSlice operation: m_startIndex + m_numRows exceeds number of rows in the input.");

            FunctionValues().Resize(m_numRows, Inputs(0)->FunctionValues().GetNumCols());
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, true);
            m_outputHeight = m_numRows;        

            //WARNING: this node will destroy the image size information from the child
            if (m_inputWidth * m_inputChannels != 1)
                fprintf(stderr, "WARNING: RowSlice operation cannot inherit image size information from its child. Image size info is lost.\n");
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

    private:
        size_t m_startIndex, m_numRows;
    };

    template class RowSliceNode<float>; 
    template class RowSliceNode<double>;

    //this node is used to extract part of the input by rows as the output
    //it has to be continuous segments of rows since each column is treated as one sample
    template<class ElemType>
    class RowStackNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        RowStackNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        { }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            auto node = dynamic_pointer_cast<RowStackNode<ElemType>>(nodeP);

            if (flags & CopyNodeFlags::copyNodeChildren)
            {
                node->m_children = m_children;
                node->m_startRowIndeces = m_startRowIndeces;
                node->m_inputMatrices = m_inputMatrices;
            }
        }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"RowStack"; }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex >= ChildrenSize())
                InvalidArgument("RowStack-ComputeInputPartial: inputIndex out of range.");
            ComputeInputPartialS(Inputs(inputIndex)->GradientValues(), GradientValues(), m_startRowIndeces[inputIndex], m_startRowIndeces[inputIndex + 1] - m_startRowIndeces[inputIndex]);
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex >= ChildrenSize())
                InvalidArgument("RowStack-ComputeInputPartial: inputIndex out of range.");

            Matrix<ElemType> sliceInputGrad = Inputs(inputIndex)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(sliceInputGrad, sliceOutputGrad, m_startRowIndeces[inputIndex], m_startRowIndeces[inputIndex+1] - m_startRowIndeces[inputIndex]);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const size_t startIndex, const size_t numRows)
        {
            inputGradientValues.AddWithRowSliceValuesOf(gradientValues, startIndex, numRows);
        }

        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(m_functionValues, m_inputMatrices,  0, Inputs(0)->FunctionValues().GetNumCols());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            Matrix<ElemType> sliceFunctionValues = FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceFunctionValues, m_inputMatrices, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const std::vector<const Matrix<ElemType>*>& inputMatrices, const size_t sliceStartCol, const size_t sliceNumCols)
        {
            functionValues.AssignRowStackValuesOf(inputMatrices, sliceStartCol, sliceNumCols);
#if NANCHECK
            functionValues.HasNan("RowStack");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() < 2)
                LogicError("RowStack operation: must have two or more inputs.");

            if (Inputs(0) == nullptr)
                LogicError("RowStack operation: the input node is NULL.");

            size_t numCols = Inputs(0)->FunctionValues().GetNumCols();
            m_startRowIndeces.resize(ChildrenSize()+1);
            m_inputMatrices.resize(ChildrenSize());

            size_t totalRows = 0;
            m_startRowIndeces[0] = 0;

            for (int i = 0; i < ChildrenSize(); i++)
            {
                if (Inputs(i) == nullptr)
                    LogicError("RowStack operation: the input node is NULL.");

                Matrix<ElemType>& childMatrix = Inputs(i)->FunctionValues();
                size_t numRows = childMatrix.GetNumRows();
                if (numRows == 0)
                    LogicError("RowStack operation: the input node %ls has 0 rows.", Inputs(i)->NodeName().c_str());
                
                if (childMatrix.GetNumCols() != numCols)
                    LogicError("RowStack operation: the input node %ls has different number of columns.", Inputs(i)->NodeName().c_str());

                totalRows += numRows;
                m_inputMatrices[i] = &childMatrix;
                m_startRowIndeces[i + 1] = m_startRowIndeces[i] + numRows;
            }

            FunctionValues().Resize(totalRows, numCols);
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, true);
            m_outputHeight = FunctionValues().GetNumRows();

            //WARNING: this node will destroy the image size information from the child
            if (m_inputWidth * m_inputChannels != 1)
                fprintf(stderr, "WARNING: RowStack operation cannot inherit image size information from its child. Image size info is lost.\n");
        }

        virtual void AttachInputs(const std::vector<ComputationNodePtr>& inputs)
        {
            unsigned int numInputs = inputs.size();
            m_children.resize(numInputs);
            for (unsigned int i = 0; i < numInputs; i++)
                m_children[i] = inputs[i];
        }

    private:
        std::vector<size_t> m_startRowIndeces; //start row number in the stacked matrix of each input (child)
        std::vector<const Matrix<ElemType>*> m_inputMatrices;
    };

    template class RowStackNode<float>;
    template class RowStackNode<double>;

    template<class ElemType>
    class ScaleNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        ScaleNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Scale";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("ScaleNode operation only takes two inputs.");

            //left Node must be a scalar Constant
            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
            }
            else
            {
                ComputeInputPartialRight(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues());
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex > 1)
                InvalidArgument("ScaleNode operation only takes two inputs.");

            //left Node must be a scalar
            if (inputIndex == 0)  //left derivative
            {
                Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialLeft(sliceInput1Value, Inputs(0)->GradientValues(), sliceOutputGrad);
            }
            else
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialRight(Inputs(0)->FunctionValues(), sliceInput1Grad, sliceOutputGrad);
            }
        }

        static void WINAPI ComputeInputPartialLeft(const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            inputGradientValues += Matrix<ElemType>::InnerProductOfMatrices(gradientValues, inputFunctionValues);
        }

        static void WINAPI ComputeInputPartialRight(const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            Matrix<ElemType>::ScaleAndAdd(inputFunctionValues.Get00Element(), gradientValues, inputGradientValues);
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)  
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, const Matrix<ElemType>& input1)  
        {
            functionValues.AssignProductOf(input0.Get00Element(), input1);
#if NANCHECK
            functionValues.HasNan("Scale");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 2) 
                throw std::logic_error("Scale operation requires two inputs.");

            if (Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements())
                throw std::logic_error("Scale operation: one of the operands has 0 element.");

            if (Inputs(0)->FunctionValues().GetNumRows() != 1 || Inputs(0)->FunctionValues().GetNumCols() != 1)
                throw std::logic_error("The left value of ScaleNode must be a scalar value.");

            FunctionValues().Resize(Inputs(1)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());
            //left Node must be a scalar
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(1); 
        }

        virtual void AttachInputs(const ComputationNodePtr scalarValue, const ComputationNodePtr Value) 
        {
            m_children.resize(2);
            m_children[0] = scalarValue;
            m_children[1] = Value;
        }
    };


    template class ScaleNode<float>; 
    template class ScaleNode<double>;

    template<class ElemType>
    class TimesNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        TimesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Times";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("Times operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
            }
            else  //right derivative
            {
                ComputeInputPartialRight(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues());
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex > 1)
                InvalidArgument("Times operation only takes two inputs.");

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

        static void WINAPI ComputeInputPartialLeft(const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
            {
#if DUMPOUTPUT
            gradientValues.Print("Gradient-in");
            inputGradientValues.Print("child Gradient-in/out");
            inputFunctionValues.Print("child Function values");
#endif
            //currently we only support one combination when the input is sparse.
            if (inputFunctionValues.GetMatrixType() == SPARSE && inputGradientValues.GetMatrixType() == DENSE && gradientValues.GetMatrixType() == DENSE)
                inputGradientValues.SwitchToMatrixType(SPARSE, MatrixFormat::matrixFormatSparseBlockCol, false);

                Matrix<ElemType>::MultiplyAndAdd(gradientValues, false, inputFunctionValues, true, inputGradientValues);
#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
        }

        static void WINAPI ComputeInputPartialRight(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
            {
#if DUMPOUTPUT
            gradientValues.Print("Gradient-in");
            inputGradientValues.Print("child Gradient-in/out");
            inputFunctionValues.Print("child Function values");
#endif
                Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, true, gradientValues, false, inputGradientValues);
#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
        }


        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
#ifdef DEBUG_DECODER
            fprintf(stderr, "Times node %ls output norm = %.8e, input(0) norm = %.8e, input(1) norm = %.8e\n", this->NodeName().c_str(), FunctionValues().FrobeniusNorm(), 
                Inputs(0)->FunctionValues().FrobeniusNorm(), Inputs(1)->FunctionValues().FrobeniusNorm());
#endif
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)  
        {
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();
            FunctionValues().Resize(rows0, cols1);

            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, const Matrix<ElemType>& input1)  
        {
#if DUMPOUTPUT
            input0.Print("TimesNode - Input0");
#endif
            functionValues.AssignProductOf(input0, false, input1, false);
#if NANCHECK
            functionValues.HasNan("Times");
#endif
#if DUMPOUTPUT
            functionValues.Print("TimesNode");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 2) 
                throw std::logic_error("Times operation requires two inputs.");

            //support automatic dimention inference for learnable parameters
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if ((rows0 == 0 || cols1 == 0 ) && this->LoopId() < 0)
                throw logic_error("Times operation: Inputs(0)->FunctionValues().GetNumRows() and Inputs(1)->FunctionValues().GetNumCols() should not be 0 since it cannot be automatically inferred");

            // TODO: use dynamic_pointer_cast
            // TODO: why should these nodes even care whether their inputs are LearnableParmaeters? If needed, can the base class do this?
            if ((Inputs(0)->OperationName() == OperationNameOf(LearnableParameter) && cols0 == 0 && rows1 != 0) && this->LoopId() < 0)
                Inputs(0)->FunctionValues().Resize(rows0, rows1);

            if (Inputs(1)->OperationName() == OperationNameOf(LearnableParameter) && cols0 != 0 && rows1 == 0)
                Inputs(1)->FunctionValues().Resize(cols0, cols1);

            if ((Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements())&& this->LoopId() < 0)
                throw std::logic_error("Times operation: One of the operants has 0 elements.");

            //cols0 and rows1 may have been changed so don't use them in the following check
            if ((Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumCols()) && this->LoopId() < 0)
            {
                throw std::logic_error("The Matrix dimension in the Times operation does not match.");
            }
            FunctionValues().Resize(rows0, cols1);
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs()  
        {
            InferImageDimsFromInput(1, false); //the second one is the input since it's column wize

            //after multiplication the structure is lost
            m_outputWidth = 1;
            m_outputHeight = Inputs(0)->FunctionValues().GetNumRows();
            m_outputChannels =  1;
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }
    };

    template class TimesNode<float>; 
    template class TimesNode<double>;

    template<class ElemType>
    class TransposeTimesNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        TransposeTimesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"TransposeTimes"; }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("TransposeTimesNode operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
            }
            else  //right derivative
            {
                ComputeInputPartialRight(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues());
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex > 1)
                InvalidArgument("TransposeTimesNode operation only takes two inputs.");

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

        static void WINAPI ComputeInputPartialLeft(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
        {
#if DUMPOUTPUT
            gradientValues.Print("Gradient-in");
            inputGradientValues.Print("child Gradient-in/out");
            inputFunctionValues.Print("child Function values");
#endif
            //currently we only support one combination when the input is sparse.
            if (inputFunctionValues.GetMatrixType() == SPARSE && inputGradientValues.GetMatrixType() == DENSE && gradientValues.GetMatrixType() == DENSE)
                inputGradientValues.SwitchToMatrixType(SPARSE, MatrixFormat::matrixFormatSparseBlockCol, false);

            Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, false, gradientValues, true, inputGradientValues);


#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
        }

        static void WINAPI ComputeInputPartialRight(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
        {
#if DUMPOUTPUT
            gradientValues.Print("Gradient-in");
            inputGradientValues.Print("child Gradient-in/out");
            inputFunctionValues.Print("child Function values");
#endif
            Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, false, gradientValues, false, inputGradientValues);

#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
        }

        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, const Matrix<ElemType>& input1)
        {
#if DUMPOUTPUT
            input0.Print("TransposeTimesNode - Input0");
#endif
            functionValues.AssignProductOf(input0, true, input1, false);
#if NANCHECK
            functionValues.HasNan("TransposeTimes");
#endif
#if DUMPOUTPUT
            functionValues.Print("TransposeTimes");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 2)
                throw std::logic_error("TransposeTimes operation requires two inputs.");

            //support automatic dimention inference for learnable parameters
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if ((rows0 == 0 || cols1 == 0) && this->LoopId() < 0)
                throw logic_error("TransposeTimes operation: Inputs(0)->FunctionValues().GetNumRows() and Inputs(1)->FunctionValues().GetNumCols() should not be 0 since it cannot be automatically inferred");

            if ((Inputs(0)->OperationName() == OperationNameOf(LearnableParameter) && cols0 == 0 && rows1 != 0) && this->LoopId() < 0)
                Inputs(0)->FunctionValues().Resize(rows0, rows1);

            if (Inputs(1)->OperationName() == OperationNameOf(LearnableParameter) && cols0 != 0 && rows1 == 0)
                Inputs(1)->FunctionValues().Resize(cols0, cols1);

            if ((Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements()) && this->LoopId() < 0)
                throw std::logic_error("TransposeTimes operation: One of the operants has 0 elements.");

            //cols0 and rows1 may have been changed so don't use them in the following check
            if ((Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumRows()) && this->LoopId() < 0)
            {
                throw std::logic_error("The Matrix dimension in the TransposeTimes operation does not match.");
            }
            FunctionValues().Resize(cols0, cols1);
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(1, false); //the second one is the input since it's column wize

            //after multiplication the structure is lost
            m_outputWidth = 1;
            m_outputHeight = Inputs(0)->FunctionValues().GetNumRows();
            m_outputChannels = 1;
        }


        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode)
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }
    };

    template class TransposeTimesNode<float>;
    template class TransposeTimesNode<double>;

    template<class ElemType>
    class ElementTimesNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        ElementTimesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"ElementTimes";} 

        virtual void ComputeInputPartial(const size_t inputIndex)  
        {
            if (inputIndex > 1)
                InvalidArgument("ElementTimes operation only takes two inputs.");

            ComputeInputPartialS(Inputs(1-inputIndex)->FunctionValues(), Inputs(inputIndex)->GradientValues(), GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)  
        {
            if (inputIndex > 1)
                InvalidArgument("ElementTimes operation only takes two inputs.");

            Matrix<ElemType> sliceInput0Grad = Inputs(inputIndex)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInput1Value = Inputs(1-inputIndex)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(sliceInput1Value, sliceInput0Grad, sliceOutputGrad);
        }

        // depending on inputIndex, all the input variables change meaning
        // inputIndex == 0 (left) -  inputGradientValues[0], inputFunctionValues[1]
        // inputIndex == 1 (right) - inputGradientValues[1], inputFunctionValues[0]
        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            inputGradientValues.AddElementProductOf(gradientValues, inputFunctionValues);

#if NANCHECK
            inputGradientValues.HasNan("ElementTimes");
#endif
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)  
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, const Matrix<ElemType>& input1)  
        {
            functionValues.AssignElementProductOf(input0, input1);

#if NANCHECK
            functionValues.HasNan("ElementTimes");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 2) 
                throw std::logic_error("ElementTimes operation requires two inputs.");

            //derive number of rows if possible
            for (size_t index = 0; index < 2; index++)
            {
                if (Inputs(index)->OperationName() == OperationNameOf(LearnableParameter))
                {
                    size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0 ? Inputs(1 - index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                    size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0 ? Inputs(1 - index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                    Inputs(index)->FunctionValues().Resize(rows, cols);
                }
            }

            if (Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements())
                throw std::logic_error("ElementTimes operation: one of the operants has 0 element.");

            if (Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumRows() ||
                Inputs(1)->FunctionValues().GetNumCols() != Inputs(0)->FunctionValues().GetNumCols())
                throw std::logic_error("The Matrix<ElemType> dimension in the ElementTimes operation does not match.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs()
        {
            if (IsChildAnImage(0))  //when conflict, give priority to child 0
                InferImageDimsFromInput(0);
            else
                InferImageDimsFromInput(1);
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }
    };

    template class ElementTimesNode<float>; 
    template class ElementTimesNode<double>;

    template<class ElemType>
    class RowElementTimesNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        RowElementTimesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_tempMatrix(deviceId)
        { }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"RowElementTimes"; }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("RowElementTimes operation only takes two inputs.");

            if (inputIndex == 0)
            {
                ComputeInputPartialLeftS(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues(), m_tempMatrix);
            }
            else
            {
                ComputeInputPartialRightS(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues(), m_tempMatrix);
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex > 1)
                InvalidArgument("RowElementTimes operation only takes two inputs.");

            Matrix<ElemType> sliceInput0Grad = Inputs(inputIndex)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInput1Value = Inputs(1 - inputIndex)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (inputIndex == 0)
            {
                ComputeInputPartialLeftS(sliceInput1Value, sliceInput0Grad, sliceOutputGrad, m_tempMatrix);
            }
            else
            {
                ComputeInputPartialRightS(sliceInput1Value, sliceInput0Grad, sliceOutputGrad, m_tempMatrix);
            }
        }

        //left (input 0) is a matrix
        static void WINAPI ComputeInputPartialLeftS(Matrix<ElemType>& input1FunctionValues,
            Matrix<ElemType>& input0GradientValues, 
            const Matrix<ElemType>& gradientValues, 
            Matrix<ElemType>& tempMatrix)
        {
            tempMatrix.SetValue(gradientValues);
            tempMatrix.RowElementMultiplyWith(input1FunctionValues);
            input0GradientValues += tempMatrix;

#if NANCHECK
            input0GradientValues.HasNan("RowElementTimes");
#endif
        }

        //right (input 1) is a row vector
        static void WINAPI ComputeInputPartialRightS(Matrix<ElemType>& input0FunctionValues, 
            Matrix<ElemType>& input1GradientValues, 
            const Matrix<ElemType>& gradientValues, 
            Matrix<ElemType>& tempMatrix)
        {
            tempMatrix.AssignInnerProductOf(gradientValues, input0FunctionValues, true);
            input1GradientValues += tempMatrix;

#if NANCHECK
            input1GradientValues.HasNan("RowElementTimes");
#endif
        }
        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, const Matrix<ElemType>& input1)
        {
            functionValues.SetValue(input0);
            functionValues.RowElementMultiplyWith(input1);

#if NANCHECK
            functionValues.HasNan("RowElementTimes");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 2)
                throw std::logic_error("RowElementTimes operation requires two inputs.");

            if (Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements())
                throw std::logic_error("RowElementTimes operation: one of the operants has 0 element.");

            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if (cols0 != cols1 || rows1 != 1)
                throw std::logic_error("RowElementTimes: Either the second operand is not a row vector or the number of columns of operands does not match.");

            FunctionValues().Resize(rows0, cols0);
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            //input 0 is the matrix and input 1 is a row vector
            InferImageDimsFromInput(0);
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode)
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            Base::MoveMatricesToDevice(deviceId);
            m_tempMatrix.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
        }

        private:
            Matrix<ElemType> m_tempMatrix;
    };

    template class RowElementTimesNode<float>;
    template class RowElementTimesNode<double>;

    template<class ElemType>
    class ColumnElementTimesNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        ColumnElementTimesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_tempMatrix(deviceId)
        { }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"ColumnElementTimes"; }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("ColumnElementTimes operation only takes two inputs.");

            if (inputIndex == 0)
            {
                ComputeInputPartialLeftS(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues(), m_tempMatrix);
            }
            else
            {
                ComputeInputPartialRightS(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues(), m_tempMatrix);
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex > 1)
                InvalidArgument("ColumnElementTimes operation only takes two inputs.");

            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (inputIndex == 0)
            {
                Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialLeftS(Inputs(1)->FunctionValues(), sliceInput0Grad, sliceOutputGrad, m_tempMatrix);
            }
            else
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                ComputeInputPartialRightS(sliceInput0Value, Inputs(1)->GradientValues(), sliceOutputGrad, m_tempMatrix);
            }
        }

        //left (input 0) is a matrix
        static void WINAPI ComputeInputPartialLeftS(Matrix<ElemType>& input1FunctionValues,
            Matrix<ElemType>& input0GradientValues,
            const Matrix<ElemType>& gradientValues,
            Matrix<ElemType>& tempMatrix)
        {
            tempMatrix.SetValue(gradientValues);
            tempMatrix.ColumnElementMultiplyWith(input1FunctionValues);
            input0GradientValues += tempMatrix;

#if NANCHECK
            input0GradientValues.HasNan("ColumnElementTimes");
#endif
        }

        //right (input 1) is a col vector
        static void WINAPI ComputeInputPartialRightS(Matrix<ElemType>& input0FunctionValues,
            Matrix<ElemType>& input1GradientValues,
            const Matrix<ElemType>& gradientValues,
            Matrix<ElemType>& tempMatrix)
        {
            tempMatrix.AssignInnerProductOf(gradientValues, input0FunctionValues, false);
            input1GradientValues += tempMatrix;

#if NANCHECK
            input1GradientValues.HasNan("ColumnElementTimes");
#endif
        }
        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, Inputs(1)->FunctionValues());
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, const Matrix<ElemType>& input1)
        {
            functionValues.SetValue(input0);
            functionValues.ColumnElementMultiplyWith(input1);

#if NANCHECK
            functionValues.HasNan("ColumnElementTimes");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 2)
                throw std::logic_error("ColumnElementTimes operation requires two inputs.");

            //derive number of rows if possible
            for (size_t index = 0; index < 2; index++)
            {
                if (Inputs(index)->OperationName() == OperationNameOf(LearnableParameter))
                {
                    size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0 ? Inputs(1 - index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                    size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0 ? Inputs(1 - index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                    Inputs(index)->FunctionValues().Resize(rows, cols);
                }
            }

            if (Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements())
                throw std::logic_error("ColumnElementTimes operation: one of the operants has 0 element.");

            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if (rows0 != rows1 || cols1 != 1)
                throw std::logic_error("ColumnElementTimes: Either the second operand is not a column vector or the number of rows of operands does not match.");

            FunctionValues().Resize(rows0, cols0);
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            //input 0 is the matrix and input 1 is a column vector
            InferImageDimsFromInput(0);
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode)
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            Base::MoveMatricesToDevice(deviceId);
            m_tempMatrix.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
        }

    private:
        Matrix<ElemType> m_tempMatrix;
    };

    template class ColumnElementTimesNode<float>;
    template class ColumnElementTimesNode<double>;

    template<class ElemType>
    class PlusNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        PlusNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Plus";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("Plus operation only takes two inputs.");
            ComputationNodePtr child = Inputs(inputIndex);
            ComputeInputPartialS(FunctionValues(), GradientValues(), child->FunctionValues(), child->GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex > 1)
                InvalidArgument("Plus operation only takes two inputs.");

            //only the one with more columns can be sliced, if both have same columns both are sliced
            size_t cols0 = Inputs(inputIndex)->FunctionValues().GetNumCols(), cols1=Inputs(1-inputIndex)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (cols0 >= cols1)
            {
                Matrix<ElemType> sliceInput0Grad = Inputs(inputIndex)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput0Value = Inputs(inputIndex)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialS(sliceOutputValue, sliceOutputGrad, sliceInput0Value, sliceInput0Grad);
            }
            else 
            {
                ComputeInputPartialS(sliceOutputValue, sliceOutputGrad, Inputs(inputIndex)->FunctionValues(), Inputs(inputIndex)->GradientValues());
            }
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& functionValues, Matrix<ElemType>& gradientValues, Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues)
        {
#if DUMPOUTPUT

            functionValues.Print("PlusNode");
#endif

            size_t rowsc = inputFunctionValues.GetNumRows(), colsc = inputFunctionValues.GetNumCols();
            size_t rowsp = functionValues.GetNumRows(), colsp = functionValues.GetNumCols();
#if DUMPOUTPUT
            fprintf(stderr, "input dimensions %lld x %lld,  this node dimensions %lld x %lld\n", rowsc, colsc, rowsp, colsp);
            gradientValues.Print("Gradient-in");
            inputGradientValues.Print("child Gradient-in/out");
#endif

            if (colsc == colsp && rowsc == rowsp)
                inputGradientValues += gradientValues;
            else if (colsc == 1 && rowsc == 1)
                inputGradientValues += gradientValues.SumOfElements();
            else if (colsc == 1 && colsp != 1)
            {
                size_t colspExpand = rowsp*colsp/rowsc;
                gradientValues.Reshape(rowsc, colspExpand);
                Matrix<ElemType>::MultiplyAndAdd(gradientValues, false, ConstOnes(colspExpand, 1, functionValues.GetDeviceId()), false, inputGradientValues);
                gradientValues.Reshape(rowsp, colsp);
            }
            else if (rowsc == 1 && rowsp != 1)
                Matrix<ElemType>::MultiplyAndAdd(ConstOnes(1, rowsp,functionValues.GetDeviceId()), false, gradientValues, false, inputGradientValues);
            else if (colsc != 1 && colsp % colsc == 0)
            {
                /// the children matrix is [a b] and the parent considers it as [a a a b b b]
                size_t ratio = colsp / colsc; 
                for (size_t i = 0; i < colsc; i++)
                {
                    size_t colspExpand = rowsp*colsp / rowsc / colsc;
                    Matrix<ElemType> tmp = gradientValues.ColumnSlice(i * ratio, ratio);
                    tmp.Reshape(rowsc, colspExpand);
                    Matrix<ElemType> res = inputGradientValues.ColumnSlice(i, 1);
                    Matrix<ElemType>::MultiplyAndAdd(tmp, false, ConstOnes(colspExpand, 1, functionValues.GetDeviceId()), false, res);
                    inputGradientValues.ColumnSlice(i, 1).SetValue(res);
                }
            }
            else
                RuntimeError("Plus partial: unexpected condition.");
#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
        }


        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)  
        {
            size_t cols0 = Inputs(0)->FunctionValues().GetNumCols(), cols1=Inputs(1)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            //only the one with more columns can be sliced, if both have same columns both are sliced
            if (cols0 == cols1)
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value);
            }
            else if (cols0 > cols1)
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, Inputs(1)->FunctionValues());
            }
            else //cols0 < cols1)
            {
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value);
            }
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, Matrix<ElemType>& inputFunctionValues0, Matrix<ElemType>& inputFunctionValues1)  
        {
            size_t rows0 = inputFunctionValues0.GetNumRows(), cols0 = inputFunctionValues0.GetNumCols();
            size_t rows1 = inputFunctionValues1.GetNumRows(), cols1 = inputFunctionValues1.GetNumCols();
            functionValues.Resize(max(rows0, rows1), max(cols0,cols1));

            if ((rows0 == rows1 && cols0 == cols1) || ((rows0 == 1 || rows1 == 1) && cols0 == cols1))
            {
                functionValues.AssignSumOf(inputFunctionValues0, inputFunctionValues1);
            }
            else if (cols0 == 1 && rows1 % rows0 == 0)  //one is col vec with divisable rows, including scalar
            {
                inputFunctionValues1.Reshape(rows0, rows1 * cols1 / rows0);
                functionValues.AssignSumOf(inputFunctionValues0, inputFunctionValues1);
                inputFunctionValues1.Reshape(rows1, cols1);
                functionValues.Reshape(max(rows0, rows1), max(cols0,cols1));
            }
            else if (cols1 == 1 && rows0 % rows1 == 0)  //one is col vec with divisable rows, including scalar
            {
                inputFunctionValues0.Reshape(rows1, rows0 * cols0 / rows1);
                functionValues.AssignSumOf(inputFunctionValues0, inputFunctionValues1);
                inputFunctionValues0.Reshape(rows0, cols0);
                functionValues.Reshape(max(rows0, rows1), max(cols0,cols1));
            }       
            else if (cols1 < cols0 && rows0 == rows1 && cols0 % cols1 == 0)  //one is a matrix with number of columns that is a multiples of the column number of another matrix
            {
                /// the children matrix is [a b] and the parent considers it as [a a a b b b]
                Matrix<ElemType> tmpMat(inputFunctionValues1.GetDeviceId());
                size_t ratio = cols0 / cols1; 
                for (size_t i = 0; i < cols1; i++)
                {
                    tmpMat = Matrix<ElemType>::RepMat(inputFunctionValues1.ColumnSlice(i, 1), 1, ratio);
                    functionValues.ColumnSlice(i*ratio, ratio).SetValue(tmpMat + inputFunctionValues0.ColumnSlice(i * ratio, ratio)); 
                }
            }       
            else
            {
                LogicError("Plus node not supported format");
            }
#if NANCHECK
            functionValues.HasNan("Plus");
#endif
#if DUMPOUTPUT
            functionValues.Print("PlusNode");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 2) 
                throw std::logic_error("Plus operation requires two inputs.");

            //if dimention not specified we assume two operants' dimentions should be the same
            size_t index = 0;
            if (Inputs(index)->OperationName() == OperationNameOf(LearnableParameter))
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            index = 1;
            if (Inputs(index)->OperationName() == OperationNameOf(LearnableParameter))
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            if ((Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements()) && this->LoopId() < 0)
                throw std::logic_error("Plus operation: one of the operants has 0 element.");

            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if ((!(rows0 == rows1 && cols0 == cols1) &&  //match size
                !((rows0 == 1 || rows1 == 1) && cols0 == cols1) && //one is row vec
                !(  (cols0 > cols1 && cols0 % cols1 == 0) || 
                    (cols0 == 1 && rows1 % rows0 == 0) || 
                    (cols1 == 1 && rows0 % rows1 == 0))) && this->LoopId() < 0) //one is col vec with divisable rows, including scalar
            {
                LogicError("The Matrix dimension in the Plus operation does not match.");
            }       

            FunctionValues().Resize(max(rows0, rows1), max(cols0,cols1) );
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs() //based on the matrix with larger size
        {
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if (rows0 > rows1 || cols0 > cols1) //child 0 is larger
                InferImageDimsFromInput(0);
            else if (rows0 < rows1 || cols0 < cols1) //child 1 is larger
                InferImageDimsFromInput(1);
            else //same size
            {
                if (IsChildAnImage(0))  //when conflict, give priority to child 0
                    InferImageDimsFromInput(0);
                else
                    InferImageDimsFromInput(1);
            }
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }
    };

    template class PlusNode<float>; 
    template class PlusNode<double>;

    template<class ElemType>
    class MinusNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        MinusNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Minus";}

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("Minus operation only takes two inputs.");

            // prepare a matrix of ones as needed
            ComputationNodePtr child = Inputs(inputIndex);
            size_t rowsc = child->FunctionValues().GetNumRows(), colsc = child->FunctionValues().GetNumCols();
            size_t rowsp = FunctionValues().GetNumRows(), colsp = FunctionValues().GetNumCols();

            Matrix<ElemType> ones = Matrix<ElemType>();
            if (colsc == 1 && colsp != 1)
            {
                size_t colspExpand = rowsp*colsp/rowsc;
                ones = ConstOnes(colspExpand, 1,FunctionValues().GetDeviceId());
            }
            else if (rowsc == 1 && rowsp != 1)
            {
                ones = ConstOnes(1, rowsp,FunctionValues().GetDeviceId());
            }

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(child->FunctionValues(), child->GradientValues(), FunctionValues(), GradientValues(), ones); 
            }
            else  //right derivative
        {
                ComputeInputPartialRight(child->FunctionValues(), child->GradientValues(), FunctionValues(), GradientValues(), ones); 
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) 
        {
            //only the one with more columns can be sliced, if both have same columns both are sliced
            size_t cols0 = Inputs(inputIndex)->FunctionValues().GetNumCols(), cols1=Inputs(1-inputIndex)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInput0Grad = Inputs(inputIndex)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput0Value = Inputs(inputIndex)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> ones = Matrix<ElemType>();

            size_t rowsc = Inputs(inputIndex)->FunctionValues().GetNumRows(), rowsp = FunctionValues().GetNumRows();
            size_t colsp = FunctionValues().GetNumCols();

            if (cols0 >= cols1) //indicates cols0 == functionValue.cols
            {

                if (rowsc == 1 && rowsp != 1)
        {
                    ones = ConstOnes(1, rowsp, FunctionValues().GetDeviceId());
                }
    
                if (inputIndex == 0)  //left derivative
            {
                    ComputeInputPartialLeft(sliceInput0Value, sliceInput0Grad, sliceOutputValue, sliceOutputGrad, ones); 
            }
                else  //right derivativeAzqz
            {
                    ComputeInputPartialRight(sliceInput0Value, sliceInput0Grad, sliceOutputValue, sliceOutputGrad, ones); 
            }
            }
            else // cols0 < cols1 -> cols0=1
            {
                if (cols0 == 1 && colsp != 1)
                {
                    size_t colspExpand = rowsp*colsp/rowsc;
                    ones = ConstOnes(colspExpand, 1,FunctionValues().GetDeviceId());
                }

                if (inputIndex == 0)  //left derivative
            {
                    ComputeInputPartialLeft(sliceInput0Value, sliceInput0Grad, sliceOutputValue, sliceOutputGrad, ones); 
                }
                else  //right derivative
                {
                    ComputeInputPartialRight(sliceInput0Value, sliceInput0Grad, sliceOutputValue, sliceOutputGrad, ones); 
                }
            }
        }

        static void WINAPI ComputeInputPartialLeft(Matrix<ElemType>& childFunctionValues, Matrix<ElemType>& childGradientValues, const Matrix<ElemType>& functionValues, /*const*/ Matrix<ElemType>& gradientValues, /*const*/ Matrix<ElemType>& ones)
            {
            ComputeInputPartialS(0, childFunctionValues, childGradientValues, functionValues, gradientValues, ones);
        }

        static void WINAPI ComputeInputPartialRight(Matrix<ElemType>& childFunctionValues, Matrix<ElemType>& childGradientValues, const Matrix<ElemType>& functionValues, /*const*/ Matrix<ElemType>& gradientValues, /*const*/ Matrix<ElemType>& ones)  
                {
            ComputeInputPartialS(1, childFunctionValues, childGradientValues, functionValues, gradientValues, ones);
        }

        static void WINAPI ComputeInputPartialS(const size_t inputIndex, Matrix<ElemType>& childFunctionValues, Matrix<ElemType>& childGradientValues, const Matrix<ElemType>& functionValues, /*const*/ Matrix<ElemType>& gradientValues, /*const*/ Matrix<ElemType>& ones)
        {
            ElemType weight = ElemType(inputIndex == 0? 1:-1);
            size_t rowsc = childFunctionValues.GetNumRows(), colsc = childFunctionValues.GetNumCols();
            size_t rowsp = functionValues.GetNumRows(), colsp = functionValues.GetNumCols();

            if (colsc == 1 && colsp != 1)
            {
                size_t colspExpand = rowsp*colsp/rowsc;
                ones.Resize(colspExpand, 1);
                }
            else if (rowsc == 1 && rowsp != 1)
            {
                ones.Resize(1, rowsp);
            }

            if (colsc == colsp && rowsc == rowsp)
            {
                if (inputIndex == 0)
                    childGradientValues += gradientValues;
                else
                    childGradientValues -= gradientValues;
            }
            else if (colsc == 1 && rowsc == 1)
                {
                if (inputIndex == 0)
                    childGradientValues += gradientValues.SumOfElements();
                else
                    childGradientValues -= gradientValues.SumOfElements();
                }
            else if (colsc == 1 && colsp != 1)
            {
                size_t colspExpand = rowsp*colsp/rowsc;
                gradientValues.Reshape(rowsc, colspExpand);
                Matrix<ElemType>::MultiplyAndWeightedAdd(weight, gradientValues, false, ones, false, 1, childGradientValues);
                gradientValues.Reshape(rowsp, colsp);
            }
            else if (rowsc == 1 && rowsp != 1)
                Matrix<ElemType>::MultiplyAndWeightedAdd(weight, ones, false, gradientValues, false, 1, childGradientValues);
            else
                RuntimeError("Minus partial: unexpected condition.");
        }


        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());  
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            size_t cols0 = Inputs(0)->FunctionValues().GetNumCols(), cols1=Inputs(1)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            //only the one with more columns can be sliced, if both have same columns both are sliced
            if (cols0 == cols1)
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value);
            }
            else if (cols0 > cols1)
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, Inputs(1)->FunctionValues());
            }
            else //cols0 < cols1)
            {
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value);
            }
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, Matrix<ElemType>& in0, Matrix<ElemType>& in1)  
        {
            size_t rows0 = in0.GetNumRows(), cols0 = in0.GetNumCols();
            size_t rows1 = in1.GetNumRows(), cols1 = in1.GetNumCols();
            functionValues.Resize(max(rows0, rows1), max(cols0,cols1));

            if ((rows0 == rows1 && cols0 == cols1) || ((rows0 == 1 || rows1 == 1) && cols0 == cols1))
            {
                functionValues.AssignDifferenceOf(in0, in1);
            }
            else if (cols0 == 1 && rows1 % rows0 == 0)  //one is col vec with divisable rows, including scalar
            {
                in1.Reshape(rows0, rows1 * cols1 / rows0);
                functionValues.AssignDifferenceOf(in0, in1);
                in1.Reshape(rows1, cols1);
                functionValues.Reshape(max(rows0, rows1), max(cols0,cols1));
            }
            else if (cols1 == 1 && rows0 % rows1 == 0)  //one is col vec with divisable rows, including scalar
            {
                in0.Reshape(rows1, rows0 * cols0 / rows1);
                functionValues.AssignDifferenceOf(in0, in1);
                in0.Reshape(rows0, cols0);
                functionValues.Reshape(max(rows0, rows1), max(cols0,cols1));
            }      
#if NANCHECK
            functionValues.HasNan("Minus");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 2) 
                throw std::logic_error("Minus operation requires two inputs.");

            //if dimention is missing make the two operatants to have same size
            size_t index = 0;
            if (Inputs(index)->OperationName() == OperationNameOf(LearnableParameter))
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            index = 1;
            if (Inputs(index)->OperationName() == OperationNameOf(LearnableParameter))
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            if (Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements())
                throw std::logic_error("Minus operation: one of the operants has 0 element.");

            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if (!(rows0 == rows1 && cols0 == cols1) &&  //match size
                !((rows0 == 1 || rows1 == 1) && cols0 == cols1) && //one is row vec
                !((cols0 == 1 && rows1 % rows0 == 0) || (cols1 == 1 && rows0 % rows1 == 0)))  //one is col vec with divisable rows, including scalar
            {
                throw std::logic_error("The Matrix dimension in the Minus operation does not match.");
            }       

            FunctionValues().Resize(max(rows0, rows1), max(cols0,cols1) );
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs() //based on the matrix with larger size
        {
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if (rows0 > rows1 || cols0 > cols1) //child 0 is larger
                InferImageDimsFromInput(0);
            else if (rows0 < rows1 || cols0 < cols1) //child 1 is larger
                InferImageDimsFromInput(1);
            else //same size
            {
                if (IsChildAnImage(0))  //when conflict, give priority to child 0
                    InferImageDimsFromInput(0);
                else
                    InferImageDimsFromInput(1);
            }
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }
    };

    template class MinusNode<float>; 
    template class MinusNode<double>;

    //The first matrix should be a vector regpresting the diagonal of a square matrix in the DiagTimes operation
    template<class ElemType>
    class DiagTimesNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        DiagTimesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_innerproduct(deviceId), m_rightGradient(deviceId)
        { }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"DiagTimes";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("DiagTimes operation only takes two inputs.");
            else if (inputIndex == 0)  //left derivative
                ComputeInputPartialLeft(m_innerproduct, Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
            else  //right derivative
                ComputeInputPartialRight(m_rightGradient, Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex > 1)
                InvalidArgument("DiagTimes operation only takes two inputs.");

            //left parameter (diag matix cannot be sliced)
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (inputIndex == 0)  //left derivative
            {
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                ComputeInputPartialLeft(m_innerproduct, sliceInput1Value, Inputs(0)->GradientValues(), sliceOutputGrad);
            }
            else  //right derivative
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                ComputeInputPartialRight(m_rightGradient, Inputs(0)->FunctionValues(), sliceInput1Grad, sliceOutputGrad);
            }
        }

        static void WINAPI ComputeInputPartialLeft(Matrix<ElemType>& temp, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            temp.AssignInnerProductOf(gradientValues, inputFunctionValues, false);
            inputGradientValues += temp;
        }

        static void WINAPI ComputeInputPartialRight(Matrix<ElemType>& temp, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            temp.SetValue(gradientValues);
            temp.ColumnElementMultiplyWith(inputFunctionValues);
            inputGradientValues += temp;
        }


        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues()); 
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)  
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value); 
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues0, const Matrix<ElemType>& inputFunctionValues1)  
        {
            functionValues.SetValue(inputFunctionValues1);
            functionValues.ColumnElementMultiplyWith(inputFunctionValues0);
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 2) 
                throw std::logic_error("DiagTimes operation requires two inputs.");

            //if dimention not specified we assume two operants' dimentions should match
            if (Inputs(0)->OperationName() == OperationNameOf(LearnableParameter) && Inputs(0)->FunctionValues().GetNumRows() == 0 && Inputs(1)->FunctionValues().GetNumRows() != 0)
            {
                Inputs(0)->FunctionValues().Resize(Inputs(1)->FunctionValues().GetNumRows(), 1);
            }

            if (Inputs(1)->OperationName() == OperationNameOf(LearnableParameter) && Inputs(0)->FunctionValues().GetNumRows() != 0 && Inputs(1)->FunctionValues().GetNumRows() == 0)
            {
                Inputs(1)->FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());
            }

            if (Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements())
                throw std::logic_error("DiagTimes operation: one of the operants has 0 element.");

            if (Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumRows())
                throw std::logic_error("The Matrix dimension in the DiagTimes operation does not match.");

            if (1 != Inputs(0)->FunctionValues().GetNumCols())
                throw std::logic_error("The first matrix should be a vector regpresting the diagonal of a square matrix in the DiagTimes operation.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());
            m_innerproduct.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());
            m_rightGradient.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());

            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs() //this is element wise scaling, so based on child 1
        {
            InferImageDimsFromInput(1);
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            Base::MoveMatricesToDevice(deviceId);
            m_innerproduct.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
            m_rightGradient.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<DiagTimesNode<ElemType>>(nodeP);
                node->m_innerproduct = m_innerproduct;
                node->m_rightGradient = m_rightGradient;
            }
        }
private:
        Matrix<ElemType> m_innerproduct;
        Matrix<ElemType> m_rightGradient;
    };

    template class DiagTimesNode<float>; 
    template class DiagTimesNode<double>;

    //The first matrix should be a vector regpresting the diagonal of a square matrix in the DiagTimes operation
    template<class ElemType>
    class CosDistanceNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        CosDistanceNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_invNorm0(deviceId), m_invNorm1(deviceId), m_leftTerm(deviceId), m_rightTerm(deviceId), m_temp(deviceId)
        { }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"CosDistance";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("CosDistance operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(m_invNorm0, m_invNorm1, FunctionValues(), m_temp, m_rightTerm, m_leftTerm, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), GradientValues(), Inputs(inputIndex)->GradientValues());
            }
            else  //right derivative
            {
                ComputeInputPartialRight(m_invNorm0, m_invNorm1, FunctionValues(), m_temp, m_rightTerm, m_leftTerm, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), GradientValues(), Inputs(inputIndex)->GradientValues());
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) 
        {
            if (inputIndex > 1)
                InvalidArgument("CosDistance operation only takes two inputs.");

            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInputGrad = Inputs(inputIndex)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = this->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(m_invNorm0, m_invNorm1, sliceOutputValue, m_temp, m_rightTerm, m_leftTerm, sliceInput0Value, sliceInput1Value, sliceOutputGrad, sliceInputGrad);
            }
            else  //right derivative
            {
                ComputeInputPartialRight(m_invNorm0, m_invNorm1, sliceOutputValue, m_temp, m_rightTerm, m_leftTerm, sliceInput0Value, sliceInput1Value, sliceOutputGrad, sliceInputGrad);
            }
        }

        static void WINAPI ComputeInputPartialLeft(const Matrix<ElemType>& invNorm0, const Matrix<ElemType>& invNorm1, const Matrix<ElemType>& functionValues, 
            Matrix<ElemType>& temp, Matrix<ElemType>& rightTerm, Matrix<ElemType>& leftTerm, // the temporary variables
            const Matrix<ElemType>& in0, const Matrix<ElemType>& in1, const Matrix<ElemType>& gradientValues,
            Matrix<ElemType>& inputGradientValues)
        {
            ComputeInputPartialS(0, invNorm0, invNorm1, functionValues, temp, rightTerm, leftTerm, in0, in1, gradientValues, inputGradientValues);
        }

        static void WINAPI ComputeInputPartialRight(const Matrix<ElemType>& invNorm0, const Matrix<ElemType>& invNorm1, const Matrix<ElemType>& functionValues, 
            Matrix<ElemType>& temp, Matrix<ElemType>& rightTerm, Matrix<ElemType>& leftTerm, // the temporary variables
            const Matrix<ElemType>& in0, const Matrix<ElemType>& in1, const Matrix<ElemType>& gradientValues,
            Matrix<ElemType>& inputGradientValues)  
        {
            ComputeInputPartialS(1, invNorm0, invNorm1, functionValues, temp, rightTerm, leftTerm, in0, in1, gradientValues, inputGradientValues);  
        }

        // functionValues, invNorm0, invNorm1 - output from the EvaluateNode() method
        // temp, rightTerm, leftTerm - temporary matrices
        // in0, in1 - input functionValues from other nodes
        // inputGradientValues(x) - gradients to update, where x matches inputIndex
        static void WINAPI ComputeInputPartialS(const size_t inputIndex, const Matrix<ElemType>& invNorm0, const Matrix<ElemType>& invNorm1, const Matrix<ElemType>& functionValues, 
            Matrix<ElemType>& temp, Matrix<ElemType>& rightTerm, Matrix<ElemType>& leftTerm, // the temporary variables
            const Matrix<ElemType>& in0, const Matrix<ElemType>& in1, const Matrix<ElemType>& gradientValues,
            Matrix<ElemType>& inputGradientValues)  
        {
            if (inputIndex == 0)  //left derivative
            {
                temp.AssignElementProductOf(invNorm0, invNorm0);
            }
            else  //right derivative
            {
                temp.AssignElementProductOf(invNorm1, invNorm1);
            }

            temp.ElementMultiplyWith(functionValues);
            rightTerm.SetValue(inputIndex?in1:in0);
            rightTerm.RowElementMultiplyWith(temp);

            temp.AssignElementProductOf(invNorm0, invNorm1);
            leftTerm.SetValue(inputIndex?in0:in1);
            leftTerm.RowElementMultiplyWith(temp);

            leftTerm -= rightTerm;
            leftTerm.RowElementMultiplyWith(gradientValues);
            inputGradientValues += leftTerm;
            
            //alternatively the above three lines can be replaced by
            //leftTerm.RowElementMultiplyWith(gradientValues);
            //rightTerm.RowElementMultiplyWith(gradientValues);
            //Matrix<ElemType>::AddScaledDifference(1, leftTerm, rightTerm, inputGradientValues);
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_invNorm0, m_invNorm1, FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());  
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) 
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(m_invNorm0, m_invNorm1, sliceOutputValue, sliceInput0Value, sliceInput1Value);  
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& invNorm0, Matrix<ElemType>& invNorm1, 
            Matrix<ElemType>& functionValues, Matrix<ElemType>& in0, Matrix<ElemType>& in1)  
        {
            invNorm0.AssignVectorNorm2Of(in0, true); // seems to modify input (in0)
            invNorm0.AssignElementInverseOf(invNorm0);

            invNorm1.AssignVectorNorm2Of(in1, true); // seems to modify the input (in1)
            invNorm1.AssignElementInverseOf(invNorm1);

            functionValues.AssignInnerProductOf(in0, in1, true);
            functionValues.ElementMultiplyWith(invNorm0);
            functionValues.ElementMultiplyWith(invNorm1);
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 2) 
                throw std::logic_error("CosDistance operation requires two inputs.");

            //if dimention is missing make the two operatants to have same size
            size_t index = 0;
            if (Inputs(index)->OperationName() == OperationNameOf(LearnableParameter))
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            index = 1;
            if (Inputs(index)->OperationName() == OperationNameOf(LearnableParameter))
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            if (Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements())
                throw std::logic_error("CosDistance operation: one of the operants has 0 element.");

            if (Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumRows() || 
                Inputs(1)->FunctionValues().GetNumCols() != Inputs(0)->FunctionValues().GetNumCols())
                throw std::logic_error("The Matrix dimension in the CosDistance operation does not match.");

            FunctionValues().Resize(1, Inputs(1)->FunctionValues().GetNumCols());

            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs() 
        {
            InferImageDimsFromInput(0, false);

            m_outputChannels = 1;
            m_outputWidth = 1;
            m_outputHeight = 1;        
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            Base::MoveMatricesToDevice(deviceId);
            m_invNorm0.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
            m_invNorm1.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
            m_leftTerm.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
            m_rightTerm.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
            m_temp.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<CosDistanceNode<ElemType>>(nodeP);
                node->m_invNorm0 = m_invNorm0;
                node->m_invNorm1 = m_invNorm1;
                node->m_leftTerm = m_leftTerm;
                node->m_rightTerm = m_rightTerm;
                node->m_temp = m_temp;
            }
        }
private:
        // invNorm nodes tranfer data between EvaluateThisNode and ComputeInputPartial
        Matrix<ElemType> m_invNorm0;
        Matrix<ElemType> m_invNorm1;
        // the rest are temporaries, values don't need to be maintained
        Matrix<ElemType> m_leftTerm;
        Matrix<ElemType> m_rightTerm;
        Matrix<ElemType> m_temp;
    };

    template class CosDistanceNode<float>; 
    template class CosDistanceNode<double>;


    template<class ElemType>
    class KhatriRaoProductNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        KhatriRaoProductNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"KhatriRaoProduct";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("KhatriRaoProduct operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues()); 
            }
            else  //right derivative
            {
                ComputeInputPartialRight(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues()); 
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) 
        {
            if (inputIndex > 1)
                InvalidArgument("KhatriRaoProduct operation only takes two inputs.");

            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (inputIndex == 0)  //left derivative
            {
                Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialLeft(sliceInput1Value, sliceInput0Grad, sliceOutputGrad); 
            }
            else  //right derivative
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialRight(sliceInput0Value, sliceInput1Grad, sliceOutputGrad); 
            }
        }

        static void WINAPI ComputeInputPartialLeft(Matrix<ElemType>& childFunctionValues, Matrix<ElemType>& childGradientValues, const Matrix<ElemType>& gradientValues)
        {
            childGradientValues.AddColumnReshapeProductOf(gradientValues, childFunctionValues, false);
        }

        static void WINAPI ComputeInputPartialRight(Matrix<ElemType>& childFunctionValues, Matrix<ElemType>& childGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            childGradientValues.AddColumnReshapeProductOf(gradientValues, childFunctionValues, true);
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());  
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)  
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value); 
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, Matrix<ElemType>& in0, Matrix<ElemType>& in1)  
        {
            functionValues.AssignKhatriRaoProductOf(in0,in1);
#if NANCHECK
            functionValues.HasNan("KhatriRaoProduct");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 2) 
                throw std::logic_error("KhatriRaoProduct operation requires two inputs.");

            //support automatic dimention inference for learnable parameters
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if (rows0 == 0 || rows1 == 0)
                throw logic_error("KhatriRaoProduct operation: The number of rows in the input should not be 0.");

            if (Inputs(0)->OperationName() == OperationNameOf(LearnableParameter) && cols0 == 0 && cols1 != 0)
                Inputs(0)->FunctionValues().Resize(rows0, cols1);

            if (Inputs(1)->OperationName() == OperationNameOf(LearnableParameter) && cols0 != 0 && cols1 == 0)
                Inputs(1)->FunctionValues().Resize(rows1, cols0);

            //cols may be changed before this line and so cannot use cached cols values below
            if (Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements())
                throw std::logic_error("KhatriRaoProduct operation: One of the operants has 0 elements.");

            if (Inputs(1)->FunctionValues().GetNumCols() != Inputs(0)->FunctionValues().GetNumCols())
            {
                throw std::logic_error("The Matrices should have same number of columns.");
            }

            FunctionValues().Resize(rows0 * rows1, Inputs(0)->FunctionValues().GetNumCols());
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs()  
        {
            //since it's symmetrical any one of the input may be the true input. 
            //since we dont' use the input image size info in the operation, the input part doesn't matter.
            InferImageDimsFromInput(1, false); 

            //after KhatriRaoProduct the structure is lost
            m_outputWidth = 1;
            m_outputHeight = m_functionValues.GetNumRows();
            m_outputChannels =  1;
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }
    };

    template class KhatriRaoProductNode<float>; 
    template class KhatriRaoProductNode<double>;

    template<class ElemType>
    class CosDistanceWithNegativeSamplesNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;

    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        CosDistanceWithNegativeSamplesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_invNorm0(deviceId), m_invNorm1(deviceId), m_invNormSquare(deviceId), 
            m_leftTerm(deviceId), m_rightTerm(deviceId), m_temp(deviceId)
        { }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"CosDistanceWithNegativeSamples"; }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("CosDistanceWithNegativeSamples operation only takes grdients on the first two inputs.");

            ComputeInputPartialS(inputIndex, m_invNorm0, m_invNorm1, FunctionValues(), m_temp, m_rightTerm, m_leftTerm, m_invNormSquare, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), Inputs(3)->FunctionValues(), Inputs(inputIndex)->GradientValues(), GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex > 1)
                InvalidArgument("CosDistanceWithNegativeSamples operation only takes grdients on the first two inputs.");

            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInputGrad = Inputs(inputIndex)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceThisGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(inputIndex, m_invNorm0, m_invNorm1, sliceOutputValue, m_temp, m_rightTerm, m_leftTerm, m_invNormSquare, sliceInput0Value, sliceInput1Value, Inputs(2)->FunctionValues(), Inputs(3)->FunctionValues(), sliceInputGrad, sliceThisGrad);
        }

        // functionValues, invNorm0, invNorm1 - output from the EvaluateNode() method
        // temp, rightTerm, leftTerm - temporary matrices
        // in0, in1, in2, in3 - input functionValues from other nodes
        // inputGradientValues(x) - gradients to update, where x matches inputIndex
        static void WINAPI ComputeInputPartialS(const size_t inputIndex, const Matrix<ElemType>& invNorm0, const Matrix<ElemType>& invNorm1, const Matrix<ElemType>& functionValues,
            Matrix<ElemType>& temp, Matrix<ElemType>& rightTerm, Matrix<ElemType>& leftTerm, Matrix<ElemType>& invNormSquare, // the temporary variables
            const Matrix<ElemType>& in0, const Matrix<ElemType>& in1, const Matrix<ElemType>& in2, const Matrix<ElemType>& in3,
            Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& thisGradientValues)
        {
            size_t shift = (size_t)in2.Get00Element();
            size_t negNumber = (size_t)in3.Get00Element();
            size_t numCols = in0.GetNumCols(); // used in computing right child's graident

            if (inputIndex == 0) // left derivative
            {
                invNormSquare.AssignElementProductOf(invNorm0, invNorm0);

                for (long m = 0; m < negNumber + 1; m++)
                {
                    temp.GetARowByIndex(functionValues, m); // set this matrx to be the m-th row in functionValues
                    temp.ElementMultiplyWith(invNormSquare);

                    Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, in0, rightTerm, 0, true);

                    if (m == 0)
                    {
                        temp.AssignElementProductOf(invNorm0, invNorm1);

                        Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, in1, leftTerm, 0, true);
                    }
                    else
                    {
                        size_t currshift = m + shift - 1;  // for current line, how much should we shift

                        temp.AssignElementProductOfWithShift(invNorm0, invNorm1, currshift); // this is a row vector

                        Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, in1, leftTerm, currshift, true);
                    }

                    leftTerm = leftTerm - rightTerm;

                    temp.GetARowByIndex(thisGradientValues, m);

                    Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, leftTerm, rightTerm, 0, true);

                    inputGradientValues += rightTerm;
                }
            }
            else // right part
            {
                invNormSquare.AssignElementProductOf(invNorm1, invNorm1);  //this matrix should be save and unchanged. It should not be changed

                for (long m = 0; m < negNumber + 1; m++)
                {
                    temp.GetARowByIndex(functionValues, m); // set this matrx to be the m-th row in functionValues

                    if (m == 0) // this is the first line. computation should be symmetric
                    {
                        // the following is for the right part
                        temp.ElementMultiplyWith(invNormSquare);

                        Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, in1, rightTerm, 0, true);

                        // the following is for the left part
                        temp.AssignElementProductOf(invNorm0, invNorm1);

                        Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, in0, leftTerm, 0, true);

                        leftTerm = leftTerm - rightTerm;

                        temp.GetARowByIndex(thisGradientValues, m);

                        Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, leftTerm, rightTerm, 0, true);

                        inputGradientValues += rightTerm;
                    }
                    else // this requires shift
                    {
                        size_t currshift = (m + shift - 1) % numCols;
                        size_t reverseshift = numCols - currshift;

                        leftTerm.AssignElementProductOfWithShift(invNormSquare, temp, reverseshift);  //use leftTerm as a temp variable here

                        Matrix<ElemType>::ConductRowElementMultiplyWithShift(leftTerm, in1, rightTerm, 0, true);

                        temp.AssignElementProductOfWithShift(invNorm1, invNorm0, reverseshift);

                        Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, in0, leftTerm, reverseshift, true);

                        leftTerm = leftTerm - rightTerm;

                        temp.GetARowByIndex(thisGradientValues, m);

                        Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, leftTerm, rightTerm, reverseshift, false);

                        inputGradientValues += rightTerm;
                    }
                }
            }
        }

        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(m_invNorm0, m_invNorm1, FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), Inputs(3)->FunctionValues(), m_leftTerm, m_rightTerm);
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(m_invNorm0, m_invNorm1, sliceOutputValue, sliceInput0Value, sliceInput1Value, Inputs(2)->FunctionValues(), Inputs(3)->FunctionValues(), m_leftTerm, m_rightTerm);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& invNorm0, Matrix<ElemType>& invNorm1,
            Matrix<ElemType>& functionValues, Matrix<ElemType>& in0, Matrix<ElemType>& in1, Matrix<ElemType>& in2, Matrix<ElemType>& in3, Matrix<ElemType>& leftTermTemp, Matrix<ElemType>& rightTermTemp)
        {
            invNorm0.AssignVectorNorm2Of(in0, true); // seems to modify input (in0)
            invNorm0.AssignElementInverseOf(invNorm0);

            invNorm1.AssignVectorNorm2Of(in1, true); // seems to modify the input (in1)
            invNorm1.AssignElementInverseOf(invNorm1);

            size_t shift = (size_t)in2.Get00Element();
            size_t negNumber = (size_t)in3.Get00Element();

            // mutiply invNorm0 and invNorm1 with shift and neg. 
            // The result is a matrix of (numberneg+1, invNorm0.Cols)
            leftTermTemp.AssignElementProductOfWithShiftNeg(invNorm0, invNorm1, shift, negNumber);

            // compute the right values
            // Again, the ouput is a matrix of (negNumber+1, invNorm0.cols)
            rightTermTemp.AssignInnerProductOfWithShiftNeg(in0, in1, true, shift, negNumber);

            // compute the evaluation result matrix by multiply these two matrices, element by element
            // we get a (negNumber+1, n) matrix
            functionValues.AssignElementProductOf(leftTermTemp, rightTermTemp);
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 4)
                throw std::logic_error("CosDistanceWithNegativeSamples operation requires 4 inputs.");

            //if dimention is missing make the two operatants to have same size
            size_t index = 0;
            if (Inputs(index)->OperationName() == OperationNameOf(LearnableParameter))
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0 ? Inputs(1 - index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0 ? Inputs(1 - index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            index = 1;
            if (Inputs(index)->OperationName() == OperationNameOf(LearnableParameter))
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0 ? Inputs(1 - index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0 ? Inputs(1 - index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            if (Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements())
                throw std::logic_error("CosDistanceWithNegativeSamples operation: one of the operants has 0 element.");

            if (Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumRows() ||
                Inputs(1)->FunctionValues().GetNumCols() != Inputs(0)->FunctionValues().GetNumCols())
                throw std::logic_error("The Matrix dimension in the CosDistanceWithNegativeSamples operation does not match.");

            // input(2) is shift, input(3) is the #neg
            size_t negNumber = (size_t)Inputs(3)->FunctionValues()(0, 0);

            FunctionValues().Resize(negNumber + 1, Inputs(1)->FunctionValues().GetNumCols());

            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            m_outputChannels = 1;
            m_outputWidth = 1;
            m_outputHeight = 1;
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode, const ComputationNodePtr shiftNode, const ComputationNodePtr negNode)
        {
            m_children.resize(4);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
            m_children[2] = shiftNode;
            m_children[3] = negNode;
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            Base::MoveMatricesToDevice(deviceId);
            m_invNorm0.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
            m_invNorm1.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
            m_invNormSquare.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
            m_leftTerm.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
            m_rightTerm.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
            m_temp.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<CosDistanceWithNegativeSamplesNode<ElemType>>(nodeP);
                node->m_invNorm0 = m_invNorm0;
                node->m_invNorm1 = m_invNorm1;
                node->m_invNormSquare = m_invNormSquare;
                node->m_leftTerm = m_leftTerm;
                node->m_rightTerm = m_rightTerm;
                node->m_temp = m_temp;
            }
        }
private:
        // invNorm nodes tranfer data between EvaluateThisNode and ComputeInputPartial
        Matrix<ElemType> m_invNorm0;
        Matrix<ElemType> m_invNorm1;
        // the rest are temporaries, values don't need to be maintained
        Matrix<ElemType> m_leftTerm;
        Matrix<ElemType> m_rightTerm;
        Matrix<ElemType> m_invNormSquare;
        Matrix<ElemType> m_temp;
    };

    template class CosDistanceWithNegativeSamplesNode<float>;
    template class CosDistanceWithNegativeSamplesNode<double>;

    template<class ElemType>
    class TransposeNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;

        Matrix<ElemType> mOnes; 
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        TransposeNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNodeNonLooping<ElemType>(deviceId, name),
            mOnes(deviceId)
        { }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"Transpose"; }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("Times operation only takes two inputs.");

            ComputeInputPartialS(Inputs(0)->GradientValues(), mOnes, GradientValues());
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& ones, const Matrix<ElemType>& gradientValues)
        {
#if DUMPOUTPUT
            gradientValues.Print("Gradient-in");
            inputGradientValues.Print("child Gradient-in/out");
            inputFunctionValues.Print("child Function values");
#endif

            if (ones.GetNumRows() != inputGradientValues.GetNumRows() || ones.GetNumCols() != inputGradientValues.GetNumRows())
                ones = Matrix<ElemType>::Ones(inputGradientValues.GetNumRows(), inputGradientValues.GetNumRows(), inputGradientValues.GetDeviceId());
            Matrix<ElemType>::MultiplyAndAdd(ones, false, gradientValues, true, inputGradientValues);
#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
        }

        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues());
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0)
        {
#if DUMPOUTPUT
            input0.Print("TransposeNode- Input0");
#endif
            functionValues.AssignTransposeOf(input0);
#if NANCHECK
            functionValues.HasNan("Transpose");
#endif
#if DUMPOUTPUT
            functionValues.Print("TransposeNode");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 1)
                throw std::logic_error("Transpose operation requires one input.");

            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();

            if (rows0 == 0 || cols0 == 0)
                throw logic_error("Transpose operation: Inputs(0)->FunctionValues().GetNumRows() and Inputs(1)->FunctionValues().GetNumCols() should not be 0 ");

            FunctionValues().Resize(cols0, rows0);
            mOnes = Matrix<ElemType>::Ones(rows0, rows0, m_deviceId);
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false); //the second one is the input since it's column wize

            //after multiplication the structure is lost
            m_outputWidth = 1;
            m_outputHeight = Inputs(0)->FunctionValues().GetNumCols();
            m_outputChannels = 1;
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode)
        {
            m_children.resize(1);
            m_children[0] = leftNode;
        }
    };

    template class TransposeNode<float>;
    template class TransposeNode<double>;

    /**
    Has a stride in particular dimensions of left matrix when doing times operation. 
    Example 1: column stride s
    A in d x [s x T1] 
    B in T1 x s
    C = A x B  in d x s, and each element is computed as 
    c_{i,k} = \sum_j a_{i,j*s+k} b_{j,k}
    where s is the stride in column.

    Example 2:
    A in [s x T1] x d
    B in d x s
    C = A x B  in T1 x s, and each element is computed as
    c_{i,k} = \sum_j a_{i*s+k,j} b_{j,k}
    where s is the stride in rows.

    Notice that s is equal to k. 
    */
    template<class ElemType>
    class StrideTimesNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;

        size_t m_StrideDim; // the dimension index on which stride works 
        size_t m_Stride;    // the stride 
    private:
        void UpdateStride(const Matrix<ElemType>& input1) 
        {
            m_Stride = input1.GetNumCols();
        }
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        StrideTimesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_Stride(1)
        { }
        // BUGBUG: This node needs to serialize and CopyTo m_Stride

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"StrideTimes"; }

        virtual void ComputeInputPartial(const size_t) { NOT_IMPLEMENTED; }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex > 1)
                InvalidArgument("StrideTimes operation only takes two inputs.");

            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (m_StrideDim == 1) /// column stride
            {
                if (inputIndex == 0)  //left derivative
                {
                    Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);


//                    TimesNode<ElemType>::ComputeInputPartialLeft(sliceInput1Value, Inputs(0)->GradientValues(), sliceOutputGrad);

                    Matrix<ElemType> mTmp1(sliceInput1Value.GetDeviceId());
                    size_t r = Inputs(0)->FunctionValues().GetNumRows();
                    size_t T1 = Inputs(0)->FunctionValues().GetNumCols() / m_samplesInRecurrentStep;
                    mTmp1.Resize(r, T1);
                    Matrix<ElemType> mTmp2(sliceInput1Value.GetDeviceId());
                    Matrix<ElemType> mTmp3(sliceInput1Value.GetDeviceId());

                    for (size_t k = 0; k < m_samplesInRecurrentStep; k++)
                    {
                        mTmp1.SetValue(0);
                        mTmp2 = sliceInput1Value.ColumnSlice(k, 1);
                        mTmp3 = sliceOutputGrad.ColumnSlice(k, 1);

                        TimesNode<ElemType>::ComputeInputPartialLeft(mTmp2, mTmp1, mTmp3);

                        for (size_t t = 0; t < T1; t++)
                        {
                            Inputs(0)->GradientValues().ColumnSlice(t*m_samplesInRecurrentStep + k, 1) += mTmp1.ColumnSlice(t, 1);
                        }
                    }
                }
                else  //right derivative
                {
                    Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                    //                    TimesNode<ElemType>::ComputeInputPartialRight(Inputs(0)->FunctionValues(), sliceInput1Grad, sliceOutputGrad);

                    for (size_t k = 0; k < m_samplesInRecurrentStep; k++)
                    {
                        Matrix<ElemType> mTmp1(sliceOutputGrad.GetDeviceId());
                        size_t r = Inputs(0)->FunctionValues().GetNumRows();
                        size_t T1 = Inputs(0)->FunctionValues().GetNumCols() / m_samplesInRecurrentStep;
                        mTmp1.Resize(r, T1);
                        for (size_t t = 0; t < T1; t++)
                        {
                            mTmp1.ColumnSlice(t, 1).SetValue(Inputs(0)->FunctionValues().ColumnSlice(t*m_samplesInRecurrentStep + k, 1));
                        }
                        Matrix<ElemType> mTmp2(sliceOutputGrad.GetDeviceId());
                        mTmp2 = sliceInput1Grad.ColumnSlice(k, 1);
                        Matrix<ElemType> mTmp3(sliceOutputGrad.GetDeviceId());
                        mTmp3 = sliceOutputGrad.ColumnSlice(k, 1);

                        TimesNode<ElemType>::ComputeInputPartialRight(mTmp1, mTmp2, mTmp3);
                    }
                }
            }
            else if (m_StrideDim == 0) /// row stride
            {
                if (inputIndex == 0)  //left derivative
                {
                    Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                    for (size_t k = 0; k < m_samplesInRecurrentStep; k++)
                    {
                        Matrix<ElemType> mTmp1(sliceInput1Value.GetDeviceId());
                        size_t d = Inputs(1)->FunctionValues().GetNumRows();
                        size_t T1 = Inputs(0)->FunctionValues().GetNumRows() / m_samplesInRecurrentStep;
                        mTmp1.Resize(d, T1);
                        Matrix<ElemType> mTmp2(sliceInput1Value.GetDeviceId());
                        mTmp2 = sliceInput1Value.ColumnSlice(k, 1);
                        Matrix<ElemType> mTmp3(sliceInput1Value.GetDeviceId());
                        mTmp3 = sliceOutputGrad.ColumnSlice(k, 1);
                        ComputeInputPartialLeft(mTmp2, mTmp1, mTmp3);

                        Matrix<ElemType> mTmp4(sliceInput1Value.GetDeviceId());
                        for (size_t t = 0; t < T1; t++)
                        {
                            mTmp4 = mTmp1.ColumnSlice(t, 1);
                            mTmp4.Reshape(1, d);
                            Inputs(0)->GradientValues().AddToRowSliceValuesOf(mTmp4, t*m_samplesInRecurrentStep + k, 1);
                        }
                    }
                }
                else  //right derivative
                {
                    Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                    for (size_t k = 0; k < m_samplesInRecurrentStep; k++)
                    {
                        size_t d = Inputs(1)->FunctionValues().GetNumRows();
                        size_t T1 = Inputs(0)->FunctionValues().GetNumRows() / m_samplesInRecurrentStep;

                        Matrix<ElemType> mTmp0(sliceOutputGrad.GetDeviceId());
                        mTmp0.Resize(1, d);

                        Matrix<ElemType> mTmp1(sliceOutputGrad.GetDeviceId());
                        mTmp1.Resize(T1, d);
                        for (size_t t = 0; t < T1; t++)
                        {
                            mTmp0.SetValue(0);
                            mTmp0.AddWithRowSliceValuesOf(Inputs(0)->FunctionValues(), t * m_samplesInRecurrentStep + k, 1);
                            mTmp1.AssignToRowSliceValuesOf(mTmp0, t, 1);
                        }
                        Matrix<ElemType> mTmp2(sliceOutputGrad.GetDeviceId());
                        mTmp2 = sliceInput1Grad.ColumnSlice(k, 1);
                        Matrix<ElemType> mTmp3(sliceOutputGrad.GetDeviceId());
                        mTmp3 = sliceOutputGrad.ColumnSlice(k, 1);

                        ComputeInputPartialRight(mTmp1, mTmp2, mTmp3);
                    }
                }
            }
        }

        static void WINAPI ComputeInputPartialLeft(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
        {
#if DUMPOUTPUT   
            gradientValues.Print("Gradient-in");   
            inputGradientValues.Print("child Gradient-in/out");   
            inputFunctionValues.Print("child Function values");   
#endif
            //currently we only support one combination when the input is sparse.   
            if (inputFunctionValues.GetMatrixType() == SPARSE && inputGradientValues.GetMatrixType() == DENSE && gradientValues.GetMatrixType() == DENSE)
                inputGradientValues.SwitchToMatrixType(SPARSE, MatrixFormat::matrixFormatSparseBlockCol, false);

            Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, false, gradientValues, true, inputGradientValues);

#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
        }

        static void WINAPI ComputeInputPartialRight(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
        {
#if DUMPOUTPUT   
            gradientValues.Print("Gradient-in");   
            inputGradientValues.Print("child Gradient-in/out");
            inputFunctionValues.Print("child Function values");
#endif   
            Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, true, gradientValues, false, inputGradientValues);
#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
        }

        virtual void EvaluateThisNode()
        {
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();
            UpdateStride(Inputs(1)->FunctionValues());
            if (m_StrideDim == 0)
                FunctionValues().Resize(rows0 / m_samplesInRecurrentStep, cols1);
            if (m_StrideDim == 1)
                FunctionValues().Resize(rows0, cols1);

            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_Stride, m_StrideDim);
#ifdef DEBUG_DECODER
            fprintf(stderr, "Times node %ls output norm = %.8e, input(0) norm = %.8e, input(1) norm = %.8e\n", this->NodeName().c_str(), FunctionValues().FrobeniusNorm(),
                Inputs(0)->FunctionValues().FrobeniusNorm(), Inputs(1)->FunctionValues().FrobeniusNorm());
#endif
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            UpdateStride(sliceInput1Value);
            if (m_StrideDim == 0)
                FunctionValues().Resize(rows0 / m_samplesInRecurrentStep, cols1);
            if (m_StrideDim == 1)
                FunctionValues().Resize(rows0, cols1);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value, m_Stride, m_StrideDim);
        }

        /**
        A in d x [s x T1]
        B in T1 x s
        C = A x B  in d x s, and each element is computed as 
        c_{i,k} = \sum_j a_{i,j*s+k} b_{j,k}
        C in d x s
        where s is the stride in column.

        Example 2:
        A in [s x T1] x d
        B in d x s
        C = A x B  in T1 x s, and each element is computed as
        c_{i,k} = \sum_j a_{i*s+k,j} b_{j,k}
        where s is the stride in rows.
        C in T1 x s

        strideDim : 0 or 1
        */
        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, const Matrix<ElemType>& input1, const size_t stride, const size_t strideDim)
        {
#if DUMPOUTPUT
            input0.Print("StrideTimesNode - Input0");
#endif
            assert(strideDim == 0 || strideDim == 1);
            Matrix<ElemType> mTmp1(input0.GetDeviceId());
            Matrix<ElemType> mTmp2(input0.GetDeviceId());
            if (strideDim == 1) /// the example 1 case at column
            {
                assert(stride == input1.GetNumCols());
                size_t T1 = input0.GetNumCols() / stride;
                assert(T1 == input1.GetNumRows());
                size_t d = input0.GetNumRows();
                functionValues.Resize(d, stride);
                for (size_t k = 0; k < stride; k++)
                {
                    mTmp1.Resize(d, T1);
                    for (size_t j = 0; j < T1; j++)
                    {
                        mTmp1.ColumnSlice(j, 1).SetValue(input0.ColumnSlice(j * stride + k, 1));
                    }

                    mTmp2 = input1.ColumnSlice(k, 1);
                    functionValues.ColumnSlice(k, 1).AssignProductOf(mTmp1, false, mTmp2, false);

                }
            }
            else if (strideDim == 0)/// the example 2 case at row
            {
                assert(stride == input1.GetNumCols());
                size_t T1 = input0.GetNumRows() / stride;
                size_t d = input1.GetNumRows();
                assert(d == input0.GetNumCols());
                functionValues.Resize(T1, stride);
                mTmp1.Resize(d, T1);
                for (size_t k = 0; k < stride; k++)
                {
                    for (size_t j = 0; j < T1; j++)
                    {
                        mTmp1.ColumnSlice(j, 1).AssignRowSliceValuesOf(input0, k + j * stride, 1);
                    }

                    mTmp2 = input1.ColumnSlice(k, 1);
                    functionValues.ColumnSlice(k, 1).AssignProductOf(mTmp1, true, mTmp2, false);

                }
            }
#if NANCHECK
            functionValues.HasNan("StrideTimes");
#endif
#if DUMPOUTPUT
            functionValues.Print("StrideTimesNode");
#endif
        }

        /**
        three inputs
        input0: left matrix
        input1: right matrix
        stridedim: single element no gradient matrix, 0 row stride / 1 column stride
        */
        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 3)
                throw std::logic_error("StrideTimes operation requires three inputs.");

            //support automatic dimention inference for learnable parameters
            if (Inputs(2)->FunctionValues().GetNumElements() != 1)
                LogicError("StrideTimes : input(2) should be a single element matrix");

            m_StrideDim = (size_t) Inputs(2)->FunctionValues().Get00Element();
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if (m_StrideDim != 0 && m_StrideDim != 1)
                LogicError("StrideTimes : stride dim must be either 0 (row) or 1 (column)");

            if (Inputs(2)->NeedGradient())
                LogicError("StrideTImes : no gradient update should be on input(2)");

            //cols0 and rows1 may have been changed so don't use them in the following check
            if (m_StrideDim == 0)
            {
                if (rows1 != cols0)
                    LogicError("The Matrix dimension in the StrideTimes operation in dim %d does not match for cols %d in A and rows %d in B.", m_StrideDim, cols0, rows1);
                size_t T1 = rows0 / m_Stride;
                FunctionValues().Resize(T1, cols1);
            }

            //cols0 and rows1 may have been changed so don't use them in the following check
            if (m_StrideDim == 1)
            {
                if (cols0/m_Stride != rows1)
                    LogicError("The Matrix dimension in the StrideTimes operation in dim %d does not match for cols %d in A and row number %d in B.", m_StrideDim, cols0, rows1);
                FunctionValues().Resize(rows0, cols1);
            }

            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(1, false); //the second one is the input since it's column wize

            //after multiplication the structure is lost
            m_outputWidth = 1;
            m_outputHeight = Inputs(0)->FunctionValues().GetNumRows();
            m_outputChannels = 1;
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode, const ComputationNodePtr strideNode)
        {
            m_children.resize(3);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
            m_children[2] = strideNode;
        }
    };

    template class StrideTimesNode<float>;
    template class StrideTimesNode<double>;

}}}
