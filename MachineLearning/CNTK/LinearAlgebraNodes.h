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
        UsingComputationNodeMembers;
    public:
        NegateNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        NegateNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        NegateNode(const NegateNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new NegateNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Negate";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Negate operation only has one input.");
            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Negate operation only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input)  
        {
            functionValues.AssignDifferenceOf(0, input);
#if NANCHECK
            functionValues.HasNan("Negate");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("Negate operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Negate operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            
            CopyImageSizeFromInputs(); 
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
        UsingComputationNodeMembers;
    public:
        SumElementsNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        SumElementsNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        SumElementsNode(const SumElementsNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new SumElementsNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"SumElements";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("SumElements only has one input.");
            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("SumElements only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.AssignSumOfElements(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("SumElements");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("SumElements operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("SumElements operation: the input node has 0 element.");

            FunctionValues().Resize(1, 1);
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, false);

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

    //this node is used to extract part of the input by rows as the output
    //it has to be continuous segments of rows since each column is treated as one sample
    template<class ElemType>
    class RowSliceNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        RowSliceNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId), m_startIndex(0), m_numRows (0) 
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        RowSliceNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        RowSliceNode(const RowSliceNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }
        
        RowSliceNode(const DEVICEID_TYPE deviceId, size_t start_index, size_t num_rows, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            m_startIndex = start_index;
            m_numRows = num_rows;
            

            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new RowSliceNode<ElemType>(this, name, flags);
            return node;
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            RowSliceNode<ElemType>* node = (RowSliceNode<ElemType>*) nodeP;

            node->m_startIndex = m_startIndex;
            node->m_numRows = m_numRows;
        }

        virtual void SaveToFile(File& fstream) const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << m_startIndex << m_numRows;
        }
        
        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            fstream >> m_startIndex >> m_numRows;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"RowSlice";} 

        virtual void PrintSelfBeforeValidation(bool allowNulls = false) const
        {
            fprintf(stderr, "\nValidating --> %ls = %ls", NodeName().c_str(), OperationName().c_str());

            if (!IsLeaf())
            {
                fprintf(stderr, "(");
                for (size_t i = 0; i<ChildrenSize(); i++)
                {
                    ComputationNodePtr child = Inputs(i);
                    if (i > 0)
                        fprintf(stderr, ", ");

                    if (child == nullptr)
                    {
                        if (allowNulls)
                        {
                            fprintf(stderr, "NULL");
                            continue;
                        }
                        throw runtime_error("One of the children is missing.");
                    }

                    fprintf(stderr, "%ls[%lu, %lu]", child->NodeName().c_str(), child->FunctionValues().GetNumRows(), child->FunctionValues().GetNumCols());

                }
                fprintf(stderr, ", StartIndex=%lu, NumOfRows=%lu)", m_startIndex, m_numRows);
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("RowSlice only has one input.");

            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues(), m_startIndex, m_numRows);
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("RowSlice only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue, m_startIndex, m_numRows);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues, const size_t startIndex, const size_t numRows)  
        {
            functionValues.AssignRowSliceValuesOf(inputFunctionValues, startIndex, numRows);
#if NANCHECK
            functionValues.HasNan("RowSlice");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("RowSlice operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("RowSlice operation: the input node has 0 element.");

            if (Inputs(0)->FunctionValues().GetNumRows() < m_startIndex + m_numRows)
                throw std::logic_error("RowSlice operation: m_startIndex + m_numRows exceeds number of rows in the input.");

            FunctionValues().Resize(m_numRows, Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, true);
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
        UsingComputationNodeMembers;
    public:
        RowStackNode(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        RowStackNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        RowStackNode(const RowStackNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;

            ComputationNodePtr node = new RowStackNode<ElemType>(this, name, flags);
            return node;
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            RowStackNode<ElemType>* node = (RowStackNode<ElemType>*) nodeP;

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
                throw std::invalid_argument("RowStack-ComputeInputPartial: inputIndex out of range.");

            ComputeInputPartialS(Inputs(inputIndex)->GradientValues(), GradientValues(), m_startRowIndeces[inputIndex], m_startRowIndeces[inputIndex + 1] - m_startRowIndeces[inputIndex]);
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex >= ChildrenSize())
                throw std::invalid_argument("RowStack-ComputeInputPartial: inputIndex out of range.");

            Matrix<ElemType> sliceInputGrad = Inputs(inputIndex)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            Matrix<ElemType> sliceFunctionValues = FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceFunctionValues, m_inputMatrices, timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const std::vector<const Matrix<ElemType>*>& inputMatrices, const size_t sliceStartCol, const size_t sliceNumCols)
        {
            functionValues.AssignRowStackValuesOf(inputMatrices, sliceStartCol, sliceNumCols);
#if NANCHECK
            functionValues.HasNan("RowStack");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();
            
            unsigned int numInputs = ChildrenSize();
            if (numInputs < 2)
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
            CopyImageSizeFromInputs();
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, true);
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
        UsingComputationNodeMembers;
    public:
        ScaleNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        ScaleNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        ScaleNode(const ScaleNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new ScaleNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Scale";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("ScaleNode operation only takes two inputs.");

            //left Node must be a scalar
            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
            }
            else
            {
                ComputeInputPartialRight(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues());
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("ScaleNode operation only takes two inputs.");

            //left Node must be a scalar
            if (inputIndex == 0)  //left derivative
            {
                Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialLeft(sliceInput1Value, Inputs(0)->GradientValues(), sliceOutputGrad);
            }
            else
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, const Matrix<ElemType>& input1)  
        {
            functionValues.AssignProductOf(input0.Get00Element(), input1);
#if NANCHECK
            functionValues.HasNan("Scale");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("Scale operation requires two inputs.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Scale operation: one of the operands has 0 element.");

            if (Inputs(0)->FunctionValues().GetNumRows() != 1 || Inputs(0)->FunctionValues().GetNumCols() != 1)
                throw std::logic_error("The left value of ScaleNode must be a scalar value.");

            FunctionValues().Resize(Inputs(1)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());
            //left Node must be a scalar
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(1); 
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
        UsingComputationNodeMembers;
    public:
        TimesNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        TimesNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        TimesNode(const TimesNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new TimesNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Times";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("Times operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
            }
            else  //right derivative
            {
                ComputeInputPartialRight(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues());
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("Times operation only takes two inputs.");

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
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("Times operation requires two inputs.");

            //support automatic dimention inference for learnable parameters
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if ((rows0 == 0 || cols1 == 0 ) && this->LoopId() < 0)
                throw logic_error("Times operation: Inputs(0)->FunctionValues().GetNumRows() and Inputs(1)->FunctionValues().GetNumCols() should not be 0 since it cannot be automatically inferred");

            if ((Inputs(0)->OperationName() == LearnableParameter<ElemType>::TypeName() && cols0 == 0 && rows1 != 0) && this->LoopId() < 0)
                Inputs(0)->FunctionValues().Resize(rows0, rows1);

            if (Inputs(1)->OperationName() == LearnableParameter<ElemType>::TypeName() && cols0 != 0 && rows1 == 0)
                Inputs(1)->FunctionValues().Resize(cols0, cols1);

            if ((Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)&& this->LoopId() < 0)
                throw std::logic_error("Times operation: One of the operants has 0 elements.");

            //cols0 and rows1 may have been changed so don't use them in the following check
            if ((Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumCols()) && this->LoopId() < 0)
            {
                throw std::logic_error("The Matrix dimension in the Times operation does not match.");
            }
            FunctionValues().Resize(rows0, cols1);
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()  
        {
            CopyImageSizeFromInput(1, false); //the second one is the input since it's column wize

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
    class ElementTimesNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        ElementTimesNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        ElementTimesNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        ElementTimesNode(const ElementTimesNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new ElementTimesNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"ElementTimes";} 

        virtual void ComputeInputPartial(const size_t inputIndex)  
        {
            if (inputIndex > 1)
                throw std::invalid_argument("ElementTimes operation only takes two inputs.");

            ComputeInputPartialS(Inputs(1-inputIndex)->FunctionValues(), Inputs(inputIndex)->GradientValues(), GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)  
        {
            if (inputIndex > 1)
                throw std::invalid_argument("ElementTimes operation only takes two inputs.");

            Matrix<ElemType> sliceInput0Grad = Inputs(inputIndex)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInput1Value = Inputs(1-inputIndex)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, const Matrix<ElemType>& input1)  
        {
            functionValues.AssignElementProductOf(input0, input1);
#if NANCHECK
            functionValues.HasNan("ElementTimes");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("ElementTimes operation requires two inputs.");

            size_t index = 0;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            index = 1;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("ElementTimes operation: one of the operants has 0 element.");

            if (Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumRows() ||
                Inputs(1)->FunctionValues().GetNumCols() != Inputs(0)->FunctionValues().GetNumCols())
                throw std::logic_error("The Matrix<ElemType> dimension in the ElementTimes operation does not match.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()
        {
            if (IsChildAnImage(0))  //when conflict, give priority to child 0
                CopyImageSizeFromInput(0);
            else
                CopyImageSizeFromInput(1);
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
    class PlusNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        PlusNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        PlusNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        PlusNode(const PlusNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new PlusNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Plus";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("Plus operation only takes two inputs.");
            ComputationNodePtr child = Inputs(inputIndex);
            ComputeInputPartialS(FunctionValues(), GradientValues(), child->FunctionValues(), child->GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("Plus operation only takes two inputs.");

            //only the one with more columns can be sliced, if both have same columns both are sliced
            size_t cols0 = Inputs(inputIndex)->FunctionValues().GetNumCols(), cols1=Inputs(1-inputIndex)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (cols0 >= cols1)
            {
                Matrix<ElemType> sliceInput0Grad = Inputs(inputIndex)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput0Value = Inputs(inputIndex)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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
            else
                throw std::runtime_error("Plus partial: unexpected condition.");
#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
                }


        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            size_t cols0 = Inputs(0)->FunctionValues().GetNumCols(), cols1=Inputs(1)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            //only the one with more columns can be sliced, if both have same columns both are sliced
            if (cols0 == cols1)
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value);
            }
            else if (cols0 > cols1)
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, Inputs(1)->FunctionValues());
            }
            else //cols0 < cols1)
            {
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

#if NANCHECK
            functionValues.HasNan("Plus");
#endif
#if DUMPOUTPUT
            functionValues.Print("PlusNode");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("Plus operation requires two inputs.");

            //if dimention not specified we assume two operants' dimentions should be the same
            size_t index = 0;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            index = 1;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            if ((Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0) && this->LoopId() < 0)
                throw std::logic_error("Plus operation: one of the operants has 0 element.");

            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if ((!(rows0 == rows1 && cols0 == cols1) &&  //match size
                !((rows0 == 1 || rows1 == 1) && cols0 == cols1) && //one is row vec
                !((cols0 == 1 && rows1 % rows0 == 0) || (cols1 == 1 && rows0 % rows1 == 0)))&& this->LoopId() < 0) //one is col vec with divisable rows, including scalar
            {
                throw std::logic_error("The Matrix dimension in the Plus operation does not match.");
            }       

            FunctionValues().Resize(max(rows0, rows1), max(cols0,cols1) );
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs() //based on the matrix with larger size
        {
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if (rows0 > rows1 || cols0 > cols1) //child 0 is larger
                CopyImageSizeFromInput(0);
            else if (rows0 < rows1 || cols0 < cols1) //child 1 is larger
                CopyImageSizeFromInput(1);
            else //same size
            {
                if (IsChildAnImage(0))  //when conflict, give priority to child 0
                    CopyImageSizeFromInput(0);
                else
                    CopyImageSizeFromInput(1);
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
        UsingComputationNodeMembers;
    public:
        MinusNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        MinusNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        MinusNode(const MinusNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new MinusNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Minus";}

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("Minus operation only takes two inputs.");

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

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq) 
        {
            //only the one with more columns can be sliced, if both have same columns both are sliced
            size_t cols0 = Inputs(inputIndex)->FunctionValues().GetNumCols(), cols1=Inputs(1-inputIndex)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInput0Grad = Inputs(inputIndex)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput0Value = Inputs(inputIndex)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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
                throw std::runtime_error("Minus partial: unexpected condition.");
        }


        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());  
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            size_t cols0 = Inputs(0)->FunctionValues().GetNumCols(), cols1=Inputs(1)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            //only the one with more columns can be sliced, if both have same columns both are sliced
            if (cols0 == cols1)
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value);
            }
            else if (cols0 > cols1)
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, Inputs(1)->FunctionValues());
            }
            else //cols0 < cols1)
            {
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("Minus operation requires two inputs.");

            //if dimention is missing make the two operatants to have same size
            size_t index = 0;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            index = 1;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
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
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs() //based on the matrix with larger size
        {
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if (rows0 > rows1 || cols0 > cols1) //child 0 is larger
                CopyImageSizeFromInput(0);
            else if (rows0 < rows1 || cols0 < cols1) //child 1 is larger
                CopyImageSizeFromInput(1);
            else //same size
            {
                if (IsChildAnImage(0))  //when conflict, give priority to child 0
                    CopyImageSizeFromInput(0);
                else
                    CopyImageSizeFromInput(1);
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
        UsingComputationNodeMembers;
    public:
        DiagTimesNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode<ElemType>(deviceId), m_innerproduct(deviceId), m_rightGradient(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        DiagTimesNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_innerproduct(deviceId), m_rightGradient(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"DiagTimes";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("DiagTimes operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(m_innerproduct, Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
            }
            else  //right derivative
            {
                ComputeInputPartialRight(m_rightGradient, Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues());
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("DiagTimes operation only takes two inputs.");

            //left parameter (diag matix cannot be sliced)
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (inputIndex == 0)  //left derivative
            {
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                ComputeInputPartialLeft(m_innerproduct, sliceInput1Value, Inputs(0)->GradientValues(), sliceOutputGrad);
            }
            else  //right derivative
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
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

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value); 
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues0, const Matrix<ElemType>& inputFunctionValues1)  
        {
            functionValues.SetValue(inputFunctionValues1);
            functionValues.ColumnElementMultiplyWith(inputFunctionValues0);
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("DiagTimes operation requires two inputs.");

            //if dimention not specified we assume two operants' dimentions should match
            if (Inputs(0)->OperationName() == LearnableParameter<ElemType>::TypeName() && Inputs(0)->FunctionValues().GetNumRows() == 0 && Inputs(1)->FunctionValues().GetNumRows() != 0)
            {
                Inputs(0)->FunctionValues().Resize(Inputs(1)->FunctionValues().GetNumRows(), 1);
            }

            if (Inputs(1)->OperationName() == LearnableParameter<ElemType>::TypeName() && Inputs(0)->FunctionValues().GetNumRows() != 0 && Inputs(1)->FunctionValues().GetNumRows() == 0)
            {
                Inputs(1)->FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());
            }

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("DiagTimes operation: one of the operants has 0 element.");

            if (Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumRows())
                throw std::logic_error("The Matrix dimension in the DiagTimes operation does not match.");

            if (1 != Inputs(0)->FunctionValues().GetNumCols())
                throw std::logic_error("The first matrix should be a vector regpresting the diagonal of a square matrix in the DiagTimes operation.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());
            m_innerproduct.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());
            m_rightGradient.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());

            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs() //this is element wise scaling, so based on child 1
        {
            CopyImageSizeFromInput(1);
        }

        virtual void AttachInputs(const ComputationNodePtr leftNode, const ComputationNodePtr rightNode) 
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_innerproduct.GetDeviceId() != deviceId)
                    m_innerproduct.TransferFromDeviceToDevice(m_innerproduct.GetDeviceId(), deviceId);
                if (m_rightGradient.GetDeviceId() != deviceId)
                    m_rightGradient.TransferFromDeviceToDevice(m_rightGradient.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            DiagTimesNode<ElemType>* node = (DiagTimesNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_innerproduct = m_innerproduct;
                node->m_rightGradient = m_rightGradient;
            }
        }

        // copy constructor
        DiagTimesNode(const DiagTimesNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_innerproduct(node->m_deviceId), m_rightGradient(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new DiagTimesNode<ElemType>(this, name, flags);
            return node;
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
        UsingComputationNodeMembers;
    public:
        CosDistanceNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode<ElemType>(deviceId), m_invNorm0(deviceId), m_invNorm1(deviceId), m_leftTerm(deviceId), m_rightTerm(deviceId), m_temp(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        CosDistanceNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_invNorm0(deviceId), m_invNorm1(deviceId), m_leftTerm(deviceId), m_rightTerm(deviceId), m_temp(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"CosDistance";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("CosDistance operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(m_invNorm0, m_invNorm1, FunctionValues(), m_temp, m_rightTerm, m_leftTerm, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), GradientValues(), Inputs(inputIndex)->GradientValues());
            }
            else  //right derivative
            {
                ComputeInputPartialRight(m_invNorm0, m_invNorm1, FunctionValues(), m_temp, m_rightTerm, m_leftTerm, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), GradientValues(), Inputs(inputIndex)->GradientValues());
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq) 
        {
            if (inputIndex > 1)
                throw std::invalid_argument("CosDistance operation only takes two inputs.");

            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInputGrad = Inputs(inputIndex)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = this->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("CosDistance operation requires two inputs.");

            //if dimention is missing make the two operatants to have same size
            size_t index = 0;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            index = 1;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0? Inputs(1-index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0? Inputs(1-index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("CosDistance operation: one of the operants has 0 element.");

            if (Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumRows() || 
                Inputs(1)->FunctionValues().GetNumCols() != Inputs(0)->FunctionValues().GetNumCols())
                throw std::logic_error("The Matrix dimension in the CosDistance operation does not match.");

            FunctionValues().Resize(1, Inputs(1)->FunctionValues().GetNumCols());

            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs() 
        {
            CopyImageSizeFromInput(0, false);

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
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_invNorm0.GetDeviceId() != deviceId)
                    m_invNorm0.TransferFromDeviceToDevice(m_invNorm0.GetDeviceId(), deviceId);
                if (m_invNorm1.GetDeviceId() != deviceId)
                    m_invNorm1.TransferFromDeviceToDevice(m_invNorm1.GetDeviceId(), deviceId);
                if (m_leftTerm.GetDeviceId() != deviceId)
                    m_leftTerm.TransferFromDeviceToDevice(m_leftTerm.GetDeviceId(), deviceId);
                if (m_rightTerm.GetDeviceId() != deviceId)
                    m_rightTerm.TransferFromDeviceToDevice(m_rightTerm.GetDeviceId(), deviceId);
                if (m_temp.GetDeviceId() != deviceId)
                    m_temp.TransferFromDeviceToDevice(m_temp.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            CosDistanceNode<ElemType>* node = (CosDistanceNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_invNorm0 = m_invNorm0;
                node->m_invNorm1 = m_invNorm1;
                node->m_leftTerm = m_leftTerm;
                node->m_rightTerm = m_rightTerm;
                node->m_temp = m_temp;
            }
        }

        // copy constructor
        CosDistanceNode(const CosDistanceNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_invNorm0(node->m_deviceId), m_invNorm1(node->m_deviceId), m_leftTerm(node->m_deviceId), m_rightTerm(node->m_deviceId), m_temp(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new CosDistanceNode<ElemType>(this, name, flags);
            return node;
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
        UsingComputationNodeMembers;
    public:
        KhatriRaoProductNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        KhatriRaoProductNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        KhatriRaoProductNode(const KhatriRaoProductNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new KhatriRaoProductNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"KhatriRaoProduct";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("KhatriRaoProduct operation only takes two inputs.");

            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues()); 
            }
            else  //right derivative
            {
                ComputeInputPartialRight(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues()); 
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq) 
        {
            if (inputIndex > 1)
                throw std::invalid_argument("KhatriRaoProduct operation only takes two inputs.");

            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (inputIndex == 0)  //left derivative
            {
                Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                ComputeInputPartialLeft(sliceInput1Value, sliceInput0Grad, sliceOutputGrad); 
            }
            else  //right derivative
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value); 
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, Matrix<ElemType>& in0, Matrix<ElemType>& in1)  
        {
            functionValues.AssignKhatriRaoProductOf(in0,in1);
#if NANCHECK
            functionValues.HasNan("KhatriRaoProduct");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("KhatriRaoProduct operation requires two inputs.");

            //support automatic dimention inference for learnable parameters
            size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();
            size_t rows1 = Inputs(1)->FunctionValues().GetNumRows(), cols1 = Inputs(1)->FunctionValues().GetNumCols();

            if (rows0 == 0 || rows1 == 0)
                throw logic_error("KhatriRaoProduct operation: The number of rows in the input should not be 0.");

            if (Inputs(0)->OperationName() == LearnableParameter<ElemType>::TypeName() && cols0 == 0 && cols1 != 0)
                Inputs(0)->FunctionValues().Resize(rows0, cols1);

            if (Inputs(1)->OperationName() == LearnableParameter<ElemType>::TypeName() && cols0 != 0 && cols1 == 0)
                Inputs(1)->FunctionValues().Resize(rows1, cols0);

            //cols may be changed before this line and so cannot use cached cols values below
            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("KhatriRaoProduct operation: One of the operants has 0 elements.");

            if (Inputs(1)->FunctionValues().GetNumCols() != Inputs(0)->FunctionValues().GetNumCols())
            {
                throw std::logic_error("The Matrices should have same number of columns.");
            }

            FunctionValues().Resize(rows0 * rows1, Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()  
        {
            //since it's symmetrical any one of the input may be the true input. 
            //since we dont' use the input image size info in the operation, the input part doesn't matter.
            CopyImageSizeFromInput(1, false); 

            //after KhatriRaoProduct the structure is lost
            m_outputWidth = 1;
            m_outputHeight = m_functionValues.GetNumRows();
            m_outputChannels = 1;
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
        //typedef ComputationNode<ElemType>* ComputationNodePtr;
        UsingComputationNodeMembers;

    public:
        CosDistanceWithNegativeSamplesNode(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_invNorm0(deviceId), m_invNorm1(deviceId), m_invNormSquare(deviceId), 
            m_leftTerm(deviceId), m_rightTerm(deviceId), m_temp(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        CosDistanceWithNegativeSamplesNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_invNorm0(deviceId), m_invNorm1(deviceId), m_invNormSquare(deviceId), 
            m_leftTerm(deviceId), m_rightTerm(deviceId), m_temp(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"CosDistanceWithNegativeSamples"; }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("CosDistanceWithNegativeSamples operation only takes grdients on the first two inputs.");

            ComputeInputPartialS(inputIndex, m_invNorm0, m_invNorm1, FunctionValues(), m_temp, m_rightTerm, m_leftTerm, m_invNormSquare, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), Inputs(3)->FunctionValues(), Inputs(inputIndex)->GradientValues(), GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("CosDistanceWithNegativeSamples operation only takes grdients on the first two inputs.");

            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInputGrad = Inputs(inputIndex)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceThisGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 4)
                throw std::logic_error("CosDistanceWithNegativeSamples operation requires 4 inputs.");

            //if dimention is missing make the two operatants to have same size
            size_t index = 0;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0 ? Inputs(1 - index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0 ? Inputs(1 - index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            index = 1;
            if (Inputs(index)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(index)->FunctionValues().GetNumRows() == 0 ? Inputs(1 - index)->FunctionValues().GetNumRows() : Inputs(index)->FunctionValues().GetNumRows();
                size_t cols = Inputs(index)->FunctionValues().GetNumCols() == 0 ? Inputs(1 - index)->FunctionValues().GetNumCols() : Inputs(index)->FunctionValues().GetNumCols();
                Inputs(index)->FunctionValues().Resize(rows, cols);
            }

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("CosDistanceWithNegativeSamples operation: one of the operants has 0 element.");

            if (Inputs(1)->FunctionValues().GetNumRows() != Inputs(0)->FunctionValues().GetNumRows() ||
                Inputs(1)->FunctionValues().GetNumCols() != Inputs(0)->FunctionValues().GetNumCols())
                throw std::logic_error("The Matrix dimension in the CosDistanceWithNegativeSamples operation does not match.");

            // input(2) is shift, input(3) is the #neg
            size_t negNumber = (size_t)Inputs(3)->FunctionValues()(0, 0);

            FunctionValues().Resize(negNumber + 1, Inputs(1)->FunctionValues().GetNumCols());

            CopyImageSizeFromInputs();
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, false);

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
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_invNorm0.GetDeviceId() != deviceId)
                    m_invNorm0.TransferFromDeviceToDevice(m_invNorm0.GetDeviceId(), deviceId);
                if (m_invNorm1.GetDeviceId() != deviceId)
                    m_invNorm1.TransferFromDeviceToDevice(m_invNorm1.GetDeviceId(), deviceId);
                if (m_invNormSquare.GetDeviceId() != deviceId)
                    m_invNormSquare.TransferFromDeviceToDevice(m_invNormSquare.GetDeviceId(), deviceId);
                if (m_leftTerm.GetDeviceId() != deviceId)
                    m_leftTerm.TransferFromDeviceToDevice(m_leftTerm.GetDeviceId(), deviceId);
                if (m_rightTerm.GetDeviceId() != deviceId)
                    m_rightTerm.TransferFromDeviceToDevice(m_rightTerm.GetDeviceId(), deviceId);
                if (m_temp.GetDeviceId() != deviceId)
                    m_temp.TransferFromDeviceToDevice(m_temp.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            CosDistanceWithNegativeSamplesNode<ElemType>* node = (CosDistanceWithNegativeSamplesNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_invNorm0 = m_invNorm0;
                node->m_invNorm1 = m_invNorm1;
                node->m_invNormSquare = m_invNormSquare;
                node->m_leftTerm = m_leftTerm;
                node->m_rightTerm = m_rightTerm;
                node->m_temp = m_temp;
            }
        }

        // copy constructor
        CosDistanceWithNegativeSamplesNode(const CosDistanceWithNegativeSamplesNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_invNorm0(node->m_deviceId), m_invNorm1(node->m_deviceId), m_leftTerm(node->m_deviceId), m_rightTerm(node->m_deviceId), m_temp(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;

            ComputationNodePtr node = new CosDistanceWithNegativeSamplesNode<ElemType>(this, name, flags);
            return node;
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

}}}
