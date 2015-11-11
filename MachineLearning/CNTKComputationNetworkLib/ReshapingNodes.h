// ReshapingNodes.h -- collection of nodes that reshape or sub-sample matrices leading to layout changes
//
// <copyright file="NonlinearityNodes.h" company="Microsoft">
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

    // -----------------------------------------------------------------------
    // ReshapingNodeBase (input) -- base class for nodes that reshape
    // -----------------------------------------------------------------------

    template<class ElemType>
    class ReshapingNodeBase : public ComputationNode<ElemType>, public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        ReshapingNodeBase(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        // stack K consecutive frames into a single frame that is K times taller
        // FrameRange and MBLayout refer to the 'to' (reduced) timeline.
        // BUGBUG: THIS IS UNTESTED!!
        static void Stack(const FrameRange & frameRange, const shared_ptr<MBLayout> & pMBLayout, /*const*/ Matrix<ElemType> & from, Matrix<ElemType> & to, size_t K, bool addTo)
        {
            // example
            //  input: T=2, D=2, K=3, S=2 (abcdef and uvwxyz)
            //   abc def
            //   ABC DEF
            //  
            //   uvw xyz
            //   UVW XYZ
            //  target:
            //   a d
            //   A D
            //   b e
            //   B E
            //   c f
            //   C F
            //  
            //   u x
            //   U X
            //   v y
            //   V Y
            //   w z
            //   W Z
            // underlying matrix storage is actually this:
            //  input:
            //   aubvcw dxeyfz
            //   AUBVCW DXEYFZ
            //  target:
            //   abcuvw defxyz
            //   ABCUVW DEFXYZ

            // I.e. this operation swaps index dimensions of a tensor:
            //   The input is a tensor of the form (D,       S, M, K, T).
            //   The output is of the form         (D, K, M, S,       T).
            //     K = stacking factor
            //     T = target steps
            //     S = #sequences
            //     D = featDim
            //     M = 1, thrown in for generality of underlying Matrix function

            // We operate on the 'to' layout, frameRange refers to result, not the input.
            // The input layout is different, but reshaping the input to output dimensions will allow us to pull out the right values anyway.
            auto from0      = from.Reshaped(to.GetNumRows(), to.GetNumCols());   // we operate on 'to' layout
            auto fromSlice0 = DataSlice(from0, frameRange, pMBLayout);
            auto   toSlice0 = DataSlice(to,    frameRange, pMBLayout);
            // now we got views on the right ranges of values, but with weird dimensions

            // reshape them into a unified view with D being the row dimension, and (S,M,K,T) the column dimension
            size_t    D = from.GetNumRows();
            size_t SMKT = from.GetNumCols();
            auto fromSlice = fromSlice0.Reshaped(D, SMKT);
            auto   toSlice =   toSlice0.Reshaped(D, SMKT);

            // now to the shuffle dance
            size_t S = pMBLayout->GetNumParallelSequences();
            size_t T = pMBLayout->GetNumTimeSteps();
            size_t M = 1;
            Matrix<ElemType>::TensorShuffleScaleAndAdd(addTo ? 1.0f : 0, fromSlice, D, S, M, K, T, 1.0f, toSlice, toSlice);
        }

        // split frames of D*K elements into K consecutive frames of dimension D.
        // FrameRange and MBLayout refer to the 'from' (reduced) timeline.
        // This function is the inverse of Stack(). See comments there and exchange from and to.
        static void Unstack(const FrameRange & frameRange, const shared_ptr<MBLayout> & pMBLayout, /*const*/ Matrix<ElemType> & from, Matrix<ElemType> & to, size_t K, bool addTo)
        {
            auto fromSlice0 = DataSlice(from, frameRange, pMBLayout);
            auto   to0      = to.Reshaped(from.GetNumRows(), from.GetNumCols());
            auto   toSlice0 = DataSlice(to0, frameRange, pMBLayout);

            size_t    D = to.GetNumRows();
            size_t SMKT = to.GetNumCols();
            auto fromSlice = fromSlice0.Reshaped(D, SMKT);
            auto   toSlice =   toSlice0.Reshaped(D, SMKT);

            size_t S = pMBLayout->GetNumParallelSequences();
            size_t T = pMBLayout->GetNumTimeSteps();
            size_t M = 1;
            Matrix<ElemType>::TensorShuffleScaleAndAdd(addTo ? 1.0f : 0, fromSlice, D, K, M, S, T, 1.0f, toSlice, toSlice);
        }
    };

#define UsingReshapingNodeBaseMembers UsingComputationNodeMembersBoilerplate

    // -----------------------------------------------------------------------
    // ReshapeNode (input) -- reshape input matrix
    //
    // If input has no layout, then this reshapes the input matrix
    // from (rows x cols) to (newRows x (cols / newRows * rows)).
    //
    // If input has a layout, then it changes the number of time steps, i.e.
    // from (rows x T time steps) to (newRows x (T / newRows * rows) time steps).
    // E.g. going from rows=20 to newRows=40 groups two consecutive time steps into one.
    // In this case, multiple parallel sequences are treated independently.
    //
    // Unlike most other nodes, this node has intimate inside knowlegde of MBLayouts and frameRanges.
    //
    // BUGBUG: THIS IS UNTESTED for non-layout case!!  --TODO: remove these comments once tested
    // BUGBUG: THIS IS UNTESTED for MBLayout case!!
    // -----------------------------------------------------------------------

    template<class ElemType>
    class ReshapeNode : public ReshapingNodeBase<ElemType>
    {
        typedef ReshapingNodeBase<ElemType> Base; UsingReshapingNodeBaseMembers;
        static const std::wstring TypeName() { return L"Reshape"; }
    public:
        ReshapeNode(DEVICEID_TYPE deviceId, const wstring & name, size_t numRows = 0, const ImageLayout & imageLayout = ImageLayout(0,0,0)) :
            Base(deviceId, name),
            m_numRows(numRows),
            m_imageLayout(imageLayout)
        { }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<ReshapeNode<ElemType>>(nodeP);
                node->m_numRows = m_numRows;
                node->m_imageLayout = m_imageLayout;
            }
        }

        virtual void SaveToFile(File& fstream) const override
        {
            Base::SaveToFile(fstream);
            fstream << m_numRows << m_imageLayout.width << m_imageLayout.height << m_imageLayout.channels;
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_numRows >> m_imageLayout.width >> m_imageLayout.height >> m_imageLayout.channels;
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, true);
            InferImageDimensions();

            if (m_imageLayout.width == 0 || m_imageLayout.height == 0 || m_imageLayout.channels == 0)
            {
                m_outputImageLayout = ImageLayout(1, 1, m_numRows);
                if (m_inputImageLayout.width * m_inputImageLayout.channels != 1)
                    fprintf(stderr, "WARNING: Reshape operation cannot inherit image size information from its child. Image size info is lost.\n");
            }
            else
            {
                m_outputImageLayout = m_imageLayout;
            }
        }

        virtual void PrintSelfBeforeValidation(bool allowNulls = false) const
        {
            fprintf(stderr, "\nValidating --> %ls = %ls", NodeName().c_str(), OperationName().c_str());

            if (!IsLeaf())
            {
                fprintf(stderr, "(");
                for (size_t i = 0; i < ChildrenSize(); i++)
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
                        RuntimeError("One of the children is missing.");
                    }

                    fprintf(stderr, "%ls[%lu, %lu]", child->NodeName().c_str(), child->GetNumRows(), child->GetNumCols());
                }

                fprintf(stderr, ", NumOfRows=%lu, imageWidth=%lu, imageHeight=%lu, imageChannels=%lu)", m_numRows, m_imageLayout.width, m_imageLayout.height, m_imageLayout.channels);
            }
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            size_t rows = Inputs(0)->GetNumRows(), cols = Inputs(0)->GetNumCols();
            // Note: During initial validation, cols may not be a multiple. E.g. cols may be 1 or 3. So we cannot check here whether the integer-multiple conditions are fulfilled.
            size_t newCols = cols * rows / m_numRows;
            if (isFinalValidationPass)
            {
                if ((m_numRows > rows && m_numRows % rows != 0) ||  // grouping columns
                    (m_numRows < rows && rows % m_numRows != 0))    // splitting columns
                    InvalidArgument("%ls %ls operation: output row dimension %d is not an integer multiple or divisor of input dimension %d", NodeName().c_str(), OperationName().c_str(), (int)m_numRows, (int)rows);
                if (!m_pMBLayout && rows * cols != m_numRows * newCols)    // sadly, cannot verify here if we have a layout, since current #cols may be bogus
                    LogicError("%ls %ls operation: unexpected dimension mismatch", NodeName().c_str(), OperationName().c_str());
            }

            Resize(m_numRows, newCols);
            if (Inputs(0)->HasMBLayout())
            {
                if (!m_pMBLayout)
                    m_pMBLayout = make_shared<MBLayout>();  // mini-batch data: this generates its own layout
            }
            else
                assert(!m_pMBLayout);                       // reshaping non-mini-batch data
            InferImageDimsFromInputs();
        }

        virtual size_t UpdateFunctionMBSize(size_t numCols) override
        {
            // BUGBUG: numCols parameter is legacy and not really supported
            size_t rows = Inputs(0)->GetNumRows(), cols = Inputs(0)->GetNumCols();
            size_t newCols = cols * rows / m_numRows;
            if (!m_pMBLayout)
            {
#if 0
                VerifySize(m_numRows, newCols);
#endif
            }
            else
                Resize(m_numRows, newCols);
            return numCols;
        }

        // TODO: there seems to be semantic overlap between OnEvaluateBeginIteration() and UpdateFunctionMBSize()
        virtual void /*IComputationNode::*/OnEvaluateBeginIteration() override
        {
            if (m_pMBLayout)
            {
                // create the derived layout
                // BUGBUG: This assumes that the layout is complete at this point in time. RecurrentNodeBase makes the same assumption.
                //         This assumption is correct at present, but will becomes invalid once we go sequence-to-sequence.
                m_pMBLayout->Init(Inputs(0)->GetNumParallelSequences(), Inputs(0)->GetNumTimeSteps() * Inputs(0)->GetNumRows() / m_numRows);
                if (!m_pMBLayout->IsAllNone())
                    LogicError("ReshapeNode::OnEvaluateBeginIteration() to be completed for MBLayout case.");
                // TODO: ^^ MBLayout update
            }
        }

        // notes:
        //  - input and output have different time base and different layouts
        //  - frameRange refers to *functionValues*, not the inputs
        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            size_t rows = Inputs(0)->GetNumRows(), cols = Inputs(0)->GetNumCols();
            size_t newCols = cols * rows / m_numRows;
            assert(newCols * m_numRows == cols * rows); // follows from above check
            VerifySize(m_numRows, newCols);

            // no layout case: this is indeed just a reshape
            // (We still need to copy the values since there is currently no way to point to an input function value while reshaping at the same time.)
            if (!m_pMBLayout)
            {
                FunctionValues().Reshaped(newCols * m_numRows, 1).SetValue(Inputs(0)->FunctionValues().Reshaped(cols * rows, 1));   // copy the values as one long vector
            }
            // layout case: reshape semantics happens across parallel seqeunces, i.e. requiring data shuffling
            else
            {
                // TODO: It does not make sense to run ReshapeNode frame-by-frame inside a loop, because it changes the time base.
                //       However, in the future, we should be able to run inside an outer loop.
                if (!frameRange.IsAllFrames())
                    InvalidArgument("%ls %ls operation cannot be run from inside a loop since it changes the time base.", NodeName().c_str(), OperationName().c_str());
                if (weStack())
                    Base::Stack(frameRange, m_pMBLayout, Inputs(0)->FunctionValues(), FunctionValues(), factor(), false/*addTo*/);
                else
                    Base::Unstack(frameRange, Inputs(0)->GetMBLayout(), Inputs(0)->FunctionValues(), FunctionValues(), factor(), false/*addTo*/);
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange & frameRange) override
        {
            size_t rows = Inputs(0)->GetNumRows(), cols = Inputs(0)->GetNumCols();
            size_t newCols = cols * rows / m_numRows;

            // no layout case: this is indeed just a reshape
            if (!m_pMBLayout || isNoop())
            {
                Inputs(0)->GradientValues().Reshaped(cols * rows, 1) += GradientValues().Reshaped(newCols * m_numRows, 1);   // treat the values as one long vector
            }
            // layout case: reshape semantics happens across parallel seqeunces, i.e. requiring data shuffling
            else
            {
                if (weStack())
                    Base::Unstack(frameRange, m_pMBLayout, GradientValues(), Inputs(0)->GradientValues(), factor(), true/*addTo*/);
                else
                    Base::Stack(frameRange, Inputs(0)->GetMBLayout(), GradientValues(), Inputs(0)->GradientValues(), factor(), true/*addTo*/);
            }
        }

    private:
        size_t m_numRows;
        bool weStack() const { return m_numRows > Inputs(0)->GetNumRows(); }        // do we stack (multiple frames into one)
        size_t factor() const { return m_numRows > Inputs(0)->GetNumRows() ? m_numRows / Inputs(0)->GetNumRows() : Inputs(0)->GetNumRows() / m_numRows; }   // factor by which we stack or unstack
        bool isNoop() const { return m_numRows == Inputs(0)->GetNumRows(); }        // TODO: we also must test for changes in image layout
        ImageLayout m_imageLayout;

        void InferImageDimensions()
        {
            if (m_imageLayout.width > 0)
            {
                if (m_imageLayout.height > 0)
                {
                    if (m_imageLayout.channels > 0)
                    {
                        if (m_imageLayout.GetNumElements() != m_numRows)
                            RuntimeError("Image dimensions do not match row size.");
                    }
                    else
                    {
                        if (m_numRows % (m_imageLayout.width * m_imageLayout.height) > 0)
                            RuntimeError("Image row size is not a multiple of specified image dimensions.");
                        else
                            m_imageLayout.channels = m_numRows / (m_imageLayout.width * m_imageLayout.height);
                    }
                }
                else
                {
                    if (m_imageLayout.channels > 0)
                    {
                        if (m_numRows % (m_imageLayout.width * m_imageLayout.channels) > 0)
                            RuntimeError("Image row size is not a multiple of specified image dimensions.");
                        else
                            m_imageLayout.height = m_numRows / (m_imageLayout.width * m_imageLayout.channels);
                    }
                    else
                    {
                        RuntimeError("At least two image dimensions must be specified.");
                    }
                }
            }
            else
            {
                if (m_imageLayout.height > 0)
                {
                    if (m_imageLayout.channels > 0)
                    {
                        if (m_numRows % (m_imageLayout.height * m_imageLayout.channels) > 0)
                            RuntimeError("Image row size is not a multiple of specified image dimensions.");
                        else
                            m_imageLayout.width = m_numRows / (m_imageLayout.height * m_imageLayout.channels);
                    }
                    else
                        RuntimeError("At least two image dimensions must be specified.");
                }
                else if (m_imageLayout.channels > 0)
                    RuntimeError("At least two image dimensions must be specified.");
            }
        }
    };

    template class ReshapeNode<float>;
    template class ReshapeNode<double>;

    // -----------------------------------------------------------------------
    // RowSliceNode (input)
    // this node extracts part of the input by rows as the output
    // it has to be continuous segments of rows since each column is treated as one sample
    // -----------------------------------------------------------------------

    template<class ElemType>
    class RowSliceNode : public ComputationNode<ElemType>, public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"RowSlice"; }
    public:
        RowSliceNode(DEVICEID_TYPE deviceId, const wstring & name, size_t startIndex = 0, size_t numRows = 0) :
            Base(deviceId, name),
            m_startIndex(startIndex),
            m_numRows(numRows)
        { }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            auto node = dynamic_pointer_cast<RowSliceNode<ElemType>>(nodeP);

            node->m_startIndex = m_startIndex;
            node->m_numRows = m_numRows;
        }

        virtual void SaveToFile(File& fstream) const override
        {
            Base::SaveToFile(fstream);
            fstream << m_startIndex << m_numRows;
        }
        
        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_startIndex >> m_numRows;
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange & frameRange) override
        {
            Inputs(0)->GradientSlice(frameRange).AddToRowSliceValuesOf(GradientSlice(frameRange), m_startIndex, m_numRows);
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            ValueSlice(frameRange).AssignRowSliceValuesOf(Inputs(0)->ValueSlice(frameRange), m_startIndex, m_numRows);
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            if (isFinalValidationPass && Inputs(0)->GetNumRows() < m_startIndex + m_numRows)
                RuntimeError("RowSlice operation: m_startIndex + m_numRows exceeds number of rows in the input.");

            Resize(m_numRows, Inputs(0)->GetNumCols());
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, true);
            m_outputImageLayout.height = m_numRows;        

            //WARNING: this node will destroy the image size information from the child
            if (m_inputImageLayout.width * m_inputImageLayout.channels != 1)
                fprintf(stderr, "WARNING: RowSlice operation cannot inherit image size information from its child. Image size info is lost.\n");
        }

    private:
        size_t m_startIndex, m_numRows;
    };

    template class RowSliceNode<float>; 
    template class RowSliceNode<double>;

    // -----------------------------------------------------------------------
    // RowStackNode (input0, input1, ...)
    // stacks multiple inputs on top of each other
    // -----------------------------------------------------------------------

    template<class ElemType>
    class RowStackNode : public ComputationNode<ElemType>   // note: not deriving from NumInputs<> like most other nodes, because this one takes a variable number of inputs
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"RowStack"; }
    public:
        RowStackNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeChildren)
            {
                auto node = dynamic_pointer_cast<RowStackNode<ElemType>>(nodeP);
                node->m_startRowIndices = m_startRowIndices;
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            Inputs(inputIndex)->GradientSlice(frameRange).AddWithRowSliceValuesOf(GradientSlice(frameRange), m_startRowIndices[inputIndex], Inputs(inputIndex)->GetNumRows());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            for (size_t inputIndex = 0; inputIndex < ChildrenSize(); inputIndex++)
                ValueSlice(frameRange).AssignToRowSliceValuesOf(Inputs(inputIndex)->ValueSlice(frameRange), m_startRowIndices[inputIndex], Inputs(inputIndex)->GetNumRows());
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);
            InferMBLayoutFromInputsForStandardCase();

            size_t numCols = Inputs(0)->GetNumCols();

            // count totalRows and form m_startRowIndices[] array, which is the cumulative sum of matrix heights
            m_startRowIndices.resize(ChildrenSize());
            size_t totalRows = 0;

            for (int i = 0; i < ChildrenSize(); i++)
            {
                if (isFinalValidationPass && Inputs(i)->GetNumCols() != numCols)
                    LogicError("RowStack operation: the input node %ls has different number of columns.", Inputs(i)->NodeName().c_str());

                m_startRowIndices[i] = totalRows;
                totalRows += Inputs(i)->GetNumRows();
            }

            Resize(totalRows, numCols);
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, true);
            m_outputImageLayout.height = GetNumRows();

            //WARNING: this node will destroy the image size information from the child
            if (m_inputImageLayout.width * m_inputImageLayout.channels != 1)
                fprintf(stderr, "WARNING: RowStack operation cannot inherit image size information from its child. Image size info is lost.\n");
        }

    private:
        std::vector<size_t> m_startRowIndices;  // start row number in the stacked matrix of each input (child) (cumsum of matrix heights)
    };

    template class RowStackNode<float>;
    template class RowStackNode<double>;

    // -----------------------------------------------------------------------
    // RowRepeatNode (input) -- duplicate row(s) of a matrix multiple times
    // -----------------------------------------------------------------------

    template<class ElemType>
    class RowRepeatNode : public ComputationNode<ElemType>, public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"RowRepeat"; }
    public:
        RowRepeatNode(DEVICEID_TYPE deviceId, const wstring & name, size_t numRepeats = 1) :
            Base(deviceId, name),
            m_numRepeat(numRepeats)
        { }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<RowRepeatNode<ElemType>>(nodeP);
                node->m_numRepeat = m_numRepeat;
            }
        }

        virtual void SaveToFile(File& fstream) const override
        {
            Base::SaveToFile(fstream);
            fstream << m_numRepeat;
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_numRepeat;
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, true);
            m_outputImageLayout.height = m_inputImageLayout.height * m_numRepeat;

            // WARNING: this node will destroy the image size information from the child
            if (m_inputImageLayout.width * m_inputImageLayout.channels != 1)
                fprintf(stderr, "WARNING: RowRepeat operation cannot inherit image size information from its child. Image size info is lost.\n");
        }

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
                        RuntimeError("One of the children is missing.");
                    }

                    fprintf(stderr, "%ls[%lu, %lu]", child->NodeName().c_str(), child->GetNumRows(), child->GetNumCols());
                }

                fprintf(stderr, ", numRepeats=%lu)", m_numRepeat);
            }
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            Resize(Inputs(0)->GetNumRows() * m_numRepeat, Inputs(0)->GetNumCols());
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            if (!isNoop())    // if m_numRepeat == 1 then virtual FunctionValues() will return the child   --TODO: do this as an in-place optimization instead
                ValueSlice(frameRange).AssignRepeatOf(Inputs(0)->ValueSlice(frameRange), m_numRepeat, 1);
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange & frameRange) override
        {
            Inputs(0)->GradientSlice(frameRange).AddToRowRepeatValuesOf(GradientSlice(frameRange), m_numRepeat);
        }

        // TODO: Can we remove this const-related duplication as well?
        virtual const Matrix<ElemType>& FunctionValues() const
        {
            if (!isNoop())
                return *m_functionValues;
            else
                return Inputs(0)->FunctionValues();
        }

        virtual Matrix<ElemType>& FunctionValues() 
        {
            if (!isNoop())
                return *m_functionValues;
            else
                return Inputs(0)->FunctionValues();
        }

    private:
        bool isNoop() const { return m_numRepeat == 1; }    // in this case this node does nothing
        size_t m_numRepeat;
    };

    template class RowRepeatNode<float>;
    template class RowRepeatNode<double>;

}}}
