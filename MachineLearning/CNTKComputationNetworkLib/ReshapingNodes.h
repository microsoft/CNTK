// ReshapingNodes.h -- collection of nodes that reshape or sub-sample matrices leading to layout changes
//
// <copyright file="ReshapingNodes.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "ComputationNode.h"
#include "Sequences.h"

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
    // ReinterpretNodeBase (input) -- base class for nodes that reinterpret
    // -----------------------------------------------------------------------

    template<class ElemType>
    class ReinterpretNodeBase : public ComputationNode<ElemType>, public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        //DeclareConstructorFromConfigWithNumInputs(ReinterpretNodeBase);
        ReinterpretNodeBase(DEVICEID_TYPE deviceId, const wstring & name) :
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
            auto fromSlice0 = DataSliceWithMBLayout(from0, frameRange, pMBLayout);
            auto   toSlice0 = DataSliceWithMBLayout(to,    frameRange, pMBLayout);
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
            auto fromSlice0 = DataSliceWithMBLayout(from, frameRange, pMBLayout);
            auto   to0      = to.Reshaped(from.GetNumRows(), from.GetNumCols());
            auto   toSlice0 = DataSliceWithMBLayout(to0, frameRange, pMBLayout);

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

#define UsingReinterpretNodeBaseMembers UsingComputationNodeMembersBoilerplate

    // -----------------------------------------------------------------------
    // ReshapeNode (input) -- reinterpret input matrix as having different dimensions
    // where the new row dimension is given, and the column dimension is inferred.
    // Also optionally associate a different ImageLayout with the data.
    //
    // If input has no layout, then this reshapes the input matrix
    // from (rows x cols) to (newRows x (cols / newRows * rows)).
    //
    // If input has a layout, then it adds or removes a nested time dimension.
    //  - If newRows > rows, then we remove a time dimension by stacking all frames from the dimension into one:
    //       (rows x (newRows/rows nested time steps) x T time steps)
    //    -> (newRows x T time steps).
    //  - If newRows < rows, then we add a time dimension, going
    //       (rows x T time steps)
    //    -> (newRows x (rows/newRows nested time steps) x T time steps).
    //    which requires the nested time sequence to have the correct number of steps.
    // E.g. going from rows=20 to newRows=40 assumes a nested time sequence of 2 steps, which are grouped into one step, with the two vectors stacked.
    // Multiple parallel sequences are treated independently.
    // TODO: This definition is poor; we should use a different node name, and specify the factor directly.
    //       We may hide that in BrainScript, but better use different node types.
    //       E.g. ReinterpretRowStackAsSequence and ReinterpretSequenceAsRowStack.
    // BUGBUG: This is not actually implemented yet. Instead, it goes from 1 to K steps or from K to 1 step. This is temporary/experimental, until the plumbing for nesting is there.
    //
    // Thirdly, ReshapeNode can also be used to update only the ImageLayout. In that case, the MBLayout is kept as is.
    //
    // Note: The new row dimension must be a straight multiple or divisor of the current row dimension.
    // To reshape to a non-multiple go to row dim 1 first.
    //
    // Unlike most other nodes, this node has intimate inside knowlegde of MBLayouts and frameRanges.
    // TODO: Changing the ImageLayout does not seem to belong here.
    // -----------------------------------------------------------------------

    template<class ElemType>
    class ReshapeNode : public ReinterpretNodeBase<ElemType>
    {
        typedef ReinterpretNodeBase<ElemType> Base; UsingReinterpretNodeBaseMembers;
        static const std::wstring TypeName() { return L"Reshape"; }
    public:
        ReshapeNode(DEVICEID_TYPE deviceId, const wstring & name, size_t numRows = 0, const ImageLayout & imageLayout = ImageLayoutWHC(0,0,0)) :
            Base(deviceId, name),
            m_numTargetRows(numRows),
            m_targetImageLayout(imageLayout)
        { }
        ReshapeNode(const ScriptableObjects::IConfigRecordPtr configp) :
            ReshapeNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"numRows"), ImageLayoutWHC(configp->Get(L"imageWidth"), configp->Get(L"imageHeight"), configp->Get(L"imageChannels")))
        {
            AttachInputs(configp, this->GetExpectedNumInputs());
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<ReshapeNode<ElemType>>(nodeP);
                node->m_numTargetRows = m_numTargetRows;
                node->m_targetImageLayout = m_targetImageLayout;
            }
        }

        virtual void SaveToFile(File& fstream) const override
        {
            Base::SaveToFile(fstream);
            fstream << m_numTargetRows;
            m_targetImageLayout.SaveToFile(fstream);
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_numTargetRows;
            m_targetImageLayout.LoadFromFile(fstream);
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, true);
            InferImageDimensions();

            if (m_targetImageLayout.GetWidth() == 0 || m_targetImageLayout.GetHeight() == 0 || m_targetImageLayout.GetNumChannels() == 0)
            {
                m_imageLayout = ImageLayoutWHC(1, 1, m_numTargetRows);
                if (m_inputImageLayout.GetWidth() * m_inputImageLayout.GetNumChannels() != 1)
                    fprintf(stderr, "WARNING: Reshape operation cannot inherit image size information from its child. Image size info is lost.\n");
            }
            else
            {
                m_imageLayout = m_targetImageLayout;
            }
        }

        virtual void /*IComputationNode::*/PrintSelfBeforeValidation() const override
        {
            fprintf(stderr, "\nValidating --> %ls = %ls", NodeName().c_str(), OperationName().c_str());
            fprintf(stderr, "(");
            for (size_t i = 0; i < ChildrenSize(); i++)
            {
                ComputationNodePtr child = Inputs(i);
                if (i > 0)
                    fprintf(stderr, ", ");
                if (!child)
                    fprintf(stderr, "NULL");
                else
                    fprintf(stderr, "%ls[%lu, %lu]", child->NodeName().c_str(), child->GetNumRows(), child->GetNumCols());
            }
            fprintf(stderr, ", NumOfRows=%lu, imageWidth=%lu, imageHeight=%lu, imageChannels=%lu)", m_numTargetRows, m_targetImageLayout.GetWidth(), m_targetImageLayout.GetHeight(), m_targetImageLayout.GetNumChannels());
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            size_t rows = Inputs(0)->GetNumRows(), cols = Inputs(0)->GetNumCols();
            // Note: During initial validation, cols may not be a multiple. E.g. cols may be 1 or 3. So we cannot check here whether the integer-multiple conditions are fulfilled.
            size_t newCols = cols * rows / m_numTargetRows;
            if (isFinalValidationPass)
            {
                if ((m_numTargetRows > rows && m_numTargetRows % rows != 0) ||  // grouping columns
                    (m_numTargetRows < rows && rows % m_numTargetRows != 0))    // splitting columns
                    InvalidArgument("%ls %ls operation: output row dimension %d is not an integer multiple or divisor of input dimension %d", NodeName().c_str(), OperationName().c_str(), (int)m_numTargetRows, (int)rows);
                if (!m_pMBLayout && rows * cols != m_numTargetRows * newCols)    // sadly, cannot verify here if we have a layout, since current #cols may be bogus
                    LogicError("%ls %ls operation: unexpected dimension mismatch", NodeName().c_str(), OperationName().c_str());
            }

            SetDims(m_numTargetRows, newCols);
            if (factor() == 1)          // canonical case: no reshaping actually (e.g. only changing the ImageLayout)
                m_pMBLayout = Inputs(0)->GetMBLayout();
            else if (Inputs(0)->HasMBLayout())
            {
                if (!m_pMBLayout)
                    m_pMBLayout = make_shared<MBLayout>();  // mini-batch data: this generates its own layout
            }
            else
                assert(!m_pMBLayout);                       // reshaping non-mini-batch data
            InferImageDimsFromInputs();
        }

        virtual void UpdateFunctionMBSize() override
        {
            size_t rows = Inputs(0)->GetNumRows(), cols = Inputs(0)->GetNumCols();
            size_t newCols = cols * rows / m_numTargetRows;
            if (!m_pMBLayout)
            {
#if 0
                VerifyDims(m_numTargetRows, newCols);
#endif
            }
            else
                SetDims(m_numTargetRows, newCols);
        }

        // TODO: there seems to be semantic overlap between OnEvaluateBeginIteration() and UpdateFunctionMBSize()
        virtual void /*IComputationNode::*/OnEvaluateBeginIteration() override
        {
            Base::OnEvaluateBeginIteration();
            // create the derived layout
            if (m_pMBLayout && factor() != 1)
            {
                // BUGBUG: This assumes that the layout is complete at this point in time (RecurrentNodeBase makes the same assumption).
                //         This assumption is correct at present, but will becomes invalid once we go sequence-to-sequence.
                m_pMBLayout->Init(Inputs(0)->GetNumParallelSequences(), Inputs(0)->GetNumTimeSteps() * Inputs(0)->GetNumRows() / m_numTargetRows);
                if (weStack())
                {
                    // going from many samples to one: layout entry will get no flags
                    if (m_pMBLayout->GetNumTimeSteps() != 1)
                        LogicError("ReshapeNode::OnEvaluateBeginIteration() faking to remove a nested time dimension only works when going back to a single frame per sequence.");
                    // leave flags empty (single-frame 'utterances' come form frame randomization, hence no flags)
                }
                else
                {
                    // going from one sample to many: layout will get SentenceStart/SentenceEnd flags for the sequence we expand into
                    if (Inputs(0)->GetMBLayout()->GetNumTimeSteps() != 1)
                        LogicError("ReshapeNode::OnEvaluateBeginIteration() faking to add a nested time dimension only works when coming from a single frame per sequence.");
                    for (size_t s = 0; s < m_pMBLayout->GetNumParallelSequences(); s++)
                        m_pMBLayout->SetAsSentence(s, 0, m_pMBLayout->GetNumTimeSteps());
                }
            }
        }

        // notes:
        //  - input and output have different time base and different layouts (unless the canonical case of factor() == 1)
        //  - frameRange refers to *functionValues*, not the inputs
        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            size_t rows = Inputs(0)->GetNumRows(), cols = Inputs(0)->GetNumCols();
            size_t newCols = cols * rows / m_numTargetRows;
            assert(newCols * m_numTargetRows == cols * rows); // follows from above check
            VerifyDims(m_numTargetRows, newCols);

            // no layout case: this is indeed just a reshape. Same for canonical case
            // (We still need to copy the values since there is currently no way to point to an input function value while reshaping at the same time.)
            if (!m_pMBLayout || factor() == 1)
            {
                FunctionValues().Reshaped(newCols * m_numTargetRows, 1).SetValue(Inputs(0)->FunctionValues().Reshaped(cols * rows, 1));   // copy the values as one long vector
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
                    Base::Unstack(frameRange.WithLayout(Inputs(0)->GetMBLayout()), Inputs(0)->GetMBLayout(), Inputs(0)->FunctionValues(), FunctionValues(), factor(), false/*addTo*/);
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange & frameRange) override
        {
            size_t rows = Inputs(0)->GetNumRows(), cols = Inputs(0)->GetNumCols();
            size_t newCols = cols * rows / m_numTargetRows;

            // no layout case: this is indeed just a reshape. Same for canonical case
            if (!m_pMBLayout || factor() == 1)
            {
                Inputs(0)->GradientValues().Reshaped(cols * rows, 1) += GradientValues().Reshaped(newCols * m_numTargetRows, 1);   // treat the values as one long vector
            }
            // layout case: reshape semantics happens across parallel seqeunces, i.e. requiring data shuffling
            else
            {
                if (weStack())
                    Base::Unstack(frameRange, m_pMBLayout, GradientValues(), Inputs(0)->GradientValues(), factor(), true/*addTo*/);
                else
                    Base::Stack(frameRange.WithLayout(Inputs(0)->GetMBLayout()), Inputs(0)->GetMBLayout(), GradientValues(), Inputs(0)->GradientValues(), factor(), true/*addTo*/);
            }
        }

    private:
        size_t m_numTargetRows;
        bool weStack() const { return m_numTargetRows > Inputs(0)->GetNumRows(); }        // do we stack (multiple frames into one)
        size_t factor() const { return m_numTargetRows > Inputs(0)->GetNumRows() ? m_numTargetRows / Inputs(0)->GetNumRows() : Inputs(0)->GetNumRows() / m_numTargetRows; }   // factor by which we stack or unstack
        ImageLayout m_targetImageLayout;

        void InferImageDimensions()
        {
            if (m_targetImageLayout.GetWidth() > 0)
            {
                if (m_targetImageLayout.GetHeight() > 0)
                {
                    if (m_targetImageLayout.GetNumChannels() > 0)
                    {
                        if (m_targetImageLayout.GetNumElements() != m_numTargetRows)
                            RuntimeError("Image dimensions do not match row size.");
                    }
                    else
                    {
                        if (m_numTargetRows % (m_targetImageLayout.GetWidth() * m_targetImageLayout.GetHeight()) > 0)
                            RuntimeError("Image row size is not a multiple of specified image dimensions.");
                        else
                            m_targetImageLayout = ImageLayoutWHC(m_targetImageLayout.GetWidth(), m_targetImageLayout.GetHeight(), m_numTargetRows / (m_targetImageLayout.GetWidth() * m_targetImageLayout.GetHeight()));
                    }
                }
                else
                {
                    if (m_targetImageLayout.GetNumChannels() > 0)
                    {
                        if (m_numTargetRows % (m_targetImageLayout.GetWidth() * m_targetImageLayout.GetNumChannels()) > 0)
                            RuntimeError("Image row size is not a multiple of specified image dimensions.");
                        else
                            m_targetImageLayout = ImageLayoutWHC(m_targetImageLayout.GetWidth(), m_numTargetRows / (m_targetImageLayout.GetWidth() * m_targetImageLayout.GetNumChannels()), m_targetImageLayout.GetNumChannels());
                    }
                    else
                    {
                        RuntimeError("At least two image dimensions must be specified.");
                    }
                }
            }
            else
            {
                if (m_targetImageLayout.GetHeight() > 0)
                {
                    if (m_targetImageLayout.GetNumChannels() > 0)
                    {
                        if (m_numTargetRows % (m_targetImageLayout.GetHeight() * m_targetImageLayout.GetNumChannels()) > 0)
                            RuntimeError("Image row size is not a multiple of specified image dimensions.");
                        else
                            m_targetImageLayout = ImageLayoutWHC(m_numTargetRows / (m_targetImageLayout.GetHeight() * m_targetImageLayout.GetNumChannels()), m_targetImageLayout.GetHeight(), m_targetImageLayout.GetNumChannels());
                    }
                    else
                        RuntimeError("At least two image dimensions must be specified.");
                }
                else if (m_targetImageLayout.GetNumChannels() > 0)
                    RuntimeError("At least two image dimensions must be specified.");
            }
        }
    };

    template class ReshapeNode<float>;
    template class ReshapeNode<double>;

    // -----------------------------------------------------------------------
    // ReconcileMBLayout (dataInput, layoutInput)
    // This node copies data from 'dataInput' while it propagates the minibatch-layout information from 'layoutInput'.
    // It does perform a runtime check to enforce that the layout of 'dataInput' is compatible (identical content) to that of 'layoutInput'.
    // This node is meant to be used from BrainScript macros that bracket expand/reduce pairs of nodes. It is not meant to really be used directly.
    // TODO: What to do with sequence-boundary flags?
    // -----------------------------------------------------------------------

    template<class ElemType>
    class ReconcileMBLayoutNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"ReconcileMBLayout"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(ReconcileMBLayoutNode);
        ReconcileMBLayoutNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange & frameRange) override
        {
            Inputs(0)->GradientSlice(frameRange.WithLayout(Inputs(0)->GetMBLayout())) += GradientSlice(frameRange);
            // TODO: Once we do in-place, the above must include a copy-to-self check (pay special attention to adding vs. copying).
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            // enforce compatibility of 'dataInput' with 'layoutInput'
            // TODO: how to deal with boundary flags?
            if (*m_pMBLayout != *Inputs(0)->GetMBLayout())   // this does a deep value-level comparison
                InvalidArgument("%ls %ls operation discovered that %ls %ls operation produced an MB layout that is incompaitble with that of %ls %ls.",
                                NodeName().c_str(), OperationName().c_str(),
                                Inputs(0)->NodeName().c_str(), Inputs(0)->OperationName().c_str(),
                                Inputs(1)->NodeName().c_str(), Inputs(1)->OperationName().c_str());

            // copy the data from 'dataInput'
            ValueSlice(frameRange).SetValue(Inputs(0)->ValueSlice(frameRange.WithLayout(Inputs(0)->GetMBLayout())));  // just propagate through
            // TODO: Once we do in-place, the above must include a copy-to-self check (either here or inside the matrix lib).
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            if (isFinalValidationPass && (!Inputs(0)->HasMBLayout() || !Inputs(1)->HasMBLayout()))
                RuntimeError("%ls %ls operation requires two inputs that both have an associated MB layout.");
            m_pMBLayout = Inputs(1)->GetMBLayout(); // output layout is that of 'layoutInput'
            // Note: We could also enforce that both inputs in fact have different layouts. But maybe there are edge cases where it isn't. Then this just becomes a nop. Also OK.

            SetDims(Inputs(0));
            InferImageDimsFromInputs();
        }
    };

    template class ReconcileMBLayoutNode<float>; 
    template class ReconcileMBLayoutNode<double>;

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
            m_sliceHeight(numRows)
        { }
        RowSliceNode(const ScriptableObjects::IConfigRecordPtr configp) :
            RowSliceNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"startIndex"), configp->Get(L"numRows"))
        {
            AttachInputs(configp, this->GetExpectedNumInputs());
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            auto node = dynamic_pointer_cast<RowSliceNode<ElemType>>(nodeP);

            node->m_startIndex = m_startIndex;
            node->m_sliceHeight = m_sliceHeight;
        }

        virtual void SaveToFile(File& fstream) const override
        {
            Base::SaveToFile(fstream);
            fstream << m_startIndex << m_sliceHeight;
        }
        
        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_startIndex >> m_sliceHeight;
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange & frameRange) override
        {
            Inputs(0)->GradientSlice(frameRange).AddToRowSliceValuesOf(GradientSlice(frameRange), m_startIndex, m_sliceHeight);
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            ValueSlice(frameRange).AssignRowSliceValuesOf(Inputs(0)->ValueSlice(frameRange), m_startIndex, m_sliceHeight);
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            if (isFinalValidationPass && Inputs(0)->GetNumRows() < m_startIndex + m_sliceHeight)
                RuntimeError("RowSlice operation: m_startIndex + m_sliceHeight exceeds number of rows in the input.");

            SetDims(m_sliceHeight, Inputs(0)->GetNumCols());
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, true);
            m_imageLayout = ImageLayoutWHC(m_imageLayout.GetWidth(), m_sliceHeight, m_imageLayout.GetNumChannels());

            // warn that this node will destroy the image size information from the child
            if (m_inputImageLayout.GetWidth() * m_inputImageLayout.GetNumChannels() != 1)
                fprintf(stderr, "WARNING: RowSlice operation cannot inherit image size information from its child. Image size info is lost.\n");
        }

    private:
        size_t m_startIndex, m_sliceHeight;
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
        DeclareConstructorFromConfig(RowStackNode);
        RowStackNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
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

            SetDims(totalRows, numCols);
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, true);
            m_imageLayout = ImageLayoutWHC(m_imageLayout.GetWidth(), GetNumRows(), m_imageLayout.GetNumChannels());

            // warn that this node will destroy the image size information from the child
            if (m_inputImageLayout.GetWidth() * m_inputImageLayout.GetNumChannels() != 1)
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
        RowRepeatNode(const ScriptableObjects::IConfigRecordPtr configp) :
            RowRepeatNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"numRepeats"))
        {
            AttachInputs(configp, this->GetExpectedNumInputs());
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
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
            m_imageLayout = ImageLayoutWHC(m_imageLayout.GetWidth(), m_inputImageLayout.GetHeight() * m_numRepeat, m_imageLayout.GetNumChannels());

            // watn that this node will destroy the image size information from the child
            if (m_inputImageLayout.GetWidth() * m_inputImageLayout.GetNumChannels() != 1)
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

            SetDims(Inputs(0)->GetNumRows() * m_numRepeat, Inputs(0)->GetNumCols());
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            //if (!isNoop())    // if m_numRepeat == 1 then virtual FunctionValues() will return the child   --TODO: do this as an in-place optimization instead
            ValueSlice(frameRange).AssignRepeatOf(Inputs(0)->ValueSlice(frameRange), m_numRepeat, 1);
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange & frameRange) override
        {
            Inputs(0)->GradientSlice(frameRange).AddToRowRepeatValuesOf(GradientSlice(frameRange), m_numRepeat);
        }

    private:
        //bool isNoop() const { return m_numRepeat == 1; }    // in this case this node does nothing
        size_t m_numRepeat;
    };

    template class RowRepeatNode<float>;
    template class RowRepeatNode<double>;

}}}
