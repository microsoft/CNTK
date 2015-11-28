//
// <copyright file="CompositeComputationNodes.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

//The basic idea of this implementation is learned from Brian Guenter <bguenter@microsoft.com>

#include "ComputationNode.h"
#include "TrainingCriterionNodes.h"

#include <map>
#include <string>
#include <stdexcept>
#include <list>
#include <iostream> 

//this file will contain computation nodes that require several atomic computation.
//composite nodes can save memory, computation, or both
namespace Microsoft { namespace MSR { namespace CNTK {

    // -----------------------------------------------------------------------
    // ParallelNode (input0, input1)
    // -----------------------------------------------------------------------

    /**
    parallel node to join two streams into one 
    
    join parallel children node, avoids any operations except putting outputs from children to corresponding columns
    input(0) : [nDim0 X T]
    input(1) : [nDim1 X T]
    output   : [[nDim0 + nDim1] X T]
    */
    template<class ElemType>
    class ParallelNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>, public NumInputs<2>
    {
        typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"Parallel"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(ParallelNode);
        ParallelNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void ComputeInputPartialNonLooping(size_t inputIndex) override
        {
            if (inputIndex > 1)
                InvalidArgument("Parallel operation only takes two input.");
            ComputationNodePtr child = Inputs(inputIndex);
            size_t startidx = (inputIndex == 0) ? 0 : Inputs(0)->GetNumRows();
            size_t nrows = child->GetNumRows();

            if (child->GradientValues().GetNumRows() != child->GetNumRows() || child->GradientValues().GetNumCols() != GetNumCols())
            {
                child->GradientValues().Resize(child->GetNumRows(), child->GetNumCols());
                child->GradientValues().SetValue(0);
            }

            Matrix<ElemType> tmpMat(m_deviceId);
            tmpMat.AssignRowSliceValuesOf(GradientValues(), startidx, nrows);

            ComputeInputPartialS(tmpMat, child->GradientValues());
        }

        /*TODO: merge with call site*/void ComputeInputPartialS(Matrix<ElemType>& gradientValues, Matrix<ElemType>& inputGradientValues)
        {
            inputGradientValues += gradientValues;
        }

        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        /*TODO: merge with call site*/void EvaluateThisNodeS(Matrix<ElemType>& functionValues, Matrix<ElemType>& inputFunctionValues0, Matrix<ElemType>& inputFunctionValues1)
        {
            size_t rows0 = inputFunctionValues0.GetNumRows(), cols0 = inputFunctionValues0.GetNumCols();
            size_t rows1 = inputFunctionValues1.GetNumRows(), cols1 = inputFunctionValues1.GetNumCols();

            if (cols0 != cols1)
                LogicError("ParallelNode: column dimension mismatched!");

            functionValues.Resize(rows0 + rows1, cols0);
            functionValues.SetValue(0);

            functionValues.AssignToRowSliceValuesOf(inputFunctionValues0, 0, rows0);
            functionValues.AssignToRowSliceValuesOf(inputFunctionValues1, rows0, rows1);
        }

        /// input(0) : [nDim1 X T]
        /// input(1) : [nDim2 X T]
        /// output   : [[nDim1 + nDim2] X T]
        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            size_t rows1, cols1;
            rows1 = Inputs(1)->GetNumRows();
            cols1 = Inputs(1)->GetNumCols();

            size_t rows0, cols0;
            rows0 = Inputs(0)->GetNumRows();
            cols0 = Inputs(0)->GetNumCols();

            if (isFinalValidationPass && cols0 != cols1)
                LogicError("ParallelNode: column dimension mismatched!");

            size_t rows = rows0 + rows1;
            size_t cols = cols0;
            SetDims(rows, cols);

            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInput(0);
        }

    public:
        virtual bool UnitTest() {
            size_t nT = 3;
            size_t nInput0 = 3;
            size_t nInput1 = 3;

            Matrix<ElemType> f0(m_deviceId), func(m_deviceId), f1(m_deviceId);

            f0 = Inputs(0)->FunctionValues();
            f1 = Inputs(1)->FunctionValues();
            func = FunctionValues();

            Inputs(0)->SetDims(nInput0, nT);
            Inputs(0)->UpdateFunctionValuesSize();
            Inputs(0)->FunctionValues().SetValue(0);
            Inputs(0)->FunctionValues()(0, 0) = 1;
            Inputs(0)->FunctionValues()(0, 1) = 2;
            Inputs(0)->FunctionValues()(0, 2) = 3;

            Inputs(1)->SetDims(nInput1, nT);
            Inputs(1)->UpdateFunctionValuesSize();
            Inputs(1)->FunctionValues().SetValue(0);
            Inputs(1)->FunctionValues()(0, 0) = 4;
            Inputs(1)->FunctionValues()(0, 1) = 5;
            Inputs(1)->FunctionValues()(0, 2) = 6;
            SetDims(nInput0 + nInput1, nT);
            UpdateFunctionValuesSize();

            EvaluateThisNode(FrameRange(m_pMBLayout));

            /// check with expected values
            if (!ISCLOSE(FunctionValues()(0, 0), 1, EPSILON) ||
                !ISCLOSE(FunctionValues()(0, 1), 2, EPSILON) ||
                !ISCLOSE(FunctionValues()(0, 2), 3, EPSILON) ||
                !ISCLOSE(FunctionValues()(3, 0), 4, EPSILON) ||
                !ISCLOSE(FunctionValues()(3, 1), 5, EPSILON) ||
                !ISCLOSE(FunctionValues()(3, 2), 6, EPSILON))
                return false;
            FunctionValues().TransferToDeviceIfNotThere(m_deviceId, true);

            GradientValues().Resize(nInput0 + nInput1, nT);
            GradientValues().SetValue(0);
            Inputs(0)->GradientValues().Resize(nInput0, nT);
            Inputs(1)->GradientValues().Resize(nInput1, nT);
            Inputs(0)->GradientValues().SetValue(0);
            Inputs(1)->GradientValues().SetValue(0);
            GradientValues()(0, 0) = 1;
            GradientValues()(0, 1) = 2;
            GradientValues()(0, 2) = 3;
            GradientValues()(3, 0) = 4;
            GradientValues()(3, 1) = 5;
            GradientValues()(3, 2) = 6;

            ComputeInputPartial(0, FrameRange(m_pMBLayout));
            ComputeInputPartial(1, FrameRange(m_pMBLayout));

            /// check with expected values
            if (!ISCLOSE(Inputs(0)->GradientValues()(0, 0), 1, EPSILON)
                || !ISCLOSE(Inputs(0)->GradientValues()(0, 1), 2, EPSILON)
                || !ISCLOSE(Inputs(0)->GradientValues()(0, 2), 3, EPSILON)
                || !ISCLOSE(Inputs(1)->GradientValues()(0, 0), 4, EPSILON)
                || !ISCLOSE(Inputs(1)->GradientValues()(0, 1), 5, EPSILON)
                || !ISCLOSE(Inputs(1)->GradientValues()(0, 2), 6, EPSILON))
                return false;

            Inputs(0)->GradientValues().TransferToDeviceIfNotThere( m_deviceId, true);
            Inputs(1)->GradientValues().TransferToDeviceIfNotThere( m_deviceId, true);

            return true;
        }

    };

    template class ParallelNode<float>;
    template class ParallelNode<double>;

    // -----------------------------------------------------------------------
    // PreComputedNode
    // -----------------------------------------------------------------------

    //this is a noninstantiable virtual class, all nodes require precomputation should derive from it
    template<class ElemType>
    class PreComputedNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>
    {
        typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembers; using Base::OperationName;
    public:
        //virtual ComputationNodeBase * NewThis(DEVICEID_TYPE deviceId, const wstring & name) = 0;
        //DeclareConstructorFromConfigWithNumInputs(PreComputedNode);
        PreComputedNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name),
            m_hasComputed(false)
        { }

        // interface through which this node is operated on are these two functions

        // check whether node has already undergone precomputation
        virtual bool HasComputed() const { return m_hasComputed; }

        // call this with 'false' at start and with 'true' at end
        // This is used for resetting and updating from accumulators.
        virtual void MarkComputed(const bool hasComputed)
        {
            m_hasComputed = hasComputed;
            CreateMatrixIfNull(m_functionValues);
        }

        virtual bool RequiresPreCompute() const override { return true; }

        virtual void SaveToFile(File& fstream) const override
        {
            Base::SaveToFile(fstream);
            fstream << m_hasComputed;
            fstream << FunctionValues();   // TODO: why serialize if not yet computed?
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_hasComputed;
            LoadFunctionValues(fstream);
         }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const override
        {
            Base::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "[%lu,%lu]  ", GetNumRows(), GetNumCols());
            fstream << string(str);
            sprintf(str, "HasComputed=%ls", HasComputed()? L"true" : L"false");
            fstream << string(str);

            PrintNodeValuesToFile(printValues, fstream);
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            if (!Inputs(0)->HasMBLayout())
                InvalidArgument("%ls %ls operation requires its input to come in minibatches of samples.", NodeName().c_str(), OperationName().c_str());
            m_pMBLayout = nullptr;    // this node does not hold mini-batch data
            if (!m_hasComputed) // this node retains state, and state gets destroyed by Resize(), so we must be careful
                SetDims(Inputs(0)->GetNumRows(), 1);
            else
                VerifyDims(Inputs(0)->GetNumRows(), 1);
            InferImageDimsFromInputs();
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<PreComputedNode<ElemType>>(nodeP);
                node->m_hasComputed = m_hasComputed;
            }
        }

        // this is for the special case: convertDBN needs this; because we initialize values directly from another well-trained model
        virtual void SideLoadFromMatrix(const Matrix<ElemType>& value)
        {
            CreateMatrixIfNull(m_functionValues); 
            m_functionValues->SetValue(value);
            m_hasComputed = true; 
        }
    public:
        bool m_hasComputed;
    };

#define UsingPreComputedNodeMembers UsingComputationNodeMembers; using Base::m_hasComputed; using Base::OperationName

    // -----------------------------------------------------------------------
    // MeanInvStdDevNodeBase (features)  -- common base class for Mean and InvStdDev
    // -----------------------------------------------------------------------

    template<class ElemType>
    class MeanInvStdDevNodeBase : public PreComputedNode<ElemType>, public NumInputs<1>
    {
        typedef PreComputedNode<ElemType> Base; UsingPreComputedNodeMembers;
        //static const std::wstring TypeName() { return L"MeanInvStdDev (base)"; }
    public:
        //DeclareConstructorFromConfigWithNumInputs(MeanInvStdDevNodeBase);
        MeanInvStdDevNodeBase(DEVICEID_TYPE deviceId, const wstring & name) :
            PreComputedNode<ElemType>(deviceId, name),
            m_numSamples(SIZE_MAX)
        { }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);
            m_numSamples = SIZE_MAX;
        }
    
        // this is used by convertDBN
        virtual void SideLoadFromMatrix(const Matrix<ElemType>& m)
        {
            Base::SideLoadFromMatrix(m);
            m_numSamples = SIZE_MAX;
        }

        virtual void /*PreComputedNode::*/MarkComputed(const bool hasComputed , size_t numSamples = 0)
        {
            Base::MarkComputed(hasComputed);
            if (!m_hasComputed)     // initialize
            {
                if (IsAccumulating())
                    LogicError("%ls %ls operation: MarkComputed(false) has been called while accumulating.", NodeName().c_str(), OperationName().c_str());
                m_numSamples = 0;
            }
            else                    // finalize
            {
                if (!IsAccumulating())
                    LogicError("%ls %ls operation: MarkComputed(true) has been called without MarkComputed(false) first.", NodeName().c_str(), OperationName().c_str());
                if (m_numSamples == 0)
                    LogicError("%ls %ls operation: No data accumulated during precomputation.", NodeName().c_str(), OperationName().c_str());
                m_numSamples = SIZE_MAX;
            }
        }

        virtual void ComputeInputPartialNonLooping(size_t /*inputIndex*/) override
        {
            //LogicError("Mean operation should not be involved in the gradient calculation.");
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                if (m_numSamples != SIZE_MAX)
                    LogicError("%ls %ls operation: CopyTo() called while accumulating.", NodeName().c_str(), OperationName().c_str());
                auto node = dynamic_pointer_cast<MeanInvStdDevNodeBase<ElemType>>(nodeP);
                node->m_numSamples = SIZE_MAX;
            }
        }
    protected:
        size_t m_numSamples;    // (SIZE_MAX while outside accumulation state)
        bool IsAccumulating() const { return m_numSamples != SIZE_MAX; }
    };

#define UsingMeanInvStdDevNodeBaseNodeMembers ComputationNodeBoilerplate; UsingPreComputedNodeMembers; using Base::m_numSamples; using Base::IsAccumulating

    // -----------------------------------------------------------------------
    // MeanNode (features)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class MeanNode : public MeanInvStdDevNodeBase<ElemType>
    {
        typedef MeanInvStdDevNodeBase<ElemType> Base; UsingMeanInvStdDevNodeBaseNodeMembers;
        static const std::wstring TypeName() { return L"Mean"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(MeanNode);
        MeanNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        MeanNode(DEVICEID_TYPE deviceId, const wstring & name, size_t) : Base(deviceId, name)
        {

        }
        virtual void /*PreComputedNode::*/MarkComputed(const bool hasComputed)
        {
            Base::MarkComputed(hasComputed);
            if (!m_hasComputed)     // initialize accumulation
            {
                UpdateFunctionValuesSize();
                FunctionValues().SetValue(0);
            }
            // no else branch because EvaluateThisNodeNonLooping() already leaves a valid mean in m_functionValues
        }

        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            FrameRange frameRange(Inputs(0)->GetMBLayout());
            if (m_hasComputed)
                return;     // not accumulating

            if (!IsAccumulating())
                LogicError("%ls %ls operation: MarkComputed(false) has not been called.", NodeName().c_str(), OperationName().c_str());

            // set gaps to zero, since we are reducing in time
            Inputs(0)->MaskMissingValuesColumnsToZero(frameRange);

            auto & samples = Inputs(0)->FunctionValues();
            auto & avg = FunctionValues();

#if 1//NANCHECK
            samples.HasNan("Mean-Samples");
#endif
            size_t numNewSamples = Inputs(0)->GetMBLayout()->DetermineActualNumSamples();
            size_t totalNumSamples = m_numSamples + numNewSamples;
            if (totalNumSamples == 0) totalNumSamples = 1;  // 0/0=1 in this context
            Matrix<ElemType>::MultiplyAndWeightedAdd(1.0f / totalNumSamples, samples, false,
                                                     ConstOnes(samples.GetNumCols(), 1, samples.GetDeviceId()),
                                                     false, (ElemType)m_numSamples / totalNumSamples, avg);
#if 1//NANCHECK
            avg.HasNan("Mean-avg");
#endif

            m_numSamples += numNewSamples;
        }
    };

    template class MeanNode<float>;
    template class MeanNode<double>;

    // -----------------------------------------------------------------------
    // InvStdDevNode (features)
    // TODO: share stuff with MeanNode
    // -----------------------------------------------------------------------

    template<class ElemType>
    class InvStdDevNode : public MeanInvStdDevNodeBase<ElemType>
    {
        typedef MeanInvStdDevNodeBase<ElemType> Base; UsingMeanInvStdDevNodeBaseNodeMembers;
        static const std::wstring TypeName() { return L"InvStdDev"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(InvStdDevNode);
        InvStdDevNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name),
            m_mean(deviceId), m_var(deviceId), m_temp(deviceId)
        { }

        virtual void /*PreComputedNode::*/MarkComputed(const bool hasComputed) override
        {
            Base::MarkComputed(hasComputed);

            if (!m_hasComputed) // initialize
            {
                // reset accumulators
                size_t inputDim = Inputs(0)->GetNumRows();
                m_mean.Resize(inputDim, 1);
                m_var.Resize(inputDim, 1);
                m_mean.SetValue(0);
                m_var.SetValue(0);
                UpdateFunctionValuesSize();
                FunctionValues().SetValue(0);   // also set this because not doing it may flag during debugging; avoids special-casing this
            }
            else                // finalize
            {
                ElemType sqrtFloor = 1e-10f;
                m_var.InplaceTruncateBottom(sqrtFloor);     // prevent too small variance (and negative square roots due to numeric inaccuracy)
#if 1//NANCHECK
                m_var.HasNan("MarkComputed-InplaceTruncateBottom");
#endif
                m_var.InplaceSqrt();

#if 1//NANCHECK
                m_var.HasNan("MarkComputed-InplaceSqrt");
#endif
                m_var.ElementInverse();

#if NANCHECK
                m_var.HasNan("MarkComputed-ElementInverse()");
#endif
                FunctionValues().SetValue(m_var);
            }
        }

        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            FrameRange frameRange(Inputs(0)->GetMBLayout());
            if (m_hasComputed)
                return;     // not accumulating

            if (!IsAccumulating())
                LogicError("%ls %ls operation: MarkComputed(false) has not been called.", NodeName().c_str(), OperationName().c_str());

            // set gaps to zero, since we are reducing in time
            Inputs(0)->MaskMissingValuesColumnsToZero(frameRange);

            auto & samples = Inputs(0)->FunctionValues();
#if 1//NANCHECK
            samples.HasNan("InvStdDev-Samples");
#endif
            m_temp.SetValue(m_mean);
            size_t numNewSamples = Inputs(0)->GetMBLayout()->DetermineActualNumSamples();
            size_t totalNumSamples = m_numSamples + numNewSamples;
            if (totalNumSamples == 0) totalNumSamples = 1;  // 0/0=1 in this context
            Matrix<ElemType>::MultiplyAndWeightedAdd(1.0f / totalNumSamples, samples, false,
                                                     ConstOnes(samples.GetNumCols(), 1, samples.GetDeviceId()),
                                                     false, (ElemType)m_numSamples / totalNumSamples, m_mean);

            m_temp -= m_mean;
            m_temp.AssignElementPowerOf(m_temp, 2);
            m_var += m_temp;

            m_temp.AssignDifferenceOf(samples, m_mean);
            m_temp.AssignElementPowerOf(m_temp, 2);

            Matrix<ElemType>::MultiplyAndWeightedAdd(1.0f / totalNumSamples, m_temp, false,
                                                     ConstOnes(samples.GetNumCols(), 1, samples.GetDeviceId()),
                                                     false, (ElemType)m_numSamples / totalNumSamples, m_var);

#if 1//NANCHECK
            m_var.HasNan("InvStdDev-m_var");
#endif

            m_numSamples += samples.GetNumCols();
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<InvStdDevNode<ElemType>>(nodeP);
                node->m_mean = m_mean;
                node->m_var = m_var;
                node->m_temp =  m_temp;
            }
        }
    private:
        Matrix<ElemType> m_mean;
        Matrix<ElemType> m_var;
        Matrix<ElemType> m_temp;
    };

    template class InvStdDevNode<float>;
    template class InvStdDevNode<double>;

    // -----------------------------------------------------------------------
    // PerDimMeanVarNormalizationNode (feature, mean, invStdDev)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class PerDimMeanVarNormalizationNode : public ComputationNode<ElemType>, public NumInputs<3>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"PerDimMeanVarNormalization"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(PerDimMeanVarNormalizationNode);
        PerDimMeanVarNormalizationNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange &) override
        {
            InvalidArgument("PerDimMeanVarNormalizationNode should only be called in the evaluation stage.");
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            //only feature (input0) and output needs to be sliced
            Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues());
        }

        /*TODO: merge with call site*/void EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0,
                                             const Matrix<ElemType>& input1, const Matrix<ElemType>& input2)
        {
#if DUMPOUTPUT
            //input0.Print("PerDimMeanVarNormalization-input0");
            //input1.Print("PerDimMeanVarNormalization-input1");
            //input2.Print("PerDimMeanVarNormalization-input2");
#endif

#if NANCHECK
            input0.HasNan("PerDimMeanVarNormalization-input0");
            input1.HasNan("PerDimMeanVarNormalization-input1");
            input2.HasNan("PerDimMeanVarNormalization-input2");
#endif
            functionValues.AssignDifferenceOf(input0, input1);
            functionValues.ColumnElementMultiplyWith(input2);
#if NANCHECK
            functionValues.HasNan("PerDimMeanVarNormalization");
#endif
#if DUMPOUTPUT
            functionValues.Print("PerDimMeanVarNormalizationNode");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            if (Inputs(0)->RequiresPreCompute())
            {
                LogicError(
                    "PerDimMeanVarNormalizationNode criterion forbids first input from being a pre-compute node. "
                    "The first input should be the node whose output should be normalized, and the second and third inputs "
                    "should be LearnableParameter type or (Mean, InvStdDev) so that the values will be saved.");
            }

            if (!(Inputs(1)->OperationName() == OperationNameOf(LearnableParameter) &&
                  Inputs(2)->OperationName() == OperationNameOf(LearnableParameter)) &&
                !(Inputs(1)->OperationName() == OperationNameOf(MeanNode) &&
                  Inputs(2)->OperationName() == OperationNameOf(InvStdDevNode)))
            {
                LogicError(
                    "PerDimMeanVarNormalizationNode criterion requires the last two inputs to be LearnableParameter "
                    "type or (Mean, InvStdDev) so that the values will be saved.");
            }

            {
                size_t rows = (Inputs(1)->GetNumRows() == 0) ? Inputs(0)->GetNumRows() : Inputs(1)->GetNumRows();
                ValidateInferChildDims(1, rows, 1);
            }

            {
                size_t rows = (Inputs(2)->GetNumRows() == 0) ? Inputs(0)->GetNumRows() : Inputs(2)->GetNumRows();
                ValidateInferChildDims(2, rows, 1);
            }

            if (isFinalValidationPass)
            {
                //match rows
                if (!(Inputs(0)->GetNumRows() == Inputs(1)->GetNumRows() &&
                    Inputs(2)->GetNumRows() == Inputs(1)->GetNumRows()))
                {
                    LogicError("PerDimMeanVarNormalizationNode: All inputs should have same number of rows.");
                }

                if (!(Inputs(1)->GetNumCols() == 1 && Inputs(2)->GetNumCols() == 1))
                    LogicError("PerDimMeanVarNormalizationNode: Mean and InvStdDev should be a colum  vector.");
            }

            // TODO: Is this correct? Why not just skip propagating a gradient into these? We should not poke around in our children.
            Inputs(1)->SetParameterUpdateRequired(false);
            Inputs(2)->SetParameterUpdateRequired(false);  //prevent learning
            SetDims(Inputs(0));
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }
    };

    template class PerDimMeanVarNormalizationNode<float>;
    template class PerDimMeanVarNormalizationNode<double>;

    // -----------------------------------------------------------------------
    // PerDimMeanVarDeNormalizationNode (feature, mean, invStdDev)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class PerDimMeanVarDeNormalizationNode : public ComputationNode<ElemType>, public NumInputs<3>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"PerDimMeanVarDeNormalization"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(PerDimMeanVarDeNormalizationNode);
        PerDimMeanVarDeNormalizationNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange &) override
        {
            InvalidArgument("PerDimMeanVarDeNormalizationNode should only be called in the evaluation stage.");
        }

        //(feature-mean).*InvStdDev
        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            //only feature (input0) and output needs to be sliced
            Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues());
        }

        /*TODO: merge with call site*/void EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0,
                                             const Matrix<ElemType>& input1, const Matrix<ElemType>& input2)
        {
    #if DUMPOUTPUT
            //input0.Print("PerDimMeanVarDeNormalization-input0");
            //input1.Print("PerDimMeanVarDeNormalization-input1");
            //input2.Print("PerDimMeanVarDeNormalization-input2");
    #endif

    #if NANCHECK
            input0.HasNan("PerDimMeanVarDeNormalization-input0");
            input1.HasNan("PerDimMeanVarDeNormalization-input1");
            input2.HasNan("PerDimMeanVarDeNormalization-input2");
    #endif
            //functionValues.AssignDifferenceOf(input0, input1);
            //functionValues.ColumnElementMultiplyWith(input2);
            //functionValues.AssignDifferenceOf(input0, input0);
            //functionValues += input2;
            //functionValues.ElementInverse();
            //functionValues.ElementMultiplyWith(input0);
            functionValues.SetValue(input0);
            functionValues.ColumnElementDivideBy(input2);
            functionValues += input1;
    #if NANCHECK
            functionValues.HasNan("PerDimMeanVarDeNormalization");
    #endif
    #if DUMPOUTPUT
            functionValues.Print("PerDimMeanVarDeNormalizationNode");
    #endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            if (Inputs(0)->RequiresPreCompute())
            {
                LogicError(
                    "PerDimMeanVarDeNormalizationNode criterion forbids first input from being a pre-compute node. "
                    "The first input should be the node whose output should be de-normalized, and the second and third inputs "
                    "should be LearnableParameter type or (Mean, InvStdDev) so that the values will be saved.");
            }

            if (!(Inputs(1)->OperationName() == OperationNameOf(LearnableParameter) &&
                  Inputs(2)->OperationName() == OperationNameOf(LearnableParameter)) &&
                !(Inputs(1)->OperationName() == OperationNameOf(MeanNode) &&
                  Inputs(2)->OperationName() == OperationNameOf(InvStdDevNode)))
            {
                LogicError(
                    "PerDimMeanVarDeNormalizationNode criterion requires the last two inputs to be "
                    "LearnableParameter type or (Mean, InvStdDev) so that the values will be saved.");
            }

            {
                size_t rows = Inputs(1)->GetNumRows() == 0 ? Inputs(0)->GetNumRows() : Inputs(1)->GetNumRows();
                ValidateInferChildDims(1, rows, 1);
            }

            {
                size_t rows = Inputs(2)->GetNumRows() == 0? Inputs(0)->GetNumRows() : Inputs(2)->GetNumRows();
                ValidateInferChildDims(2, rows, 1);
            }

            if (isFinalValidationPass)
            {
                if (!(Inputs(0)->GetNumRows() == Inputs(1)->GetNumRows() &&  //match rows
                    Inputs(2)->GetNumRows() == Inputs(1)->GetNumRows()))
                {
                    LogicError("PerDimMeanVarDeNormalizationNode: All inputs should have same number of rows.");
                }

                if (!(Inputs(1)->GetNumCols() == 1 && Inputs(2)->GetNumCols() == 1))
                {
                    LogicError("PerDimMeanVarDeNormalizationNode: Mean and InvStdDev should be a colum  vector.");
                }
            }

            //prevent learning
            // TODO: Is this correct? Why not just skip propagating a gradient into these?
            Inputs(1)->SetParameterUpdateRequired(false);
            Inputs(2)->SetParameterUpdateRequired(false);

            SetDims(Inputs(0));
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }
    };

    template class PerDimMeanVarDeNormalizationNode<float>;
    template class PerDimMeanVarDeNormalizationNode<double>;

    // -----------------------------------------------------------------------
    // BatchModeNode
    // -----------------------------------------------------------------------

    /**
    BatchModeNode is a derivative of ComputationNode.
    It additionally check if needs to process data in batch before processing its parent
    This is used in case of beam search decoding. Batchmode node must be processed before other nodes.
    It differs from PreComputeNode in that precompute done is done before the entire corpus.
    This is done before forward computation of all nodes.
    */
    template<class ElemType>
    class BatchModeNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>
    {
        // all nodes require precomputation should derive from this class
        typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembers;
    public:
        //virtual ComputationNodeBase * NewThis(DEVICEID_TYPE deviceId, const wstring & name) = 0;
        //DeclareConstructorFromConfigWithNumInputs(BatchModeNode);
        BatchModeNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name),
            m_memory(deviceId)
        { }

        virtual bool HasComputed() const = 0;
        virtual void MarkComputed(const bool hasComputed) = 0;

        virtual void SaveToFile(File& fstream) const override
        {
            Base::SaveToFile(fstream);
            fstream << m_hasComputed;
            fstream << FunctionValues();
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_hasComputed;
            LoadFunctionValues(fstream);
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const override
        {
            Base::DumpNodeInfo(printValues, fstream);

            const size_t BUFLEN = 4096;
            WCHAR str[BUFLEN];
            swprintf(str, BUFLEN, L"[%lu,%lu]  ", GetNumRows(), GetNumCols());
            fstream << wstring(str);
            swprintf(str, BUFLEN, L"HasComputed=%ls", HasComputed() ? L"true" : L"false");
            fstream << wstring(str);

            PrintNodeValuesToFile(printValues, fstream);
        }

    protected:
        Matrix<ElemType> m_memory;   // the memory of input or output
        bool m_hasComputed;
    };

    // add this at the start of each derived class, to get access to the members of ComputationNode
    // See #define of 'UsingComputationNodeMembersBoilerplate' for more explanation.
    #define UsingBatchModeNodeMembers UsingComputationNodeMembersBoilerplate; \
        protected:  \
            using Base::m_memory; using Base::m_hasComputed; \
        public: \
            using Base::HasComputed; using Base::MarkComputed

    // -----------------------------------------------------------------------
    // TimeReverseNode (input)
    // BUGBUG: This must actually implement reversing the layout.
    // Challenge: This reverses the layout. If we time-reverse back, we'd reverse the layout again.
    // We will get the original layout. Unfortunately, it is not the same layout pointer.
    // To turn it back to the same layout pointer, insert a ReconcileMBLayout node.
    // -----------------------------------------------------------------------

    /**
    Developed by Kaisheng Yao.
    This node is used in the following work
    K. Yao and G. Zweig, "Sequence-to-Sequence Neural Net Models for Grapheme-to-Phoneme Conversion", submitted to INTERSPEECH 2015
    */
    template<class ElemType>
    class TimeReverseNode : public BatchModeNode<ElemType>, public NumInputs<1>
    {
        typedef BatchModeNode<ElemType> Base; UsingBatchModeNodeMembers;
        static const std::wstring TypeName() { return L"TimeReverse"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(TimeReverseNode);
        TimeReverseNode(DEVICEID_TYPE deviceId, const wstring & name) :
            BatchModeNode<ElemType>(deviceId, name)
        { }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<TimeReverseNode<ElemType>>(nodeP);
                // TODO: m_memory is never used inside this class, just assigned. Can it not be assigned?
                node->m_memory = m_memory;
            }
        }

        virtual bool HasComputed() const { return m_hasComputed; }
        virtual void MarkComputed(const bool hasComputed) { m_hasComputed = hasComputed; }

        virtual void ComputeInputPartialNonLooping(size_t inputIndex) override
        {
            assert(inputIndex == 0); inputIndex;
            VerifyDims(Inputs(0));

            size_t nT = GetNumTimeSteps();
            for (size_t t = 0; t < nT; t++)
            {
                Matrix<ElemType>  g =            GradientSlice(FrameRange(GetMBLayout(), t));
                Matrix<ElemType> ig = Inputs(0)->GradientSlice(FrameRange(Inputs(0)->GetMBLayout(), nT - 1 - t));
                ig += g;
            }
        }

        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            // BUGBUG: We must flip the layout, too.
            if (GetNumParallelSequences() != 1)
                LogicError("%ls %ls operation not implemented for multiple parallel sequences. It does not flip the layout either. I.e. only works for a single utterance.", NodeName().c_str(), OperationName().c_str());
            if (!m_hasComputed)
            {
                // this assumes this reverse node is called once, so it can set, instead add to, the function values
                SetDims(Inputs(0));
                UpdateFunctionValuesSize();

                size_t nT = GetNumTimeSteps();
                for (size_t t = 0; t < nT; t++)
                {
                    Matrix<ElemType> v = Inputs(0)->ValueSlice(FrameRange(Inputs(0)->GetMBLayout(), t));
                    ValueSlice(FrameRange(GetMBLayout(), nT - 1 - t)).SetValue(v);
                }

#if NANCHECK
                FunctionValues().HasNan("TimeReverse");
#endif
#if DUMPOUTPUT
                FunctionValues().Print("TimeReverseNode");
#endif

                m_memory.SetValue(FunctionValues());
            }
            // TODO: don't need to set m_hasCompute? Or what is it for?
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);
            InferMBLayoutFromInputsForStandardCase();
            if (isFinalValidationPass && !m_pMBLayout)
                RuntimeError("%ls %ls operation makes no sense without a MB layout.", NodeName().c_str(), OperationName().c_str());
            SetDims(Inputs(0));
            InferImageDimsFromInput(0);
        }

    public:
        bool UnitTest() {
            size_t nT = 3;
            size_t nInput = 3;
            size_t nOutput = nInput;

            /// backup
            Matrix<ElemType> f0(m_deviceId), func(m_deviceId);

            f0 = Inputs(0)->FunctionValues();
            func = FunctionValues();

            Inputs(0)->SetDims(nInput, nT);
            Inputs(0)->UpdateFunctionValuesSize();
            Inputs(0)->FunctionValues().SetValue(0);
            Inputs(0)->FunctionValues()(0, 0) = 1;
            Inputs(0)->FunctionValues()(0, 1) = 2;
            Inputs(0)->FunctionValues()(0, 2) = 3;
            SetDims(nOutput, nT);
            UpdateFunctionValuesSize();
            Inputs(0)->FunctionValues().TransferToDeviceIfNotThere( m_deviceId, true);
            EvaluateThisNode(FrameRange(m_pMBLayout));

            /// check with expected values
            if (!ISCLOSE(FunctionValues()(0, 0), 3, EPSILON) ||
                !ISCLOSE(FunctionValues()(0, 1), 2, EPSILON) ||
                !ISCLOSE(FunctionValues()(0, 2), 1, EPSILON))
            {
                return false;
            }

            FunctionValues().TransferToDeviceIfNotThere( m_deviceId, true);

            Inputs(0)->GradientValues().Resize(nOutput, nT);
            Inputs(0)->GradientValues().SetValue(1.0);
            GradientValues().Resize(nOutput, nT);
            GradientValues().SetValue(0);
            GradientValues()(0, 0) = 1;
            GradientValues()(0, 1) = 2;
            GradientValues()(0, 2) = 3;
            GradientValues().TransferToDeviceIfNotThere( m_deviceId, true);

            ComputeInputPartial(0, FrameRange(m_pMBLayout));

            /// check with expected values
            if (!ISCLOSE(Inputs(0)->GradientValues()(0, 0), 4, EPSILON) ||
                !ISCLOSE(Inputs(0)->GradientValues()(0, 1), 3, EPSILON) ||
                !ISCLOSE(Inputs(0)->GradientValues()(0, 2), 2, EPSILON))
            {
                return false;
            }

            Inputs(0)->GradientValues().TransferToDeviceIfNotThere(m_deviceId, true);
            GradientValues().TransferToDeviceIfNotThere(m_deviceId, true);

            return true;
        }
    };

    template class TimeReverseNode<float>;
    template class TimeReverseNode<double>;

}}}
