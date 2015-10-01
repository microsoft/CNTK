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
        ParallelNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void ComputeInputPartial(const size_t inputIndex)
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
            Resize(rows, cols);

            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInput(0);
        }

        //virtual void AttachInputs(const ComputationNodePtr c1, const ComputationNodePtr c2)
        //{
        //    m_children.resize(2);
        //    m_children[0] = c1;
        //    m_children[1] = c2;
        //}

    public:
        virtual bool UnitTest() {
            size_t nT = 3;
            size_t nInput0 = 3;
            size_t nInput1 = 3;

            Matrix<ElemType> f0(m_deviceId), func(m_deviceId), f1(m_deviceId);

            f0 = Inputs(0)->FunctionValues();
            f1 = Inputs(1)->FunctionValues();
            func = FunctionValues();

            Inputs(0)->Resize(nInput0, nT);
            Inputs(0)->FunctionValues().SetValue(0);
            Inputs(0)->FunctionValues()(0, 0) = 1;
            Inputs(0)->FunctionValues()(0, 1) = 2;
            Inputs(0)->FunctionValues()(0, 2) = 3;

            Inputs(1)->Resize(nInput1, nT);
            Inputs(1)->FunctionValues().SetValue(0);
            Inputs(1)->FunctionValues()(0, 0) = 4;
            Inputs(1)->FunctionValues()(0, 1) = 5;
            Inputs(1)->FunctionValues()(0, 2) = 6;
            Resize(nInput0 + nInput1, nT);

            EvaluateThisNode(FrameRange());

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

            ComputeInputPartial(0);
            ComputeInputPartial(1);

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
        typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembers;
    public:
        //virtual ComputationNodeBase * NewThis(DEVICEID_TYPE deviceId, const wstring & name) = 0;
        PreComputedNode(DEVICEID_TYPE deviceId, const wstring & name) : Base(deviceId, name)
        {
            // further initializations
            m_hasComputed = false;
            CreateMatrixIfNull(m_functionValues);
        }

        virtual bool HasComputed() const = 0;
        virtual void MarkComputed(const bool hasComputed) = 0;

        virtual bool RequiresPreCompute() const { return true;}

        virtual void SaveToFile(File& fstream)  const override
        {
            Base::SaveToFile(fstream);
            fstream << m_hasComputed;
            fstream << FunctionValues();
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_hasComputed;
            fstream >> FunctionValues();
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


        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<PreComputedNode<ElemType>>(nodeP);
                node->m_hasComputed = m_hasComputed;
            }
        }
    public:
        bool m_hasComputed;
    };

    #define UsingPreComputedNodeMembers UsingComputationNodeMembersBoilerplate; using Base::m_hasComputed

    //template class PreComputedNode<float>;
    //template class PreComputedNode<double>;

    // -----------------------------------------------------------------------
    // MeanNode (features)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class MeanNode : public PreComputedNode<ElemType>, public NumInputs<1>
    {
        typedef PreComputedNode<ElemType> Base; UsingPreComputedNodeMembers;
        static const std::wstring TypeName() { return L"Mean"; }
    public:
        MeanNode(DEVICEID_TYPE deviceId, const wstring & name) :
            PreComputedNode<ElemType>(deviceId, name),
            m_numSamples(0)
        { }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);
            m_numSamples = 0;   // TODO: intended? Not loaded from file?
        }

        virtual bool HasComputed() const { return m_hasComputed; }        // why are these not in the base class?

        virtual void MarkComputed(const bool hasComputed)
        {
            m_hasComputed = hasComputed;
            if (m_hasComputed)
                m_numSamples = 0;
        }

        virtual bool RequiresPreCompute() const { return true; }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/)
        {
            LogicError("Mean operation should not be involved in the gradient calculation.");
        }

        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            if (!m_hasComputed)
            {
                Matrix<ElemType> &samples =Inputs(0)->FunctionValues();
                Matrix<ElemType> &avg =FunctionValues();
    #if NANCHECK
                samples.HasNan("Mean-Samples");
    #endif

                size_t numNewSamples = samples.GetNumCols();
                Matrix<ElemType>::MultiplyAndWeightedAdd(1.0f / (m_numSamples + samples.GetNumCols()), samples, false,
                                                         ConstOnes(numNewSamples, 1, samples.GetDeviceId()),
                                                         false, (ElemType)m_numSamples / (m_numSamples + numNewSamples), avg);

    #if NANCHECK
                avg.HasNan("Mean-avg");
                ones.HasNan("Mean-ones");
    #endif

                m_numSamples += numNewSamples;
            }
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            //if (Inputs(0)->GetNumRows() == 0)
            //    LogicError("Mean operation: the input node has 0 element.");

            Resize(Inputs(0)->GetNumRows(), 1);
            m_pMBLayout = nullptr;    // this node does not hold mini-batch data
            InferImageDimsFromInputs();
        }

        //virtual void AttachInputs(const ComputationNodePtr singleInput)
        //{
        //    m_children.resize(1);
        //    m_children[0] = singleInput;
        //}

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<MeanNode<ElemType>>(nodeP);
                node->m_numSamples = m_numSamples;
            }
        }
    private:
        size_t m_numSamples;    // TODO: move to base class?
    };

    template class MeanNode<float>;
    template class MeanNode<double>;

    // -----------------------------------------------------------------------
    // InvStdDevNode (features)
    // TODO: share stuff with MeanNode
    // -----------------------------------------------------------------------

    template<class ElemType>
    class InvStdDevNode : public PreComputedNode<ElemType>, public NumInputs<1>
    {
        typedef PreComputedNode<ElemType> Base; UsingPreComputedNodeMembers;
        static const std::wstring TypeName() { return L"InvStdDev"; }
    public:
        InvStdDevNode(DEVICEID_TYPE deviceId, const wstring & name) :
            PreComputedNode<ElemType>(deviceId, name),
            m_mean(deviceId), m_var(deviceId), m_temp(deviceId),
            m_numSamples(0)
        { }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);
            m_numSamples = 0;   // TODO: intended? not loading from file?
        }

        virtual bool HasComputed() const { return m_hasComputed; }

        virtual void MarkComputed(const bool hasComputed)
        {
            m_hasComputed = hasComputed;

            if (m_hasComputed && m_numSamples > 0)  //m_numSamples>0 means it's not called from model loading
            {
                ElemType sqrtFloor = 1e-10f;

                m_var.InplaceTruncateBottom(sqrtFloor); //prevent too small variance (and negative square roots)
    #if NANCHECK
                m_var.HasNan("MarkComputed-InplaceTruncateBottom");
    #endif
                m_var.InplaceSqrt();

    #if NANCHECK
                m_var.HasNan("MarkComputed-InplaceSqrt");
    #endif
                m_var.ElementInverse();

    #if NANCHECK
                m_var.HasNan("MarkComputed-ElementInverse()");
    #endif
                FunctionValues().SetValue(m_var);

                m_numSamples = 0;
            }
        }

        virtual bool RequiresPreCompute() const { return true; }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/)
        {
            LogicError("InvStdDev operation should not be involved in the gradient calculation.");
        }

        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            if (!m_hasComputed)
            {
                Matrix<ElemType> &samples = Inputs(0)->FunctionValues();
    #if NANCHECK
                samples.HasNan("InvStdDev-Samples");
    #endif
                m_temp.SetValue(m_mean);
                size_t numNewSample = samples.GetNumCols();
                Matrix<ElemType>::MultiplyAndWeightedAdd(1.0f / (m_numSamples + numNewSample), samples, false,
                                                         ConstOnes(numNewSample, 1, samples.GetDeviceId()),
                                                         false, (ElemType)m_numSamples / (m_numSamples + numNewSample), m_mean);

                m_temp -= m_mean;
                m_temp.AssignElementPowerOf(m_temp, 2);
                m_var += m_temp;

                m_temp.AssignDifferenceOf(samples, m_mean);
                m_temp.AssignElementPowerOf(m_temp, 2);

                Matrix<ElemType>::MultiplyAndWeightedAdd(1.0f / (m_numSamples + numNewSample), m_temp, false,
                                                         ConstOnes(numNewSample, 1, samples.GetDeviceId()),
                                                         false, (ElemType)m_numSamples / (m_numSamples + numNewSample), m_var);

    #if NANCHECK
                m_var.HasNan("InvStdDev-m_var");
    #endif

                m_numSamples += samples.GetNumCols();
            }
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            //if (Inputs(0)->GetNumRows() == 0)
            //    LogicError("InvStdDev operation: the input node has 0 element.");

            size_t inputDim = Inputs(0)->GetNumRows();
            m_mean.Resize(inputDim, 1);
            m_var.Resize(inputDim, 1);

            Resize(inputDim, 1);
            m_pMBLayout = nullptr;    // this node does not hold mini-batch data
            InferImageDimsFromInputs();
        }

        //virtual void AttachInputs(const ComputationNodePtr singleInput)
        //{
        //    m_children.resize(1);
        //    m_children[0] = singleInput;
        //}

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            Base::MoveMatricesToDevice(deviceId);
            m_mean.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
            m_var.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
            m_temp.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<InvStdDevNode<ElemType>>(nodeP);
                node->m_numSamples = m_numSamples;

                node->m_mean = m_mean;
                node->m_var = m_var;
                node-> m_temp =  m_temp;
            }
        }
    private:
        size_t m_numSamples;
        Matrix<ElemType> m_mean;
        Matrix<ElemType> m_var;
        Matrix<ElemType>  m_temp;
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
        PerDimMeanVarNormalizationNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/)  //scaled by 2*number of colmns (samples) in the Matrix<ElemType>
        {
            InvalidArgument("PerDimMeanVarNormalizationNode should only be called in the evaluation stage.");   // TODO: don't we have a base class for this?
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange &)
        {
            InvalidArgument("PerDimMeanVarNormalizationNode should only be called in the evaluation stage.");
        }

        //(feature-mean).*InvStdDev
        void EvaluateThisNodeMap()    // TODO: This is a stop-gap; in most cases, we should just be able to delete this (but need to review one by one)
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(),
                              Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { EvaluateThisNodeMap(); return; }
            //only feature (input0) and output needs to be sliced
            Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange/*TODO: delete this:*/.Check(frameRange.t() * GetNumParallelSequences(), GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange/*TODO: delete this:*/.Check(frameRange.t() * GetNumParallelSequences(), GetNumParallelSequences(), m_pMBLayout));

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

            if (Inputs(1)->OperationName() == OperationNameOf(LearnableParameter))
            {
                size_t rows = (Inputs(1)->GetNumRows() == 0) ? Inputs(0)->GetNumRows() : Inputs(1)->GetNumRows();
                Inputs(1)->Resize(rows, 1);
            }

            if (Inputs(2)->OperationName() == OperationNameOf(LearnableParameter))
            {
                size_t rows = (Inputs(2)->GetNumRows() == 0) ? Inputs(0)->GetNumRows() : Inputs(2)->GetNumRows();
                Inputs(2)->Resize(rows, 1);
            }

            //if (Inputs(0)->GetNumRows() == 0 ||
            //    Inputs(1)->GetNumRows() == 0 ||
            //    Inputs(2)->GetNumRows() == 0)
            //{
            //    LogicError(
            //        "PerDimMeanVarNormalizationNode operation: one of the operands has 0 elements.");
            //}

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

            Inputs(1)->NeedGradient() = false;
            Inputs(2)->NeedGradient() = false;  //prevent learning
            Resize(Inputs(0));
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }

        //leftNode should be the empirical
        //virtual void AttachInputs(const ComputationNodePtr feature, const ComputationNodePtr mean, const ComputationNodePtr InvStdDev)
        //{
        //    m_children.resize(3);
        //    m_children[0] = feature;
        //    m_children[1] = mean;
        //    m_children[2] = InvStdDev;
        //}
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
        PerDimMeanVarDeNormalizationNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/)  //scaled by 2*number of colmns (samples) in the Matrix<ElemType>
        {
            InvalidArgument("PerDimMeanVarDeNormalizationNode should only be called in the evaluation stage.");
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange &)
        {
            InvalidArgument("PerDimMeanVarDeNormalizationNode should only be called in the evaluation stage.");
        }

        //(feature-mean).*InvStdDev
        void EvaluateThisNodeMap()    // TODO: This is a stop-gap; in most cases, we should just be able to delete this (but need to review one by one)
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(),
                              Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { EvaluateThisNodeMap(); return; }
            //only feature (input0) and output needs to be sliced
            Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange/*TODO: delete this:*/.Check(frameRange.t() * GetNumParallelSequences(), GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange/*TODO: delete this:*/.Check(frameRange.t() * GetNumParallelSequences(), GetNumParallelSequences(), m_pMBLayout));

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

            if (Inputs(1)->OperationName() == OperationNameOf(LearnableParameter))
            {
                size_t rows = Inputs(1)->GetNumRows() == 0 ? Inputs(0)->GetNumRows() : Inputs(1)->GetNumRows();
                Inputs(1)->Resize(rows, 1);
            }

            if (Inputs(2)->OperationName() == OperationNameOf(LearnableParameter))
            {
                size_t rows = Inputs(2)->GetNumRows() == 0? Inputs(0)->GetNumRows() : Inputs(2)->GetNumRows();
                Inputs(2)->Resize(rows, 1);
            }

            //if (Inputs(0)->GetNumRows() == 0 ||
            //    Inputs(1)->GetNumRows() == 0 ||
            //    Inputs(2)->GetNumRows() == 0)
            //{
            //    LogicError("PerDimMeanVarDeNormalizationNode operation: one of the operands has 0 elements.");
            //}

            if (isFinalValidationPass)
            {
                if (!(Inputs(0)->GetNumRows() == Inputs(1)->GetNumRows() &&  //match rows
                    Inputs(2)->GetNumRows() == Inputs(1)->GetNumRows()))
                {
                    //Inputs(1)->Resize(Inputs(0)->GetNumRows(), 1);
                    //Inputs(2)->Resize(Inputs(0)->GetNumRows(), 1);
                    LogicError("PerDimMeanVarDeNormalizationNode: All inputs should have same number of rows.");
                }

                if (!(Inputs(1)->GetNumCols() == 1 && Inputs(2)->GetNumCols() == 1))
                {
                    LogicError("PerDimMeanVarDeNormalizationNode: Mean and InvStdDev should be a colum  vector.");
                }
            }

            Inputs(1)->NeedGradient() = false;
            //prevent learning
            Inputs(2)->NeedGradient() = false;

            Resize(Inputs(0));
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }

        //leftNode should be the empirical
        //virtual void AttachInputs(const ComputationNodePtr feature, const ComputationNodePtr mean, const ComputationNodePtr InvStdDev)
        //{
        //    m_children.resize(3);
        //    m_children[0] = feature;
        //    m_children[1] = mean;
        //    m_children[2] = InvStdDev;
        //}
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
        BatchModeNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name),
            m_memory(deviceId)
        { }

        virtual bool HasComputed() const = 0;
        virtual void MarkComputed(const bool hasComputed) = 0;

        //virtual bool RequiresBatchMode() const { return true; }

#if 0   // I think this is a left-over. It does not seem to fit BatchMode
        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            assert(m_memory.GetNumCols() > 0);

            // BUGBUG: this is broken
            // TODO: what is this? Derives from ComputationNodeNonLooping, yet implemented a frame loop?
            //Resize(m_memory.GetNumRows(), GetNumParallelSequences());
            Resize(m_memory.GetNumRows(), GetNumParallelSequences());   // extra space for one time step
            if (frameRange.t() == 0)    // for first frame, check that we got all in memory  --TODO: is this comment correct? How about going backwards?
                assert(ValueSlice(FrameRange(0, GetNumParallelSequences())).FrobeniusNorm() == DataSlice(m_memory, FrameRange(0, GetNumParallelSequences())).FrobeniusNorm());
                //assert(FunctionValues().ColumnSlice(0, GetNumParallelSequences()), m_pMBLayout).FrobeniusNorm() == m_memory.ColumnSlice(0, GetNumParallelSequences()), m_pMBLayout).FrobeniusNorm());
            FunctionValues().SetValue(DataSlice(m_memory, frameRange/*TODO: delete this:*/.Check(frameRange.t() * GetNumParallelSequences(), GetNumParallelSequences(), m_pMBLayout)));
            assert(GetNumCols() == GetNumParallelSequences());
        }
#endif

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
            fstream >> FunctionValues();
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
        TimeReverseNode(DEVICEID_TYPE deviceId, const wstring & name) :
            BatchModeNode<ElemType>(deviceId, name)
        { }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<TimeReverseNode<ElemType>>(nodeP);
                node->m_memory = m_memory;
            }
        }

        virtual bool HasComputed() const { return m_hasComputed; }
        virtual void MarkComputed(const bool hasComputed) { m_hasComputed = hasComputed; }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            Base::MoveMatricesToDevice(deviceId);
            m_memory.TransferToDeviceIfNotThere(deviceId, true, m_memory.HasNoElements());
        }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                InvalidArgument("TimeReverse operation only takes one input.");
            ComputationNodePtr child = Inputs(inputIndex);
            ComputeInputPartialS(GradientValues(), child->GradientValues(), GetNumParallelSequences());
        }

        /*TODO: merge with call site*/void ComputeInputPartialS(Matrix<ElemType>& gradientValues, Matrix<ElemType>& inputGradientValues, int nSamples)
        {
    #if DUMPOUTPUT

            functionValues.Print("TimeReverseNode");
    #endif
            size_t nc = inputGradientValues.GetNumCols();
            size_t nr = inputGradientValues.GetNumRows();
            if (nc != gradientValues.GetNumCols() || nr != gradientValues.GetNumRows())
            {
                inputGradientValues.Resize(nr, nc);
                inputGradientValues.SetValue(0);
            }

            for (size_t i = 0; i < nc; i += nSamples)
            {
                Matrix<ElemType> ig = gradientValues.ColumnSlice(i, nSamples);
                Matrix<ElemType> ii = inputGradientValues.ColumnSlice(nc - i - nSamples, nSamples);
                ii += ig;
            }

    #if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
    #endif
        }

        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            if (m_hasComputed == false)
            {
                EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), GetNumParallelSequences());
                m_memory.SetValue(FunctionValues());
            }
        }

        /*TODO: merge with call site*/void EvaluateThisNodeS(Matrix<ElemType>& functionValues, Matrix<ElemType>& inputFunctionValues, int nSamples)
        {
            /// this assumes this reverse node is called once, so it can set, instead add to, the function values
            size_t rows0 = inputFunctionValues.GetNumRows(), cols0 = inputFunctionValues.GetNumCols();
            functionValues.Resize(rows0, cols0);

            for (size_t i = 0; i < cols0; i += nSamples)
            {
                Matrix<ElemType> ig = inputFunctionValues.ColumnSlice(i, nSamples);
                functionValues.ColumnSlice(cols0 - i - nSamples, nSamples).SetValue(ig);
            }

    #if NANCHECK
            m_functionValues->HasNan("TimeReverse");
    #endif
    #if DUMPOUTPUT
            functionValues.Print("TimeReverseNode");
    #endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);
            InferMBLayoutFromInputsForStandardCase();
            Resize(Inputs(0));
            InferImageDimsFromInput(0);
        }

        //virtual void AttachInputs(const ComputationNodePtr cNode)
        //{
        //    m_children.resize(1);
        //    m_children[0] = cNode;
        //}

    public:
        bool UnitTest() {
            size_t nT = 3;
            size_t nInput = 3;
            size_t nOutput = nInput;

            /// backup
            Matrix<ElemType> f0(m_deviceId), func(m_deviceId);

            f0 = Inputs(0)->FunctionValues();
            func = FunctionValues();

            Inputs(0)->Resize(nInput, nT);
            Inputs(0)->FunctionValues().SetValue(0);
            Inputs(0)->FunctionValues()(0, 0) = 1;
            Inputs(0)->FunctionValues()(0, 1) = 2;
            Inputs(0)->FunctionValues()(0, 2) = 3;
            Resize(nOutput, nT);
            Inputs(0)->FunctionValues().TransferToDeviceIfNotThere( m_deviceId, true);
            EvaluateThisNode(FrameRange());

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

            ComputeInputPartial(0);

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

    protected:
        virtual bool NodeDoesItsOwnCustomizedMissingColumnsMasking() 
        { 
           return true; 
        }

    };

    template class TimeReverseNode<float>;
    template class TimeReverseNode<double>;

}}}
