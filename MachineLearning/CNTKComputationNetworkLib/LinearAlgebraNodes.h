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

    // -----------------------------------------------------------------------
    // PlusNode (summand1, summand2)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class PlusNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        using Base::ValueSliceToDense;
        static const std::wstring TypeName() { return L"Plus"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(PlusNode);
        PlusNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            Matrix<ElemType> gradientValues = GradientSlice(frameRange);
            Matrix<ElemType> functionValues = ValueSlice(frameRange);
            Matrix<ElemType> inputGradientValues = Inputs(inputIndex)->GradientSlice(frameRange.AllowBroadcast());
            Matrix<ElemType> inputFunctionValues = Inputs(inputIndex)->ValueSlice(frameRange.AllowBroadcast());

#if DUMPOUTPUT
            functionValues.Print("PlusNode");
#endif
            size_t rowsc = inputFunctionValues.GetNumRows(), colsc = inputFunctionValues.GetNumCols();
            size_t rowsp = functionValues.GetNumRows(),      colsp = functionValues.GetNumCols();
#if DUMPOUTPUT
            fprintf(stderr, "input dimensions %lld x %lld,  this node dimensions %lld x %lld\n", rowsc, colsc, rowsp, colsp);
            gradientValues.Print("Gradient-in");
            inputGradientValues.Print("child Gradient-in/out");
#endif

            if (colsc == colsp && rowsc == rowsp)                   // matching dimensions  --this may also trigger for column vector added to a frame, if frameRange denotes a single frame
            {
                // BUGBUG: if we reduce from a frame of a MB into a one-column vector, then we must also mask gaps
                inputGradientValues += gradientValues;
            }
            else if (colsc == 1 && rowsc == 1)                      // child is a scalar
            {
                MaskMissingGradientColumnsToZero(frameRange);       // reducing over frames, so we must zero out the gaps
                inputGradientValues += gradientValues.SumOfElements();
            }
            else if (colsc == 1 && colsp != 1)                      // child is a broadcasting column vector
            {
                size_t colspExpand = rowsp*colsp/rowsc;
                MaskMissingGradientColumnsToZero(frameRange);       // reducing over frames, so we must zero out the gaps
                Matrix<ElemType>::MultiplyAndAdd(gradientValues.Reshaped(rowsc, colspExpand), false, ConstOnes(colspExpand, 1, functionValues.GetDeviceId()), false, inputGradientValues);
            }
            else if (rowsc == 1 && rowsp != 1)                      // child is a broadcasting row vector
            {
                Matrix<ElemType>::MultiplyAndAdd(ConstOnes(1, rowsp, functionValues.GetDeviceId()), false, gradientValues, false, inputGradientValues);
            }
            else if (colsc != 1 && colsp % colsc == 0)
            {
                // the children matrix is [a b] and the parent considers it as [a a a b b b]
                // Note: There is no need to mask gaps here because this operation is only allowed on non-MBLayout inputs
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

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override  
        {
            Matrix<ElemType> functionValues = ValueSliceToDense(frameRange, false); // Switch to dense as a work-around because ColumnSlice doesn't support all the sparse formats
            Matrix<ElemType> inputFunctionValues0 = Inputs(0)->ValueSlice(frameRange.AllowBroadcast());
            Matrix<ElemType> inputFunctionValues1 = Inputs(1)->ValueSlice(frameRange.AllowBroadcast());
            // Note: If one input is a column vector (no MBLayout) and the other a sequence of frames (MBLayout), then the above will be a slice for the other only.

            size_t rows0 = inputFunctionValues0.GetNumRows(), cols0 = inputFunctionValues0.GetNumCols();
            size_t rows1 = inputFunctionValues1.GetNumRows(), cols1 = inputFunctionValues1.GetNumCols();

            if ((rows0 == rows1 && cols0 == cols1/*matching dimensions*/) || ((rows0 == 1 || rows1 == 1)/*one is a broadcasting row vector*/ && cols0 == cols1))
            {
                functionValues.AssignSumOf(inputFunctionValues0, inputFunctionValues1);
            }
            else if (cols0 == 1 && rows1 % rows0 == 0)  // one is col vec with divisable rows, including scalar   --allowing divisable rows can be useful for images
            {
                functionValues.AssignSumOf(inputFunctionValues0, inputFunctionValues1.Reshaped(rows0, rows1 * cols1 / rows0));
                functionValues.Reshape(max(rows0, rows1), max(cols0, cols1));
            }
            else if (cols1 == 1 && rows0 % rows1 == 0)  // one is col vec with divisable rows, including scalar
            {
                functionValues.Reshape(rows1, rows0 * cols0 / rows1);
                functionValues.AssignSumOf(inputFunctionValues0.Reshaped(rows1, rows0 * cols0 / rows1), inputFunctionValues1);
                functionValues.Reshape(max(rows0, rows1), max(cols0, cols1));
            }       
            else if (cols1 < cols0 && rows0 == rows1 && cols0 % cols1 == 0)  // first summand is a matrix with number of columns that is a multiple of the column number of the second matrix
            {
                if (m_pMBLayout)
                    InvalidArgument("%ls %ls operation applied to mismatching number of columns when columns are samples of a minibatch", NodeName().c_str(), OperationName().c_str());
                // the children matrix is [a b] and the parent considers it as [a a a b b b]
                // This can be useful for dealing with images.
                Matrix<ElemType> tmpMat(inputFunctionValues1.GetDeviceId());
                size_t ratio = cols0 / cols1;
                // TODO: Why is this different from MinusNode?
                for (size_t i = 0; i < cols1; i++)
                {
                    tmpMat = Matrix<ElemType>::RepMat(inputFunctionValues1.ColumnSlice(i, 1), 1, ratio);
                    functionValues.ColumnSlice(i*ratio, ratio).SetValue(tmpMat + inputFunctionValues0.ColumnSlice(i * ratio, ratio)); 
                }
            }
            else
                LogicError("%ls %ls operation's Validate() function let invalid dimensions slip by.", NodeName().c_str(), OperationName().c_str());
#if DUMPOUTPUT
            functionValues.Print("PlusNode");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            ValidateBinaryZip(isFinalValidationPass, true/*allowMultiples*/);
        }

        virtual void InferImageDimsFromInputs() //based on the matrix with larger size
        {
            size_t rows0 = Inputs(0)->GetNumRows(), cols0 = Inputs(0)->GetNumCols();
            size_t rows1 = Inputs(1)->GetNumRows(), cols1 = Inputs(1)->GetNumCols();

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
    };

    template class PlusNode<float>; 
    template class PlusNode<double>;

    // -----------------------------------------------------------------------
    // MinusNode (minuend, subtrahend)
    // -----------------------------------------------------------------------

    // TODO: merge with PlusNode
    template<class ElemType>
    class MinusNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"Minus"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(MinusNode);
        MinusNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            Matrix<ElemType> gradientValues = GradientSlice(frameRange);
            Matrix<ElemType> functionValues = ValueSlice(frameRange);

            Matrix<ElemType> childGradientValues = Inputs(inputIndex)->GradientSlice(frameRange.AllowBroadcast());
            Matrix<ElemType> childFunctionValues = Inputs(inputIndex)->ValueSlice(frameRange.AllowBroadcast());

            size_t rowsc = childFunctionValues.GetNumRows(), colsc = childFunctionValues.GetNumCols();
            size_t rowsp = functionValues.GetNumRows(),      colsp = functionValues.GetNumCols();

            ElemType sign = inputIndex == 0 ? 1.0f : -1.0f;
            if (colsc == colsp && rowsc == rowsp)                   // matching dimensions
            {
                // BUGBUG: if we reduce from a frame of a MB into a one-column vector, then we must also mask gaps
                if (sign > 0)
                    childGradientValues += gradientValues;
                else
                    childGradientValues -= gradientValues;
            }
            else if (colsc == 1 && rowsc == 1)                      // child is a scalar (1 x 1)
            {
                MaskMissingGradientColumnsToZero(frameRange);       // reducing over frames, so we must zero out the gaps
                if (sign > 0)
                    childGradientValues += gradientValues.SumOfElements();
                else
                    childGradientValues -= gradientValues.SumOfElements();
            }
            else if (colsc == 1 && colsp != 1)                      // child is broadcasting column vector
            {
                size_t colspExpand = rowsp * colsp / rowsc;
                MaskMissingGradientColumnsToZero(frameRange);       // reducing over frames, so we must zero out the gaps
                Matrix<ElemType>::MultiplyAndWeightedAdd(sign, gradientValues.Reshaped(rowsc, colspExpand), false, ConstOnes(colspExpand, 1, FunctionValues().GetDeviceId()), false, 1, childGradientValues);
            }
            else if (rowsc == 1 && rowsp != 1)                      // child is a broadcasting row vector
            {
                Matrix<ElemType>::MultiplyAndWeightedAdd(sign, ConstOnes(1, rowsp, FunctionValues().GetDeviceId()), false, gradientValues, false, 1, childGradientValues);
            }
            else
                LogicError("%ls %ls operation's Validate() function let invalid dimensions slip by.", NodeName().c_str(), OperationName().c_str());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            Matrix<ElemType> functionValues = ValueSlice(frameRange);
            Matrix<ElemType> inputFunctionValues0 = Inputs(0)->ValueSlice(frameRange.AllowBroadcast());
            Matrix<ElemType> inputFunctionValues1 = Inputs(1)->ValueSlice(frameRange.AllowBroadcast());

            size_t rows0 = inputFunctionValues0.GetNumRows(), cols0 = inputFunctionValues0.GetNumCols();
            size_t rows1 = inputFunctionValues1.GetNumRows(), cols1 = inputFunctionValues1.GetNumCols();
            functionValues.VerifySize(max(rows0, rows1), max(cols0,cols1));

            if ((rows0 == rows1 && cols0 == cols1/*match*/) || ((rows0 == 1 || rows1 == 1)/*one is a broadcasting row vector*/ && cols0 == cols1))
            {
                functionValues.AssignDifferenceOf(inputFunctionValues0, inputFunctionValues1);
            }
            else if (cols0 == 1 && rows1 % rows0 == 0)  // one is col vec with divisable rows, including scalar
            {
                functionValues.AssignDifferenceOf(inputFunctionValues0, inputFunctionValues1.Reshaped(rows0, rows1 * cols1 / rows0));
                functionValues.Reshape(max(rows0, rows1), max(cols0,cols1));
            }
            else if (cols1 == 1 && rows0 % rows1 == 0)  // one is col vec with divisable rows, including scalar
            {
                functionValues.AssignDifferenceOf(inputFunctionValues0.Reshaped(rows1, rows0 * cols0 / rows1), inputFunctionValues1);
                functionValues.Reshape(max(rows0, rows1), max(cols0, cols1));
            }
            else
                LogicError("%ls %ls operation's Validate() function let invalid dimensions slip by.", NodeName().c_str(), OperationName().c_str());
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            ValidateBinaryZip(isFinalValidationPass, true/*allowMultiples*/);
        }

        virtual void InferImageDimsFromInputs() //based on the matrix with larger size
        {
            size_t rows0 = Inputs(0)->GetNumRows(), cols0 = Inputs(0)->GetNumCols();
            size_t rows1 = Inputs(1)->GetNumRows(), cols1 = Inputs(1)->GetNumCols();

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
    };

    template class MinusNode<float>; 
    template class MinusNode<double>;

    // -----------------------------------------------------------------------
    // ScaleNode (scalar scaling factor, matrix)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class ScaleNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"Scale"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(ScaleNode);
        ScaleNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (inputIndex == 0)        // left derivative
            {
                // this is a reduction over frames, so we must mask gaps to zero
                Inputs(0)->GradientValues() += Matrix<ElemType>::InnerProductOfMatrices(MaskedGradientSlice(frameRange), Inputs(1)->MaskedValueSlice(frameRange)); // element-wise product summed up over all
            }
            else if (inputIndex == 1)   // right derivative
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientSlice(frameRange);
                //Matrix<ElemType>::ScaleAndAdd(Inputs(0)->FunctionValues().Get00Element(), GradientSlice(frameRange), sliceInput1Grad);
                Matrix<ElemType>::Multiply1x1AndWeightedAdd(+1.0f, Inputs(0)->FunctionValues()/*1x1*/, GradientSlice(frameRange), 1.0f, sliceInput1Grad);
            }
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override  
        {
            //ValueSlice(frameRange).AssignProductOf(Inputs(0)->FunctionValues().Get00Element(), Inputs(1)->ValueSlice(frameRange));
            ValueSlice(frameRange).Assign1x1ProductOf(Inputs(0)->FunctionValues()/*1x1*/, Inputs(1)->ValueSlice(frameRange));
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            // left Node must be a scalar
            if (isFinalValidationPass && (Inputs(0)->GetNumRows() != 1 || Inputs(0)->GetNumCols() != 1))
                RuntimeError("The left value of ScaleNode must be a scalar value.");

            SetDims(Inputs(1));
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(1); 
        }
    };

    template class ScaleNode<float>; 
    template class ScaleNode<double>;

    // -----------------------------------------------------------------------
    // NegateNode (input)
    // computes the negative of its input
    // -----------------------------------------------------------------------

    template<class ElemType>
    class NegateNode : public ComputationNode<ElemType>, public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"Negate"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(NegateNode);
        NegateNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange & frameRange) override
        {
            Inputs(0)->GradientSlice(frameRange) -= GradientSlice(frameRange);
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override 
        {
            ValueSlice(frameRange).AssignDifferenceOf(0, Inputs(0)->ValueSlice(frameRange));
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            ValidateUnaryMap(isFinalValidationPass);
        }
    };

    template class NegateNode<float>; 
    template class NegateNode<double>;

    // -----------------------------------------------------------------------
    // TimesNode (A, B)
    // right operand and output can have MB layout, while left operand cannot
    // -----------------------------------------------------------------------

    template<class ElemType>
    class TimesNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"Times"; }
    public:

        DeclareConstructorFromConfigWithNumInputs(TimesNode);
        TimesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        {
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (inputIndex == 0)    // left derivative
            {
                // this potentially computes inner products over time, so we use the Masked- variants
                Matrix<ElemType> sliceOutputGrad = MaskedGradientSlice(frameRange);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->MaskedValueSlice(frameRange);

                // currently we only support one combination when the input is sparse.
                if (sliceInput1Value.GetMatrixType() == SPARSE && Inputs(0)->GradientValues().GetMatrixType() == DENSE && sliceOutputGrad.GetMatrixType() == DENSE)
                    Inputs(0)->GradientValues().SwitchToMatrixType(SPARSE, MatrixFormat::matrixFormatSparseBlockCol, false);

                Matrix<ElemType>::MultiplyAndAdd(sliceOutputGrad, false, sliceInput1Value, true, Inputs(0)->GradientValues());
            }
            else                    // right derivative
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientSlice(frameRange);
                Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange);

                Matrix<ElemType>::MultiplyAndAdd(Inputs(0)->FunctionValues(), true, sliceOutputGrad, false, sliceInput1Grad);
            }
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            size_t rows0 = Inputs(0)->GetNumRows(), cols1 = Inputs(1)->GetNumCols();
            VerifyDims(rows0, cols1);

            // right operand and output can have MB layout, while left operand cannot
            Matrix<ElemType> sliceInput1Value = Inputs(1)->ValueSlice(frameRange);
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange);
#if DUMPOUTPUT
            Inputs(0)->FunctionValues().Print("TimesNode - Input0");
#endif
            sliceOutputValue.AssignProductOf(Inputs(0)->FunctionValues(), false, sliceInput1Value, false);
#if NANCHECK
            sliceOutputValue.HasNan("Times");
#endif
#if DUMPOUTPUT
            sliceOutputValue.Print("TimesNode");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            //support automatic dimension inference for learnable parameters
            size_t rows0 = Inputs(0)->GetNumRows(), cols0 = Inputs(0)->GetNumCols();
            size_t rows1 = Inputs(1)->GetNumRows(), cols1 = Inputs(1)->GetNumCols();

            if (isFinalValidationPass && (rows0 == 0 || (cols1 == 0 && !Inputs(1)->GetMBLayout())))
                RuntimeError("Times operation: Inputs(0)->GetNumRows() and Inputs(1)->GetNumCols() should not be 0 since it cannot be automatically inferred");

            // limited automatic dimension inference for *children*, useful for CNN since it can be hard to know the size of each input parameter without deep knowledge how CNN is implemented (padding, stride)
            // TODO: ^^ There must be a better solution. Maybe MBLayout as well?
            // TODO: use dynamic_pointer_cast
            // infer cols0 as rows1
            if (cols0 == 0 && !Inputs(0)->GetMBLayout() && rows1 != 0 && isFinalValidationPass)
                ValidateInferChildDims(0, rows0, rows1);

            // infer rows1 as cols0
            if (cols0 != 0 && rows1 == 0)
                ValidateInferChildDims(1, cols0, cols1);

            if (isFinalValidationPass && Inputs(1)->GetNumRows() != Inputs(0)->GetNumCols())
                LogicError("The inner matrix dimension in the %ls %ls operation does not match (%d vs. %d).", NodeName().c_str(), OperationName().c_str(), (int)Inputs(1)->GetNumRows(), (int)Inputs(0)->GetNumCols());
            SetDims(rows0, cols1);

            if (isFinalValidationPass && Inputs(0)->HasMBLayout())
                InvalidArgument("%ls %ls operation requires the first factor to not be minibatch data (must not have an MBLayout).", NodeName().c_str(), OperationName().c_str());
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs()  
        {
            InferImageDimsFromInput(1, false); //the second one is the input since it's columnwise

            //after multiplication the structure is lost
            m_imageLayout = ImageLayoutWHC(1, Inputs(0)->GetNumRows(), 1);
        }

        virtual void AllocateGradientMatricesForChildren(MatrixPool& matrixPool) override
        {
            //this is a special handling case. We need to allocate sparse matrix directly instead of from pool.
            if (m_children[0]->NeedGradient() && Inputs(1)->FunctionValues().GetMatrixType() == SPARSE)
            {
                CreateMatrixIfNull(Inputs(0)->GradientValuesPtr());
                Inputs(0)->GradientValues().SwitchToMatrixType(SPARSE, MatrixFormat::matrixFormatSparseBlockCol, false);
            }
           
            //we need to call base allocation at end since we will need to allocate special ones first 
            //so that the default allocator will not allocate it again.
            Base::AllocateGradientMatricesForChildren(matrixPool);
        }
    };

    template class TimesNode<float>; 
    template class TimesNode<double>;

    // -----------------------------------------------------------------------
    // TransposeTimesNode (A', B)
    // right operand and output can have MB layout, while left operand cannot
    // TODO: merge with TimesNode?
    // -----------------------------------------------------------------------

    template<class ElemType>
    class TransposeTimesNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"TransposeTimes"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(TransposeTimesNode);
        TransposeTimesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (inputIndex == 0)  //left derivative
            {
                // this potentially computes inner products over time, so we use the Masked- variants
                Matrix<ElemType> sliceOutputGrad = MaskedGradientSlice(frameRange);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->MaskedValueSlice(frameRange);

                ComputeInputPartialLeft(sliceInput1Value, Inputs(0)->GradientValues(), sliceOutputGrad);
            }
            else  //right derivative
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientSlice(frameRange);
                Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange);

                ComputeInputPartialRight(Inputs(0)->FunctionValues(), sliceInput1Grad, sliceOutputGrad);
            }
        }

        /*TODO: merge with call site*/void ComputeInputPartialLeft(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
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

        /*TODO: merge with call site*/void ComputeInputPartialRight(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
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

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->ValueSlice(frameRange);
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange);

            sliceOutputValue.AssignProductOf(Inputs(0)->FunctionValues(), true, sliceInput1Value, false);
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            //support automatic dimension inference for learnable parameters
            size_t rows0 = Inputs(0)->GetNumRows(), cols0 = Inputs(0)->GetNumCols();
            size_t rows1 = Inputs(1)->GetNumRows(), cols1 = Inputs(1)->GetNumCols();

            if (isFinalValidationPass && (rows0 == 0 || (!Inputs(1)->HasMBLayout() && cols1 == 0)))
                RuntimeError("TransposeTimes operation: Inputs(0)->GetNumRows() and Inputs(1)->GetNumCols() should not be 0 since it cannot be automatically inferred");

            if (cols0 == 0 && rows1 != 0 && isFinalValidationPass)
                ValidateInferChildDims(0, rows0, rows1);

            if (cols0 != 0 && rows1 == 0)
                ValidateInferChildDims(1, cols0, cols1);

            //cols0 and rows1 may have been changed so don't use them in the following check
            if (isFinalValidationPass && Inputs(1)->GetNumRows() != Inputs(0)->GetNumRows())
                LogicError("The Matrix dimension in the TransposeTimes operation does not match.");

            SetDims(cols0, cols1);
            InferMBLayoutFromInputsForStandardCase();   // TODO: what does the MBLayout mean in the context of TransposeTimes? Can the left arg have an MBLayout?
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(1, false); //the second one is the input since it's column wize

            //after multiplication the structure is lost
            m_imageLayout = ImageLayoutWHC(1, Inputs(0)->GetNumRows(), 1);
        }
    };

    template class TransposeTimesNode<float>;
    template class TransposeTimesNode<double>;

    // -----------------------------------------------------------------------
    // ElementTimesNode (factor1, factor2)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class ElementTimesNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"ElementTimes"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(ElementTimesNode);
        ElementTimesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            Matrix<ElemType> sliceInput0Grad = Inputs(inputIndex)->GradientSlice(frameRange);
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange);
            Matrix<ElemType> sliceInput1Value = Inputs(1-inputIndex)->ValueSlice(frameRange);

            // depending on inputIndex, all the input variables change meaning
            // inputIndex == 0 (left) -  inputGradientValues[0], inputFunctionValues[1]
            // inputIndex == 1 (right) - inputGradientValues[1], inputFunctionValues[0]
            sliceInput0Grad.AddElementProductOf(sliceOutputGrad, sliceInput1Value);
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override  
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->ValueSlice(frameRange);
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange);

            //EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value);
            sliceOutputValue.AssignElementProductOf(sliceInput0Value, sliceInput1Value);
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            ValidateBinaryZip(isFinalValidationPass, false/*allowMultiple*/);
        }

        virtual void InferImageDimsFromInputs()
        {
            if (IsChildAnImage(0))  // if conflict, give priority to child 0
                InferImageDimsFromInput(0);
            else
                InferImageDimsFromInput(1);
        }
    };

    template class ElementTimesNode<float>; 
    template class ElementTimesNode<double>;

    // -----------------------------------------------------------------------
    // RowElementTimesNode (left, right)  --TODO: what are left and right?
    // -----------------------------------------------------------------------

    template<class ElemType>
    class RowElementTimesNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"RowElementTimes"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(RowElementTimesNode);
        RowElementTimesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        void ComputeInputPartialMap(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("RowElementTimes operation only takes two inputs.");

            if (inputIndex == 0)
            {
                ComputeInputPartialLeftS(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues(), *m_tempMatrix);
            }
            else
            {
                ComputeInputPartialRightS(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues(), *m_tempMatrix);
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            Matrix<ElemType> sliceInput0Grad = Inputs(inputIndex)->GradientSlice(frameRange);
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange);

            Matrix<ElemType> sliceInput1Value = Inputs(1 - inputIndex)->ValueSlice(frameRange);

            if (inputIndex == 0)
            {
                ComputeInputPartialLeftS(sliceInput1Value, sliceInput0Grad, sliceOutputGrad, *m_tempMatrix);
            }
            else
            {
                ComputeInputPartialRightS(sliceInput1Value, sliceInput0Grad, sliceOutputGrad, *m_tempMatrix);
            }
        }

        //left (input 0) is a matrix
        /*TODO: merge with call site*/void ComputeInputPartialLeftS(Matrix<ElemType>& input1FunctionValues,
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
        /*TODO: merge with call site*/void ComputeInputPartialRightS(Matrix<ElemType>& input0FunctionValues, 
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
        void EvaluateThisNodeMap()    // TODO: This is a stop-gap; in most cases, we should just be able to delete this (but need to review one by one)
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            //if (frameRange.IsAllFrames()) { EvaluateThisNodeMap(); return; }
            Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->ValueSlice(frameRange);
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange);

            EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, sliceInput1Value);
        }

        /*TODO: merge with call site*/void EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, const Matrix<ElemType>& input1)
        {
            functionValues.SetValue(input0);
            functionValues.RowElementMultiplyWith(input1);

#if NANCHECK
            functionValues.HasNan("RowElementTimes");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            size_t rows0 = Inputs(0)->GetNumRows(), cols0 = Inputs(0)->GetNumCols();
            size_t rows1 = Inputs(1)->GetNumRows(), cols1 = Inputs(1)->GetNumCols(); rows0;
            if (isFinalValidationPass && cols0 != cols1 || rows1 != 1)
                LogicError("RowElementTimes: Either the second operand is not a row vector or the number of columns of operands does not match.");

            SetDims(Inputs(0));
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            // input 0 is the matrix and input 1 is a row vector
            InferImageDimsFromInput(0);
        }

        //request matrices that are needed for gradient computation
        virtual void RequestMatricesBeforeGradientComp(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeGradientComp(matrixPool);
            RequestMatrixFromPool(m_tempMatrix, matrixPool);
        }

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool)
        {
            Base::ReleaseMatricesAfterGradientComp(matrixPool);
            ReleaseMatrixToPool(m_tempMatrix, matrixPool);
        }

    private:
        shared_ptr<Matrix<ElemType>> m_tempMatrix;
    };

    template class RowElementTimesNode<float>;
    template class RowElementTimesNode<double>;

    // -----------------------------------------------------------------------
    // ColumnElementTimesNode (left, right)  --TODO: what are left and right?
    // -----------------------------------------------------------------------

    template<class ElemType>
    class ColumnElementTimesNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"ColumnElementTimes"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(ColumnElementTimesNode);
        ColumnElementTimesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        void ComputeInputPartialMap(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("ColumnElementTimes operation only takes two inputs.");

            if (inputIndex == 0)
            {
                ComputeInputPartialLeftS(Inputs(1)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues(), *m_tempMatrix);
            }
            else
            {
                ComputeInputPartialRightS(Inputs(0)->FunctionValues(), Inputs(1)->GradientValues(), GradientValues(), *m_tempMatrix);
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange);

            if (inputIndex == 0)
            {
                Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientSlice(frameRange);

                ComputeInputPartialLeftS(Inputs(1)->FunctionValues(), sliceInput0Grad, sliceOutputGrad, *m_tempMatrix);
            }
            else
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange);
                ComputeInputPartialRightS(sliceInput0Value, Inputs(1)->GradientValues(), sliceOutputGrad, *m_tempMatrix);
            }
        }

        //left (input 0) is a matrix
        /*TODO: merge with call site*/void ComputeInputPartialLeftS(Matrix<ElemType>& input1FunctionValues,
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
        /*TODO: merge with call site*/void ComputeInputPartialRightS(Matrix<ElemType>& input0FunctionValues,
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
        void EvaluateThisNodeMap()    // TODO: This is a stop-gap; in most cases, we should just be able to delete this (but need to review one by one)
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues());
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            //if (frameRange.IsAllFrames()) { EvaluateThisNodeMap(); return; }
            Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange);
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange);

            EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, Inputs(1)->FunctionValues());
        }

        /*TODO: merge with call site*/void EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0, const Matrix<ElemType>& input1)
        {
            functionValues.SetValue(input0);
            functionValues.ColumnElementMultiplyWith(input1);

#if NANCHECK
            functionValues.HasNan("ColumnElementTimes");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            //derive number of rows if possible
            for (size_t index = 0; index < 2; index++)
            {
                size_t rows = Inputs(index)->GetNumRows() == 0 ? Inputs(1 - index)->GetNumRows() : Inputs(index)->GetNumRows();
                size_t cols = Inputs(index)->GetNumCols() == 0 ? Inputs(1 - index)->GetNumCols() : Inputs(index)->GetNumCols();
                ValidateInferChildDims(index, rows, cols);
            }

            size_t rows0 = Inputs(0)->GetNumRows(), cols0 = Inputs(0)->GetNumCols();
            size_t rows1 = Inputs(1)->GetNumRows(), cols1 = Inputs(1)->GetNumCols(); cols0;
            if (isFinalValidationPass && (rows0 != rows1 || cols1 != 1))
                LogicError("ColumnElementTimes: Either the second operand is not a column vector or the number of rows of operands does not match.");

            SetDims(Inputs(0));
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            // input 0 is a matrix and input 1 is a column vector
            InferImageDimsFromInput(0);
        }

        //request matrices that are needed for gradient computation
        virtual void RequestMatricesBeforeGradientComp(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeGradientComp(matrixPool);
            RequestMatrixFromPool(m_tempMatrix, matrixPool);
        }

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool)
        {
            Base::ReleaseMatricesAfterGradientComp(matrixPool);
            ReleaseMatrixToPool(m_tempMatrix, matrixPool);
        }

    private:
        shared_ptr<Matrix<ElemType>> m_tempMatrix;
    };

    template class ColumnElementTimesNode<float>;
    template class ColumnElementTimesNode<double>;

    // -----------------------------------------------------------------------
    // DiagTimesNode (vector representing the diagonal of a square matrix, data)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class DiagTimesNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"DiagTimes"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(DiagTimesNode);
        DiagTimesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (inputIndex == 0)    // left derivative
            {
                Matrix<ElemType> sliceOutputGrad  = MaskedGradientSlice(frameRange);            // use Masked- version since this is reducing over frames
                Matrix<ElemType> sliceInput1Value = Inputs(1)->MaskedValueSlice(frameRange);
                m_innerproduct->AssignInnerProductOf(sliceOutputGrad, sliceInput1Value, false);
                Inputs(0)->GradientValues() += *m_innerproduct;
            }
            else                    // right derivative
            {
                Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange);
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientSlice(frameRange);
                m_rightGradient->SetValue(sliceOutputGrad);
                m_rightGradient->ColumnElementMultiplyWith(Inputs(0)->FunctionValues());
                sliceInput1Grad += *m_rightGradient;
            }
        }

        ///*TODO: merge with call site*/void ComputeInputPartialLeft(Matrix<ElemType>& temp, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        //{
        //    temp.AssignInnerProductOf(gradientValues, inputFunctionValues, false);
        //    inputGradientValues += temp;
        //}
        //
        ///*TODO: merge with call site*/void ComputeInputPartialRight(Matrix<ElemType>& temp, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        //{
        //    temp.SetValue(gradientValues);
        //    temp.ColumnElementMultiplyWith(inputFunctionValues);
        //    inputGradientValues += temp;
        //}

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override  
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->ValueSlice(frameRange);
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange);

            sliceOutputValue.SetValue(sliceInput1Value);
            sliceOutputValue.ColumnElementMultiplyWith(Inputs(0)->FunctionValues());
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            //if dimension not specified we assume two operands' dimensions should match
            if (Inputs(0)->GetNumRows() == 0 && Inputs(1)->GetNumRows() != 0)
                ValidateInferChildDims(0, Inputs(1)->GetNumRows(), 1);

            if (Inputs(0)->GetNumRows() != 0 && Inputs(1)->GetNumRows() == 0)
                ValidateInferChildDims(1, Inputs(0)->GetNumRows(), Inputs(1)->GetNumCols());

            if (isFinalValidationPass)
            {
                if (Inputs(1)->GetNumRows() != Inputs(0)->GetNumRows())
                    LogicError("The Matrix dimension in the DiagTimes operation does not match.");

                if (Inputs(0)->GetNumCols() != 1)
                    LogicError("The first matrix should be a vector representing the diagonal of a square matrix in the DiagTimes operation.");
            }

            SetDims(Inputs(0)->GetNumRows(), Inputs(1)->GetNumCols());
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs() //this is element wise scaling, so based on child 1
        {
            InferImageDimsFromInput(1);
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<DiagTimesNode<ElemType>>(nodeP);
                *node->m_innerproduct = *m_innerproduct;
                *node->m_rightGradient = *m_rightGradient;
            }
        }
        //request matrices that are needed for gradient computation
        virtual void RequestMatricesBeforeGradientComp(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeGradientComp(matrixPool);
            RequestMatrixFromPool(m_innerproduct, matrixPool);
            RequestMatrixFromPool(m_rightGradient, matrixPool);
        }

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool)
        {
            Base::ReleaseMatricesAfterGradientComp(matrixPool);
            ReleaseMatrixToPool(m_innerproduct, matrixPool);
            ReleaseMatrixToPool(m_rightGradient, matrixPool);
        }
private:
        shared_ptr<Matrix<ElemType>> m_innerproduct;
        shared_ptr<Matrix<ElemType>> m_rightGradient;
    };

    template class DiagTimesNode<float>; 
    template class DiagTimesNode<double>;

    // -----------------------------------------------------------------------
    // SumElementsNode (input)
    // sums up all elements in the input into a single scalar
    // -----------------------------------------------------------------------

    template<class ElemType>
    class SumElementsNode : public ComputationNode<ElemType>, public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"SumElements"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(SumElementsNode);
        SumElementsNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange & frameRange) override
        {
            // BUGBUG: In the future we may want to allow this to operate on a scalar that is one step of an outer time loop.
            Inputs(0)->GradientSlice(frameRange) += GradientValues(); // here the assumption is that gradientValues are 1x1 matrix
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            FunctionValues().AssignSumOfElements(Inputs(0)->MaskedValueSlice(frameRange));  // since we are reducing over frames, we must first mask gaps in input to zero
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            SetDims(1, 1);
            m_pMBLayout = nullptr;    // this node does not hold mini-batch data
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            m_imageLayout = ImageLayout();
        }
    };

    template class SumElementsNode<float>; 
    template class SumElementsNode<double>;

    // -----------------------------------------------------------------------
    // SumColumnElementsNode (input)
    // sums up each column of the input
    // -----------------------------------------------------------------------

    template<class ElemType>
    class SumColumnElementsNode : public ComputationNode<ElemType>, public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"SumColumnElements"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(SumColumnElementsNode);
        SumColumnElementsNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange & frameRange) override
        {
            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientSlice(frameRange);
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange);

            sliceInputGrad += sliceOutputGrad; // here the assumption is that gradientValues is a row vector
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->ValueSlice(frameRange);
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange);

            //EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
            Matrix<ElemType>::VectorSum(sliceInputValue, sliceOutputValue, true);
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            SetDims(1, Inputs(0)->GetNumCols());
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            m_imageLayout = ImageLayout();
        }
    };

    template class SumColumnElementsNode<float>;
    template class SumColumnElementsNode<double>;

    // -----------------------------------------------------------------------
    // TransposeNode (input matrix)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class TransposeNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>, public NumInputs<1>
    {
        typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"Transpose"; }

    public:
        DeclareConstructorFromConfigWithNumInputs(TransposeNode);
        TransposeNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNodeNonLooping::*/ComputeInputPartialNonLooping(size_t /*inputIndex*/) override
        {
            Matrix<ElemType>& inputGradientValues = Inputs(0)->GradientValues();
            const Matrix<ElemType>& gradientValues = GradientValues();
#if DUMPOUTPUT
            gradientValues.Print("Gradient-in");
            inputGradientValues.Print("child Gradient-in/out");
            inputFunctionValues.Print("child Function values");
#endif
            const Matrix<ElemType>& ones = ConstOnes(inputGradientValues.GetNumRows(), inputGradientValues.GetNumRows(), inputGradientValues.GetDeviceId());
            Matrix<ElemType>::MultiplyAndAdd(ones, false, gradientValues, true, inputGradientValues);
#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
        }

        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
#if DUMPOUTPUT
            Inputs(0)->FunctionValues().Print("TransposeNode- Input0");
#endif
            FunctionValues().AssignTransposeOf(Inputs(0)->FunctionValues());
#if NANCHECK
            FunctionValues().HasNan("Transpose");
#endif
#if DUMPOUTPUT
            FunctionValues().Print("TransposeNode");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            size_t rows0 = Inputs(0)->GetNumRows(), cols0 = Inputs(0)->GetNumCols();

            SetDims(cols0, rows0);
            if (Inputs(0)->HasMBLayout())
                InvalidArgument("%ls %ls operation cannot operate on minibatch data (which have a layout)", NodeName().c_str(), OperationName().c_str());
            m_pMBLayout = nullptr;    // this node does not hold mini-batch data
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false); // the second one is the input since it's column wize

            // after transposition, the structure is lost
            m_imageLayout = ImageLayoutWHC(1, Inputs(0)->GetNumCols(), 1);
        }
    };

    template class TransposeNode<float>;
    template class TransposeNode<double>;

    // -----------------------------------------------------------------------
    // DiagonalNode -- extract diagonal elements of a square matrix
    // -----------------------------------------------------------------------

    template<class ElemType>
    class DiagonalNode : public ComputationNodeNonLooping<ElemType>, public NumInputs<1>
    {
        typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"Diagonal"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(DiagonalNode);
        DiagonalNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<DiagonalNode<ElemType>>(nodeP);
            }
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, true);

            m_imageLayout = ImageLayoutWHC(1, m_imageLayout.GetHeight(), 1);

            if (m_inputImageLayout.GetWidth() * m_inputImageLayout.GetNumChannels() != 1)
                fprintf(stderr, "WARNING: Diagonal operation cannot inherit image size information from its child. Image size info is lost.\n");
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

                    fprintf(stderr, "%ls[%lu, %lu]", child->NodeName().c_str(), child->FunctionValues().GetNumRows(), child->FunctionValues().GetNumCols());
                }

                fprintf(stderr, ")");
            }
        }

        virtual void Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);
            m_pMBLayout = nullptr;

            if (isFinalValidationPass && Inputs(0)->HasMBLayout())
                InvalidArgument("%ls %ls operation cannot operate on minibatch data (which have a layout)", NodeName().c_str(), OperationName().c_str());

            size_t dim = Inputs(0)->GetNumCols();
            if (isFinalValidationPass && dim != Inputs(0)->GetNumRows())
                InvalidArgument("%ls %ls operation requires a square matrix as its input.", NodeName().c_str(), OperationName().c_str());

            SetDims(1, dim);
            InferImageDimsFromInputs();
        }

        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            Inputs(0)->FunctionValues().AssignDiagonalValuesTo(FunctionValues());
#if NANCHECK
            FunctionValues().HasNan("Diagonal");
#endif
        }

        virtual void /*ComputationNodeNonLooping::*/ComputeInputPartialNonLooping(size_t /*inputIndex*/) override
        {
            Matrix<ElemType>& inputGradientValues = Inputs(0)->GradientValues();
            const Matrix<ElemType>& gradientValues = GradientValues();

            // BUGBUG: This should use the memshare mechanism
            Matrix<ElemType> diag(gradientValues.GetNumRows(), gradientValues.GetNumCols(), gradientValues.GetDeviceId());
            diag = gradientValues;
            diag.Resize(gradientValues.GetNumCols(), 1);

            inputGradientValues.SetValue(0);
            // BUGBUG: Must *add* to gradient!
            inputGradientValues.SetDiagonalValue(diag);
        }
    };

    template class DiagonalNode<float>;
    template class DiagonalNode<double>;

    // -----------------------------------------------------------------------
    // CosDistanceNode (left, right)
    // column-wise cos distance
    // TODO: Would it be useful to allow one of the two to be a single column?
    // -----------------------------------------------------------------------

    //The first matrix should be a vector regpresting the diagonal of a square matrix in the DiagTimes operation
    template<class ElemType>
    class CosDistanceNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"CosDistance"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(CosDistanceNode);
        CosDistanceNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            // functionValues, invNorm0, invNorm1 - output from the EvaluateNode() method
            // temp, rightTerm, leftTerm - temporary matrices
            if (inputIndex == 0)  //left derivative
                m_temp->AssignElementProductOf(*m_invNorm0, *m_invNorm0);
            else  //right derivative
                m_temp->AssignElementProductOf(*m_invNorm1, *m_invNorm1);

            m_temp->ElementMultiplyWith(ValueSlice(frameRange));
            m_rightTerm->SetValue(Inputs(inputIndex)->ValueSlice(frameRange));
            m_rightTerm->RowElementMultiplyWith(*m_temp);

            m_temp->AssignElementProductOf(*m_invNorm0, *m_invNorm1);
            m_leftTerm->SetValue(Inputs(1 - inputIndex)->ValueSlice(frameRange));
            m_leftTerm->RowElementMultiplyWith(*m_temp);

            *m_leftTerm -= *m_rightTerm;
            m_leftTerm->RowElementMultiplyWith(GradientSlice(frameRange));
            Inputs(inputIndex)->GradientSlice(frameRange) += *m_leftTerm;
        }


        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override 
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->ValueSlice(frameRange);
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange);

            m_invNorm0->AssignVectorNorm2Of(sliceInput0Value, true);
            m_invNorm0->AssignElementInverseOf(*m_invNorm0);

            m_invNorm1->AssignVectorNorm2Of(sliceInput1Value, true);
            m_invNorm1->AssignElementInverseOf(*m_invNorm1);

            sliceOutputValue.AssignInnerProductOf(sliceInput0Value, sliceInput1Value, true);
            sliceOutputValue.ElementMultiplyWith(*m_invNorm0);
            sliceOutputValue.ElementMultiplyWith(*m_invNorm1);
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);
            ValidateInferBinaryChildrenDims();

#if 0
            if (isFinalValidationPass && (Inputs(1)->GetNumRows() != Inputs(0)->GetNumRows() || (HasMBLayout() && (Inputs(1)->GetNumCols() != Inputs(0)->GetNumCols()))))
                LogicError("%ls %ls operation: The input dimensions do not match.", NodeName().c_str(), OperationName().c_str());
#endif

            SetDims(1, Inputs(1)->GetNumCols());
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs() 
        {
            InferImageDimsFromInput(0, false);

            m_imageLayout = ImageLayout();
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<CosDistanceNode<ElemType>>(nodeP);
                *node->m_invNorm0 = *m_invNorm0;
                *node->m_invNorm1 = *m_invNorm1;
                *node->m_leftTerm = *m_leftTerm;
                *node->m_rightTerm = *m_rightTerm;
                *node->m_temp = *m_temp;
            }
        }
        //request matrices needed to do node function value evaluation
        virtual void RequestMatricesBeforeEval(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeEval(matrixPool);
            RequestMatrixFromPool(m_invNorm0, matrixPool);
            RequestMatrixFromPool(m_invNorm1, matrixPool);
        }

        //request matrices that are needed for gradient computation
        virtual void RequestMatricesBeforeGradientComp(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeGradientComp(matrixPool);
            RequestMatrixFromPool(m_leftTerm, matrixPool);
            RequestMatrixFromPool(m_rightTerm, matrixPool);
            RequestMatrixFromPool(m_temp, matrixPool);
        }

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool)
        {
            Base::ReleaseMatricesAfterGradientComp(matrixPool);
            ReleaseMatrixToPool(m_invNorm0, matrixPool);
            ReleaseMatrixToPool(m_invNorm1, matrixPool);
            ReleaseMatrixToPool(m_leftTerm, matrixPool);
            ReleaseMatrixToPool(m_rightTerm, matrixPool);
            ReleaseMatrixToPool(m_temp, matrixPool);
        }
private:
        // invNorm nodes tranfer data between EvaluateThisNode and ComputeInputPartial
        shared_ptr<Matrix<ElemType>> m_invNorm0;
        shared_ptr<Matrix<ElemType>> m_invNorm1;
        // the rest are temporaries, values don't need to be maintained
        shared_ptr<Matrix<ElemType>> m_leftTerm;
        shared_ptr<Matrix<ElemType>> m_rightTerm;
        shared_ptr<Matrix<ElemType>> m_temp;
    };

    template class CosDistanceNode<float>; 
    template class CosDistanceNode<double>;

    // -----------------------------------------------------------------------
    // KhatriRaoProductNode (left, right)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class KhatriRaoProductNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"KhatriRaoProduct"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(KhatriRaoProductNode);
        KhatriRaoProductNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange);

            if (inputIndex == 0)  //left derivative
            {
                Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientSlice(frameRange);
                Matrix<ElemType> sliceInput1Value = Inputs(1)->ValueSlice(frameRange);

                sliceInput0Grad.AddColumnReshapeProductOf(sliceOutputGrad, sliceInput1Value, false);
            }
            else  //right derivative
            {
                Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange);
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientSlice(frameRange);

                sliceInput1Grad.AddColumnReshapeProductOf(sliceOutputGrad, sliceInput0Value, true);
            }
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override  
        {
            ValueSlice(frameRange).AssignKhatriRaoProductOf(Inputs(0)->ValueSlice(frameRange), Inputs(1)->ValueSlice(frameRange));
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();

            //support automatic dimension inference for learnable parameters
            size_t rows0 = Inputs(0)->GetNumRows(), cols0 = Inputs(0)->GetNumCols();
            size_t rows1 = Inputs(1)->GetNumRows(), cols1 = Inputs(1)->GetNumCols();

            if (cols0 == 0 && cols1 != 0)
                ValidateInferChildDims(0, rows0, cols1);

            if (cols0 != 0 && cols1 == 0)
                ValidateInferChildDims(1, rows1, cols0);

            if (isFinalValidationPass && !HasMBLayout() && Inputs(1)->GetNumCols() != Inputs(0)->GetNumCols())
                LogicError("The Matrices should have same number of columns.");

            SetDims(rows0 * rows1, Inputs(0)->GetNumCols());
        }

        virtual void InferImageDimsFromInputs()  
        {
            //since it's symmetrical any one of the input may be the true input. 
            //since we dont' use the input image size info in the operation, the input part doesn't matter.
            InferImageDimsFromInput(1, false); 

            //after KhatriRaoProduct the structure is lost
            m_imageLayout = ImageLayoutWHC(1, m_functionValues->GetNumRows(), 1);
        }
    };

    template class KhatriRaoProductNode<float>; 
    template class KhatriRaoProductNode<double>;

    // -----------------------------------------------------------------------
    // CosDistanceWithNegativeSamplesNode (left, right, shift, neg)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class CosDistanceWithNegativeSamplesNode : public ComputationNode<ElemType>, public NumInputs<4>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"CosDistanceWithNegativeSamples"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(CosDistanceWithNegativeSamplesNode);
        CosDistanceWithNegativeSamplesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        void ComputeInputPartialMap(const size_t inputIndex)
        {
            if (inputIndex > 1)
                InvalidArgument("CosDistanceWithNegativeSamples operation only takes grdients on the first two inputs.");

            ComputeInputPartialS(inputIndex, *m_invNorm0, *m_invNorm1, FunctionValues(), *m_temp, *m_rightTerm, *m_leftTerm, *m_invNormSquare, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), Inputs(3)->FunctionValues(), Inputs(inputIndex)->GradientValues(), GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->ValueSlice(frameRange);
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange);
            Matrix<ElemType> sliceInputGrad = Inputs(inputIndex)->GradientSlice(frameRange);
            Matrix<ElemType> sliceThisGrad = GradientSlice(frameRange);

            ComputeInputPartialS(inputIndex, *m_invNorm0, *m_invNorm1, sliceOutputValue, *m_temp, *m_rightTerm, *m_leftTerm, *m_invNormSquare, sliceInput0Value, sliceInput1Value, Inputs(2)->FunctionValues(), Inputs(3)->FunctionValues(), sliceInputGrad, sliceThisGrad);
        }

        // functionValues, invNorm0, invNorm1 - output from the EvaluateNode() method
        // temp, rightTerm, leftTerm - temporary matrices
        // in0, in1, in2, in3 - input functionValues from other nodes
        // inputGradientValues(x) - gradients to update, where x matches inputIndex
        /*TODO: merge with call site*/void ComputeInputPartialS(const size_t inputIndex, const Matrix<ElemType>& invNorm0, const Matrix<ElemType>& invNorm1, const Matrix<ElemType>& functionValues,
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

        void EvaluateThisNodeMap()    // TODO: This is a stop-gap; in most cases, we should just be able to delete this (but need to review one by one)
        {
            EvaluateThisNodeS(*m_invNorm0, *m_invNorm1, FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), Inputs(3)->FunctionValues(), *m_leftTerm, *m_rightTerm);
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            //if (frameRange.IsAllFrames()) { EvaluateThisNodeMap(); return; }
            Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->ValueSlice(frameRange);
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange);

            EvaluateThisNodeS(*m_invNorm0, *m_invNorm1, sliceOutputValue, sliceInput0Value, sliceInput1Value, Inputs(2)->FunctionValues(), Inputs(3)->FunctionValues(), *m_leftTerm, *m_rightTerm);
        }

        /*TODO: merge with call site*/void EvaluateThisNodeS(Matrix<ElemType>& invNorm0, Matrix<ElemType>& invNorm1, Matrix<ElemType>& functionValues, Matrix<ElemType>& in0, Matrix<ElemType>& in1, Matrix<ElemType>& in2, Matrix<ElemType>& in3, Matrix<ElemType>& leftTermTemp, Matrix<ElemType>& rightTermTemp)
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

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            //if dimension is missing make the two operatants to have same size
            // TODO: use a for loop??
            size_t index = 0;
            {
                size_t rows = Inputs(index)->GetNumRows() == 0 ? Inputs(1 - index)->GetNumRows() : Inputs(index)->GetNumRows();
                size_t cols = Inputs(index)->GetNumCols() == 0 ? Inputs(1 - index)->GetNumCols() : Inputs(index)->GetNumCols();
                ValidateInferChildDims(index, rows, cols);
            }

            index = 1;
            {
                size_t rows = Inputs(index)->GetNumRows() == 0 ? Inputs(1 - index)->GetNumRows() : Inputs(index)->GetNumRows();
                size_t cols = Inputs(index)->GetNumCols() == 0 ? Inputs(1 - index)->GetNumCols() : Inputs(index)->GetNumCols();
                ValidateInferChildDims(index, rows, cols);
            }

            if (isFinalValidationPass &&
                (Inputs(1)->GetNumRows() != Inputs(0)->GetNumRows() ||
                 (!Inputs(1)->GetMBLayout() && Inputs(1)->GetNumCols() != Inputs(0)->GetNumCols())))
            {
                LogicError("The Matrix dimension in the %ls %ls operation does not match.", NodeName().c_str(), OperationName().c_str());
            }

            // input(2) is shift, input(3) is the #neg
            size_t negNumber = (size_t)Inputs(3)->Get00Element();

            SetDims(negNumber + 1, Inputs(1)->GetNumCols());
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            m_imageLayout = ImageLayout();
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<CosDistanceWithNegativeSamplesNode<ElemType>>(nodeP);
                *node->m_invNorm0 = *m_invNorm0;
                *node->m_invNorm1 = *m_invNorm1;
                *node->m_invNormSquare = *m_invNormSquare;
                *node->m_leftTerm = *m_leftTerm;
                *node->m_rightTerm = *m_rightTerm;
                *node->m_temp = *m_temp;
            }
        }
        //request matrices needed to do node function value evaluation
        virtual void RequestMatricesBeforeEval(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeEval(matrixPool);
            RequestMatrixFromPool(m_invNorm0, matrixPool);
            RequestMatrixFromPool(m_invNorm1, matrixPool);
            RequestMatrixFromPool(m_leftTerm, matrixPool);
            RequestMatrixFromPool(m_rightTerm, matrixPool);
        }

        //request matrices that are needed for gradient computation
        virtual void RequestMatricesBeforeGradientComp(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeGradientComp(matrixPool);
            RequestMatrixFromPool(m_invNormSquare, matrixPool);
            RequestMatrixFromPool(m_temp, matrixPool);
        }

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool)
        {
            Base::ReleaseMatricesAfterGradientComp(matrixPool);
            ReleaseMatrixToPool(m_invNorm0, matrixPool);
            ReleaseMatrixToPool(m_invNorm1, matrixPool);
            ReleaseMatrixToPool(m_leftTerm, matrixPool);
            ReleaseMatrixToPool(m_rightTerm, matrixPool);
            ReleaseMatrixToPool(m_invNormSquare, matrixPool);
            ReleaseMatrixToPool(m_temp, matrixPool);
        }
private:
        // invNorm nodes tranfer data between EvaluateThisNode and ComputeInputPartial
        shared_ptr<Matrix<ElemType>> m_invNorm0;
        shared_ptr<Matrix<ElemType>> m_invNorm1;
        shared_ptr<Matrix<ElemType>> m_leftTerm;
        shared_ptr<Matrix<ElemType>> m_rightTerm;
        // the rest are temporaries, values don't need to be maintained
        shared_ptr<Matrix<ElemType>> m_invNormSquare;
        shared_ptr<Matrix<ElemType>> m_temp;
    };

    template class CosDistanceWithNegativeSamplesNode<float>;
    template class CosDistanceWithNegativeSamplesNode<double>;

}}}
