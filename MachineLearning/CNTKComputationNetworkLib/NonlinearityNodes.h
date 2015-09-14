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

    // =======================================================================
    // NonlinearityNode -- abstract base class that holds what's shared between non-linearity nodes like Sigmoid
    // =======================================================================

    // NOTE:
    // This represents a partially failed attempt to unify this.
    // The idea is that the shared class manages everything, and the derived classes only implement two virtual functions,
    // one for Eval and one for Partial.
    // However, it turned out that all those functions have slightly different signatures. Grmpf.
    // So we will either have to fix the virtual function to be the superset, or abstract on a different level.

    // shared base for all elemen-twise non-linearities
    template<class ElemType>
    class NonlinearityNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) = 0;
        NonlinearityNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_gradient(deviceId)
        { }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                InvalidArgument("Nonlinearities only have one input.");
            ComputeInputPartialV(m_gradient, Inputs(0)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex != 0)
                InvalidArgument("Nonlinearities only have one input.");

            // TODO: this seems always the same pattern. This belongs into a base slice-extractor function.
            //       We should also unify these two functions into one that decides 1 frame or all frames at runtime... through the slice-extractor function itself.
            //       For now we could define ALL_SAMPLES e.g. as SIZE_MAX.
            //       GetGradientSlice(), GetInputSlice() or something.
            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().FrameSlice(frameRange, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            // why GradientValues() but m_functionValues below and not FunctionValues()?

            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialV(m_gradient, sliceInputValue, sliceInputGrad, sliceOutputGrad);
        }

        virtual void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) = 0;

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeV(m_functionValues, Inputs(0)->FunctionValues());   // actual work is done in derived class depending on what non-linearity it is
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().FrameSlice(frameRange, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeV(sliceOutputValue, sliceInputValue);
        }

        virtual void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues) = 0;

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 1) 
                LogicError("Nonlinearity operations should have one input.");

            if (Inputs(0)->FunctionValues().HasNoElements())
                LogicError("Nonlinearity operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_gradient.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            InferImageDimsFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            Base::MoveMatricesToDevice(deviceId);
            m_gradient.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<NonlinearityNode<ElemType>>(nodeP);
                node->m_gradient = m_gradient;
            }
        }

    protected:
        Matrix<ElemType> m_gradient;
    };

#define UsingNonlinearityNodeMembers UsingComputationNodeMembers; using Base::m_gradient

    // =======================================================================
    // RectifiedLinearNode -- ReLU non-linearity
    // =======================================================================

    template<class ElemType>
    class RectifiedLinearNode : public NonlinearityNode<ElemType>
    {
        typedef NonlinearityNode<ElemType> Base; UsingNonlinearityNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        RectifiedLinearNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() {return L"RectifiedLinear";} 

        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
        {
            gradient.AssignLinearRectifierDerivativeOf(inputFunctionValues);
#if DUMPOUTPUT
            inputGradientValues.Print("RecitifiedLinearNode-Partial-in");
#endif
            inputGradientValues.AddElementProductOf(gradientValues, gradient); 
#if DUMPOUTPUT
            inputGradientValues.Print("RecitifiedLinearNode-Partial-out");
#endif
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.AssignTruncateBottomOf(inputFunctionValues, 0);
#if NANCHECK
            functionValues.HasNan("RectifiedLinear");
#endif
#if DUMPOUTPUT
            functionValues.Print("RectifiedLinearNode");
#endif
        }
    };

    template class RectifiedLinearNode<float>; 
    template class RectifiedLinearNode<double>;

    // =======================================================================
    // SigmoidNode -- sigmoid non-linearity
    // =======================================================================

    template<class ElemType>
    class SigmoidNode : public NonlinearityNode<ElemType>
    {
        typedef NonlinearityNode<ElemType> Base; UsingNonlinearityNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        SigmoidNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() {return L"Sigmoid";} 

        // we should get rid of this code dup, need to unify the -V functions
        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                InvalidArgument("Sigmoid only has one input.");
            ComputeInputPartialS(m_gradient, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex != 0)
                InvalidArgument("Sigmoid only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradient, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
        }

        // should be:
        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) { gradient; inputFunctionValues;  inputGradientValues;  gradientValues;  LogicError("wrong signature :( need to unify code more"); }
        // but is:
        /*virtual*/ void ComputeInputPartialS(Matrix<ElemType>& gradient, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)
        {
            gradient.AssignSigmoidDerivativeOf(functionValues);
            inputGradientValues.AddElementProductOf(gradientValues, gradient);
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.AssignSigmoidOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Sigmoid");
#endif
        }
    };

    template class SigmoidNode<float>; 
    template class SigmoidNode<double>;

    // =======================================================================
    // TanhNode -- tanh non-linearity
    // =======================================================================

    template<class ElemType>
    class TanhNode : public NonlinearityNode<ElemType>
    {
        typedef NonlinearityNode<ElemType> Base; UsingNonlinearityNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        TanhNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() {return L"Tanh";}

        // TODO: unify signature & get rid of code dup
        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                InvalidArgument("Tanh only has one input.");
            ComputeInputPartialS(m_gradient, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex != 0)
                InvalidArgument("Tanh only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradient, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
        }

        // should be:
        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) { gradient; inputFunctionValues;  inputGradientValues;  gradientValues;  LogicError("wrong signature :( need to unify code more"); }
        // but is:
        /*virtual*/ void ComputeInputPartialS(Matrix<ElemType>& gradient, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)
        {
            gradient.AssignElementProductOf(functionValues, functionValues); // v .* v
            gradient.AssignDifferenceOf(1, gradient); // 1-v^2

            inputGradientValues.AddElementProductOf(gradientValues, gradient); // += d .* ((1-v) .* v))
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.AssignTanhOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Tanh");
#endif
        }
    };

    template class TanhNode<float>; 
    template class TanhNode<double>;

    // =======================================================================
    // LogNode -- component-wise log() of input
    // =======================================================================

    template<class ElemType>
    class LogNode : public NonlinearityNode<ElemType>
    {
        typedef NonlinearityNode<ElemType> Base; UsingNonlinearityNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        LogNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"Log"; }

        // TODO: get rid of code dup
        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                InvalidArgument("Log only has one input.");
            ComputeInputPartialS(m_gradient, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex != 0)
                InvalidArgument("Log only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradient, sliceInputGrad, sliceInputValue, sliceOutputGrad);
        }

        // should be:
        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) { gradient; inputFunctionValues;  inputGradientValues;  gradientValues;  LogicError("wrong signature :( need to unify code more"); }
        // but is:
        /*virtual*/ void ComputeInputPartialS(Matrix<ElemType>& gradient, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& gradientValues)
        {
            gradient.AssignElementInverseOf(inputFunctionValues); // 1/x (x is input to log(x))

            inputGradientValues.AddElementProductOf(gradientValues, gradient);
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)
        {
            functionValues.AssignLogOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Log");
#endif
        }
    };

    template class LogNode<float>;
    template class LogNode<double>;

    // =======================================================================
    // ExpNode -- component-wise exp() of input
    // =======================================================================

    template<class ElemType>
    class ExpNode : public NonlinearityNode<ElemType>
    {
        typedef NonlinearityNode<ElemType> Base; UsingNonlinearityNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        ExpNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"Exp"; }

        // TODO: get rid of code dup
        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                InvalidArgument("Exp only has one input.");
            ComputeInputPartialS(m_gradient, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex != 0)
                InvalidArgument("Exp only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradient, sliceInputGrad, sliceInputValue, sliceOutputGrad);
        }

        // should be:
        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) { gradient; inputFunctionValues;  inputGradientValues;  gradientValues;  LogicError("wrong signature :( need to unify code more"); }
        // but is:
        /*virtual*/ void ComputeInputPartialS(Matrix<ElemType>& gradient, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& gradientValues)
        {
            gradient.AssignExpOf(inputFunctionValues); // Exp(x) is its own partial
            inputGradientValues.AddElementProductOf(gradientValues, gradient);
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)
        {
            functionValues.AssignExpOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Exp");
#endif
        }
    };

    template class ExpNode<float>;
    template class ExpNode<double>;

    // =======================================================================
    // CosineNode -- component-wise cos() of input
    // =======================================================================

    template<class ElemType>
    class CosineNode : public NonlinearityNode<ElemType>
    {
        typedef NonlinearityNode<ElemType> Base; UsingNonlinearityNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        CosineNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Cosine";}

        // TODO: code dup
        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                InvalidArgument("Cosine only has one input.");
            ComputeInputPartialS(m_gradient, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex != 0)
                InvalidArgument("Cosine only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradient, sliceInputGrad, sliceInputValue, sliceOutputGrad);
        }


        // should be:
        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) { gradient; inputFunctionValues;  inputGradientValues;  gradientValues;  LogicError("wrong signature :( need to unify code more"); }
        // but is:
        /*virtual*/ void ComputeInputPartialS(Matrix<ElemType>& gradient, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& gradientValues)
        {
            gradient.AssignNegativeSineOf(inputFunctionValues); // -sin(x) (x is input to Cosine(x))
            inputGradientValues.AddElementProductOf(gradientValues, gradient);
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.AssignCosineOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Cosine");
#endif
        }
    };

    template class CosineNode<float>; 
    template class CosineNode<double>;

    // =======================================================================
    // SoftmaxNode -- soft-max over input vector(s)
    // =======================================================================

    //we assume it's  column-wise by default
    //the derivative will increase the Matrix<ElemType> size to the power of column size and should not be used.
    template<class ElemType>
    class SoftmaxNode : public NonlinearityNode<ElemType>
    {
        typedef NonlinearityNode<ElemType> Base; UsingNonlinearityNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        SoftmaxNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNode<ElemType>(deviceId, name),
            m_diff(deviceId)
        { }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() {return L"Softmax";}

        // TODO: code dup
        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                InvalidArgument("Softmax only has one input.");
            ComputeInputPartialS(m_gradient, m_diff, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex != 0)
                InvalidArgument("Softmax only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradient, m_diff, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
        }

        // should be:
        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) { gradient; inputFunctionValues;  inputGradientValues;  gradientValues;  LogicError("wrong signature :( need to unify code more"); }
        // but is:
        /*virtual*/ void ComputeInputPartialS(Matrix<ElemType>& gradient, Matrix<ElemType>& diff, Matrix<ElemType>& inputGradientValues,
            const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)
        {
            gradient.AssignInnerProductOf(gradientValues, functionValues, true);
            diff.AssignDifferenceOf(gradientValues, gradient);

            inputGradientValues.AddElementProductOf(diff, functionValues);
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            // need to resize
            size_t r = Inputs(0)->FunctionValues().GetNumRows(), c = Inputs(0)->FunctionValues().GetNumCols();
            // note: I moved this test before sliceInputValue=..., assuming it will not be affected by the assignment
            if (m_functionValues.GetNumCols() != c ||
                m_functionValues.GetNumRows() != r)
                m_functionValues.Resize(r, c);
            NonlinearityNode<ElemType>::EvaluateThisNode(frameRange);
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.AssignLogSoftmaxOf(inputFunctionValues, true);
            functionValues.InplaceExp();
#if NANCHECK
            functionValues.HasNan("SoftMax");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 1) 
                LogicError("SoftmaxNode operation should have one input.");

            if (Inputs(0)->FunctionValues().HasNoElements())
                LogicError("SoftmaxNode operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            // TODO: differs from base in that it does not resize the gradient--why?
            InferImageDimsFromInputs(); 
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            NonlinearityNode<ElemType>::MoveMatricesToDevice(deviceId);
            m_diff.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<SoftmaxNode<ElemType>>(nodeP);
                node->m_diff = m_diff;
            }
        }
private:
        Matrix<ElemType> m_diff;
    };

    template class SoftmaxNode<float>; 
    template class SoftmaxNode<double>;

    // =======================================================================
    // LogSoftmaxNode -- log of soft-max over input vector(s)
    // =======================================================================

    template<class ElemType>
    class LogSoftmaxNode : public NonlinearityNode<ElemType>
    {
        typedef NonlinearityNode<ElemType> Base; UsingNonlinearityNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        LogSoftmaxNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNode<ElemType>(deviceId, name)
        { }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"LogSoftmax"; }

        // TODO: code dup
        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                InvalidArgument("Softmax only has one input.");
            ComputeInputPartialS(m_gradient, m_softmax, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());
        }


        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex != 0)
                InvalidArgument("Softmax only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradient, m_softmax, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
        }

        // should be:
        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) { gradient; inputFunctionValues;  inputGradientValues;  gradientValues;  LogicError("wrong signature :( need to unify code more"); }
        // but is:
        /*virtual*/ void ComputeInputPartialS(Matrix<ElemType>& gradient, Matrix<ElemType>& softmax, Matrix<ElemType>& inputGradientValues,
            const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)
        {
            softmax.AssignExpOf(functionValues);
            Matrix<ElemType>::VectorSum(gradientValues, gradient, true);
            softmax.RowElementMultiplyWith(gradient);
            Matrix<ElemType>::AddScaledDifference(1.0, gradientValues, softmax, inputGradientValues);
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)
        {
            functionValues.AssignLogSoftmaxOf(inputFunctionValues, true);
#if NANCHECK
            functionValues.HasNan("LogSoftMax");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 1)
                LogicError("LogSoftmaxNode operation should have one input.");

            if (Inputs(0)->FunctionValues().HasNoElements())
                LogicError("LogSoftmaxNode operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            // differs from base in that it does not resize the gradient
            InferImageDimsFromInputs();
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            Base::MoveMatricesToDevice(deviceId);
            m_softmax.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<LogSoftmaxNode<ElemType>>(nodeP);
                node->m_softmax = m_softmax;
            }
        }
private:
        Matrix<ElemType> m_softmax;
    };

    template class LogSoftmaxNode<float>;
    template class LogSoftmaxNode<double>;

    // =======================================================================
    // GMMLogLikelihoodNode -- GMM log LL over input vector(s)
    // BUGBUG: This seems to just to a single Gaussian?
    // =======================================================================

    //calculates: the log likelihood of a feature given GMM parameters
    template<class ElemType>
    class GMMLogLikelihoodNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        GMMLogLikelihoodNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_prior(deviceId), m_normedDeviation(deviceId), m_normedDeviationVectors(deviceId),
            m_stddev(deviceId), m_posterior(deviceId), m_temp(deviceId)
        { }
virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"GMMLogLikelihood"; }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            switch (inputIndex)
            {
            case 0:
                ComputeInputPartialUnnormedPrior(Inputs(0)->GradientValues(), m_gradientValues, m_prior, m_posterior, m_temp);
                break;
            case 1:
                ComputeInputPartialMean(Inputs(1)->GradientValues(), m_gradientValues, m_normedDeviationVectors, m_posterior, m_temp);
                break;
            case 2:
                ComputeInputPartialLogStddev(Inputs(2)->GradientValues(), m_gradientValues, m_normedDeviation, m_posterior, m_temp);
                break;
            case 3:
                ComputeInputPartialFeature(Inputs(3)->GradientValues(), m_gradientValues, m_normedDeviationVectors, m_posterior, m_temp);
                break;
            default:
                InvalidArgument("GMMLogLikelihoodNode only takes four inputs.");
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            //get the right slice 
            const size_t colsPrior = Inputs(0)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceGradientValue = m_gradientValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> slicePosterior = m_posterior.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                
            switch (inputIndex)
            {
            case 0:
                {
                    if (colsPrior == 1)
                        ComputeInputPartialUnnormedPrior(Inputs(0)->GradientValues(), sliceGradientValue, m_prior, slicePosterior, m_temp);
                    else
                    {
                        Matrix<ElemType> sliceUnnormedPriorGradient = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                        Matrix<ElemType> slicePrior = m_prior.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                        ComputeInputPartialUnnormedPrior(sliceUnnormedPriorGradient, sliceGradientValue, slicePrior, slicePosterior, m_temp);
                    }
                }
                break;
            case 1:
                {
                      Matrix<ElemType> sliceNormedDeviationVectors = m_normedDeviationVectors.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                      if (colsPrior == 1)
                        ComputeInputPartialMean(Inputs(1)->GradientValues(), sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, m_temp);
                    else
                    {
                        Matrix<ElemType> sliceMeanGradient = Inputs(1)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                        ComputeInputPartialMean(sliceMeanGradient, sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, m_temp);
                    }
                }
                break;
            case 2:
                {
                    Matrix<ElemType> sliceNormedDeviation = m_normedDeviation.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                    if (colsPrior == 1)
                        ComputeInputPartialLogStddev(Inputs(2)->GradientValues(), sliceGradientValue, sliceNormedDeviation, slicePosterior, m_temp);
                    else
                    {
                        Matrix<ElemType> sliceLotStddevGradient = Inputs(2)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                        ComputeInputPartialLogStddev(sliceLotStddevGradient, sliceGradientValue, sliceNormedDeviation, slicePosterior, m_temp);
                    }
                }
                break;
            case 3:
                {
                    Matrix<ElemType> sliceNormedDeviationVectors = m_normedDeviationVectors.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceFeatureGradient = Inputs(3)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                    ComputeInputPartialFeature(sliceFeatureGradient, sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, m_temp);
                }
                break;
            default:
                InvalidArgument("GMMLogLikelihoodNode criterion only takes four inputs.");
            }
        }

        static void WINAPI ComputeInputPartialUnnormedPrior(Matrix<ElemType>& unnormedPriorGradientValues, const Matrix<ElemType>& gradientValues,
            const Matrix<ElemType>& prior, const Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
        {
            temp.AssignDifferenceOf(posterior, prior);
            temp.RowElementMultiplyWith(gradientValues);
            if (prior.GetNumCols() == posterior.GetNumCols())
                unnormedPriorGradientValues += temp; 
            else if (prior.GetNumCols() == 1)
                Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(posterior.GetNumCols(), 1, unnormedPriorGradientValues.GetDeviceId()), false, unnormedPriorGradientValues);
            else
                RuntimeError("GMMLogLikelihoodNode: UnnormedPrior should either have same number of columns as the features or have only one column.");
        }

        static void WINAPI ComputeInputPartialMean(Matrix<ElemType>& meanGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& normedDeviationVectors,
            Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
        {
            size_t numComponent = posterior.GetNumRows(); 
            size_t numSamples = posterior.GetNumCols();
            size_t featureSize = normedDeviationVectors.GetNumRows() / numComponent;

            temp.SetValue(normedDeviationVectors); //recall normedDeviationVectors <-- (x-u_c)/(stddev^2)
            temp.Reshape(featureSize, numSamples* numComponent);

            posterior.Reshape(1, numSamples* numComponent);
            temp.RowElementMultiplyWith(posterior); //temp <-- posterior * (x-u_c)/(stddev^2)

            posterior.Reshape(numComponent, numSamples);  //reshape back
            temp.Reshape(featureSize * numComponent, numSamples); //reshape back

            temp.RowElementMultiplyWith(gradientValues);

            if (numSamples == meanGradientValues.GetNumCols())
                meanGradientValues += temp;
            else if (meanGradientValues.GetNumCols() == 1)
                Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(numSamples, 1, meanGradientValues.GetDeviceId()), false, meanGradientValues);
            else
                RuntimeError("GMMLogLikelihoodNode: stddev should either have same number of columns as the features or have only one column.");
        }

        static void WINAPI ComputeInputPartialLogStddev(Matrix<ElemType>& logStddevGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& normedDeviation,
            const Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
        {
            size_t numComponent = posterior.GetNumRows();
            size_t numSamples = posterior.GetNumCols();

            temp.AssignDifferenceOf(normedDeviation, (ElemType)numComponent);
            temp.ElementMultiplyWith(posterior);
            temp.RowElementMultiplyWith(gradientValues);
            if (logStddevGradientValues.GetNumCols() == numSamples)
                logStddevGradientValues += temp;
            else if (logStddevGradientValues.GetNumCols() == 1)
                Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(numSamples, 1, logStddevGradientValues.GetDeviceId()), false, logStddevGradientValues);
            else
                RuntimeError("GMMLogLikelihoodNode: stddev should either have same number of columns as the features or have only one column.");
        }

        static void WINAPI ComputeInputPartialFeature(Matrix<ElemType>& featureGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& normedDeviationVectors,
            Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
        {
            size_t numComponent = posterior.GetNumRows();
            size_t numSamples = posterior.GetNumCols();
            size_t featureSize = normedDeviationVectors.GetNumRows() / numComponent;

            temp.SetValue(normedDeviationVectors);
            temp *= -1;
            temp.Reshape(featureSize, numSamples* numComponent);
            posterior.Reshape(1, numSamples* numComponent);
            temp.RowElementMultiplyWith(posterior);

            posterior.Reshape(numComponent, numSamples);
            temp.Reshape(featureSize * numComponent, numSamples);
            temp.RowElementMultiplyWith(gradientValues);

            for (int i = 0; i < numComponent; i++)
                featureGradientValues.AddWithRowSliceValuesOf(temp, i*featureSize, featureSize);
        }

        virtual void SetFunctionAndGradientSize(const int numSamples)
        {
            ComputationNode<ElemType>::SetFunctionAndGradientSize(numSamples);

            size_t numComponents = Inputs(0)->FunctionValues().GetNumRows();
            size_t colsPrior = Inputs(0)->FunctionValues().GetNumCols();
            //size_t numSamples = Inputs(3)->FunctionValues().GetNumCols();
            size_t featureSize = Inputs(3)->FunctionValues().GetNumRows();

            m_prior.Resize(numComponents, colsPrior);
            m_stddev.Resize(numComponents, colsPrior);
            m_normedDeviation.Resize(numComponents, numSamples);
            m_normedDeviationVectors.Resize(numComponents * featureSize, numSamples);
            m_posterior.Resize(numComponents, numSamples);
        }

        //input0=unnormedPrior, input1=mean, input2=logstddev, input3=feature
        virtual void EvaluateThisNode()
        {
            // all internal matrices will be automatically resized since all of them are assigned to a value so no resize is needed here.
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), Inputs(3)->FunctionValues(), 
                m_prior, m_stddev, m_normedDeviationVectors, m_normedDeviation, m_posterior, m_temp);
        }

        //input0=unnormedPrior, input1=mean, input2=logstddev, input3=feature
        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            size_t colsPrior = Inputs(0)->FunctionValues().GetNumCols();
            size_t numSamples = Inputs(3)->FunctionValues().GetNumCols();

            //get the right slice 
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceFeature = Inputs(3)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceNormedDeviation = m_normedDeviation.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceNormedDeviationVectors = m_normedDeviationVectors.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> slicePosterior = m_posterior.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (colsPrior == 1)
            {
                EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), sliceFeature,
                    m_prior, m_stddev, sliceNormedDeviationVectors, sliceNormedDeviation, slicePosterior, m_temp);
            }
            else if (colsPrior == numSamples)
            {
                Matrix<ElemType> sliceUnnormedPrior = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceMean = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceLogstddev = Inputs(2)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                Matrix<ElemType> slicePrior = m_prior.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceStddev = m_stddev.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, sliceUnnormedPrior, sliceMean, sliceLogstddev, sliceFeature,
                    slicePrior, sliceStddev, sliceNormedDeviationVectors, sliceNormedDeviation, slicePosterior, m_temp);
            }
            else  //should not reach the code since validation should fail already
                RuntimeError("GMMLogLikelihoodNode: UnnormedPrior should either have same number of columns as the features or have only one column.");
        }

        //input0=unnormedPrior, input1=mean, input2=logstddev, input3=feature
        //If we want to speed up we need to replace following code with a several specialized GPU functions
        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& unnormedPrior, const Matrix<ElemType>& mean,  Matrix<ElemType>& logstddev,
            const Matrix<ElemType>& feature, Matrix<ElemType>& prior, Matrix<ElemType>& stddev, Matrix<ElemType>& normedDeviationVectors,
            Matrix<ElemType>& normedDeviation, Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
        {
            int numComponent = unnormedPrior.GetNumRows();
            size_t numSamples = feature.GetNumCols();
            size_t featureDim = feature.GetNumRows();

            //compute prior which is softmax of unnormedPrior
            prior.AssignLogSoftmaxOf(unnormedPrior, true);  //log prior

            prior.InplaceExp();

            //compute stddev
            stddev.AssignExpOf(logstddev);

#if DUMPOUTPUT
            unnormedPrior.Print("unnormedPrior", 0, min(5, unnormedPrior.GetNumRows() - 1), 0, min(10, unnormedPrior.GetNumCols() - 1));
            mean.Print("mean", 0, min(5, mean.GetNumRows() - 1), 0, min(10, mean.GetNumCols() - 1));
            logstddev.Print("logstddev", 0, min(5, logstddev.GetNumRows() - 1), 0, min(10, logstddev.GetNumCols() - 1));

            prior.Print("prior", 0, min(5, prior.GetNumRows() - 1), 0, min(10, prior.GetNumCols() - 1));
            stddev.Print("stddev", 0, min(5, stddev.GetNumRows() - 1), 0, min(10, stddev.GetNumCols() - 1));
#endif

            //compute normedDeviation <-- ||x-u_c||^2/(stddev^2)
            normedDeviationVectors.AssignRepeatOf(feature, numComponent, 1);
            normedDeviationVectors -= mean; //each column of the mean has multiple mean components
            normedDeviationVectors.Reshape(featureDim, numSamples* numComponent);  //now each column is feature-mean_i

            normedDeviation.AssignVectorNorm2Of(normedDeviationVectors, true);
            normedDeviation ^= 2;
            temp.AssignRepeatOf(stddev, 1, numSamples / stddev.GetNumCols());  //stddev.GetNumCols() is either 1 or =numSamples
            temp.Reshape(1, temp.GetNumElements());  //one stddev value for each component for each sample
            temp ^= 2;
            normedDeviation.ElementDivideBy(temp);  //normedDeviation and temp have same dim (1, numSamples* numComponent)

            //compute  normedDeviationVectors <-- (x-u_c)/(stddev^2)
            normedDeviationVectors.RowElementDivideBy(temp);  //divide twice
            normedDeviationVectors.Reshape(featureDim*numComponent, numSamples);  //reshape back

            //compute per-component likelihood
            posterior.AssignProductOf(-0.5f, normedDeviation); //posterior  <-- -||x-u_c||^2/(stddev^2)/2 and in (1, numSamples* numComponent) dim
            temp.InplaceLog();
            temp *= ((ElemType)numComponent / 2.0f); //temp <-- stddev^c and in (1, numSamples* numComponent) dim
            posterior -= temp;  // posterior  <-- exp[-||x-u_c||^2/(stddev^2)/2]/(stddev^c)
            posterior -= (ElemType)(numComponent / 2.0f*log(TWO_PI)); //likelihood for each component and sample is now computed and stored in posterior
            posterior.InplaceExp(); //posterior  <-- exp(-||x-u_c||^2/(stddev^2)/2)

            normedDeviation.Reshape(numComponent, numSamples);  //reshape back
            posterior.Reshape(numComponent, numSamples);  //reshape back

            //compute posterior <-- prior_i * likelihood_i
            if (unnormedPrior.GetNumCols() == numSamples)  //each sample has different prior
                posterior.ElementMultiplyWith(prior);
            else  //all samples share the same prior
                posterior.ColumnElementMultiplyWith(prior);

            //compute GMM log-likelihood
            Matrix<ElemType>::Multiply(ConstOnes(1, numComponent, posterior.GetDeviceId()), false, posterior, false, functionValues);  //functionValues <-- total likelihood
            posterior.RowElementDivideBy(functionValues); //posterior <-- per-comp likelihood / total likelihood
            functionValues.InplaceLog(); //log likelihood

#if DUMPOUTPUT
            temp.Print("temp", 0, min(5, temp.GetNumRows() - 1), 0, min(10, temp.GetNumCols() - 1));
            normedDeviation.Print("normedDeviation", 0, min(5, normedDeviation.GetNumRows() - 1), 0, min(10, normedDeviation.GetNumCols() - 1));

            posterior.Print("posterior", 0, min(5, posterior.GetNumRows() - 1), 0, min(10, posterior.GetNumCols() - 1));
            functionValues.Print("functionValues", 0, min(5, functionValues.GetNumRows() - 1), 0, min(10, functionValues.GetNumCols() - 1));

            functionValues.Print("GMMLogLikelihoodNode");
#endif

#if NANCHECK
            functionValues.HasNan("GMMLogLikelihood");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 4)
                LogicError("GMMLogLikelihoodNode requires four inputs.");

            size_t rows[4], cols[4];
            for (int i = 0; i < 4; i++)
            {
                rows[i] = Inputs(i)->FunctionValues().GetNumRows();
                cols[i] = Inputs(i)->FunctionValues().GetNumCols();
            }

            if (cols[0] != cols[1] || cols[0] != cols[2])
                LogicError("GMMLogLikelihoodNode: UnnormedPrior (first input), mean (second input), and logStddev (third input) should have same number of columns.");

            if (cols[0] != 1 && cols[0] != cols[3])
                LogicError("GMMLogLikelihoodNode: UnnormedPrior (first input) should either have same number of columns as the features (fourth input) or have only one column.");

            if (rows[0] != rows[2])
                LogicError("GMMLogLikelihoodNode: UnnormedPrior (first input) should have same dimension as logStddev (third input), i.e., all dimensions in each Gaussian component share the same stddev.");

            if (rows[1] != rows[0]*rows[3])
                LogicError("GMMLogLikelihoodNode: the number of rows in mean (second input) should equal rows(unnormedPrior(first input) * rows(feature(fourth input)).");

            FunctionValues().Resize(1, cols[3]);
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(3, false);

            m_outputChannels = 1;
            m_outputWidth = 1;
            m_outputHeight = 1;
        }

        //leftNode should be the empirical
        virtual void AttachInputs(const ComputationNodePtr unnormedPrior, const ComputationNodePtr mean, const ComputationNodePtr logStddev, const ComputationNodePtr feature)
        {
            m_children.resize(4);
            m_children[0] = unnormedPrior;
            m_children[1] = mean;
            m_children[2] = logStddev;
            m_children[3] = feature;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            Base::MoveMatricesToDevice(deviceId);
            m_prior.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId, true);
            m_normedDeviation.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId, true);
            m_normedDeviationVectors.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId, true);
            m_stddev.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId, true);
            m_posterior.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId, true);
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<GMMLogLikelihoodNode<ElemType>>(nodeP);
                node->m_prior = m_prior;
                node->m_normedDeviation = m_normedDeviation;
                node->m_normedDeviationVectors = m_normedDeviationVectors;
                node->m_stddev = m_stddev;
                node->m_posterior = m_posterior;
            }
        }

    protected:
        Matrix<ElemType> m_prior;
        Matrix<ElemType> m_normedDeviation;
        Matrix<ElemType> m_normedDeviationVectors;
        Matrix<ElemType> m_stddev;
        Matrix<ElemType> m_posterior;
        Matrix<ElemType> m_temp;
    };

    template class GMMLogLikelihoodNode<float>;
    template class GMMLogLikelihoodNode<double>;

    // =======================================================================
    // DropoutNode -- perform drop-out
    // Output is scaled such that no post-scaling is necessary.
    // =======================================================================

    template<class ElemType>
    class DropoutNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        DropoutNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_maskOfDropout(deviceId),
            m_dropoutRate(0)
        {
            m_randomSeed = (unsigned long)atomic_fetch_add(&s_timeStampCounter, (unsigned long long int)1);
        }

        virtual const std::wstring OperationName() const { return TypeName(); }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                InvalidArgument("Dropout operation only takes one input.");
            ComputeInputPartialS(m_dropoutRate, Inputs(0)->GradientValues(), m_maskOfDropout, GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex > 0)
                InvalidArgument("Dropout operation only takes one input.");

            Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceMask = Matrix<ElemType>();
            if (m_dropoutRate > 0)
            {
                sliceMask = m_maskOfDropout.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            }

            ComputeInputPartialS(m_dropoutRate, sliceInput0Grad, sliceMask, sliceOutputGrad);
        }

        static void WINAPI ComputeInputPartialS(const double dropoutRate, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& maskOfDropout, const Matrix<ElemType>& gradientValues)
        {
            if (dropoutRate > 0)
            {
                inputGradientValues.AddElementProductOf(gradientValues, maskOfDropout);
            }
            else
            {
                inputGradientValues += gradientValues;
            }
        }

        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(m_dropoutRate, m_randomSeed, FunctionValues(), m_maskOfDropout, Inputs(0)->FunctionValues());
        }
        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = Matrix <ElemType>();

            Matrix<ElemType> sliceMask = Matrix<ElemType>();
            if (m_dropoutRate > 0)
            {
                FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
                m_maskOfDropout.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
                sliceMask = m_maskOfDropout.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            }

            sliceOutputValue = FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(m_dropoutRate, m_randomSeed, sliceOutputValue, sliceMask, sliceInput0Value);
        }

        static void WINAPI EvaluateThisNodeS(const double dropoutRate, unsigned long& randomSeed, Matrix<ElemType>& functionValues, Matrix<ElemType>& maskOfDropout, const Matrix<ElemType>& inputFunctionValues)
        {
            if (dropoutRate > 0)
            {
                maskOfDropout.Resize(inputFunctionValues.GetNumRows(), inputFunctionValues.GetNumCols());

                maskOfDropout.SetUniformRandomMask((ElemType)dropoutRate, (ElemType)(1.0 / (1.0 - dropoutRate)), randomSeed);
                randomSeed += 1073807359;  //1073807359 is a very large prime number to avoid collision with other dropout nodes

                functionValues.AssignElementProductOf(maskOfDropout, inputFunctionValues);
#if NANCHECK
                functionValues.HasNan("DropOut");
#endif
            }
            else
            {
                //remove this line since we can get same effect by overwritting the FunctionValues functions without copying the values
                //functionValues = inputFunctionValues;
            }
        }

        virtual const Matrix<ElemType>& FunctionValues() const
        {
            if (m_dropoutRate > 0)
                return m_functionValues;
            else
                return Inputs(0)->FunctionValues();
        }

        virtual Matrix<ElemType>& FunctionValues()
        {
            if (m_dropoutRate > 0)
                return m_functionValues;
            else
                return Inputs(0)->FunctionValues();
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 1)
                LogicError("Dropout operation should have one input.");

            if (Inputs(0)->FunctionValues().HasNoElements())
                LogicError("Dropout operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_maskOfDropout.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            InferImageDimsFromInputs();
        }

        virtual void AttachInputs(const ComputationNodePtr inputNode)
        {
            m_children.resize(1);
            m_children[0] = inputNode;
        }

        void SetDropoutRate(const double val)
        {
            if (val < 0 || val >= 1)
                LogicError("DropoutRate must be >= 0 and < 1.");
            m_dropoutRate = val;
        }

        void SetRandomSeed(const unsigned long val)
        {
            m_randomSeed = (unsigned long)val;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            Base::MoveMatricesToDevice(deviceId);
            m_maskOfDropout.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId, true);
        }

        static const std::wstring TypeName() { return L"Dropout"; }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<DropoutNode<ElemType>>(nodeP);
                node->m_dropoutRate = m_dropoutRate;
                node->m_randomSeed = m_randomSeed;
                node->m_maskOfDropout = m_maskOfDropout;
            }
        }
private:
        double m_dropoutRate;
        unsigned long m_randomSeed;

        Matrix<ElemType> m_maskOfDropout;
    };

    template class DropoutNode<float>;
    template class DropoutNode<double>;

    // =======================================================================
    // ReshapeNode -- reshape input matrix
    // TODO: Why is this in NonlinearityNodes.h? Should he linear algebra no?
    // =======================================================================

    template<class ElemType>
    class ReshapeNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        ReshapeNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_numRows(0),
            m_imageWidth(0),
            m_imageHeight(0),
            m_imageChannels(0)
        { }
        ReshapeNode(DEVICEID_TYPE deviceId, const wstring & name, size_t numRows, size_t imageWidth, size_t imageHeight, size_t imageChannels) :
            ComputationNode<ElemType>(deviceId, name),
            m_numRows(numRows),
            m_imageWidth(imageWidth),
            m_imageHeight(imageHeight),
            m_imageChannels(imageChannels)
        { }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<ReshapeNode<ElemType>>(nodeP); // TODO: change to Base for all
                node->m_numRows = m_numRows;
                node->m_imageWidth = m_imageWidth;
                node->m_imageHeight = m_imageHeight;
                node->m_imageChannels = m_imageChannels;
            }
        }

        virtual void SaveToFile(File& fstream) const
        {
            Base::SaveToFile(fstream);
            fstream << m_numRows << m_imageWidth << m_imageHeight << m_imageChannels;
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion)
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_numRows >> m_imageWidth >> m_imageHeight >> m_imageChannels;
        }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"Reshape"; }

        virtual void AttachInputs(const ComputationNodePtr singleInput)
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, true);
            InferImageDimensions();

            if (m_imageWidth == 0 || m_imageHeight == 0 || m_imageChannels == 0)
            {
                m_outputWidth = 1;
                m_outputChannels = 1;
                m_outputHeight = m_numRows;
                if (m_inputWidth * m_inputChannels != 1)
                    fprintf(stderr, "WARNING: Reshape operation cannot inherit image size information from its child. Image size info is lost.\n");
            }
            else
            {
                m_outputWidth = m_imageWidth;
                m_outputChannels = m_imageChannels;
                m_outputHeight = m_imageHeight;
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
                        throw runtime_error("One of the children is missing.");
                    }

                    fprintf(stderr, "%ls[%lu, %lu]", child->NodeName().c_str(), child->FunctionValues().GetNumRows(), child->FunctionValues().GetNumCols());
                }

                fprintf(stderr, ", NumOfRows=%lu, imageWidth=%lu, imageHeight=%lu, imageChannels=%lu)", m_numRows, m_imageWidth, m_imageHeight, m_imageChannels);
            }
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 1)
                LogicError("Reshape operation: Should have one input.");

            if (Inputs(0)->FunctionValues().HasNoElements())
                LogicError("Reshape operation: The input node has 0 element.");

            size_t cols = Inputs(0)->FunctionValues().GetNumElements() / m_numRows;

            // We can not do a proper pre-validation check for the reshaping node. There are cases when 
            // reshaping only makes sense if we consider the whole minibatch but not based on a single
            // sample. This is a hack to prevent the validation step from throwing an unnecessary error
            // for cases where at runtime the operation would be valid
            if (cols == 0)
                cols = 1;
            FunctionValues().Resize(m_numRows, cols);
            InferImageDimsFromInputs();
        }

        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), m_numRows);
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            size_t rows = Inputs(0)->FunctionValues().GetNumRows();
            if ((rows * m_samplesInRecurrentStep) % m_numRows > 0)
            {
                LogicError("Reshape operation: Number of elements in the recurrent input step is not a multiple of the specified number of rows.");
            }

            size_t outputSamplesInRecurrentStep = m_samplesInRecurrentStep * rows / m_numRows;
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * outputSamplesInRecurrentStep, outputSamplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue, m_numRows);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues, const size_t numRows)
        {
            functionValues.Resize(inputFunctionValues.GetNumRows(), inputFunctionValues.GetNumCols());
            functionValues.AssignRowSliceValuesOf(inputFunctionValues, 0, inputFunctionValues.GetNumRows());

            if (functionValues.GetNumRows() != numRows)
            {
                if (functionValues.GetNumElements() % numRows > 0)
                    LogicError("Reshape operation: Number of elements in the input is not a multiple of the specified number of rows.");

                functionValues.Reshape(numRows, functionValues.GetNumElements() / numRows);
            }
#if NANCHECK
            functionValues.HasNan("Reshape");
#endif
        }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                InvalidArgument("Reshape operation only takes one input.");

            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues(), m_numRows);
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex > 0)
                InvalidArgument("Reshape operation only takes one input.");

            size_t rows = Inputs(0)->GradientValues().GetNumRows();
            if ((rows * m_samplesInRecurrentStep) % m_numRows > 0)
            {
                LogicError("Reshape operation: Number of elements in the recurrent input step is not a multiple of the specified number of rows.");
            }

            size_t outputSamplesInRecurrentStep = m_samplesInRecurrentStep * rows / m_numRows;

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * outputSamplesInRecurrentStep, outputSamplesInRecurrentStep);

            ComputeInputPartialS(sliceInputGrad, sliceOutputGrad, m_numRows);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const size_t /*numRows*/)
        {
            size_t numRows = inputGradientValues.GetNumRows();
            inputGradientValues.Reshape(gradientValues.GetNumRows(), gradientValues.GetNumCols());
            inputGradientValues += gradientValues;
            inputGradientValues.Reshape(numRows, inputGradientValues.GetNumElements() / numRows);
        }

        virtual const Matrix<ElemType>& FunctionValues() const
        {
            if (Inputs(0)->FunctionValues().GetNumRows() != m_numRows)
                return m_functionValues;
            else
                return Inputs(0)->FunctionValues();
        }

    private:
        size_t m_numRows;
        size_t m_imageWidth;
        size_t m_imageHeight;
        size_t m_imageChannels;

        void InferImageDimensions()
        {
            if (m_imageWidth > 0)
            {
                if (m_imageHeight > 0)
                {
                    if (m_imageChannels > 0)
                    {
                        if (m_imageWidth * m_imageHeight * m_imageChannels != m_numRows)
                            throw runtime_error("Image dimensions do not match row size.");
                    }
                    else
                    {
                        if (m_numRows % (m_imageWidth * m_imageHeight) > 0)
                            throw runtime_error("Image row size is not a multiple of specified image dimensions.");
                        else
                            m_imageChannels = m_numRows / (m_imageWidth * m_imageHeight);
                    }
                }
                else
                {
                    if (m_imageChannels > 0)
                    {
                        if (m_numRows % (m_imageWidth * m_imageChannels) > 0)
                            throw runtime_error("Image row size is not a multiple of specified image dimensions.");
                        else
                            m_imageHeight = m_numRows / (m_imageWidth * m_imageChannels);
                    }
                    else
                    {
                        throw runtime_error("At least two image dimensions must be specified.");
                    }
                }
            }
            else
            {
                if (m_imageHeight > 0)
                {
                    if (m_imageChannels > 0)
                    {
                        if (m_numRows % (m_imageHeight * m_imageChannels) > 0)
                            throw runtime_error("Image row size is not a multiple of specified image dimensions.");
                        else
                            m_imageWidth = m_numRows / (m_imageHeight * m_imageChannels);
                    }
                    else
                        throw runtime_error("At least two image dimensions must be specified.");
                }
                else if (m_imageChannels > 0)
                    throw runtime_error("At least two image dimensions must be specified.");
            }
        }
    };

    template class ReshapeNode<float>;
    template class ReshapeNode<double>;

    // =======================================================================
    // RowRepeatNode -- duplicate row(s) of a matrix multiple times
    // =======================================================================

    template<class ElemType>
    class RowRepeatNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        RowRepeatNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_numRepeat(1)
        { }
        RowRepeatNode(DEVICEID_TYPE deviceId, const wstring & name, size_t numRepeats) :
            ComputationNode<ElemType>(deviceId, name),
            m_numRepeat(numRepeats)
        { }
        // ^^ TODO: merge those two above using optional args

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<RowRepeatNode<ElemType>>(nodeP);
                node->m_numRepeat = m_numRepeat;
            }
        }

        virtual void SaveToFile(File& fstream) const
        {
            Base::SaveToFile(fstream);
            fstream << m_numRepeat;
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion)
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_numRepeat;
        }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"RowRepeat"; }

        virtual void AttachInputs(const ComputationNodePtr singleInput)
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, true);
            m_outputHeight = m_inputHeight * m_numRepeat;

            //WARNING: this node will destroy the image size information from the child
            if (m_inputWidth * m_inputChannels != 1)
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
                        throw runtime_error("One of the children is missing.");
                    }

                    fprintf(stderr, "%ls[%lu, %lu]", child->NodeName().c_str(), child->FunctionValues().GetNumRows(), child->FunctionValues().GetNumCols());
                }

                fprintf(stderr, ", numRepeats=%lu)", m_numRepeat);
            }
        }

        virtual void /*ComputationNodeBase::*/Validate()
        {
            Base::Validate();

            if (m_children.size() != 1)
                LogicError("RowRepeat operation should have one input.");

            if (Inputs(0)->FunctionValues().HasNoElements())
                LogicError("RowRepeat  operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows() * m_numRepeat, Inputs(0)->FunctionValues().GetNumCols());
            InferImageDimsFromInputs();
        }

        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues(), m_numRepeat);
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue, m_numRepeat);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues, const size_t numRepeats)
        {
            functionValues.AssignRepeatOf(inputFunctionValues, numRepeats, 1);
#if NANCHECK
            functionValues.HasNan("RowRepeat");
#endif
        }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                InvalidArgument("RowRepeat only has one input.");

            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues(), m_numRepeat);
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex != 0)
                InvalidArgument("RowRepeat only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(sliceInputGrad, sliceOutputGrad, m_numRepeat);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const size_t numRepeats)
        {
            inputGradientValues.AddToRowRepeatValuesOf(gradientValues, numRepeats);
        }

        virtual const Matrix<ElemType>& FunctionValues() const
        {
            if (m_numRepeat == 1)
                return m_functionValues;
            else
                return Inputs(0)->FunctionValues();
        }

    private:
        size_t m_numRepeat;
    };

    template class RowRepeatNode<float>;
    template class RowRepeatNode<double>;

}}}
