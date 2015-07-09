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

    template<class ElemType>
    class RectifiedLinearNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        RectifiedLinearNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode<ElemType>(deviceId), m_gradientOfRectifiedLinear(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        RectifiedLinearNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_gradientOfRectifiedLinear(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("RectifiedLinear only has one input.");
            ComputeInputPartialS(m_gradientOfRectifiedLinear, Inputs(0)->FunctionValues(), Inputs(0)->GradientValues(), GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("RectifiedLinear only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradientOfRectifiedLinear, sliceInputValue, sliceInputGrad, sliceOutputGrad);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientOfRectifiedLinear, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
        {
            gradientOfRectifiedLinear.AssignLinearRectifierDerivativeOf(inputFunctionValues);
#if DUMPOUTPUT
            inputGradientValues.Print("RecitifiedLinearNode-Partial-in");
#endif
            inputGradientValues.AddElementProductOf(gradientValues, gradientOfRectifiedLinear); 
#if DUMPOUTPUT
            inputGradientValues.Print("RecitifiedLinearNode-Partial-out");
#endif
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
            functionValues.AssignTruncateBottomOf(inputFunctionValues, 0);
#if NANCHECK
            functionValues.HasNan("RectifiedLinear");
#endif
#if DUMPOUTPUT
            functionValues.Print("RectifiedLinearNode");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("RectifiedLinear operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("RectifiedLinear operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_gradientOfRectifiedLinear.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_gradientOfRectifiedLinear.GetDeviceId() != deviceId)
                    m_gradientOfRectifiedLinear.TransferFromDeviceToDevice(m_gradientOfRectifiedLinear.GetDeviceId(), deviceId);
            }
        }

        static const std::wstring TypeName() {return L"RectifiedLinear";} 

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            RectifiedLinearNode<ElemType>* node = (RectifiedLinearNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_gradientOfRectifiedLinear = m_gradientOfRectifiedLinear;
            }
        }

        // copy constructor
        RectifiedLinearNode(const RectifiedLinearNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_gradientOfRectifiedLinear(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new RectifiedLinearNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_gradientOfRectifiedLinear;
    };

    template class RectifiedLinearNode<float>; 
    template class RectifiedLinearNode<double>;

    template<class ElemType>
    class SigmoidNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        SigmoidNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode<ElemType>(deviceId), m_gradientOfSigmoid(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        SigmoidNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_gradientOfSigmoid(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Sigmoid";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Sigmoid only has one input.");
            ComputeInputPartialS(m_gradientOfSigmoid, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());  
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Sigmoid only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradientOfSigmoid, sliceInputGrad, sliceOutputGrad, sliceOutputValue);  
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientOfSigmoid, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)  
        {
            gradientOfSigmoid.AssignSigmoidDerivativeOf(functionValues);

            inputGradientValues.AddElementProductOf(gradientValues, gradientOfSigmoid);
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
            functionValues.AssignSigmoidOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Sigmoid");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("Sigmoid operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Sigmoid operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_gradientOfSigmoid.Resize(FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_gradientOfSigmoid.GetDeviceId() != deviceId)
                    m_gradientOfSigmoid.TransferFromDeviceToDevice(m_gradientOfSigmoid.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            SigmoidNode<ElemType>* node = (SigmoidNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_gradientOfSigmoid = m_gradientOfSigmoid;
            }
        }

        // copy constructor
        SigmoidNode(const SigmoidNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_gradientOfSigmoid(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new SigmoidNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_gradientOfSigmoid;
    };

    template class SigmoidNode<float>; 
    template class SigmoidNode<double>;


    template<class ElemType>
    class TanhNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        TanhNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode<ElemType>(deviceId), m_gradientOfTanh(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        TanhNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_gradientOfTanh(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Tanh";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Tanh only has one input.");
            ComputeInputPartialS(m_gradientOfTanh, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Tanh only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradientOfTanh, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientOfTanh, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)  
        {
            gradientOfTanh.AssignElementProductOf(functionValues, functionValues); // v .* v
            gradientOfTanh.AssignDifferenceOf(1, gradientOfTanh); // 1-v^2

            inputGradientValues.AddElementProductOf(gradientValues, gradientOfTanh); // += d .* ((1-v) .* v))
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
            functionValues.AssignTanhOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Tanh");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("Tanh operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Tanh operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_gradientOfTanh.Resize(FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_gradientOfTanh.GetDeviceId() != deviceId)
                    m_gradientOfTanh.TransferFromDeviceToDevice(m_gradientOfTanh.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            TanhNode<ElemType>* node = (TanhNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_gradientOfTanh = m_gradientOfTanh;
            }
        }

        // copy constructor
        TanhNode(const TanhNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_gradientOfTanh(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new TanhNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_gradientOfTanh;
    };

    template class TanhNode<float>; 
    template class TanhNode<double>;


    template<class ElemType>
    class LogNode : public ComputationNode<ElemType>
    {
                UsingComputationNodeMembers;
        public:
        LogNode(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_gradientOfLog(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        LogNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_gradientOfLog(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"Log"; }


        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Log only has one input.");
            ComputeInputPartialS(m_gradientOfLog, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Log only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradientOfLog, sliceInputGrad, sliceInputValue, sliceOutputGrad);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientOfLog, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& gradientValues)
        {
            gradientOfLog.AssignElementInverseOf(inputFunctionValues); // 1/x (x is input to log(x))

            inputGradientValues.AddElementProductOf(gradientValues, gradientOfLog);
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
            functionValues.AssignLogOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Log");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1)
                throw std::logic_error("Log operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Log operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_gradientOfLog.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs();
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput)
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_gradientOfLog.GetDeviceId() != deviceId)
                    m_gradientOfLog.TransferFromDeviceToDevice(m_gradientOfLog.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            LogNode<ElemType>* node = (LogNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_gradientOfLog = m_gradientOfLog;
            }
        }

        // copy constructor
        LogNode(const LogNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_gradientOfLog(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;

            ComputationNodePtr node = new LogNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_gradientOfLog;
    };

    template class LogNode<float>;
    template class LogNode<double>;



    template<class ElemType>
    class ExpNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        ExpNode(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_gradientOfExp(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        ExpNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_gradientOfExp(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"Exp"; }


        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Exp only has one input.");
            ComputeInputPartialS(m_gradientOfExp, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Exp only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradientOfExp, sliceInputGrad, sliceInputValue, sliceOutputGrad);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientOfExp, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& gradientValues)
        {
            gradientOfExp.AssignExpOf(inputFunctionValues); // Exp(x) is its own partial

            inputGradientValues.AddElementProductOf(gradientValues, gradientOfExp);
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
            functionValues.AssignExpOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Exp");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1)
                throw std::logic_error("Exp operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Exp operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_gradientOfExp.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs();
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput)
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_gradientOfExp.GetDeviceId() != deviceId)
                    m_gradientOfExp.TransferFromDeviceToDevice(m_gradientOfExp.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            ExpNode<ElemType>* node = (ExpNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_gradientOfExp = m_gradientOfExp;
            }
        }

        // copy constructor
        ExpNode(const ExpNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_gradientOfExp(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;

            ComputationNodePtr node = new ExpNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_gradientOfExp;
    };

    template class ExpNode<float>;
    template class ExpNode<double>;


    template<class ElemType>
    class CosineNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        CosineNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode<ElemType>(deviceId), m_gradientOfCosine(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        CosineNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_gradientOfCosine(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Cosine";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Cosine only has one input.");
            ComputeInputPartialS(m_gradientOfCosine, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Cosine only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradientOfCosine, sliceInputGrad, sliceInputValue, sliceOutputGrad);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientOfCosine, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& gradientValues)  
        {
            gradientOfCosine.AssignNegativeSineOf(inputFunctionValues); // -sin(x) (x is input to Cosine(x))
            inputGradientValues.AddElementProductOf(gradientValues, gradientOfCosine);
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
            functionValues.AssignCosineOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Cosine");
#endif
        }


        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("Cosine operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Cosine operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_gradientOfCosine.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_gradientOfCosine.GetDeviceId() != deviceId)
                    m_gradientOfCosine.TransferFromDeviceToDevice(m_gradientOfCosine.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            CosineNode<ElemType>* node = (CosineNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_gradientOfCosine = m_gradientOfCosine;
            }
        }

        // copy constructor
        CosineNode(const CosineNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_gradientOfCosine(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new CosineNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_gradientOfCosine;
    };

    template class CosineNode<float>; 
    template class CosineNode<double>;


    //we assume it's  column-wise by default
    //the derivative will increase the Matrix<ElemType> size to the power of column size and should not be used.
    template<class ElemType>
    class SoftmaxNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        SoftmaxNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_gradientDotValue(deviceId), m_diff(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        SoftmaxNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_gradientDotValue(deviceId), m_diff(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Softmax";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Softmax only has one input.");
            ComputeInputPartialS(m_gradientDotValue, m_diff, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Softmax only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradientDotValue, m_diff, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientDotValue, Matrix<ElemType>& diff, Matrix<ElemType>& inputGradientValues,
            const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)
        {
            gradientDotValue.AssignInnerProductOf(gradientValues, functionValues, true);
            diff.AssignDifferenceOf(gradientValues, gradientDotValue);

            inputGradientValues.AddElementProductOf(diff, functionValues);
        }

        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            size_t r = Inputs(0)->FunctionValues().GetNumRows(), c = Inputs(0)->FunctionValues().GetNumCols();
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            if (m_functionValues.GetNumCols() != c ||
                m_functionValues.GetNumRows() != r)
                m_functionValues.Resize(r, c);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.AssignLogSoftmaxOf(inputFunctionValues, true);
            functionValues.InplaceExp();
#if NANCHECK
            functionValues.HasNan("SoftMax");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("SoftmaxNode operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("SoftmaxNode operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_gradientDotValue.GetDeviceId() != deviceId)
                    m_gradientDotValue.TransferFromDeviceToDevice(m_gradientDotValue.GetDeviceId(), deviceId);
                if (m_diff.GetDeviceId() != deviceId)
                    m_diff.TransferFromDeviceToDevice(m_diff.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            SoftmaxNode<ElemType>* node = (SoftmaxNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_gradientDotValue = m_gradientDotValue;
                node->m_diff = m_diff;
            }
        }

        // copy constructor
        SoftmaxNode(const SoftmaxNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_gradientDotValue(node->m_deviceId), m_diff(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new SoftmaxNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_gradientDotValue;
        Matrix<ElemType> m_diff;
    };

    template class SoftmaxNode<float>; 
    template class SoftmaxNode<double>;

    template<class ElemType>
    class LogSoftmaxNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        LogSoftmaxNode(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_gradientDotValue(deviceId), m_softmax(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        LogSoftmaxNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_gradientDotValue(deviceId), m_softmax(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"LogSoftmax"; }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Softmax only has one input.");
            ComputeInputPartialS(m_gradientDotValue, m_softmax, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("Softmax only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(m_gradientDotValue, m_softmax, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientDotValue, Matrix<ElemType>& softmax, Matrix<ElemType>& inputGradientValues,
            const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)
        {
            softmax.AssignExpOf(functionValues);
            Matrix<ElemType>::VectorSum(gradientValues, gradientDotValue, true);
            softmax.RowElementMultiplyWith(gradientDotValue);
            Matrix<ElemType>::AddScaledDifference(1.0, gradientValues, softmax, inputGradientValues);
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
            functionValues.AssignLogSoftmaxOf(inputFunctionValues, true);
#if NANCHECK
            functionValues.HasNan("LogSoftMax");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1)
                throw std::logic_error("LogSoftmaxNode operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("LogSoftmaxNode operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs();
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput)
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_gradientDotValue.GetDeviceId() != deviceId)
                    m_gradientDotValue.TransferFromDeviceToDevice(m_gradientDotValue.GetDeviceId(), deviceId);
                if (m_softmax.GetDeviceId() != deviceId)
                    m_softmax.TransferFromDeviceToDevice(m_softmax.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            LogSoftmaxNode<ElemType>* node = (LogSoftmaxNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_gradientDotValue = m_gradientDotValue;
                node->m_softmax = m_softmax;
            }
        }

        // copy constructor
        LogSoftmaxNode(const LogSoftmaxNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_gradientDotValue(node->m_deviceId), m_softmax(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;

            ComputationNodePtr node = new LogSoftmaxNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_gradientDotValue;
        Matrix<ElemType> m_softmax;
    };

    template class LogSoftmaxNode<float>;
    template class LogSoftmaxNode<double>;


    //calculates: the log likelihood of a feature given GMM parameters
    template<class ElemType>
    class GMMLogLikelihoodNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        GMMLogLikelihoodNode(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_prior(deviceId), m_normedDeviation(deviceId), m_normedDeviationVectors(deviceId), m_stddev(deviceId), m_posterior(deviceId), m_temp(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        GMMLogLikelihoodNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_prior(deviceId), m_normedDeviation(deviceId), m_normedDeviationVectors(deviceId), m_stddev(deviceId), m_posterior(deviceId), m_temp(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        GMMLogLikelihoodNode(const GMMLogLikelihoodNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_prior(node->m_deviceId), m_normedDeviation(node->m_deviceId), m_normedDeviationVectors(node->m_deviceId),
            m_stddev(node->m_deviceId), m_posterior(node->m_deviceId), m_temp(m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;

            ComputationNodePtr node = new GMMLogLikelihoodNode<ElemType>(this, name, flags);
            return node;
        }

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
                throw std::invalid_argument("GMMLogLikelihoodNode only takes four inputs.");
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            //get the right slice 
            size_t startIndex = timeIdxInSeq * m_samplesInRecurrentStep;

            size_t colsPrior = Inputs(0)->FunctionValues().GetNumCols();

            Matrix<ElemType> sliceGradientValue = m_gradientValues.ColumnSlice(startIndex, m_samplesInRecurrentStep);
            Matrix<ElemType> slicePosterior = m_posterior.ColumnSlice(startIndex, m_samplesInRecurrentStep);
                
            switch (inputIndex)
            {
            case 0:
                {
                    if (colsPrior == 1)
                        ComputeInputPartialUnnormedPrior(Inputs(0)->GradientValues(), sliceGradientValue, m_prior, slicePosterior, m_temp);
                    else
                    {
                        Matrix<ElemType> sliceUnnormedPriorGradient = Inputs(0)->GradientValues().ColumnSlice(startIndex, m_samplesInRecurrentStep);
                        Matrix<ElemType> slicePrior = m_prior.ColumnSlice(startIndex, m_samplesInRecurrentStep);
                        ComputeInputPartialUnnormedPrior(sliceUnnormedPriorGradient, sliceGradientValue, slicePrior, slicePosterior, m_temp);
                    }
                }
                break;
            case 1:
                {
                      Matrix<ElemType> sliceNormedDeviationVectors = m_normedDeviationVectors.ColumnSlice(startIndex, m_samplesInRecurrentStep);
                      if (colsPrior == 1)
                        ComputeInputPartialMean(Inputs(1)->GradientValues(), sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, m_temp);
                    else
                    {
                        Matrix<ElemType> sliceMeanGradient = Inputs(1)->GradientValues().ColumnSlice(startIndex, m_samplesInRecurrentStep);
                        ComputeInputPartialMean(sliceMeanGradient, sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, m_temp);
                    }
                }
                break;
            case 2:
                {
                    Matrix<ElemType> sliceNormedDeviation = m_normedDeviation.ColumnSlice(startIndex, m_samplesInRecurrentStep);
                    if (colsPrior == 1)
                        ComputeInputPartialLogStddev(Inputs(2)->GradientValues(), sliceGradientValue, sliceNormedDeviation, slicePosterior, m_temp);
                    else
                    {
                        Matrix<ElemType> sliceLotStddevGradient = Inputs(2)->GradientValues().ColumnSlice(startIndex, m_samplesInRecurrentStep);
                        ComputeInputPartialLogStddev(sliceLotStddevGradient, sliceGradientValue, sliceNormedDeviation, slicePosterior, m_temp);
                    }
                }
                break;
            case 3:
                {
                    Matrix<ElemType> sliceNormedDeviationVectors = m_normedDeviationVectors.ColumnSlice(startIndex, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceFeatureGradient = Inputs(3)->GradientValues().ColumnSlice(startIndex, m_samplesInRecurrentStep);
                    ComputeInputPartialFeature(sliceFeatureGradient, sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, m_temp);
                }
                break;
            default:
                throw std::invalid_argument("GMMLogLikelihoodNode criterion only takes four inputs.");
            }
        }

        static void WINAPI ComputeInputPartialUnnormedPrior(Matrix<ElemType>& unnormedPriorGradientValues, const Matrix<ElemType>& gradientValues,
            const Matrix<ElemType>& prior, const Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
        {
            temp.AssignDifferenceOf(posterior, prior);
            temp.RowElementMultiplyWith(gradientValues);
            if (prior.GetNumCols() == posterior.GetNumCols())
            {
                unnormedPriorGradientValues += temp; 
            }
            else if (prior.GetNumCols() == 1)
            {
                Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(posterior.GetNumCols(), 1, unnormedPriorGradientValues.GetDeviceId()), false, unnormedPriorGradientValues);
            }
            else
            {
                throw std::runtime_error("GMMLogLikelihoodNode: UnnormedPrior should either have same number of columns as the features or have only one column.");
            }
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
            {
                meanGradientValues += temp;
            }
            else if (meanGradientValues.GetNumCols() == 1)
            {
                Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(numSamples, 1, meanGradientValues.GetDeviceId()), false, meanGradientValues);
            }
            else
            {
                throw std::runtime_error("GMMLogLikelihoodNode: stddev should either have same number of columns as the features or have only one column.");
            }
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
            {
                logStddevGradientValues += temp;
            }
            else if (logStddevGradientValues.GetNumCols() == 1)
            {
                Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(numSamples, 1, logStddevGradientValues.GetDeviceId()), false, logStddevGradientValues);
            }
            else
            {
                throw std::runtime_error("GMMLogLikelihoodNode: stddev should either have same number of columns as the features or have only one column.");
            }
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
        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            size_t colsPrior = Inputs(0)->FunctionValues().GetNumCols();
            size_t numSamples = Inputs(3)->FunctionValues().GetNumCols();

            //get the right slice 
            size_t startIndex = timeIdxInSeq * m_samplesInRecurrentStep;

            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(startIndex, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceFeature = Inputs(3)->FunctionValues().ColumnSlice(startIndex, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceNormedDeviation = m_normedDeviation.ColumnSlice(startIndex, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceNormedDeviationVectors = m_normedDeviationVectors.ColumnSlice(startIndex, m_samplesInRecurrentStep);
            Matrix<ElemType> slicePosterior = m_posterior.ColumnSlice(startIndex, m_samplesInRecurrentStep);

            if (colsPrior == 1)
            {
                EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), sliceFeature,
                    m_prior, m_stddev, sliceNormedDeviationVectors, sliceNormedDeviation, slicePosterior, m_temp);
            }
            else if (colsPrior == numSamples)
            {
                Matrix<ElemType> sliceUnnormedPrior = Inputs(0)->FunctionValues().ColumnSlice(startIndex, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceMean = Inputs(1)->FunctionValues().ColumnSlice(startIndex, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceLogstddev = Inputs(2)->FunctionValues().ColumnSlice(startIndex, m_samplesInRecurrentStep);

                Matrix<ElemType> slicePrior = m_prior.ColumnSlice(startIndex, m_samplesInRecurrentStep);
                Matrix<ElemType> sliceStddev = m_stddev.ColumnSlice(startIndex, m_samplesInRecurrentStep);

                EvaluateThisNodeS(sliceOutputValue, sliceUnnormedPrior, sliceMean, sliceLogstddev, sliceFeature,
                    slicePrior, sliceStddev, sliceNormedDeviationVectors, sliceNormedDeviation, slicePosterior, m_temp);
            }
            else  //should not reach the code since validation should fail already
            {
                throw std::runtime_error("GMMLogLikelihoodNode: UnnormedPrior should either have same number of columns as the features or have only one column.");
            }

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

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 4)
                throw std::logic_error("GMMLogLikelihoodNode requires four inputs.");

            size_t rows[4], cols[4];
            for (int i = 0; i < 4; i++)
            {
                rows[i] = Inputs(i)->FunctionValues().GetNumRows();
                cols[i] = Inputs(i)->FunctionValues().GetNumCols();
            }

            if (cols[0] != cols[1] || cols[0] != cols[2])
                throw std::logic_error("GMMLogLikelihoodNode: UnnormedPrior (first input), mean (second input), and logStddev (third input) should have same number of columns.");

            if (cols[0] != 1 && cols[0] != cols[3])
                throw std::logic_error("GMMLogLikelihoodNode: UnnormedPrior (first input) should either have same number of columns as the features (fourth input) or have only one column.");

            if (rows[0] != rows[2])
                throw std::logic_error("GMMLogLikelihoodNode: UnnormedPrior (first input) should have same dimension as logStddev (third input), i.e., all dimensions in each Gaussian component share the same stddev.");

            if (rows[1] != rows[0]*rows[3])
                throw std::logic_error("GMMLogLikelihoodNode: the number of rows in mean (second input) should equal rows(unnormedPrior(first input) * rows(feature(fourth input)).");

            FunctionValues().Resize(1, cols[3]);
            CopyImageSizeFromInputs();
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(3, false);

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
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_prior.GetDeviceId() != deviceId)
                {
                    m_prior.TransferFromDeviceToDevice(m_prior.GetDeviceId(), deviceId, true);
                }
                if (m_normedDeviation.GetDeviceId() != deviceId)
                {
                    m_normedDeviation.TransferFromDeviceToDevice(m_normedDeviation.GetDeviceId(), deviceId, true);
                }
                if (m_normedDeviationVectors.GetDeviceId() != deviceId)
                {
                    m_normedDeviationVectors.TransferFromDeviceToDevice(m_normedDeviationVectors.GetDeviceId(), deviceId, true);
                }
                if (m_stddev.GetDeviceId() != deviceId)
                {
                    m_stddev.TransferFromDeviceToDevice(m_stddev.GetDeviceId(), deviceId, true);
                }
                if (m_posterior.GetDeviceId() != deviceId)
                {
                    m_posterior.TransferFromDeviceToDevice(m_posterior.GetDeviceId(), deviceId, true);
                }
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            GMMLogLikelihoodNode<ElemType>* node = (GMMLogLikelihoodNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
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

    template<class ElemType>
    class DropoutNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:

        DropoutNode(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_maskOfDropout(deviceId)
        {
                m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
                m_deviceId = deviceId;
                MoveMatricesToDevice(deviceId);
                m_dropoutRate = 0;
                m_randomSeed = (unsigned long)atomic_fetch_add(&s_timeStampCounter, (unsigned long long int)1);
                InitRecurrentNode();
            }

        DropoutNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_maskOfDropout(deviceId)
        {
                m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
                m_dropoutRate = 0;  //dropout is consisered as a training parameter and thus not reinitialized if loadfromfile
                m_randomSeed = (unsigned long)atomic_fetch_add(&s_timeStampCounter, (unsigned long long int)1);

                LoadFromFile(fstream, modelVersion, deviceId);
            }

        virtual const std::wstring OperationName() const { return TypeName(); }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("Dropout operation only takes one input.");
            ComputeInputPartialS(m_dropoutRate, Inputs(0)->GradientValues(), m_maskOfDropout, GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("Dropout operation only takes one input.");

            Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceMask = Matrix<ElemType>();
            if (m_dropoutRate > 0)
            {
                sliceMask = m_maskOfDropout.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            }

            ComputeInputPartialS(m_dropoutRate, sliceInput0Grad, sliceMask, sliceOutputGrad);
        }

        static void WINAPI ComputeInputPartialS(const ElemType dropoutRate, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& maskOfDropout, const Matrix<ElemType>& gradientValues)
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
        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = Matrix <ElemType>();

            Matrix<ElemType> sliceMask = Matrix<ElemType>();
            if (m_dropoutRate > 0)
            {
                FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
                m_maskOfDropout.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
                sliceMask = m_maskOfDropout.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            }

            sliceOutputValue = FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(m_dropoutRate, m_randomSeed, sliceOutputValue, sliceMask, sliceInput0Value);
        }

        static void WINAPI EvaluateThisNodeS(const ElemType dropoutRate, unsigned long& randomSeed, Matrix<ElemType>& functionValues, Matrix<ElemType>& maskOfDropout, const Matrix<ElemType>& inputFunctionValues)
        {
            if (dropoutRate > 0)
            {
                maskOfDropout.Resize(inputFunctionValues.GetNumRows(), inputFunctionValues.GetNumCols());

                maskOfDropout.SetUniformRandomMask(dropoutRate, ElemType(1.0) / (ElemType(1) - dropoutRate), randomSeed);
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

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1)
                throw std::logic_error("Dropout operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Dropout operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_maskOfDropout.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs();
        }

        virtual void AttachInputs(const ComputationNodePtr inputNode)
        {
            m_children.resize(1);
            m_children[0] = inputNode;
        }

        void SetDropoutRate(const ElemType val)
        {
            if (val < 0 || val >= 1)
                throw std::logic_error("DropoutRate must be >= 0 and < 1.");
            m_dropoutRate = val;
        }

        void SetRandomSeed(const unsigned long val)
        {
            m_randomSeed = (unsigned long)val;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_maskOfDropout.GetDeviceId() != deviceId)
                    m_maskOfDropout.TransferFromDeviceToDevice(m_maskOfDropout.GetDeviceId(), deviceId, true);
            }
        }

        static const std::wstring TypeName() { return L"Dropout"; }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            DropoutNode<ElemType>* node = (DropoutNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_dropoutRate = m_dropoutRate;
                node->m_randomSeed = m_randomSeed;
                node->m_maskOfDropout = m_maskOfDropout;
            }
        }

        // copy constructor
        DropoutNode(const DropoutNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_maskOfDropout(node->m_deviceId)
        {
                node->CopyTo(this, newName, flags);
            }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;

            ComputationNodePtr node = new DropoutNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        ElemType m_dropoutRate;
        unsigned long m_randomSeed;

        Matrix<ElemType> m_maskOfDropout;
    };

    template class DropoutNode<float>;
    template class DropoutNode<double>;

    template<class ElemType>
    class ReshapeNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;

    public:

        ReshapeNode(const DEVICEID_TYPE deviceId, size_t numRows, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId)
        {
                m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
                m_deviceId = deviceId;
                m_numRows = numRows;

                MoveMatricesToDevice(deviceId);
                InitRecurrentNode();
            }

        ReshapeNode(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_numRows(0)
        {
                m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
                m_deviceId = deviceId;
                MoveMatricesToDevice(deviceId);
                InitRecurrentNode();
            }

        ReshapeNode(const ReshapeNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId)
        {
                node->CopyTo(this, newName, flags);
            }

        ReshapeNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId)
        {
                m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
                LoadFromFile(fstream, modelVersion, deviceId);
            }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;
            ComputationNodePtr node = new ReshapeNode<ElemType>(this, name, flags);
            return node;
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            ReshapeNode<ElemType>* node = (ReshapeNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_numRows = m_numRows;
            }
        }

        virtual void SaveToFile(File& fstream) const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << m_numRows;
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            fstream >> m_numRows;
        }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"Reshape"; }

        virtual void AttachInputs(const ComputationNodePtr singleInput)
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, true);

            //WARNING: this node will destroy the image size information from the child
            m_outputWidth = 1;
            m_outputChannels = 1;
            m_outputHeight = m_numRows;

            if (m_inputWidth * m_inputChannels != 1)
                fprintf(stderr, "WARNING: Reshape operation cannot inherit image size information from its child. Image size info is lost.\n");
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

                fprintf(stderr, ", NumOfRows=%lu)", m_numRows);
            }
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1)
                throw std::logic_error("Reshape operation: Should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Reshape operation: The input node has 0 element.");

            size_t cols = Inputs(0)->FunctionValues().GetNumElements() / m_numRows;

            // We can not do a proper pre-validation check for the reshaping node. There are cases when 
            // reshaping only makes sense if we consider the whole minibatch but not based on a single
            // sample. This is a hack to prevent the validation step from throwing an unnecessary error
            // for cases where at runtime the operation would be valid
            if (cols == 0)
                cols = 1;
            FunctionValues().Resize(m_numRows, cols);
            CopyImageSizeFromInputs();
        }

        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), m_numRows);
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            size_t rows = Inputs(0)->FunctionValues().GetNumRows();
            if ((rows * m_samplesInRecurrentStep) % m_numRows > 0)
            {
                throw std::logic_error("Reshape operation: Number of elements in the recurrent input step is not a multiple of the specified number of rows.");
            }

            size_t outputSamplesInRecurrentStep = m_samplesInRecurrentStep * rows / m_numRows;
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * outputSamplesInRecurrentStep, outputSamplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInputValue, m_numRows);
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues, const size_t numRows)
        {
            functionValues.Resize(inputFunctionValues.GetNumRows(), inputFunctionValues.GetNumCols());
            functionValues.AssignRowSliceValuesOf(inputFunctionValues, 0, inputFunctionValues.GetNumRows());

            if (functionValues.GetNumRows() != numRows)
            {
                if (functionValues.GetNumElements() % numRows > 0)
                    throw std::logic_error("Reshape operation: Number of elements in the input is not a multiple of the specified number of rows.");

                functionValues.Reshape(numRows, functionValues.GetNumElements() / numRows);
            }
#if NANCHECK
            functionValues.HasNan("Reshape");
#endif
        }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("Reshape operation only takes one input.");

            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues(), m_numRows);
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("Reshape operation only takes one input.");

            size_t rows = Inputs(0)->GradientValues().GetNumRows();
            if ((rows * m_samplesInRecurrentStep) % m_numRows > 0)
            {
                throw std::logic_error("Reshape operation: Number of elements in the recurrent input step is not a multiple of the specified number of rows.");
            }

            size_t outputSamplesInRecurrentStep = m_samplesInRecurrentStep * rows / m_numRows;

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * outputSamplesInRecurrentStep, outputSamplesInRecurrentStep);

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

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);
        }

    private:
        size_t m_numRows;
    };

    template class ReshapeNode<float>;
    template class ReshapeNode<double>;

    template<class ElemType>
    class RowRepeatNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;

    public:

        RowRepeatNode(const DEVICEID_TYPE deviceId, size_t numRepeats, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId)
        {
                m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
                m_deviceId = deviceId;
                m_numRepeat = numRepeats;

                MoveMatricesToDevice(deviceId);
                InitRecurrentNode();
            }

        RowRepeatNode(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_numRepeat(1)
        {
                m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
                m_deviceId = deviceId;
                MoveMatricesToDevice(deviceId);
                InitRecurrentNode();
            }

        RowRepeatNode(const RowRepeatNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId)
        {
                node->CopyTo(this, newName, flags);
            }

        RowRepeatNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId)
        {
                m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
                LoadFromFile(fstream, modelVersion, deviceId);
            }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;
            ComputationNodePtr node = new RowRepeatNode<ElemType>(this, name, flags);
            return node;
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            RowRepeatNode<ElemType>* node = (RowRepeatNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_numRepeat = m_numRepeat;
            }
        }

        virtual void SaveToFile(File& fstream) const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << m_numRepeat;
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            fstream >> m_numRepeat;
        }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"RowRepeat"; }

        virtual void AttachInputs(const ComputationNodePtr singleInput)
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, true);
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

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1)
                throw std::logic_error("RowRepeat operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("RowRepeat  operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows() * m_numRepeat, Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs();
        }

        virtual void EvaluateThisNode()
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues(), m_numRepeat);
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            Matrix<ElemType> sliceInputValue = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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
                throw std::invalid_argument("RowRepeat only has one input.");

            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues(), m_numRepeat);
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("RowRepeat only has one input.");

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

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

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);
        }

    private:
        size_t m_numRepeat;
    };

    template class RowRepeatNode<float>;
    template class RowRepeatNode<double>;

}}}
