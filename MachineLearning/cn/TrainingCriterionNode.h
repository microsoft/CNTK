//
// <copyright file="TrainingCriterionNode.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include <hash_set>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include "ComputationNode.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    //note: to save computation the gradient may be scaled by an constant. 

    template<class ElemType>
    class SquareErrorNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


    public:
        SquareErrorNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode(deviceId), m_leftMinusRight(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        SquareErrorNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_leftMinusRight(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }
                
        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"SquareError";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("SquareError criteria only takes two inputs.");

            //left Node must be a scalar
            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(Inputs(0)->GradientValues(), GradientValues(), m_leftMinusRight);
            }
            else
            {
                ComputeInputPartialRight(Inputs(1)->GradientValues(), GradientValues(), m_leftMinusRight);
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            throw std::logic_error("SquareError node should never be in a loop.");
        }

        static void WINAPI ComputeInputPartialLeft(Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& leftMinusRight)  
        {
            inputGradientValues.AddWithScaleOf(gradientValues.Get00Element(), leftMinusRight);
        }

        static void WINAPI ComputeInputPartialRight(Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& leftMinusRight)  
        {
            inputGradientValues.AddWithScaleOf(-gradientValues.Get00Element(), leftMinusRight);
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                descriptor->GradientParam(inputIndex, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->MatrixParam(m_leftMinusRight, "leftMinusRight", paramOptionsInput);
                descriptor->SetFunction(inputIndex?(FARPROC)ComputeInputPartialRight:(FARPROC)ComputeInputPartialLeft);
                break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->MatrixParam(m_leftMinusRight, "leftMinusRight", paramOptionsOutput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_leftMinusRight);
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            throw std::logic_error("SquareError node should never be in a loop.");
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues0, const Matrix<ElemType>& inputFunctionValues1, Matrix<ElemType>& leftMinusRight)  
        {
            leftMinusRight.AssignDifferenceOf(inputFunctionValues0, inputFunctionValues1);
            ElemType v = leftMinusRight.FrobeniusNorm();
            functionValues.Resize(1,1);
            functionValues.SetValue(v*v/2);
#if NANCHECK
            functionValues.HasNan("SquareError");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("SquareError operation requires two inputs.");

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
                throw std::logic_error("SquareError operation: one of the operants has 0 element.");

            if (!(Inputs(0)->FunctionValues().GetNumRows() == Inputs(1)->FunctionValues().GetNumRows()  &&  //match size
                Inputs(0)->FunctionValues().GetNumCols() == Inputs(1)->FunctionValues().GetNumCols()) )
            {
                throw std::logic_error("The Matrix dimension in the SquareError operation does not match.");
            }       

            FunctionValues().Resize(1,1);
            m_leftMinusRight.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
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

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_leftMinusRight.GetDeviceId() != deviceId)
                    m_leftMinusRight.TransferFromDeviceToDevice(m_leftMinusRight.GetDeviceId(), deviceId,true);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            SquareErrorNode<ElemType>* node = (SquareErrorNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_leftMinusRight = m_leftMinusRight;
            }
        }

        // copy constructor
        SquareErrorNode(const SquareErrorNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), m_leftMinusRight(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new SquareErrorNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_leftMinusRight;
    };

    template class SquareErrorNode<float>; 
    template class SquareErrorNode<double>;

    //calculates: -sum(left_i * log(softmax_i(right)))
    template<class ElemType>
    class CrossEntropyWithSoftmaxNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


    public:
        CrossEntropyWithSoftmaxNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode(deviceId), m_logSoftmaxOfRight(deviceId), m_softmaxOfRight(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        CrossEntropyWithSoftmaxNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_logSoftmaxOfRight(deviceId), m_softmaxOfRight(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }
                
        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"CrossEntropyWithSoftmax";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("CrossEntropyWithSoftmaxNode criterion only takes two inputs.");

            //left Node must be a scalar
            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(m_logSoftmaxOfRight, Inputs(inputIndex)->GradientValues(), GradientValues());
            }
            else
            {
                ComputeInputPartialRight(m_softmaxOfRight, Inputs(0)->FunctionValues(), Inputs(inputIndex)->GradientValues(), GradientValues());
            }

        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq) 
        {
            throw std::logic_error("CrossEntropyWithSoftmax node should never be in a loop.");
        }

        static void WINAPI ComputeInputPartialLeft(const Matrix<ElemType>& logSoftmaxOfRight, Matrix<ElemType>& inputGradientValues, 
            const Matrix<ElemType>& gradientValues)  
        {
#if DUMPOUTPUT
            logSoftmaxOfRight.Print("CrossEntropyWithSoftmax Partial-logSoftmaxOfRight");
            gradientValues.Print("CrossEntropyWithSoftmax Partial-gradientValues");
            inputGradientValues.Print("CrossEntropyWithSoftmaxNode Partial-Left-in");
#endif

            Matrix<ElemType>::ScaleAndAdd(-gradientValues.Get00Element(), logSoftmaxOfRight, inputGradientValues);
#if DUMPOUTPUT
            inputGradientValues.Print("CrossEntropyWithSoftmaxNode Partial-Left-out");
#endif

        }

        static void WINAPI ComputeInputPartialRight(const Matrix<ElemType>& softmaxOfRight, const Matrix<ElemType>& inputFunctionValues, 
            Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        {
#if DUMPOUTPUT
            softmaxOfRight.Print("CrossEntropyWithSoftmax Partial-softmaxOfRight");
            inputFunctionValues.Print("CrossEntropyWithSoftmax Partial-inputFunctionValues");
            gradientValues.Print("CrossEntropyWithSoftmax Partial-gradientValues");
            inputGradientValues.Print("CrossEntropyWithSoftmaxNode Partial-Right-in");
#endif

            Matrix<ElemType>::AddScaledDifference(gradientValues, softmaxOfRight, inputFunctionValues, inputGradientValues);
#if DUMPOUTPUT
            inputGradientValues.Print("CrossEntropyWithSoftmaxNode Partial-Right");
#endif
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                if (inputIndex == 0)
                {
                    descriptor->MatrixParam(m_logSoftmaxOfRight, "logSoftmaxOfRight", paramOptionsInput);
                }
                else
                {
                    descriptor->MatrixParam(m_softmaxOfRight, "softmaxOfRight", paramOptionsInput);
                    descriptor->FunctionParam(0, paramOptionsInput);
                }
                descriptor->GradientParam(inputIndex, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->SetFunction(inputIndex?(FARPROC)ComputeInputPartialRight:(FARPROC)ComputeInputPartialLeft);
                break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->MatrixParam(m_softmaxOfRight, "softmaxOfRight", paramOptionsOutput);
                descriptor->MatrixParam(m_logSoftmaxOfRight, "logSoftmaxOfRight", paramOptionsOutput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()   //-sum(left_i * log(softmax_i(right)))
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_softmaxOfRight, m_logSoftmaxOfRight);
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            throw std::logic_error("CrossEntropyWithSoftmax node should never be in a loop.");
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues0, const Matrix<ElemType>& inputFunctionValues1, 
            Matrix<ElemType>& softmaxOfRight, Matrix<ElemType>& logSoftmaxOfRight)  
        {
			logSoftmaxOfRight.AssignLogSoftmaxOf(inputFunctionValues1, true);
			softmaxOfRight.SetValue(logSoftmaxOfRight);
			softmaxOfRight.InplaceExp();
            functionValues.AssignInnerProductOfMatrices(inputFunctionValues0, logSoftmaxOfRight);
            functionValues*=(-1);
#if NANCHECK
            functionValues.HasNan("CrossEntropyWithSoftmax");
#endif
#if DUMPOUTPUT
            functionValues.Print("CrossEntropyWithSoftmaxNode");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("CrossEntropyWithSoftmaxNode criterion requires two inputs.");

            if (Inputs(0)->OperationName() != L"InputValue" && Inputs(0)->OperationName() != L"SparseInputValue")
                throw std::logic_error("CrossEntropyWithSoftmaxNode criterion requires the first input to be the label.");

            //we may release the constraint that the first operant is an inputValue later so the following code should be kept
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
                throw std::logic_error("CrossEntropyWithSoftmaxNode operation: one of the operants has 0 element.");

            if (!(Inputs(0)->FunctionValues().GetNumRows() == Inputs(1)->FunctionValues().GetNumRows()  &&  //match size
                Inputs(0)->FunctionValues().GetNumCols() == Inputs(1)->FunctionValues().GetNumCols()) )
            {
                throw std::logic_error("The Matrix<ElemType>  dimension in the CrossEntropyWithSoftmaxNode operation does not match.");
            }       

            FunctionValues().Resize(1,1);
            CopyImageSizeFromInputs(); 

            m_logSoftmaxOfRight.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            m_softmaxOfRight.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, false);

            m_outputChannels = 1;
            m_outputWidth = 1;
            m_outputHeight = 1;        
        }

        //leftNode should be the empirical
        virtual void AttachInputs(const ComputationNodePtr label, const ComputationNodePtr prediction) 
        {
            m_children.resize(2);
            m_children[0] = label;
            m_children[1] = prediction;
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_logSoftmaxOfRight.GetDeviceId() != deviceId)
                {
                    m_logSoftmaxOfRight.TransferFromDeviceToDevice(m_logSoftmaxOfRight.GetDeviceId(), deviceId,true);
                }
                if (m_softmaxOfRight.GetDeviceId() != deviceId)
                {
                    m_softmaxOfRight.TransferFromDeviceToDevice(m_softmaxOfRight.GetDeviceId(), deviceId,true);
                }
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            CrossEntropyWithSoftmaxNode<ElemType>* node = (CrossEntropyWithSoftmaxNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_logSoftmaxOfRight = m_logSoftmaxOfRight;
                node->m_softmaxOfRight = m_softmaxOfRight;
            }
        }

        // copy constructor
        CrossEntropyWithSoftmaxNode(const CrossEntropyWithSoftmaxNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), m_logSoftmaxOfRight(node->m_deviceId), m_softmaxOfRight(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new CrossEntropyWithSoftmaxNode<ElemType>(this, name, flags);
            return node;
        }

    protected:
        Matrix<ElemType> m_logSoftmaxOfRight;
        Matrix<ElemType> m_softmaxOfRight;       
    };

    template class CrossEntropyWithSoftmaxNode<float>; 
    template class CrossEntropyWithSoftmaxNode<double>;

    //calculates: -sum(left_i * log(right_i))
    //assume softmax is already done
    template<class ElemType>
    class CrossEntropyNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


    public:
        CrossEntropyNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode(deviceId), m_logOfRight(deviceId), m_leftDivRight(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        CrossEntropyNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_logOfRight(deviceId), m_leftDivRight(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"CrossEntropy";} 

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 1)
                throw std::invalid_argument("CrossEntropy criterion only takes two inputs.");

            //left Node must be a scalar
            if (inputIndex == 0)  //left derivative
            {
                ComputeInputPartialLeft(m_logOfRight, Inputs(inputIndex)->GradientValues(), GradientValues());
            }
            else
            {
                ComputeInputPartialRight(m_leftDivRight, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(inputIndex)->GradientValues(), GradientValues());
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq) 
        {
            throw std::logic_error("CrossEntropy node should never be in a loop.");
        }

        static void WINAPI ComputeInputPartialLeft(const Matrix<ElemType>& logOfRight, Matrix<ElemType>& inputGradientValues, 
            const Matrix<ElemType>& gradientValues)  
        {
            Matrix<ElemType>::ScaleAndAdd(-gradientValues.Get00Element(), logOfRight, inputGradientValues);
        }

        static void WINAPI ComputeInputPartialRight(Matrix<ElemType>& leftDivRight, 
            const Matrix<ElemType>& inputFunctionValues0, const Matrix<ElemType>& inputFunctionValues1,
            Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)  
        {
            leftDivRight.AssignElementDivisionOf(inputFunctionValues0, inputFunctionValues1);

            Matrix<ElemType>::ScaleAndAdd(-gradientValues.Get00Element(), leftDivRight, inputGradientValues);
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                if (inputIndex == 0)
                {
                    descriptor->MatrixParam(m_logOfRight, "logOfRight", paramOptionsInput);
                }
                else
                {
                    descriptor->MatrixParam(m_leftDivRight, "leftDivRight", paramOptionsInput | paramOptionsTemporary);
                    descriptor->FunctionParam(0, paramOptionsInput);
                    descriptor->FunctionParam(1, paramOptionsInput);
                }
                descriptor->GradientParam(inputIndex, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->SetFunction(inputIndex?(FARPROC)ComputeInputPartialRight:(FARPROC)ComputeInputPartialLeft);
                break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->MatrixParam(m_logOfRight, "logOfRight", paramOptionsOutput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }


        virtual void EvaluateThisNode()   //-sum(left_i * log(right_i))
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_logOfRight);
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            throw std::logic_error("CrossEntropy node should never be in a loop.");
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues0, const Matrix<ElemType>& inputFunctionValues1, 
            Matrix<ElemType>& logOfRight)  
        {
            logOfRight.SetValue(inputFunctionValues1);
            logOfRight.InplaceLog();
            functionValues.AssignInnerProductOfMatrices(inputFunctionValues0, logOfRight);
            functionValues*=(-1);
#if NANCHECK
            functionValues.HasNan("CrossEntropy");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("CrossEntropyNode criterion requires two inputs.");

            if (Inputs(0)->OperationName() != L"InputValue")
                throw std::logic_error("CrossEntropyNode criterion requires the first input to be the label.");

            //we may release the constraint that the first operant is an inputValue later so the following code should be kept
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
                throw std::logic_error("CrossEntropyNode operation: one of the operants has 0 element.");

            if (!(Inputs(0)->FunctionValues().GetNumRows() == Inputs(1)->FunctionValues().GetNumRows()  &&  //match size
                Inputs(0)->FunctionValues().GetNumCols() == Inputs(1)->FunctionValues().GetNumCols()) )
            {
                throw std::logic_error("The Matrix dimension in the CrossEntropyNode operation does not match.");
            }       

            FunctionValues().Resize(1,1);
            m_logOfRight.Resize(Inputs(1)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());
            m_leftDivRight.Resize(Inputs(1)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, false);

            m_outputChannels = 1;
            m_outputWidth = 1;
            m_outputHeight = 1;        
        }

        //leftNode should be the empirical
        virtual void AttachInputs(const ComputationNodePtr label, const ComputationNodePtr prediction) 
        {
            m_children.resize(2);
            m_children[0] = label;
            m_children[1] = prediction;
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_logOfRight.GetDeviceId() != deviceId)
                {
                    m_logOfRight.TransferFromDeviceToDevice(m_logOfRight.GetDeviceId(), deviceId,true);
                }
                if (m_leftDivRight.GetDeviceId() != deviceId)
                {
                    m_leftDivRight.TransferFromDeviceToDevice(m_leftDivRight.GetDeviceId(), deviceId,true);
                }
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            CrossEntropyNode<ElemType>* node = (CrossEntropyNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_logOfRight = m_logOfRight;
                node->m_leftDivRight = m_leftDivRight;
            }
        }

        // copy constructor
        CrossEntropyNode(const CrossEntropyNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
                    : ComputationNode(node->m_deviceId), m_logOfRight(node->m_deviceId), m_leftDivRight(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new CrossEntropyNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        // matrix value passed from evaluate to computePartial
        Matrix<ElemType> m_logOfRight;
        // temporary
        Matrix<ElemType> m_leftDivRight;
    };

    template class CrossEntropyNode<float>; 
    template class CrossEntropyNode<double>;

    template<class ElemType>
    class MatrixL1RegNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 

    public:
        MatrixL1RegNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode(deviceId), m_gradientOfL1Norm(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        MatrixL1RegNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_gradientOfL1Norm(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"MatrixL1Reg";} 

        virtual void ComputeInputPartial(const size_t inputIndex) // scale by number of cols (or samples)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("MatrixL1RegNode only has one input.");

            ComputeInputPartialS(m_gradientOfL1Norm, Inputs(0)->GradientValues(), GradientValues(), Inputs(0)->FunctionValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq) 
        {
            throw std::logic_error("MatrixL1Reg node should never be in a loop.");
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientOfL1Norm, 
            Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            gradientOfL1Norm.AssignSignOf(inputFunctionValues);
            inputGradientValues.AddWithScaleOf(gradientValues.Get00Element(), gradientOfL1Norm);
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                descriptor->MatrixParam(m_gradientOfL1Norm, "gradientOfL1Norm", paramOptionsInput | paramOptionsTemporary);
                descriptor->GradientParam(0, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->SetFunction((FARPROC)ComputeInputPartialS);
                break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            throw std::logic_error("MatrixL1Reg node should never be in a loop.");
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.Resize(1,1);
            functionValues.SetValue(inputFunctionValues.MatrixNorm1());
#if NANCHECK
            functionValues.HasNan("MatrixL1Reg");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("MatrixL1Reg criterion should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("MatrixL1Reg operation: the input node has 0 element.");

            FunctionValues().Resize(1,1);
            m_gradientOfL1Norm.Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, false);

            m_outputChannels = 1;
            m_outputWidth = 1;
            m_outputHeight = 1;
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_gradientOfL1Norm.GetDeviceId() != deviceId)
                    m_gradientOfL1Norm.TransferFromDeviceToDevice(m_gradientOfL1Norm.GetDeviceId(), deviceId,true);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            MatrixL1RegNode<ElemType>* node = (MatrixL1RegNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_gradientOfL1Norm = m_gradientOfL1Norm;
            }
        }

        // copy constructor
        MatrixL1RegNode(const MatrixL1RegNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), m_gradientOfL1Norm(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new MatrixL1RegNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        // temporary
        Matrix<ElemType> m_gradientOfL1Norm;
    };

    template class MatrixL1RegNode<float>; 
    template class MatrixL1RegNode<double>;

    template<class ElemType>
    class MatrixL2RegNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 


    public:
        MatrixL2RegNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode(deviceId), m_temp(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        MatrixL2RegNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_temp(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"MatrixL2Reg";} 


        virtual void ComputeInputPartial(const size_t inputIndex) // scale by number of cols (or samples)
        {
            if (inputIndex != 0)
                throw std::invalid_argument("MatrixL2RegNode only has one input.");

            ComputeInputPartialS(Inputs(0)->GradientValues(), GradientValues(), Inputs(0)->FunctionValues(), FunctionValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq) 
        {
            throw std::logic_error("MatrixL2RegNode node should never be in a loop.");
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& functionValues)  
        {
            ElemType v = gradientValues.Get00Element() / (functionValues.Get00Element() + EPS_IN_INVERSE);
            inputGradientValues.AddWithScaleOf(v, gradientValues);
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                descriptor->GradientParam(0, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->GradientParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam();
                descriptor->SetFunction((FARPROC)ComputeInputPartialS);
                break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }


        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            throw std::logic_error("MatrixL2RegNode node should never be in a loop.");
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            functionValues.Resize(1,1);
            functionValues.SetValue(inputFunctionValues.FrobeniusNorm());
#if NANCHECK
            functionValues.HasNan("MatrixL2Reg");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("MatrixL2Reg criterion should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("MatrixL2Reg operation: the input node has 0 element.");

            FunctionValues().Resize(1,1);
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, false);

            m_outputChannels = 1;
            m_outputWidth = 1;
            m_outputHeight = 1;        
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        // copy constructor
        MatrixL2RegNode(const MatrixL2RegNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), m_temp(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new MatrixL2RegNode<ElemType>(this, name, flags);
            return node;
        }
                
        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_temp.GetDeviceId() != deviceId)
                {
                    m_temp.TransferFromDeviceToDevice(m_temp.GetDeviceId(), deviceId,true);
                }
            }
        }

    private:
        Matrix<ElemType> m_temp;
    };

    template class MatrixL2RegNode<float>; 
    template class MatrixL2RegNode<double>;

    //calculates: -sum(left_i * log(softmax_i(right))) for class given history and for word given history
    template<class ElemType>
    class ClassBasedCrossEntropyWithSoftmaxNode: public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType>* ComputationNodePtr; 

    public:
        ClassBasedCrossEntropyWithSoftmaxNode(const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode(deviceId), m_logSoftmax(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        ClassBasedCrossEntropyWithSoftmaxNode(File& fstream, const size_t modelVersion, const short deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), m_logSoftmax(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }
                
        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"ClassBasedCrossEntropyWithSoftmax";} 

        virtual void ComputeInputPartial(const size_t inputIndex)  //scaled by 2*number of colmns (samples) in the Matrix<ElemType>
        {
            if (inputIndex != 1 && inputIndex != 2)
                throw std::invalid_argument("ClassCrossEntropyWithSoftmaxNode criterion only takes with respect to input and weight.");

            if (inputIndex == 1)
                ComputeClassEntropyGradientOfInput(Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), m_ptrClsinfo, m_ptrIdx2Cls, m_logSoftmax, Inputs(inputIndex)->GradientValues());
            else
                ComputeClassEntropyGradientOfWeight(Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), m_ptrClsinfo, m_ptrIdx2Cls, m_logSoftmax, Inputs(inputIndex)->GradientValues());

        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq) 
        {
            throw std::logic_error("ClassCrossEntropyWithSoftmax node should never be in a loop.");
        }

        static void ComputeClassEntropyGradientOfInput(const Matrix<ElemType>& inputFunctionValues0, const Matrix<ElemType>& inputFunctionValues1, 
            const Matrix<ElemType>& inputFunctionValues2, const Matrix<ElemType>* clsInfo, const Matrix<ElemType>* idx2Cls, 
            const Matrix<ElemType>& logSoftmax, Matrix<ElemType>& grd)  
        {
            logSoftmax.ClassEntropyError(logSoftmax);
            logSoftmax.ClassEntropyGradientOfInput(logSoftmax, inputFunctionValues2, grd);
        }

        static void ComputeClassEntropyGradientOfWeight(const Matrix<ElemType>& inputFunctionValues0, const Matrix<ElemType>& inputFunctionValues1, 
            const Matrix<ElemType>& inputFunctionValues2, const Matrix<ElemType>* clsInfo, const Matrix<ElemType>* idx2Cls, 
            const Matrix<ElemType>& logSoftmax, Matrix<ElemType>& grd)  
        {
            logSoftmax.ClassEntropyGradientOfWeight(logSoftmax, 
                    inputFunctionValues1, inputFunctionValues2, 
                    inputFunctionValues0, 
                    clsInfo, idx2Cls, grd);
        }

        // GetTaskDescriptor - Get a task descriptor for this node
        // taskType - task type we are generating a task for
        virtual TaskDescriptor<ElemType>* GetPTaskDescriptor(TaskType taskType, size_t inputIndex=0) const
        {
            TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType, inputIndex);
            switch(taskType)
            {
            case taskComputeInputPartial:
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->FunctionParam(2, paramOptionsInput);
                descriptor->MatrixParam(*m_ptrClsinfo, "clsInfo", paramOptionsInput | paramOptionsConstant);
                descriptor->MatrixParam(*m_ptrIdx2Cls, "idx2Cls", paramOptionsInput | paramOptionsConstant);
                descriptor->MatrixParam(m_logSoftmax, "logSoftmax", paramOptionsInput);
                descriptor->GradientParam(inputIndex, paramOptionsInput | paramOptionsOutput | paramOptionsInitialize);
                descriptor->SetFunction(inputIndex==1?(FARPROC)ComputeClassEntropyGradientOfInput:(FARPROC)ComputeClassEntropyGradientOfWeight);
                break;
            case taskEvaluate:
                descriptor->FunctionParam();
                descriptor->FunctionParam(0, paramOptionsInput);
                descriptor->FunctionParam(1, paramOptionsInput);
                descriptor->FunctionParam(2, paramOptionsInput);
                descriptor->MatrixParam(*m_ptrClsinfo, "clsInfo", paramOptionsInput | paramOptionsConstant);
                descriptor->MatrixParam(*m_ptrIdx2Cls, "idx2Cls", paramOptionsInput | paramOptionsConstant);
                descriptor->MatrixParam(m_logSoftmax, "logSoftmax", paramOptionsOutput);
                descriptor->SetFunction((FARPROC)EvaluateThisNodeS);
                break;
            default:
                assert(false);
                throw std::logic_error("Unsupported task requested");
            }
            return descriptor;
        }

        virtual void EvaluateThisNode()   //-sum(left_i * log(softmax_i(right)))
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), m_ptrClsinfo, m_ptrIdx2Cls, m_logSoftmax);
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            throw std::logic_error("ClassCrossEntropyWithSoftmax node should never be in a loop.");
        }

        static void EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues0, 
            const Matrix<ElemType>& inputFunctionValues1, const Matrix<ElemType>& inputFunctionValues2, 
            const Matrix<ElemType>* clsInfo, const Matrix<ElemType>* idx2Cls, Matrix<ElemType>& logSoftmax)  
        {            
            logSoftmax.Resize(inputFunctionValues0.GetNumRows(), inputFunctionValues0.GetNumCols());
            logSoftmax.ClassEntropy(inputFunctionValues1, inputFunctionValues2, inputFunctionValues0, clsInfo, idx2Cls, logSoftmax, functionValues);
#if NANCHECK
            functionValues.HasNan("ClassBasedCrossEntropyWithSoftmax");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 3) 
                throw std::logic_error("ClassBasedCrossEntropyWithSoftmaxNode criterion requires three inputs.");

            if (Inputs(0)->OperationName() != L"SparseInputValue" 
                && Inputs(0)->OperationName() != L"InputValue")
                throw std::logic_error("ClassBasedCrossEntropyWithSoftmaxNode criterion requires the first input to be the label.");

            if (!(Inputs(1)->FunctionValues().GetNumRows() == Inputs(2)->FunctionValues().GetNumCols()  &&  // input and matrix can be timed
                Inputs(0)->FunctionValues().GetNumCols() == Inputs(1)->FunctionValues().GetNumCols() &&  // label and input same obs numbers
                Inputs(0)->FunctionValues().GetNumRows() == Inputs(2)->FunctionValues().GetNumRows() ) ) // label and matrix match output size
            {
                throw std::logic_error("The Matrix<ElemType>  dimension in the ClassBasedCrossEntropyWithSoftmaxNode operation does not match.");
            }       

            FunctionValues().Resize(1,1);
            CopyImageSizeFromInputs(); 
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, false);

            m_outputChannels = 1;
            m_outputWidth = 1;
            m_outputHeight = 1;        
        }

        //leftNode should be the empirical
        // classinfo is a matrix of N columns and 2 rows. N columns correspond to N class
        // the first row indicates the starting row and the second row indicates the end row of a class
        virtual void AddClassInfo(Matrix<ElemType>* classinfo,
            Matrix<ElemType>* idx2cls) 
        {
            m_ptrClsinfo = classinfo;
            m_ptrIdx2Cls = idx2cls;
        }

        //leftNode should be the empirical
        // classinfo is a matrix of N columns and 2 rows. N columns correspond to N class
        // the first row indicates the starting row and the second row indicates the end row of a class
        virtual void AttachInputs(const ComputationNodePtr label, const ComputationNodePtr input, 
            const ComputationNodePtr matrix) 
        {
            m_children.resize(3);
            m_children[0] = label;
            m_children[1] = input;
            m_children[2] = matrix;

            //initializes m_logSoftmax
            m_logSoftmax.SwitchToMatrixType(SPARSE, matrixFormatSparseCSC);
            m_logSoftmax.Resize(label->FunctionValues().GetNumRows(), label->FunctionValues().GetNumCols());
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_logSoftmax.GetDeviceId() != deviceId)
                {
                    m_logSoftmax.TransferFromDeviceToDevice(m_logSoftmax.GetDeviceId(), deviceId,true);
                }
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            ClassBasedCrossEntropyWithSoftmaxNode<ElemType>* node = (ClassBasedCrossEntropyWithSoftmaxNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_logSoftmax = m_logSoftmax;                
            }
        }

        // copy constructor
        ClassBasedCrossEntropyWithSoftmaxNode(const ClassBasedCrossEntropyWithSoftmaxNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), m_logSoftmax(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new ClassBasedCrossEntropyWithSoftmaxNode<ElemType>(this, name, flags);
            return node;
        }

    protected:
        Matrix<ElemType> m_logSoftmax;

        Matrix<ElemType>* m_ptrClsinfo;
        Matrix<ElemType>* m_ptrIdx2Cls;
    };

    template class ClassBasedCrossEntropyWithSoftmaxNode<float>; 
    template class ClassBasedCrossEntropyWithSoftmaxNode<double>;

}}}