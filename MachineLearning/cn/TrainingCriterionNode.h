//
// <copyright file="TrainingCriterionNode.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

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
        UsingComputationNodeMembers;
    public:
        SquareErrorNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode<ElemType>(deviceId), m_leftMinusRight(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        SquareErrorNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_leftMinusRight(deviceId)
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

        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/)
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

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_leftMinusRight);
        }

        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/)
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

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
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
            : ComputationNode<ElemType>(node->m_deviceId), m_leftMinusRight(node->m_deviceId)
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
        UsingComputationNodeMembers;
    public:
        CrossEntropyWithSoftmaxNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode<ElemType>(deviceId), m_logSoftmaxOfRight(deviceId), m_softmaxOfRight(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        CrossEntropyWithSoftmaxNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_logSoftmaxOfRight(deviceId), m_softmaxOfRight(deviceId)
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

        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/) 
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


        virtual void EvaluateThisNode()   //-sum(left_i * log(softmax_i(right)))
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_softmaxOfRight, m_logSoftmaxOfRight);
        }

        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/) 
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

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
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
            : ComputationNode<ElemType>(node->m_deviceId), m_logSoftmaxOfRight(node->m_deviceId), m_softmaxOfRight(node->m_deviceId)
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
        UsingComputationNodeMembers;
    public:
        CrossEntropyNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode<ElemType>(deviceId), m_logOfRight(deviceId), m_leftDivRight(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        CrossEntropyNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_logOfRight(deviceId), m_leftDivRight(deviceId)
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

        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/) 
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

        virtual void EvaluateThisNode()   //-sum(left_i * log(right_i))
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_logOfRight);
        }

        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/) 
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

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
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
                    : ComputationNode<ElemType>(node->m_deviceId), m_logOfRight(node->m_deviceId), m_leftDivRight(node->m_deviceId)
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
        UsingComputationNodeMembers;
    public:
        MatrixL1RegNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode<ElemType>(deviceId), m_gradientOfL1Norm(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        MatrixL1RegNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_gradientOfL1Norm(deviceId)
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

        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/) 
        {
            throw std::logic_error("MatrixL1Reg node should never be in a loop.");
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientOfL1Norm, 
            Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& inputFunctionValues)  
        {
            gradientOfL1Norm.AssignSignOf(inputFunctionValues);
            inputGradientValues.AddWithScaleOf(gradientValues.Get00Element(), gradientOfL1Norm);
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/) 
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

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
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
            : ComputationNode<ElemType>(node->m_deviceId), m_gradientOfL1Norm(node->m_deviceId)
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
        UsingComputationNodeMembers;
    public:
        MatrixL2RegNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode<ElemType>(deviceId), m_temp(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        MatrixL2RegNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_temp(deviceId)
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

        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/) 
        {
            throw std::logic_error("MatrixL2RegNode node should never be in a loop.");
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& /*inputFunctionValues*/, const Matrix<ElemType>& functionValues)  
        {
            ElemType v = gradientValues.Get00Element() / (functionValues.Get00Element() + EPS_IN_INVERSE);
            inputGradientValues.AddWithScaleOf(v, gradientValues);
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/)
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
            : ComputationNode<ElemType>(node->m_deviceId), m_temp(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new MatrixL2RegNode<ElemType>(this, name, flags);
            return node;
        }
                
        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
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
    // need to provide class probabilty from external node
    template<class ElemType>
    class ClassBasedCrossEntropyWithSoftmaxNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        ClassBasedCrossEntropyWithSoftmaxNode(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_logSoftmax(deviceId), m_softMax(deviceId), m_grdToSoftMaxInput(deviceId), m_clsLogSoftmax(deviceId), m_clsSoftmax(deviceId)
        {
                m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
                m_deviceId = deviceId;
                MoveMatricesToDevice(deviceId);
                InitRecurrentNode();
            }

        ClassBasedCrossEntropyWithSoftmaxNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_logSoftmax(deviceId), m_softMax(deviceId), m_grdToSoftMaxInput(deviceId), m_clsLogSoftmax(deviceId), m_clsSoftmax(deviceId)
        {
                m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
                LoadFromFile(fstream, modelVersion, deviceId);
            }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"ClassBasedCrossEntropyWithSoftmax"; }

        /**
        compute gradients to input observations, the weights to the observations, and the class log posterior probabilites
        */
        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex != 1 && inputIndex != 2 && inputIndex != 3)
                throw std::invalid_argument("ClassCrossEntropyWithSoftmaxNode criterion only takes with respect to input, weight to the input and class log posterior probability.");

            size_t nT = Inputs(0)->FunctionValues().GetNumCols();
            Matrix<ElemType> grd_t;
            Matrix<ElemType> grd_to_wgt_t;

            ComputeSoftMaxPartial();

            size_t sz = 0;
            for (size_t t = 0; t < nT; t++)
            {
                /// compute prb - 1 and prb
                Matrix<ElemType> lbl_t = Inputs(0)->FunctionValues().ColumnSlice(t, 1);
                assert(lbl_t.GetDeviceId() == CPUDEVICE);
                size_t c_t = (size_t)lbl_t(1, 0);
                size_t lft_bnd = (size_t)lbl_t(2, 0);
                size_t rgt_bnd = (size_t)lbl_t(3, 0);
                size_t nbr_wrd = rgt_bnd - lft_bnd; // number of words in the class

                Matrix<ElemType> input_weight_t = Inputs(2)->FunctionValues().ColumnSlice(lft_bnd, nbr_wrd);

                Matrix<ElemType> obs = Inputs(1)->FunctionValues().ColumnSlice(t, 1);

                Matrix<ElemType> grd_to_soft_max_input = m_grdToSoftMaxInput.ColumnSlice(sz, nbr_wrd);

                Matrix<ElemType> grd_to_cls_prob = m_clsLogSoftmax.ColumnSlice(t, 1);

                switch (inputIndex){
                case 1:
                    /// gradient to input
                    grd_t = Inputs(1)->GradientValues().ColumnSlice(t, 1);
                    ComputeInputPartialRight(input_weight_t, grd_t, grd_to_soft_max_input);
                    break;
                case 2:
                    /// gradient to input weight
                    grd_to_wgt_t = Inputs(2)->GradientValues().ColumnSlice(lft_bnd, nbr_wrd);
                    ComputeInputPartialLeft(obs, grd_to_wgt_t, grd_to_soft_max_input);
                    break;
                case 3:
                    grd_t = Inputs(3)->GradientValues().ColumnSlice(t, 1);
                    grd_t.SetValue(m_clsSoftmax.ColumnSlice(t, 1));
                    ComputeCEPartialToSoftmaxInputs(grd_t, GradientValues(), c_t);
                    break;
                default:
                    throw std::runtime_error("ClassCrossEntropyWithSoftmaxNode criterion only takes with respect to input, weight to the input and class log posterior probability.");
                }

                sz += nbr_wrd;
            }

        }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/)
        {
            throw std::logic_error("ClassCrossEntropyWithSoftmax node should never be in a loop.");
        }

        static void WINAPI ComputeInputPartialRight(const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
        {
            Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, false, gradientValues, true, inputGradientValues);
        }

        static void WINAPI ComputeInputPartialLeft(const Matrix<ElemType>& obs, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
        {
            Matrix<ElemType>::MultiplyAndAdd(obs, false, gradientValues, false, inputGradientValues);
        }

        static void WINAPI ComputeCEPartialToSoftmaxInputs(Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues, size_t y_t)
        {
            Matrix<ElemType>::MinusOneAt(inputGradientValues, y_t);
            Matrix<ElemType>::Scale(gradientValues, inputGradientValues);
        }

        /// gradient of cross entropy w.r.t. to input to softmax
        void ComputeSoftMaxPartial()
        {
            if (m_needRecomputeGradientToSoftmaxInput)
            {
                m_grdToSoftMaxInput.Resize(1, m_totalNbrWords);

                size_t nT = Inputs(1)->FunctionValues().GetNumCols();
                size_t sz = 0;
                for (size_t t = 0; t < nT; t++)
                {
                    /// compute prb - 1 and prb
                    Matrix<ElemType> lbl_t = Inputs(0)->FunctionValues().ColumnSlice(t, 1);
                    size_t y_t = (size_t)lbl_t(0, 0);
                    size_t lft_bnd = (size_t)lbl_t(2, 0);
                    size_t rgt_bnd = (size_t)lbl_t(3, 0);
                    size_t nbr_wrd = rgt_bnd - lft_bnd;// number of words in the class

                    Matrix<ElemType> softMax = m_softMax.ColumnSlice(sz, nbr_wrd);

                    ComputeCEPartialToSoftmaxInputs(softMax, GradientValues(), y_t - lft_bnd);

                    m_grdToSoftMaxInput.ColumnSlice(sz, nbr_wrd).SetValue(softMax);

                    sz += nbr_wrd;
                }

                m_needRecomputeGradientToSoftmaxInput = false;
            }
        }

        virtual void EvaluateThisNode()   //-sum(left_i * log(softmax_i(right)))
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(),
                Inputs(3)->FunctionValues(), m_logSoftmax, m_softMax, m_clsLogSoftmax, m_clsSoftmax, m_totalNbrWords);
            m_needRecomputeGradientToSoftmaxInput = true;
        }

        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/)
        {
            throw std::logic_error("ClassCrossEntropyWithSoftmax node should never be in a loop.");
        }

        static void EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& lbls,
            const Matrix<ElemType>& inputs, const Matrix<ElemType>& input_weight, const Matrix<ElemType>& input_cls_log_post_prob,
            Matrix<ElemType>& logSoftmax,
            Matrix<ElemType>& softMax, Matrix<ElemType>& clsLogSoftmax, Matrix<ElemType>& clsSoftmax, size_t& totalWords)
        {
            totalWords = 0;
            size_t nT = lbls.GetNumCols();

            for (size_t t = 0; t < lbls.GetNumCols(); t++)
            {
                Matrix<ElemType> lblInfo = lbls.ColumnSlice(t, 1);
                size_t lft_bnd = (size_t)lblInfo(2, 0);
                size_t rgt_bnd = (size_t)lblInfo(3, 0);
                totalWords += (rgt_bnd - lft_bnd);
            }

            size_t nRow = inputs.GetNumRows();

            size_t sz = totalWords;
            softMax.Resize(1, sz);
            logSoftmax.Resize(1, sz);
            clsLogSoftmax.Resize(input_cls_log_post_prob.GetNumRows(), nT);
            clsSoftmax.Resize(input_cls_log_post_prob.GetNumRows(), nT);

            /// loop over time
            functionValues.SetValue(0);
            sz = 0;
            for (size_t t = 0; t < lbls.GetNumCols(); t++)
            {
                Matrix<ElemType> lblInfo = lbls.ColumnSlice(t, 1);
                size_t y_t = (size_t)lblInfo(0, 0);
                size_t c_t = (size_t)lblInfo(1, 0);
                size_t lft_bnd = (size_t)lblInfo(2, 0);
                size_t rgt_bnd = (size_t)lblInfo(3, 0);
                size_t nbr_wrd = rgt_bnd - lft_bnd;

                /// e.g., 200 x 148
                Matrix<ElemType> weightForClass = input_weight.ColumnSlice(lft_bnd, nbr_wrd);

                /// W x_t 
                Matrix<ElemType> softMax_t = softMax.ColumnSlice(sz, nbr_wrd);
                Matrix<ElemType> logSoftMax_t = logSoftmax.ColumnSlice(sz, nbr_wrd);
                Matrix<ElemType> obs = inputs.ColumnSlice(t, 1);  /// e.g., 200 x 1
                obs.Reshape(1, nRow);  /// 1 x 200

                logSoftMax_t.AssignProductOf(obs, false, weightForClass, false); /// 1 x 148

                // log softmax(W x_t)
                logSoftMax_t.InplaceLogSoftmax(false); /// 1 x 148
                softMax_t.SetValue(logSoftMax_t);
                // softmax(W x_t)
                softMax_t.InplaceExp();  /// 1 x 148

                /// add the word log posterior probability
                size_t idx_in_class = y_t - lft_bnd;
                Matrix<ElemType>::AddElementToElement(logSoftMax_t, 0, idx_in_class, functionValues, 0, 0);

                /// add the class log posterior probability
                Matrix<ElemType> clsLogSoftmax_t = clsLogSoftmax.ColumnSlice(t, 1);
                clsLogSoftmax_t.SetValue(input_cls_log_post_prob.ColumnSlice(t, 1));
                clsLogSoftmax_t.InplaceLogSoftmax(true); /// 50 x 1
                Matrix<ElemType> clsSoftmax_t = clsSoftmax.ColumnSlice(t, 1);
                clsSoftmax_t.AssignExpOf(clsLogSoftmax_t);
                Matrix<ElemType>::AddElementToElement(clsLogSoftmax_t, c_t, 0, functionValues, 0, 0);

                sz += nbr_wrd;
            }

            functionValues *= (-1);

#if NANCHECK
            functionValues.HasNan("ClassBasedCrossEntropyWithSoftmax");
#endif
        }

        /**
        Inputs: [0] label in dense matrix in [4 x T]
        the first row is the word index, the second row is the class index, the third row is the first word index of the class
        the last row is the first word index of the next class
        [1] hidden layer activity to the node in [hdsize x T]. for a simple rnn, this is the hidden layer activty
        [2] weight matrix in [hdsize x vocab_size], for speed-up, as per word matrix can be simply obtained as column slice
        [3] clsprob in dense matrix in [nbr_cls x T]. this is the output from logsoftmax node for the log-posterior probabilty of class given observations
        */
        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 4)
                throw std::logic_error("ClassBasedCrossEntropyWithSoftmaxNode criterion requires four inputs.");

            if (Inputs(0)->OperationName() != InputValue<ElemType>::TypeName())
                throw std::logic_error("ClassBasedCrossEntropyWithSoftmaxNode criterion requires the first input to be the label.");

            if (!(Inputs(1)->FunctionValues().GetNumRows() == Inputs(2)->FunctionValues().GetNumRows())) // input and matrix can be timed
            {
                throw std::logic_error("The Matrix<ElemType>  dimension for observation and weight in the ClassBasedCrossEntropyWithSoftmaxNode operation does not match.");
            }
            if (!(Inputs(0)->FunctionValues().GetNumCols() == Inputs(1)->FunctionValues().GetNumCols())) // label and input same obs numbers
            {
                throw std::logic_error("The Matrix<ElemType>  dimension for label and observation in the ClassBasedCrossEntropyWithSoftmaxNode operation does not match.");
            }
            if (!(Inputs(0)->FunctionValues().GetNumRows() == 4)) // label needs to be 4 rows
            {
                throw std::logic_error("The label in the ClassBasedCrossEntropyWithSoftmaxNode operation needs to be 4 rows.");
            }
            if (!(Inputs(3)->FunctionValues().GetNumCols() == Inputs(0)->FunctionValues().GetNumCols())) // number of observations
            {
                throw std::logic_error("The number of observations in class log post probability and label in the ClassBasedCrossEntropyWithSoftmaxNode operation don't match.");
            }

            FunctionValues().Resize(1, 1);
            CopyImageSizeFromInputs();

            m_nbrCls = Inputs(3)->FunctionValues().GetNumRows();
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, false);

            m_outputChannels = 1;
            m_outputWidth = 1;
            m_outputHeight = 1;
        }

        virtual void AttachInputs(const ComputationNodePtr label, const ComputationNodePtr input,
            const ComputationNodePtr inputweight, const ComputationNodePtr clsLogPostProbability)
        {
            m_children.resize(4);
            m_children[0] = label;
            m_children[1] = input;
            m_children[2] = inputweight;
            m_children[3] = clsLogPostProbability;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_logSoftmax.GetDeviceId() != deviceId)
                    m_logSoftmax.TransferFromDeviceToDevice(m_logSoftmax.GetDeviceId(), deviceId, true);
                if (m_softMax.GetDeviceId() != deviceId)
                    m_softMax.TransferFromDeviceToDevice(m_softMax.GetDeviceId(), deviceId, true);
                if (m_clsLogSoftmax.GetDeviceId() != deviceId)
                    m_clsLogSoftmax.TransferFromDeviceToDevice(m_clsLogSoftmax.GetDeviceId(), deviceId, true);
                if (m_clsSoftmax.GetDeviceId() != deviceId)
                    m_clsSoftmax.TransferFromDeviceToDevice(m_clsSoftmax.GetDeviceId(), deviceId, true);
                if (m_grdToSoftMaxInput.GetDeviceId() != deviceId)
                    m_grdToSoftMaxInput.TransferFromDeviceToDevice(m_grdToSoftMaxInput.GetDeviceId(), deviceId, true);
            }
        }

        // copy constructor
        ClassBasedCrossEntropyWithSoftmaxNode(const ClassBasedCrossEntropyWithSoftmaxNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_logSoftmax(node->m_deviceId), m_softMax(node->m_deviceId), m_grdToSoftMaxInput(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;

            ComputationNodePtr node = new ClassBasedCrossEntropyWithSoftmaxNode<ElemType>(this, name, flags);
            return node;
        }

    protected:
        Matrix<ElemType> m_logSoftmax;
        Matrix<ElemType> m_softMax;

        Matrix<ElemType> m_clsLogSoftmax;
        Matrix<ElemType> m_clsSoftmax;

        /// gradient of cross entropy with respect to the input of softmax
        /// a 1 row by \sum_t m_nbrWordsInEachTime[t] vector
        /// one slice of size m_nbrWordsInEachTime[t] saves the input to softmax for word y_t
        Matrix<ElemType> m_grdToSoftMaxInput;
        bool m_needRecomputeGradientToSoftmaxInput;

        size_t           m_nbrCls;
        size_t           m_totalNbrWords;
    };

    template class ClassBasedCrossEntropyWithSoftmaxNode<float>;
    template class ClassBasedCrossEntropyWithSoftmaxNode<double>;


}}}