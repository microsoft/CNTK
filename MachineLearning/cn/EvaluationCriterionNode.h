//
// <copyright file="EvaluationCriterionNode.h" company="Microsoft">
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
    class ErrorPredictionNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        ErrorPredictionNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") 
            : ComputationNode<ElemType>(deviceId), m_maxIndexes0(deviceId), m_maxIndexes1(deviceId), m_maxValues(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        ErrorPredictionNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_maxIndexes0(deviceId), m_maxIndexes1(deviceId), m_maxValues(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }
                
        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"ErrorPrediction";} 

        void Reset()
        {
        }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/)  //scaled by 2*number of elements in the Matrix<ElemType>
        {
            throw std::logic_error("ErrorPrediction is used for evaluation only.");
        }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/)
        {
            throw std::logic_error("ErrorPrediction is used for evaluation only.");
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(m_functionValues, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_maxIndexes0, m_maxIndexes1, m_maxValues);
        }

        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/)
        {
            throw std::logic_error("ErrorPrediction node should never be in a loop.");
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues0, const Matrix<ElemType>& inputFunctionValues1, Matrix<ElemType>& maxIndexes0, Matrix<ElemType>& maxIndexes1, Matrix<ElemType>& maxValues)  
        {
            inputFunctionValues0.VectorMax(maxIndexes0, maxValues, true);
            inputFunctionValues1.VectorMax(maxIndexes1, maxValues, true);
            functionValues.AssignNumOfDiff(maxIndexes0, maxIndexes1);
        #if NANCHECK
            functionValues.HasNan("ErrorPrediction");
        #endif
#if DUMPOUTPUT
            functionValues.Print("ErrorPredictionNode");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("ErrorPrediction operation requires two inputs.");

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
                m_maxIndexes0.Resize(1,cols);
                m_maxIndexes1.Resize(1,cols);
                m_maxValues.Resize(1,cols);
            }

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("ErrorPrediction operation: one of the operants has 0 element.");

            if (((!(Inputs(0)->FunctionValues().GetNumRows() == Inputs(1)->FunctionValues().GetNumRows()  &&  //match size
                Inputs(0)->FunctionValues().GetNumCols() == Inputs(1)->FunctionValues().GetNumCols()) )) && Inputs(0)->LoopId() < 0)
            {
                throw std::logic_error("The Matrix dimension in the ErrorPrediction operation does not match.");
            }       

            FunctionValues().Resize(1,1);
            CopyImageSizeFromInputs(); 

            // resize the temporaries to their proper size
            size_t cols = Inputs(0)->FunctionValues().GetNumCols();
            m_maxIndexes0.Resize(1,cols);
            m_maxIndexes1.Resize(1,cols);
            m_maxValues.Resize(1,cols);
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
                if (m_maxIndexes0.GetDeviceId() != deviceId)
                    m_maxIndexes0.TransferFromDeviceToDevice(m_maxIndexes0.GetDeviceId(), deviceId,true);

                if (m_maxIndexes1.GetDeviceId() != deviceId)
                    m_maxIndexes1.TransferFromDeviceToDevice(m_maxIndexes1.GetDeviceId(), deviceId,true);

                if (m_maxValues.GetDeviceId() != deviceId)
                    m_maxValues.TransferFromDeviceToDevice(m_maxValues.GetDeviceId(), deviceId,true);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            ErrorPredictionNode<ElemType>* node = (ErrorPredictionNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_maxIndexes0 = m_maxIndexes0;
                node->m_maxIndexes1 = m_maxIndexes1;
                node->m_maxValues = m_maxValues;
            }
        }

        // copy constructor
        ErrorPredictionNode(const ErrorPredictionNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) 
            : ComputationNode<ElemType>(node->m_deviceId), m_maxIndexes0(node->m_deviceId), m_maxIndexes1(node->m_deviceId), m_maxValues(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new ErrorPredictionNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        Matrix<ElemType> m_maxIndexes0, m_maxIndexes1;
        Matrix<ElemType> m_maxValues;
    };

    template class ErrorPredictionNode<float>; 
    template class ErrorPredictionNode<double>;

    /**
    * this node does sequence decoding only
    * it corresponds to a decoder
    */
    template<class ElemType>
    class SequenceDecoderNode : public ComputationNode<ElemType>
    {
    private:
        Matrix<ElemType> mAlpha;
        Matrix<ElemType> mBacktrace;

        int mStartLab; /// the starting output label
        int mEndLab;   /// the ending output label, if avaliable

    public:
        SequenceDecoderNode(const DEVICEID_TYPE  deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), mAlpha(deviceId), mBacktrace(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
            mStartLab = -1;
            mEndLab = -1;
        }

        SequenceDecoderNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE  deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), mAlpha(deviceId), mBacktrace(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
            mStartLab = -1;
            mEndLab = -1;
        }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"SequenceDecoderNode"; }

        static void DecideStartEndingOutputLab(const Matrix<ElemType>& lbls, int & stt, int & stp)
        {
            if (stt != -1 && stp != -1)
                return; /// have computed before

            int iNumPos = lbls.GetNumCols();

            int firstLbl = -1;
            for (int ik = 0; ik < lbls.GetNumRows(); ik++)
            if (lbls(ik, 0) != 0){
                firstLbl = ik; break;
            }

            int lastLbl = -1;
            for (int ik = 0; ik < lbls.GetNumRows(); ik++)
            if (lbls(ik, iNumPos - 1) != 0){
                lastLbl = ik; break;
            }

            stt = firstLbl;
            stp = lastLbl;
        };

        virtual void ComputeInputPartial(const size_t /*inputIndex*/)  //scaled by 2*number of elements in the Matrix<ElemType>
        {
            throw std::logic_error("SequenceDecoder is used for evaluation only.");
        }

        /// compute posterior probability of label y at position t
        virtual void EvaluateThisNode()
        {
            DecideStartEndingOutputLab(Inputs(0)->FunctionValues(), mStartLab, mEndLab);
            EvaluateThisNodeS(mAlpha, mBacktrace, FunctionValues(), Inputs(1)->FunctionValues(),
                Inputs(2)->FunctionValues(), mStartLab, mEndLab);
        }

        /// compute forward backward algorithm
        static void EvaluateThisNodeS(Matrix<ElemType>& alpha, Matrix<ElemType>& backtrace, Matrix<ElemType>& functionValues, const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, const size_t stt, const size_t stp)
        {
            /// to-do, each slice is for one sentence
            /// to-do, number of slices correspond to number of frames 
            /// this implementation only supports one sentence per minibatch

            /// change to other values so can support multiple sentences in each minibatch
            assert(iStep == 1);
            ForwardCompute(alpha, backtrace, pos_scores, pair_scores, stt);
            BackwardCompute(functionValues, backtrace, stp);

        };

        /// compute forward backward algorithm
        static void ForwardCompute(Matrix<ElemType>& alpha,
            Matrix<ElemType>& backtrace,
            const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores,
            const size_t stt)
        {
            /// to-do, shift more than 1 to support muliple sentences per minibatch
            int iNumPos = pos_scores.GetNumCols();
            int iNumLab = pos_scores.GetNumRows();
            size_t iTmp = 0;

            /// need to have 
            alpha.Resize(iNumLab, iNumPos);
            backtrace.Resize(iNumLab, iNumPos);

            for (int t = 0; t < iNumPos; t++)
            {
                for (int k = 0; k < iNumLab; k++)
                {
                    ElemType fTmp = (ElemType)LZERO;
                    if (t > 1){
                        for (int j = 0; j < iNumLab; j++)
                        {
                            ElemType fAlpha = alpha(j, t - 1) + pair_scores(k, j);
                            if (fAlpha > fTmp){
                                fTmp = fAlpha;
                                iTmp = j;
                            }
                        }
                        fTmp += pos_scores(k, t);  /// include position dependent score
                    }
                    else
                    {
                        /// with constrain that the first word is labeled as a given symbol
                        iTmp = stt;
                        fTmp = 0;
                        if (t == 1){
                            fTmp = alpha(iTmp, t - 1);
                            fTmp += pair_scores(k, iTmp);
                            fTmp += pos_scores(k, t);
                        }
                        else {
                            fTmp = (k == stt) ? pos_scores(k, t) : (ElemType)LZERO;
                        }
                    }
                    alpha(k, t) = fTmp;
                    backtrace(k, t) = (ElemType)iTmp;
                }
            }

        };

        /// compute backward algorithm
        static void BackwardCompute(
            Matrix<ElemType>& decodedpath,
            const Matrix<ElemType>& backtrace, const size_t stp)
        {
            int iNumPos = backtrace.GetNumCols();
            int iNumLab = backtrace.GetNumRows();

            decodedpath.Resize(iNumLab, iNumPos);
            decodedpath.SetValue(0);

            size_t lastlbl = stp;
            decodedpath(lastlbl, iNumPos - 1) = 1;

            for (int t = iNumPos - 1; t > 0; t--)
            {
                lastlbl = (size_t) backtrace(lastlbl, t);
                decodedpath(lastlbl, t - 1) = 1;
            }
        };

        /// need to feed in quesudo label data, which tells the decoder what is the begining
        /// and ending output symbol. these symbols will constrain the search space
        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 3)
                throw std::logic_error("SequenceDecoderNode requires three inputs.");

            if (!(Inputs(1)->FunctionValues().GetNumRows() == Inputs(2)->FunctionValues().GetNumRows() &&  // position dependent and pair scores have same number of labels
                Inputs(0)->FunctionValues().GetNumRows() == Inputs(1)->FunctionValues().GetNumRows() &&
                Inputs(0)->FunctionValues().GetNumCols() == Inputs(1)->FunctionValues().GetNumCols() && // position dependent and pair scores have the same observation numbers
                Inputs(2)->FunctionValues().GetNumCols() == Inputs(2)->FunctionValues().GetNumRows()))
            {
                throw std::logic_error("The Matrix<ElemType>  dimension in the SequenceDecoderNode operation does not match.");
            }

            CopyImageSizeFromInputs();
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, false);

            m_outputChannels = 1;
            m_outputWidth = 1;
            m_outputHeight = 1;
        }

        /// label : output label vector of [0:T-1]
        /// position_dependent_score : score from position dependent node,
        /// in the R-CRF case, it is the RNN output score before softmax
        /// transition score : score from the transition node, 
        /// in the R-CRF case, it is the transition probability between labels
        virtual void AttachInputs(const ComputationNodePtr label,
            const ComputationNodePtr position_dependent_score,
            const ComputationNodePtr transition_score)
        {
            m_children.resize(3);
            m_children[0] = label;
            m_children[1] = position_dependent_score;
            m_children[2] = transition_score;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);

        }

        // copy constructor
        SequenceDecoderNode(const SequenceDecoderNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;

            ComputationNodePtr node = new SequenceDecoderNode<ElemType>(this, name, flags);
            return node;
        }

    };
    template class SequenceDecoderNode<float>;
    template class SequenceDecoderNode<double>;

}}}