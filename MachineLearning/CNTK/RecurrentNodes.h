//
// <copyright file="RecurrentNodes.h" company="Microsoft">
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

/**
to-dos:
delay_node : has another input that points to additional observations. 
memory_node: M x N node, with a argument telling whether to save the last observation, or save a window size of observations, or save all observations
pair_node : copy function values and gradient values from one node in source network to target network

decoder delay_node -> memory_node -> pair(source, target) pair(source, target) -> memory_node -> encoder output node


*/

namespace Microsoft { namespace MSR { namespace CNTK {


    template<class ElemType>
    class DelayNode : public ComputationNode<ElemType>
    {
        ElemType  m_default_activity; 

        UsingComputationNodeMembers;
    public:
        DelayNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")  
            : ComputationNode<ElemType>(deviceId), m_pastActivity(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_reqMultiSeqHandling = true;
            m_default_activity = (ElemType)DEFAULT_HIDDEN_ACTIVITY;
            m_delay = 1;
            m_functionValues.Resize(1,1);
            m_pastActivity.Resize(1,1);
            InitRecurrentNode();
        }
                
        DelayNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_pastActivity(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);

            m_default_activity = (ElemType)DEFAULT_HIDDEN_ACTIVITY;
            m_delay = 1;
            m_functionValues.Resize(1,1);
            m_pastActivity.Resize(1,1);
            m_reqMultiSeqHandling = true;

            LoadFromFile(fstream, modelVersion, deviceId);
        }

        void SaveToFile(File& fstream) const
        {
            fstream << OperationName() << NodeName();
            fstream << m_delay; 
            fstream << FunctionValues().GetNumRows() << FunctionValues().GetNumCols(); 

            fstream << m_default_activity;
        }

        void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
        {
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();

            fstream >> m_delay;

            size_t iRow, timeIdxInSeq;
            fstream >> iRow >> timeIdxInSeq;
            FunctionValues().Resize(iRow,timeIdxInSeq);
            m_pastActivity.Resize(iRow, timeIdxInSeq);

            if (modelVersion >= CNTK_MODEL_VERSION_2)
                fstream >> m_default_activity;
        }

        DelayNode(const DEVICEID_TYPE deviceId, ElemType initHiddenActivity, size_t row_size, size_t col_size, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_reqMultiSeqHandling = true;
            m_default_activity = initHiddenActivity;
            m_delay = 1;

            m_functionValues.Resize(row_size, col_size);
            m_functionValues.SetValue(m_default_activity);

            m_pastActivity.Resize(row_size, col_size);
            m_pastActivity.SetValue(m_default_activity);

            m_gradientValues.Resize(row_size, col_size);
            m_gradientValues.SetValue(0.0f);

            InitRecurrentNode();
        }

        virtual const std::wstring OperationName() const {return TypeName();}

        /**
        Set sentence boundary information according to a specified time delay. 

        This function resets segmentation information according to the specified delay, which is a stream of 0, -1, and 1. 
        For the delay node, the default delay time is 1. In the case of other delay times, the output from delay is multipled by a sentence-length constrained shifted mask padded with zero.
        For example, mask for delay time = 1 is
        0 1 1 0 1 1 1
        Its corresponding mask for delay time = 2 is
        0 0 1 0 0 1 1

        Using the mask for segmentation, it is easy to orgnize data streams. Examples are in the following
        Case 1: Data parallelism with two sentences processed in parallel. Suppose the second sentence is 5 words only. The first format is an interleaved format of two parallel processes
        Use the following 2-row matrix to represent the two processes
        0 1 1 1 0 1 1 0 1
        0 1 1 1 1 1 0 0 0
        Case 2: Data parallelism with two sentences. One is not using sentence truncation (the first stream) and the second stream with two sentences uses sentence truncation
        0 1 1 1 1 1 1 1 1
        0 1 1 1 1 1 0 1 1 

        When delay node reads above, it simply use matrix multiplication on its hidden state and the natural outcome is a reset of hidden state, since 0 times anything is 0.
        With the above way of representing sentence-beginning, the node is flexible to have either BPTT with sentence truncation or BPTT w/o sentence truncation. Also, it can deal with the situation that there are long sentences and short sentences in one minibatch.
        */
        void ResetBound(Matrix<ElemType> * seg, vector<MinibatchPackingFlag> * minibatchPackingFlag)
        {
            ComputationNode<ElemType>::ResetBound(seg, minibatchPackingFlag);
            if (m_delay > 1)
            {
                m_shiftedMinibatchPackingFlag = *minibatchPackingFlag;
                m_boundaryInfo = *seg;

                //each row has a number to indicate how many values should be reset for that utterance
                int numRows = (int)seg->GetNumRows();
                vector<int> numResetLeft;
                numResetLeft.resize(numRows);
                std::fill(numResetLeft.begin(), numResetLeft.end(), 0);

                for (int i = 0; i < minibatchPackingFlag->size(); i++)
                {
                    if ((*minibatchPackingFlag)[i] & MinibatchPackingFlag::UtteranceStartOrNoLabel)
                    {
                        //we set delay-1 elements following it to be UtteranceStart until met NoLabel
                        for (int j = 0; j < numRows; j++)
                        {
                            if ((*seg)(j, i) == SENTENCE_BEGIN)
                            {
                                numResetLeft[j] = m_delay;
                            }
                            else if ((*seg)(j, i) == NO_LABELS)
                            {
                                numResetLeft[j] = 0;
                            }
                        }
                    }

                    //now set the UtteranceStart
                    bool valueChanged = false;
                    for (int j = 0; j < numRows; j++)
                    {
                        if (numResetLeft[j]-- > 0)
                        {
                            m_boundaryInfo(j, i) = SENTENCE_BEGIN;
                            valueChanged = true;
                        }
                    }

                    if (valueChanged)
                    {
                        m_shiftedMinibatchPackingFlag[i] |= MinibatchPackingFlag::UtteranceStart;
                    }
                }

                m_minibatchPackingFlag = &m_shiftedMinibatchPackingFlag;
                m_sentenceSeg = &m_boundaryInfo;
            }

            if (m_delay <= 0)
                LogicError("Delay should be 1 or larger");
        }

        /// to-do: need to change to the new way of resetting state
        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("Delay operation only takes one input.");

            size_t nbrSamples = GradientValues().GetNumCols() / m_samplesInRecurrentStep; 
            for (int i = nbrSamples - 1; i >= 0; i--)
            {
                ComputeInputPartialSR(i, m_delay, Inputs(0)->GradientValues(), GradientValues(), m_samplesInRecurrentStep);
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("Delay operation only takes one input.");
            assert(m_functionValues.GetNumRows() == GradientValues().GetNumRows()); // original used m_functionValues.GetNumRows() for loop dimension
            assert(m_sentenceSeg != nullptr);
            assert(m_minibatchPackingFlag != nullptr);

            Matrix<ElemType> colBegin(m_sentenceSeg->GetDeviceId());
            colBegin = m_sentenceSeg->ColumnSlice(timeIdxInSeq, 1);

            ComputeInputPartialSRP(timeIdxInSeq, m_delay, Inputs(0)->GradientValues(), GradientValues(), m_samplesInRecurrentStep, colBegin, (*m_minibatchPackingFlag)[timeIdxInSeq]);
        }

        /// to-do: need to change to the new way of resetting state
        static void WINAPI ComputeInputPartialSR(int timeIdxInSeq, int delay,
            Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const size_t mNbr)
        {
            assert(timeIdxInSeq >= 0);
            if ((timeIdxInSeq - delay) >= 0 && (timeIdxInSeq - delay) * mNbr <= inputGradientValues.GetNumCols())
            {
                Matrix<ElemType> to = inputGradientValues.ColumnSlice((timeIdxInSeq - delay)*mNbr, mNbr);
                Matrix<ElemType> frm= gradientValues.ColumnSlice(timeIdxInSeq * mNbr, mNbr);
                to += frm; 
            }
        }

        static void WINAPI ComputeInputPartialSRP(int timeIdxInSeq, int delay, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const size_t mNbr, const Matrix<ElemType>& colBegin, MinibatchPackingFlag minibatchPackingFlag)
        {
            assert(timeIdxInSeq >= 0);
            if ((timeIdxInSeq - delay) >= 0)
            {
                if (minibatchPackingFlag & MinibatchPackingFlag::UtteranceStartOrNoLabel)
                {                   
                    for (int i = 0; i < mNbr; i++)
                    {
                        if (colBegin(i,0) == SENTENCE_MIDDLE)
                        {
                            Matrix<ElemType> to = inputGradientValues.ColumnSlice((timeIdxInSeq - delay)*mNbr + i, 1);
                            Matrix<ElemType> frm= gradientValues.ColumnSlice(timeIdxInSeq * mNbr + i, 1);

                            to += frm;
                        }
                    }

                }
                else
                {
                    Matrix<ElemType> frm = gradientValues.ColumnSlice(timeIdxInSeq * mNbr, mNbr);
                    Matrix<ElemType> to = inputGradientValues.ColumnSlice((timeIdxInSeq - delay)*mNbr, mNbr);

                    to += frm;
                }
            }
        }


        virtual void EvaluateThisNode()  
        {
            ASSERT(m_delay > 0);
            size_t blogSize = Inputs(0)->FunctionValues().GetNumCols();

            for (size_t i = 0; i < blogSize / m_samplesInRecurrentStep; i++)
                EvaluateThisNodeSR(i, m_delay, true, m_default_activity, m_functionValues, m_pastActivity, Inputs(0)->FunctionValues(), m_samplesInRecurrentStep);
            /// reset past activity
            m_pastActivity = Inputs(0)->FunctionValues();
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            /// reset past activity as it reached to the begining of a minibatch
            /// the node pointed hasn't yet updated, so it is the past activity 
            assert(m_sentenceSeg != nullptr);
            assert(m_minibatchPackingFlag != nullptr);

            if (timeIdxInSeq == 0)
            {
                m_pastActivity = Inputs(0)->FunctionValues();
            }
            
            Matrix<ElemType> colBegin(m_sentenceSeg->GetDeviceId());
            colBegin = m_sentenceSeg->ColumnSlice(timeIdxInSeq, 1);
            EvaluateThisNodeSRP(timeIdxInSeq, m_delay, m_functionValues, m_pastActivity, Inputs(0)->FunctionValues(), m_samplesInRecurrentStep, m_default_activity, colBegin, (*m_minibatchPackingFlag)[timeIdxInSeq]);

        }

        /// to-do: need to change to the new way of resetting state
        static void WINAPI EvaluateThisNodeSR(const size_t timeIdxInSeq, const int delay, const bool reset, const ElemType default_activity, Matrix<ElemType>& functionValues, const Matrix<ElemType>& pastActivity, const Matrix<ElemType>& inputFunctionValues, const size_t mNbr)
        {
            ASSERT(delay > 0);

            if (functionValues.GetNumRows() != inputFunctionValues.GetNumRows() ||
                functionValues.GetNumCols() != inputFunctionValues.GetNumCols())
                functionValues.Resize(inputFunctionValues.GetNumRows(),
                    inputFunctionValues.GetNumCols());

            int iPastIndex = (int) (timeIdxInSeq - delay) * mNbr;
            int d = iPastIndex; 
            if (d < 0)
                d = (int)functionValues.Mod((float)iPastIndex, (float)pastActivity.GetNumCols());  
           /// this can point to the past activity of the previous mninibatch

            Matrix<ElemType> out = functionValues.ColumnSlice(timeIdxInSeq * mNbr, mNbr);
            Matrix<ElemType> inp((DEVICEID_TYPE)functionValues.GetDeviceId()) ;

            if (reset)
                out.SetValue(default_activity);
            else
            {
                if (iPastIndex < 0)
                    inp = pastActivity.ColumnSlice(d, mNbr);
                else
                    inp = inputFunctionValues.ColumnSlice(d, mNbr);
                out.SetValue(inp);
            }
        }

        /**
        This function returns output from the previous time instance.For recurrent network, the initial state needs to be set in the case of sentence begining, which is carried over from colBegin.In case of sentence begining, the state activity is set to an initial value.The colBegin has element of SENTENCE_BEGIN, SENTENCE_MIDDLE and NO_LABELS, which are 0, 1, and - 1, respectively.
            To compute the initial value, we use
            prevState = colBegin * pastActivity + ~colBegin * initialStateValue
            and ~sentenceBegin is computed as - 1 * (colBegin - 1), assuming that colBegin is either 0 or 1. For example, when colBegin == 1, ~sentenceBegin == 0.
        colBegin is truncated to the range of 0 to 1 to satisify the assumption. For NO_LABELS case, it is converted to its absolute value, which is 1, and treated in ~colBegin as SENTENCE_MIDDLE, which results in 0 so that zero is output for NO_LABELS case. 
        */
        static void WINAPI EvaluateThisNodeSRP(const size_t timeIdxInSeq, const int delay, Matrix<ElemType>& functionValues, const Matrix<ElemType>& pastActivity, const Matrix<ElemType>& inputFunctionValues, const size_t mNbr, const ElemType & initStateValue, const Matrix<ElemType> & colBegin, const MinibatchPackingFlag minibatchPackingFlag)
        {
            ASSERT(delay > 0);

            if (functionValues.GetNumRows() != inputFunctionValues.GetNumRows() ||
                functionValues.GetNumCols() != inputFunctionValues.GetNumCols())
                functionValues.Resize(inputFunctionValues.GetNumRows(),
                    inputFunctionValues.GetNumCols());

            int iPastIndex = (int) (timeIdxInSeq - delay) * mNbr;
            int d = iPastIndex; 
            if (d < 0)
                d = (int)functionValues.Mod((float)iPastIndex, (float)pastActivity.GetNumCols());  
            /// this can point to the past activity of the previous mninibatch

            Matrix<ElemType> out = functionValues.ColumnSlice(timeIdxInSeq * mNbr, mNbr);
            Matrix<ElemType> inp((DEVICEID_TYPE)functionValues.GetDeviceId()) ;

            if (minibatchPackingFlag & MinibatchPackingFlag::UtteranceStartOrNoLabel)
            {
                for (int i = 0; i < mNbr; i ++)
                {
                    out = functionValues.ColumnSlice(timeIdxInSeq * mNbr + i,1);
                    if (iPastIndex < 0)
                        inp = pastActivity.ColumnSlice(d+i, 1);
                    else
                        inp = inputFunctionValues.ColumnSlice(d+i, 1);

                    if (colBegin(i,0) == SENTENCE_BEGIN)
                    {
                        out.SetValue(initStateValue);
                    }else
                    {
                        out.SetValue(inp);
                    }
                }
                //colSeg.SetDiagonalValue(colSegPastActivity);
                //Matrix<ElemType>::Multiply(inp, false, colSeg, false, out);

                //SetToInitStateValueForResetSeg(colBegin, mNbr, initStateValue, out);
            }
            else
            {
                if (iPastIndex < 0)
                    inp = pastActivity.ColumnSlice(d, mNbr);
                else
                    inp = inputFunctionValues.ColumnSlice(d, mNbr);


                out.SetValue(inp);
            }
        }



        virtual const Matrix<ElemType>& FunctionValues() const 
        {
            return m_functionValues;
        }

        virtual Matrix<ElemType>& FunctionValues() 
        {
            return m_functionValues;
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation(true/*allowNulls*/);

            if (m_children.size() != 1) 
                throw std::logic_error("Delay operation should have one input.");

            if (!(Inputs(0) == nullptr))
            {
                size_t rows0 = Inputs(0)->FunctionValues().GetNumRows(), cols0 = Inputs(0)->FunctionValues().GetNumCols();

                if (rows0 > 0 && cols0 > 0) FunctionValues().Resize(rows0, cols0);
            }
            CopyImageSizeFromInputs(); 
        }



        virtual void AttachInputs(const ComputationNodePtr inputNode)
        {
            m_children.resize(1);
            m_children[0] = inputNode;
        }

        void SetDelay(const int val)
        {
            if (val <= 0)
                throw std::logic_error("Delay must be > 0.");
            m_delay = val;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_boundaryInfo.GetDeviceId() != deviceId)
                    m_boundaryInfo.TransferFromDeviceToDevice(m_boundaryInfo.GetDeviceId(), deviceId);
                if (m_pastActivity.GetDeviceId() != deviceId)
                    m_pastActivity.TransferFromDeviceToDevice(m_pastActivity.GetDeviceId(), deviceId, true);
            }
        }

        static const std::wstring TypeName() {return L"Delay";} 

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            DelayNode<ElemType>* node = (DelayNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_delay = m_delay;
                node->m_default_activity = m_default_activity;
                node->m_pastActivity = m_pastActivity;
            }
        }

        // copy constructor
        DelayNode(const DelayNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_pastActivity(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new DelayNode<ElemType>(this, name, flags);
            return node;
        }

    protected:
        virtual bool UseCustomizedMultiSeqHandling() { return true; }

    private:
        Matrix<ElemType> m_pastActivity;  /// saves the past activity this delay node points to
        int      m_delay;    /// steps for delay 
        vector<MinibatchPackingFlag> m_shiftedMinibatchPackingFlag;
        Matrix<ElemType> m_boundaryInfo; /// individual sentence boundary information 
    };

    template class DelayNode<float>; 
    template class DelayNode<double>;

    /**
    LSTM specific node. This node uses matrix operations to have LSTM functionality. 
    It avoids using general recurrent loop operations in the network operations in computationnetwork. 


    Developed by Kaisheng Yao
    Used in the following works:
    K. Yao, G. Zweig, "Sequence to sequence neural net models for graphone to phoneme conversion", submitted to Interspeech 2015
    */
    template<class ElemType>
    class LSTMNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;

    public:
        LSTMNode(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_State(deviceId), m_PastState(deviceId),
            m_PastOutput(deviceId), m_Gi(deviceId), m_Gf(deviceId), m_Go(deviceId), grdToObs(deviceId), grdToInputGate(deviceId),
            grdToForgetGate(deviceId), grdToOutputGate(deviceId), grdToCellWgt(deviceId), tanhObs(deviceId),
            tanhState(deviceId), m_tempMatrix(deviceId),
            mSlicePrevState(deviceId), mSlicePrevOutput(deviceId),
            grdBeforeInputGate(deviceId),
            grdBeforeForget(deviceId), grdBeforeGo(deviceId), grdToCell(deviceId),
            grdBeforeTanhInputGate(deviceId), m_obs_error_from_future_minibatch(deviceId),
            m_state_error_from_future_minibatch(deviceId), mLastState(deviceId), mLastOutput(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_reqMultiSeqHandling = true;
            InitRecurrentNode();
            m_inputDim = 0;
            m_outputDim = 0;
            m_use_errors_from_future_minibatch = false;
            m_DefaultState = (ElemType) DEFAULT_HIDDEN_ACTIVITY;
        }

        LSTMNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode<ElemType>(deviceId), m_State(deviceId), m_PastState(deviceId), m_PastOutput(deviceId), m_Gi(deviceId), m_Gf(deviceId), m_Go(deviceId), grdToObs(deviceId), grdToInputGate(deviceId), grdToForgetGate(deviceId), grdToOutputGate(deviceId), grdToCellWgt(deviceId), tanhObs(deviceId), tanhState(deviceId), m_tempMatrix(deviceId), mSlicePrevState(deviceId), mSlicePrevOutput(deviceId),
            grdBeforeInputGate(deviceId),
            grdBeforeForget(deviceId), grdBeforeGo(deviceId), grdToCell(deviceId),
            grdBeforeTanhInputGate(deviceId), m_obs_error_from_future_minibatch(deviceId),
            m_state_error_from_future_minibatch(deviceId), mLastState(deviceId), mLastOutput(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            m_inputDim = 0;
            m_outputDim = 0;
            m_reqMultiSeqHandling = true;
            m_use_errors_from_future_minibatch = false;
            m_DefaultState = (ElemType)DEFAULT_HIDDEN_ACTIVITY;
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        LSTMNode(const LSTMNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode<ElemType>(node->m_deviceId), m_State(node->m_deviceId), m_PastState(node->m_deviceId), m_PastOutput(node->m_deviceId), m_Gi(node->m_deviceId), m_Gf(node->m_deviceId), m_Go(node->m_deviceId), grdToObs(node->m_deviceId), grdToInputGate(node->m_deviceId), grdToForgetGate(node->m_deviceId), grdToOutputGate(node->m_deviceId), grdToCellWgt(node->m_deviceId), tanhObs(node->m_deviceId), tanhState(node->m_deviceId), m_tempMatrix(node->m_deviceId), mSlicePrevState(node->m_deviceId), mSlicePrevOutput(node->m_deviceId),
            grdBeforeInputGate(node->m_deviceId),
            grdBeforeForget(node->m_deviceId), grdBeforeGo(node->m_deviceId), grdToCell(node->m_deviceId),
            grdBeforeTanhInputGate(node->m_deviceId), m_obs_error_from_future_minibatch(node->m_deviceId),
            m_state_error_from_future_minibatch(node->m_deviceId), mLastState(node->m_deviceId), mLastOutput(node->m_deviceId)
        {
            m_use_errors_from_future_minibatch = false;
            node->CopyTo(this, newName, flags);
            m_DefaultState = (ElemType) DEFAULT_HIDDEN_ACTIVITY;
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;

            ComputationNodePtr node = new LSTMNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"LSTM"; }

        virtual void SaveToFile(File& fstream) const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << m_inputDim << m_outputDim;
            fstream << m_DefaultState;
        }

        void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            if (modelVersion == 2)
                fstream >> m_inputDim >> m_outputDim;
            fstream >> m_DefaultState;
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            LSTMNode<ElemType>* node = (LSTMNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_inputDim = m_inputDim;
                node->m_outputDim = m_outputDim;

                node->m_State = m_State;  /// hidden state activity
                node->m_PastState = m_PastState; /// state activity in the previous minibatch
                node->m_PastOutput = m_PastOutput; /// output in the previou minibatch 

                node->m_Gi = m_Gi;     /// input gate activity
                node->m_Gf = m_Gf;     /// forget gate activity
                node->m_Go = m_Go;     /// output gate activity

                node->mSlicePrevOutput = mSlicePrevOutput;
                node->mSlicePrevState = mSlicePrevState;

                node->m_use_errors_from_future_minibatch = m_use_errors_from_future_minibatch;

                node->m_DefaultState = m_DefaultState;
                node->m_reqMultiSeqHandling = m_reqMultiSeqHandling;
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 4)
                throw std::invalid_argument("LSTM operation only takes five inputs.");

            size_t nT = Inputs(0)->FunctionValues().GetNumCols();
            size_t inputDim = Inputs(0)->FunctionValues().GetNumRows();
            size_t outputDim = Inputs(1)->FunctionValues().GetNumRows();

            if (m_GradientComputed == false)
            {
                if (FunctionValues().GetNumCols() != GradientValues().GetNumCols() ||
                    FunctionValues().GetNumRows() != GradientValues().GetNumRows())
                {
                    throw std::runtime_error("LSTMNode::GradientValue size doesn't match to the function value size");
                }

                /// reset gradients
                grdToObs.Resize(inputDim, nT); grdToObs.SetValue(0);
                grdToInputGate.Resize(Inputs(1)->FunctionValues().GetNumRows(), Inputs(1)->FunctionValues().GetNumCols()); grdToInputGate.SetValue(0);
                grdToForgetGate.Resize(Inputs(2)->FunctionValues().GetNumRows(), Inputs(2)->FunctionValues().GetNumCols()); grdToForgetGate.SetValue(0);
                grdToOutputGate.Resize(Inputs(3)->FunctionValues().GetNumRows(), Inputs(3)->FunctionValues().GetNumCols()); grdToOutputGate.SetValue(0);
                grdToCellWgt.Resize(Inputs(4)->FunctionValues().GetNumRows(), Inputs(4)->FunctionValues().GetNumCols()); grdToCellWgt.SetValue(0);

                Matrix<ElemType> slicePrevOutput(m_deviceId), slicePrevState(m_deviceId);
                Matrix<ElemType> grdToPrevOutput(m_deviceId), grdToPrevState(m_deviceId);
                Matrix<ElemType> stateError(m_deviceId);
                slicePrevState.Resize(outputDim, m_samplesInRecurrentStep);
                slicePrevOutput.Resize(outputDim, m_samplesInRecurrentStep);
                slicePrevOutput.SetValue(0);

                stateError.Resize(slicePrevState.GetNumRows(), slicePrevState.GetNumCols());

                grdToPrevOutput.Resize(slicePrevOutput.GetNumRows(), slicePrevOutput.GetNumCols());
                grdToPrevState.Resize(slicePrevState.GetNumRows(), slicePrevState.GetNumCols());
                grdToPrevOutput.SetValue(0);
                grdToPrevState.SetValue(0);

                for (int timeIdxInSeq = nT - m_samplesInRecurrentStep; timeIdxInSeq >= 0; timeIdxInSeq -= m_samplesInRecurrentStep)
                {
                    Matrix<ElemType> sliceObs = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceOutput = FunctionValues().ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceState = m_State.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);

                    Matrix<ElemType> sliceGi = m_Gi.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceGf = m_Gf.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceGo = m_Go.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);

                    Matrix<ElemType> sliceTanhState = tanhState.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceTanhObs = tanhObs.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);

                    Matrix<ElemType> error = GradientValues().ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);

                    Matrix<ElemType> grdToObsSlice(this->m_deviceId);


#ifdef DEBUG_DECODER
                    fprintf(stderr, "original output error [%ld] norm = %.8e\n", timeIdxInSeq, error.FrobeniusNorm());
#endif

                    PrepareThisErrorsBeforeBackProp(timeIdxInSeq, nT, error, stateError, grdToPrevOutput, grdToPrevState,
                        m_obs_error_from_future_minibatch, m_state_error_from_future_minibatch, m_samplesInRecurrentStep, m_sentenceSeg);

#ifdef DEBUG_DECODER
                    fprintf(stderr, "output error [%ld] norm = %.8e\n", timeIdxInSeq, error.FrobeniusNorm());
                    fprintf(stderr, "state error [%ld] norm = %.8e\n", timeIdxInSeq, stateError.FrobeniusNorm());
#endif

                    grdToPrevOutput.Resize(slicePrevOutput.GetNumRows(), slicePrevOutput.GetNumCols());
                    grdToPrevState.Resize(slicePrevState.GetNumRows(), slicePrevState.GetNumCols());
                    grdToPrevOutput.SetValue(0);
                    grdToPrevState.SetValue(0);

                    PrepareHistory(timeIdxInSeq, mSlicePrevOutput, mSlicePrevState, FunctionValues(), m_State, m_PastOutput, m_PastState, m_samplesInRecurrentStep, m_DefaultState, m_sentenceSeg);

                    try{
                        ComputeInputGradientWrtGates(
                            error,
                            sliceObs,
                            grdToObsSlice,
                            Inputs(1)->FunctionValues(),
                            grdToInputGate,
                            Inputs(2)->FunctionValues(),
                            grdToForgetGate,
                            Inputs(3)->FunctionValues(),
                            grdToOutputGate,
                            Inputs(4)->FunctionValues(),
                            grdToCellWgt,
                            mSlicePrevOutput,
                            mSlicePrevState,
                            stateError,
                            sliceState,
                            sliceTanhState,
                            sliceTanhObs,
                            sliceGi,
                            sliceGf,
                            sliceGo,
                            grdToPrevOutput,
                            grdToPrevState,
                            m_tempMatrix
                            );
                    }
                    catch (...)
                    {
                        fprintf(stderr, "Error in computing gradient in function ComputeInputPartial for LSTMnode at position %ld, length %ld", timeIdxInSeq, nT);
                        throw;
                    }
                    grdToObs.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep).SetValue(grdToObsSlice);

                    PrepareErrors(timeIdxInSeq, grdToPrevOutput, grdToPrevState, m_samplesInRecurrentStep, m_sentenceSeg);
                }
#ifdef DEBUG_DECODER
                fprintf(stderr, "after error prop b_c norm = %.8e\n", Inputs(4)->FunctionValues().ColumnSlice(0, 1).FrobeniusNorm());
#endif
                m_obs_error_from_future_minibatch = grdToPrevOutput;
                m_state_error_from_future_minibatch = grdToPrevState;


#ifdef DEBUG_DECODER
                fprintf(stderr, "pass error to encoder error = %.4e state error = %.4e\n", m_obs_error_from_future_minibatch.FrobeniusNorm(), m_state_error_from_future_minibatch.FrobeniusNorm());
#endif
                m_GradientComputed = true;
            }

            if (inputIndex == 0)  //derivative with regard to the observation
            {
                if (Inputs(inputIndex)->GradientValues().GetNumElements() == 0)
                    Inputs(inputIndex)->GradientValues().SetValue(grdToObs);
                else
                    Inputs(inputIndex)->GradientValues() += grdToObs;
            }

            if (inputIndex == 1)
            {
                if (Inputs(inputIndex)->GradientValues().GetNumElements() == 0)
                    Inputs(inputIndex)->GradientValues().SetValue(grdToInputGate);
                else
                    Inputs(inputIndex)->GradientValues() += grdToInputGate;
            }

            if (inputIndex == 2)
            {
                if (Inputs(inputIndex)->GradientValues().GetNumElements() == 0)
                    Inputs(inputIndex)->GradientValues().SetValue(grdToForgetGate);
                else
                    Inputs(inputIndex)->GradientValues() += grdToForgetGate;
            }

            if (inputIndex == 3)
            {
                if (Inputs(inputIndex)->GradientValues().GetNumElements() == 0)
                    Inputs(inputIndex)->GradientValues().SetValue(grdToOutputGate);
                else
                    Inputs(inputIndex)->GradientValues() += grdToOutputGate;
            }

            if (inputIndex == 4)
            {
                if (Inputs(inputIndex)->GradientValues().GetNumElements() == 0)
                    Inputs(inputIndex)->GradientValues().SetValue(grdToCellWgt);
                else
                    Inputs(inputIndex)->GradientValues() += grdToCellWgt;
            }
#ifdef DEBUG_DECODER
            fprintf(stderr, "LSTM gradient[%d] norm = %.8e\n", inputIndex, Inputs(inputIndex)->GradientValues().FrobeniusNorm());
#endif

        }

        static void WINAPI GradientOfTanh(const Matrix<ElemType>& functionValues,
            const Matrix<ElemType>& gradientOut,
            Matrix<ElemType>& inputGradientValues,
            Matrix<ElemType>& extTmp)
        {
            Matrix<ElemType> mTmp(inputGradientValues.GetDeviceId());
            extTmp.AssignElementProductOf(functionValues, functionValues); // v .* v
            mTmp.AssignDifferenceOf(1, extTmp); // 1-v^2
            if (inputGradientValues.GetNumRows() != functionValues.GetNumRows() ||
                inputGradientValues.GetNumCols() != functionValues.GetNumCols())
                throw std::logic_error("LSTMNode::GradientOfTanh : inputGradientValues need to be pre-allocated!");
            inputGradientValues.AddElementProductOf(gradientOut, mTmp); //  d .* ((1-v) .* v))
        }

        static void WINAPI ComputeInputGradientWrtGates(
            const Matrix<ElemType>& outGrd,  /// the error to h_t from upper layer
            const Matrix<ElemType> & obs,
            Matrix<ElemType> &grdToObs,
            const Matrix<ElemType>& mInputGate,
            Matrix<ElemType> &grdToInputGate,
            const Matrix<ElemType> &mForgetGate,
            Matrix<ElemType> &grdToForgetGate,
            const Matrix<ElemType> &mOutputGate,
            Matrix<ElemType>& grdToOutputGate,
            const Matrix<ElemType> &mCellWgt,
            Matrix<ElemType> &grdToCellWgt,
            const Matrix<ElemType>& prevOutput,
            const Matrix<ElemType>& prevState,
            const Matrix<ElemType>& stateError,  /// the error propagated to cell from t+1
            const Matrix<ElemType> &state,
            const Matrix<ElemType> &tanhState,
            const Matrix<ElemType> & tanhBeforeApplyingInputGating,
            const Matrix<ElemType> &gi,
            const Matrix<ElemType> &gf,
            const Matrix<ElemType> &go,
            Matrix<ElemType> &grdToPrevOutput,
            Matrix<ElemType> &grdToPrevState,
            Matrix<ElemType> & tmpMat
            )
        {
            int inputDim = obs.GetNumRows();
            int outputDim = mOutputGate.GetNumRows();

            assert(grdToPrevOutput.FrobeniusNorm() == 0);
            assert(grdToPrevState.FrobeniusNorm() == 0);
            assert(state.FrobeniusNorm() > 0);
            Matrix<ElemType> Who = mOutputGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> Wco = mOutputGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> Wxo = mOutputGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdToWho = grdToOutputGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdToWco = grdToOutputGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> grdToWxo = grdToOutputGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdTobo = grdToOutputGate.ColumnSlice(0, 1);

            Matrix<ElemType> Whf = mForgetGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> Wcf = mForgetGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> Wxf = mForgetGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdToWhf = grdToForgetGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdToWcf = grdToForgetGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> grdToWxf = grdToForgetGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdTobf = grdToForgetGate.ColumnSlice(0, 1);

            Matrix<ElemType> Wxc = mCellWgt.ColumnSlice(1, inputDim);
            Matrix<ElemType> Whc = mCellWgt.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdToWxc = grdToCellWgt.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdToWhc = grdToCellWgt.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdTobc = grdToCellWgt.ColumnSlice(0, 1);

            Matrix<ElemType> Whi = mInputGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> Wci = mInputGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> Wxi = mInputGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdToWhi = grdToInputGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdToWci = grdToInputGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> grdToWxi = grdToInputGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdTobi = grdToInputGate.ColumnSlice(0, 1);

            /// error backpropagate to output gate
            Matrix<ElemType> grdToGo(tmpMat.GetDeviceId()), gradientOfSigmoid(tmpMat.GetDeviceId());
            Matrix<ElemType> grdBeforeGo(tmpMat.GetDeviceId()), grdBeforeInputGate(tmpMat.GetDeviceId());
            Matrix<ElemType> grdToCell(tmpMat.GetDeviceId());

            tmpMat.AssignElementProductOf(outGrd, tanhState);  // error to o_t
            gradientOfSigmoid.AssignSigmoidDerivativeOf(go);
            grdBeforeGo.AssignElementProductOf(tmpMat, gradientOfSigmoid);  // error before softmax
#ifdef DEBUG_DECODER
            fprintf(stderr, "output gate error = %.4e\n", grdBeforeGo(0, 0));
#endif
            Matrix<ElemType>::MultiplyAndAdd(Who, true, grdBeforeGo, false, grdToPrevOutput);  /// error to previous output
            Matrix<ElemType>::MultiplyAndAdd(Wxo, true, grdBeforeGo, false, grdToObs);      /// error to observation 
            tmpMat = grdBeforeGo;
            tmpMat.ColumnElementMultiplyWith(Wco);
            grdToCell = tmpMat;                                                            /// error to memory cell

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeGo, false, prevOutput, true, grdToWho); /// gradient to Who
            Matrix<ElemType>::MultiplyAndAdd(grdBeforeGo, false, obs, true, grdToWxo); /// gradient to Wxo
            tmpMat.AssignInnerProductOf(grdBeforeGo, state, false);
            grdToWco += tmpMat;                    /// to Wco
            for (size_t i = 0; i < grdBeforeGo.GetNumCols(); i++)
            {
                grdTobo += grdBeforeGo.ColumnSlice(i, 1);  /// gradient to bo
            }

            grdToGo.AssignElementProductOf(outGrd, go);  // error to tanh
            GradientOfTanh(tanhState, grdToGo, grdToCell, tmpMat); // error to memory cell
            grdToCell += stateError; /// add error to memory cell from t+1
#ifdef DEBUG_DECODER
            fprintf(stderr, "previous state[0] = %.4e norm = %.4e\n", prevState(0, 0), prevState.FrobeniusNorm());
            fprintf(stderr, "state error = %.4e\n", grdToCell(0, 0));
            fprintf(stderr, "state error norm = %.4e\n", grdToCell.FrobeniusNorm());
#endif
            /// error backpropagate to memory cells
            grdToPrevState.AssignElementProductOf(gf, grdToCell);  // error to previous memory cell
            /// be careful, need to double check if errors are missing

            Matrix<ElemType> grdBeforeForget(tmpMat.GetDeviceId());
            tmpMat.AssignElementProductOf(prevState, grdToCell);  // error to f_t
            gradientOfSigmoid.AssignSigmoidDerivativeOf(gf);
            grdBeforeForget.AssignElementProductOf(gradientOfSigmoid, tmpMat); /// error before forget gate
#ifdef DEBUG_DECODER
            fprintf(stderr, "forget gate error = %.4e\n", grdBeforeForget(0, 0));
#endif

            Matrix<ElemType>::MultiplyAndAdd(Whf, true, grdBeforeForget, false, grdToPrevOutput);  /// error to previous output
            tmpMat = grdBeforeForget;
            tmpMat.ColumnElementMultiplyWith(Wcf);
            grdToPrevState += tmpMat;                                                            /// error to previous state

            Matrix<ElemType>::MultiplyAndAdd(Wxf, true, grdBeforeForget, false, grdToObs);  /// error to observation

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeForget, false, prevOutput, true, grdToWhf); /// gradient to Whf
            tmpMat.AssignInnerProductOf(grdBeforeForget, prevState, false);
            grdToWcf += tmpMat;                                                             /// gradient to Wcf

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeForget, false, obs, true, grdToWxf); /// gradient to Wxf
            for (size_t i = 0; i < grdBeforeForget.GetNumCols(); i++)
                grdTobf += grdBeforeForget.ColumnSlice(i, 1);                                                    /// gradient to bf

            /// error backpropagate to input gate
            tmpMat.AssignElementProductOf(tanhBeforeApplyingInputGating, grdToCell);
            gradientOfSigmoid.AssignSigmoidDerivativeOf(gi);
            grdBeforeInputGate.AssignElementProductOf(gradientOfSigmoid, tmpMat); /// error before input gate
#ifdef DEBUG_DECODER
            fprintf(stderr, "input gate error = %.4e\n", grdBeforeInputGate(0, 0));
#endif

            Matrix<ElemType>::MultiplyAndAdd(Whi, true, grdBeforeInputGate, false, grdToPrevOutput);  /// error to previous output
            tmpMat = grdBeforeInputGate;
            tmpMat.ColumnElementMultiplyWith(Wci);
            grdToPrevState += tmpMat;                                                            /// error to previous state

#ifdef DEBUG_DECODER
            fprintf(stderr, "to previous state error = %.4e\n", grdToPrevState(0, 0));
            fprintf(stderr, "to previous state error norm = %.4e\n", grdToPrevState.FrobeniusNorm());
#endif
            Matrix<ElemType>::MultiplyAndAdd(Wxi, true, grdBeforeInputGate, false, grdToObs);  /// error to observation

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeInputGate, false, prevOutput, true, grdToWhi); /// gradient to Whi
            tmpMat.AssignInnerProductOf(grdBeforeInputGate, prevState, false);
            grdToWci += tmpMat;                                                             /// gradient to Wci
            Matrix<ElemType>::MultiplyAndAdd(grdBeforeInputGate, false, obs, true, grdToWxi); /// gradient to Wxi
            for (size_t i = 0; i < grdBeforeInputGate.GetNumCols(); i++)
                grdTobi += grdBeforeInputGate.ColumnSlice(i, 1);                                                  /// gradient to bi

            /// error backpropagate to inputs
            Matrix<ElemType> grdTmp2(tmpMat.GetDeviceId());
            Matrix<ElemType> grdBeforeTanhInputGate(tmpMat.GetDeviceId());
            grdTmp2.AssignElementProductOf(gi, grdToCell);
            grdBeforeTanhInputGate.Resize(tanhBeforeApplyingInputGating.GetNumRows(), tanhBeforeApplyingInputGating.GetNumCols());
            GradientOfTanh(tanhBeforeApplyingInputGating, grdTmp2, grdBeforeTanhInputGate, tmpMat); // error to memory cell
            Matrix<ElemType>::MultiplyAndAdd(Wxc, true, grdBeforeTanhInputGate, false, grdToObs);  /// error to observation
#ifdef DEBUG_DECODER
            fprintf(stderr, "to observation error = %.4e\n", grdToObs(0, 0));
#endif

            Matrix<ElemType>::MultiplyAndAdd(Whc, true, grdBeforeTanhInputGate, false, grdToPrevOutput);  /// error to previous output
            Matrix<ElemType>::MultiplyAndAdd(grdBeforeTanhInputGate, false, obs, true, grdToWxc); /// gradient to Wxc

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeTanhInputGate, false, prevOutput, true, grdToWhc); /// gradient to Whc
            for (size_t i = 0; i < grdBeforeTanhInputGate.GetNumCols(); i++)
                grdTobc += grdBeforeTanhInputGate.ColumnSlice(i, 1);                                                    /// gradient to bc

        }

        /**
        get the segmentation information, SENTENECE_BEGIN, SENTENCE_MIDDLE, NO_LABELS 
        for time at t and stream of streamid
        */
        int GetSegInfo(size_t t, size_t streamid)
        {
            if (streamid >= m_samplesInRecurrentStep)
                LogicError("GetSegInfo: stream id %d is larger than the number of streams %d", streamid, m_samplesInRecurrentStep);

            size_t nT = Inputs(0)->FunctionValues().GetNumCols();
            if (t >= nT)
                LogicError("GetSegInfo: time %d times is larger than the total number of observations %d", t, nT);

            int utt_t = (int)t / m_samplesInRecurrentStep;
            Matrix<ElemType> thisCol = m_sentenceSeg->ColumnSlice(utt_t, 1);
            thisCol.Reshape(1, m_samplesInRecurrentStep);
            return (int) thisCol.ColumnSlice(streamid, 1).Get00Element();
        }

        /**
        save the last hidden layer activity and output
        */
        void SaveLastStateActity()
        {
            size_t nT = Inputs(0)->FunctionValues().GetNumCols();
            size_t outputDim = Inputs(1)->FunctionValues().GetNumRows();
            
            /// save the hidden activities and output for the next minibatch
            mLastOutput.Resize(outputDim, m_samplesInRecurrentStep);
            mLastState.Resize(outputDim, m_samplesInRecurrentStep);

            for (size_t i = 0; i < m_samplesInRecurrentStep; i++)
            {
                for (int t = nT - m_samplesInRecurrentStep + i; t >= 0; t -= m_samplesInRecurrentStep)
                {
                    if (GetSegInfo(t, i) == SENTENCE_MIDDLE)
                    {
                        mLastOutput.ColumnSlice(i, 1).SetValue(FunctionValues().ColumnSlice(t, 1));
                        mLastState.ColumnSlice(i, 1).SetValue(m_State.ColumnSlice(t, 1));
                        break;
                    }
                }
            }
        }

        virtual void EvaluateThisNode()
        {
            size_t nT = Inputs(0)->FunctionValues().GetNumCols();
            size_t outputDim = Inputs(1)->FunctionValues().GetNumRows();

            try{
                FunctionValues().Resize(outputDim, nT);
                FunctionValues().SetValue(NAN);  /// set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                m_State.Resize(outputDim, nT);
                m_State.SetValue(NAN);  /// set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                m_Gi.Resize(outputDim, nT);
                m_Gi.SetValue(NAN);  /// set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                m_Gf.Resize(outputDim, nT);
                m_Gf.SetValue(NAN);  /// set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                m_Go.Resize(outputDim, nT);
                m_Go.SetValue(NAN);  /// set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                tanhState.Resize(outputDim, nT);
                tanhState.SetValue(NAN);  /// set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                tanhObs.Resize(outputDim, nT);
                tanhObs.SetValue(NAN);  /// set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 

                if (m_PastState.IsEmpty() || m_PastState.GetNumCols() != m_samplesInRecurrentStep)
                {
                    m_PastState.Resize(outputDim, m_samplesInRecurrentStep);
                    m_PastState.SetValue(m_DefaultState);
                }
                if (m_PastOutput.IsEmpty() || m_PastOutput.GetNumCols() != m_samplesInRecurrentStep)
                {
                    m_PastOutput.Resize(outputDim, m_samplesInRecurrentStep);
                }

#ifdef DEBUG_DECODER
                if (m_PastOutput.IsEmpty() == false)
                    fprintf(stderr, "LSTM node %ls past output norm = %.8e\n", this->NodeName().c_str(), m_PastOutput.FrobeniusNorm());
                if (m_PastState.IsEmpty() == false)
                    fprintf(stderr, "LSTM node %ls past state norm = %.8e\n", this->NodeName().c_str(), m_PastState.FrobeniusNorm());
#endif

                for (size_t timeIdxInSeq = 0; timeIdxInSeq < nT; timeIdxInSeq += m_samplesInRecurrentStep)
                {

                    Matrix<ElemType> sliceObs = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceOutput = FunctionValues().ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceState = m_State.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);

                    Matrix<ElemType> sliceGi = m_Gi.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceGf = m_Gf.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceGo = m_Go.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);

                    Matrix<ElemType> sliceTanhState = tanhState.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceTanhInput =
                        tanhObs.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);

                    PrepareHistory(timeIdxInSeq, mSlicePrevOutput, mSlicePrevState, FunctionValues(), m_State, m_PastOutput, m_PastState, m_samplesInRecurrentStep, m_DefaultState, m_sentenceSeg);

                    try{
                        EvaluateThisNodeS(Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), Inputs(3)->FunctionValues(), Inputs(4)->FunctionValues(),
                            sliceObs, mSlicePrevOutput, mSlicePrevState, sliceOutput, sliceState, sliceGi, sliceGf, sliceGo, sliceTanhState, sliceTanhInput, m_tempMatrix);
                    }
                    catch (...)
                    {
                        fprintf(stderr, "Error in evaluating LSTMnode at position %ld out of %ld", timeIdxInSeq, nT);
                        throw;
                    }
                }

                /// save the hidden activities and output for the next minibatch
                SaveLastStateActity();

#ifdef DEBUG_DECODER
                if (mLastOutput.IsEmpty() == false)
                    fprintf(stderr, "LSTM node %ls last output norm = %.8e\n", this->NodeName().c_str(), mLastOutput.FrobeniusNorm());
                if (mLastState.IsEmpty() == false)
                    fprintf(stderr, "LSTM node %ls last state norm = %.8e\n", this->NodeName().c_str(), mLastState.FrobeniusNorm());
#endif

#ifdef DEBUG_DECODER
                ElemType tmpnorm = FunctionValues().FrobeniusNorm();
                if (ISCLOSE(tmpnorm, 0.834251, 0.002))
                    fprintf(stderr, "check!");
                fprintf(stderr, "LSTM function norm = %.8e\n", tmpnorm);
                for (size_t i = 0; i < 5; i++)
                    fprintf(stderr, "LSTM input[%d] norm = %.8e ", i, Inputs(i)->FunctionValues().FrobeniusNorm());
                fprintf(stderr, "\n");
#endif

                m_GradientComputed = false;
            }
            catch (...)
            {
                fprintf(stderr, "Error in evaluation of LSTMNode with %ld observations", nT);
                throw;
            }
        }

        /**
        Prepare history for LSTMnode

        This function returns state and output from the previous time instance. For recurrent network, the initial state needs to be set in the case of sentence begining, which is carried over from sentenceBegin. In case of sentence begining, the state activity is set to an initial value. The sentenceBegin has element of SENTENCE_BEGIN, SENTENCE_MIDDLE and NO_LABELS, which are 0, 1, and -1, respectively. 
        To compute the initial value, we use
        prevState = sentenceBegin * pastActivity + ~sentenceBegin * initialStateValue
        and ~sentenceBegin is computed as -1*(sentenceBegin - 1), assuming that sentenceBegin is either 0 or 1. For example, when sentenceBegin == 1, ~sentenceBegin == 0. 
        The previous-time output doesn't have initial value, so it is computed as 
        prevOutput = sentenceBegin * pastOutput

        */
        /// prepare prevstate and prevoutput
        static void WINAPI PrepareHistory(
            size_t timeIdxInSeq,
            Matrix<ElemType> & slicePrevOutput,
            Matrix<ElemType> & slicePrevState,
            const Matrix<ElemType> & output,
            const Matrix<ElemType> & state,
            const Matrix<ElemType> & pastOutput,
            const Matrix<ElemType> & pastState,
            size_t nsamples, const ElemType & initStateValue, Matrix<ElemType>* sentenceBegin)
        {
            size_t nRow = pastOutput.GetNumRows();
            size_t nStream = sentenceBegin->GetNumRows();

            assert(nStream == nsamples);

            int utt_t = (int)floor(timeIdxInSeq / nsamples);
            if (slicePrevOutput.IsEmpty() || slicePrevOutput.GetNumRows() != nRow || slicePrevOutput.GetNumCols() != nsamples)
                slicePrevOutput.Resize(nRow, nsamples);
            if (slicePrevState.IsEmpty() || slicePrevState.GetNumRows() != nRow || slicePrevState.GetNumCols() != nsamples)
                slicePrevState.Resize(nRow, nsamples);

            if (sentenceBegin->GetNumRows() != nsamples)
                LogicError("Number of rows should be the same as the number of data streams");

            Matrix<ElemType> colBegin(sentenceBegin->GetDeviceId());
            colBegin.SetValue(sentenceBegin->ColumnSlice(utt_t, 1));
            Matrix<ElemType> colSeg(colBegin.GetDeviceId()); 
            colSeg.Resize(nStream, nStream);
            /// will reset to 0 if sentence begining at a posiiton is 0
            /// will keep the output if it is not the sentence begining
            colBegin.InplaceTruncateBottom(SENTENCE_BEGIN);
            colBegin.InplaceTruncateTop(SENTENCE_MIDDLE);
            colSeg.SetDiagonalValue(colBegin);

            Matrix<ElemType> newPrevOutput(colBegin.GetDeviceId());
            Matrix<ElemType> newPrevState(colBegin.GetDeviceId());
            if (utt_t == 0)
            {
                /// this is the begining of this minibatch
                Matrix<ElemType>::Multiply(pastOutput.ColumnSlice(0, nsamples), false, colSeg, false, newPrevOutput);
                Matrix<ElemType>::Multiply(pastState.ColumnSlice(0, nsamples), false, colSeg, false, newPrevState);

            }
            else
            {
                /// this is in the minibatch
                Matrix<ElemType>::Multiply(output.ColumnSlice(timeIdxInSeq - nsamples, nsamples), false, colSeg, false, newPrevOutput);
                Matrix<ElemType>::Multiply(state.ColumnSlice(timeIdxInSeq - nsamples, nsamples), false, colSeg, false, newPrevState);
            }

            SetToInitStateValueForResetSeg(sentenceBegin->ColumnSlice(utt_t, 1), nStream, initStateValue, newPrevState);

            slicePrevOutput.ColumnSlice(0, nsamples).SetValue(newPrevOutput);
            slicePrevState.ColumnSlice(0, nsamples).SetValue(newPrevState);
        }

        /// prepare prevstate and prevoutput
        void PrepareThisErrorsBeforeBackProp(
            size_t timeIdxInSeq,
            size_t nT, /// number of columns
            Matrix<ElemType> & error,
            Matrix<ElemType> & stateError,
            const Matrix<ElemType>& grdToPrevOutput,
            const Matrix<ElemType>& grdToPrevState,
            const Matrix<ElemType>& obs_error_from_future_minibatch,
            const Matrix<ElemType>& state_error_from_future_minibatch,
            size_t nsamples, Matrix<ElemType>* sentenceBegin)
        {
            int utt_t = (int)floor(timeIdxInSeq / nsamples);
            int total_utt_t = (int)floor(nT / nsamples);

            error += grdToPrevOutput;
            stateError = grdToPrevState;

            if (m_use_errors_from_future_minibatch)
            {
                for (size_t utt_id = 0; utt_id < nsamples; utt_id++)
                {
                    /// if uses errors from future minibatch
                    if ((GetSegInfo(timeIdxInSeq, utt_id) == SENTENCE_MIDDLE && utt_t == total_utt_t - 1) /// last time 
                        || (utt_t < total_utt_t - 1 && GetSegInfo(timeIdxInSeq, utt_id) == SENTENCE_MIDDLE && GetSegInfo(timeIdxInSeq + nsamples, utt_id) == NO_LABELS) /// future observation is no observation
                        )
                    {
                        error.ColumnSlice(utt_id, 1) += obs_error_from_future_minibatch.ColumnSlice(utt_id, 1);
                        stateError.ColumnSlice(utt_id, 1) += state_error_from_future_minibatch.ColumnSlice(utt_id, 1);
                    }
                }
            }


            Matrix<ElemType> colBegin(sentenceBegin->GetDeviceId());
            colBegin.SetValue(sentenceBegin->ColumnSlice(utt_t, 1));
            colBegin.InplaceTruncateBottom(NO_LABELS);
            colBegin.InplaceTruncateTop(SENTENCE_BEGIN);
            colBegin += fabs((ElemType)NO_LABELS); /// raise this so that -1 -> 0 and therefore 
            Matrix<ElemType> colSeg(colBegin.GetDeviceId());
            colSeg.Resize(nsamples, nsamples);
            colSeg.SetDiagonalValue(colBegin);

            /// times the errors with the mask
            Matrix<ElemType> newOutputError(colBegin.GetDeviceId());
            Matrix<ElemType> newStateError(colBegin.GetDeviceId());

            Matrix<ElemType>::Multiply(error, false, colSeg, false, newOutputError);
            Matrix<ElemType>::Multiply(stateError, false, colSeg, false, newStateError);
            
            error.ColumnSlice(0, nsamples).SetValue(newOutputError);
            stateError.ColumnSlice(0, nsamples).SetValue(newStateError);
        }

        /// prepare prevstate and prevoutput
        static void WINAPI PrepareErrors(
            size_t timeIdxInSeq,
            Matrix<ElemType> & errors,
            Matrix<ElemType> & stateError,
            size_t nsamples, Matrix<ElemType>* sentenceBegin)
        {
            int utt_t = (int)floor(timeIdxInSeq / nsamples);
            Matrix<ElemType> colBegin(sentenceBegin->GetDeviceId());
            colBegin.SetValue(sentenceBegin->ColumnSlice(utt_t, 1));
            /// will reset to 0 if sentence begining at a posiiton is 0
            /// will keep the output if it is not the sentence begining
            colBegin.InplaceTruncateBottom(SENTENCE_BEGIN);
            colBegin.InplaceTruncateTop(SENTENCE_MIDDLE);

            Matrix<ElemType> colSeg(colBegin.GetDeviceId());
            colSeg.Resize(nsamples, nsamples);
            colSeg.SetDiagonalValue(colBegin);

            /// times the errors with the mask
            Matrix<ElemType> newOutputError(colBegin.GetDeviceId());
            Matrix<ElemType> newStateError(colBegin.GetDeviceId());

            Matrix<ElemType>::Multiply(errors, false, colSeg, false, newOutputError);
            Matrix<ElemType>::Multiply(stateError, false, colSeg, false, newStateError);

            errors.ColumnSlice(0, nsamples).SetValue(newOutputError);
            stateError.ColumnSlice(0, nsamples).SetValue(newStateError);
        }

        static void WINAPI EvaluateThisNodeS(
            const Matrix<ElemType>& mInputGate,
            const Matrix<ElemType> &mForgetGate, const Matrix<ElemType> &mOutputGate,
            const Matrix<ElemType> &mCellWgt,
            const Matrix<ElemType> &obs,
            const Matrix<ElemType>& prevOutput,
            const Matrix<ElemType>& prevState,
            Matrix<ElemType> &output,
            Matrix<ElemType> &state,
            Matrix<ElemType> &gi,
            Matrix<ElemType> &gf,
            Matrix<ElemType> &go,
            Matrix<ElemType> &tanhState,
            Matrix<ElemType> &tanhObs,
            Matrix<ElemType> &tmp)
        {
            int inputDim = obs.GetNumRows();
            int outputDim = mOutputGate.GetNumRows();

            /// for input gate
            Matrix<ElemType>::Multiply(mInputGate.ColumnSlice(1, inputDim), false, obs, false, gi);
            Matrix<ElemType>::MultiplyAndAdd(mInputGate.ColumnSlice(1 + inputDim, outputDim), false, prevOutput, false, gi);
            gi += mInputGate.ColumnSlice(0, 1);
            tmp = prevState;
            tmp.ColumnElementMultiplyWith(mInputGate.ColumnSlice(1 + inputDim + outputDim, 1));
            gi += tmp;
            gi.AssignSigmoidOf(gi);

            /// for forget gate
            Matrix<ElemType>::Multiply(mForgetGate.ColumnSlice(1, inputDim), false, obs, false, gf);
            Matrix<ElemType>::MultiplyAndAdd(mForgetGate.ColumnSlice(1 + inputDim, outputDim), false, prevOutput, false, gf);
            gf += mForgetGate.ColumnSlice(0, 1);
            tmp = prevState;
            tmp.ColumnElementMultiplyWith(mForgetGate.ColumnSlice(1 + inputDim + outputDim, 1));
            gf += tmp;
            gf.AssignSigmoidOf(gf);

            /// for cell state
            Matrix<ElemType>::Multiply(mCellWgt.ColumnSlice(1, inputDim), false, obs, false, state);
            Matrix<ElemType>::MultiplyAndAdd(mCellWgt.ColumnSlice(1 + inputDim, outputDim), false, prevOutput, false, state);
            state += mCellWgt.ColumnSlice(0, 1);
#ifdef DEBUG_DECODER
//            fprintf(stderr, "W_xc norm = %.8e\n", mCellWgt.ColumnSlice(1, inputDim).FrobeniusNorm());
//            fprintf(stderr, "W_hc norm = %.8e\n", mCellWgt.ColumnSlice(1 + inputDim, outputDim).FrobeniusNorm());
//            fprintf(stderr, "b_c norm = %.8e\n", mCellWgt.ColumnSlice(0, 1).FrobeniusNorm());
#endif
            tanhObs.AssignTanhOf(state);
            state.AssignElementProductOf(gi, tanhObs);
            state.AddElementProductOf(gf, prevState);

            /// for output gate
            Matrix<ElemType>::Multiply(mOutputGate.ColumnSlice(1, inputDim), false, obs, false, go);
            Matrix<ElemType>::MultiplyAndAdd(mOutputGate.ColumnSlice(1 + inputDim, outputDim), false, prevOutput, false, go);
            go += mOutputGate.ColumnSlice(0, 1);
            tmp = state;
            tmp.ColumnElementMultiplyWith(mOutputGate.ColumnSlice(1 + inputDim + outputDim, 1));
            go += tmp;
            go.AssignSigmoidOf(go);

            /// to return output
            tanhState.AssignTanhOf(state);
            output.AssignElementProductOf(go, tanhState);
        }


        /// input(0) : child with dimension [inputdim x T]
        /// input(1) : input gate [outputdim x [inputdim + outputdim + 2]] bi, Wxi, Whi, Wci
        /// input(2) : forget gate [outputdim x [inputdim + outputdim + 2]] for bf, Wxf, Whf, Wcf
        /// input(3) : output gate [outputdim x [inputdim + outputdim + 2]] for bo, Wxo, Who, and Wco
        /// input(4) : memory cell weight [outputdim x [inputdim + outputdim + 1]] for bc, Wxc, and Whc 
        /// output : dimension [outputdim x T]
        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 5)
                throw std::logic_error("LSTMNode requires four inputs.");

            CopyImageSizeFromInputs();

            if (Inputs(0)->FunctionValues().GetMatrixType() == SPARSE)
                LogicError("LSTMNode: input to LSTM has to be dense matrix. Consider adding a project layer using lookuptable before LSTM node. ");

            if (Inputs(1)->OperationName() != LearnableParameter<ElemType>::TypeName() ||
                Inputs(2)->OperationName() != LearnableParameter<ElemType>::TypeName() ||
                Inputs(3)->OperationName() != LearnableParameter<ElemType>::TypeName() ||
                Inputs(4)->OperationName() != LearnableParameter<ElemType>::TypeName())
                throw std::logic_error("LSTM validation: need to have learnable parameters ");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("LSTM validation: input size is zero!");

            if (Inputs(1)->FunctionValues().GetNumElements() == 0 ||
                Inputs(2)->FunctionValues().GetNumElements() == 0 ||
                Inputs(3)->FunctionValues().GetNumElements() == 0 ||
                Inputs(4)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("LSTM validation : parameter size is zero!");


            size_t nindim = Inputs(0)->FunctionValues().GetNumRows();
            size_t noutdim = Inputs(1)->FunctionValues().GetNumRows();
            size_t nT = Inputs(0)->FunctionValues().GetNumCols();
            size_t nCol = nindim + noutdim + 2;
            if (Inputs(1)->FunctionValues().GetNumCols() != nCol)
            {
                throw std::logic_error("LSTM validation : dimension mismatched between child and inputGate");
            }
            if (Inputs(2)->FunctionValues().GetNumCols() != nCol)
            {
                throw std::logic_error("LSTM validation : dimension mismatched between child and forgetGate");
            }
            if (Inputs(3)->FunctionValues().GetNumCols() != nCol)
            {
                throw std::logic_error("LSTM validation : dimension mismatched between child and outputGate");
            }

            if (noutdim != Inputs(2)->FunctionValues().GetNumRows() ||
                noutdim != Inputs(3)->FunctionValues().GetNumRows() ||
                noutdim != Inputs(4)->FunctionValues().GetNumRows())
            {
                throw std::logic_error("LSTM validation: output dimension mismatched!");
            }

            FunctionValues().Resize(noutdim, nT);
            FunctionValues().SetValue(NAN);  /// set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
        }

        bool UnitTest()
        {
            try{
                size_t nT = 3;
                size_t nInput = 2;
                size_t nHidden = 3;
                size_t nOutput = 3;

                /// backup 
                Matrix<ElemType> f0(m_deviceId), f1(m_deviceId), f2(m_deviceId), f3(m_deviceId), f4(m_deviceId), func(m_deviceId), f5(m_deviceId);
                Matrix<ElemType> target(m_deviceId);
                Matrix<ElemType> giWeight, ghWeight, goWeight;
                ElemType initStateValue = m_DefaultState;
                Matrix<ElemType> boundary(m_deviceId);
                boundary.Resize(1, nT);
                boundary.SetValue(SENTENCE_MIDDLE);
                boundary.ColumnSlice(0, 1).SetValue(SENTENCE_BEGIN);

                vector<MinibatchPackingFlag> minibatchPackingFlag;
                minibatchPackingFlag.resize(nT);
                std::fill(minibatchPackingFlag.begin(), minibatchPackingFlag.end(), MinibatchPackingFlag::None);
                minibatchPackingFlag[1] = MinibatchPackingFlag::UtteranceStart;
                ResetBound(&boundary, &minibatchPackingFlag);

                f0 = Inputs(0)->FunctionValues();
                f1 = Inputs(1)->FunctionValues();
                f2 = Inputs(2)->FunctionValues();
                f3 = Inputs(3)->FunctionValues();
                f4 = Inputs(4)->FunctionValues();
                func = FunctionValues();

                target.Resize(nOutput, nT);
                for (size_t i = 0; i < nT; i++)
                    target(0, i) = 1;

                Inputs(0)->FunctionValues().Resize(nInput, nT);
                Inputs(0)->FunctionValues().SetValue(ConstOnes(nInput, nT, m_deviceId));
                Inputs(0)->FunctionValues().SetValue((ElemType)0.1);
                Inputs(1)->FunctionValues().Resize(nHidden, nInput + nOutput + 2);
                Inputs(1)->FunctionValues().SetValue((ElemType)0.1);
                Inputs(2)->FunctionValues().Resize(nHidden, nInput + nHidden + 2);
                Inputs(2)->FunctionValues().SetValue((ElemType)0.1);
                Inputs(3)->FunctionValues().Resize(nOutput, nInput + nHidden + 2);
                Inputs(3)->FunctionValues().SetValue((ElemType)0.1);
                Inputs(4)->FunctionValues().Resize(nOutput, nHidden + nInput + 1);
                Inputs(4)->FunctionValues().SetValue((ElemType)0.1);
                FunctionValues().Resize(nOutput, nT);

                m_DefaultState = 0.0;
                EvaluateThisNode();

                /// check with expected values
                if (!ISCLOSE(FunctionValues()(0, 0), 0.0335975, EPSILON) ||
                    !ISCLOSE(FunctionValues()(0, 1), 0.05485132, EPSILON) ||
                    !ISCLOSE(FunctionValues()(0, 2), 0.06838435, EPSILON) ||
                    !(FunctionValues()(0, 0) == FunctionValues()(1, 0)))
                    throw("LSTMNode forward computation error");

                if (FunctionValues().GetDeviceId() != m_deviceId)
                    FunctionValues().TransferFromDeviceToDevice(FunctionValues().GetDeviceId(), m_deviceId, true);

                GradientValues().Resize(nOutput, nT);
                GradientValues().SetValue(1.0);
                for (size_t i = 0; i < 5; i++)
                {
                    Inputs(i)->GradientValues().Resize(Inputs(i)->FunctionValues().GetNumRows(), Inputs(i)->FunctionValues().GetNumCols());
                    Inputs(i)->GradientValues().SetValue(0);
                }
                for (size_t i = 0; i < 5; i++)
                    ComputeInputPartial(i);

                /// check with expected values
                if (!ISCLOSE(Inputs(1)->GradientValues()(0, 0), 0.07843818, EPSILON) /// bi
                    || !ISCLOSE(Inputs(1)->GradientValues()(0, 1), 0.00784382, EPSILON)  // Wxi
                    || !ISCLOSE(Inputs(1)->GradientValues()(0, 3), 0.00192997, EPSILON)  // Whi
                    || !ISCLOSE(Inputs(1)->GradientValues()(0, 6), 0.00362767, EPSILON)  // Wci
                    )
                    throw("LSTMNode gradient error on input gates");
                if (!ISCLOSE(Inputs(2)->GradientValues()(0, 0), 0.02738655, EPSILON)  // bf
                    || !ISCLOSE(Inputs(2)->GradientValues()(0, 1), 0.00273866, EPSILON)  // Wxf
                    || !ISCLOSE(Inputs(2)->GradientValues()(0, 3), 0.00120922, EPSILON)  // Whf
                    || !ISCLOSE(Inputs(2)->GradientValues()(0, 6), 0.00227184, EPSILON)  // Wcf
                    )
                    throw("LSTMNode gradient error on forget gates");
                if (!ISCLOSE(Inputs(3)->GradientValues()(0, 0), 0.07801557, EPSILON)  // bo
                    || !ISCLOSE(Inputs(3)->GradientValues()(0, 1), 0.00780156, EPSILON)  // Wxo
                    || !ISCLOSE(Inputs(3)->GradientValues()(0, 3), 0.00268089, EPSILON)  // Who
                    || !ISCLOSE(Inputs(3)->GradientValues()(0, 6), 0.00809852, EPSILON)  // Wco
                    )
                    throw("LSTMNode gradient error on output gates");
                if (!ISCLOSE(Inputs(4)->GradientValues()(0, 0), 1.3075038, EPSILON)  // bc
                    || !ISCLOSE(Inputs(4)->GradientValues()(0, 1), 0.13075038, EPSILON)  // Wxc
                    || !ISCLOSE(Inputs(4)->GradientValues()(0, 3), 0.03080355, EPSILON)  // Whc
                    )
                    throw("LSTMNode gradient error on memory cells");

                for (size_t i = 0; i < 5; i++)
                {
                    if (Inputs(i)->GradientValues().GetDeviceId() != m_deviceId)
                        Inputs(i)->GradientValues().TransferFromDeviceToDevice(Inputs(i)->GradientValues().GetDeviceId(), m_deviceId, true);
                }
                m_DefaultState = initStateValue;
            }
            catch (...)
            {
                fprintf(stderr, "LSTMNode unit test is not passed!");
                return false;
            }

            fprintf(stderr, "LSTMNode unit test passed!\n");
            return true;
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(1, false);
        }

        /// input(0) : child with dimension [inputdim x T]
        /// input(1) : input gate [outputdim x [inputdim + outputdim + 2]] bi, Wxi, Whi, Wci
        /// input(2) : forget gate [outputdim x [inputdim + outputdim + 2]] for bf, Wxf, Whf, Wcf
        /// input(3) : output gate [outputdim x [inputdim + outputdim + 2]] for bo, Wxo, Who, and Wco
        /// input(4) : memory cell weight [outputdim x [inputdim + outputdim + 1]] for bc, Wxc, and Whc 
        /// output : dimension [outputdim x T]
        virtual void AttachInputs(const ComputationNodePtr obs, const ComputationNodePtr inputGate, const ComputationNodePtr forgetGate, const ComputationNodePtr outputGate, const ComputationNodePtr memoryCellWgt)
        {
            m_children.resize(5);
            m_children[0] = obs;
            m_children[1] = inputGate;
            m_children[2] = forgetGate;
            m_children[3] = outputGate;
            m_children[4] = memoryCellWgt;
        }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_functionValues.GetDeviceId() != deviceId)
                {
                    bool fEmpty = m_functionValues.GetNumElements() == 0;
                    m_functionValues.TransferFromDeviceToDevice(m_functionValues.GetDeviceId(), deviceId, true, fEmpty);
                }

                if (m_gradientValues.GetDeviceId() != deviceId)
                {
                    bool fEmpty = m_gradientValues.GetNumElements() == 0;
                    m_gradientValues.TransferFromDeviceToDevice(m_gradientValues.GetDeviceId(), deviceId, true, fEmpty);
                }

                if (grdToObs.GetDeviceId() != deviceId)
                    grdToObs.TransferFromDeviceToDevice(grdToObs.GetDeviceId(), deviceId);
                if (grdToInputGate.GetDeviceId() != deviceId)
                    grdToInputGate.TransferFromDeviceToDevice(grdToInputGate.GetDeviceId(), deviceId);
                if (grdToForgetGate.GetDeviceId() != deviceId)
                    grdToForgetGate.TransferFromDeviceToDevice(grdToForgetGate.GetDeviceId(), deviceId);
                if (grdToOutputGate.GetDeviceId() != deviceId)
                    grdToOutputGate.TransferFromDeviceToDevice(grdToOutputGate.GetDeviceId(), deviceId);
                if (grdToCellWgt.GetDeviceId() != deviceId)
                    grdToCellWgt.TransferFromDeviceToDevice(grdToCellWgt.GetDeviceId(), deviceId);

                if (m_State.GetDeviceId() != deviceId)
                    m_State.TransferFromDeviceToDevice(m_State.GetDeviceId(), deviceId);
                if (m_PastState.GetDeviceId() != deviceId)
                    m_PastState.TransferFromDeviceToDevice(m_PastState.GetDeviceId(), deviceId);
                if (m_PastOutput.GetDeviceId() != deviceId)
                    m_PastOutput.TransferFromDeviceToDevice(m_PastOutput.GetDeviceId(), deviceId);
                if (m_Gi.GetDeviceId() != deviceId)
                    m_Gi.TransferFromDeviceToDevice(m_Gi.GetDeviceId(), deviceId);
                if (m_Gf.GetDeviceId() != deviceId)
                    m_Gf.TransferFromDeviceToDevice(m_Gf.GetDeviceId(), deviceId);
                if (m_Go.GetDeviceId() != deviceId)
                    m_Go.TransferFromDeviceToDevice(m_Go.GetDeviceId(), deviceId);

                if (tanhState.GetDeviceId() != deviceId)
                    tanhState.TransferFromDeviceToDevice(tanhState.GetDeviceId(), deviceId);
                if (tanhObs.GetDeviceId() != deviceId)
                    tanhObs.TransferFromDeviceToDevice(tanhObs.GetDeviceId(), deviceId);
                if (m_tempMatrix.GetDeviceId() != deviceId)
                    m_tempMatrix.TransferFromDeviceToDevice(m_tempMatrix.GetDeviceId(), deviceId);

                if (mSlicePrevState.GetDeviceId() != deviceId)
                    mSlicePrevState.TransferFromDeviceToDevice(mSlicePrevState.GetDeviceId(), deviceId);
                if (mSlicePrevOutput.GetDeviceId() != deviceId)
                    mSlicePrevOutput.TransferFromDeviceToDevice(mSlicePrevOutput.GetDeviceId(), deviceId);
                if (grdBeforeInputGate.GetDeviceId() != deviceId)
                    grdBeforeInputGate.TransferFromDeviceToDevice(grdBeforeInputGate.GetDeviceId(), deviceId);
                if (grdBeforeForget.GetDeviceId() != deviceId)
                    grdBeforeForget.TransferFromDeviceToDevice(grdBeforeForget.GetDeviceId(), deviceId);
                if (grdBeforeGo.GetDeviceId() != deviceId)
                    grdBeforeGo.TransferFromDeviceToDevice(grdBeforeGo.GetDeviceId(), deviceId);
                if (grdToCell.GetDeviceId() != deviceId)
                    grdToCell.TransferFromDeviceToDevice(grdToCell.GetDeviceId(), deviceId);
                if (grdBeforeTanhInputGate.GetDeviceId() != deviceId)
                    grdBeforeTanhInputGate.TransferFromDeviceToDevice(grdBeforeTanhInputGate.GetDeviceId(), deviceId);
            }
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            ComputationNode<ElemType>::DumpNodeInfo(printValues, fstream);

            fstream << L"Input[Width:" << m_inputDim << L"]  \n" ; 
            fstream << L"Hidden[Width:" << m_outputDim << L"]    Output[Width:" << m_outputDim << L"]  \n";
        }


    public:

        bool GetHistory(Matrix<ElemType>& hist, bool bLastTime)
        {
            size_t tRow = m_PastOutput.GetNumRows();
            size_t tCol = m_PastOutput.GetNumCols();
            size_t rCol = m_PastState.GetNumCols();

            DEVICEID_TYPE device = hist.GetDeviceId();
            hist.TransferFromDeviceToDevice(device, m_deviceId, true);
            hist.Resize(tRow, tCol + rCol);

            if (bLastTime)
            {
                hist.ColumnSlice(0, tCol).SetValue(mLastOutput);
                hist.ColumnSlice(tCol, rCol).SetValue(mLastState);
            }
            else{
                hist.ColumnSlice(0, tCol).SetValue(m_PastOutput);
                hist.ColumnSlice(tCol, rCol).SetValue(m_PastState);
            }

            hist.TransferFromDeviceToDevice(m_deviceId, device, true);
            return true;
        }

        void SetHistory(const Matrix<ElemType>& hist)
        {
            size_t tRow = hist.GetNumRows();
            size_t tCol = hist.GetNumCols();
            size_t eCols = tCol / 2;

            DEVICEID_TYPE device = hist.GetDeviceId();
            hist.TransferFromDeviceToDevice(device, m_deviceId, true);

            m_PastOutput.Resize(tRow, eCols);
            m_PastState.Resize(tRow, eCols);
            m_PastOutput.SetValue(hist.ColumnSlice(0, eCols));
            m_PastState.SetValue(hist.ColumnSlice(eCols, eCols));

            hist.TransferFromDeviceToDevice(m_deviceId, device, true);
        }

        virtual void GetErrorsToPreviousMinibatch(Matrix<ElemType>& hist)
        {
            size_t tRow = m_obs_error_from_future_minibatch.GetNumRows();
            size_t tCol = m_obs_error_from_future_minibatch.GetNumCols();
            size_t rCol = m_state_error_from_future_minibatch.GetNumCols();

            DEVICEID_TYPE device = hist.GetDeviceId();

            hist.TransferFromDeviceToDevice(device, m_deviceId, true);
            hist.Resize(tRow, tCol + rCol);

            hist.ColumnSlice(0, tCol).SetValue(m_obs_error_from_future_minibatch);
            hist.ColumnSlice(tCol, rCol).SetValue(m_state_error_from_future_minibatch);

            hist.TransferFromDeviceToDevice(m_deviceId, device, true);
        }

        virtual void SetErrorsFromFutureMinibatch(Matrix<ElemType>& hist)
        {
            size_t tCol = hist.GetNumCols();
            size_t rCol = tCol / 2;

            DEVICEID_TYPE device = hist.GetDeviceId();

            hist.TransferFromDeviceToDevice(device, m_deviceId, true);

            m_obs_error_from_future_minibatch.SetValue(hist.ColumnSlice(0, rCol));
            m_state_error_from_future_minibatch.SetValue(hist.ColumnSlice(rCol, rCol));

            m_use_errors_from_future_minibatch = true;

            hist.TransferFromDeviceToDevice(m_deviceId, device, true);
        }

    protected:
        virtual bool UseCustomizedMultiSeqHandling() { return true; }

    protected:
        size_t m_inputDim;
        size_t m_outputDim;

        Matrix<ElemType> m_State;  /// hidden state activity
        Matrix<ElemType> m_PastState; /// state activity in the previous minibatch
        Matrix<ElemType> m_PastOutput; /// output in the previou minibatch 

        Matrix<ElemType> mLastState; /// last state activity 
        Matrix<ElemType> mLastOutput; /// last output 

        Matrix<ElemType> m_Gi;     /// input gate activity
        Matrix<ElemType> m_Gf;     /// forget gate activity
        Matrix<ElemType> m_Go;     /// output gate activity

        Matrix<ElemType> grdToObs, grdToInputGate, grdToForgetGate, grdToOutputGate, grdToCellWgt;
        Matrix<ElemType> tanhState, tanhObs;

        Matrix<ElemType> m_tempMatrix; /// temp matrix for speed-up

        bool     m_GradientComputed; /// true if LSTM node has computed gradients, set to false if forward computation is just finished 

        Matrix<ElemType> mSlicePrevOutput, mSlicePrevState;

        Matrix<ElemType> grdBeforeInputGate, grdBeforeForget, grdBeforeGo, grdToCell, grdBeforeTanhInputGate;

    public:
        /// errors from future minibatch
        Matrix<ElemType> m_obs_error_from_future_minibatch;
        Matrix<ElemType> m_state_error_from_future_minibatch;
        bool m_use_errors_from_future_minibatch;

        ElemType m_DefaultState;

    };

    template class LSTMNode<float>;
    template class LSTMNode<double>;
}}}
