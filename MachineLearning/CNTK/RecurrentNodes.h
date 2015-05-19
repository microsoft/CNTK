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
            m_default_activity = (ElemType)DEFAULT_HIDDEN_ACTIVITY;
            m_delay = 1;
            m_functionValues.Resize(1,1);
            m_pastActivity.Resize(1,1);
            Reset();
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
            Reset();

            LoadFromFile(fstream, modelVersion, deviceId);
        }

        void SaveToFile(File& fstream) const
        {
            fstream << OperationName() << NodeName();
            fstream << m_delay; 
            fstream << FunctionValues().GetNumRows() << FunctionValues().GetNumCols(); 
        }

        void LoadFromFile(File& fstream, const size_t /*modelVersion*/, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
        {
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();

            fstream >> m_delay;

            size_t iRow, timeIdxInSeq;
            fstream >> iRow >> timeIdxInSeq;
            FunctionValues().Resize(iRow,timeIdxInSeq);
            m_pastActivity.Resize(iRow, timeIdxInSeq);
        }

        DelayNode(const DEVICEID_TYPE deviceId, ElemType initHiddenActivity, size_t row_size, size_t col_size, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_default_activity = initHiddenActivity;
            m_delay = 1;

            m_functionValues.Resize(row_size, col_size);
            m_functionValues.SetValue(m_default_activity);

            m_pastActivity.Resize(row_size, col_size);
            m_pastActivity.SetValue(m_default_activity);

            m_gradientValues.Resize(row_size, col_size);
            m_gradientValues.SetValue(0.0f);

            Reset();
            InitRecurrentNode();
        }

        virtual const std::wstring OperationName() const {return TypeName();}

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
            if (m_samplesInRecurrentStep == 1)
            {
                ComputeInputPartialSR(timeIdxInSeq, m_delay, Inputs(0)->GradientValues(), GradientValues(), m_samplesInRecurrentStep);
            }else
            {
                for (size_t i = 0 ; i < m_samplesInRecurrentStep; i++)
                {
                    bool reset = false;

                    if ((((int)m_sentenceEnd[i] +(int)m_delay - 1) >= (int)timeIdxInSeq) && (m_sentenceEnd[i] <= timeIdxInSeq))
                    {
                        reset = true;
                    }

                    ComputeInputPartialSRP(timeIdxInSeq, m_delay, reset, Inputs(0)->GradientValues(), GradientValues(), i, m_samplesInRecurrentStep);
                }
            }
        }

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

        static void WINAPI ComputeInputPartialSRP(int timeIdxInSeq, int delay,  bool reset,
            Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const size_t indexInBatch, const size_t mNbr)
        {
            assert(timeIdxInSeq >= 0);
            if ((timeIdxInSeq - delay) >= 0 && (timeIdxInSeq - delay) * mNbr <= inputGradientValues.GetNumCols() && !reset)
            {
                Matrix<ElemType> to = inputGradientValues.ColumnSlice((timeIdxInSeq - delay)*mNbr + indexInBatch, 1);
                Matrix<ElemType> frm= gradientValues.ColumnSlice(timeIdxInSeq * mNbr + indexInBatch, 1);
                to += frm; 
            }
        }


        virtual void EvaluateThisNode()  
        {
            ASSERT(m_delay > 0);
            size_t blogSize = Inputs(0)->FunctionValues().GetNumCols();

            for (size_t i = 0; i < blogSize / m_samplesInRecurrentStep; i++)
                EvaluateThisNodeSR(i, m_delay, m_Reset, m_default_activity, m_functionValues, m_pastActivity, Inputs(0)->FunctionValues(), m_samplesInRecurrentStep);
            /// reset past activity
            m_pastActivity = Inputs(0)->FunctionValues();
        }

        void Reset()
        {
            m_Reset = true;
        }


        void NotReset()
        {
            m_Reset = false;
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)  
        {
            /// reset past activity as it reached to the begining of a minibatch
            /// the node pointed hasn't yet updated, so it is the past activity 
            if (timeIdxInSeq == 0)
                m_pastActivity = Inputs(0)->FunctionValues();
            
            if (m_samplesInRecurrentStep == 1)
            {
                //bool reset = (m_sentenceEnd[0] == timeIdxInSeq);
                bool reset = false;

                if ((((int)m_sentenceEnd[0] +(int)m_delay - 1) >= (int)timeIdxInSeq) && (m_sentenceEnd[0] <= timeIdxInSeq))
                {
                    reset = true;
                }
                EvaluateThisNodeSR(timeIdxInSeq, m_delay, reset, m_default_activity, m_functionValues, m_pastActivity, Inputs(0)->FunctionValues(), m_samplesInRecurrentStep);
            } else
            {
                for (size_t i = 0 ; i < m_samplesInRecurrentStep; i++)
                {
                    // bool reset = (m_sentenceEnd[i] == timeIdxInSeq);
                    bool reset = false;

                    if ((((int)m_sentenceEnd[i] +(int)m_delay - 1) >= (int)timeIdxInSeq) && (m_sentenceEnd[i] <= timeIdxInSeq))
                    {
                        reset = true;
                    }
                    EvaluateThisNodeSRP(timeIdxInSeq, m_delay, reset, m_default_activity, m_functionValues, m_pastActivity, Inputs(0)->FunctionValues(), i, m_samplesInRecurrentStep);
                }
            }

        }

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

        static void WINAPI EvaluateThisNodeSRP(const size_t timeIdxInSeq, const int delay, const bool reset, const ElemType default_activity, Matrix<ElemType>& functionValues, const Matrix<ElemType>& pastActivity, const Matrix<ElemType>& inputFunctionValues, const size_t indexInBatch, const size_t mNbr)
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

            Matrix<ElemType> out = functionValues.ColumnSlice(timeIdxInSeq * mNbr+indexInBatch, 1);
            Matrix<ElemType> inp((DEVICEID_TYPE)functionValues.GetDeviceId()) ;

            if (reset)
                out.SetValue(default_activity);
            else
            {
                if (iPastIndex < 0)
                    inp = pastActivity.ColumnSlice(d+indexInBatch, 1);
                else
                    inp = inputFunctionValues.ColumnSlice(d+indexInBatch, 1);
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
                if (m_functionValues.GetDeviceId() != deviceId)
                    m_functionValues.TransferFromDeviceToDevice(m_functionValues.GetDeviceId(), deviceId);
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
                node->m_Reset = m_Reset;
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

    private:
        Matrix<ElemType> m_pastActivity;  /// saves the past activity this delay node points to
        int      m_delay;    /// steps for delay 
        bool     m_Reset; 

    };

    template class DelayNode<float>; 
    template class DelayNode<double>;

}}}
