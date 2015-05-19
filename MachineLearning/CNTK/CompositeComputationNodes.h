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

    template<class ElemType>
    class PreComputedNode : public ComputationNode<ElemType>  //this is a noninstantiable virtual class, all nodes require precomputation should derive from it
    {
        UsingComputationNodeMembers;
    public:
        PreComputedNode<ElemType>(DEVICEID_TYPE deviceId) : ComputationNode<ElemType>(deviceId) {}
        virtual bool HasComputed() const = 0;
        virtual void MarkComputed(const bool hasComputed) = 0;

        virtual bool RequirePreCompute() const { return true;}
                
        virtual void SaveToFile(File& fstream)  const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << m_hasComputed;
            fstream << m_functionValues;
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            fstream >> m_hasComputed;
            fstream >> m_functionValues;
        }


        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            ComputationNode<ElemType>::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "[%lu,%lu]  ", FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            fstream << string(str);
            sprintf(str, "HasComputed=%ls", HasComputed()? L"true" : L"false");
            fstream << string(str);

            PrintNodeValuesToFile(printValues, fstream);
        }

    public:
        bool m_hasComputed;
    };
#define UsingPreComputedNodeMembers UsingComputationNodeMembers; using PreComputedNode<ElemType>::m_hasComputed

    template class PreComputedNode<float>; 
    template class PreComputedNode<double>;

    template<class ElemType>
    class MeanNode : public PreComputedNode<ElemType>
    {
        UsingPreComputedNodeMembers;
    public:
        MeanNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : PreComputedNode<ElemType>(deviceId)  
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_hasComputed = false;
            m_numSamples = 0;
            InitRecurrentNode();
        }

        MeanNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : PreComputedNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX)
        {
            PreComputedNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            m_numSamples = 0;
        }

        virtual bool HasComputed() const {return m_hasComputed;}
        virtual void MarkComputed(const bool hasComputed)
        {
            m_hasComputed = hasComputed;

            if (m_hasComputed) 
            {
                m_numSamples = 0;
            }
        }

        virtual bool RequirePreCompute() const { return true;}

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Mean";} 
        virtual void ComputeInputPartial(const size_t /*inputIndex*/)
        {
            throw std::logic_error("Mean operation should not be involved in the gradient calculation.");
        }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/) 
        {
            throw std::logic_error("Mean operation should not be involved in the gradient calculation.");
        }

        virtual void EvaluateThisNode()  
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
                    ConstOnes(numNewSamples, 1, samples.GetDeviceId()), false, (ElemType)m_numSamples / (m_numSamples + numNewSamples), avg);

#if NANCHECK
                avg.HasNan("Mean-avg");
                ones.HasNan("Mean-ones");
#endif
                
                m_numSamples += numNewSamples;
            }
        }
        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/)
        {
            throw std::logic_error("Mean operation should not be involved in a recurrent loop.");
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("Mean operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("Mean operation: the input node has 0 element.");

            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), 1);
            CopyImageSizeFromInputs(); 
        }

        virtual void AttachInputs(const ComputationNodePtr singleInput) 
        {
            m_children.resize(1);
            m_children[0] = singleInput;
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            MeanNode<ElemType>* node = (MeanNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_hasComputed = m_hasComputed;
                node->m_numSamples = m_numSamples;
            }
        }

        // copy constructor
        MeanNode(const MeanNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) 
            : PreComputedNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new MeanNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        size_t m_numSamples;
    };

    template class MeanNode<float>; 
    template class MeanNode<double>;

    template<class ElemType>
    class InvStdDevNode : public PreComputedNode<ElemType>
    {
        UsingPreComputedNodeMembers;
    public:
        InvStdDevNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : PreComputedNode<ElemType>(deviceId), m_mean(deviceId), m_var(deviceId),  m_temp(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            m_hasComputed = false;
            m_numSamples = 0;
            InitRecurrentNode();
        }

        InvStdDevNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"")
            : PreComputedNode<ElemType>(deviceId), m_mean(deviceId), m_var(deviceId),  m_temp(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX)
        {
            PreComputedNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            m_numSamples = 0;
        }

        virtual bool HasComputed() const {return m_hasComputed;}

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
        virtual bool RequirePreCompute() const { return true;}

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"InvStdDev";} 
        virtual void ComputeInputPartial(const size_t /*inputIndex*/)
        {
            throw std::logic_error("InvStdDev operation should not be involved in the gradient calculation.");
        }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/)
        {
            throw std::logic_error("InvStdDev operation should not be involved in the gradient calculation.");
        }

        virtual void EvaluateThisNode()  
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
                    ConstOnes(numNewSample, 1, samples.GetDeviceId()), false, (ElemType)m_numSamples / (m_numSamples + numNewSample), m_mean);

                m_temp -= m_mean;
                m_temp.AssignElementPowerOf(m_temp, 2);
                m_var += m_temp;

                m_temp.AssignDifferenceOf(samples, m_mean);
                m_temp.AssignElementPowerOf(m_temp, 2);

                Matrix<ElemType>::MultiplyAndWeightedAdd(1.0f / (m_numSamples + numNewSample), m_temp, false,
                    ConstOnes(numNewSample, 1, samples.GetDeviceId()), false, (ElemType)m_numSamples / (m_numSamples + numNewSample), m_var);

#if NANCHECK
                m_var.HasNan("InvStdDev-m_var");
#endif

                m_numSamples += samples.GetNumCols();
            }
        }

        virtual void EvaluateThisNode(const size_t /*timeIdxInSeq*/)
        {
            throw std::logic_error("InvStdDev operation should not be involved in a recurrent loop.");
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("InvStdDev operation should have one input.");

            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("InvStdDev operation: the input node has 0 element.");

            size_t inputDim = Inputs(0)->FunctionValues().GetNumRows();
            m_mean.Resize(inputDim, 1);        
            m_var.Resize(inputDim, 1); 

            FunctionValues().Resize(inputDim, 1);
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
                if (m_mean.GetDeviceId() != deviceId)
                    m_mean.TransferFromDeviceToDevice(m_mean.GetDeviceId(), deviceId);
                if (m_var.GetDeviceId() != deviceId)
                    m_var.TransferFromDeviceToDevice(m_var.GetDeviceId(), deviceId);
                if (m_temp.GetDeviceId() != deviceId)
                    m_temp.TransferFromDeviceToDevice( m_temp.GetDeviceId(), deviceId);
            }
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            InvStdDevNode<ElemType>* node = (InvStdDevNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_hasComputed = m_hasComputed;
                node->m_numSamples = m_numSamples;

                node->m_mean = m_mean;
                node->m_var = m_var;
                node-> m_temp =  m_temp;
            }
        }

        // copy constructor
        InvStdDevNode(const InvStdDevNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) 
            : PreComputedNode<ElemType>(node->m_deviceId), m_mean(node->m_deviceId), m_var(node->m_deviceId),  m_temp(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }                                      

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new InvStdDevNode<ElemType>(this, name, flags);
            return node;
        }

    private:
        size_t m_numSamples;
        Matrix<ElemType> m_mean;
        Matrix<ElemType> m_var;
        Matrix<ElemType>  m_temp;
    };

    template class InvStdDevNode<float>;
    template class InvStdDevNode<double>;

    template<class ElemType>
    class PerDimMeanVarNormalizationNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        PerDimMeanVarNormalizationNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId) 
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        PerDimMeanVarNormalizationNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        PerDimMeanVarNormalizationNode(const PerDimMeanVarNormalizationNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new PerDimMeanVarNormalizationNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"PerDimMeanVarNormalization";} 
        virtual void ComputeInputPartial(const size_t /*inputIndex*/)  //scaled by 2*number of colmns (samples) in the Matrix<ElemType>
        {
            throw std::invalid_argument("PerDimMeanVarNormalizationNode should only be called in the evaluation stage.");
        }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/)
        {
            throw std::invalid_argument("PerDimMeanVarNormalizationNode should only be called in the evaluation stage.");
        }

        virtual void EvaluateThisNode()   //(feature-mean).*InvStdDev
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            //only feature (input0) and output needs to be sliced
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues());
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0, const Matrix<ElemType> &input1, const Matrix<ElemType> &input2)
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

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 3) 
                throw std::logic_error("PerDimMeanVarNormalizationNode criterion requires three inputs.");

            if (Inputs(0)->RequirePreCompute())
                throw std::logic_error("PerDimMeanVarNormalizationNode criterion forbids first input from being a pre-compute node. "
                                       "The first input should be the node whose output should be normalized, and the second and third inputs "
                                       "should be LearnableParameter type or (Mean, InvStdDev) so that the values will be saved.");

            if (!(Inputs(1)->OperationName() == LearnableParameter<ElemType>::TypeName() && Inputs(2)->OperationName() == LearnableParameter<ElemType>::TypeName()) &&
                !(Inputs(1)->OperationName() == MeanNode<ElemType>::TypeName() && Inputs(2)->OperationName() == InvStdDevNode<ElemType>::TypeName()))
                throw std::logic_error("PerDimMeanVarNormalizationNode criterion requires the last two inputs to be LearnableParameter type or (Mean, InvStdDev) so that the values will be saved.");

            if (Inputs(1)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(1)->FunctionValues().GetNumRows() == 0? Inputs(0)->FunctionValues().GetNumRows() : Inputs(1)->FunctionValues().GetNumRows();
                Inputs(1)->FunctionValues().Resize(rows, 1);
            }

            if (Inputs(2)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(2)->FunctionValues().GetNumRows() == 0? Inputs(0)->FunctionValues().GetNumRows() : Inputs(2)->FunctionValues().GetNumRows();
                Inputs(2)->FunctionValues().Resize(rows, 1);
            }

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0 || Inputs(2)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("PerDimMeanVarNormalizationNode operation: one of the operants has 0 element.");

            if (!(Inputs(0)->FunctionValues().GetNumRows() == Inputs(1)->FunctionValues().GetNumRows()  &&  //match rows
                Inputs(2)->FunctionValues().GetNumRows() == Inputs(1)->FunctionValues().GetNumRows()) )
            {
                //Inputs(1)->FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), 1);
                //Inputs(2)->FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), 1);
                throw std::logic_error("PerDimMeanVarNormalizationNode: All inputs should have same number of rows.");
            }       

            if (!(Inputs(1)->FunctionValues().GetNumCols() == 1 && Inputs(2)->FunctionValues().GetNumCols() == 1))
            {
                throw std::logic_error("PerDimMeanVarNormalizationNode: Mean and InvStdDev should be a colum  vector.");
            }       

            Inputs(1)->NeedGradient() = false;
            Inputs(2)->NeedGradient() = false;  //prevent learning 
            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        //leftNode should be the empirical
        virtual void AttachInputs(const ComputationNodePtr feature, const ComputationNodePtr mean, const ComputationNodePtr InvStdDev) 
        {
            m_children.resize(3);
            m_children[0] = feature;
            m_children[1] = mean;
            m_children[2] = InvStdDev;
        }
    };

    template class PerDimMeanVarNormalizationNode<float>; 
    template class PerDimMeanVarNormalizationNode<double>;

    template<class ElemType>
    class PerDimMeanVarDeNormalizationNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        PerDimMeanVarDeNormalizationNode(const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId) 
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        PerDimMeanVarDeNormalizationNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        PerDimMeanVarDeNormalizationNode(const PerDimMeanVarDeNormalizationNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new PerDimMeanVarDeNormalizationNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"PerDimMeanVarDeNormalization";} 
        virtual void ComputeInputPartial(const size_t /*inputIndex*/)  //scaled by 2*number of colmns (samples) in the Matrix<ElemType>
        {
            throw std::invalid_argument("PerDimMeanVarDeNormalizationNode should only be called in the evaluation stage.");
        }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const size_t /*timeIdxInSeq*/)
        {
            throw std::invalid_argument("PerDimMeanVarDeNormalizationNode should only be called in the evaluation stage.");
        }

        virtual void EvaluateThisNode()   //(feature-mean).*InvStdDev
        {
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues());
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            //only feature (input0) and output needs to be sliced
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(sliceOutputValue, sliceInput0Value, Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues());
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0, const Matrix<ElemType> &input1, const Matrix<ElemType> &input2)
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

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 3) 
                throw std::logic_error("PerDimMeanVarDeNormalizationNode criterion requires three inputs.");

            if (Inputs(0)->RequirePreCompute())
                throw std::logic_error("PerDimMeanVarDeNormalizationNode criterion forbids first input from being a pre-compute node. "
                                       "The first input should be the node whose output should be de-normalized, and the second and third inputs "
                                       "should be LearnableParameter type or (Mean, InvStdDev) so that the values will be saved.");

            if (!(Inputs(1)->OperationName() == LearnableParameter<ElemType>::TypeName() && Inputs(2)->OperationName() == LearnableParameter<ElemType>::TypeName()) &&
                !(Inputs(1)->OperationName() == MeanNode<ElemType>::TypeName() && Inputs(2)->OperationName() == InvStdDevNode<ElemType>::TypeName()))
                throw std::logic_error("PerDimMeanVarDeNormalizationNode criterion requires the last two inputs to be LearnableParameter type or (Mean, InvStdDev) so that the values will be saved.");

            if (Inputs(1)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(1)->FunctionValues().GetNumRows() == 0? Inputs(0)->FunctionValues().GetNumRows() : Inputs(1)->FunctionValues().GetNumRows();
                Inputs(1)->FunctionValues().Resize(rows, 1);
            }

            if (Inputs(2)->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                size_t rows = Inputs(2)->FunctionValues().GetNumRows() == 0? Inputs(0)->FunctionValues().GetNumRows() : Inputs(2)->FunctionValues().GetNumRows();
                Inputs(2)->FunctionValues().Resize(rows, 1);
            }

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0 || Inputs(2)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("PerDimMeanVarDeNormalizationNode operation: one of the operants has 0 element.");

            if (!(Inputs(0)->FunctionValues().GetNumRows() == Inputs(1)->FunctionValues().GetNumRows()  &&  //match rows
                Inputs(2)->FunctionValues().GetNumRows() == Inputs(1)->FunctionValues().GetNumRows()) )
            {
                //Inputs(1)->FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), 1);
                //Inputs(2)->FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), 1);
                throw std::logic_error("PerDimMeanVarDeNormalizationNode: All inputs should have same number of rows.");
            }       

            if (!(Inputs(1)->FunctionValues().GetNumCols() == 1 && Inputs(2)->FunctionValues().GetNumCols() == 1))
            {
                throw std::logic_error("PerDimMeanVarDeNormalizationNode: Mean and InvStdDev should be a colum  vector.");
            }       

            Inputs(1)->NeedGradient() = false;
            Inputs(2)->NeedGradient() = false;  //prevent learning 
            FunctionValues().Resize(Inputs(0)->FunctionValues().GetNumRows(), Inputs(0)->FunctionValues().GetNumCols());
            CopyImageSizeFromInputs(); 
        }

        //leftNode should be the empirical
        virtual void AttachInputs(const ComputationNodePtr feature, const ComputationNodePtr mean, const ComputationNodePtr InvStdDev) 
        {
            m_children.resize(3);
            m_children[0] = feature;
            m_children[1] = mean;
            m_children[2] = InvStdDev;
        }
    };

    template class PerDimMeanVarDeNormalizationNode<float>; 
    template class PerDimMeanVarDeNormalizationNode<double>;


    /**
    BatchModeNode is a derivative of ComputationNode.
    It additionally check if needs to process data in batch before processing its parent
    This is used in case of beam search decoding. Batchmode node must be processed before other nodes.
    It differs from PreComputeNode in that precompute done is done before the entire corpus.
    This is done before forward computation of all nodes. 
    This node is similar to the PreComputeNode, but is an abstract of it.
    */
    template<class ElemType>
    class BatchModeNode : public ComputationNode<ElemType>
        // all nodes require precomputation should derive from it
    {
        UsingComputationNodeMembers;

    protected:
        /// the memory of input or output
        Matrix<ElemType> mMemory;

    public:
        BatchModeNode(DEVICEID_TYPE deviceId) : ComputationNode<ElemType>(deviceId), mMemory(deviceId) {}
        virtual bool HasComputed() const = 0;
        virtual void MarkComputed(const bool hasComputed) = 0;

        virtual bool RequireBatchMode() const { return true; }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq)
        {
            assert(mMemory.GetNumCols() > 0);

            FunctionValues().Resize(mMemory.GetNumRows(), m_samplesInRecurrentStep);
            if (timeIdxInSeq == 0)
            {
                assert(FunctionValues().ColumnSlice(0, m_samplesInRecurrentStep).FrobeniusNorm() == mMemory.ColumnSlice(0, m_samplesInRecurrentStep).FrobeniusNorm());
            }
            FunctionValues().SetValue(mMemory.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep));
            assert(FunctionValues().GetNumCols() == m_samplesInRecurrentStep);
        }

        virtual void SaveToFile(File& fstream)  const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << m_hasComputed;
            fstream << m_functionValues;
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const short deviceId = AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            fstream >> m_hasComputed;
            fstream >> m_functionValues;
        }


        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            ComputationNode<ElemType>::DumpNodeInfo(printValues, fstream);

            const size_t BUFLEN = 4096;
            WCHAR str[BUFLEN];
            swprintf(str, BUFLEN, L"[%lu,%lu]  ", FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            fstream << wstring(str);
            swprintf(str, BUFLEN, L"HasComputed=%ls", HasComputed() ? L"true" : L"false");
            fstream << wstring(str);

            PrintNodeValuesToFile(printValues, fstream);
        }

    protected:
        bool m_hasComputed;
    };

    // add this at the start of each derived class, to get access to the members of ComputationNode
    // TODO: comment here why this is needed and how to maintain it
#define UsingBatchModeNodeMembers \
    UsingComputationNodeMembers; \
    typedef BatchModeNode<ElemType> C; \
protected:  \
    typedef BatchModeNode<ElemType>* BatchModeNodePtr;  \
public: \
    using C::HasComputed; using C::MarkComputed; \
    using C::RequireBatchMode; using C::EvaluateThisNode; using C::SaveToFile; \
    using C::LoadFromFile; using C::DumpNodeInfo; \
protected:  \
    using C::mMemory; using C::m_hasComputed; \

    template class BatchModeNode<float>;
    template class BatchModeNode<double>;

    /**
    Developed by Kaisheng Yao. 
    This node is used in the following work
    K. Yao and G. Zweig, "Sequence-to-Sequence Neural Net Models for Grapheme-to-Phoneme Conversion", submitted to INTERSPEECH 2015
    */
    template<class ElemType>
    class TimeReverseNode : public BatchModeNode<ElemType>
    {
        UsingBatchModeNodeMembers;

    public:
        TimeReverseNode(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"") : BatchModeNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        TimeReverseNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE  deviceId = AUTOPLACEMATRIX, const std::wstring name = L"") : BatchModeNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        TimeReverseNode(const TimeReverseNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) 
            : BatchModeNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            TimeReverseNode<ElemType>* node = (TimeReverseNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->mMemory = mMemory;
            }
        }

        virtual void SaveToFile(File& fstream)  const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const short deviceId = AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);
        }

        virtual bool HasComputed() const {
            return m_hasComputed;
        }

        virtual void MarkComputed(const bool hasComputed)
        {
            m_hasComputed = hasComputed;
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;

            ComputationNodePtr node = new TimeReverseNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const { return TypeName(); }
        static const std::wstring TypeName() { return L"TimeReverse"; }

        virtual void MoveMatricesToDevice(const short deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (mMemory.GetDeviceId() != deviceId)
            {
                bool fEmpty = mMemory.GetNumElements() == 0;
                mMemory.TransferFromDeviceToDevice(mMemory.GetDeviceId(), deviceId, true, fEmpty);
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("TimeReverse operation only takes one input.");
            ComputationNodePtr child = Inputs(inputIndex);
            ComputeInputPartialS(GradientValues(), child->GradientValues(), m_samplesInRecurrentStep);
        }

        static void WINAPI ComputeInputPartialS(Matrix<ElemType>& gradientValues, Matrix<ElemType>& inputGradientValues, int nSamples)
        {
#if DUMPOUTPUT

            functionValues.Print("TimeReverseNode");
#endif
            size_t nc = inputGradientValues.GetNumCols();
            size_t nr = inputGradientValues.GetNumRows();
            if (nc != gradientValues.GetNumCols() ||
                nr != gradientValues.GetNumRows())
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

        virtual void EvaluateThisNode()
        {
            if (m_hasComputed == false)
            {
                EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), m_samplesInRecurrentStep);
                mMemory.SetValue(FunctionValues());
            }
        }

        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, Matrix<ElemType>& inputFunctionValues, int nSamples)
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
            m_functionValues.HasNan("TimeReverse");
#endif
#if DUMPOUTPUT
            functionValues.Print("TimeReverseNode");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1)
                throw std::logic_error("TimeReverse operation requires one input.");

            size_t rows, cols;
            rows = Inputs(0)->FunctionValues().GetNumRows();
            cols = Inputs(0)->FunctionValues().GetNumCols();

            FunctionValues().Resize(rows, cols);
            CopyImageSizeFromInput(0);
        }

        virtual void AttachInputs(const ComputationNodePtr cNode)
        {
            m_children.resize(1);
            m_children[0] = cNode;
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

            Inputs(0)->FunctionValues().Resize(nInput, nT);
            Inputs(0)->FunctionValues().SetValue(0);
            Inputs(0)->FunctionValues()(0, 0) = 1;
            Inputs(0)->FunctionValues()(0, 1) = 2;
            Inputs(0)->FunctionValues()(0, 2) = 3;
            FunctionValues().Resize(nOutput, nT);
            if (Inputs(0)->FunctionValues().GetDeviceId() != m_deviceId)
                Inputs(0)->FunctionValues().TransferFromDeviceToDevice(Inputs(0)->FunctionValues().GetDeviceId(), m_deviceId, true);

            EvaluateThisNode();

            /// check with expected values
            if (!ISCLOSE(FunctionValues()(0, 0), 3, EPSILON) ||
                !ISCLOSE(FunctionValues()(0, 1), 2, EPSILON) ||
                !ISCLOSE(FunctionValues()(0, 2), 1, EPSILON))
                return false;
            if (FunctionValues().GetDeviceId() != m_deviceId)
                FunctionValues().TransferFromDeviceToDevice(FunctionValues().GetDeviceId(), m_deviceId, true);

            Inputs(0)->GradientValues().Resize(nOutput, nT);
            Inputs(0)->GradientValues().SetValue(1.0);
            GradientValues().Resize(nOutput, nT);
            GradientValues().SetValue(0);
            GradientValues()(0, 0) = 1;
            GradientValues()(0, 1) = 2;
            GradientValues()(0, 2) = 3;
            if (GradientValues().GetDeviceId() != m_deviceId)
                GradientValues().TransferFromDeviceToDevice(GradientValues().GetDeviceId(), m_deviceId, true);

            ComputeInputPartial(0);

            /// check with expected values
            if (!ISCLOSE(Inputs(0)->GradientValues()(0, 0), 4, EPSILON)
                || !ISCLOSE(Inputs(0)->GradientValues()(0, 1), 3, EPSILON)
                || !ISCLOSE(Inputs(0)->GradientValues()(0, 2), 2, EPSILON))
                return false;

            if (Inputs(0)->GradientValues().GetDeviceId() != m_deviceId)
                Inputs(0)->GradientValues().TransferFromDeviceToDevice(Inputs(0)->GradientValues().GetDeviceId(), m_deviceId, true);

            if (GradientValues().GetDeviceId() != m_deviceId)
                GradientValues().TransferFromDeviceToDevice(GradientValues().GetDeviceId(), m_deviceId, true);

            return true;
        }
    };

    template class TimeReverseNode<float>;
    template class TimeReverseNode<double>;


}}}
