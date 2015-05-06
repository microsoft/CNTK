//
// <copyright file="CompositeComputationNode.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

//The basic idea of this implementation is learned from Brian Guenter <bguenter@microsoft.com>

#include <map>
#include <string>
#include <stdexcept>
#include <list>
#include "ComputationNode.h"
#include "TrainingCriterionNode.h"
#include <iostream> 

#define TWO_PI 6.283185307f


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


    // convolution parameters structure, to make it easier to pass these around all these parameters
    struct ConvolutionParams
    {
        size_t inputWidth, inputHeight, inputChannels;
        size_t kernelWidth, kernelHeight;
        size_t horizontalSubsample, verticalSubsample;
        size_t outputWidth, outputHeight, outputChannels;
        size_t maxTempMemSizeInSamples;
        bool zeroPadding;
    };

    //convolutional network 
    //follow "high performance convolutional neural networks for document processing" by Kumar chellapilla, Sidde Puri, and Patrice Simard
    //assume each column is an input sample. Each sample is stored in [channel, row, col]  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
    template<class ElemType>
    class ConvolutionNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        ConvolutionNode(const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels, 
                        const size_t horizontalSubsample, const size_t verticalSubsample, 
                        const bool zeroPadding = false, 
                        const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"",
                        const size_t maxTempMemSizeInSamples = 0)
                        : ComputationNode<ElemType>(deviceId), m_tempMatrix(deviceId),
                          m_kernelWidth(kernelWidth), m_kernelHeight(kernelHeight), 
                          m_horizontalSubsample(horizontalSubsample), m_verticalSubsample(verticalSubsample),
                          m_zeroPadding(zeroPadding), m_maxTempMemSizeInSamples(maxTempMemSizeInSamples)
        {
            m_outputChannels = outputChannels;
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        ConvolutionNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") 
            : ComputationNode<ElemType>(deviceId), m_tempMatrix(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }
                
        virtual void SaveToFile(File& fstream) const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream <<  m_kernelWidth << m_kernelHeight << m_horizontalSubsample << m_verticalSubsample; 
            fstream << m_outputChannels << m_zeroPadding << m_maxTempMemSizeInSamples; 
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            fstream >>  m_kernelWidth >> m_kernelHeight >> m_horizontalSubsample >> m_verticalSubsample; 
            fstream >> m_outputChannels >> m_zeroPadding >> m_maxTempMemSizeInSamples; 
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            ConvolutionNode<ElemType>* node = (ConvolutionNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_kernelWidth = m_kernelWidth;
                node->m_kernelHeight = m_kernelHeight;

                node->m_horizontalSubsample = m_horizontalSubsample;
                node->m_verticalSubsample = m_verticalSubsample;

                node->m_zeroPadding = m_zeroPadding;

                node->m_maxTempMemSizeInSamples = m_maxTempMemSizeInSamples;

                node->m_tempMatrix = m_tempMatrix;
            }
        }

        // copy constructor
        ConvolutionNode(const ConvolutionNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) 
            : ComputationNode<ElemType>(node->m_deviceId), m_tempMatrix(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new ConvolutionNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Convolution";} 

        ConvolutionParams GetConvolutionParams() const
        {
            ConvolutionParams convParam;
            convParam.inputWidth = m_inputWidth;
            convParam.inputHeight = m_inputHeight;
            convParam.inputChannels = m_inputChannels;

            convParam.kernelWidth = m_kernelWidth;
            convParam.kernelHeight = m_kernelHeight;

            convParam.horizontalSubsample = m_horizontalSubsample;
            convParam.verticalSubsample = m_verticalSubsample;

            convParam.outputWidth = m_outputWidth;
            convParam.outputHeight = m_outputHeight;
            convParam.outputChannels = m_outputChannels;

            convParam.zeroPadding = m_zeroPadding;

            convParam.maxTempMemSizeInSamples = m_maxTempMemSizeInSamples;
            return convParam;
        }

        virtual void ComputeInputPartial(const size_t inputIndex) 
        {
            if (inputIndex > 1)
                throw std::invalid_argument("Convolution operation only takes two inputs.");

            if (inputIndex == 0)  //derivative with regard to the weight matrix
            {
                ComputeInputPartialOverWeight(this, GradientValues(), Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_tempMatrix, true);
            }
            else  // derivative with regard to the input feature
            {
                ComputeInputPartialOverInputFeature(this, GradientValues(), Inputs(1)->GradientValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_tempMatrix);
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq) 
        {
            if (inputIndex > 1)
                throw std::invalid_argument("Convolution operation only takes two inputs.");

            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (inputIndex == 0)  //derivative with regard to the weight matrix
            {
                ComputeInputPartialOverWeight(this, sliceOutputGrad, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), sliceInput1Value, m_tempMatrix);
            }
            else  // derivative with regard to the input feature
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                ComputeInputPartialOverInputFeature(this, sliceOutputGrad, sliceInput1Grad, Inputs(0)->FunctionValues(), sliceInput1Value, m_tempMatrix);
            }
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(this, FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_tempMatrix);
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(this, sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value, m_tempMatrix);
        }

        static void WINAPI EvaluateThisNodeS(const ConvolutionNode<ElemType>* pConv, Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0, 
            const Matrix<ElemType> &input1, Matrix<ElemType> &tempMatrix)
        {
#if NANCHECK
            input0.HasNan("Convolution-input0");
            input1.HasNan("Convolution-input1");
#endif
            ConvolutionParams convParam = pConv->GetConvolutionParams();

            size_t packedInputRows = convParam.kernelWidth * convParam.kernelHeight * convParam.inputChannels;
            size_t packedInputColsPerSample = convParam.outputWidth * convParam.outputHeight;
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = convParam.inputWidth * convParam.inputHeight * convParam.inputChannels;  //size of each input sample

            long batchSize = (long)input1.GetNumCols();  //right child is the input sample

            long maxTempMemSizeInSamples = (long)(convParam.maxTempMemSizeInSamples == 0? batchSize : convParam.maxTempMemSizeInSamples);

            const Matrix<ElemType> & weightMatrix = input0;
            assert(weightMatrix.GetNumCols() == packedInputRows && weightMatrix.GetNumRows() == convParam.outputChannels);
            functionValues.Resize(convParam.outputChannels, outputSizePerChannel * batchSize);

            long subBatchSize = (long)min(batchSize, maxTempMemSizeInSamples); 
            long numSubBatches = (batchSize+subBatchSize-1)/subBatchSize; 

            for (long i=0; i<numSubBatches; i++) 
            {
                long startSampleID = i*subBatchSize; 
                long endSampleID = min(batchSize, startSampleID + subBatchSize); 
                long smallBatchSize = endSampleID-startSampleID; 

                tempMatrix.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                Matrix<ElemType>  inputSubBatch = input1.ColumnSlice(startSampleID, smallBatchSize);
                tempMatrix.AssignPackedConvolutionInput(inputSubBatch, 
                                                                 convParam.inputWidth, convParam.inputHeight, convParam.inputChannels,
                                                                 convParam.outputWidth, convParam.outputHeight, convParam.outputChannels,
                                                                 convParam.kernelWidth, convParam.kernelHeight, convParam.horizontalSubsample, convParam.verticalSubsample, 
                                                                 convParam.zeroPadding); 

                Matrix<ElemType>  outputSubBatch = functionValues.ColumnSlice(outputSizePerChannel * startSampleID, outputSizePerChannel * smallBatchSize);
                Matrix<ElemType>::Multiply(weightMatrix, false, tempMatrix, false, outputSubBatch);
            }

            functionValues.Reshape(convParam.outputChannels * outputSizePerChannel, batchSize);  //each sample becomes a column

#if NANCHECK
            functionValues.HasNan("Convolution");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("ConvolutionNode requires two inputs.");

            //we may want to remove this check in the future if we want to support the case that the weight itself is result of some computation 
            //if (Inputs(0)->OperationName() != LearnableParameter<ElemType>::TypeName())
            //    throw std::logic_error("ConvolutionNode requires the first input to be LearnableParameter type.");

            if (m_horizontalSubsample > m_kernelWidth || m_verticalSubsample > m_kernelHeight)
                throw std::invalid_argument("In ConvolutionNode horizontalSubsample must <= kernelWidth and verticalSubsample must <= kernelHeight.");

            CopyImageSizeFromInputs();

            size_t weightCols = m_kernelWidth * m_kernelHeight * m_inputChannels;

            if (Inputs(0)->OperationName() == LearnableParameter<ElemType>::TypeName() && Inputs(0)->FunctionValues().GetNumElements() == 0)
            {
                Inputs(0)->FunctionValues().Resize(m_outputChannels, weightCols);
            }

            if (m_children[0]->FunctionValues().GetNumCols() != weightCols || m_children[0]->FunctionValues().GetNumRows() != m_outputChannels)
            {
                msra::strfun::strprintf msg("convolutionWeight matrix %ls should have dimension [%d, %d] which is [outputChannels, kernelWidth * kernelHeight * inputChannels]", 
                    m_children[0]->NodeName().c_str(), m_outputChannels, weightCols);
                throw std::logic_error(msg.c_str());            
            }

            size_t inputDim = m_inputWidth * m_inputHeight * m_inputChannels;
            if (Inputs(1)->OperationName() == LearnableParameter<ElemType>::TypeName() && Inputs(1)->FunctionValues().GetNumRows() == 0)
            {
                Inputs(1)->FunctionValues().Resize(inputDim, Inputs(1)->FunctionValues().GetNumCols());
            }

            if (m_children[1]->FunctionValues().GetNumRows() != inputDim)
            {
                msra::strfun::strprintf msg("each column of input to the convolution node %ls is a sample and should have dimension %d, which is inputWidth * inputHeight * inputChannels", 
                    NodeName().c_str(), inputDim);
                throw std::logic_error(msg.c_str());            
            }

            if (Inputs(0)->FunctionValues().GetNumElements() == 0 || Inputs(1)->FunctionValues().GetNumElements() == 0 )
                throw std::logic_error("Convolution operation: one of the operants has 0 element.");
            
            size_t outputDim = m_outputWidth * m_outputHeight * m_outputChannels;
            FunctionValues().Resize(outputDim, m_children[1]->FunctionValues().GetNumCols());
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(1, false);

            if (m_inputWidth < m_kernelWidth || m_inputHeight < m_kernelHeight)
                throw std::invalid_argument("inputWidth must >= kernelWidth and inputHeight must >= kernelHeight.");

            if (m_zeroPadding)
            {
                const int kernelWidthCenter = m_kernelWidth % 2;
                const int kernelHeightCenter = m_kernelHeight % 2;
                m_outputWidth = (m_inputWidth-kernelWidthCenter)/m_horizontalSubsample + 1;
                m_outputHeight = (m_inputHeight-kernelHeightCenter)/m_verticalSubsample + 1;
            }
            else
            {
                m_outputWidth = (m_inputWidth-m_kernelWidth)/m_horizontalSubsample + 1;
                m_outputHeight = (m_inputHeight-m_kernelHeight)/m_verticalSubsample + 1;
            }    
        }

        virtual void AttachInputs(const ComputationNodePtr convolutionWeight, const ComputationNodePtr inputFeature) 
        {
            m_children.resize(2);
            m_children[0] = convolutionWeight;
            m_children[1] = inputFeature;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);

            if (deviceId != AUTOPLACEMATRIX)
            {
                if (m_tempMatrix.GetDeviceId() != deviceId)
                    m_tempMatrix.TransferFromDeviceToDevice(m_tempMatrix.GetDeviceId(), deviceId);
            }
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            ComputationNode<ElemType>::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "Input[Width:%lu, Height:%lu, Channels:%lu]  \n", m_inputWidth, m_inputHeight, m_inputChannels);
            fstream << string(str);
            sprintf(str, "Kernel[Width:%lu, Height:%lu]  SubSample[Horizontal:%lu, Vertical:%lu]\n", m_kernelWidth, m_kernelHeight, m_horizontalSubsample, m_verticalSubsample);
            fstream << string(str);
            sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", m_outputWidth, m_outputHeight, m_outputChannels);
            fstream << string(str);
            sprintf(str, "ZeroPadding=%ls  maxTempMemSizeInSamples=%lu\n", m_zeroPadding? L"true" : L"false", m_maxTempMemSizeInSamples);
            fstream << string(str);
        }

        void SetmMaxTempMemSizeInSamples(const size_t maxTempMemSizeInSamples)
        {
            m_maxTempMemSizeInSamples = maxTempMemSizeInSamples;
        }

    private:
        static void WINAPI ComputeInputPartialOverWeight(const ConvolutionNode<ElemType>* pConv, Matrix<ElemType> &gradientValues, 
            Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &/*input0*/, const Matrix<ElemType> &input1, Matrix<ElemType> &tempMatrix, const bool inLoop=false)
        {
            ConvolutionParams convParam = pConv->GetConvolutionParams();

            size_t packedInputRows = convParam.kernelWidth * convParam.kernelHeight * convParam.inputChannels;
            size_t packedInputColsPerSample = convParam.outputWidth * convParam.outputHeight;
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = convParam.inputWidth * convParam.inputHeight * convParam.inputChannels;  //size of each input sample

            long batchSize = (long) input1.GetNumCols(); //right child is the input sample

            long maxTempMemSizeInSamples = (long) (convParam.maxTempMemSizeInSamples == 0? batchSize : convParam.maxTempMemSizeInSamples);

            //const Matrix<ElemType> & weightMatrix = input0;
            //inputGradientValues.Resize(weightMatrix.GetNumRows(), weightMatrix.GetNumCols()); //should have been resized when preparing gradient computation

            gradientValues.Reshape(convParam.outputChannels,  outputSizePerChannel * batchSize);  //reshape to match the longernal operation

            long subBatchSize = min(batchSize, maxTempMemSizeInSamples); 
            long numSubBatches = (batchSize+subBatchSize-1)/subBatchSize; 

            if (numSubBatches == 1 && !inLoop)  //reuse packed input from evaluation step if it's not changed by either subbatch or recurrent steps.
            {
                Matrix<ElemType>::MultiplyAndAdd(gradientValues, false, tempMatrix, true, inputGradientValues);
            }
            else
            {
                for (long i=0; i<numSubBatches; i++) 
                {
                    long startSampleID = i*subBatchSize; 
                    long endSampleID = min(batchSize, startSampleID + subBatchSize); 
                    long smallBatchSize = endSampleID-startSampleID; 

                    tempMatrix.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                    Matrix<ElemType> inputSubBatch = input1.ColumnSlice(startSampleID, smallBatchSize);
                    tempMatrix.AssignPackedConvolutionInput(inputSubBatch, 
                                                                     convParam.inputWidth, convParam.inputHeight, convParam.inputChannels,
                                                                     convParam.outputWidth, convParam.outputHeight, convParam.outputChannels,
                                                                     convParam.kernelWidth, convParam.kernelHeight, convParam.horizontalSubsample, convParam.verticalSubsample, 
                                                                     convParam.zeroPadding); 

                    Matrix<ElemType> outputGradientSubBatch = gradientValues.ColumnSlice(startSampleID * outputSizePerChannel, smallBatchSize * outputSizePerChannel);
                    Matrix<ElemType>::MultiplyAndAdd(outputGradientSubBatch, false, tempMatrix, true, inputGradientValues);
                }
            }

            gradientValues.Reshape(convParam.outputChannels * outputSizePerChannel, batchSize);  //change back
        }

        //compute gradient over the packed input and then convert the result to the original input
        static void WINAPI ComputeInputPartialOverInputFeature(const ConvolutionNode<ElemType>* pConv, Matrix<ElemType> &gradientValues, const Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &input0, const Matrix<ElemType> &input1, Matrix<ElemType> &tempMatrix)
        {
            
            ConvolutionParams convParam = pConv->GetConvolutionParams();
            size_t packedInputRows = convParam.kernelWidth * convParam.kernelHeight * convParam.inputChannels;
            size_t packedInputColsPerSample = convParam.outputWidth * convParam.outputHeight;
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = convParam.inputWidth * convParam.inputHeight * convParam.inputChannels;  //size of each input sample

            long batchSize = (long) input1.GetNumCols(); //right child is the input sample

            long maxTempMemSizeInSamples = (long) (convParam.maxTempMemSizeInSamples == 0? batchSize : convParam.maxTempMemSizeInSamples);

            const Matrix<ElemType> & weightMatrix = input0;

            gradientValues.Reshape(convParam.outputChannels,  outputSizePerChannel * batchSize);  //reshape to match the longernal operation

            long subBatchSize = min(batchSize, maxTempMemSizeInSamples); 
            long numSubBatches = (batchSize+subBatchSize-1)/subBatchSize; 

            for (long i=0; i<numSubBatches; i++) 
            {
                long startSampleID = i*subBatchSize; 
                long endSampleID = min(batchSize, startSampleID + subBatchSize); 
                long smallBatchSize = endSampleID-startSampleID; 

                tempMatrix.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                Matrix<ElemType> outputGradientSubBatch = gradientValues.ColumnSlice(startSampleID * outputSizePerChannel, smallBatchSize * outputSizePerChannel);
                Matrix<ElemType>::Multiply(weightMatrix, true, outputGradientSubBatch, false,  tempMatrix);

                Matrix<ElemType> inputGradientSubBatch = inputGradientValues.ColumnSlice(startSampleID, smallBatchSize);
                tempMatrix.UnpackConvolutionInput(inputGradientSubBatch, 
                                                                 convParam.inputWidth, convParam.inputHeight, convParam.inputChannels,
                                                                 convParam.outputWidth, convParam.outputHeight, convParam.outputChannels,
                                                                 convParam.kernelWidth, convParam.kernelHeight, convParam.horizontalSubsample, convParam.verticalSubsample, 
                                                                 convParam.zeroPadding); 
            }

            gradientValues.Reshape(convParam.outputChannels * outputSizePerChannel, batchSize);  //change back
        }
        

    private:
        size_t m_kernelWidth, m_kernelHeight;
        size_t m_horizontalSubsample, m_verticalSubsample;
        bool m_zeroPadding;

        Matrix<ElemType> m_tempMatrix; 
        size_t m_maxTempMemSizeInSamples; // can change during runtime
    };

    template class ConvolutionNode<float>; 
    template class ConvolutionNode<double>;

    struct PoolParams
    {
        size_t inputWidth, inputHeight, inputChannels;
        size_t windowWidth, windowHeight;
        size_t horizontalSubsample, verticalSubsample;
        size_t outputWidth, outputHeight, outputChannels;
        size_t inputSizePerSample, outputSizePerSample;
    };

    //Max Pooling: support multi channel
    //assume each column is an input sample. Each sample is stored in  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
    template<class ElemType>
    class MaxPoolingNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        MaxPoolingNode( const size_t windowWidth, const size_t windowHeight, 
                        const size_t horizontalSubsample, const size_t verticalSubsample, 
                        const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId),
                          m_windowWidth(windowWidth), m_windowHeight(windowHeight),
                          m_horizontalSubsample(horizontalSubsample), m_verticalSubsample(verticalSubsample)                       
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }
                
        MaxPoolingNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }
                
        virtual void SaveToFile(File& fstream) const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << m_windowWidth << m_windowHeight << m_horizontalSubsample << m_verticalSubsample; 
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            fstream >> m_windowWidth >> m_windowHeight >> m_horizontalSubsample >> m_verticalSubsample; 
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            MaxPoolingNode<ElemType>* node = (MaxPoolingNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_inputWidth = m_inputWidth;
                node->m_inputHeight = m_inputHeight;
                node->m_inputChannels = m_inputChannels;

                node->m_windowWidth = m_windowWidth;
                node->m_windowHeight = m_windowHeight;

                node->m_horizontalSubsample = m_horizontalSubsample;
                node->m_verticalSubsample = m_verticalSubsample;

                node->m_outputWidth = m_outputWidth;
                node->m_outputHeight = m_outputHeight;
                node->m_outputChannels = m_outputChannels;

                node->m_inputSizePerSample = m_inputSizePerSample;
                node->m_outputSizePerSample = m_outputSizePerSample;
            }
        }
        // copy constructor
        MaxPoolingNode(const MaxPoolingNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new MaxPoolingNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"MaxPooling";}

        PoolParams GetPoolParams() const
        {
            PoolParams poolParams;
            poolParams.inputWidth = m_inputWidth;
            poolParams.inputHeight = m_inputHeight;
            poolParams.inputChannels = m_inputChannels;

            poolParams.windowWidth = m_windowWidth;
            poolParams.windowHeight = m_windowHeight;

            poolParams.horizontalSubsample = m_horizontalSubsample;
            poolParams.verticalSubsample = m_verticalSubsample;

            poolParams.outputWidth = m_outputWidth;
            poolParams.outputHeight = m_outputHeight;
            poolParams.outputChannels = m_outputChannels;

            poolParams.inputSizePerSample = m_inputSizePerSample;
            poolParams.outputSizePerSample = m_outputSizePerSample;
            return poolParams;
        }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("MaxPooling operation only takes one inputs.");

            ComputeInputPartialS(this, GradientValues(), Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), FunctionValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq) 
        {
            if (inputIndex > 0)
                throw std::invalid_argument("MaxPooling operation only takes one inputs.");

            Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(this, sliceOutputGrad, sliceInput0Grad, sliceInput0Value, sliceOutputValue);
        }

        static void WINAPI ComputeInputPartialS(const MaxPoolingNode<ElemType>* ppool, const Matrix<ElemType> &gradientValues, Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &input0, const Matrix<ElemType> &functionValues)
        {
            PoolParams poolParams = ppool->GetPoolParams();

            inputGradientValues.AddMaxPoolingGradient(gradientValues, input0, functionValues, poolParams.inputChannels,
                                                    poolParams.inputWidth, poolParams.inputHeight, poolParams.inputSizePerSample, 
                                                    poolParams.outputWidth, poolParams.outputHeight, poolParams.outputSizePerSample, 
                                                    poolParams.windowWidth, poolParams.windowHeight, poolParams.horizontalSubsample, poolParams.verticalSubsample);
        }

        virtual void EvaluateThisNode()  
        {
#if NANCHECK
            Inputs(0)->FunctionValues().HasNan("MaxPooling-input0");
#endif
            EvaluateThisNodeS(this, FunctionValues(), Inputs(0)->FunctionValues());
#if NANCHECK
            m_functionValues.HasNan("MaxPooling");
#endif
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(this, sliceOutputValue, sliceInput0Value);
        }

        static void WINAPI EvaluateThisNodeS(const MaxPoolingNode<ElemType>* ppool, Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0)
        {
            PoolParams poolParams = ppool->GetPoolParams();
            functionValues.AssignMaxPoolingResult(input0, poolParams.inputChannels,
                                                 poolParams.inputWidth, poolParams.inputHeight, poolParams.inputSizePerSample, 
                                                 poolParams.outputWidth, poolParams.outputHeight, poolParams.outputSizePerSample, 
                                                 poolParams.windowWidth, poolParams.windowHeight, poolParams.horizontalSubsample, poolParams.verticalSubsample);
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("MaxPoolingNode requires one input.");

            if (m_horizontalSubsample > m_windowWidth || m_verticalSubsample > m_windowHeight)
                throw std::invalid_argument("MaxPoolingNode: horizontalSubsample must <= windowWidth and verticalSubsample must <= windowHeight.");

            CopyImageSizeFromInputs();

            m_inputSizePerSample = m_inputWidth * m_inputHeight * m_inputChannels;
            m_outputSizePerSample = m_outputWidth * m_outputHeight * m_outputChannels;

            if (Inputs(0)->OperationName() == LearnableParameter<ElemType>::TypeName() && Inputs(0)->FunctionValues().GetNumRows() == 0)
            {
                Inputs(0)->FunctionValues().Resize(m_inputSizePerSample, Inputs(0)->FunctionValues().GetNumCols());
            }

            if (m_children[0]->FunctionValues().GetNumRows() != m_inputSizePerSample)
            {
                msra::strfun::strprintf msg("each column of input to the MaxPooling node %ls is a sample and should have dimension %d, which is inputWidth * inputHeight * inputChannels", 
                    NodeName().c_str(), m_inputSizePerSample);
                throw std::logic_error(msg.c_str());            
            }
            
            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("MaxPoolingNode operation: the input node has 0 element.");

            m_functionValues.Resize(m_outputSizePerSample, m_children[0]->FunctionValues().GetNumCols());
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, false);

            if (m_inputWidth < m_windowWidth || m_inputHeight < m_windowHeight)
                throw std::invalid_argument("MaxPoolingNode: inputWidth must >= windowWidth and inputHeight must >= windowHeight.");

            m_outputWidth = (m_inputWidth-m_windowWidth)/m_horizontalSubsample + 1;
            m_outputHeight = (m_inputHeight-m_windowHeight)/m_verticalSubsample + 1;
            m_outputChannels = m_inputChannels;
        }

        virtual void AttachInputs(const ComputationNodePtr inputFeature) 
        {
            m_children.resize(1);
            m_children[0] = inputFeature;
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            ComputationNode<ElemType>::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "Input[Width:%lu, Height:%lu, Channels:%lu]  \n", m_inputWidth, m_inputHeight, m_inputChannels);
            fstream << string(str);
            sprintf(str, "PoolingWindow[Width:%lu, Height:%lu]  SubSampling[Horizontal:%lu, Vertical:%lu]\n", m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
            fstream << string(str);
            sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", m_outputWidth, m_outputHeight, m_outputChannels);
            fstream << string(str);
            sprintf(str, "TotalSizePerSample[Input:%lu, Output:%lu]  \n", m_inputSizePerSample, m_outputSizePerSample);
            fstream << string(str);
        }

    private:
        size_t m_windowWidth, m_windowHeight;
        size_t m_horizontalSubsample, m_verticalSubsample;
        size_t m_inputSizePerSample, m_outputSizePerSample;
    };

    template class MaxPoolingNode<float>; 
    template class MaxPoolingNode<double>;    

    //Average Pooling: support multi channel
    //assume each column is an input sample. Each sample is stored in  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
    template<class ElemType>
    class AveragePoolingNode : public ComputationNode<ElemType>
    {
        UsingComputationNodeMembers;
    public:
        AveragePoolingNode(const size_t windowWidth, const size_t windowHeight, 
                        const size_t horizontalSubsample, const size_t verticalSubsample, 
                        const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId),
                          m_windowWidth(windowWidth), m_windowHeight(windowHeight),
                          m_horizontalSubsample(horizontalSubsample), m_verticalSubsample(verticalSubsample)                     
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            m_deviceId = deviceId;
            MoveMatricesToDevice(deviceId);
            InitRecurrentNode();
        }

        AveragePoolingNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const std::wstring name = L"") : ComputationNode<ElemType>(deviceId)
        {
            m_nodeName = (name == L""? CreateUniqNodeName() : name);
            LoadFromFile(fstream, modelVersion, deviceId);
        }                
                
        virtual void SaveToFile(File& fstream) const
        {
            ComputationNode<ElemType>::SaveToFile(fstream);

            fstream << m_windowWidth << m_windowHeight << m_horizontalSubsample << m_verticalSubsample; 
        }

        virtual void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId=AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            fstream >> m_windowWidth >> m_windowHeight >> m_horizontalSubsample >> m_verticalSubsample; 
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            AveragePoolingNode<ElemType>* node = (AveragePoolingNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_inputWidth = m_inputWidth;
                node->m_inputHeight = m_inputHeight;
                node->m_inputChannels = m_inputChannels;

                node->m_windowWidth = m_windowWidth;
                node->m_windowHeight = m_windowHeight;

                node->m_horizontalSubsample = m_horizontalSubsample;
                node->m_verticalSubsample = m_verticalSubsample;

                node->m_outputWidth = m_outputWidth;
                node->m_outputHeight = m_outputHeight;
                node->m_outputChannels = m_outputChannels;

                node->m_inputSizePerSample = m_inputSizePerSample;
                node->m_outputSizePerSample = m_outputSizePerSample;
            }
        }

        // copy constructor
        AveragePoolingNode(const AveragePoolingNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags) : ComputationNode<ElemType>(node->m_deviceId)
        {
            node->CopyTo(this, newName, flags);
        }

        virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
        {
            const std::wstring& name = (newName == L"")?NodeName():newName;
                
            ComputationNodePtr node = new AveragePoolingNode<ElemType>(this, name, flags);
            return node;
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"AveragePooling";}
        PoolParams GetPoolParams() const
        {
            PoolParams poolParams;
            poolParams.inputWidth = m_inputWidth;
            poolParams.inputHeight = m_inputHeight;
            poolParams.inputChannels = m_inputChannels;

            poolParams.windowWidth = m_windowWidth;
            poolParams.windowHeight = m_windowHeight;

            poolParams.horizontalSubsample = m_horizontalSubsample;
            poolParams.verticalSubsample = m_verticalSubsample;

            poolParams.outputWidth = m_outputWidth;
            poolParams.outputHeight = m_outputHeight;
            poolParams.outputChannels = m_outputChannels;

            poolParams.inputSizePerSample = m_inputSizePerSample;
            poolParams.outputSizePerSample = m_outputSizePerSample;
            return poolParams;
        }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("AveragePooling operation only takes one inputs.");

            ComputeInputPartialS(this, GradientValues(), Inputs(0)->GradientValues());
        }

        virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq) 
        {
            if (inputIndex > 0)
                throw std::invalid_argument("AveragePooling operation only takes one inputs.");

            Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            ComputeInputPartialS(this, sliceOutputGrad, sliceInput0Grad);
        }

        static void WINAPI ComputeInputPartialS(const AveragePoolingNode<ElemType>* ppool, const Matrix<ElemType> &gradientValues, Matrix<ElemType> &inputGradientValues)
        {
            PoolParams poolParams = ppool->GetPoolParams();

            inputGradientValues.AddAveragePoolingGradient(gradientValues, poolParams.inputChannels,
                                                    poolParams.inputWidth, poolParams.inputHeight, poolParams.inputSizePerSample, 
                                                    poolParams.outputWidth, poolParams.outputHeight, poolParams.outputSizePerSample, 
                                                    poolParams.windowWidth, poolParams.windowHeight, poolParams.horizontalSubsample, poolParams.verticalSubsample);
        }

        virtual void EvaluateThisNode()  
        {
#if NANCHECK
            Inputs(0)->FunctionValues().HasNan("AveragePooling-input0");
#endif
            EvaluateThisNodeS(this, FunctionValues(), Inputs(0)->FunctionValues());
#if NANCHECK
            m_functionValues.HasNan("AveragePooling");
#endif
        }

        virtual void EvaluateThisNode(const size_t timeIdxInSeq) 
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(this, sliceOutputValue, sliceInput0Value);
        }

        static void WINAPI EvaluateThisNodeS(const AveragePoolingNode<ElemType>* ppool, Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0)
        {
            PoolParams poolParams = ppool->GetPoolParams();
            
            functionValues.AssignAveragePoolingResult(input0, poolParams.inputChannels,
                                                 poolParams.inputWidth, poolParams.inputHeight, poolParams.inputSizePerSample, 
                                                 poolParams.outputWidth, poolParams.outputHeight, poolParams.outputSizePerSample, 
                                                 poolParams.windowWidth, poolParams.windowHeight, poolParams.horizontalSubsample, poolParams.verticalSubsample);
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("AveragePoolingNode requires one input.");

            if (m_horizontalSubsample > m_windowWidth || m_verticalSubsample > m_windowHeight)
                throw std::invalid_argument("AveragePoolingNode: horizontalSubsample must <= windowWidth and verticalSubsample must <= windowHeight.");

            CopyImageSizeFromInputs();

            m_inputSizePerSample = m_inputWidth * m_inputHeight * m_inputChannels;
            m_outputSizePerSample = m_outputWidth * m_outputHeight * m_outputChannels;

            if (Inputs(0)->OperationName() == LearnableParameter<ElemType>::TypeName() && Inputs(0)->FunctionValues().GetNumRows() == 0)
            {
                Inputs(0)->FunctionValues().Resize(m_inputSizePerSample, Inputs(0)->FunctionValues().GetNumCols());
            }

            if (m_children[0]->FunctionValues().GetNumRows() != m_inputSizePerSample)
            {
                msra::strfun::strprintf msg("each column of input to the AveragePooling node %ls is a sample and should have dimension %d, which is inputWidth * inputHeight * inputChannels", 
                    NodeName().c_str(), m_inputSizePerSample);
                throw std::logic_error(msg.c_str());            
            }
                        
            if (Inputs(0)->FunctionValues().GetNumElements() == 0)
                throw std::logic_error("AveragePoolingNode operation: the input node has 0 element.");

            FunctionValues().Resize(m_outputSizePerSample, m_children[0]->FunctionValues().GetNumCols());
        }

        virtual void CopyImageSizeFromInputs()
        {
            CopyImageSizeFromInput(0, false);

            if (m_inputWidth < m_windowWidth || m_inputHeight < m_windowHeight)
                throw std::invalid_argument("AveragePoolingNode: inputWidth must >= windowWidth and inputHeight must >= windowHeight.");

            m_outputWidth = (m_inputWidth-m_windowWidth)/m_horizontalSubsample + 1;
            m_outputHeight = (m_inputHeight-m_windowHeight)/m_verticalSubsample + 1;
            m_outputChannels = m_inputChannels;
        }

        virtual void AttachInputs(const ComputationNodePtr inputFeature) 
        {
            m_children.resize(1);
            m_children[0] = inputFeature;
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            ComputationNode<ElemType>::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "Input[Width:%lu, Height:%lu, Channels:%lu]  \n", m_inputWidth, m_inputHeight, m_inputChannels);
            fstream << string(str);
            sprintf(str, "PoolingWindow[Width:%lu, Height:%lu]  SubSample[Horizontal:%lu, Vertical:%lu]\n", m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
            fstream << string(str);
            sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", m_outputWidth, m_outputHeight, m_outputChannels);
            fstream << string(str);
            sprintf(str, "TotalSizePerSample[Input:%lu, Output:%lu]\n", m_inputSizePerSample, m_outputSizePerSample);
            fstream << string(str);
        }

    private:
        size_t m_windowWidth, m_windowHeight;
        size_t m_horizontalSubsample, m_verticalSubsample;
        size_t m_inputSizePerSample, m_outputSizePerSample;
    };

    template class AveragePoolingNode<float>; 
    template class AveragePoolingNode<double>;    

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
        static void WINAPI EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& unnormedPrior, const Matrix<ElemType>& mean, const Matrix<ElemType>& logstddev,
            const Matrix<ElemType>& feature, Matrix<ElemType>& prior, Matrix<ElemType>& stddev, Matrix<ElemType>& normedDeviationVectors,
            Matrix<ElemType>& normedDeviation, Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
        {
            int numComponent = unnormedPrior.GetNumRows();
            size_t numSamples = feature.GetNumCols();
            size_t featureDim = feature.GetNumRows();

            //compute prior which is softmax of unnormedPrior
            prior.AssignLogSoftmaxOf(unnormedPrior, true);
            prior.InplaceExp();

            //compute stddev
            stddev.AssignExpOf(logstddev);

            //compute normedDeviation <-- ||x-u_c||^2/(stddev^2)
            normedDeviationVectors.AssignRepeatOf(feature, numComponent, 1);
            normedDeviationVectors -= mean; //each column of the mean has multiple mean components
            normedDeviationVectors.Reshape(featureDim, numSamples* numComponent);  //now each column is feature-mean_i

            normedDeviation.AssignVectorNorm2Of(normedDeviationVectors, true);
            temp.AssignRepeatOf(stddev, 1, numSamples / stddev.GetNumCols());  //stddev.GetNumCols() is either 1 or =numSamples
            temp.Reshape(1, temp.GetNumElements());  //one stddev value for each component for each sample
            normedDeviation.ElementDivideBy(temp);  //normedDeviation and temp have same dim (1, numSamples* numComponent)
            normedDeviation ^= 2;

            //compute  normedDeviationVectors <-- (x-u_c)/(stddev^2)
            normedDeviationVectors.RowElementDivideBy(temp); //temp is stddev. divide once  
            normedDeviationVectors.RowElementDivideBy(temp);  //divide twice
            normedDeviationVectors.Reshape(featureDim*numComponent, numSamples);  //reshape back

            //compute per-component likelihood
            posterior.AssignProductOf(-0.5f, normedDeviation); //posterior  <-- -||x-u_c||^2/(stddev^2)/2 and in (1, numSamples* numComponent) dim
            posterior.InplaceExp(); //posterior  <-- exp(-||x-u_c||^2/(stddev^2)/2)
            temp ^= (ElemType)numComponent; //temp <-- stddev^c and in (1, numSamples* numComponent) dim
            posterior.RowElementDivideBy(temp);  // posterior  <-- exp[-||x-u_c||^2/(stddev^2)/2]/(stddev^c)
            posterior /= (ElemType)pow(TWO_PI, numComponent / 2.0f); //likelihood for each component and sample is now computed and stored in posterior

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

#if NANCHECK
            functionValues.HasNan("GMMLogLikelihood");
#endif
#if DUMPOUTPUT
            functionValues.Print("GMMLogLikelihoodNode");
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

    /**
    BatchModeNode is a derivative of ComputationNode.
    It additionally check if needs to process data in batch before processing its parent
    This is used in case of beam search decoding. Batchmode node must be processed before other nodes.
    It differs from PreComputeNode in that precompute done is done before the entire corpus.
    This is done before forward computation of all nodes. 
    */
    template<class ElemType>
    class BatchModeNode : public ComputationNode<ElemType>
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
            swprintf(str, BUFLEN, L"HasComputed=%ws", HasComputed() ? L"true" : L"false");
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
            : ComputationNode(deviceId), mState(deviceId), mPastState(deviceId),
            mPastOutput(deviceId), mGi(deviceId), mGf(deviceId), mGo(deviceId), grdToObs(deviceId), grdToInputGate(deviceId),
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
            InitRecurrentNode();
            m_inputDim = 0;
            m_outputDim = 0;
            m_use_errors_from_future_minibatch = false;
            mDefaultState = (ElemType) DEFAULT_HIDDEN_ACTIVITY;
        }

        LSTMNode(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
            : ComputationNode(deviceId), mState(deviceId), mPastState(deviceId), mPastOutput(deviceId), mGi(deviceId), mGf(deviceId), mGo(deviceId), grdToObs(deviceId), grdToInputGate(deviceId), grdToForgetGate(deviceId), grdToOutputGate(deviceId), grdToCellWgt(deviceId), tanhObs(deviceId), tanhState(deviceId), m_tempMatrix(deviceId), mSlicePrevState(deviceId), mSlicePrevOutput(deviceId),
            grdBeforeInputGate(deviceId),
            grdBeforeForget(deviceId), grdBeforeGo(deviceId), grdToCell(deviceId),
            grdBeforeTanhInputGate(deviceId), m_obs_error_from_future_minibatch(deviceId),
            m_state_error_from_future_minibatch(deviceId), mLastState(deviceId), mLastOutput(deviceId)
        {
            m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
            m_inputDim = 0;
            m_outputDim = 0;
            m_use_errors_from_future_minibatch = false;
            mDefaultState = (ElemType)DEFAULT_HIDDEN_ACTIVITY;
            LoadFromFile(fstream, modelVersion, deviceId);
        }

        // copy constructor
        LSTMNode(const LSTMNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
            : ComputationNode(node->m_deviceId), mState(node->m_deviceId), mPastState(node->m_deviceId), mPastOutput(node->m_deviceId), mGi(node->m_deviceId), mGf(node->m_deviceId), mGo(node->m_deviceId), grdToObs(node->m_deviceId), grdToInputGate(node->m_deviceId), grdToForgetGate(node->m_deviceId), grdToOutputGate(node->m_deviceId), grdToCellWgt(node->m_deviceId), tanhObs(node->m_deviceId), tanhState(node->m_deviceId), m_tempMatrix(node->m_deviceId), mSlicePrevState(node->m_deviceId), mSlicePrevOutput(node->m_deviceId),
            grdBeforeInputGate(node->m_deviceId),
            grdBeforeForget(node->m_deviceId), grdBeforeGo(node->m_deviceId), grdToCell(node->m_deviceId),
            grdBeforeTanhInputGate(node->m_deviceId), m_obs_error_from_future_minibatch(node->m_deviceId),
            m_state_error_from_future_minibatch(node->m_deviceId), mLastState(node->m_deviceId), mLastOutput(node->m_deviceId)
        {
            m_use_errors_from_future_minibatch = false;
            node->CopyTo(this, newName, flags);
            mDefaultState = (ElemType) DEFAULT_HIDDEN_ACTIVITY;
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
            fstream << mDefaultState;
        }

        void LoadFromFile(File& fstream, const size_t modelVersion, const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
        {
            ComputationNode<ElemType>::LoadFromFile(fstream, modelVersion, deviceId);

            if (modelVersion == 2)
                fstream >> m_inputDim >> m_outputDim;
            fstream >> mDefaultState;
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
            LSTMNode<ElemType>* node = (LSTMNode<ElemType>*) nodeP;

            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_inputDim = m_inputDim;
                node->m_outputDim = m_outputDim;

                node->mState = mState;  /// hidden state activity
                node->mPastState = mPastState; /// state activity in the previous minibatch
                node->mPastOutput = mPastOutput; /// output in the previou minibatch 

                node->mGi = mGi;     /// input gate activity
                node->mGf = mGf;     /// forget gate activity
                node->mGo = mGo;     /// output gate activity

                node->mSlicePrevOutput = mSlicePrevOutput;
                node->mSlicePrevState = mSlicePrevState;

                node->m_use_errors_from_future_minibatch = m_use_errors_from_future_minibatch;

                node->mDefaultState = mDefaultState;
            }
        }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 4)
                throw std::invalid_argument("LSTM operation only takes five inputs.");

            size_t nT = Inputs(0)->FunctionValues().GetNumCols();
            size_t inputDim = Inputs(0)->FunctionValues().GetNumRows();
            size_t outputDim = Inputs(1)->FunctionValues().GetNumRows();

            if (mGradientComputed == false)
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
                    Matrix<ElemType> sliceState = mState.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);

                    Matrix<ElemType> sliceGi = mGi.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceGf = mGf.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceGo = mGo.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);

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

                    PrepareHistory(timeIdxInSeq, mSlicePrevOutput, mSlicePrevState, FunctionValues(), mState, mPastOutput, mPastState, m_samplesInRecurrentStep, mDefaultState, m_sentenceSeg);

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
                        RuntimeError("Error in computing gradient in function ComputeInputPartial for LSTMnode at position %ld, length %ld", timeIdxInSeq, nT);
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
                mGradientComputed = true;
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
        get the segmentation information, SENTENECE_BEGIN, SENTENCE_MIDDLE, NO_OBSERVATION 
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
            Matrix<ElemType> thisCol = m_sentenceSeg.ColumnSlice(utt_t, 1);
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
                        mLastState.ColumnSlice(i, 1).SetValue(mState.ColumnSlice(t, 1));
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
                mState.Resize(outputDim, nT);
                mState.SetValue(NAN);  /// set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                mGi.Resize(outputDim, nT);
                mGi.SetValue(NAN);  /// set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                mGf.Resize(outputDim, nT);
                mGf.SetValue(NAN);  /// set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                mGo.Resize(outputDim, nT);
                mGo.SetValue(NAN);  /// set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                tanhState.Resize(outputDim, nT);
                tanhState.SetValue(NAN);  /// set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                tanhObs.Resize(outputDim, nT);
                tanhObs.SetValue(NAN);  /// set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 

                if (mPastState.IsEmpty() || mPastState.GetNumCols() != m_samplesInRecurrentStep)
                {
                    mPastState.Resize(outputDim, m_samplesInRecurrentStep);
                    mPastState.SetValue(mDefaultState);
                }
                if (mPastOutput.IsEmpty() || mPastOutput.GetNumCols() != m_samplesInRecurrentStep)
                {
                    mPastOutput.Resize(outputDim, m_samplesInRecurrentStep);
                }

#ifdef DEBUG_DECODER
                if (mPastOutput.IsEmpty() == false)
                    fprintf(stderr, "LSTM node %ls past output norm = %.8e\n", this->NodeName().c_str(), mPastOutput.FrobeniusNorm());
                if (mPastState.IsEmpty() == false)
                    fprintf(stderr, "LSTM node %ls past state norm = %.8e\n", this->NodeName().c_str(), mPastState.FrobeniusNorm());
#endif

                for (size_t timeIdxInSeq = 0; timeIdxInSeq < nT; timeIdxInSeq += m_samplesInRecurrentStep)
                {

                    Matrix<ElemType> sliceObs = Inputs(0)->FunctionValues().ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceOutput = FunctionValues().ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceState = mState.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);

                    Matrix<ElemType> sliceGi = mGi.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceGf = mGf.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceGo = mGo.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);

                    Matrix<ElemType> sliceTanhState = tanhState.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);
                    Matrix<ElemType> sliceTanhInput =
                        tanhObs.ColumnSlice(timeIdxInSeq, m_samplesInRecurrentStep);

                    PrepareHistory(timeIdxInSeq, mSlicePrevOutput, mSlicePrevState, FunctionValues(), mState, mPastOutput, mPastState, m_samplesInRecurrentStep, mDefaultState, m_sentenceSeg);

                    try{
                        EvaluateThisNodeS(Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), Inputs(3)->FunctionValues(), Inputs(4)->FunctionValues(),
                            sliceObs, mSlicePrevOutput, mSlicePrevState, sliceOutput, sliceState, sliceGi, sliceGf, sliceGo, sliceTanhState, sliceTanhInput, m_tempMatrix);
                    }
                    catch (...)
                    {
                        RuntimeError("Error in evaluating LSTMnode at position %ld out of %ld", timeIdxInSeq, nT);
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

                /// set output to 0 if there are no observations
                ResetForNoObservation(FunctionValues());
                ResetForNoObservation(mState);

#ifdef DEBUG_DECODER
                ElemType tmpnorm = FunctionValues().FrobeniusNorm();
                if (ISCLOSE(tmpnorm, 0.834251, 0.002))
                    fprintf(stderr, "check!");
                fprintf(stderr, "LSTM function norm = %.8e\n", tmpnorm);
                for (size_t i = 0; i < 5; i++)
                    fprintf(stderr, "LSTM input[%d] norm = %.8e ", i, Inputs(i)->FunctionValues().FrobeniusNorm());
                fprintf(stderr, "\n");
#endif

                mGradientComputed = false;
            }
            catch (...)
            {
                RuntimeError("Error in evaluation of LSTMNode with %ld observations", nT);
            }
        }

        /**
        Prepare history for LSTMnode

        This function returns state and output from the previous time instance. For recurrent network, the initial state needs to be set in the case of sentence begining, which is carried over from sentenceBegin. In case of sentence begining, the state activity is set to an initial value. The sentenceBegin has element of SENTENCE_BEGIN, SENTENCE_MIDDLE and NO_OBSERVATION, which are 0, 1, and -1, respectively. 
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
            size_t nsamples, const ElemType & initStateValue, const Matrix<ElemType>& sentenceBegin)
        {
            size_t nRow = pastOutput.GetNumRows();
            size_t nStream = sentenceBegin.GetNumRows();

            assert(nStream == nsamples);

            int utt_t = (int)floor(timeIdxInSeq / nsamples);
            if (slicePrevOutput.IsEmpty() || slicePrevOutput.GetNumRows() != nRow || slicePrevOutput.GetNumCols() != nsamples)
                slicePrevOutput.Resize(nRow, nsamples);
            if (slicePrevState.IsEmpty() || slicePrevState.GetNumRows() != nRow || slicePrevState.GetNumCols() != nsamples)
                slicePrevState.Resize(nRow, nsamples);

            if (sentenceBegin.GetNumRows() != nsamples)
                LogicError("Number of rows should be the same as the number of data streams");

            Matrix<ElemType> colBegin(sentenceBegin.GetDeviceId());
            colBegin.SetValue(sentenceBegin.ColumnSlice(utt_t, 1));
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

            SetToInitStateValueForResetSeg(sentenceBegin.ColumnSlice(utt_t, 1), nStream, initStateValue, newPrevState);

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
            size_t nsamples, const Matrix<ElemType>& sentenceBegin)
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
                        || (utt_t < total_utt_t - 1 && GetSegInfo(timeIdxInSeq, utt_id) == SENTENCE_MIDDLE && GetSegInfo(timeIdxInSeq + nsamples, utt_id) == NO_OBSERVATION) /// future observation is no observation
                        )
                    {
                        error.ColumnSlice(utt_id, 1) += obs_error_from_future_minibatch.ColumnSlice(utt_id, 1);
                        stateError.ColumnSlice(utt_id, 1) += state_error_from_future_minibatch.ColumnSlice(utt_id, 1);
                    }
                }
            }


            Matrix<ElemType> colBegin(sentenceBegin.GetDeviceId());
            colBegin.SetValue(sentenceBegin.ColumnSlice(utt_t, 1));
            colBegin.InplaceTruncateBottom(NO_OBSERVATION);
            colBegin.InplaceTruncateTop(SENTENCE_BEGIN);
            colBegin += fabs((ElemType)NO_OBSERVATION); /// raise this so that -1 -> 0 and therefore 
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
            size_t nsamples, const Matrix<ElemType>& sentenceBegin)
        {
            int utt_t = (int)floor(timeIdxInSeq / nsamples);
            Matrix<ElemType> colBegin(sentenceBegin.GetDeviceId());
            colBegin.SetValue(sentenceBegin.ColumnSlice(utt_t, 1));
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
                ElemType initStateValue = mDefaultState;
                Matrix<ElemType> boundary(m_deviceId);
                boundary.Resize(1, nT);
                boundary.SetValue(SENTENCE_MIDDLE);
                boundary.ColumnSlice(0, 1).SetValue(SENTENCE_BEGIN);
                ResetBound(boundary);

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

                mDefaultState = 0.0;
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
                mDefaultState = initStateValue;
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

                if (mState.GetDeviceId() != deviceId)
                    mState.TransferFromDeviceToDevice(mState.GetDeviceId(), deviceId);
                if (mPastState.GetDeviceId() != deviceId)
                    mPastState.TransferFromDeviceToDevice(mPastState.GetDeviceId(), deviceId);
                if (mPastOutput.GetDeviceId() != deviceId)
                    mPastOutput.TransferFromDeviceToDevice(mPastOutput.GetDeviceId(), deviceId);
                if (mGi.GetDeviceId() != deviceId)
                    mGi.TransferFromDeviceToDevice(mGi.GetDeviceId(), deviceId);
                if (mGf.GetDeviceId() != deviceId)
                    mGf.TransferFromDeviceToDevice(mGf.GetDeviceId(), deviceId);
                if (mGo.GetDeviceId() != deviceId)
                    mGo.TransferFromDeviceToDevice(mGo.GetDeviceId(), deviceId);

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

            WCHAR str[4096];
            wsprintf(str, L"Input[Width:%lu]  \n", m_inputDim);
            fstream << wstring(str);
            wsprintf(str, L"Hidden[Width:%lu]    Output[Width:%lu]  \n", m_outputDim, m_outputDim);
            fstream << wstring(str);
        }


    public:

        bool GetHistory(Matrix<ElemType>& hist, bool bLastTime)
        {
            size_t tRow = mPastOutput.GetNumRows();
            size_t tCol = mPastOutput.GetNumCols();
            size_t rCol = mPastState.GetNumCols();

            DEVICEID_TYPE device = hist.GetDeviceId();
            hist.TransferFromDeviceToDevice(device, m_deviceId, true);
            hist.Resize(tRow, tCol + rCol);

            if (bLastTime)
            {
                hist.ColumnSlice(0, tCol).SetValue(mLastOutput);
                hist.ColumnSlice(tCol, rCol).SetValue(mLastState);
            }
            else{
                hist.ColumnSlice(0, tCol).SetValue(mPastOutput);
                hist.ColumnSlice(tCol, rCol).SetValue(mPastState);
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

            mPastOutput.Resize(tRow, eCols);
            mPastState.Resize(tRow, eCols);
            mPastOutput.SetValue(hist.ColumnSlice(0, eCols));
            mPastState.SetValue(hist.ColumnSlice(eCols, eCols));

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
        size_t m_inputDim;
        size_t m_outputDim;

        Matrix<ElemType> mState;  /// hidden state activity
        Matrix<ElemType> mPastState; /// state activity in the previous minibatch
        Matrix<ElemType> mPastOutput; /// output in the previou minibatch 

        Matrix<ElemType> mLastState; /// last state activity 
        Matrix<ElemType> mLastOutput; /// last output 

        Matrix<ElemType> mGi;     /// input gate activity
        Matrix<ElemType> mGf;     /// forget gate activity
        Matrix<ElemType> mGo;     /// output gate activity

        Matrix<ElemType> grdToObs, grdToInputGate, grdToForgetGate, grdToOutputGate, grdToCellWgt;
        Matrix<ElemType> tanhState, tanhObs;

        Matrix<ElemType> m_tempMatrix; /// temp matrix for speed-up

        bool     mGradientComputed; /// true if LSTM node has computed gradients, set to false if forward computation is just finished 

        Matrix<ElemType> mSlicePrevOutput, mSlicePrevState;

        Matrix<ElemType> grdBeforeInputGate, grdBeforeForget, grdBeforeGo, grdToCell, grdBeforeTanhInputGate;

    public:
        /// errors from future minibatch
        Matrix<ElemType> m_obs_error_from_future_minibatch;
        Matrix<ElemType> m_state_error_from_future_minibatch;
        bool m_use_errors_from_future_minibatch;

        ElemType mDefaultState;

    };

    template class LSTMNode<float>;
    template class LSTMNode<double>;

}}}
