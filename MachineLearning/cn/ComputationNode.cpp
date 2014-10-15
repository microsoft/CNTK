//
// <copyright file="ComputationNode.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "ComputationNode.h"
#include "SimpleEvaluator.h"
#include "IComputationNetBuilder.h"
#include "SGD.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    typedef Matrix<float> FloatMatrix;
    typedef Matrix<double> DoubleMatrix;

    LONG64 ComputationNode<float>::s_timeStampCounter = 0;
    LONG64 ComputationNode<double>::s_timeStampCounter = 0;

    std::map<size_t, std::map<size_t, FloatMatrix*>> ComputationNode<float>::s_constOnes;
    std::map<size_t, std::map<size_t, DoubleMatrix*>> ComputationNode<double>::s_constOnes;

    template<class ElemType>
    TaskDescriptor<ElemType>* LearnableParameter<ElemType>::GetPTaskDescriptor(TaskType taskType, size_t inputIndex/*=0*/) const
    {
        TaskDescriptor<ElemType>* descriptor = new TaskDescriptor<ElemType>(this, taskType);
        switch(taskType)
        {
        case taskUpdate:
            {
            descriptor->Param(paramTypePointer, "SGD", paramOptionsInput | paramOptionsConstant);
            descriptor->FunctionParam(-1, paramOptionsInput | paramOptionsOutput | paramOptionsMaintainValue | paramOptionsInitOnBOF | paramOptionsSaveOnEOF | paramOptionsInitalValuesOnDestinations);
            descriptor->GradientParam(-1, paramOptionsInput);

            // use dimensions of m_gradientValues, smoothed gradients are always the same size
            ParamData<ElemType>* param = descriptor->MatrixParam(m_gradientValues, "smoothedGradient", paramOptionsInput | paramOptionsOutput | paramOptionsInitOnBOF | paramOptionsMaintainValue);
            ElemType val(0.0);
            param->SetInitialize(val);

            descriptor->Param(sizeof(ElemType)==4?paramTypeSingle:paramTypeDouble, "learnRatePerSample", paramOptionsInput | paramOptionsConstant);
            descriptor->Param(paramTypeLongLong, "actualMBSize", paramOptionsInput);
            descriptor->Param(paramTypeLongLong, "expectedMBSize", paramOptionsInput | paramOptionsConstant);
            descriptor->SetFunction((FARPROC)MSR::CNTK::SGD<ElemType>::UpdateWeightsS);
            break;
            }
        default:
            assert(false);
            throw std::logic_error("Unsupported task requested");
        }
        return descriptor;
    }
    template class LearnableParameter<float>;
    template class LearnableParameter<double>;
}}}