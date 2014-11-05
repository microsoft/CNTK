//
// <copyright file="ComputationNetwork.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "ComputationNetwork.h"

namespace Microsoft { namespace MSR { namespace CNTK {
   
    template<class ElemType>
    ComputationNode<ElemType>* ComputationNetwork<ElemType>::CreateNodeFromFile(const std::wstring nodeType, 
            const std::wstring nodeName, File & fstream)
    {
            ComputationNode<ElemType>* newNode = nullptr;

            if (nodeType == LearnableParameter<ElemType>::TypeName())
                newNode = new LearnableParameter<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == InputValue<ElemType>::TypeName())
                newNode = new InputValue<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == SparseLearnableParameter<ElemType>::TypeName())
                newNode = new SparseLearnableParameter<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == SparseInputValue<ElemType>::TypeName())
                newNode = new SparseInputValue<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == ConvolutionNode<ElemType>::TypeName())
                newNode = new ConvolutionNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == MaxPoolingNode<ElemType>::TypeName())
                newNode = new MaxPoolingNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == AveragePoolingNode<ElemType>::TypeName())
                newNode = new AveragePoolingNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == NegateNode<ElemType>::TypeName())
                newNode = new NegateNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == RectifiedLinearNode<ElemType>::TypeName())
                newNode = new RectifiedLinearNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == SigmoidNode<ElemType>::TypeName())
                newNode = new SigmoidNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == TanhNode<ElemType>::TypeName())
                newNode = new TanhNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == LogNode<ElemType>::TypeName())
                newNode = new LogNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == SoftmaxNode<ElemType>::TypeName())
                newNode = new SoftmaxNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == SumNode<ElemType>::TypeName())
                newNode = new SumNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == ScaleNode<ElemType>::TypeName())
                newNode = new ScaleNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == TimesNode<ElemType>::TypeName())
                newNode = new TimesNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == ElementTimesNode<ElemType>::TypeName())
                newNode = new ElementTimesNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == DiagTimesNode<ElemType>::TypeName())
                newNode = new DiagTimesNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == CosDistanceNode<ElemType>::TypeName())
                newNode = new CosDistanceNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == KhatriRaoProductNode<ElemType>::TypeName())
                newNode = new KhatriRaoProductNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == PlusNode<ElemType>::TypeName())
                newNode = new PlusNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == MinusNode<ElemType>::TypeName())
                newNode = new MinusNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == SquareErrorNode<ElemType>::TypeName())
                newNode = new SquareErrorNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == CrossEntropyWithSoftmaxNode<ElemType>::TypeName())
                newNode = new CrossEntropyWithSoftmaxNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == ClassBasedCrossEntropyWithSoftmaxNode<ElemType>::TypeName())
                newNode = new ClassBasedCrossEntropyWithSoftmaxNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == CrossEntropyNode<ElemType>::TypeName())
                newNode = new CrossEntropyNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == MatrixL1RegNode<ElemType>::TypeName())
                newNode = new MatrixL1RegNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == MatrixL2RegNode<ElemType>::TypeName())
                newNode = new MatrixL2RegNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == PerDimMeanVarNormalizationNode<ElemType>::TypeName() || nodeType==L"PerDimMeanVarNormalizationNode") // mseltzer - hack b/c this changed (Dong?) and old models didn't load...
                newNode = new PerDimMeanVarNormalizationNode<ElemType>(fstream, m_deviceId, nodeName);            
            else if (nodeType == PerDimMeanNormalizationNode<ElemType>::TypeName())
                newNode = new PerDimMeanNormalizationNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == ErrorPredictionNode<ElemType>::TypeName())
                newNode = new ErrorPredictionNode<ElemType>(fstream, m_deviceId, nodeName);    
            else if (nodeType == DropoutNode<ElemType>::TypeName())
                newNode = new DropoutNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == MeanNode<ElemType>::TypeName())
                newNode = new MeanNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == InvStdDevNode<ElemType>::TypeName())
                newNode = new InvStdDevNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == DelayNode<ElemType>::TypeName())
                newNode = new DelayNode<ElemType>(fstream, m_deviceId, nodeName);
            else if (nodeType == LookupTableNode<ElemType>::TypeName())
                newNode = new LookupTableNode<ElemType>(fstream, m_deviceId, nodeName);
            else
            {
                fprintf(stderr, "Error creating new ComputationNode of type %ls, with name %ls\n", nodeType.c_str(), nodeName.c_str());
                throw std::invalid_argument("Invalid node type.");
            }

            AddNodeToNet(newNode);
            return newNode;
    }

    template<class ElemType>
    ComputationNode<ElemType>* ComputationNetwork<ElemType>::CreateComputationNode(const std::wstring nodeType, const std::wstring nodeName) 
    {         
            
            ComputationNode<ElemType>* newNode;

            if (nodeType == NegateNode<ElemType>::TypeName())
                newNode = new NegateNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == RectifiedLinearNode<ElemType>::TypeName())
                newNode = new RectifiedLinearNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == SigmoidNode<ElemType>::TypeName())
                newNode = new SigmoidNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == TanhNode<ElemType>::TypeName())
                newNode = new TanhNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == LogNode<ElemType>::TypeName())
                newNode = new LogNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == SoftmaxNode<ElemType>::TypeName())
                newNode = new SoftmaxNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == SumNode<ElemType>::TypeName())
                newNode = new SumNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == ScaleNode<ElemType>::TypeName())
                newNode = new ScaleNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == TimesNode<ElemType>::TypeName())
                newNode = new TimesNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == ElementTimesNode<ElemType>::TypeName())
                newNode = new ElementTimesNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == DiagTimesNode<ElemType>::TypeName())
                newNode = new DiagTimesNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == CosDistanceNode<ElemType>::TypeName())
                newNode = new CosDistanceNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == KhatriRaoProductNode<ElemType>::TypeName())
                newNode = new KhatriRaoProductNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == PlusNode<ElemType>::TypeName())
                newNode = new PlusNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == MinusNode<ElemType>::TypeName())
                newNode = new MinusNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == SquareErrorNode<ElemType>::TypeName())
                newNode = new SquareErrorNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == CrossEntropyWithSoftmaxNode<ElemType>::TypeName())
                newNode = new CrossEntropyWithSoftmaxNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == CrossEntropyNode<ElemType>::TypeName())
                newNode = new CrossEntropyNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == ClassBasedCrossEntropyWithSoftmaxNode<ElemType>::TypeName())
                newNode = new ClassBasedCrossEntropyWithSoftmaxNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == MatrixL1RegNode<ElemType>::TypeName())
                newNode = new MatrixL1RegNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == MatrixL2RegNode<ElemType>::TypeName())
                newNode = new MatrixL2RegNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == PerDimMeanVarNormalizationNode<ElemType>::TypeName())
                newNode = new PerDimMeanVarNormalizationNode<ElemType>(m_deviceId, nodeName);        
            else if (nodeType == PerDimMeanNormalizationNode<ElemType>::TypeName())
                newNode = new PerDimMeanNormalizationNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == ErrorPredictionNode<ElemType>::TypeName())
                newNode = new ErrorPredictionNode<ElemType>(m_deviceId, nodeName);    
            else if (nodeType == DropoutNode<ElemType>::TypeName())
                newNode = new DropoutNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == MeanNode<ElemType>::TypeName())
                newNode = new MeanNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == InvStdDevNode<ElemType>::TypeName())
                newNode = new InvStdDevNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == DelayNode<ElemType>::TypeName())
                newNode = new DelayNode<ElemType>(m_deviceId, nodeName);
            else if (nodeType == LookupTableNode<ElemType>::TypeName())
                newNode = new LookupTableNode<ElemType>(m_deviceId, nodeName);
            else
            {
                fprintf(stderr, "Error creating new ComputationNode of type %ls, with name %ls\n", nodeType.c_str(), nodeName.c_str());
                throw std::invalid_argument("Invalid node type.");
            }

            AddNodeToNet(newNode);
            return newNode;
    }


}}}
