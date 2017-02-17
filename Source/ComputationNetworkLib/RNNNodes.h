//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "Matrix.h"
#include "TensorView.h"
#include "RNNCommon.h"

#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <algorithm>
#include <utility>
#include <assert.h>

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// OptimizedRNNStack (weights, data)
// -----------------------------------------------------------------------

template <class ElemType>
class OptimizedRNNStackNode : public ComputationNode<ElemType>, public NumInputs<2>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"OptimizedRNNStack"; }
    using Base::OperationName;

public:
    OptimizedRNNStackNode(DEVICEID_TYPE deviceId, const wstring& name);
    OptimizedRNNStackNode(const ScriptableObjects::IConfigRecordPtr configp);
    OptimizedRNNStackNode(DEVICEID_TYPE deviceId, const std::wstring& name, bool bidirectional, size_t numLayers, size_t hiddenSize, const std::wstring& recurrentOp);

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override;
    virtual void Save(File& fstream) const;
    virtual void Load(File& fstream, size_t modelVersion) override;

public:
    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override;
    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override;
    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override;

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_transposedInput, matrixPool);
        RequestMatrixFromPool(m_transposedOutput, matrixPool);
        RequestMatrixFromPool(m_reserve, matrixPool);
        RequestMatrixFromPool(m_workspace, matrixPool);
        RequestMatrixFromPool(m_packingIndex, matrixPool);
    }

    // request matrices needed to do node derivative value evaluation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_transposedDInput, matrixPool);
        RequestMatrixFromPool(m_transposedDOutput, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_transposedInput, matrixPool);
        ReleaseMatrixToPool(m_transposedOutput, matrixPool);
        ReleaseMatrixToPool(m_transposedDInput, matrixPool);
        ReleaseMatrixToPool(m_transposedDOutput, matrixPool);
        ReleaseMatrixToPool(m_reserve, matrixPool);
        ReleaseMatrixToPool(m_workspace, matrixPool);
        ReleaseMatrixToPool(m_packingIndex, matrixPool);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const { return 0 == childIndex; }
    RnnAttributes Attributes() const { return m_rnnAttributes; }

protected:
    bool m_BackwardDataCalledYet;
    TensorShape shapeXT;
    TensorShape shapeYT;
    shared_ptr<Matrix<ElemType>> m_transposedInput;
    shared_ptr<Matrix<ElemType>> m_transposedOutput;
    shared_ptr<Matrix<ElemType>> m_transposedDInput;
    shared_ptr<Matrix<ElemType>> m_transposedDOutput;
    shared_ptr<Matrix<ElemType>> m_workspace;
    shared_ptr<Matrix<ElemType>> m_reserve;
    shared_ptr<Matrix<ElemType>> m_packingIndex;

private:
    void TransposeHelper(const MatrixBasePtr matX, const TensorShape &shapeX, MatrixBasePtr matY, TensorShape &shapeY);

    void PackSequencesForCuDNN(const Matrix<ElemType>& src, Matrix<ElemType>& dst, vector<size_t>& numSequencesForFrame);
    void UnpackSequencesFromCuDNN(const Matrix<ElemType>& src, Matrix<ElemType>& dst);

    RnnAttributes m_rnnAttributes;

    bool m_legacySwapInputsPending = false; // to support an internal legacy version
};

}}}
