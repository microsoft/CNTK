// <copyright file="MatrixPool.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>

#pragma once

#include <string>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <stdlib.h>

#include "Basics.h"
#include "Matrix.h"
#include "ComputationNode.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class MatrixPool
{
    vector<shared_ptr<Matrix<float>>> m_releasedFloatMatrices;
    vector<shared_ptr<Matrix<double>>> m_releasedDoubleMatrices;

    template <class ElemType>
    vector<shared_ptr<Matrix<ElemType>>>& GetReleasedMatrices();

public:
    //release here means the matrix can be put back and shared by others
    template <class ElemType>
    void Release(shared_ptr<Matrix<ElemType>> freeMatrix)
    {
        vector<shared_ptr<Matrix<ElemType>>>& releasedMatrices = GetReleasedMatrices<ElemType>();
        if (freeMatrix == nullptr || freeMatrix->GetMatrixType() == SPARSE)
            RuntimeError("MatrixPool::Release: freeMatrix should not be null or sparse.");
#ifdef _DEBUG
        for (int i = 0; i < releasedMatrices.size(); i++)
        {
            if (releasedMatrices[i] == freeMatrix)
                RuntimeError("MatrixPool::Release: freeMatrix is already in the released pool.");
        }

#endif
        releasedMatrices.push_back(freeMatrix);
    }

    template <class ElemType>
    shared_ptr<Matrix<ElemType>> Request(DEVICEID_TYPE deviceId)
    {
        vector<shared_ptr<Matrix<ElemType>>>& releasedMatrices = GetReleasedMatrices<ElemType>();
        shared_ptr<Matrix<ElemType>> matrixPtr;
        if (releasedMatrices.empty())
        {
            matrixPtr = make_shared<Matrix<ElemType>>(deviceId);
        }
        else
        {
            matrixPtr = releasedMatrices.back();
            releasedMatrices.pop_back();
        }

        if (!matrixPtr) // this can't really happen
            LogicError("MatrixPool::Request: failed to get a valid matrix.");

        return matrixPtr;
    }
};
} } }
