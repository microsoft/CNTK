//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

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

// MatrixPool -- class to support memory sharing
// Despite the gather general name of this class, it is specifically designed to support the memory sharing of ComputationNodes.
// Note: see #define SUPRESS_MEMSHARING below as for how to temporarily disable memory sharing altogether, for debugging
class MatrixPool
{
    vector<shared_ptr<Matrix<float>>>  m_releasedFloatMatrices;
    vector<shared_ptr<Matrix<double>>> m_releasedDoubleMatrices;

    template <class ElemType>
    vector<shared_ptr<Matrix<ElemType>>>& GetReleasedMatrices();

public:
    // release here means the matrix can be put back and shared by others
    template <class ElemType>
    void Release(shared_ptr<Matrix<ElemType>> freeMatrix)
    {
        if (freeMatrix == nullptr || freeMatrix->GetMatrixType() == SPARSE)
            LogicError("MatrixPool::Release: freeMatrix should not be null or sparse.");
//#define SUPRESS_MEMSHARING // #define this to disable memory sharing through this structure
        // TODO: Make this a runtime option.
#ifndef SUPRESS_MEMSHARING
        vector<shared_ptr<Matrix<ElemType>>>& releasedMatrices = GetReleasedMatrices<ElemType>();
#ifdef _DEBUG
        for (int i = 0; i < releasedMatrices.size(); i++)
        {
            if (releasedMatrices[i] == freeMatrix)
                RuntimeError("MatrixPool::Release: freeMatrix is already in the released pool.");
        }

#endif
        releasedMatrices.push_back(freeMatrix);
#endif
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

}}}
