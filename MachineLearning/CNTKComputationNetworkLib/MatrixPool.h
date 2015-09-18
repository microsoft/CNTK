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
        void GetReleasedMatrices(vector<shared_ptr<Matrix<float>>>  * releasedMatrices) { releasedMatrices = &m_releasedFloatMatrices; }
        void GetReleasedMatrices(vector<shared_ptr<Matrix<double>>> * releasedMatrices) { releasedMatrices = &m_releasedDoubleMatrices; }
    public:
        template<class ElemType>
        void Release(const shared_ptr<Matrix<ElemType>> & freeMatrix)
        {
            if (!freeMatrix)
                LogicError("MatrixPool::Release: freeMatrix should not be null.");
            vector<shared_ptr<Matrix<float>>> * releasedMatrices;
            GetReleasedMatrices(releasedMatrices);
            releasedMatrices->push_back(freeMatrix);
        }

        template<class ElemType>
        shared_ptr<Matrix<ElemType>> Request(DEVICEID_TYPE deviceId)
        {
            vector<shared_ptr<Matrix<float>>> * releasedMatrices;
            GetReleasedMatrices(releasedMatrices);
            shared_ptr<Matrix<ElemType>> matrixPtr;
            if (releasedMatrices->empty())
            {
                matrixPtr = make_shared<Matrix<ElemType>>(deviceId);
            }
            else
            {
                matrixPtr = releasedMatrices->back();
                releasedMatrices->pop_back();
            }

            if (!matrixPtr)     // this can't really happen
                LogicError("MatrixPool::Request: failed to get a valid matrix.");

            return matrixPtr;
        }
    };

}}}
