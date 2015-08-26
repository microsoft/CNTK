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

namespace Microsoft {
    namespace MSR {
        namespace CNTK {

            template<class ElemType>
            class MatrixPool
            {
            protected:
                typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
                typedef shared_ptr<Matrix<ElemType>> MatrixPtr;

            public:
                void Release(const MatrixPtr& freeMatrix)
                {
                    if (freeMatrix == nullptr)
                        RuntimeError("MatrixPool::Release: freeMatrix should not be null.");

                    m_releasedMatrices.push_back(freeMatrix);
                }

                MatrixPtr Request(DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
                {
                    MatrixPtr matrixPtr = null;
                    if (m_releasedMatrices.empty())
                    {
                        matrixPtr = make_shared<Matrix<ElemType>>(deviceId)
                    }
                    else
                    {
                        matrixPtr = m_releasedMatrices.back();
                        m_releasedMatrices.pop_back();
                    }

                    if (matrixPtr == nullptr)
                        RuntimeError("MatrixPool::Request: failed to get a valid matrix.");

                    return matrixPtr;
                }

            protected:

                vector<MatrixPtr> m_releasedMatrices;
            };
        }
    }
}