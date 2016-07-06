//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <CPUMatrix.h>
#include <GPUMatrix.h>

#include <unordered_map>


namespace Microsoft { namespace MSR { namespace CNTK {

class SyncAction
{

protected:
    // this is needed so we can execute async actions first and while they are running we execute sync actions
    bool m_isAsynchronous; 
    Matrix<float> *m_bufferGPU;
    Matrix<float> *m_bufferCPU;
    

public:
    ~SyncAction(){};
    virtual void executeAction() = 0;
    Matrix<float> *GetGPUMatrix(){ return m_bufferGPU; }
    Matrix<float> *GetCPUMatrix(){ return m_bufferCPU; }

public:    
    bool GetIsAsynchronous()
    {
        return m_isAsynchronous;
    }


};

}}}


