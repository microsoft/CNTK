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
    float *m_bufferCPU;
    int m_rows;
    int m_cols;
    size_t m_bytes;
    

public:
    ~SyncAction(){};
    virtual void BeginAction() = 0;
    virtual void EndAction() = 0; // for synchronization and cleanup
    virtual void ReleaseMemory() = 0;
    Matrix<float> *GetGPUMatrix(){ return m_bufferGPU; }
    float *GetCPUMatrix(){ return m_bufferCPU; }
    bool GetIsAsynchronous()
    {
        return m_isAsynchronous;
    }

};

}}}


