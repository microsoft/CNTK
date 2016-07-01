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
    GPUMatrix<float> *m_bufferGPU;
    CPUMatrix<float> *m_bufferCPU;
    

public:
    ~SyncAction(){};
    virtual void executeAction() = 0;

public:    
    bool GetIsAsynchronous()
    {
        return m_isAsynchronous;
    }


};

}}}


