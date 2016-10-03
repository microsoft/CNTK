//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "DataTransferer.h"
#include "GPUDataTransferer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    DataTransfererPtr CreatePrefetchDataTransferer(int deviceId)
    {
        return std::make_shared<PrefetchGPUDataTransferer>(deviceId);
    }

} } }
