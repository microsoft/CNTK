//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CPUMatrixImpl.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // explicit instantiations, due to CPUMatrix being too big and causing VS2015 cl crash.
    template class MATH_API CPUMatrix<double>;
}}}