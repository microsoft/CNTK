//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "BinaryConvolveOp.h"

using namespace CNTK;

extern "C" 
#ifdef _WIN32
__declspec (dllexport)
#endif
// define the call in to the binary convolve function, operands are the kernels and the inputs, attributes is a dictionary of parameters
Function* CreateBinaryConvolveFunction(const Variable* operands, size_t /*numOperands*/, const Dictionary* attributes, const wchar_t* name)
{
    return new BinaryConvolveFunction(operands[0], operands[1], *attributes, name);
}
