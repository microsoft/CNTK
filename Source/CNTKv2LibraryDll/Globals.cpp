//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"

// TODO: Currently there are some known issues with memory sharing for forward pass output matrices that 
// need to be addressed before we can switch to using memory sharing by default here.
bool g_shareNodeValueMatrices = false;
