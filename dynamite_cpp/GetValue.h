//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// code to dynamically rebatch a dynamic graph

#include "CNTKLibrary.h"

// this will eventually become Variable::Value()
CNTK::NDArrayViewPtr GetValue(const CNTK::Variable&);
