//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// This implements a plain-text deserializer
//

#include "stdafx.h"
#include "DataDeserializer.h"

#include <memory>

#define let const auto

using namespace CNTK;
using namespace std;

namespace CNTK {

shared_ptr<DataDeserializer> CreatePlainTextDeserializer(const ConfigParameters& deserializerConfig, bool primary)
{
    deserializerConfig; primary;
    return nullptr;
}

}; // namespace