//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CommonMatrix.cpp
//
#include "stdafx.h"
#include "CommonMatrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

std::unordered_map<DEVICEID_TYPE, BufferManager*> BufferManager::m_instances;

template <>
std::multimap<size_t, float*>& BufferManager::BufferContainor<float>() { return m_bufferFloatContainor; }
template <>
std::multimap<size_t, double*>& BufferManager::BufferContainor<double>() { return m_bufferDoubleContainor; }
template <>
std::multimap<size_t, char*>& BufferManager::BufferContainor<char>() { return m_bufferCharContainor; }
template <>
std::multimap<size_t, short*>& BufferManager::BufferContainor<short>() { return m_bufferShortContainor; }

}
}
}
