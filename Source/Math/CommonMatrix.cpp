

#include "stdafx.h"
#include "CommonMatrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {
	BufferManager* BufferManager::m_instance = new BufferManager();

	template <>
	unordered_map<DEVICEID_TYPE, unordered_map<size_t, vector<float*>>>& BufferManager::BufferContainor<float>() { return m_bufferFloatContainor; }
	template <>
	unordered_map<DEVICEID_TYPE, unordered_map<size_t, vector<double*>>>& BufferManager::BufferContainor<double>() { return m_bufferDoubleContainor; }
	template <>
	unordered_map<DEVICEID_TYPE, unordered_map<size_t, vector<char*>>>& BufferManager::BufferContainor<char>() { return m_bufferCharContainor; }
}
}
}