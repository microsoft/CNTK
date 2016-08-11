//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <unordered_map>
#ifndef CPUONLY
    #include <cuda_runtime.h>
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

class CUDATimer
{
public:
    CUDATimer(){}
    ~CUDATimer(){}
	void tick(std::string name);
	void tick();
	float tock(std::string name);
	float tock();
	float tockp();
	float tockp(std::string name);
private:
#ifndef CPUONLY
	std::unordered_map<std::string,cudaEvent_t*> m_dictTickTock;
	cudaEvent_t* create_tick();
	float tock(cudaEvent_t* startstop);
#endif
	std::unordered_map<std::string,float> m_dictTickTockCumulative;
};

}}}
