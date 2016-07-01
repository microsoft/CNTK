//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <unordered_map>
#ifndef CPUONLY
    #include <cuda_runtime_api.h>
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

class Timer
{
public:
    Timer(){}
    ~Timer(){}
	void tick(std::string name);
	void tick();
	float tock(std::string name);
	float tock();
private:
	std::unordered_map<std::string,cudaEvent_t*> m_dictTickTock;
	std::unordered_map<std::string,float> m_dictTickTockCumulative;

	cudaEvent_t* create_tick();
	float tock(cudaEvent_t* startstop);
};

}}}
