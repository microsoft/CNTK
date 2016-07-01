//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Timer.h"
#include <assert.h>



namespace Microsoft { namespace MSR { namespace CNTK {

void Timer::tick(){	tick("default"); }
void Timer::tick(std::string name)
{
	if (m_dictTickTock.count(name) > 0)
	{
		if (m_dictTickTockCumulative.count(name) > 0)
			m_dictTickTockCumulative[name] += tock(m_dictTickTock[name]);
		else
			m_dictTickTockCumulative[name] = tock(m_dictTickTock[name]);
            
        m_dictTickTock.erase(name);
	} 
    else
	{
		m_dictTickTock[name] = create_tick();
	}
}


float Timer::tock(){ return tock("default"); }
float Timer::tock(std::string name)
{
	if (m_dictTickTockCumulative.count(name) > 0)
	{
		float cumulative_value = m_dictTickTockCumulative[name];
		float value = tock(m_dictTickTock[name]);
		m_dictTickTockCumulative.erase(name);
		return value + cumulative_value;
	}
	else
	{
		assert(("No tick event was registered for the name" + name, m_dictTickTock.count(name) > 0));
		float value = tock(m_dictTickTock[name]);
		m_dictTickTock.erase(name);
		return value;
	}
}

cudaEvent_t* Timer::create_tick()
{
    cudaEvent_t* startstop;
    startstop = (cudaEvent_t*)malloc(2*sizeof(cudaEvent_t));
    cudaEventCreate(&startstop[0]);
    cudaEventCreate(&startstop[1]);
    cudaEventRecord(startstop[0], 0);

    return startstop;
}

float Timer::tock(cudaEvent_t* startstop)
{
	float time;
	cudaEventRecord(startstop[1], 0);
	cudaEventSynchronize(startstop[1]);
	cudaEventElapsedTime(&time, startstop[0], startstop[1]);
	return time;
}

}}}
