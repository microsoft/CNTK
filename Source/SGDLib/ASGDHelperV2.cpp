//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ASGDHelperV2.cpp : Implements ASGDHelper interface. The implementation is based on Multiverso.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ASGDHelper.h"
#include "MPIWrapper.h"
#include "ComputationNetwork.h"
#include "TimerUtility.h"

#include <functional>
#include <thread>
#include <unordered_map>
#include <numeric>
#include <algorithm>

#ifdef ASGD_PARALLEL_SUPPORT

#include <multiverso/multiverso.h>
#include <multiverso/util/configure.h>
#include <multiverso/table/array_table.h>
#include <multiverso/updater/updater.h>

#pragma comment(lib, "Multiverso.lib")
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

#ifdef ASGD_PARALLEL_SUPPORT
template<class ElemType = float>
class MultiversoHelperV2 : public ASGDHelper<ElemType>
{
public:
MultiversoHelperV2(size_t nodeNumRanks):
m_totalWorkerNumbers(nodeNumRanks)
{
    multiverso::SetCMDFlag("logtostderr", true);
    multiverso::SetCMDFlag<std::string>(std::string("updater_type"), std::string("sgd"));
    multiverso::MV_Init();
}

private:
    multiverso::ArrayServer<ElemType>* m_serverArray;
    multiverso::ArrayWorker<ElemType>* m_workerArray;
    std::vector<multiverso::GetOption*> m_getOptions; // used by sparse table
    std::vector<multiverso::AddOption*> m_addOptions; // used by sparse tabl

    int m_totalWorkerNumbers;
    
    int m_traceLevel;
    int m_syncPerfStats;
    Timer m_reportTimer;
    size_t m_parameterSyncCounter;
    size_t m_sampleSinceLastReport
};
#else
template<class ElemType = float>
class NoneASGDHelperV2 : public ASGDHelper<ElemType>
{

};

#endif

template<class ElemType>
ASGDHelper<ElemType>* NewASGDHelperV2(size_t nodeNumRanks)
{
#ifdef ASGD_PARALLEL_SUPPORT
    return new MultiversoHelperV2<ElemType>(size_t nodeNumRanks);
#else
    return new NoneASGDHelperV2<ElemType>(size_t nodeNumRanks);
#endif
}

template ASGDHelper<float>* NewASGDHelperV2<float>(size_t nodeNumRanks);

template ASGDHelper<double>* NewASGDHelperV2<double>(size_t nodeNumRanks);

}}}