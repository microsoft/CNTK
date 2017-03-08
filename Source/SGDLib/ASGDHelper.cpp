//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ASGDHelper.cpp : Implements ASGDHelper interface. The implementation is based on Multiverso.
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


#ifndef CPUONLY
#include <cuda_runtime.h>
#pragma comment (lib, "cudart.lib")     // for cudaMemcpyAsync()
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

#ifndef CPUONLY

#include <cuda_runtime.h>

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE>
static void CudaCall(ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode)
{
    if (retCode != successCode)
    {
        try
        {
#ifdef _WIN32
            const char* hostname = getenv("COMPUTERNAME");
#else
            char hostname[HOST_NAME_MAX];
            if (gethostname(hostname, HOST_NAME_MAX) != 0)
                strcpy(hostname, "?");
#endif
            int currentCudaDevice;
            cudaGetDevice(&currentCudaDevice);
            Microsoft::MSR::CNTK::RuntimeError("%s failure %d; GPU=%d ; hostname=%s ; expr=%s", libName, (int)retCode, currentCudaDevice, hostname ? hostname : "?", exprString);
        }
        catch (const std::exception& e) // catch, log, and rethrow since CUDA code sometimes hangs in destruction, so we'd never get to see the error
        {
            std::cerr << e.what() << std::endl;
            throw;
        }
    }
}

#define CUDA_CALL(expr)     (CudaCall((expr), #expr, "CUDA",     cudaSuccess))
#endif // CPUONLY

#ifdef ASGD_PARALLEL_SUPPORT

// MultiversoHelper is the implementation of ASGDHelper interface with Multiverso
template<class ElemType = float>
class MultiversoHelper : public ASGDHelper<ElemType>
{
public:
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;

    MultiversoHelper(const std::list<ComputationNodeBasePtr> & learnableNodes,          // Parameters that needs to be train
        size_t nodeNumRanks,                                                            // Number of working nodes
        bool useAsyncBuffer = true,                                                   // Using asynchonous buffer to hide communication cost
        bool isSimulatedModelAveragingSGD = false,                                      // Using parameter server-based MA rather than ASGD
        AdjustLearningRateAtBeginning adjusttype = AdjustLearningRateAtBeginning::None, // Adjust learning per minibatches at very begining of training process
        // this could be used to tackle the unstableness of ASGD
        double adjustCoef = 0.2,                                                        // see in DecayCoefficient()
        size_t adjustPerMinibatches = 600,                                              //
        int traceLevel = 0,                                                             // log level
        int syncPerfStats = 0) :                                                        // shown perf data every syncPerfStats
        m_parameterSyncCounter(0), m_adjustLearningRateAtBeginningType(adjusttype),
        m_adjustCoefficient(adjustCoef), m_adjustMBNumber(adjustPerMinibatches),
        m_totalClientNumber(nodeNumRanks), m_useAsyncBuffer(useAsyncBuffer),
        m_traceLevel(traceLevel), m_ModelAveragingSGDSimulating(isSimulatedModelAveragingSGD), m_doesEveryNodesShouldSynced(false),
        m_syncPerfStats(syncPerfStats)
    {
        if (m_ModelAveragingSGDSimulating)
        {
            m_doesEveryNodesShouldSynced = true;
            m_useAsyncBuffer = false;
        }
        // Pipeline releated variables
        m_localBufferNum = m_useAsyncBuffer ? 2 : 1;
        m_bufferSwapIndex = new int[m_localBufferNum];

        // CPU asynchronous buffer
        m_cpuAsyncBuffer = new ElemType*[m_localBufferNum];

        // Get option used by multiverso sparse update
        m_getOptions.reserve(m_localBufferNum);
        m_addOptions.reserve(m_localBufferNum);

#ifndef CPUONLY
        // GPU asynchronous buffer
        m_gpuAsyncBuffer.resize(m_localBufferNum);
        // creat an communication stream for the data tranfer between GPU and CPU
        CUDA_CALL(cudaStreamCreate(&_commStream));
#endif
        m_bufferIndexInUse = 0;
        for (int i = 0; i < m_localBufferNum; i++)
            m_bufferSwapIndex[i] = (i + 1) % m_localBufferNum;

        m_aysncBufferThread = nullptr;

        multiverso::SetCMDFlag("logtostderr", true);

        if (m_doesEveryNodesShouldSynced)
            multiverso::SetCMDFlag("sync", true);

        MultiversoInit(learnableNodes);
    }

    ~MultiversoHelper()
    {
        fprintf(stderr, "~MultiversoHelper\n");
        fflush(stderr);

        if (m_useAsyncBuffer && m_aysncBufferThread != nullptr && m_aysncBufferThread->joinable())
            m_aysncBufferThread->join();

        delete m_bufferSwapIndex, m_deltaArray;

        for (size_t i = 0; i < m_localBufferNum; i++)
        {
#ifndef CPUONLY
            CUDA_CALL(cudaFreeHost(m_cpuAsyncBuffer[i]));
#else
            delete m_cpuAsyncBuffer[i];
#endif
        }
        delete m_cpuAsyncBuffer;
#ifndef CPUONLY
        CUDA_CALL(cudaStreamDestroy(_commStream));
#endif
        multiverso::MV_ShutDown(false);
    }

    void InitModel(const std::list<ComputationNodeBasePtr> & learnableNodes) override
    {
        float factor = 1.0f / m_totalClientNumber;

        int i = 0; // indicate the index of learnable nodes
        for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
        {
            ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
            Matrix<ElemType> &mat = node->Value();

#ifndef CPUONLY
            for (int j = 0; j < m_localBufferNum; j++)
                m_gpuAsyncBuffer[j].push_back(mat.DeepClone());
#endif
            ElemType* px = m_cpuAsyncBuffer[0] + m_tableOffsets[i];
            mat.CopyToArray(px, m_tableLength[i]);
        }

        for (int i2 = 1; i2 < m_localBufferNum; i2++)
            memcpy(m_cpuAsyncBuffer[i2], m_cpuAsyncBuffer[0], sizeof(ElemType) * m_totalModelSize);

        memcpy(m_deltaArray, m_cpuAsyncBuffer[0], sizeof(ElemType) * m_totalModelSize);

        // because the parameter server will minus the delta on the server, so that we should send the minus initial model to the server.
        std::transform(m_deltaArray, m_deltaArray + m_totalModelSize, m_deltaArray, std::bind1st(std::multiplies<ElemType>(), -factor));

        m_workerArray->Add(m_deltaArray, m_totalModelSize);
        m_workerArray->Get(m_deltaArray, m_totalModelSize);
        WaitAll();
        m_workerArray->Get(m_deltaArray, m_totalModelSize);

        if (std::equal(m_deltaArray, m_deltaArray + m_totalModelSize, m_cpuAsyncBuffer[0]))
            fprintf(stderr, "multiverso initial model loaded.\n");
        m_reportTimer.Start();
    }

    bool PushAndPullModel(const std::list<ComputationNodeBasePtr> & learnableNodes, size_t sampleSinceLastSynced) override
    {
        m_parameterSyncCounter++;

        double fromCPUToGPUTime;
        double fromGPUToCPUTime;
        double networkTime;
        double swapTimeOnGPU;
        m_reportTimer.Restart();
        WaitAsyncBuffer();
        m_reportTimer.Stop();

        // reset statics for profiling
        if (m_traceLevel > 2 && m_syncPerfStats > 0 && m_parameterSyncCounter % m_syncPerfStats == 0)
        {
            fromCPUToGPUTime = 0;
            fromGPUToCPUTime = 0;
            networkTime = 0;
            swapTimeOnGPU = 0;
        }

        m_bufferIndexInUse = m_bufferSwapIndex[m_bufferIndexInUse];

        int i = 0; // indicate the index of learnable nodes
        if (m_useAsyncBuffer)
        {
            m_reportTimer.Restart();
            for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
            {
                ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
                Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->Value();
#ifndef CPUONLY
                // CNTK model -> GPU buffer
                CUDA_CALL(cudaMemcpy(m_gpuAsyncBuffer[m_bufferIndexInUse][i].Data(),
                    mat.Data(),
                    mat.GetNumElements() * sizeof(ElemType),
                    cudaMemcpyDeviceToDevice));

                // GPU buffer -> CNTK model
                CUDA_CALL(cudaMemcpy(mat.Data(),
                    m_gpuAsyncBuffer[m_bufferSwapIndex[m_bufferIndexInUse]][i].Data(),
                    mat.GetNumElements() * sizeof(ElemType),
                    cudaMemcpyDeviceToDevice));
#else
                ElemType * px = m_cpuAsyncBuffer[m_bufferIndexInUse] + m_tableOffsets[i];
                mat.CopyToArray(px, m_tableLength[i]);
                ElemType * py = m_cpuAsyncBuffer[m_bufferSwapIndex[m_bufferIndexInUse]] + m_tableOffsets[i];
                mat.SetValue(mat.GetNumRows(), mat.GetNumCols(), mat.GetDeviceId(), py);
                delete px;
#endif
            }
            m_reportTimer.Stop();
            if (m_traceLevel > 2)
            {
                swapTimeOnGPU = m_reportTimer.ElapsedSeconds();
            }
#ifndef CPUONLY
            m_aysncBufferThread = new thread([&]()
            {
                float factor = DecayCoefficient();
                int deviceId = m_gpuAsyncBuffer[m_bufferIndexInUse][0].GetDeviceId();

                CUDA_CALL(cudaSetDevice(deviceId));

                Timer threadTimer;
                threadTimer.Restart();
                for (int widx = 0; widx < m_tableCount; widx++)
                {
                    ElemType * px = m_deltaArray + m_tableOffsets[widx];
                    // GPU buffer -> CPU buffer
                    CUDA_CALL(cudaMemcpyAsync(px,
                        m_gpuAsyncBuffer[m_bufferIndexInUse][widx].Data(),
                        m_gpuAsyncBuffer[m_bufferIndexInUse][widx].GetNumElements() * sizeof(ElemType),
                        cudaMemcpyDeviceToHost,
                        _commStream));
                }
                // waiting copy from GPU to CPU has finished
                CUDA_CALL(cudaStreamSynchronize(_commStream));
                threadTimer.Stop();

                if (m_traceLevel > 3)
                {
                    double time = threadTimer.ElapsedSeconds();
                    fprintf(stderr, "\t\t -- pullAndRequest, GPU -> CPU time %lf \n", time);
                }

                // delta =  gradient * learning_rate
                std::transform(m_cpuAsyncBuffer[m_bufferIndexInUse], 
                    m_cpuAsyncBuffer[m_bufferIndexInUse] + m_totalModelSize,
                    m_deltaArray, m_deltaArray, 
                    std::minus<ElemType>());

                threadTimer.Restart();
                // lr decay
                std::transform(m_deltaArray, 
                    m_deltaArray + m_totalModelSize, 
                    m_deltaArray, 
                    std::bind1st(std::multiplies<ElemType>(), factor));


                ElemType* px = m_deltaArray;
                ElemType* py = m_cpuAsyncBuffer[m_bufferIndexInUse];
                m_workerArray->AddAsync(px, m_totalModelSize);
                m_workerArray->Get(py, m_totalModelSize);

                threadTimer.Stop();
                if (m_traceLevel > 3)
                {
                    double time = threadTimer.ElapsedSeconds();
                    fprintf(stderr, "\t\t -- pullAndRequest, Worker <--> Multiverso time %lf \n", time);
                }

                threadTimer.Restart();
                // copy parameters from CPU buffer to GPU buffer
                for (int widx = 0; widx < m_tableCount; widx++)
                {
                    ElemType * py2 = m_cpuAsyncBuffer[m_bufferIndexInUse] + m_tableOffsets[widx];

                    CUDA_CALL(cudaMemcpyAsync(m_gpuAsyncBuffer[m_bufferIndexInUse][widx].Data(),
                        py2,
                        m_gpuAsyncBuffer[m_bufferIndexInUse][widx].GetNumElements() * sizeof(ElemType),
                        cudaMemcpyHostToDevice,
                        _commStream));
                }
                CUDA_CALL(cudaStreamSynchronize(_commStream));
                threadTimer.Stop();
                if (m_traceLevel > 3)
                {
                    double time = threadTimer.ElapsedSeconds();
                    fprintf(stderr, "\t\t -- pullAndRequest, CPU -> GPU time %lf \n", time);
                }
            });
#else
            m_aysncBufferThread = new thread([&]()
            {
                float factor = DecayCoefficient();
                int t_cacheIdx = m_bufferIndexInUse;

                std::transform(m_cpuAsyncBuffer[t_cacheIdx], m_cpuAsyncBuffer[t_cacheIdx] + m_totalModelSize, m_deltaArray, m_deltaArray, std::minus<ElemType>());
                std::transform(m_deltaArray, m_deltaArray + m_totalModelSize, m_deltaArray, std::bind1st(std::multiplies<ElemType>(), factor));

                ElemType* px = m_deltaArray;
                ElemType* py = m_cpuAsyncBuffer[t_cacheIdx];
                m_workerArray->AddAsync(px, m_totalModelSize);
                m_workerArray->Get(py, m_totalModelSize);

            });
#endif
        }
        else
        {
            m_reportTimer.Restart();
            float factor = DecayCoefficient();
            i = 0;
            for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
            {
                ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
                Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->Value();

                ElemType * px = m_deltaArray + m_tableOffsets[i];
                mat.CopyToArray(px, m_tableLength[i]);
            }

            m_reportTimer.Stop();
            if (m_traceLevel > 3)
            {
                double time = m_reportTimer.ElapsedSeconds();
                fprintf(stderr, "\t\t -- pullAndRequest, GPU -> CPU time %lf \n", time);
            }
            std::transform(m_cpuAsyncBuffer[0], m_cpuAsyncBuffer[0] + m_totalModelSize, m_deltaArray, m_deltaArray, std::minus<ElemType>());

            // lr decay
            if (m_ModelAveragingSGDSimulating)
            {
                factor = ModelAggregationCoefficient(sampleSinceLastSynced);
                std::transform(m_deltaArray, m_deltaArray + m_totalModelSize, m_deltaArray, std::bind1st(std::multiplies<ElemType>(), factor));
                if (m_traceLevel > 2 && m_syncPerfStats != 0)
                {
                    if (m_parameterSyncCounter % m_syncPerfStats == 0)
                        ReportPerfStats(m_totalClientNumber * m_sampleSinceLastReport, m_sampleSinceLastReport);
                    else
                        m_sampleSinceLastReport += sampleSinceLastSynced;
                }

            }
            else
            {
                std::transform(m_deltaArray, m_deltaArray + m_totalModelSize, m_deltaArray, std::bind1st(std::multiplies<ElemType>(), factor));
            }
            m_reportTimer.Restart();

            ElemType* px = m_deltaArray;
            ElemType* py = m_cpuAsyncBuffer[0];
            m_workerArray->AddAsync(px, m_totalModelSize);
            m_workerArray->Get(py, m_totalModelSize);

            m_reportTimer.Stop();
            if (m_traceLevel > 3)
            {
                double time = m_reportTimer.ElapsedSeconds();
                fprintf(stderr, "\t\t -- pullAndRequest, Worker <--> Multiverso time %lf \n", time);
            }
            m_reportTimer.Restart();
            i = 0;
            for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
            {
                ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
                Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->Value();

                ElemType * px2 = m_cpuAsyncBuffer[0] + m_tableOffsets[i];
                mat.SetValue(mat.GetNumRows(), mat.GetNumCols(), mat.GetDeviceId(), px2);
            }
            m_reportTimer.Stop();
            if (m_traceLevel > 3)
            {
                double time = m_reportTimer.ElapsedSeconds();
                fprintf(stderr, "\t\t -- pullAndRequest, CPU -> GPU time %lf \n", time);
            }
        }
        return true;
    }

    void WaitAll() override
    {
        multiverso::MV_Barrier();
    }

    void WaitAsyncBuffer() override
    {
        if (m_aysncBufferThread != nullptr && m_aysncBufferThread->joinable())
        {
            m_aysncBufferThread->join();
            delete m_aysncBufferThread;
            m_aysncBufferThread = nullptr;
        }
    }

private:
    void MultiversoInit(const std::list<ComputationNodeBasePtr> & learnableNodes)
    {
        // parameter server offer vary of updaters, we only use the SGD updater for this simple case.
        multiverso::SetCMDFlag<std::string>(std::string("updater_type"), std::string("sgd"));
        multiverso::MV_Init();

        int i = 0;
        for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
        {
            ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
            Matrix<ElemType> &mat = node->Value();
            size_t layerSize = mat.GetNumElements();

            m_tableLength.push_back(layerSize);
        }

        m_tableCount = m_tableLength.size();

        // cacluate total of learnable node's size
        m_totalModelSize = accumulate(m_tableLength.begin(), m_tableLength.end(), 0);

        m_serverArray = new multiverso::ArrayServer<ElemType>(m_totalModelSize);
        m_workerArray = new multiverso::ArrayWorker<ElemType>(m_totalModelSize);

        multiverso::MV_Barrier();

        size_t idx = 0;
        for (size_t len : m_tableLength)
        {
            m_tableOffsets.push_back(idx);
            idx += len;
        }

#ifndef CPUONLY
        for (int i2 = 0; i2 < m_localBufferNum; i2++)
            m_gpuAsyncBuffer[i2].reserve(m_tableCount);

        // create pinned memory
        for (int i3 = 0; i3 < m_localBufferNum; ++i3)
            CUDA_CALL(cudaMallocHost((void **)&m_cpuAsyncBuffer[i3], sizeof(ElemType) * (m_totalModelSize), cudaHostAllocPortable));

        CUDA_CALL(cudaMallocHost((void **)&m_deltaArray, sizeof(ElemType) * (m_totalModelSize), cudaHostAllocPortable));
#else
        for (int i4 = 0; i4 < m_localBufferNum; i4++)
            m_cpuAsyncBuffer[i4] = new ElemType[m_totalModelSize];
        m_deltaArray = new ElemType[m_totalModelSize];
#endif
    }

    float DecayCoefficient()
    {
        float f = 1.f;
        switch (m_adjustLearningRateAtBeginningType)
        {
        case AdjustLearningRateAtBeginning::None:
            break;
        case AdjustLearningRateAtBeginning::Linearly:
            f = min(f, max(0.f, (float)(m_adjustCoefficient + (1 - m_adjustCoefficient) / m_adjustMBNumber * m_parameterSyncCounter)));
            break;
        case AdjustLearningRateAtBeginning::Staircase:
            f = min(f, max(0.f, (float)(m_adjustCoefficient * (m_parameterSyncCounter / m_adjustMBNumber + 1))));
            break;
        default:
            break;
        }
        return f;
    }

    float ModelAggregationCoefficient(size_t samplesSinceLastSync)
    {
        float factor = 0;
        int   nTotalSamples = samplesSinceLastSync;
        // m_pMPI->AllReduce(&nTotalSamples, 1);

        if (nTotalSamples <= 0)
        {
            factor = 1.0f / m_pMPI->NumNodesInUse();
            // give an estimated one 
        }
        else
        {
            factor = (samplesSinceLastSync + 0.0f) / nTotalSamples;
        }
        factor = 1.0f / m_pMPI->NumNodesInUse();
        return factor;
    }

    inline void transpose(ElemType *src, ElemType *dst, const int N, const int M)
    {
        for (auto n = 0; n < N*M; n++) {
            auto i = n / N;
            auto j = n%N;
            dst[n] = src[M*j + i];
        }
    }

    void ReportPerfStats(size_t totalSamplesProcessedSinceLastReport,
        size_t localSamplesProcessedSinceLastReport)
    {
        m_reportTimer.Stop();
        double secondsSinceLastReport = m_reportTimer.ElapsedSeconds();
        m_reportTimer.Restart();

        float totalThroughput = secondsSinceLastReport > 0 ? (float)totalSamplesProcessedSinceLastReport / ((float)secondsSinceLastReport * 1000.0f) : 0.0f;
        float throughputPerWorker = totalThroughput / m_totalClientNumber;

        string prefix = "\t\t(sim-model aggregation stats) %d-th sync: %8.2f seconds since last report ; %d samples processed by %d workers (%d by me);\n"
            "\t\t(sim-model aggregation stats) %d-th sync: totalThroughput = %.2fk samplesPerSecond , throughputPerWorker = %.2fk samplesPerSecond\n";
        fprintf(stderr, prefix.c_str(), (int)m_parameterSyncCounter, secondsSinceLastReport, (int)totalSamplesProcessedSinceLastReport, (int)m_totalClientNumber, (int)localSamplesProcessedSinceLastReport,
            (int)m_parameterSyncCounter, totalThroughput, throughputPerWorker);
        m_sampleSinceLastReport = 0;

    }

    multiverso::ArrayServer<ElemType>* m_serverArray;
    multiverso::ArrayWorker<ElemType>* m_workerArray;

    thread * m_aysncBufferThread;
    bool m_doesEveryNodesShouldSynced;
    bool m_ModelAveragingSGDSimulating;

    int m_totalClientNumber;
    int m_traceLevel;
    int m_syncPerfStats;
    Timer m_reportTimer;
    size_t m_parameterSyncCounter;
    size_t m_sampleSinceLastReport;

    bool m_useAsyncBuffer;
    int m_localBufferNum;
    int * m_bufferSwapIndex;
    int m_bufferIndexInUse;
    std::vector<multiverso::GetOption*> m_getOptions; // used by sparse table
    std::vector<multiverso::AddOption*> m_addOptions; // used by sparse table


    AdjustLearningRateAtBeginning m_adjustLearningRateAtBeginningType;
    double m_adjustCoefficient;
    size_t m_adjustMBNumber;

    vector<size_t> m_tableLength;
    size_t m_totalModelSize;
    vector<size_t> m_tableOffsets;
    //shared_ptr<ElemType>  m_deltaArray;
    ElemType * m_deltaArray;
    //std::vector<shared_ptr<ElemType>  > m_cpuAsyncBuffer;
    ElemType ** m_cpuAsyncBuffer;

    MPIWrapperPtr m_pMPI;

    // GPU double buffer
    std::vector<std::vector<Matrix<ElemType>   >> m_gpuAsyncBuffer;
    int m_tableCount;

#ifndef CPUONLY
    cudaStream_t _commStream;
#endif
};  // Class MultiversoHelper

#endif 

// A None implementation of ASGDHelper interface which does nothing
// This is used when CNTK_ENABLE_ASGD = false
template<class ElemType = float>
class NoneASGDHelper : public ASGDHelper<ElemType>
{
public:
    NoneASGDHelper(const std::list<ComputationNodeBasePtr> & learnableNodes,
        int nodeNumRanks,
        bool useAsyncBuffer = true,
        bool isSimModelAveragingSGD = false,
        AdjustLearningRateAtBeginning adjusttype = AdjustLearningRateAtBeginning::None,
        double adjustcoef = 0.2,
        size_t adjustnbmb = 600,
        int traceLevel = 0,
        int syncPerfStats = 0,
        const MPIWrapperPtr& pMPI = nullptr) { }

    ~NoneASGDHelper() { }

    void InitModel(const std::list<ComputationNodeBasePtr> & learnableNode) override { }

    bool PushAndPullModel(const std::list<ComputationNodeBasePtr> & learnableNodes, size_t sampleSinceLastSynced) override { 
        return true;
    }

    void WaitAll() override { }

    void WaitAsyncBuffer() override { }
};

template<class ElemType>
ASGDHelper<ElemType>* NewASGDHelper(
    const std::list<ComputationNodeBasePtr> & learnableNodes,                // Parameters that needs to be train
    size_t nodeNumRanks,                                                     // Number of working nodes
    bool useAsyncBuffer,                                            // Using asynchonous buffer to hide communication cost
    bool isSimulatedModelAveragingSGD,
    AdjustLearningRateAtBeginning adjusttype,
    double adjustCoef,
    size_t adjustPerMinibatches,
    int traceLevel,
    int syncPerfStats) 
{
#ifdef ASGD_PARALLEL_SUPPORT
    return new MultiversoHelper<ElemType>(learnableNodes, nodeNumRanks, useAsyncBuffer, isSimulatedModelAveragingSGD, 
                                      adjusttype, adjustCoef, adjustPerMinibatches, traceLevel, syncPerfStats);
#else
    return new NoneASGDHelper<ElemType>(learnableNodes, nodeNumRanks, useAsyncBuffer, isSimulatedModelAveragingSGD, 
                                      adjusttype, adjustCoef, adjustPerMinibatches, traceLevel, syncPerfStats); 
#endif
}

template ASGDHelper<float>* NewASGDHelper<float>(
    const std::list<ComputationNodeBasePtr> & learnableNodes,
    size_t nodeNumRanks,
    bool useAsyncBuffer,
    bool isSimulatedModelAveragingSGD,
    AdjustLearningRateAtBeginning adjusttype,
    double adjustCoef,
    size_t adjustPerMinibatches,
    int traceLevel,
    int syncPerfStats); 

template ASGDHelper<double>* NewASGDHelper<double>(
    const std::list<ComputationNodeBasePtr> & learnableNodes,
    size_t nodeNumRanks,
    bool useAsyncBuffer,
    bool isSimulatedModelAveragingSGD,
    AdjustLearningRateAtBeginning adjusttype,
    double adjustCoef,
    size_t adjustPerMinibatches,
    int traceLevel,
    int syncPerfStats); 

}}} 
