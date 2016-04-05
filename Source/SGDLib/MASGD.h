//
// <copyright file="MASGD.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma  once 

#include "Basics.h"
#include "ComputationNetwork.h"
#include "Config.h"
#include "SGD.h"
#include "Matrix.h"
#include "MPIWrapper.h"
#include "TimerUtility.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono> 
#include <random>


namespace Microsoft { namespace MSR { namespace CNTK {

    enum class MAWorkerStatus
    {
        DataProcessing = 0,
        DataEnd = 1, 
        NOTSTARTED = 2 
    };

    class MASGDPerfStats
    {
    private:
        size_t m_numWorkers; 
        size_t m_myRank; 
        size_t m_numSyncPerformedInCurrentEpoch; 
        size_t m_reportFrequency; 
        size_t m_totalSamplesProcessedSinceLastReport; 
        size_t m_localSamplesProcessedSinceLastReport; 
        Timer  m_Timer; 

    public:
        MASGDPerfStats(size_t myRank, size_t numWorkers):
            m_numWorkers(numWorkers), m_myRank(myRank), m_numSyncPerformedInCurrentEpoch(0), m_reportFrequency(1)
        {
            m_Timer.Start();
        }

        void SetReportFrequency(size_t freq)
        {
            m_reportFrequency = freq;
        }

        void OnEpochStart()
        {
            m_Timer.Restart(); 
            m_numSyncPerformedInCurrentEpoch = 0; 
        }
        void OnEpochEnd()
        {
            m_Timer.Stop();
        }
        void OnMAPerformed(size_t localSamplesProcessedSinceLastSync, size_t totalSamplesProcessedSinceLastSync, float secondsOnCommunication)
        {
            m_numSyncPerformedInCurrentEpoch++;
            m_totalSamplesProcessedSinceLastReport += totalSamplesProcessedSinceLastSync; 
            m_localSamplesProcessedSinceLastReport += localSamplesProcessedSinceLastSync; 
            if ( m_reportFrequency > 0 && m_numSyncPerformedInCurrentEpoch % m_reportFrequency == 0)
            {
                ReportMAPerfStats(
                    m_totalSamplesProcessedSinceLastReport, 
                    m_localSamplesProcessedSinceLastReport, 
                    secondsOnCommunication
                );

                m_totalSamplesProcessedSinceLastReport = 0; 
                m_localSamplesProcessedSinceLastReport = 0; 
            }
        }

        void ReportMAPerfStats( size_t totalSamplesProcessedSinceLastReport, 
                                size_t localSamplesProcessedSinceLastReport, 
                                float secondOnCommunication)
        {
            m_Timer.Stop(); 
            double secondsSinceLastReport = m_Timer.ElapsedSeconds(); 
            m_Timer.Restart(); 

            float totalThroughput = secondsSinceLastReport > 0 ? totalSamplesProcessedSinceLastReport / (float)secondsSinceLastReport / 1000 : 0.0f ; 
            float throughputPerWorker = totalThroughput / m_numWorkers; 

            string prefix = "\t\t(model aggregation stats) %d-th sync: %8.2f seconds since last report (%.2f seconds on comm.); %d samples processed by %d workers (%d by me);\n"
                            "\t\t(model aggregation stats) %d-th sync: totalThroughput = %.2fk samplesPerSecond , throughputPerWorker = %.2fk samplesPerSecond\n";
            fprintf(stderr, prefix.c_str(), m_numSyncPerformedInCurrentEpoch, secondsSinceLastReport, secondOnCommunication, totalSamplesProcessedSinceLastReport, m_numWorkers, localSamplesProcessedSinceLastReport,
                                            m_numSyncPerformedInCurrentEpoch, totalThroughput, throughputPerWorker); 

        }
    };
    // base class for MA-SGD algorithm family 
    template<typename ElemType>
    class IMASGD
    {
        typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
     public:
         IMASGD(const MPIWrapperPtr& pMPI, size_t perfReportFreq)
             :m_MAworkerStatus(pMPI->NumNodesInUse(), MAWorkerStatus::NOTSTARTED), 
             m_numSyncPerformed(0), 
             m_numWorkers(pMPI->NumNodesInUse()), 
             m_myRank(pMPI->CurrentNodeRank()),
             m_pMPI(pMPI), 
             m_perfReporter(pMPI->CurrentNodeRank(), pMPI->NumNodesInUse())
         {
             m_perfReporter.SetReportFrequency(perfReportFreq);
         }
         virtual ~IMASGD()
         {
         }
         
         virtual void OnEpochStart(const std::list<ComputationNodeBasePtr>& /*LearnableNodes*/)
         {
             m_MAworkerStatus.resize(m_numWorkers);
             std::fill(m_MAworkerStatus.begin(), m_MAworkerStatus.end(), MAWorkerStatus::DataProcessing);
             m_pMPI->WaitAll(); 
             m_perfReporter.OnEpochStart();
         }

         virtual void OnEpochEnd(const std::list<ComputationNodeBasePtr>&    LearnableNodes,
                                    std::list<Matrix<ElemType>>&                smoothedGradient, 
                                    size_t                                      samplesSinceLastSync 
                                    )
         {
             m_MAworkerStatus[m_myRank] = MAWorkerStatus::DataEnd;
             bool read2sync=UpdateWorkerStatus(MAWorkerStatus::DataEnd);
             // assert(read2sync); 
             size_t totalSamplesProcessed = 0;
             float secondsOnCommunication = 0.0f; 
             if (read2sync)
             {
                 m_numSyncPerformed++;
                 ModelAggregationProcessing(samplesSinceLastSync, LearnableNodes, smoothedGradient, totalSamplesProcessed, secondsOnCommunication);
                 m_perfReporter.OnMAPerformed(samplesSinceLastSync, totalSamplesProcessed, secondsOnCommunication);
             }
             
             m_pMPI->WaitAll();             
             m_perfReporter.OnEpochEnd();
         }

         virtual bool OnArrivingAtSyncPoint(
            const std::list<ComputationNodeBasePtr>& LearnableNodes,        /* input/output: */
            std::list<Matrix<ElemType>>& smoothedGradient,                  /* input/output: under some setup, it will reset to zero*/
            size_t  samplesSinceLastSync                                    /* input:  samples processed since last sync on this worker only */
             )
         {
             bool read2Sync=UpdateWorkerStatus(MAWorkerStatus::DataProcessing);
             size_t totalSamplesProcessed=0; 
             float secondsOnCommunication = 0.0f;
             if (read2Sync)
             {
                 m_numSyncPerformed++;
                 ModelAggregationProcessing(samplesSinceLastSync, LearnableNodes, smoothedGradient, totalSamplesProcessed, secondsOnCommunication);
                 m_perfReporter.OnMAPerformed(samplesSinceLastSync, totalSamplesProcessed, secondsOnCommunication);
             }
             return read2Sync;
         }
         
         virtual void ModelAggregationProcessing(
             size_t samplesSinceLastSync,                                       /* in: */
             const std::list<ComputationNodeBasePtr>&  learnableNodes,          /* in/out */
             std::list<Matrix<ElemType>>&              smoothedGradient,        /* in/out */
             size_t&                                   totalSamplesProcessed,   /* out */
             float&                                    secondsOnCommunication   /* out */) = 0; 
         
        

    protected:
        bool    somePeersHaveArrivedAtEnd()
        {
            auto iter = std::find(m_MAworkerStatus.begin(), m_MAworkerStatus.end(), MAWorkerStatus::DataEnd);
            return iter != m_MAworkerStatus.end();
        }
        bool    UpdateWorkerStatus(MAWorkerStatus myStatus)
        {
            bool retval = false;
            m_MAworkerStatus[m_myRank] = myStatus;
            if (myStatus == MAWorkerStatus::DataEnd)
            {
                // in this case, we always return true 
                vector<MPI_Request> sendRequests(m_numWorkers);
                int sentSignal = (int)MAWorkerStatus::DataEnd;
                // 1. send my status to notify peers 
                for (int dest = 0; dest < (int)m_numWorkers; dest++)
                {
                    if (dest != m_myRank)
                    {
                        MPI_Isend(&sentSignal, 1, MPI_INT, dest, m_numSyncPerformed, m_pMPI->Communicator() , &sendRequests[dest]);
                    }
                }
                // 2. recv others 
                for (int src = 0; src < m_numWorkers; src++)
                {
                    if (src != m_myRank && m_MAworkerStatus[src] == MAWorkerStatus::DataProcessing)
                    {
                        int recvSignal = 0;
                        MPI_Status status;
                        MPI_Recv(&recvSignal, 1, MPI_INT, src, m_numSyncPerformed, m_pMPI->Communicator(), &status);
                        m_MAworkerStatus[src] = (MAWorkerStatus)recvSignal;
#if 0
                        assert(status.MPI_SOURCE == src);
                        assert(status.MPI_TAG == m_numSyncPerformed);
#endif 
                    }
                }
                // 3. make sure sending operation finished 
                for (int dest = 0; dest < m_numWorkers; dest++)
                {
                    if (dest != m_myRank)
                    {
                        MPI_Wait(&sendRequests[dest], MPI_STATUS_IGNORE);
                    }
                }
                retval = true; 
            }
            else if (myStatus == MAWorkerStatus::DataProcessing)
            {
                // in this case, we return true if all nodes are ready to sync (meaning all of them are in DataProcessing State)
                // otherwise, return false
                retval = false;
                if (!somePeersHaveArrivedAtEnd())
                {
                    int sentSignal = (int)MAWorkerStatus::DataProcessing; 
                    vector<MPI_Request> sendRequests(m_numWorkers); 
                    // 1. send my status to peers 
                    for (int dest = 0; dest < (int)m_numWorkers; dest++)
                    {
                        if (dest != m_myRank)
                        {
                            MPI_Isend(&sentSignal, 1, MPI_INT, dest, m_numSyncPerformed, m_pMPI->Communicator(), &sendRequests[dest]);
                        }
                    }
                    // 2. recv status from others (blocking call)
                    for (int src = 0; src < (int)m_numWorkers; src++)
                    {
                        if (src != m_myRank)
                        {
                            int recvSignal = 0;
                            MPI_Status status;
                            MPI_Recv(&recvSignal, 1, MPI_INT, src, m_numSyncPerformed, m_pMPI->Communicator(), &status);
#if 0 
                            // for debugging purpose, to be removed when mature 
                            assert(status.MPI_SOURCE == src);
                            assert(status.MPI_TAG == m_numSyncPerformed);
#endif 
                            m_MAworkerStatus[src] = (MAWorkerStatus)recvSignal;
                        }
                    }
                    // 3. makes sure the sending operation has completed 
                    for (int dest = 0; dest < (int)m_numWorkers;  dest++)
                    {
                        if (dest != m_myRank)
                        {
                            MPI_Wait(&sendRequests[dest], MPI_STATUS_IGNORE);
                        }
                    }
                    // 4. check peer status again
                    retval = !somePeersHaveArrivedAtEnd(); 
                }
            }
            else
            {
                LogicError("UpdateWorkerStatus cannot accept WorkerStatus other than DataProcessing or DataEnd\n");
            }

            return retval;
        }
        // borrow DownCast function from ComputationNetwork
        ComputationNodePtr DownCast(ComputationNodeBasePtr inode)
        {
            ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(inode);
            if (!node)
                InvalidArgument("an ComputationNodeBasePtr of mismatching precision was passed");
            return node;
        }

        std::vector<MAWorkerStatus> m_MAworkerStatus; 
        int                         m_numSyncPerformed; 
        size_t                      m_numWorkers; 
        size_t                      m_myRank;
        MASGDPerfStats              m_perfReporter;
        MPIWrapperPtr m_pMPI;
 };


    // Implementation of standard model averaging 
    template<typename ElemType>
    class BasicModelAveragingSGD : public IMASGD<ElemType>
    {
        typedef IMASGD<ElemType> Base; 
        using Base::m_pMPI;
        using Base::DownCast;

    public:
        BasicModelAveragingSGD(const MPIWrapperPtr& pMPI, size_t reportFreq)
            :Base(pMPI, reportFreq)
        {}

        void ModelAggregationProcessing(
            size_t samplesSinceLastSync,                                       /* in */
            const std::list<ComputationNodeBasePtr>&  learnableNodes,          /* in/out */
            std::list<Matrix<ElemType>>&              smoothedGradient,        /* in/out */
            size_t&                                   totalSamplesProcessed,   /* out */
            float&                                    secondsOnCommunication   /* out */) override
        {
            //----------------------------------------
            // 1. communicate with other nodes to negotiate  contribution weights
            //----------------------------------------
            float factor = 0;
            int   nTotalSamples = samplesSinceLastSync;
            Timer commTimer; 
            secondsOnCommunication = 0.0f;
            commTimer.Start();
            m_pMPI->AllReduce(&nTotalSamples, 1);
            commTimer.Stop();
            secondsOnCommunication += (float)commTimer.ElapsedSeconds();

            if (nTotalSamples <= 0)
            {
                // prepare for overflow 
                factor = 1.0f / m_pMPI->NumNodesInUse();
                totalSamplesProcessed = samplesSinceLastSync * m_pMPI->NumNodesInUse();
                // give an estimated one 
            }
            else
            {
                factor = (samplesSinceLastSync + 0.0f) / nTotalSamples;
                totalSamplesProcessed = nTotalSamples;
            }

            //========================================
            // 2. process for each individual node
            //========================================
            for (auto& pBaseNode : learnableNodes)
            {
                if (!pBaseNode->IsParameterUpdateRequired())
                {
                    continue;
                }
                // 2.1 model averaging
                auto pNode = DownCast(pBaseNode);
                // 2.1.1. average model from individual models 
                Matrix<ElemType> mat(pNode->Value().DeepClone()); // pNode->Value returns lvalue, so a deep copy is invoked here
                // 2.1.2. normalize the weight matrix 
                Matrix<ElemType>::Scale(factor, mat);
                // 2.1.3. send weight matrix over MPI nodes; 
                unique_ptr<ElemType[]> px(mat.CopyToArray());
                //ElemType* px = mat.CopyToArray();
                size_t    nx = mat.GetNumElements();
                // 2.1.4. inplace sum 
                commTimer.Restart();
                m_pMPI->AllReduce(px.get(), nx);
                commTimer.Stop();
                secondsOnCommunication += (float)commTimer.ElapsedSeconds();
                // 2.1.5. set value 
                pNode->Value().SetValue(mat.GetNumRows(), mat.GetNumCols(), mat.GetDeviceId(), px.get());
                // 2.1.6. clean up 
                //delete[]px;
            }
        }
    };

} } }
