//
// <copyright file="MinibatchFetcher.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "ComputationNetwork.h"
#include "DataReader.h"
#include "MinibatchFetcher.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace Microsoft { namespace MSR { namespace CNTK {

// This derived class is an implementation of a prefetcher for minibatches. It contains a simple producer-consumer synchronization
// between reader and compute. It creates a separate thread for the reader and it allows a single compute to execute concurrently
// with a single read-ahead of a minibatch. This ensures that compute always has input data to work on, and is not blocked on
// reads off the disk, nor transfers of memory from host to device in the GPU case.
template<class ElemType>
class MinibatchPrefetcher : public MinibatchFetcher<ElemType>
{
public:
    MinibatchPrefetcher(IDataReader<ElemType>* trainSetDataReader, const std::map<std::wstring, Matrix<ElemType>*>* inputMatrices) :
        MinibatchFetcher<ElemType>(trainSetDataReader, inputMatrices),
        m_isEpochReadingDone(false),
        m_minibatchReady(false),
        m_isTerminating(false)
    {
        m_deviceId = this->m_inputMatrices->begin()->second->GetDeviceId();

        for (auto iter = this->m_inputMatrices->begin(); iter != this->m_inputMatrices->end(); iter++)
        {
            assert(m_deviceId == iter->second->GetDeviceId());
            m_prefetchInput[iter->first] = new Matrix<ElemType>(iter->second->GetNumRows(),
                                                                iter->second->GetNumCols(),
                                                                iter->second->GetDeviceId(),
                                                                iter->second->GetMatrixType(),
                                                                iter->second->GetFormat());
        }

        // Launch a worker thread
        m_prefetchThread = std::thread([this]() { this->PrefetchWorker(); });
    }

    virtual ~MinibatchPrefetcher()
    {
        // Send a signal to the worker thread that we are in shutdown mode
        m_isTerminating = true;

        // Make sure that worker thread is unblocked because we are about to wait to join with it. If
        // worker thread is in the middle of reading, let it finish so that we can safely grab the lock.
        if (!m_isEpochReadingDone)
        {
            fprintf(stderr, "Exiting minibatch loop before reading all the data, waiting to sync with the prefetch thread...\n");
            m_cv.notify_one();
        }

        m_prefetchThread.join();

        // Clean up prefetch matrix inputs
        for (auto iter = m_prefetchInput.begin(); iter != m_prefetchInput.end(); iter++)
        {
            delete iter->second;
        }
    }

    virtual bool GetMinibatch()
    {
        bool hasMoreEpochReading = false;

        // Wait until minibatch is ready to be consumed
        {
            std::unique_lock<std::mutex> mutexLock(m_mutex);
            m_cv.wait(mutexLock, [this] { return m_minibatchReady == true; });

            // This function now owns the lock

            // m_isTerminating is set on this same thread, but only in destructor
            assert(!m_isTerminating);

            if (!m_isEpochReadingDone)
            {
                // Record an event after all computation for the previous minibatch has been scheduled
                // ensuring that this event can safely be observed after all compute has finished.
                Matrix<ElemType>::RecordComputeSyncPoint(m_deviceId);

                // Swap the input matrices to make use of data that has already been read.
                // This should be as simple as "m_prefetchInput.swap(m_inputMatrices)", but unfortunately
                // underlying Matrix<ElemType> pointers are cached, so we need to dig deeper to do a swap.
                for (auto iter = this->m_inputMatrices->begin(); iter != this->m_inputMatrices->end(); iter++)
                {
                    assert(m_deviceId == iter->second->GetDeviceId());
                    std::swap(*(iter->second), *m_prefetchInput[iter->first]);
                }

                hasMoreEpochReading = true;
            }

            // Announce to worker thread to fetch another batch.
            m_minibatchReady = false;
        }
        m_cv.notify_one();

        return hasMoreEpochReading;
    }

private:

    void PrefetchWorker()
    {
        Matrix<ElemType>::EnableConcurrentRead(m_deviceId);

        while (!m_isEpochReadingDone)
        {
            // Wait until prefetch is requested
            std::unique_lock<std::mutex> mutexLock(m_mutex);
            m_cv.wait(mutexLock, [this] { return (!m_minibatchReady || m_isTerminating); });

            // We now own the lock

            // If the main thread has an early exit due to break or exception, it
            // will initiate a shutdown and it will wait for this thread to complete.
            // Thus, we need to check for that condition before proceeding.
            m_isEpochReadingDone = m_isTerminating ? true : PrefetchOneMiniBatch();

            // Signal to main thread that minibatch is ready to be consumed
            m_minibatchReady = true;

            // Manual unlocking is done before notifying, to avoid waking up
            // the waiting thread only to block again (see notify_one for details)
            mutexLock.unlock();
            m_cv.notify_one();
        }
    }

    bool PrefetchOneMiniBatch()
    {
        // This function must be called while holding a lock

        // Schedule a wait event on the read stream that ensures that nothing can be further
        // scheduled on that stream until dependent compute event has been observed.
        // Please note that first two calls will be special cases:
        //
        // 1) First mini-batch is fetched before RecordComputeSyncPoint() is ever called
        // 2) Second mini-batch is fetched depending on RecordComputeSyncPoint() reported before
        //    scheduling any actual work on the compute thread
        // 
        // Dependency chain looks like this (F = fetch, C = compute):
        //
        // F1 -> C1 -> F3 (fetch #3 depends on compute #1 completing, which depended on fetch #1 completing)
        // F2 -> C2 -> F4
        // F3 -> C3 -> F5
        // 
        // It is fetch #3 that *must* observe the event that happened between computes #1 and #2
        // before proceeding to read into the buffer that was used by compute #1.
        Matrix<ElemType>::SyncComputeBeforeRead(m_deviceId);

        // Get the next minibatch and wait for it to be available on the device
        bool isDone = !this->m_reader->GetMinibatch(const_cast<std::map<std::wstring, Matrix<ElemType>*>&>(m_prefetchInput));
        Matrix<ElemType>::SyncPendingRead(m_deviceId);

        return isDone;
    }

    // @TODO: We need to add support for a larger number of prefetch buffers, larger than 1
    std::map<std::wstring, Matrix<ElemType>*> m_prefetchInput;
    std::thread m_prefetchThread;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    DEVICEID_TYPE m_deviceId;
    std::atomic<bool> m_isEpochReadingDone;
    std::atomic<bool> m_minibatchReady;
    std::atomic<bool> m_isTerminating;
};

}}}