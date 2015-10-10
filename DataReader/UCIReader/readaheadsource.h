//
// <copyright file="readaheadsource.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// readaheadsource.h -- wrapper ('minibatchreadaheadsource') of a read-ahead thread that pre-rolls feature and lattice data
//
// F. Seide, Oct 2012
//
#pragma once

#include "Basics.h"
#include "simplethread.h"
#include "Matrix.h"
#include "DataReader.h"
#include <deque>
#include <stdexcept>
#include <vector>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

// ---------------------------------------------------------------------------
// minibatchreadaheadsource -- read-ahead thread that pre-rolls feature and lattice data
// ---------------------------------------------------------------------------
template<class ElemType, typename LabelType>
class minibatchreadaheadsource : public noncopyable/*assignment operator needed somewhere*/,
                                  CCritSec/*for multi-threaded access*/
{
    typedef unsigned LabelIdType;
    size_t epochframes;       // epoch size
    IDataReader<ElemType, LabelType>* m_dataReader;    // data source
    std::unique_ptr<msra::util::simplethread> thread;
    int verbosity;
    // the FIFO
    struct batchdata // all arguments to/from getbatch
    {
        size_t globalts;            // time for which we get the data
        // return values
        Matrix<ElemType> feat;
        Matrix<ElemType> labels;
        bool readRecords;
        // constructor takes feature and labels array and moves them
        batchdata (size_t globalts)  {Init(globalts); }
        void Init(size_t p_globalts)
        {
            globalts = p_globalts;
            readRecords = false;
        }

        // move constructor
        batchdata(batchdata&& moveFrom)
        {
            globalts = moveFrom.globalts;
            readRecords = moveFrom.readRecords;
            feat = move(moveFrom.feat);
            labels = move(moveFrom.labels);  //shallow copy the pointer       
        }
        //move assignment operator, shallow copy
        batchdata& operator=(batchdata&& moveFrom)  
        {
            if (this != &moveFrom)
            {
                globalts = moveFrom.globalts;
                readRecords = moveFrom.readRecords;
                // swap the matricies so we don't loose any of them
                swap(feat,moveFrom.feat);
                swap(labels,moveFrom.labels);       
            }
            return *this;
        }
    };

    deque<batchdata> batchCache;    // batches waiting to be used
    deque<batchdata> fifo;      // fifo queue for batches that are ready
    size_t epoch;                   // which epoch we are in currently

    // parameters for the thread proc (set by caller; taken over once newglobalts is set to non-SIZE_MAX (cleared back by thread))
    volatile size_t newglobalts;        // reset request
    volatile size_t currentepochreqframes;  // minibatch size for this epoch (taken from the first getbatch() call)
    volatile size_t currentepochendframe;   // we cannot request beyond

    // signaling
    mutable msra::util::signallingevent callerchangedsignal, threadchangedsignal;
    void waitcallerchanged() const { callerchangedsignal.wait(); }
    void flagcallerchanged() const { callerchangedsignal.flag(); }
    void waitthreadchanged() const { threadchangedsignal.wait(); }
    void flagthreadchanged() const { threadchangedsignal.flag(); }

    // GetBatch from the batch cache, or return a new one
    batchdata GetBatch(size_t globalts)
    {
        CAutoLock(*this);   // lock the batch cache
        if (batchCache.empty())
        {
            batchdata batch(globalts);
            batchCache.push_back(move(batch));
        }
        batchdata cacheBatch = move(batchCache.front());
        batchCache.pop_front();
        cacheBatch.Init(globalts);
        return move(cacheBatch);
    }

    // the thread proc
    volatile bool terminaterequest; // threadproc must respond to this
    size_t globalts;                // read cursor, owned by thread only
    void threadproc()
    {
        fprintf (stderr, "minibatchreadaheadsource: read-ahead thread entered\n");
        try
        {
            size_t epochreqframes = 0;  // minibatch size for this epoch (taken from the first getbatch() call)
            size_t epochendframe = 0;   // we cannot request beyond
            size_t globalts = 0;        // reset request
            while (!terminaterequest)
            {
                bool stillhasdata;
                {
                    CAutoLock lock (*this);
                    // if reset request then do it
                    if (newglobalts != SIZE_MAX)
                    {
                        // take over parameters from caller
                        globalts = newglobalts;
                        epochreqframes = currentepochreqframes;
                        epochendframe = currentepochendframe;
                        newglobalts = SIZE_MAX;     // remember we got it
                        // reset the FIFO
                        fifo.clear();
                        flagthreadchanged();        // signal state change (needed?)
                        fprintf (stderr, "minibatchreadaheadsource: thread entered new epoch, frame pos reset to %d\n", (int) globalts);
                        continue;
                    }
                    stillhasdata = !fifo.empty();
                }
                // we kick in once the FIFO is empty (and only once we know the mbsize)
                if (globalts >= epochendframe || stillhasdata)
                {
                    waitcallerchanged();    // nothing to do: wait for caller state change and check again
                    continue;
                }

                // we will bring in data from the current 'globalts' until the sub-getbatch() tells us
                // that we loaded new data (which means subsequent getbatch() will be free until the next load).
                // We assume the access pattern that
                //  - we start at or closely after the epoch boundary
                //  - we never go across an epoch boundary
                //  - the number of requested frames within an epoch is always the same except for the last MB
                // This pattern is implemented by the minibatchiterator. We require it.
                // (but it is possible that less is returned, i.e. at a sweep boundary or epoch end).
                bool readfromdisk = false;
                // we stop once data was read (the subsequent fetches will be cheap until the next data read)
                // For small setups, all data may be in RAM and thus no reading will happen anymore.
                // To guard against that, we limit the number of frames we pre-read.
                fprintf (stderr, "minibatchreadaheadsource: thread entering reading loop, frame read pos %d\n", (int) globalts);
                size_t batchesread = 0;
                const size_t prerollendframe = globalts + 360000;    // read max. 1 hour --to guard against setups that fit to RAM entirely (no disk reading after startup)
                while (!terminaterequest && !readfromdisk && globalts < epochendframe && globalts < prerollendframe)
                {
                    // get batch and append to FIFO (outside the lock)
                    batchdata batch = GetBatch(globalts);
                    const size_t requestedframes = min (epochreqframes, epochendframe - globalts);    // we must not request beyond the epoch
                    bool readRecords = m_dataReader->GetMinibatch(batch.feat, batch.labels);
                    readfromdisk = !readRecords;
                    batch.readRecords = readRecords;
                    if (readRecords)
                    {
                        batchesread++;
                        // Note: We may still get data beyond the end of the epoch, in utterance mode, since the epoch boundary likely falls within an utterance.
                        CAutoLock lock (*this);
                        if (!fifo.empty() && globalts != fifo.back().globalts + fifo.back().feat.GetNumCols())
                            throw std::logic_error ("minibatchreadaheadsource: FIFO got out of order while pre-reading new batch");
                        if (newglobalts != SIZE_MAX)
                            throw std::logic_error ("minibatchreadaheadsource: main thread reset to new epoch while current epoch not yet finished");
                        globalts += batch.feat.GetNumCols();
                    }
                    else
                    {
                        // we usually get here because the epoch has ended, or we hit end of file
                        // we support skipping the last minibatch if it is partial, so set globalts to correct value
                        if (globalts < epochendframe) // && batchesread > 0)
                        {
                            fprintf (stderr, "minibatchreadaheadsource: skipping to next epoch, epoch ended at %d\n", (int) globalts);
                            globalts = epochendframe;
                        }   // set to end of frame so the thread management works properly
                    }

                    // push into the fifo queue, if we didn't read anything the other side won't use data arrays, just flags
                    fifo.push_back (std::move (batch));
                    flagthreadchanged();        // signal state change so caller can pick up the new batch
                }
                fprintf (stderr, "minibatchreadaheadsource: thread exited reading loop, %d batches read up to frame position %d-1\n", (int) batchesread, (int) globalts);
            }
        }
        catch (...)
        {
            flagthreadchanged();
        }
        fprintf (stderr, "minibatchreadaheadsource: read-ahead thread exited\n");
    }
    void cancelthread()
    {
        terminaterequest = true;
        flagcallerchanged();
        thread->wait();
    }
public:
    minibatchreadaheadsource (IDataReader<ElemType, LabelType>* p_datareader)
      : terminaterequest (false), globalts (SIZE_MAX),
        epoch (SIZE_MAX), currentepochreqframes (0), currentepochendframe (0), newglobalts (SIZE_MAX), verbosity(2)
    {
        // kick off the thread
        m_dataReader = p_datareader;
        fprintf (stderr, "minibatchreadaheadsource: kicking off read-ahead thread\n");
        thread.reset (new msra::util::simplethread ([this] () { threadproc(); }));
    }

    // Init - Initialize the readahead reader for another epoch
    // p_globalts - sample number we are reading
    // p_framesrequested - minibatch size,
    // p_epoch - epoch we are requesting
    // p_epochframes - number of frames in the epoch 
    void Init(size_t p_globalts, size_t p_framesrequested, size_t p_epoch, size_t p_epochframes)
    {
        // first check whether the thread is still alive
        thread->check();

        // transfer over the parameter values
        epochframes = p_epochframes;
        globalts = p_globalts;
        epoch = p_epoch;

        assert(p_epoch == globalts / epochframes);

         // now start everthing rolling
        fprintf (stderr, "minibatchreadaheadsource: signalling thread to enter new epoch\n");
        CAutoLock lock (*this);
        if (!fifo.empty())
            throw std::logic_error ("getbatch: FIFO not cleared at end of epoch");
        newglobalts = globalts;
        currentepochreqframes = p_framesrequested;    // it is assumed that these won't change
        currentepochendframe = (epoch + 1) * epochframes;
        flagcallerchanged();
    }

    ~minibatchreadaheadsource()
    {
        cancelthread();
    }
    void setverbosity(int newverbosity){ verbosity = newverbosity; }

    // getbatch - get a batch from the queue
    // globalts - time stamp for this batch
    // framesrequested - number of frames requested
    // feat - feature matrix
    // labels - label matrix
    bool getbatch (const size_t globalts,
                   const size_t framesrequested, Matrix<ElemType> & feat, Matrix<ElemType> & labels)
    {
        // first check whether the thread is still alive
        thread->check();
        // in case of epoch change, we signal the thread
        size_t thisepoch = globalts / epochframes;
        if (thisepoch != epoch)
        {
            // if epoch changed just return;
            return false;
        }

        if (globalts + framesrequested < currentepochendframe && currentepochreqframes != framesrequested)
            throw std::logic_error ("getbatch: cannot change minibatch size mid-epoch");
        // loop
        for(;;) // wait for batch to appear
        {
            thread->check();
            {
                CAutoLock lock (*this);
                if (!fifo.empty())
                {
                    // get the first batch from the FIFO
                    batchdata front = std::move (fifo.front());
                    fifo.pop_front();
                    flagcallerchanged();
                    // it must be the correct one
                    if (front.globalts != globalts)
                        throw std::logic_error ("getbatch: data in FIFO out of sequence");

                    // if we actually read anything put it in here
                    if (front.readRecords)
                    {
                        // swap the Matricies between the passed in values and the batchdata
                        std::swap(feat, front.feat);
                        std::swap(labels, front.labels);
                    }

                    // put the batch in the batch cache;
                    batchCache.push_back(front);
                    return front.readRecords;
                }
            }
            // batch not there --keep looping
            waitthreadchanged();
        }
        return false;
    }
};

}}}
