//
// <copyright file="readaheadsource.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// readaheadsource.h -- wrapper ('minibatchreadaheadsource') of a read-ahead thread that pre-rolls feature and lattice data
//

#pragma once

#include "basetypes.h"
#include "minibatchiterator.h"
#include "latticearchive.h"
#ifdef _WIN32
#include "simplethread.h"
#endif
#include <deque>
#include <stdexcept>

namespace msra { namespace dbn {

// ---------------------------------------------------------------------------
// minibatchreadaheadsource -- read-ahead thread that pre-rolls feature and lattice data
// ---------------------------------------------------------------------------
class minibatchreadaheadsource : public minibatchsource /*the interface we implement*/,
                                 noncopyable /*assignment operator needed somewhere*/,
                                 CCritSec /*for multi-threaded access*/
{
    minibatchsource& source;  // the underlying source we read from
    const size_t epochframes; // epoch size
    unique_ptr<msra::util::simplethread> thread;
    int verbosity;
    // the FIFO
    struct batchdata // all arguments to/from getbatch
    {
        size_t globalts; // time for which we get the data
        // return values
        msra::dbn::matrix feat;
        std::vector<size_t> uids;
        std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> transcripts;
        std::vector<shared_ptr<const latticesource::latticepair>> lattices;
        batchdata(size_t globalts)
            : globalts(globalts)
        {
        }
    };
    deque<batchdata> fifo; // this is guarded by the CCritSec
    size_t epoch;          // which epoch we are in currently
    // parameters for the thread proc (set by caller; taken over once newglobalts is set to non-SIZE_MAX (cleared back by thread))
    volatile size_t newglobalts;           // reset request
    volatile size_t currentepochreqframes; // minibatch size for this epoch (taken from the first getbatch() call)
    volatile size_t currentepochendframe;  // we cannot request beyond
    // signalling
    mutable msra::util::signallingevent callerchangedsignal, threadchangedsignal;
    void waitcallerchanged() const
    {
        callerchangedsignal.wait();
    }
    void flagcallerchanged() const
    {
        callerchangedsignal.flag();
    }
    void waitthreadchanged() const
    {
        threadchangedsignal.wait();
    }
    void flagthreadchanged() const
    {
        threadchangedsignal.flag();
    }
    // the thread proc
    volatile bool terminaterequest; // threadproc must respond to this
    size_t globalts;                // read cursor, owned by thread only
    void threadproc()
    {
        // note on signaling:
        // This thread will always flag 'threadchangedsignal' if there is a state change,
        // e.g. a new batch is available, or we have successfully initialized.
        // The main ('caller') thread would check whether it finds a state it can make use of, and if not,
        // it will wait for the 'threadchangedsignal' and then check again the state etc.
        fprintf(stderr, "minibatchreadaheadsource: read-ahead thread entered\n");
        try
        {
            size_t epochreqframes = 0; // minibatch size for this epoch (taken from the first getbatch() call)
            size_t epochendframe = 0;  // we cannot request beyond
            size_t globalts = 0;       // reset request
            while (!terminaterequest)
            {
                bool stillhasdata;
                {
                    CAutoLock lock(*this);
                    // if reset request then do it
                    if (newglobalts != SIZE_MAX)
                    {
                        // take over parameters from caller
                        globalts = newglobalts;
                        epochreqframes = currentepochreqframes;
                        epochendframe = currentepochendframe;
                        newglobalts = SIZE_MAX; // remember we got it
                        // reset the FIFO
                        fifo.clear();
                        flagthreadchanged(); // signal state change (needed?)
                        fprintf(stderr, "minibatchreadaheadsource: thread entered new epoch, frame pos reset to %d\n", (int) globalts);
                        continue;
                    }
                    // did we run out of data to give to the caller?
                    stillhasdata = !fifo.empty();
                }
                // we kick in once the FIFO is empty (and only once we know the mbsize)
                // Note that the underlying source will be able to fulfill many more minibatches at no cost
                // since we stopped pulling minibatches from it once it told us it read something from the disk.
                // Thus it is OK (efficient) to run the FIFO empty before we continue asking the underlying source
                // for more data--it will give us quite some more data for free--which the caller can go and process--
                // before an expensive read operation is needed again.
                if (globalts >= epochendframe || stillhasdata)
                {
                    waitcallerchanged(); // nothing to do: wait for caller state change and check again
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
                fprintf(stderr, "minibatchreadaheadsource: thread entering reading loop, frame read pos %d\n", (int) globalts);
                size_t batchesread = 0;
                const size_t prerollendframe = globalts + 360000; // read max. 1 hour --to guard against setups that fit to RAM entirely (no disk reading after startup)
                while (!terminaterequest && !readfromdisk && globalts < epochendframe && globalts < prerollendframe)
                {
                    // get batch and append to FIFO (outside the lock)
                    batchdata batch(globalts);
                    const size_t requestedframes = min(epochreqframes, epochendframe - globalts); // we must not request beyond the epoch
                    readfromdisk = source.getbatch(globalts, requestedframes, batch.feat, batch.uids, batch.transcripts, batch.lattices);
                    batchesread++;
                    // Note: We may still get data beyond the end of the epoch, in utterance mode, since the epoch boundary likely falls within an utterance.
                    CAutoLock lock(*this);
                    if (!fifo.empty() && globalts != fifo.back().globalts + fifo.back().feat.cols())
                        throw std::logic_error("minibatchreadaheadsource: FIFO got out of order while pre-reading new batch");
                    if (newglobalts != SIZE_MAX)
                        throw std::logic_error("minibatchreadaheadsource: main thread reset to new epoch while current epoch not yet finished");
                    globalts += batch.feat.cols();
                    fifo.push_back(std::move(batch));
                    flagthreadchanged(); // signal state change so caller can pick up the new batch
                }
                fprintf(stderr, "minibatchreadaheadsource: thread exited reading loop, %d batches read up to frame position %d-1\n", (int) batchesread, (int) globalts);
            }
            fprintf(stderr, "minibatchreadaheadsource: reading loop was terminated at frame position %d-1\n", (int) globalts);
        }
        catch (const exception& e)
        {
            fprintf(stderr, "minibatchreadaheadsource: exception caught in read-ahead thread: %s\n", e.what());
            thread->fail(e); // set the error first before we signal the caller
            flagthreadchanged();
            throw; // (this will set the error a second time; OK)
        }
        fprintf(stderr, "minibatchreadaheadsource: read-ahead thread exited normally\n");
    }
    void cancelthread() // this is only ever called by the destructor
    {
        fprintf(stderr, "minibatchreadaheadsource: requesting thread termination\n");
        terminaterequest = true;
        flagcallerchanged();
        thread->wait();
    }

public:
    minibatchreadaheadsource(minibatchsource& source, size_t epochframes)
        : source(source), epochframes(epochframes), terminaterequest(false), globalts(SIZE_MAX), epoch(SIZE_MAX), currentepochreqframes(0), currentepochendframe(0), newglobalts(SIZE_MAX), verbosity(2)
    {
        // kick off the thread
        fprintf(stderr, "minibatchreadaheadsource: kicking off read-ahead thread\n");
        thread.reset(new msra::util::simplethread([this]()
                                                  {
                                                      threadproc();
                                                  }));
    }
    ~minibatchreadaheadsource()
    {
        fprintf(stderr, "~minibatchreadaheadsource: destructing read-ahead thread\n");
        cancelthread();
    }
    void setverbosity(int newverbosity)
    {
        verbosity = newverbosity;
    }
    bool getbatch(const size_t globalts,
                  const size_t framesrequested, msra::dbn::matrix& feat, std::vector<size_t>& uids,
                  std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>>& transcripts,
                  std::vector<shared_ptr<const latticesource::latticepair>>& lattices)
    {
#if 1
        // first check whether the thread is still alive
        thread->check();
        // in case of epoch change, we signal the thread
        size_t thisepoch = globalts / epochframes;
        if (thisepoch != epoch)
        {
            fprintf(stderr, "minibatchreadaheadsource: signalling thread to enter new epoch\n");
            epoch = thisepoch; // remember for next check --we have officially changed epochs
            CAutoLock lock(*this);
            if (!fifo.empty())
                throw std::logic_error("getbatch: FIFO not cleared at end of epoch");
            newglobalts = globalts;
            currentepochreqframes = framesrequested; // it is assumed that these won't change
            currentepochendframe = (epoch + 1) * epochframes;
            flagcallerchanged();
        }
        else if (globalts + framesrequested < currentepochendframe && currentepochreqframes != framesrequested)
            throw std::logic_error("getbatch: cannot change minibatch size mid-epoch");
        // loop
        bool readfromdisk = false;
        for (;;) // wait for batch to appear
        {
            thread->check();
            {
                CAutoLock lock(*this);
                if (!fifo.empty())
                {
                    // get the first batch from the FIFO
                    batchdata front = std::move(fifo.front());
                    fifo.pop_front();
                    flagcallerchanged();
                    // it must be the correct one
                    if (front.globalts != globalts)
                        throw std::logic_error("getbatch: data in FIFO out of sequence");
                    // return it
                    feat = std::move(front.feat);
                    uids = std::move(front.uids);
                    transcripts = std::move(front.transcripts);
                    lattices = std::move(front.lattices);
                    return readfromdisk;
                }
            }
            // batch not there --keep looping
            waitthreadchanged();
            readfromdisk = true; // we had to wait --use to indicate that we needed to read data (does not really matter...)
        }
#else
        return source.getbatch(globalts, framesrequested, feat, uids, transcripts, lattices);
#endif
    }
    bool getbatch(const size_t globalts,
                  const size_t framesrequested, std::vector<msra::dbn::matrix>& feat, std::vector<std::vector<size_t>>& uids,
                  std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>>& transcripts,
                  std::vector<shared_ptr<const latticesource::latticepair>>& lattices)
    {

        feat.resize(1);
        uids.resize(1);
        //transcripts.resize(1);
        //lattices.resize(1);
        return getbatch(globalts, framesrequested, feat[0], uids[0], transcripts, lattices);
    }

    size_t totalframes() const
    {
        return source.totalframes();
    }
    size_t epochsize() const
    {
        return epochframes;
    }
    double gettimegetbatch()
    {
        return source.gettimegetbatch();
    } // TODO: no, use our own time measurement
    size_t firstvalidglobalts(const size_t globalts)
    {
        return source.firstvalidglobalts(globalts);
    }
    const std::vector<size_t>& unitcounts() const
    {
        return source.unitcounts();
    }
};
};
};
