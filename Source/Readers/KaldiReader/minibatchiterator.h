//
// <copyright file="minibatchiterator.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// minibatchiterator.h -- iterator for minibatches


#pragma once
#define NONUMLATTICEMMI     // [v-hansu] move from main.cpp, no numerator lattice for mmi training

#include <vector>
#include "ssematrix.h"
#include "latticearchive.h"         // for reading HTK phoneme lattices (MMI training)
#include "latticesource.h"
#include "simple_checked_arrays.h"  // for const_array_ref

namespace msra { namespace dbn {

// ---------------------------------------------------------------------------
// minibatchsource -- abstracted interface into frame sources
// There are three implementations:
//  - the old minibatchframesource to randomize across frames and page to disk
//  - minibatchutterancesource that randomizes in chunks and pages from input files directly
//  - a wrapper that uses a thread to read ahead in parallel to CPU/GPU processing
// ---------------------------------------------------------------------------
class minibatchsource
{
public:
    // read a minibatch
    // This function returns all values in a "caller can keep them" fashion:
    //  - uids are stored in a huge 'const' array, and will never go away
    //  - transcripts are copied by value
    //  - lattices are returned as a shared_ptr
    // Thus, getbatch() can be called in a thread-safe fashion, allowing for a 'minibatchsource' implementation that wraps another with a read-ahead thread.
    // Return value is 'true' if it did read anything from disk, and 'false' if data came only from RAM cache. This is used for controlling the read-ahead thread.
    virtual bool getbatch (const size_t globalts,
                           const size_t framesrequested, msra::dbn::matrix & feat, std::vector<size_t> & uids,
                           std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> & transcripts,
                           std::vector<shared_ptr<const latticesource::latticepair>> & lattices) = 0;
    // alternate (updated) definition for multiple inputs/outputs - read as a vector of feature matrixes or a vector of label strings
    virtual bool getbatch (const size_t globalts,
                           const size_t framesrequested, std::vector<msra::dbn::matrix> & feat, std::vector<std::vector<size_t>> & uids,
                           std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> & transcripts,
                           std::vector<shared_ptr<const latticesource::latticepair>> & lattices) = 0;
    virtual size_t totalframes() const = 0;

    virtual double gettimegetbatch () = 0;                          // used to report runtime
    virtual size_t firstvalidglobalts (const size_t globalts) = 0;  // get first valid epoch start from intended 'globalts'
    virtual const std::vector<size_t> & unitcounts() const = 0;     // report number of senones
    virtual void setverbosity(int newverbosity) = 0;    
    virtual ~minibatchsource() { }
};


// ---------------------------------------------------------------------------
// minibatchiterator -- class to iterate over one epoch, minibatch by minibatch
// This iterator supports both random frames and random utterances through the minibatchsource interface whichis common to both.
// This supports multiple data passes with identical randomization; which is intended to be used for utterance-based training.
// ---------------------------------------------------------------------------
class minibatchiterator
{
    void operator= (const minibatchiterator &); // (non-copyable)

    const size_t epochstartframe;
    const size_t epochendframe;
    size_t firstvalidepochstartframe;       // epoch start frame rounded up to first utterance boundary after epoch boundary
    const size_t requestedmbframes;         // requested mb size; actual minibatches can be smaller (or even larger for lattices)
    const size_t datapasses;                // we return the data this many times; caller must sub-sample with 'datapass'

    msra::dbn::minibatchsource & source;    // feature source to read from

    std::vector<msra::dbn::matrix> featbuf;              // buffer for holding curernt minibatch's frames
    std::vector<std::vector<size_t>> uids;               // buffer for storing current minibatch's frame-level label sequence
    std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> transcripts;    // buffer for storing current minibatch's word-level label sequences (if available and used; empty otherwise)
    std::vector<shared_ptr<const latticesource::latticepair>> lattices;     // lattices of the utterances in current minibatch (empty in frame mode)

    size_t mbstartframe;                    // current start frame into generalized time line (used for frame-wise mode and for diagnostic messages)
    size_t actualmbframes;                  // actual number of frames in current minibatch
    size_t datapass;                        // current datapass = pass through the data
    double timegetbatch;                    // [v-hansu] for time measurement
    double timechecklattice;
private:
    // fetch the next mb
    // This updates featbuf, uids[], mbstartframe, and actualmbframes.
    void fillorclear()
    {
        if (!hasdata()) // we hit the end of the epoch: just cleanly clear out everything (not really needed, can't be requested ever)
        {
            foreach_index(i, featbuf)
                featbuf[i].resize (0, 0);

            foreach_index(i,uids)
                uids[i].clear();
            
            transcripts.clear();
            actualmbframes = 0;
            return;
        }
        // process one mini-batch (accumulation and update)
        assert (requestedmbframes > 0);
        const size_t requestedframes = min (requestedmbframes, epochendframe - mbstartframe);    // (< mbsize at end)
        assert (requestedframes > 0);
        source.getbatch (mbstartframe, requestedframes, featbuf, uids, transcripts, lattices);
        timegetbatch = source.gettimegetbatch();
        actualmbframes = featbuf[0].cols(); // for single i/o, there featbuf is length 1
        // note:
        //  - in frame mode, actualmbframes may still return less if at end of sweep
        //  - in utterance mode, it likely returns less than requested, and
        //    it may also be > epochendframe (!) for the last utterance, which, most likely, crosses the epoch boundary
        auto_timer timerchecklattice;
        if (!lattices.empty())
        {
            size_t totalframes = 0;
            foreach_index (i, lattices)
                totalframes += lattices[i]->getnumframes();
            if (totalframes != actualmbframes)
                throw std::logic_error ("fillorclear: frames in lattices do not match minibatch size");
        }
        timechecklattice = timerchecklattice;
    }
    bool hasdata() const { return mbstartframe < epochendframe; } // true if we can access and/or advance
    void checkhasdata() const { if (!hasdata()) throw std::logic_error ("minibatchiterator: access beyond end of epoch"); }
public:
    // interface: for (minibatchiterator i (...), i, i++) { ... }
    minibatchiterator (msra::dbn::minibatchsource & source, size_t epoch, size_t epochframes, size_t requestedmbframes, size_t datapasses)
        : source (source),
          epochstartframe (epoch * epochframes),
          epochendframe (epochstartframe + epochframes),
          requestedmbframes (requestedmbframes),
          datapasses (datapasses),
          timegetbatch (0), timechecklattice (0)
    {
        firstvalidepochstartframe = source.firstvalidglobalts (epochstartframe); // epochstartframe may fall between utterance boundaries; this gets us the first valid boundary
        fprintf (stderr, "minibatchiterator: epoch %zd: frames [%zd..%zd] (first utterance at frame %zd) with %zd datapasses\n",
                 epoch, epochstartframe, epochendframe, firstvalidepochstartframe, datapasses);
        mbstartframe = firstvalidepochstartframe;
        datapass = 0;
        fillorclear(); // get the first batch
    }
    
    // TODO not nice, but don't know how to access these frames otherwise
    // mbiterator constructor, set epochstart and -endframe explicitly
    minibatchiterator (msra::dbn::minibatchsource & source, size_t epoch, size_t epochstart, size_t epochend, size_t requestedmbframes, size_t datapasses)
        : source (source),
          epochstartframe (epochstart),
          epochendframe (epochend),
          requestedmbframes (requestedmbframes),
          datapasses (datapasses),
          timegetbatch (0), timechecklattice (0)
    {
        firstvalidepochstartframe = source.firstvalidglobalts (epochstartframe); // epochstartframe may fall between utterance boundaries; this gets us the first valid boundary
        fprintf (stderr, "minibatchiterator: epoch %zd: frames [%zd..%zd] (first utterance at frame %zd) with %zd datapasses\n",
                 epoch, epochstartframe, epochendframe, firstvalidepochstartframe, datapasses);
        mbstartframe = firstvalidepochstartframe;
        datapass = 0;
        fillorclear(); // get the first batch
    }

    // need virtual destructor to ensure proper destruction
    virtual ~minibatchiterator()
    {}

    // returns true if we still have data
    operator bool() const { return hasdata(); }

    // advance to the next minimb
    void operator++(int/*denotes postfix version*/)
    {
        checkhasdata();
        mbstartframe += actualmbframes;
        // if we hit the end, we will get mbstartframe >= epochendframe <=> !hasdata()
        // (most likely actually mbstartframe > epochendframe since the last utterance likely crosses the epoch boundary)
        // in case of multiple datapasses, reset to start when hitting the end
        if (!hasdata() && datapass + 1 < datapasses)
        {
            mbstartframe = firstvalidepochstartframe;
            datapass++;
            fprintf (stderr, "\nminibatchiterator: entering %zd-th repeat pass through the data\n", datapass+1);
        }
        fillorclear();
    }

    // accessors to current minibatch
    size_t currentmbstartframe() const { return mbstartframe; }
    size_t currentmbframes() const { return actualmbframes; }
    size_t currentmblattices() const { return lattices.size(); }
    size_t currentdatapass() const { return datapass; } // 0..datapasses-1; use this for sub-sampling
    size_t requestedframes() const {return requestedmbframes; }
    double gettimegetbatch () {return timegetbatch;}
    double gettimechecklattice () {return timechecklattice;}
    bool isfirst() const { return mbstartframe == firstvalidepochstartframe && datapass == 0; }
    float progress() const  // (note: 100%+eps possible for last utterance)
    {
        const float epochframes = (float) (epochendframe - epochstartframe);
        return (mbstartframe + actualmbframes - epochstartframe + datapass * epochframes) / (datapasses * epochframes);
    }
    std::pair<size_t,size_t> range() const { return make_pair (epochstartframe, epochendframe); }

    // return the current minibatch frames as a matrix ref into the feature buffer
    // Number of frames is frames().cols() == currentmbframes().
    // For frame-based randomization, this is 'requestedmbframes' most of the times, while for utterance randomization,
    // this depends highly on the utterance lengths.
    // User is allowed to manipulate the frames... for now--TODO: move silence filtering here as well

    msra::dbn::matrixstripe frames(size_t i) { checkhasdata(); assert(featbuf.size()>=i+1); return msra::dbn::matrixstripe (featbuf[i], 0, actualmbframes); }

    msra::dbn::matrixstripe frames() { checkhasdata(); assert(featbuf.size()==1); return msra::dbn::matrixstripe (featbuf[0], 0, actualmbframes); }

    // return the reference transcript labels (state alignment) for current minibatch
    /*const*/ std::vector<size_t> & labels() { checkhasdata(); assert(uids.size()==1);return uids[0]; }
    /*const*/ std::vector<size_t> & labels(size_t i) { checkhasdata(); assert(uids.size()>=i+1); return uids[i]; }

    // return a lattice for an utterance (caller should first get total through currentmblattices())
    shared_ptr<const msra::dbn::latticepair> lattice (size_t uttindex) const { return lattices[uttindex]; }    // lattices making up the current 

    // return the reference transcript labels (words with alignments) for current minibatch (or empty if no transcripts requested)
    const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word> transcript (size_t uttindex) { return transcripts.empty() ? const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>() : transcripts[uttindex]; }
};

};};
