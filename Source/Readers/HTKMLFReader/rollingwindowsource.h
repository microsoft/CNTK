//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// rollingwindowsource.h -- implementation of a rolling-window minibatch source ('minibatchframesource') with a disk page file
//

#pragma once

#include "Basics.h" // for attempt()
#ifdef _WIN32
#include "numahelpers.h" // for NUMA allocation
#endif
#include "minibatchsourcehelpers.h"
#include "minibatchiterator.h"
#include "biggrowablevectors.h"
#include "ssematrix.h"
#include "RandomOrdering.h"

namespace msra { namespace dbn {

// ---------------------------------------------------------------------------
// biggrowablevectorarray -- a big array of vectors for features, growable (push_back)
// Data is striped across NUMA nodes, as to not clog them up.
// This also supports paging to disk, which is used for the old minibatchframesource.
// ---------------------------------------------------------------------------
class biggrowablevectorarray : public growablevectorbase<msra::dbn::matrix>
{
    size_t m; // dim

    size_t inmembegin; // range we have in memory, rounded to enclosing blocks (not rounded at end)
    size_t inmemend;

    std::wstring pagepath; // path for paging, empty if no paging
    auto_file_ptr f;  // file handle for paging
    bool reading;     // have we begun reading?

    // allocate a block
    msra::dbn::matrix *newblock() const
    {
// we stripe the data across NUMA nodes as to not fill up one node with the feature data
#ifdef _WIN32
        msra::numa::overridenode((int) msra::numa::getmostspaciousnumanode());
#endif
        msra::dbn::matrix *res = new msra::dbn::matrix(m, elementsperblock);
#ifdef _WIN32
        msra::numa::overridenode(-1); // note: we really should reset it also in case of failure
#endif
        return res;
    }

    // handling of page file
    bool paging() const
    {
        return !pagepath.empty();
    }
    void openpagefile(bool wantread)
    {
        if (!paging())
            return;
        msra::files::make_intermediate_dirs(pagepath);

        if (!wantread)
        {
            FILE *ftry = NULL;
            std::wstring pathname(pagepath);
            ftry = _wfopen(pathname.c_str(), L"wbS");
            if (ftry)
                fclose(ftry);
        }

        /*
                code below to cycle through a-z appended to file name is no longer necessary
                since caller guarantees unique file names via HTKMLFReader
                and we want the pagepath logged to the user to be the actual one used by the code

            // try to open the pagepath from a to z
            if (!wantread)
            {
                FILE *ftry = NULL;
                char trynum = 'a';
                while (!ftry && trynum <= 'z')
                {
                    std::wstring pathname (pagepath);
                    pathname += trynum++;
                    ftry = _wfopen (pathname.c_str(), L"wbS");
                }
                if (ftry) fclose (ftry);
                pagepath += --trynum;
            }
            */
        f = fopenOrDie(pagepath, wantread ? L"rbS" : L"wbS");
        reading = wantread;
    }
    void flushlastblock() // during population phase, must be called once per block in sequence
    {
        if (!paging())
            return;
        assert(!reading);
        if (blocks.empty())
            return;
        const size_t blockid = blocks.size() - 1;
        msra::dbn::matrix &block = *blocks[blockid];
        assert(fgetpos(f) == blockid * block.sizeinpagefile());
        block.topagefile(f);
        blocks[blockid].reset(); // free the memory
        assert(blockid * elementsperblock == inmembegin);
        inmembegin = inmemend; // empty range
    }
    void releaseblock(size_t t0) // t0=block start time
    {
        assert(paging() && reading);
        size_t blockid = t0 / elementsperblock;
        assert(blockid * elementsperblock == t0);
        assert(blocks[blockid]);
        fprintf(stderr, "recoverblock: releasing feature block %d [%d..%d)\n", (int) blockid, (int) t0, (int) (t0 + elementsperblock - 1));
        blocks[blockid].reset(); // free the memory
    }
    void recoverblock(size_t t0) // t0=block start time
    {
        assert(paging() && reading);
        size_t blockid = t0 / elementsperblock;
        assert(blockid * elementsperblock == t0);
        assert(!blocks[blockid]);
        fprintf(stderr, "recoverblock: recovering feature block %d [%d..%d)\n", (int) blockid, (int) t0, (int) (t0 + elementsperblock - 1));
        blocks[blockid].reset(newblock());
        msra::dbn::matrix &block = *blocks[blockid];
        fsetpos(f, blockid * block.sizeinpagefile());
        block.frompagefile(f);
    }

public:
    biggrowablevectorarray(const std::wstring &pagepath)
        : growablevectorbase(65536), m(0), inmembegin(0), inmemend(0), pagepath(pagepath), reading(false)
    {
        openpagefile(false);
        if (paging())
            fprintf(stderr, "biggrowablevectorarray: creating disk backup store at '%ls'\n", pagepath.c_str());
    }
    ~biggrowablevectorarray()
    { // clean up the big temp file
        if (paging())
        {
            fclose(f);
            if (_wunlink(pagepath.c_str()) == 0)
                fprintf(stderr, "biggrowablevectorarray: deleted disk backup store at '%ls'\n", pagepath.c_str());
            else
                fprintf(stderr, "biggrowablevectorarray: unable to delete disk backup store at '%ls'\n", pagepath.c_str());
        }
    }

    size_t dim() const
    {
        return m;
    } // dimension of a frame

    // reading phase
    void push_back(const std::vector<float> &in)
    {
        assert(!in.empty());
        assert(m == 0 || m == in.size());
        m = in.size();
        const size_t blockid = n / elementsperblock;
        assert(blockid <= blocks.size());
        if (blockid == blocks.size()) // a new block is needed
        {
            flushlastblock();
            blocks.push_back(std::unique_ptr<msra::dbn::matrix>(newblock()));
        }
        const size_t blockn = n % elementsperblock;
        msra::dbn::matrix &block = *blocks[blockid].get();
        foreach_index (k, in)
            block(k, blockn) = in[k];
        n++;
        inmemend = n;
    }
    void no_more_push_back() // done pushing --switch to consumption mode
    {
        if (!paging())
            return;
        // finish off last block
        flushlastblock();
        fflushOrDie(f);
        fprintf(stderr, "biggrowablevectorarray: disk backup store created, %d frames, %lu bytes\n", (int) n, (unsigned long)fgetpos(f));
        fclose(f);
        foreach_index (i, blocks)
            assert(!blocks[i]);         // ensure we flushed
        assert(inmembegin == inmemend); // nothing in cache
        // switch to reading mode
        openpagefile(true);
    }

    // access phase
    // Returns 'true' if data was actually read from disk.
    bool require(std::pair<size_t, size_t> bounds) // we require this range of frames
    {
        bool readfromdisk = false;

        // get bounds rounded to block boundaries
        const size_t ts = bounds.first / elementsperblock * elementsperblock;
        const size_t te = std::min(n, (bounds.second + elementsperblock - 1) / elementsperblock * elementsperblock);
        assert(paging());
        // free all the memmory
        for (size_t t = inmembegin; t < inmemend; t += elementsperblock)
        {
            if (t >= ts && t < te) // if in wanted range then skip to end of it
                t = te - elementsperblock;
            else
                releaseblock(t);
        }
        // page in all required blocks
        for (size_t t = ts; t < te; t += elementsperblock)
        {
            if (t >= inmembegin && t < inmemend) // if in memory already then skip to end of it
                t = inmemend - elementsperblock;
            else
            {
                recoverblock(t);
                readfromdisk = true; // tell caller we did something expensive
            }
        }
        // got it
        inmembegin = ts;
        inmemend = te;
        return readfromdisk;
    }
    const msra::dbn::matrixstripe operator[](size_t t) const // get a feature vector
    {
        if (t < inmembegin || t >= inmemend)
            LogicError("biggrowablevectorarray: attempt to access vector without requesting to page it in first");
        const size_t blockt = getblockt(t);
        /*const*/ msra::dbn::matrix &block = getblock(t);
        return msra::dbn::matrixstripe(block, blockt, 1);
    }
    std::wstring pagepathname()
    {
        return pagepath;
    }
    void cleanuppagefile()
    {
        if (paging())
        {
            fclose(f);
            if (_wunlink(pagepath.c_str()) == 0)
            {
                fprintf(stderr, "biggrowablevectorarray: deleted disk backup store at '%ls'\n", pagepath.c_str());
            }
            else
            {
                fprintf(stderr, "biggrowablevectorarray: could NOT delete disk backup store at '%ls'\n", pagepath.c_str());
            }
        }
    }
};

// ---------------------------------------------------------------------------
// minibatchframesource -- feature source to provide randomized frames in minibatches
// This is the old code that pages all frames to a huge disk file first.
// (The new minibatchutterancesource pages from input files directly and can also
// operate in utterance mode for MMI training.)
// ---------------------------------------------------------------------------
class minibatchframesource : public minibatchsource
{
    size_t vdim;             // feature dimension after augmenting neighhors (0: don't read features)
    unsigned int sampperiod; // (for reference and to check against model)
    std::string featkind;
    size_t featdim;
    // cache
    biggrowablevectorarray frames;                         // [t][i] all features concatenated
    std::vector<char> boundaryflags;                       // [t] -1 for first and +1 for last frame, 0 else (for augmentneighbors())
    std::vector<CLASSIDTYPE> classids;                     // [t] the state that the frame belongs to
    size_t numframes;                                      // total frames (==frames.size()==boundaryflags.size()==classids.size()) unless special modes vdim == 0 and/or no labels
    Microsoft::MSR::CNTK::RandomOrdering m_randomOrdering; // [t] -> t'
    double timegetbatch;
    int verbosity;

public:
    // constructor
    // Pass empty labels to denote unsupervised training (so getbatch() will not return uids).
    minibatchframesource(const std::vector<std::wstring> &infiles, const std::map<std::wstring, std::vector<msra::asr::htkmlfentry>> &labels,
                         size_t vdim, size_t udim, size_t randomizationrange, const std::wstring &pagepath, const bool mayhavenoframe = false, int addEnergy = 0)
        : vdim(vdim), sampperiod(0), featdim(0), numframes(0), frames(pagepath), timegetbatch(0), verbosity(2)
    {
        if (vdim == 0 && labels.empty())
            RuntimeError("minibatchframesource: when running without features, labels are needed");
        // at this stage, we simply page in the entire training set at once and work off RAM
        // We will benefit from feature archives indirectly through htkfeatio.
        // TODO:
        //  - infiles must specify time range
        //  - at this stage only reserve() (we know the time range; allocate second-layer structure)
        //  - implement block-wise paging directly from HTK feature files through htkfeatreader
        featkind.clear();
        std::vector<float> frame;
        fprintf(stderr, "minibatchframesource: reading %d utterances..", (int) infiles.size());
        size_t numclasses = 0;           // number of units found (actually max id +1)
        size_t notfound = 0;             // number of entries missing in MLF
        msra::asr::htkfeatreader reader; // feature reader
        reader.AddEnergy(addEnergy);

        foreach_index (i, infiles)
        {
            if (i % (infiles.size() / 100 + 1) == 0)
            {
                fprintf(stderr, ".");
                fflush(stderr);
            }
            msra::basetypes::matrix<float> feat;
            msra::asr::htkfeatreader::parsedpath ppath(infiles[i]);

            // skip files for which labels don't exist (assuming bad alignment)
            std::wstring key;
            if (!labels.empty()) // empty means unsupervised mode (don't load any)
            {
#ifdef _WIN32
                key = regex_replace((std::wstring) ppath, std::wregex(L"\\.[^\\.\\\\/:]*$"), std::wstring()); // delete extension (or not if none)
#else
                key = removeExtension(ppath);
#endif
                if (labels.find(key) == labels.end())
                {
                    if (notfound < 5)
                        fprintf(stderr, "\nminibatchframesource: %d-th file not found in MLF label set: %ls", i, key.c_str());
                    notfound++;
                    continue; // skip this utterance at all
                }
            }

            // get feature frames
            if (vdim != 0) // (vdim == special mode to not read features at all)
            {
                msra::util::attempt(5, [&]()
                                    {
                                        reader.read(ppath, featkind, sampperiod, feat); // whole file read as columns of feature vectors
                                    });
                if (featdim == 0) // first time
                    featdim = feat.rows();
                else if (featdim != feat.rows())
                    RuntimeError("minibatchframesource: inconsistent feature dimension across files");
                // HVite occasionally generates mismatching output --skip such files
                if (!key.empty()) // (we have a key if supervised mode)
                {
                    const auto &labseq = labels.find(key)->second; // (we already checked above that it exists)
                    size_t labframes = labseq.empty() ? 0 : (labseq[labseq.size() - 1].firstframe + labseq[labseq.size() - 1].numframes);
                    if (abs((int) labframes - (int) feat.cols()) > 0)
                    {
                        fprintf(stderr, "\nminibatchframesource: %d-th file has small duration mismatch (%d in label vs. %d in feat file), skipping: %ls", i, (int) labframes, (int) feat.cols(), key.c_str());
                        notfound++;
                        continue; // skip this utterance at all
                    }
                }
                // append to cache
                frame.resize(featdim);
                if (feat.cols() < 2) // (2 frames needed for boundary markers)
                    RuntimeError("minibatchframesource: utterances < 2 frames not supported");
                foreach_column (t, feat)
                {
                    foreach_index (k, frame)
                        frame[k] = feat(k, t);
                    frames.push_back(frame);
                    numframes++;
                    boundaryflags.push_back((t == 0) ? -1 : (t == feat.cols() - 1) ? +1 : 0);
                }
                assert(numframes == frames.size());
                assert(numframes == boundaryflags.size());
            }

            // get label sequence
            if (!key.empty()) // (we have a key if supervised mode)
            {
                const auto &labseq = labels.find(key)->second; // (we already checked above that it exists)
                foreach_index (i2, labseq)
                {
                    const auto &e = labseq[i2];
                    if ((i2 > 0 && labseq[i2 - 1].firstframe + labseq[i2 - 1].numframes != e.firstframe) || (i2 == 0 && e.firstframe != 0))
                        RuntimeError("minibatchframesource: labels not in consecutive order MLF in label set: %ls", key.c_str());
                    for (size_t t = e.firstframe; t < e.firstframe + e.numframes; t++)
                    {
                        if (e.classid >= udim)
                            RuntimeError("minibatchframesource: class id exceeds model dimension in file %ls", key.c_str());
                        if (e.classid != (CLASSIDTYPE) e.classid)
                            RuntimeError("CLASSIDTYPE has too few bits");
                        classids.push_back((CLASSIDTYPE) e.classid);
                        numclasses = std::max(numclasses, (size_t)(1u + e.classid));
                    }
                }
                if (vdim == 0)
                    numframes = classids.size();
                if (numframes != classids.size()) // TODO: remove this once we are confident
                    RuntimeError("minibatchframesource: label duration inconsistent with feature file in MLF label set: %ls", key.c_str());
                assert(numframes == classids.size());
            }
            else
            {
                assert(classids.empty()); // that's how we detect it later
            }
        }
        assert(vdim == 0 || numframes == frames.size());
        assert(labels.empty() || numframes == classids.size());
        if ((vdim != 0 && numframes != frames.size()) || (!labels.empty() && numframes != classids.size()))
            RuntimeError("minibatchframesource: numframes variable screwup");
        fprintf(stderr, " %d frames read from %d utterances; %d classes\n", (int) numframes, (int) infiles.size(), (int) numclasses);
        if (notfound > 0)
        {
            fprintf(stderr, "minibatchframesource: %d files out of %d not found in label set\n", (int) notfound, (int) infiles.size());
            if (notfound > infiles.size() / 2)
                RuntimeError("minibatchframesource: too many files not found in label set--assuming broken configuration\n");
        }

        if (numframes == 0 && !mayhavenoframe)
            RuntimeError("minibatchframesource: no input features given!");

        // notify frames source to switch from population to consumption mode
        frames.no_more_push_back();

        // initialize randomizer
        if (numframes > 0)
            m_randomOrdering.Resize(numframes, randomizationrange);
    }
    virtual ~minibatchframesource()
    {
    }
    size_t totalframes() const
    {
        assert(vdim == 0 || numframes == frames.size());
        assert(!issupervised() || numframes == classids.size());
        return numframes;
    }

    bool issupervised() const
    {
        return !classids.empty();
    }

    void setverbosity(int newverbosity)
    {
        verbosity = newverbosity;
    }

    // retrieve one minibatch
    // Minibatches are deterministic pseudo-random samples. The entire corpus
    // is repeated infinitely, but each repetition (a 'sweep') is randomized
    // differently.
    // This function allows to retrieve a mini-batch starting from any frame
    // within this infinitely extended repetition. To the end, mini-batches are
    // specified by start frame and #frames.
    // This function returns the same data independent on #frames, i.e. the concept
    // of the mini-batch is not defined in here, but on the caller side. The caller
    // can retrieve the frames of a mini-batch in chunks that do not match the
    // caller's definition of "mini-batch," e.g. bigger or smaller chunks.
    // If a requested mini-batch spans a sweep boundary, then this function will
    // not return samples after the sweep boundary. Instead, the returned frame
    // set is shortened to not exceed the end of the sweep. The caller must make
    // a separate second call to get the rest. In trainlayer(), the one
    // sweep-boundary-spanning mini-batch will simply be shortened.
    // This function is NOT thread-safe (due to caching of random sequence).
    bool getbatch(const size_t globalts, const size_t framesrequested, msra::dbn::matrix &feat, std::vector<size_t> &uids,
                  std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> &transcripts,
                  std::vector<std::shared_ptr<const latticesource::latticepair>> &latticepairs)
    {
        auto_timer timergetbatch;

        transcripts.clear();  // word-level transcripts not supported by frame source (aimed at MMI)
        latticepairs.clear(); // neither are lattices

        assert(totalframes() > 0);
        const size_t sweep = globalts / totalframes();              // which sweep (this determines randomization)
        const size_t ts = globalts % totalframes();                 // start frame within the sweep
        const size_t te = std::min(ts + framesrequested, totalframes()); // do not go beyond sweep boundary
        assert(te > ts);
        if (verbosity >= 2)
            fprintf(stderr, "getbatch: frames [%d..%d] in sweep %d\n", (int) ts, (int) (te - 1), (int) sweep);

        // get random sequence (each time index occurs exactly once)
        // If the sweep changes, this will re-cache the sequence. We optimize for rare, monotonous sweep changes.
        const auto &tmap = m_randomOrdering(sweep);

        // page in the needed range of frames
        const size_t extent = augmentationextent(frames.dim(), vdim);
        bool readfromdisk = frames.require(m_randomOrdering.Bounds(std::max(ts, extent) - extent, te + 1 + extent));

        // generate features and uids
        feat.resize(vdim, te - ts); // note: special mode vdim == 0 means no features to be loaded
        if (issupervised())         // empty means unsupervised training -> return empty uids
            uids.resize(te - ts);
        else
            uids.clear();
        for (size_t t = ts; t < te; t++)
        {
            size_t trand = m_randomOrdering.IsRandomizationDisabled() ? t : tmap[t]; // the random-sequence sample point for this point in time
            if (vdim != 0)
            {
                auto v_t = feat.col(t - ts); // the vector to fill in
                augmentneighbors(frames, boundaryflags, trand, v_t);
            }
            if (issupervised())
                uids[t - ts] = classids[trand];
        }
        timegetbatch = timergetbatch;
        return readfromdisk;
    }

    bool getbatch(const size_t globalts, const size_t framesrequested, std::vector<msra::dbn::matrix> &feat, std::vector<std::vector<size_t>> &uids,
                  std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> &transcripts,
                  std::vector<std::shared_ptr<const latticesource::latticepair>> &latticepairs, std::vector<std::vector<size_t>> &sentendmark,
                  std::vector<std::vector<size_t>> &phoneboundaries)
    {
        // for single input/output set size to be 1 and run old getbatch
        feat.resize(1);
        uids.resize(1);
        // transcripts.resize(1);
        // latticepairs.resize(1);
        sentendmark.resize(1);
        phoneboundaries.resize(1);
        return getbatch(globalts, framesrequested, feat[0], uids[0], transcripts, latticepairs);
    }

    double gettimegetbatch()
    {
        return timegetbatch;
    }

    // return first valid globalts to ask getbatch() for
    // In frame mode, there is no constraint, i.e. it is 'globalts' itself.
    /*implement*/ size_t firstvalidglobalts(const size_t globalts)
    {
        return globalts;
    }

    /*implement*/ const std::vector<size_t> &unitcounts() const
    {
        LogicError("unitcounts: not implemented for this feature source");
        static std::vector<size_t> x;
        return x; /*keep compiler happy*/
    }
};

// ---------------------------------------------------------------------------
// minibatchframesourcemulti -- feature source to provide randomized frames in minibatches
// this is derived from minibatchframesource but worked with multiple inputs and/or outputs
// by making "frames" and "classids" a vector of vectors
// ---------------------------------------------------------------------------
class minibatchframesourcemulti : public minibatchsource
{
    std::vector<size_t> vdim;         // feature dimension after augmenting neighhors (0: don't read features)
    std::vector<size_t> leftcontext;  // number of frames to the left of the target frame in the context window
    std::vector<size_t> rightcontext; // number of frames to the right of the target frame in the context window
    unsigned int sampperiod;          // (for reference and to check against model)
    std::string featkind;
    size_t featdim;
    size_t maxvdim;
    // cache
    // std::vector<biggrowablevectorarray> frames;
    std::vector<std::unique_ptr<biggrowablevectorarray>> pframes; // [t][i] all features concatenated
    std::vector<char> boundaryflags;                         // [t] -1 for first and +1 for last frame, 0 else (for augmentneighbors())
    std::vector<std::vector<CLASSIDTYPE>> classids;          // [t] the state that the frame belongs to
    size_t numframes;                                        // total frames (==frames.size()==boundaryflags.size()==classids.size()) unless special modes vdim == 0 and/or no labels
    Microsoft::MSR::CNTK::RandomOrdering m_randomOrdering;   // [t] -> t'
    double timegetbatch;
    int verbosity;

public:
    // constructor
    // Pass empty labels to denote unsupervised training (so getbatch() will not return uids).
    minibatchframesourcemulti(const std::vector<std::vector<std::wstring>> &infiles, const std::vector<std::map<std::wstring, std::vector<msra::asr::htkmlfentry>>> &labels,
                              std::vector<size_t> vdim, std::vector<size_t> udim, std::vector<size_t> leftcontext, std::vector<size_t> rightcontext, size_t randomizationrange, const std::vector<std::wstring> &pagepath, const bool mayhavenoframe = false, int addEnergy = 0)
        : vdim(vdim), leftcontext(leftcontext), rightcontext(rightcontext), sampperiod(0), featdim(0), numframes(0), timegetbatch(0), verbosity(2), maxvdim(0)
    {

        if (vdim[0] == 0 && labels.empty())
            RuntimeError("minibatchframesourcemulti: when running without features, labels are needed");
        // at this stage, we simply page in the entire training set at once and work off RAM
        // We will benefit from feature archives indirectly through htkfeatio.
        // TODO:
        //  - infiles must specify time range
        //  - at this stage only reserve() (we know the time range; allocate second-layer structure)
        //  - implement block-wise paging directly from HTK feature files through htkfeatreader
        featkind.clear();
        std::vector<float> frame;
        std::vector<size_t> numclasses; // number of units found (actually max id +1)
        size_t notfound = 0;            // number of entries missing in MLF

        std::vector<size_t> framesaccum;

        if (infiles.size() == 0)
            RuntimeError("minibatchframesourcemulti: need at least one network input specified with features");

        if (labels.size() == 0)
            fprintf(stderr, "no MLF label files detected\n");

        foreach_index (i, infiles)
        {
            pframes.push_back(std::unique_ptr<biggrowablevectorarray>(new biggrowablevectorarray(pagepath[i])));

            if (vdim[i] > maxvdim)
                maxvdim = vdim[i];
        }

        foreach_index (i, labels)
        {
            classids.push_back(std::vector<CLASSIDTYPE>());
            numclasses.push_back(0);
        }

        fprintf(stderr, "minibatchframesourcemulti: reading %d feature sets and %d label sets...", (int) infiles.size(), (int) labels.size());

        foreach_index (m, infiles)
        {

            featdim = 0;
            numframes = 0;
            featkind.clear();
            msra::asr::htkfeatreader reader; // feature reader
            reader.AddEnergy(addEnergy);

            foreach_index (i, infiles[m]) // read each feature file in set m
            {
                if (i % (infiles[m].size() / 100 + 1) == 0)
                {
                    fprintf(stderr, ".");
                    fflush(stderr);
                }
                msra::basetypes::matrix<float> feat;
                msra::asr::htkfeatreader::parsedpath ppath(infiles[m][i]);

                // skip files for which labels don't exist (assuming bad alignment)
                std::wstring key;
                if (!labels.empty())
                {
                    if (!labels[0].empty()) // empty means unsupervised mode (don't load any)
                    {
#ifdef _WIN32
                        key = regex_replace((std::wstring) ppath, std::wregex(L"\\.[^\\.\\\\/:]*$"), std::wstring()); // delete extension (or not if none)
#else
                        key = removeExtension(ppath);
#endif
                        if (labels[0].find(key) == labels[0].end())
                        {
                            if (notfound < 5)
                                fprintf(stderr, "\nminibatchframesourcemulti: %d-th file not found in MLF label set: %ls", i, key.c_str());
                            notfound++;
                            continue; // skip this utterance at all
                        }
                    }
                }
                // get feature frames
                if (vdim[m] != 0) // (vdim == special mode to not read features at all)
                {
                    msra::util::attempt(5, [&]()
                                        {
                                            reader.read(ppath, featkind, sampperiod, feat); // whole file read as columns of feature vectors
                                        });
                    if (featdim == 0) // first time
                        featdim = feat.rows();
                    else if (featdim != feat.rows())
                        RuntimeError("minibatchframesourcemulti: inconsistent feature dimension across files");
                    // HVite occasionally generates mismatching output --skip such files
                    if (!key.empty()) // (we have a key if supervised mode)
                    {
                        const auto &labseq = labels[0].find(key)->second; // (we already checked above that it exists)
                        size_t labframes = labseq.empty() ? 0 : (labseq[labseq.size() - 1].firstframe + labseq[labseq.size() - 1].numframes);
                        if (abs((int) labframes - (int) feat.cols()) > 0)
                        {
                            fprintf(stderr, "\nminibatchframesourcemulti: %d-th file has small duration mismatch (%d in label vs. %d in feat file), skipping: %ls", i, (int) labframes, (int) feat.cols(), key.c_str());
                            notfound++;
                            continue; // skip this utterance at all
                        }
                    }
                    // append to cache
                    frame.resize(featdim);
                    if (feat.cols() < 2) // (2 frames needed for boundary markers)
                        RuntimeError("minibatchframesourcemulti: utterances < 2 frames not supported");
                    foreach_column (t, feat)
                    {
                        foreach_index (k, frame)
                            frame[k] = feat(k, t);

                        pframes[m]->push_back(frame);
                        numframes++;
                        if (m == 0)
                            boundaryflags.push_back((t == 0) ? -1 : (t == feat.cols() - 1) ? +1 : 0);
                    }
                    if (m == 0)
                        framesaccum.push_back(numframes);
                    else
                        assert(numframes == framesaccum[i]);

                    assert(numframes == pframes[m]->size());
                }
                if (m == 0)
                    assert(numframes == boundaryflags.size());

                if (m == 0) // after we get the key for this file, read all labels (only done for first feature)
                {
                    if (!key.empty())
                    {
                        foreach_index (j, labels)
                        {
                            const auto &labseq = labels[j].find(key)->second; // (we already checked above that it exists)
                            foreach_index (i2, labseq)
                            {
                                const auto &e = labseq[i2];
                                if ((i2 > 0 && labseq[i2 - 1].firstframe + labseq[i2 - 1].numframes != e.firstframe) || (i2 == 0 && e.firstframe != 0))
                                    RuntimeError("minibatchframesourcemulti: labels not in consecutive order MLF in label set: %ls", key.c_str());
                                for (size_t t = e.firstframe; t < e.firstframe + e.numframes; t++)
                                {
                                    if (e.classid >= udim[j])
                                        RuntimeError("minibatchframesourcemulti: class id exceeds model dimension in file %ls", key.c_str());
                                    if (e.classid != (CLASSIDTYPE) e.classid)
                                        RuntimeError("CLASSIDTYPE has too few bits");
                                    classids[j].push_back((CLASSIDTYPE) e.classid);
                                    numclasses[j] = std::max(numclasses[j], (size_t)(1u + e.classid));
                                }
                            }
                            if (vdim[m] == 0)
                                numframes = classids[j].size();
                            if (numframes != classids[j].size()) // TODO: remove this once we are confident
                                RuntimeError("minibatchframesourcemulti: label duration inconsistent with feature file in MLF label set: %ls", key.c_str());
                            assert(numframes == classids[j].size());
                        }
                    }
                    else
                    {
                        assert(classids.empty());
                    }
                }
            }

            assert(vdim[m] == 0 || numframes == pframes[m]->size());

            foreach_index (j, labels)
                assert(labels[j].empty() || numframes == classids[j].size());

            if (vdim[m] != 0 && numframes != pframes[m]->size()) // || (!labels.empty() && numframes != classids.size()))
                RuntimeError("\nminibatchframesource: numframes variable screwup");
            if (m == 0)
            {
                foreach_index (j, numclasses)
                    fprintf(stderr, "\nminibatchframesourcemulti: read label set %d: %d classes\n", j, (int) numclasses[j]);
            }
            fprintf(stderr, "\nminibatchframesourcemulti: feature set %d: %d frames read from %d utterances\n", m, (int) pframes[m]->size(), (int) infiles[m].size());
            if (notfound > 0)
            {
                fprintf(stderr, "minibatchframesourcemulti: %d files out of %d not found in label set\n", (int) notfound, (int) infiles[m].size());
                if (notfound > infiles[m].size() / 2)
                    RuntimeError("minibatchframesourcemulti: too many files not found in label set--assuming broken configuration\n");
            }
            // notify frames source to switch from population to consumption mode
            pframes[m]->no_more_push_back();
        }

        if (numframes == 0 && !mayhavenoframe)
            RuntimeError("minibatchframesource: no input features given!");

        // initialize randomizer
        if (numframes > 0)
            m_randomOrdering.Resize(numframes, randomizationrange);
    }
    virtual ~minibatchframesourcemulti()
    {
    }
    size_t totalframes() const
    {
        assert(maxvdim == 0 || numframes == pframes[0]->size());
        assert(!issupervised() || numframes == classids[0].size());
        return numframes;
    }

    bool issupervised() const
    {
        return !classids.empty();
    }

    void setverbosity(int newverbosity)
    {
        verbosity = newverbosity;
    }

    // retrieve one minibatch
    // Minibatches are deterministic pseudo-random samples. The entire corpus
    // is repeated infinitely, but each repetition (a 'sweep') is randomized
    // differently.
    // This function allows to retrieve a mini-batch starting from any frame
    // within this infinitely extended repetition. To the end, mini-batches are
    // specified by start frame and #frames.
    // This function returns the same data independent on #frames, i.e. the concept
    // of the mini-batch is not defined in here, but on the caller side. The caller
    // can retrieve the frames of a mini-batch in chunks that do not match the
    // caller's definition of "mini-batch," e.g. bigger or smaller chunks.
    // If a requested mini-batch spans a sweep boundary, then this function will
    // not return samples after the sweep boundary. Instead, the returned frame
    // set is shortened to not exceed the end of the sweep. The caller must make
    // a separate second call to get the rest. In trainlayer(), the one
    // sweep-boundary-spanning mini-batch will simply be shortened.
    // This function is NOT thread-safe (due to caching of random sequence).
    bool getbatch(const size_t globalts, const size_t framesrequested, std::vector<msra::dbn::matrix> &feat, std::vector<std::vector<size_t>> &uids,
                  std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> &transcripts,
                  std::vector<std::shared_ptr<const latticesource::latticepair>> &latticepairs, std::vector<std::vector<size_t>> &sentendmark,
                  std::vector<std::vector<size_t>> &phoneboundaries)
    {

        auto_timer timergetbatch;
        bool readfromdisk;
        size_t nreadfromdisk = 0;
        transcripts.clear();  // word-level transcripts not supported by frame source (aimed at MMI)
        latticepairs.clear(); // neither are lattices

        assert(totalframes() > 0);
        const size_t sweep = globalts / totalframes();              // which sweep (this determines randomization)
        const size_t ts = globalts % totalframes();                 // start frame within the sweep
        const size_t te = std::min(ts + framesrequested, totalframes()); // do not go beyond sweep boundary
        assert(te > ts);
        if (verbosity >= 2)
            fprintf(stderr, "getbatch: frames [%d..%d] in sweep %d\n", (int) ts, (int) (te - 1), (int) sweep);

        // get random sequence (each time index occurs exactly once)
        // If the sweep changes, this will re-cache the sequence. We optimize for rare, monotonous sweep changes.
        const auto &tmap = m_randomOrdering(sweep);

        feat.resize(pframes.size());
        uids.resize(classids.size());
        sentendmark.resize(classids.size());
        phoneboundaries.resize(classids.size());
        foreach_index (i, feat)
        {
            size_t leftextent, rightextent;
            // page in the needed range of frames
            if (leftcontext[i] == 0 && rightcontext[i] == 0)
            {
                leftextent = rightextent = augmentationextent(pframes[i]->dim(), vdim[i]);
            }
            else
            {
                leftextent = leftcontext[i];
                rightextent = rightcontext[i];
            }
            readfromdisk = pframes[i]->require(m_randomOrdering.Bounds(std::max(ts, leftextent) - leftextent, te + 1 + rightextent));
            // generate features and uids
            feat[i].resize(vdim[i], te - ts); // note: special mode vdim == 0 means no features to be loaded
            if (issupervised())               // empty means unsupervised training -> return empty uids
                foreach_index (j, uids)
                    uids[j].resize(te - ts);
            else
                uids.clear();

            for (size_t t = ts; t < te; t++)
            {
                size_t trand = m_randomOrdering.IsRandomizationDisabled() ? t : tmap[t]; // the random-sequence sample point for this point in time
                if (vdim[i] != 0)
                {
                    auto v_t = feat[i].col(t - ts); // the vector to fill in
                    augmentneighbors(*pframes[i], boundaryflags, trand, leftextent, rightextent, v_t);
                }
                if (i == 0)
                { // read labels for all outputs on first pass thru features. this guarantees they will be read if only one feature set but > 1 label set
                    if (issupervised())
                        foreach_index (j, uids)
                            uids[j][t - ts] = classids[j][trand];
                }
            }
            timegetbatch = timergetbatch;
            if (readfromdisk)
                nreadfromdisk++;
        }

        (nreadfromdisk == feat.size()) ? readfromdisk = true : readfromdisk = false;

        return readfromdisk;
    }

    bool getbatch(const size_t /*globalts*/, const size_t /*framesrequested*/, msra::dbn::matrix & /*feat*/, std::vector<size_t> & /*uids*/,
                  std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> & /*transcripts*/,
                  std::vector<std::shared_ptr<const latticesource::latticepair>> & /*latticepairs*/)
    {
        // should never get here
        RuntimeError("minibatchframesourcemulti: getbatch() being called for single input feature and single output feature, should use minibatchframesource instead\n");
    }

    double gettimegetbatch()
    {
        return timegetbatch;
    }

    // return first valid globalts to ask getbatch() for
    // In frame mode, there is no constraint, i.e. it is 'globalts' itself.
    /*implement*/ size_t firstvalidglobalts(const size_t globalts)
    {
        return globalts;
    }

    /*implement*/ const std::vector<size_t> &unitcounts() const
    {
        LogicError("unitcounts: not implemented for this feature source");
    }
};
};
};
