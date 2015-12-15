//
// <copyright file="chunkevalsource.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once


//#include <objbase.h>
#include "Basics.h"                  // for attempt()
#include "htkfeatio.h"                  // for reading HTK features
#include "minibatchsourcehelpers.h"
#ifndef __unix__
#include "ssematrix.h"      // TODO: why can it not be removed for Windows as well? At least needs a comment here.
#endif

#ifdef LEAKDETECT
#include <vld.h> // for memory leak detection
#endif

namespace msra { namespace dbn {

    class chunkevalsource // : public numamodelmanager
    {
        const size_t chunksize;                 // actual block size to perform computation on

        // data FIFO
        msra::dbn::matrix feat;
        std::vector<std::vector<float>> frames; // [t] all feature frames concatenated into a big block
        std::vector<char> boundaryflags;        // [t] -1 for first and +1 last frame, 0 else (for augmentneighbors())
        std::vector<size_t> numframes;          // [k] number of frames for all appended files
        std::vector<std::wstring> outpaths;     // [k] and their pathnames
        std::vector<unsigned int> sampperiods;  // [k] and sample periods (they should really all be the same...)
        size_t vdim; // input dimension
        size_t udim; // output dimension
        bool minibatchready;
        void operator=(const chunkevalsource &);
    private:
        void clear()    // empty the FIFO
        {
            frames.clear();
            boundaryflags.clear();
            numframes.clear();
            outpaths.clear();
            sampperiods.clear();
            minibatchready=false;
        }

        

        void saveandflush(msra::dbn::matrix &pred)
        {
            const size_t framesinblock = frames.size();

            // write out all files
            size_t firstframe = 0;
            foreach_index (k, numframes)
            {
                const wstring & outfile = outpaths[k];
                unsigned int sampperiod = sampperiods[k];
                size_t n = numframes[k];
                msra::files::make_intermediate_dirs (outfile);
                fprintf (stderr, "saveandflush: writing %d frames to %ls\n", (int)n, outfile.c_str());
                msra::dbn::matrixstripe thispred (pred, firstframe, n);
                // some sanity check for the data we've written
                const size_t nansinf = thispred.countnaninf();
                if (nansinf > 0)
                    fprintf (stderr, "chunkeval: %d NaNs or INF detected in '%ls' (%d frames)\n", (int) nansinf, outfile.c_str(), (int) thispred.cols());
                // save it
                msra::util::attempt (5, [&]()
                {
                    msra::asr::htkfeatwriter::write (outfile, "USER", sampperiod, thispred);
                });
                firstframe += n;
            }
            assert (firstframe == framesinblock); framesinblock;

            // and we are done --forget the FIFO content & get ready for next chunk
            clear();

        }

    public:
        chunkevalsource (size_t numinput, size_t numoutput, size_t chunksize)
            :vdim(numinput),udim(numoutput),chunksize(chunksize)
        {         
            frames.reserve (chunksize * 2);    
            feat.resize(vdim,chunksize); // initialize to size chunksize
        }

        // append data to chunk
        template<class MATRIX> void addfile (const MATRIX & feat, const string & featkind, unsigned int sampperiod, const std::wstring & outpath)
        {
            // append to frames; also expand neighbor frames
            if (feat.cols() < 2)
                RuntimeError("evaltofile: utterances < 2 frames not supported");
            foreach_column (t, feat)
            {
                std::vector<float> v (&feat(0,t), &feat(0,t) + feat.rows());
                frames.push_back (v);
                boundaryflags.push_back ((t == 0) ? -1 : (t == feat.cols() -1) ? +1 : 0);
            }

            numframes.push_back (feat.cols());
            outpaths.push_back (outpath);
            sampperiods.push_back (sampperiod);
            
        }

        void createevalminibatch()
        {
            const size_t framesinblock = frames.size();
            feat.resize(vdim, framesinblock);   // input features for whole utt (col vectors)
            // augment the features
            msra::dbn::augmentneighbors (frames, boundaryflags, 0, framesinblock, feat);
            minibatchready=true;
        }

        void writetofiles(msra::dbn::matrix &pred){ saveandflush(pred); }

        msra::dbn::matrix chunkofframes() { assert(minibatchready); return feat; }

        bool isminibatchready() { return minibatchready; }

        size_t currentchunksize() { return frames.size(); }
        void flushinput(){createevalminibatch();}
        void reset() { clear(); }

    };


    class chunkevalsourcemulti // : public numamodelmanager
    {
        const size_t chunksize;                 // actual block size to perform computation on

        // data FIFO
        std::vector<msra::dbn::matrix> feat;
        std::vector<std::vector<std::vector<float>>> framesmulti; // [t] all feature frames concatenated into a big block
        std::vector<char> boundaryflags;        // [t] -1 for first and +1 last frame, 0 else (for augmentneighbors())
        std::vector<size_t> numframes;          // [k] number of frames for all appended files
        std::vector<std::vector<std::wstring>> outpaths;     // [k] and their pathnames
        std::vector<std::vector<unsigned int>> sampperiods;  // [k] and sample periods (they should really all be the same...)
        std::vector<size_t> vdims; // input dimension
        std::vector<size_t> udims; // output dimension
        bool minibatchready;

                void operator=(const chunkevalsourcemulti &);
    private:
        void clear()    // empty the FIFO
        {
            foreach_index(i, vdims)
            {
                framesmulti[i].clear();
                outpaths[i].clear();
                sampperiods[i].clear();
            }
            boundaryflags.clear();
            numframes.clear();
            minibatchready=false;
        }

        

        void saveandflush(msra::dbn::matrix &pred, size_t index)
        {
            const size_t framesinblock = framesmulti[index].size();

            // write out all files
            size_t firstframe = 0;
            foreach_index (k, numframes)
            {
                const wstring & outfile = outpaths[index][k];
                unsigned int sampperiod = sampperiods[index][k];
                size_t n = numframes[k];
                msra::files::make_intermediate_dirs (outfile);
                fprintf (stderr, "saveandflush: writing %d frames to %ls\n", (int)n, outfile.c_str());
                msra::dbn::matrixstripe thispred (pred, firstframe, n);
                // some sanity check for the data we've written
                const size_t nansinf = thispred.countnaninf();
                if (nansinf > 0)
                    fprintf (stderr, "chunkeval: %d NaNs or INF detected in '%ls' (%d frames)\n", (int) nansinf, outfile.c_str(), (int) thispred.cols());
                // save it
                msra::util::attempt (5, [&]()
                {
                    msra::asr::htkfeatwriter::write (outfile, "USER", sampperiod, thispred);
                });
                firstframe += n;
            }
            assert (firstframe == framesinblock); framesinblock;

            // and we are done --forget the FIFO content & get ready for next chunk
            
        }

    public:
        chunkevalsourcemulti (std::vector<size_t> vdims, std::vector<size_t> udims, size_t chunksize)
            :vdims(vdims),udims(udims),chunksize(chunksize)
        {     

            foreach_index(i, vdims)
            {
                msra::dbn::matrix thisfeat;
                std::vector<std::vector<float>> frames; // [t] all feature frames concatenated into a big block
                
                frames.reserve(chunksize * 2);
                framesmulti.push_back(frames);
                //framesmulti[i].reserve (chunksize * 2);    
                
                thisfeat.resize(vdims[i], chunksize);
                feat.push_back(thisfeat);
    
                outpaths.push_back(std::vector<std::wstring>());
                sampperiods.push_back(std::vector<unsigned int>());
                //feat[i].resize(vdims[i],chunksize); // initialize to size chunksize
            }
        }

        // append data to chunk
        template<class MATRIX> void addfile (const MATRIX & feat, const string & featkind, unsigned int sampperiod, const std::wstring & outpath, size_t index)
        {
            // append to frames; also expand neighbor frames
            if (feat.cols() < 2)
                RuntimeError("evaltofile: utterances < 2 frames not supported");
            foreach_column (t, feat)
            {
                std::vector<float> v (&feat(0,t), &feat(0,t) + feat.rows());
                framesmulti[index].push_back (v);
                if (index==0)
                    boundaryflags.push_back ((t == 0) ? -1 : (t == feat.cols() -1) ? +1 : 0);
            }
            if (index==0)
                numframes.push_back (feat.cols());

            outpaths[index].push_back (outpath);
            sampperiods[index].push_back (sampperiod);
            
        }

        void createevalminibatch()
        {
            foreach_index(i, framesmulti)
            {
                const size_t framesinblock = framesmulti[i].size();
                feat[i].resize(vdims[i], framesinblock);   // input features for whole utt (col vectors)
                // augment the features
                msra::dbn::augmentneighbors (framesmulti[i], boundaryflags, 0, framesinblock, feat[i]);
            }
            minibatchready=true;
        }

        void writetofiles(msra::dbn::matrix &pred, size_t index){ saveandflush(pred, index); }

        msra::dbn::matrix chunkofframes(size_t index) { assert(minibatchready); assert(index<=feat.size()); return feat[index]; }

        bool isminibatchready() { return minibatchready; }

        size_t currentchunksize() { return framesmulti[0].size(); }
        void flushinput(){createevalminibatch();}
        void reset() { clear(); }

    };

    class FileEvalSource // : public numamodelmanager
    {
        const size_t chunksize;                 // actual block size to perform computation on

        // data FIFO
        std::vector<msra::dbn::matrix> feat;
        std::vector<std::vector<std::vector<float>>> framesMulti; // [t] all feature frames concatenated into a big block
        std::vector<char> boundaryFlags;        // [t] -1 for first and +1 last frame, 0 else (for augmentneighbors())
        std::vector<size_t> numFrames;          // [k] number of frames for all appended files
        std::vector<std::vector<unsigned int>> sampPeriods;  // [k] and sample periods (they should really all be the same...)
        std::vector<size_t> vdims; // input dimension
        std::vector<size_t> leftcontext;
        std::vector<size_t> rightcontext;
        bool minibatchReady;
        size_t minibatchSize;
        size_t frameIndex;

        void operator=(const FileEvalSource &);

    private:
        void Clear()    // empty the FIFO
        {
            foreach_index(i, vdims)
            {
                framesMulti[i].clear();
                sampPeriods[i].clear();
            }
            boundaryFlags.clear();
            numFrames.clear();
            minibatchReady=false;
            frameIndex=0;
        }

    public:
        FileEvalSource(std::vector<size_t> vdims, std::vector<size_t> leftcontext, std::vector<size_t> rightcontext, size_t chunksize) :vdims(vdims), leftcontext(leftcontext), rightcontext(rightcontext), chunksize(chunksize)
        {     
            foreach_index(i, vdims)
            {
                msra::dbn::matrix thisfeat;
                std::vector<std::vector<float>> frames; // [t] all feature frames concatenated into a big block
                
                frames.reserve(chunksize * 2);
                framesMulti.push_back(frames);
                //framesmulti[i].reserve (chunksize * 2);    
                
                thisfeat.resize(vdims[i], chunksize);
                feat.push_back(thisfeat);
    
                sampPeriods.push_back(std::vector<unsigned int>());
                //feat[i].resize(vdims[i],chunksize); // initialize to size chunksize
            }
        }

        // append data to chunk
        template<class MATRIX> void AddFile (const MATRIX & feat, const string & /*featkind*/, unsigned int sampPeriod, size_t index)
        {
            // append to frames; also expand neighbor frames
            if (feat.cols() < 2)
                RuntimeError("evaltofile: utterances < 2 frames not supported");
            foreach_column (t, feat)
            {
                std::vector<float> v (&feat(0,t), &feat(0,t) + feat.rows());
                framesMulti[index].push_back (v);
                if (index==0)
                    boundaryFlags.push_back ((t == 0) ? -1 : (t == feat.cols() -1) ? +1 : 0);
            }
            if (index==0)
                numFrames.push_back (feat.cols());

            sampPeriods[index].push_back (sampPeriod);
            
        }

        void CreateEvalMinibatch()
        {
            foreach_index(i, framesMulti)
            {
                const size_t framesInBlock = framesMulti[i].size();
                feat[i].resize(vdims[i], framesInBlock);   // input features for whole utt (col vectors)
                // augment the features
                size_t leftextent, rightextent;
                // page in the needed range of frames
                if (leftcontext[i] == 0 && rightcontext[i] == 0)
                {
                    leftextent = rightextent = augmentationextent(framesMulti[i][0].size(), vdims[i]);
                }
                else
                {
                    leftextent = leftcontext[i];
                    rightextent = rightcontext[i];
                }

                //msra::dbn::augmentneighbors(framesMulti[i], boundaryFlags, 0, leftcontext[i], rightcontext[i],)
                msra::dbn::augmentneighbors (framesMulti[i], boundaryFlags, leftextent, rightextent, 0, framesInBlock, feat[i]);
            }
            minibatchReady=true;
        }

        void SetMinibatchSize(size_t mbSize){ minibatchSize=mbSize;}
        msra::dbn::matrix ChunkOfFrames(size_t index) { assert(minibatchReady); assert(index<=feat.size()); return feat[index]; }

        bool IsMinibatchReady() { return minibatchReady; }

        size_t CurrentFileSize() { return framesMulti[0].size(); }
        void FlushInput(){CreateEvalMinibatch();}
        void Reset() { Clear(); }
    };

    
};};
