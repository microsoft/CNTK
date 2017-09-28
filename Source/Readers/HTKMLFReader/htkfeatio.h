//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// htkfeatio.h -- helper for I/O of HTK feature files
//

#pragma once

#include "Basics.h"
#include "basetypes.h"
#include "fileutil.h"
#include "simple_checked_arrays.h"

#include <string>
#include <regex>
#include <set>
#include <unordered_map>
#include <stdint.h>
#include <limits.h>
#include <wchar.h>
#include "simplesenonehmm.h"
#include <array>
#include "minibatchsourcehelpers.h"

namespace msra { namespace asr {

using namespace std;

// ===========================================================================
// htkfeatio -- common base class for reading and writing HTK feature files
// ===========================================================================

class htkfeatio
{
protected:
    auto_file_ptr f;
    wstring physicalpath;  // path of this file
    bool needbyteswapping; // need to swap the bytes?

    string featkind;         // HTK feature-kind string
    size_t featdim;          // feature dimension
    unsigned int featperiod; // sampling period

    // note that by default we assume byte swapping (seems to be HTK default)
    htkfeatio()
        : needbyteswapping(true), featdim(0), featperiod(0)
    {
    }

    // set the feature kind variables --if already set then validate that they are the same
    // Path is only for error message.
    void setkind(string kind, size_t dim, unsigned int period, const wstring& path)
    {
        if (featkind.empty()) // not set yet: just memorize them
        {
            assert(featdim == 0 && featperiod == 0);
            featkind = kind;
            featdim = dim;
            featperiod = period;
        }
        else // set already: check if consistent
        {
            if (featkind != kind || featdim != dim || featperiod != period)
                RuntimeError("setkind: inconsistent feature kind for file '%ls'", path.c_str());
        }
    }

    static short swapshort(short v) throw()
    {
        const unsigned char* b = (const unsigned char*) &v;
        return (short) ((b[0] << 8) + b[1]);
    }
    static unsigned short swapunsignedshort(unsigned short v) throw()
    {
        const unsigned char* b = (const unsigned char*)&v;
        return (unsigned short)((b[0] << 8) + b[1]);
    }
    static int swapint(int v) throw()
    {
        const unsigned char* b = (const unsigned char*) &v;
        return (int) (((((b[0] << 8) + b[1]) << 8) + b[2]) << 8) + b[3];
    }

    struct fileheader
    {
        int nsamples;
        int sampperiod;
        unsigned short sampsize;
        short sampkind;
        void read(FILE* f)
        {
            nsamples = fgetint(f);
            sampperiod = fgetint(f);
            sampsize = (unsigned short) fgetshort(f);
            sampkind = fgetshort(f);
        }

        // read header of idx feature cach
        void idxRead(FILE* f)
        {
            int magic = swapint(fgetint(f));
            if (magic != 2051)
                RuntimeError("reading idx feature cache header: invalid magic");
            nsamples = swapint(fgetint(f));
            sampperiod = 0;
            sampkind = (short) 9; // user type
            int nRows = swapint(fgetint(f));
            int nCols = swapint(fgetint(f));
            int rawsampsize = nRows * nCols;
            sampsize = (unsigned short) rawsampsize; // features are stored as bytes;
            if (sampsize != rawsampsize)
                RuntimeError("reading idx feature cache header: sample size overflow");
        }

        void write(FILE* f)
        {
            fputint(f, nsamples);
            fputint(f, sampperiod);
            fputshort(f, (short) sampsize);
            fputshort(f, sampkind);
        }
        void byteswap()
        {
            nsamples = swapint(nsamples);
            sampperiod = swapint(sampperiod);
            sampsize = swapunsignedshort(sampsize);
            sampkind = swapshort(sampkind);
        }
    };

    static const int BASEMASK = 077;
    static const int PLP = 11;
    static const int MFCC = 6;
    static const int FBANK = 7;
    static const int USER = 9;
    static const int FESTREAM = 12;
    static const int HASENERGY = 0100;   // _E log energy included
    static const int HASNULLE = 0200;    // _N absolute energy suppressed
    static const int HASDELTA = 0400;    // _D delta coef appended
    static const int HASACCS = 01000;    // _A acceleration coefs appended
    static const int HASCOMPX = 02000;   // _C is compressed
    static const int HASZEROM = 04000;   // _Z zero meaned
    static const int HASCRCC = 010000;   // _K has CRC check
    static const int HASZEROC = 020000;  // _0 0'th Cepstra included
    static const int HASVQ = 040000;     // _V has VQ index attached
    static const int HASTHIRD = 0100000; // _T has Delta-Delta-Delta index attached
};

// ===========================================================================
// htkfeatwriter -- write HTK feature file
// This is designed to write a single file only (no archive mode support).
// ===========================================================================

class htkfeatwriter : protected htkfeatio
{
    size_t curframe;
    vector<float> tmp;

public:
    short parsekind(const string& str)
    {
        vector<string> params = msra::strfun::split(str, ";");
        if (params.empty())
            RuntimeError("parsekind: invalid param kind string");
        vector<string> parts = msra::strfun::split(params[0], "_");
        // map base kind
        short sampkind;
        string basekind = parts[0];
        if (basekind == "PLP")
            sampkind = PLP;
        else if (basekind == "MFCC")
            sampkind = MFCC;
        else if (basekind == "FBANK")
            sampkind = FBANK;
        else if (basekind == "USER")
            sampkind = USER;
        else
            RuntimeError("parsekind: unsupported param base kind");
        // map qualifiers
        for (size_t i = 1; i < parts.size(); i++)
        {
            string opt = parts[i];
            if (opt.length() != 1)
                RuntimeError("parsekind: invalid param kind string");
            switch (opt[0])
            {
            case 'E':
                sampkind |= HASENERGY;
                break;
            case 'D':
                sampkind |= HASDELTA;
                break;
            case 'N':
                sampkind |= HASNULLE;
                break;
            case 'A':
                sampkind |= HASACCS;
                break;
            case 'T':
                sampkind |= HASTHIRD;
                break;
            case 'Z':
                sampkind |= HASZEROM;
                break;
            case '0':
                sampkind |= HASZEROC;
                break;
            default:
                RuntimeError("parsekind: invalid qualifier in param kind string");
            }
        }
        return sampkind;
    }

public:
    // open the file for writing
    htkfeatwriter(wstring path, string kind, size_t dim, unsigned int period)
    {
        setkind(kind, dim, period, path);
        // write header
        fileheader H;
        H.nsamples = 0; // unknown for now, updated in close()
        H.sampperiod = period;
        const int bytesPerValue = sizeof(float); // we do not support compression for now
        size_t rawsampsize = featdim * bytesPerValue;
        H.sampsize = (unsigned short) rawsampsize;
        if (H.sampsize != rawsampsize)
            RuntimeError("htkfeatwriter: sample size overflow");
        H.sampkind = parsekind(kind);
        if (needbyteswapping)
            H.byteswap();
        f = fopenOrDie(path, L"wbS");
        H.write(f);
        curframe = 0;
    }
    // write a frame
    void write(const vector<float>& v)
    {
        if (v.size() != featdim)
            LogicError("htkfeatwriter: inconsistent feature dimension");
        if (needbyteswapping)
        {
            tmp.resize(v.size());
            foreach_index (k, v)
                tmp[k] = v[k];
            msra::util::byteswap(tmp);
            fwriteOrDie(tmp, f);
        }
        else
            fwriteOrDie(v, f);
        curframe++;
    }
    // finish
    // This updates the header.
    // BUGBUG: need to implement safe-save semantics! Otherwise won't work reliably with -make mode.
    // ... e.g. set DeleteOnClose temporarily, and clear at the end?
    void close(size_t numframes)
    {
        if (curframe != numframes)
            LogicError("htkfeatwriter: inconsistent number of frames passed to close()");
        fflushOrDie(f);
        // now implant the length field; it's at offset 0
        int nSamplesFile = (int) numframes;
        if (needbyteswapping)
            nSamplesFile = swapint(nSamplesFile);
        fseekOrDie(f, 0);
        fputint(f, nSamplesFile);
        fflushOrDie(f);
        f = NULL; // this triggers an fclose() on auto_file_ptr
    }
    // read an entire utterance into a matrix
    // Matrix type needs to have operator(i,j) and resize(n,m).
    // We write to a tmp file first to ensure we don't leave broken files that would confuse make mode.
    template <class MATRIX>
    static void write(const wstring& path, const string& kindstr, unsigned int period, const MATRIX& feat)
    {
        wstring tmppath = path + L"$$"; // tmp path for make-mode compliant
        unlinkOrDie(path);              // delete if old file is already there
        // write it out
        size_t featdim = feat.rows();
        size_t numframes = feat.cols();
        vector<float> v(featdim);
        htkfeatwriter W(tmppath, kindstr, feat.rows(), period);
#ifdef SAMPLING_EXPERIMENT
        for (size_t i = 0; i < numframes; i++)
        {
            foreach_index (k, v)
            {
                float val = feat(k, i) - logf((float) SAMPLING_EXPERIMENT);
                if (i % SAMPLING_EXPERIMENT == 0)
                    v[k] = val;
                else
                    v[k] += (float) (log(1 + exp(val - v[k]))); // log add
            }
            if (i % SAMPLING_EXPERIMENT == SAMPLING_EXPERIMENT - 1)
                W.write(v);
        }
#else
        for (size_t i = 0; i < numframes; i++)
        {
            foreach_index (k, v)
                v[k] = feat(k, i);
            W.write(v);
        }
#endif
#ifdef SAMPLING_EXPERIMENT
        W.close(numframes / SAMPLING_EXPERIMENT);
#else
        W.close(numframes);
#endif
        // rename to final destination
        // (This would only fail in strange circumstances such as accidental multiple processes writing to the same file.)
        renameOrDie(tmppath, path);
    }
};

// ===========================================================================
// htkfeatreader -- read HTK feature file, with archive support
//
// To support archives, one instance of this can (and is supposed to) be used
// repeatedly. All feat files read on the same instance are validated to have
// the same feature kind.
//
// For archives, this caches the last used file handle, in expectation that most reads
// are sequential anyway. In conjunction with a big buffer, this makes a huge difference.
// ===========================================================================

class htkfeatreader : protected htkfeatio
{
    // information on current file
    // File handle and feature type information is stored in the underlying htkfeatio object.
    size_t physicalframes; // total number of frames in physical file
    // TODO make this nicer
    bool isidxformat;           // support reading of features in idxformat as well (it's a hack, but different format's are not supported yet)
    uint64_t physicaldatastart; // byte offset of first data byte
    size_t vecbytesize;         // size of one vector in bytes

    bool addEnergy;                      // add in energy as data is read (will all have zero values)
    bool compressed;                     // is compressed to 16-bit values
    bool hascrcc;                        // need to skip crcc
    vector<float> a, b;                  // for decompression
    vector<short> tmp;                   // for decompression
    vector<unsigned char> tmpByteVector; // for decompression of idx files
    size_t curframe;                     // current # samples read so far
    size_t numframes;                    // number of samples for current logical file
    size_t energyElements;               // how many energy elements to add if addEnergy is true

public:
    // parser for complex a=b[s,e] syntax
    struct parsedpath
    {
        // Note: This is not thread-safe
        static std::unordered_map<std::wstring, unsigned int> archivePathStringMap;
        static std::vector<std::wstring> archivePathStringVector;

    protected:
        friend class htkfeatreader;
        msra::strfun::cstring logicalpath; // virtual path that this file should be understood to belong to

    private:
        unsigned int archivePathIdx;

    protected:
        // physical path of archive file
        wstring archivepath() const
        {
            return archivePathStringVector[archivePathIdx];
        }

        bool isarchive;      // true if archive (range specified)
        bool isidxformat;    // support reading of features in idxformat as well (it's a hack, but different format's are not supported yet)
        size_t s, e;         // first and last frame inside the archive file; (0, INT_MAX) if not given
        void malformed(const wstring& path) const
        {
            RuntimeError("parsedpath: malformed path '%ls'", path.c_str());
        }

        // consume and return up to 'delim'; remove from 'input' (we try to avoid C++0x here for VS 2008 compat)
        static wstring consume(wstring& input, const wchar_t* delim)
        {
            vector<wstring> parts = msra::strfun::split(input, delim); // (not very efficient, but does not matter here)
            if (parts.size() == 1)
                input.clear(); // not found: consume to end
            else
                input = parts[1]; // found: break at delimiter
            return parts[0];
        }

    public:
        // constructor parses a=b[s,e] syntax and fills in the file
        // Can be used implicitly e.g. by passing a string to open().
        parsedpath(const wstring& pathParam)
            : logicalpath("")
        {
            wstring xpath(pathParam);
            wstring archivepath;

            // parse out logical path
            wstring localLogicalpath = consume(xpath, L"=");
            isidxformat = false;
            if (xpath.empty()) // no '=' detected: pass entire file (it's not an archive)
            {
                archivepath = localLogicalpath;
                s = 0;
                e = INT_MAX;
                isarchive = false;
                // check for "-ubyte" suffix in path name => it is an idx file
                wstring ubyte(L"-ubyte");
                size_t pos = archivepath.size() >= ubyte.size() ? archivepath.size() - ubyte.size() : 0;
                wstring suffix = archivepath.substr(pos, ubyte.size());
                isidxformat = ubyte == suffix;
            }
            else // a=b[s,e] syntax detected
            {
                archivepath = consume(xpath, L"[");
                if (xpath.empty()) // actually it's only a=b
                {
                    s = 0;
                    e = INT_MAX;
                    isarchive = false;
                }
                else
                {
                    s = msra::strfun::toint(consume(xpath, L","));
                    if (xpath.empty())
                        malformed(pathParam);
                    e = msra::strfun::toint(consume(xpath, L"]"));
                    // TODO \r should be handled elsewhere; refine this
                    if (!xpath.empty() && xpath != L"\r")
                        malformed(pathParam);
                    isarchive = true;
                }
            }

            auto iter = archivePathStringMap.find(archivepath);
            if (iter != archivePathStringMap.end())
            {
                archivePathIdx = iter->second;
            }
            else
            {
                archivePathIdx = (unsigned int)archivePathStringMap.size();
                archivePathStringMap[archivepath] = archivePathIdx;
                archivePathStringVector.push_back(archivepath);
            }

            logicalpath = msra::strfun::utf8(localLogicalpath);
        }

        // get the physical path for 'make' test
        wstring physicallocation() const
        {
            return archivepath();
        }

        // Gets logical path of the utterance.
        string GetLogicalPath() const
        {
            assert(!logicalpath.empty());
            return logicalpath.substr(0, logicalpath.find_last_of("."));
        }

        // Clears logical path after parsing, in order not to duplicate it 
        // with the one stored in the corpus descriptor.
        void ClearLogicalPath()
        {
            logicalpath.clear();
            logicalpath.shrink_to_fit();
        }

        // casting to wstring yields the logical path
        operator wstring() const
        {
            return msra::strfun::utf16(logicalpath);
        }

        // get duration in frames
        size_t numframes() const
        {
            if (!isarchive)
                RuntimeError("parsedpath: this mode requires an input script with start and end frames given");
            return e - s + 1;
        }
    };

    // Make sure 'parsedpath' type has a move constructor
    static_assert(std::is_move_constructible<parsedpath>::value, "Type 'parsedpath' should be move constructible!");

private:
    // open the physical HTK file
    // This is different from the logical (virtual) path name in the case of an archive.
    void openphysical(const parsedpath& ppath)
    {
        wstring physpath = ppath.physicallocation();
        // auto_file_ptr f2 = fopenOrDie (physpath, L"rbS");
        auto_file_ptr f2(fopenOrDie(physpath, L"rb")); // removed 'S' for now, as we mostly run local anyway, and this will speed up debugging

        // read the header (12 bytes for htk feature files)
        fileheader H;
        isidxformat = ppath.isidxformat;
        if (!isidxformat)
            H.read(f2);
        else // read header of idxfile
            H.idxRead(f2);

        // take a guess as to whether we need byte swapping or not
        bool needbyteswapping2 = ((unsigned int) swapint(H.sampperiod) < (unsigned int) H.sampperiod);
        if (needbyteswapping2)
            H.byteswap();

        // interpret sampkind
        int basekind = H.sampkind & BASEMASK;
        string kind;
        switch (basekind)
        {
        case PLP:
            kind = "PLP";
            break;
        case MFCC:
            kind = "MFCC";
            break;
        case FBANK:
            kind = "FBANK";
            break;
        case USER:
            kind = "USER";
            break;
        case FESTREAM:
            kind = "USER";
            break; // we return this as USER type (with guid)
        default:
            RuntimeError("htkfeatreader:unsupported feature kind");
        }
        // add qualifiers
        if (H.sampkind & HASENERGY)
            kind += "_E";
        if (H.sampkind & HASDELTA)
            kind += "_D";
        if (H.sampkind & HASNULLE)
            kind += "_N";
        if (H.sampkind & HASACCS)
            kind += "_A";
        if (H.sampkind & HASTHIRD)
            kind += "_T";
        bool compressed2 = (H.sampkind & HASCOMPX) != 0;
        bool hascrcc2 = (H.sampkind & HASCRCC) != 0;
        if (H.sampkind & HASZEROM)
            kind += "_Z";
        if (H.sampkind & HASZEROC)
            kind += "_0";
        if (H.sampkind & HASVQ)
            RuntimeError("htkfeatreader:we do not support VQ");
        // skip additional GUID in FESTREAM features
        if (H.sampkind == FESTREAM)
        { // ... note: untested
            unsigned char guid[16];
            freadOrDie(&guid, sizeof(guid), 1, f2);
            kind += ";guid=";
            for (int i = 0; i < sizeof(guid) / sizeof(*guid); i++)
                kind += msra::strfun::strprintf("%02x", guid[i]);
        }

        // other checks
        size_t bytesPerValue = isidxformat ? 1 : (compressed2 ? sizeof(short) : sizeof(float));

        if (H.sampsize % bytesPerValue != 0)
            RuntimeError("htkfeatreader:sample size not multiple of dimension");
        size_t dim = H.sampsize / bytesPerValue;

        // read the values for decompressing
        vector<float> a2, b2;
        if (compressed2)
        {
            freadOrDie(a2, dim, f2);
            freadOrDie(b2, dim, f2);
            H.nsamples -= 4; // these are counted as 4 frames--that's the space they use
            if (needbyteswapping2)
            {
                msra::util::byteswap(a2);
                msra::util::byteswap(b2);
            }
        }

        // done: swap it in
        int64_t bytepos = fgetpos(f2);
        auto location = ((std::wstring)ppath).empty() ? ppath.physicallocation() : (std::wstring)ppath;
        setkind(kind, dim, H.sampperiod, location); // this checks consistency
        this->physicalpath.swap(physpath);
        this->physicaldatastart = bytepos;
        this->physicalframes = H.nsamples;
        this->f.swap(f2); // note: this will get the previous f2 auto-closed at the end of this function
        this->needbyteswapping = needbyteswapping2;
        this->compressed = compressed2;
        this->a.swap(a2);
        this->b.swap(b2);
        this->vecbytesize = H.sampsize;
        this->hascrcc = hascrcc2;
    }
    void close() // force close the open file --use this in case of read failure
    {
        f = NULL; // assigning a new FILE* to f will close the old FILE* if any
        physicalpath.clear();
    }

public:
    htkfeatreader()
    {
        addEnergy = false;
        energyElements = 0;
    }

    // helper to create a parsed-path object
    // const auto path = parse (xpath)
    parsedpath parse(const wstring& xpath)
    {
        return parsedpath(xpath);
    }

    // read a feature file
    // Returns number of frames in that file.
    // This understands the more complex syntax a=b[s,e] and optimizes a little
    size_t open(const parsedpath& ppath)
    {
        // do not reopen the file if it is the same; use fsetpos() instead
        if (f == NULL || ppath.physicallocation() != physicalpath)
            openphysical(ppath);

        if (ppath.isarchive) // reading a sub-range from an archive
        {
            if (ppath.s > ppath.e)
                RuntimeError("open: start frame %d > end frame %d in '%ls'", (int)ppath.s, (int)ppath.e, ((wstring)ppath).c_str());
            if (ppath.e >= physicalframes)
                RuntimeError("open: end frame exceeds archive's total number of frames %d in '%ls'", (int)physicalframes, ((wstring)ppath).c_str());

            int64_t dataoffset = physicaldatastart + ppath.s * vecbytesize;
            fsetpos(f, dataoffset); // we assume fsetpos(), which is our own, is smart to not flush the read buffer
            curframe = 0;
            numframes = ppath.e + 1 - ppath.s;
        }
        else // reading a full file
        {
            curframe = 0;
            numframes = physicalframes;
            assert(fgetpos(f) == physicaldatastart);
        }
        return numframes;
    }
    // get dimension and type information for a feature file
    // This will alter the state of this object in that it opens the file. It is efficient to read it right afterwards
    void getinfo(const parsedpath& ppath, string& featkind2, size_t& featdim2, unsigned int& featperiod2)
    {
        open(ppath);
        featkind2 = this->featkind;
        featdim2 = this->featdim;
        featperiod2 = this->featperiod;
    }

    // called to add energy as we read
    void AddEnergy(size_t energyElements2)
    {
        this->energyElements = energyElements2;
        this->addEnergy = energyElements2 != 0;
    }
    const string& getfeattype() const
    {
        return featkind;
    }
    operator bool() const
    {
        return curframe < numframes;
    }
    // read a vector from the open file
    void read(std::vector<float>& v)
    {
        if (curframe >= numframes)
            RuntimeError("htkfeatreader:attempted to read beyond end");
        if (!compressed && !isidxformat) // not compressed--the easy one
        {
            freadOrDie(v, featdim, f);
            if (needbyteswapping)
                msra::util::byteswap(v);
        }
        else if (isidxformat)
        {
            // read into temp vector
            freadOrDie(tmpByteVector, featdim, f);
            v.resize(featdim);
            foreach_index (k, v)
                v[k] = (float) tmpByteVector[k];
        }
        else // need to decompress
        {
            // read into temp vector
            freadOrDie(tmp, featdim, f);
            if (needbyteswapping)
                msra::util::byteswap(tmp);
            // 'decompress' it
            v.resize(tmp.size());
            foreach_index (k, v)
                v[k] = (tmp[k] + b[k]) / a[k];
        }
        curframe++;
    }
    // read a sequence of vectors from the open file into a range of frames [ts,te)
    template <class MATRIX>
    void read(MATRIX& feat, size_t ts, size_t te)
    {
        // read vectors from file and push to our target structure
        vector<float> v(featdim + energyElements);
        for (size_t t = ts; t < te; t++)
        {
            read(v);
            // add the energy elements (all zero) if needed
            if (addEnergy)
            {
                // we add the energy elements at the end of each section of features, (features, delta, delta-delta)
                size_t posIncrement = featdim / energyElements;
                size_t pos = posIncrement;
                for (size_t i = 0; i < energyElements; i++, pos += posIncrement)
                {
                    auto iter = v.begin() + pos + i;
                    v.insert(iter, 0.0f);
                }
            }
            foreach_index(k, v)
                feat(k, t) = v[k];
        }
    }
    // read an entire utterance into an already allocated matrix
    // Matrix type needs to have operator(i,j)
    template <class MATRIX>
    void read(const parsedpath& ppath, const string& kindstr, const unsigned int period, MATRIX& feat, bool needsExpansion=false)
    {
        // open the file and check dimensions
        size_t numframes2 = open(ppath);
        if (needsExpansion)
        {
            if (numframes2 != 1)
                throw std::logic_error("read: if doing utterance-based expansion of features (e.g. ivectors), utterance must contain 1 frame only");
            if (feat.rows() != featdim)
                throw std::logic_error("read: stripe read called with wrong dimensions");
        }
        else
        {
            if (feat.cols() != numframes2 || feat.rows() != featdim)
                LogicError("read: stripe read called with wrong dimensions");
        }
        if (kindstr != featkind || period != featperiod)
            LogicError("read: attempting to mixing different feature kinds");

        // read vectors from file and push to our target structure
        try
        {
            read(feat, 0, numframes2);
            if (needsExpansion) // copy first frame to all the frames in the stripe
            {
                for (int t = 1; t < feat.cols(); t++)
                {
                    for (int k = 0; k < feat.rows(); k++)
                    {
                        feat(k, t) = feat(k, 0);
                    }
                }
            }
        }
        catch (...)
        {
            close();
            throw;
        }
    }
    // read an entire utterance into a virgen, allocatable matrix
    // Matrix type needs to have operator(i,j) and resize(n,m)
    template <class MATRIX>
    void read(const parsedpath& ppath, string& kindstr, unsigned int& period, MATRIX& feat)
    {
        // get the file
        size_t numframes2 = open(ppath);
        feat.resize(featdim + energyElements, numframes2); // result matrix--columns are features

        // read vectors from file and push to our target structure
        try
        {
            read(feat, 0, numframes2);
        }
        catch (...)
        {
            close();
            throw;
        }

        // return file info
        kindstr = featkind;
        period = featperiod;
    }
};

struct htkmlfentry
{
    unsigned int firstframe; // range [firstframe,firstframe+numframes)
    unsigned int numframes;
    msra::dbn::CLASSIDTYPE classid;  // numeric state id
    msra::dbn::HMMIDTYPE phonestart; // numeric phone start  time

private:
    // verify and save data
    void setdata(size_t ts, size_t te, size_t uid)
    {
        if (te < ts)
            RuntimeError("htkmlfentry: end time below start time??");
        // save
        firstframe = (unsigned int) ts;
        numframes = (unsigned int) (te - ts);
        classid = (msra::dbn::CLASSIDTYPE) uid;
        // check for numeric overflow
        if (firstframe != ts || firstframe + numframes != te || classid != uid)
            RuntimeError("htkmlfentry: not enough bits for one of the values");
    }

    // parse the time range
    // There are two formats:
    //  - original HTK
    //  - Dong's hacked format: ts te senonename senoneid
    // We distinguish
    static void parseframerange(const vector<char*>& toks, size_t& ts, size_t& te, const double htkTimeToFrame)
    {
        double rts = msra::strfun::todouble(toks[0]);
        double rte = msra::strfun::todouble(toks[1]);
        // if the difference between two frames is more than htkTimeToFrame, we expect conversion to time
        if (rte - rts >= htkTimeToFrame - 1) // convert time to frame
        {
            ts = (size_t)(rts / htkTimeToFrame + 0.5); // get start frame
            te = (size_t)(rte / htkTimeToFrame + 0.5); // get end frame
        }
        else
        {
            ts = (size_t)(rts);
            te = (size_t)(rte);
        }
    }

public:
    // parse format with original HTK state align MLF format and state list
    void parsewithstatelist(const vector<char*>& toks, const unordered_map<std::string, size_t>& statelisthash, const double htkTimeToFrame,
                            std::unordered_map<std::string, size_t>& hmmnamehash)
    {
        size_t ts, te;
        parseframerange(toks, ts, te, htkTimeToFrame);
        auto iter = statelisthash.find(toks[2]);
        if (iter == statelisthash.end())
            RuntimeError("htkmlfentry: state %s not found in statelist", toks[2]);
        const size_t uid = iter->second; // get state index
        setdata(ts, te, uid);
        // phone boundary
        if (hmmnamehash.size() > 0)
        {
            if (toks.size() > 4)
            {
                auto hmmiter = hmmnamehash.find(toks[4]);
                if (hmmiter == hmmnamehash.end())
                    RuntimeError("htkmlfentry: hmm %s not found in hmmlist", toks[4]);
                phonestart = (msra::dbn::HMMIDTYPE)(hmmiter->second + 1);

                // check for numeric overflow
                if ((hmmiter->second + 1) != phonestart)
                    RuntimeError("htkmlfentry: not enough bits for one of the values");
            }
            else
                phonestart = 0;
        }
    }

    // ... note: this will be too simplistic for parsing more complex MLF formats. Fix when needed.
    // add support so that it can handle conditions where time instead of frame numer is used.
    void parse(const vector<char*>& toks, const double htkTimeToFrame)
    {
        if (toks.size() != 4)
            RuntimeError("htkmlfentry: currently we only support 4-column format");
        size_t ts, te;
        parseframerange(toks, ts, te, htkTimeToFrame);
        size_t uid = msra::strfun::toint(toks[3]);
        setdata(ts, te, uid);
    }
};

template <class ENTRY, class WORDSEQUENCE>
class htkmlfreader : public map<wstring, vector<ENTRY>> // [key][i] the data
{
    wstring curpath;                                 // for error messages
    unordered_map<std::string, size_t> statelistmap; // for state <=> index
    map<wstring, WORDSEQUENCE> wordsequences;        // [key] word sequences (if we are building word entries as well, for MMI)
    std::unordered_map<std::string, size_t> symmap;

    void strtok(char* s, const char* delim, vector<char*>& toks)
    {
        toks.resize(0);
        char* context = nullptr;
        for (char* p = strtok_s(s, delim, &context); p; p = strtok_s(NULL, delim, &context))
            toks.push_back(p);
    }
    void malformed(string what)
    {
        RuntimeError("htkmlfreader: %s in '%ls'", what.c_str(), curpath.c_str());
    }

    vector<char*> readlines(const wstring& path, vector<char>& buffer)
    {
        // load it into RAM in one huge chunk
        auto_file_ptr f(fopenOrDie(path, L"rb"));
        size_t len = filesize(f);
        buffer.reserve(len + 1);
        freadOrDie(buffer, len, f);
        buffer.push_back(0); // this makes it a proper C string

        // parse into lines
        vector<char*> lines;
        lines.reserve(len / 20);
        strtok(&buffer[0], "\r\n", lines);
        return lines;
    }

    template <typename WORDSYMBOLTABLE, typename UNITSYMBOLTABLE>
    void parseentry(const vector<std::string>& lines, size_t line, const set<wstring>& restricttokeys,
                    const WORDSYMBOLTABLE* wordmap, const UNITSYMBOLTABLE* unitmap,
                    vector<typename WORDSEQUENCE::word>& wordseqbuffer, vector<typename WORDSEQUENCE::aligninfo>& alignseqbuffer,
                    const double htkTimeToFrame)
    {
        size_t idx = 0;
        string filename = lines[idx++];
        while (filename == "#!MLF!#") // skip embedded duplicate MLF headers (so user can 'cat' MLFs)
            filename = lines[idx++];

        // some mlf file have write errors, so skip malformed entry
        if (filename.length() < 3 || filename[0] != '"' || filename[filename.length() - 1] != '"')
        {
            fprintf(stderr, "warning: filename entry (%s)\n", filename.c_str());
            fprintf(stderr, "skip current mlf entry from line (%lu) until line (%lu).\n", (unsigned long)(line + idx), (unsigned long)(line + lines.size()));
            return;
        }

        filename = filename.substr(1, filename.length() - 2); // strip quotes
        if (filename.find("*/") == 0)
            filename = filename.substr(2);
#ifdef _MSC_VER
        wstring key = msra::strfun::utf16(regex_replace(filename, regex("\\.[^\\.\\\\/:]*$"), string())); // delete extension (or not if none)
#else
        wstring key = msra::strfun::utf16(msra::dbn::removeExtension(filename)); // note that c++ 4.8 is incomplete for supporting regex
#endif

        // determine lines range
        size_t s = idx;
        size_t e = lines.size() - 1;
        // lines range: [s,e)

        // don't parse unused entries (this is supposed to be used for very small debugging setups with huge MLFs)
        if (!restricttokeys.empty() && restricttokeys.find(key) == restricttokeys.end())
            return;

        vector<ENTRY>& entries = (*this)[key]; // this creates a new entry
        if (!entries.empty())
            malformed(msra::strfun::strprintf("duplicate entry '%ls'", key.c_str()));
        entries.resize(e - s);
        wordseqbuffer.resize(0);
        alignseqbuffer.resize(0);
        vector<char*> toks;
        for (size_t i = s; i < e; i++)
        {
            // We can mutate the original string as it is no longer needed after tokenization
            strtok(const_cast<char*>(lines[i].c_str()), " \t", toks);
            if (statelistmap.size() == 0)
                entries[i - s].parse(toks, htkTimeToFrame);
            else
                entries[i - s].parsewithstatelist(toks, statelistmap, htkTimeToFrame, symmap);
            // if we also read word entries, do it here
            if (wordmap)
            {
                if (toks.size() > 6 /*word entry are in this column*/)
                {
                    const char* w = toks[6]; // the word name
                    int wid = (*wordmap)[w]; // map to word id --may be -1 for unseen words in the transcript (word list typically comes from a test LM)
                    size_t wordindex = (wid == -1) ? WORDSEQUENCE::word::unknownwordindex : (size_t) wid;
                    wordseqbuffer.push_back(typename WORDSEQUENCE::word(wordindex, entries[i - s].firstframe, alignseqbuffer.size()));
                }
                if (unitmap)
                {
                    if (toks.size() > 4)
                    {
                        const char* u = toks[4];      // the triphone name
                        auto iter = unitmap->find(u); // map to unit id
                        if (iter == unitmap->end())
                            RuntimeError("parseentry: unknown unit %s in utterance %ls", u, key.c_str());
                        const size_t uid = iter->second;
                        alignseqbuffer.push_back(typename WORDSEQUENCE::aligninfo(uid, 0 /*#frames--we accumulate*/));
                    }
                    if (alignseqbuffer.empty())
                        RuntimeError("parseentry: lonely senone entry at start without phone/word entry found, for utterance %ls", key.c_str());
                    alignseqbuffer.back().frames += entries[i - s].numframes; // (we do not have an overflow check here, but should...)
                }
            }
        }
        if (wordmap) // if reading word sequences as well (for MMI), then record it (in a separate map)
        {
            if (!entries.empty() && wordseqbuffer.empty())
                RuntimeError("parseentry: got state alignment but no word-level info, although being requested, for utterance %ls", key.c_str());
            // post-process silence
            //  - first !silence -> !sent_start
            //  - last !silence -> !sent_end
            int silence = (*wordmap)["!silence"];
            if (silence >= 0)
            {
                int sentstart = (*wordmap)["!sent_start"]; // these must have been created
                int sentend = (*wordmap)["!sent_end"];
                // map first and last !silence to !sent_start and !sent_end, respectively
                if (sentstart >= 0 && wordseqbuffer.front().wordindex == (size_t) silence)
                    wordseqbuffer.front().wordindex = sentstart;
                if (sentend >= 0 && wordseqbuffer.back().wordindex == (size_t) silence)
                    wordseqbuffer.back().wordindex = sentend;
            }
            // if (sentstart < 0 || sentend < 0 || silence < 0)
            //    LogicError("parseentry: word map must contain !silence, !sent_start, and !sent_end");
            // implant
            auto& wordsequence = wordsequences[key]; // this creates the map entry
            wordsequence.words = wordseqbuffer;      // makes a copy
            wordsequence.align = alignseqbuffer;
        }
    }

public:
    // return if input statename is sil state (hard code to compared first 3 chars with "sil")
    bool issilstate(const string& statename) const // (later use some configuration table)
    {
        return (statename.size() > 3 && statename.at(0) == 's' && statename.at(1) == 'i' && statename.at(2) == 'l');
    }

    vector<bool> issilstatetable; // [state index] => true if is sil state (cached)

    // return if input stateid represent sil state (by table lookup)
    bool issilstate(const size_t id) const
    {
        assert(id < issilstatetable.size());
        return issilstatetable[id];
    }

    struct nullmap
    {
        int operator[](const char* s) const
        {
            LogicError("nullmap: should never be used");
        }
    }; // to satisfy a template, never used... :(

    // constructor reads multiple MLF files
    htkmlfreader(const vector<wstring>& paths, const set<wstring>& restricttokeys, const wstring& stateListPath = L"", const double htkTimeToFrame = 100000.0)
    {
        // read state list
        if (stateListPath != L"")
            readstatelist(stateListPath);

        // read MLF(s) --note: there can be multiple, so this is a loop
        foreach_index (i, paths)
            read(paths[i], restricttokeys, (nullmap * /*to satisfy C++ template resolution*/) NULL, (map<string, size_t>*) NULL, htkTimeToFrame);
    }

    // alternate constructor that optionally also reads word alignments (for MMI training); triggered by providing a 'wordmap'
    // (We cannot use an optional arg in the constructor above because it interferes with the template resolution.)
    template <typename WORDSYMBOLTABLE, typename UNITSYMBOLTABLE>
    htkmlfreader(const vector<wstring>& paths, const set<wstring>& restricttokeys, const wstring& stateListPath, const WORDSYMBOLTABLE* wordmap, const UNITSYMBOLTABLE* unitmap, const double htkTimeToFrame)
    {
        // read state list
        if (stateListPath != L"")
            readstatelist(stateListPath);

        // read MLF(s) --note: there can be multiple, so this is a loop
        foreach_index (i, paths)
            read(paths[i], restricttokeys, wordmap, unitmap, htkTimeToFrame);
    }

    // phone boundary
    template <typename WORDSYMBOLTABLE, typename UNITSYMBOLTABLE>
    htkmlfreader(const vector<wstring>& paths, const set<wstring>& restricttokeys, const wstring& stateListPath, const WORDSYMBOLTABLE* wordmap, const UNITSYMBOLTABLE* unitmap,
                 const double htkTimeToFrame, const msra::asr::simplesenonehmm& hset)
    {
        if (stateListPath != L"")
            readstatelist(stateListPath);
        symmap = hset.symmap;
        foreach_index (i, paths)
            read(paths[i], restricttokeys, wordmap, unitmap, htkTimeToFrame);
    }

    // note: this function is not designed to be pretty but to be fast
    template <typename WORDSYMBOLTABLE, typename UNITSYMBOLTABLE>
    void read(const wstring& path, const set<wstring>& restricttokeys, const WORDSYMBOLTABLE* wordmap, const UNITSYMBOLTABLE* unitmap, const double htkTimeToFrame)
    {
        if (!restricttokeys.empty() && this->size() >= restricttokeys.size()) // no need to even read the file if we are there (we support multiple files)
            return;

        fprintf(stderr, "htkmlfreader: reading MLF file %ls ...", path.c_str());
        curpath = path; // for error messages only

        auto_file_ptr f(fopenOrDie(path, L"rb"));
        std::string headerLine = fgetline(f);
        if (headerLine != "#!MLF!#")
            malformed("header missing");

        // Read the file in blocks and parse MLF entries
        std::vector<typename WORDSEQUENCE::word> wordsequencebuffer;
        std::vector<typename WORDSEQUENCE::aligninfo> alignsequencebuffer;
        size_t readBlockSize = 1000000;
        std::vector<char> currBlockBuf(readBlockSize + 1);
        size_t currLineNum = 1;
        std::vector<string> currMLFLines;
        bool reachedEOF = (feof(f) != 0);
        char* nextReadPtr = currBlockBuf.data();
        size_t nextReadSize = readBlockSize;
        while (!reachedEOF)
        {
            size_t numBytesRead = fread(nextReadPtr, sizeof(char), nextReadSize, f);
            reachedEOF = (numBytesRead != nextReadSize);
            if (ferror(f))
                RuntimeError("error reading from file: %s", strerror(errno));

            // Add 0 at the end to make it a proper C string
            nextReadPtr[numBytesRead] = 0;

            // Now extract lines from the currBlockBuf and parse MLF entries
            char* context = nullptr;
            const char* delim = "\r\n";

            auto consumeMLFLine = [&](const char* mlfLine)
            {
                currLineNum++;
                currMLFLines.push_back(mlfLine);
                if ((mlfLine[0] == '.') && (mlfLine[1] == 0)) // utterance end delimiter: a single dot on a line
                {
                    if (restricttokeys.empty() || (this->size() < restricttokeys.size()))
                    {
                        parseentry(currMLFLines, currLineNum - currMLFLines.size(), restricttokeys, wordmap, unitmap, wordsequencebuffer, alignsequencebuffer, htkTimeToFrame);
                    }

                    currMLFLines.clear();
                }
            };

            char* prevLine = strtok_s(currBlockBuf.data(), delim, &context);
            for (char* currLine = strtok_s(NULL, delim, &context); currLine; currLine = strtok_s(NULL, delim, &context))
            {
                consumeMLFLine(prevLine);
                prevLine = currLine;
            }

            // The last line read from the block may be a full line or part of a line
            // We can tell by whether the terminating NULL for this line is the NULL
            // we inserted after reading from the file
            size_t prevLineLen = strlen(prevLine);
            if ((prevLine + prevLineLen) == (nextReadPtr + numBytesRead))
            {
                // This is not a full line, but just a truncated part of a line.
                // Lets copy this to the start of the currBlockBuf and read new data
                // from there on
                strcpy_s(currBlockBuf.data(), currBlockBuf.size(), prevLine);
                nextReadPtr = currBlockBuf.data() + prevLineLen;
                nextReadSize = readBlockSize - prevLineLen;
            }
            else
            {
                // A full line
                consumeMLFLine(prevLine);
                nextReadPtr = currBlockBuf.data();
                nextReadSize = readBlockSize;
            }
        }

        if (!currMLFLines.empty())
            malformed("unexpected end in mid-utterance");

        curpath.clear();
        fprintf(stderr, " total %lu entries\n", (unsigned long)this->size());
    }

    // read state list, index is from 0
    void readstatelist(const wstring& stateListPath = L"")
    {
        if (stateListPath != L"")
        {
            vector<char> buffer; // buffer owns the characters--don't release until done
            vector<char*> lines = readlines(stateListPath, buffer);
            size_t index;
            issilstatetable.reserve(lines.size());
            for (index = 0; index < lines.size(); index++)
            {
                statelistmap[lines[index]] = index;
                issilstatetable.push_back(issilstate(lines[index]));
            }
            if (index != statelistmap.size())
                RuntimeError("readstatelist: lines (%d) not equal to statelistmap size (%d)", (int) index, (int) statelistmap.size());
            if (statelistmap.size() != issilstatetable.size())
                RuntimeError("readstatelist: size of statelookuparray (%d) not equal to statelistmap size (%d)", (int) issilstatetable.size(), (int) statelistmap.size());
            fprintf(stderr, "total %lu state names in state list %ls\n", (unsigned long)statelistmap.size(), stateListPath.c_str());
        }
    }

    // return state num: varify the fintune layer dim
    size_t getstatenum() const
    {
        return statelistmap.size();
    }

    size_t getstateid(string statename) // added by Hang Su adaptation
    {
        return statelistmap[statename];
    }

    // access to word sequences
    const map<wstring, WORDSEQUENCE>& allwordtranscripts() const
    {
        return wordsequences;
    }
};
};
}; // namespaces
