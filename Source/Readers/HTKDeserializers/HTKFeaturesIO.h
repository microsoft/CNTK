//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// HTKFeatureIO.h -- Legacy: helper for I/O of HTK feature files.
// TODO: Currently borrowed from the old reader, should be refactored.
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
#include <ReaderUtil.h>

namespace CNTK {

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

    static short swapshort(short v) noexcept
    {
        const unsigned char* b = (const unsigned char*)&v;
        return (short)((b[0] << 8) + b[1]);
    }
    static unsigned short swapunsignedshort(unsigned short v) noexcept
    {
        const unsigned char* b = (const unsigned char*)&v;
        return (unsigned short)((b[0] << 8) + b[1]);
    }
    static int swapint(int v) noexcept
    {
        const unsigned char* b = (const unsigned char*)&v;
        return (int)(((((b[0] << 8) + b[1]) << 8) + b[2]) << 8) + b[3];
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
            sampsize = (unsigned short)fgetshort(f);
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
            sampkind = (short)9; // user type
            int nRows = swapint(fgetint(f));
            int nCols = swapint(fgetint(f));
            int rawsampsize = nRows * nCols;
            sampsize = (unsigned short)rawsampsize; // features are stored as bytes;
            if (sampsize != rawsampsize)
                RuntimeError("reading idx feature cache header: sample size overflow");
        }

        void write(FILE* f)
        {
            fputint(f, nsamples);
            fputint(f, sampperiod);
            fputshort(f, (short)sampsize);
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
        static std::unordered_map<std::string, unsigned int> archivePathStringMap;
        static std::vector<std::wstring> archivePathStringVector;

        uint32_t s, e;       // first and last frame inside the archive file; (0, INT_MAX) if not given
        unsigned int archivePathIdx;
        bool isarchive;      // true if archive (range specified)
        bool isidxformat;    // support reading of features in idxformat as well (it's a hack, but different format's are not supported yet)

        friend class htkfeatreader;

    private:
        // physical path of archive file
        wstring archivepath() const
        {
            return archivePathStringVector[archivePathIdx];
        }

        static void malformed(const string& path)
        {
            RuntimeError("parsedpath: malformed path '%s'", path.c_str());
        }

    public:
        // constructor parses a=b[s,e] syntax and fills in the file
        // Can be used implicitly e.g. by passing a string to open().
        static parsedpath Parse(const string& pathParam, string& logicalPath)
        {
            const static string ubyte("-ubyte");

            const static std::vector<bool> equal = DelimiterHash({ '=' });
            const static std::vector<bool> leftBracket = DelimiterHash({ '[' });
            const static std::vector<bool> comma = DelimiterHash({ ',' });
            const static std::vector<bool> rightBracket = DelimiterHash({ ']' });

            parsedpath result;
            string archivepath;

            auto start = pathParam.data();
            auto end = start + pathParam.size();
            boost::iterator_range<const char*> token;

            start = ReadTillDelimiter(start, end, equal, token);
            logicalPath.assign(token.begin(), token.end());

            result.isidxformat = false;
            if (start == end) // no '=' detected: pass entire file (it's not an archive)
            {
                archivepath = logicalPath;
                result.s = 0;
                result.e = UINT_MAX;
                result.isarchive = false;
                // check for "-ubyte" suffix in path name => it is an idx file
                size_t pos = archivepath.size() >= ubyte.size() ? archivepath.size() - ubyte.size() : 0;
                string suffix = archivepath.substr(pos, ubyte.size());
                result.isidxformat = ubyte == suffix;
            }
            else // a=b[s,e] syntax detected
            {
                start = ReadTillDelimiter(start, end, leftBracket, token);
                archivepath.assign(token.begin(), token.end());
                if (start == end) // actually it's only a=b
                {
                    result.s = 0;
                    result.e = UINT_MAX;
                    result.isarchive = false;
                }
                else
                {
                    start = ReadTillDelimiter(start, end, comma, token);
                    if (start == end)
                        malformed(pathParam);

                    result.s = msra::strfun::toint(token.begin());
                    start = ReadTillDelimiter(start, end, rightBracket, token);
                    if (start != end && *start != '\r')
                        malformed(pathParam);

                    result.e = msra::strfun::toint(token.begin());
                    result.isarchive = true;
                }
            }

            auto iter = archivePathStringMap.find(archivepath);
            if (iter != archivePathStringMap.end())
            {
                result.archivePathIdx = iter->second;
            }
            else
            {
                result.archivePathIdx = (unsigned int)archivePathStringMap.size();
                archivePathStringMap[archivepath] = result.archivePathIdx;
                archivePathStringVector.push_back(Microsoft::MSR::CNTK::ToFixedWStringFromMultiByte(archivepath));
            }

            logicalPath = logicalPath.substr(0, logicalPath.find_last_of("."));
            return result;
        }

        // get the physical path for 'make' test
        wstring physicallocation() const
        {
            return archivepath();
        }

        // get duration in frames
        uint32_t numframes() const
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
        auto_file_ptr f2(fopenOrDie(physpath, L"rb")); // removed 'S' for now, as we mostly run local anyway, and this will speed up debugging

        // read the header (12 bytes for htk feature files)
        fileheader H;
        isidxformat = ppath.isidxformat;
        if (!isidxformat)
            H.read(f2);
        else // read header of idxfile
            H.idxRead(f2);

        // take a guess as to whether we need byte swapping or not
        bool needbyteswapping2 = ((unsigned int)swapint(H.sampperiod) < (unsigned int)H.sampperiod);
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
        auto location = /*((std::wstring)ppath).empty() ? */ppath.physicallocation() /*: (std::wstring)ppath*/;
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
                RuntimeError("open: start frame %d > end frame %d in '%ls'", (int)ppath.s, (int)ppath.e, ((wstring)ppath.physicallocation()).c_str());
            if (ppath.e >= physicalframes)
                RuntimeError("open: end frame exceeds archive's total number of frames %d in '%ls'", (int)physicalframes, ((wstring)ppath.physicallocation()).c_str());

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
            foreach_index(k, v)
                v[k] = (float)tmpByteVector[k];
        }
        else // need to decompress
        {
            // read into temp vector
            freadOrDie(tmp, featdim, f);
            if (needbyteswapping)
                msra::util::byteswap(tmp);
            // 'decompress' it
            v.resize(tmp.size());
            foreach_index(k, v)
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
    void read(const parsedpath& ppath, const string& kindstr, const unsigned int period, MATRIX& feat, bool needsExpansion = false)
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
    unsigned short classid;  // numeric state id

private:
    // verify and save data
    void setdata(size_t ts, size_t te, size_t uid)
    {
        if (te < ts)
            RuntimeError("htkmlfentry: end time below start time??");
        // save
        firstframe = (unsigned int)ts;
        numframes = (unsigned int)(te - ts);
        classid = (unsigned short)uid;
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
    void parsewithstatelist(const vector<char*>& toks, const unordered_map<std::string, size_t>& statelisthash, const double htkTimeToFrame)
    {
        size_t ts, te;
        parseframerange(toks, ts, te, htkTimeToFrame);
        auto iter = statelisthash.find(toks[2]);
        if (iter == statelisthash.end())
            RuntimeError("htkmlfentry: state %s not found in statelist", toks[2]);
        const size_t uid = iter->second; // get state index
        setdata(ts, te, uid);
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

}
