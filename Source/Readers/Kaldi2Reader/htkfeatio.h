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
//#include <iostream>

#include "htkfeatio_utils.h"
#include "kaldi.h"

namespace msra { namespace asr {

class FeatureSection
{
public:
    std::wstring scpFile;
    std::string rx;
    std::string feature_transform;

private:
    kaldi::RandomAccessBaseFloatMatrixReader *feature_reader;
    kaldi::nnet1::Nnet nnet_transf;
    kaldi::CuMatrix<kaldi::BaseFloat> feats_transf;
    kaldi::Matrix<kaldi::BaseFloat> buf;

public:
    FeatureSection(std::wstring scpFile, std::wstring rx_file, std::wstring feature_transform)
    {
        this->scpFile = scpFile;
        this->rx = trimmed(fileToStr(toStr(rx_file)));
        this->feature_transform = toStr(feature_transform);

        feature_reader = new kaldi::RandomAccessBaseFloatMatrixReader(rx);

        // std::wcout << "Kaldi2Reader: created feature reader " << feature_reader << " [" << rx.c_str() << "]" << std::endl;

        if (this->feature_transform == "NO_FEATURE_TRANSFORM")
        {
            this->feature_transform = "";
        }

        if (!this->feature_transform.empty())
        {
            nnet_transf.Read(this->feature_transform);
        }
    }

    kaldi::Matrix<kaldi::BaseFloat> &read(std::wstring wkey)
    {
        std::string key = toStr(wkey);

        if (!feature_reader->HasKey(key))
        {
            fprintf(stderr, "Missing features for: %s", key.c_str());
            throw std::runtime_error(msra::strfun::strprintf("Missing features for: %s", key.c_str()));
        }

        const kaldi::Matrix<kaldi::BaseFloat> &value = feature_reader->Value(key);

        if (this->feature_transform.empty())
        {
            buf.Resize(value.NumRows(), value.NumCols());
            buf.CopyFromMat(value);
        }
        else
        {
            nnet_transf.Feedforward(kaldi::CuMatrix<kaldi::BaseFloat>(value), &feats_transf);
            buf.Resize(feats_transf.NumRows(), feats_transf.NumCols());
            feats_transf.CopyToMat(&buf);
        }

        return buf;
    }

    ~FeatureSection()
    {
        // std::wcout << "Kaldi2Reader: deleted feature reader " << feature_reader << std::endl;

        delete feature_reader;
    }
};

// ===========================================================================
// htkfeatio -- common base class for reading and writing HTK feature files
// ===========================================================================

class htkfeatio
{
protected:
    htkfeatio()
    {
    }

    /*
    Kaldi is row major and stores each feature as a row. Cntk is col major, but it stores each feature as a column.
    This makes it ok to copy one to the other straight-up.
    */
    template <class MATRIX>
    void copyKaldiToCntk(kaldi::Matrix<kaldi::BaseFloat> &kaldifeat, MATRIX &cntkfeat)
    {
        int num_rows = kaldifeat.NumRows();
        int num_cols = kaldifeat.NumCols();
        int src_stride = kaldifeat.Stride();

        kaldi::BaseFloat *src = kaldifeat.Data();

        int same_size = (num_rows == cntkfeat.cols()) && (num_cols == cntkfeat.rows());
        if (!same_size)
        {
            std::wcout << __FUNCTION__ << " not same size "
                       << "kaldifeat row-maj(" << num_rows << "," << num_cols << ")"
                       << "cntkfeat col-maj(" << cntkfeat.rows() << "," << cntkfeat.cols() << ")";
            exit(1);
        }

        for (int r = 0; r < num_rows; r++)
        {
            std::copy(src, src + num_cols, &cntkfeat(0, r));
            src += src_stride;
        }
    }

    template <class MATRIX>
    void copyCntkToKaldi()
    {
    }
};

// ===========================================================================
// htkfeatwriter -- write HTK feature file
// This is designed to write a single file only (no archive mode support).
// ===========================================================================

class htkfeatwriter : protected htkfeatio
{
public:
    // open the file for writing
    htkfeatwriter(std::wstring path, std::string kind, size_t dim, unsigned int period)
    {
    }

    // read an entire utterance into a matrix
    // Matrix type needs to have operator(i,j) and resize(n,m).
    // We write to a tmp file first to ensure we don't leave broken files that would confuse make mode.
    template <class MATRIX>
    static void write(const std::wstring &path, const std::string &kindstr, unsigned int period, const MATRIX &feat)
    {
        // std::wcout << __FILE__ << ":" << __FUNCTION__ << " not implemented" << std::endl;
        exit(1);
    }
    template <class T>
    static void WriteBasicType(std::ostream &os, bool binary, T t)
    {
        if (binary)
        {
            char len_c = (std::numeric_limits<T>::is_signed ? 1 : -1) * static_cast<char>(sizeof(t));
            os.put(len_c);
            os.write(reinterpret_cast<const char *>(&t), sizeof(t));
        }
        else
        {
            if (sizeof(t) == 1)
                os << static_cast<int16>(t) << " ";
            else
                os << t << " ";
        }
        if (os.fail())
        {
            throw std::runtime_error("Write failure in WriteBasicType.");
        }
    }
    template <class MATRIX>
    static void writeKaldi(const std::wstring &path, const std::string &kindstr, unsigned int period, const MATRIX &feat, const int precision)
    {
        std::string path_utf8 = Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(path));
        std::ofstream os(path_utf8.c_str());

        if (!os.good())
        {
            throw std::runtime_error("parsedpath: this mode requires an input script with start and end frames given");
        }
        size_t featdim = feat.rows();
        size_t numframes = feat.cols();
        bool binary = true;
        os << removeExtension(basename(path_utf8)) << ' ';
        os.put('\0');
        os.put('B');
        std::string my_token = (precision == 4 ? "FM" : "DM");
        // WriteToken(os, binary, my_token);
        os << my_token << " ";
        {
            int32 rows = numframes;
            int32 cols = featdim;
            WriteBasicType(os, binary, rows);
            WriteBasicType(os, binary, cols);
        }
        std::vector<float> v(featdim);
        for (size_t i = 0; i < numframes; i++)
        {
            foreach_index (k, v)
            {
                v[k] = feat(k, i);
                if (v[k] > 50)
                {
                    v[k] = -(float) log(1.0 / featdim);
                }
            }
            os.write(reinterpret_cast<const char *>(&v[0]), precision * (featdim));
        }
        os.flush();
        if (!os.good())
        {
        }

        /* std::wstring tmppath = path + L"$$"; // tmp path for make-mode compliant
        unlinkOrDie (path);             // delete if old file is already there
        // write it out
        std::vector<float> v (featdim);
        htkfeatwriter W (tmppath, kindstr, feat.rows(), period);
        for (size_t i = 0; i < numframes; i++)
        {
            foreach_index (k, v)
                v[k] = feat(k,i);
            W.write (v);
        }
        W.close (numframes);
        // rename to final destination
        // (This would only fail in strange circumstances such as accidental multiple processes writing to the same file.)
        renameOrDie (tmppath, path);*/
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
    // TODO make this nicer

public:
    // parser for complex a=b[s,e] syntax
    struct parsedpath
    {
    public:
        FeatureSection *featuresection;

    private:
        std::wstring xpath;       // original full path specification as passed to constructor (for error messages)
        std::wstring logicalpath; // sequence ID
        size_t num_frames;

        void malformed() const
        {
            throw std::runtime_error(msra::strfun::strprintf("parsedpath: malformed path '%S'", xpath.c_str()));
        }

        // consume and return up to 'delim'; remove from 'input' (we try to avoid C++0x here for VS 2008 compat)
        std::wstring consume(std::wstring &input, const wchar_t *delim)
        {
            std::vector<std::wstring> parts = msra::strfun::split(input, delim); // (not very efficient, but does not matter here)
            if (parts.size() == 1)
                input.clear(); // not found: consume to end
            else
                input = parts[1]; // found: break at delimiter
            return parts[0];
        }

    public:
        // constructor parses a=b[s,e] syntax and fills in the file
        // Can be used implicitly e.g. by passing a string to open().
        parsedpath(std::wstring xpath, FeatureSection *featuresection)
            : xpath(xpath), featuresection(featuresection)
        {
            logicalpath = consume(xpath, L" ");
            if (xpath.empty())
                malformed();

            num_frames = msra::strfun::toint(xpath);
        }

        // casting to wstring yields the logical path
        operator const std::wstring &() const
        {
            return logicalpath;
        }

        // get duration in frames
        size_t numframes() const
        {
            return num_frames;
        }
    };

public:
    htkfeatreader()
    {
    }

    // helper to create a parsed-path object
    // const auto path = parse (xpath)
    parsedpath parse(const std::wstring &xpath, FeatureSection *featuresection)
    {
        return parsedpath(xpath, featuresection);
    }

    void getinfo(const parsedpath &ppath, size_t &featdim)
    {
        kaldi::Matrix<kaldi::BaseFloat> &kaldifeat = ppath.featuresection->read(ppath);
        featdim = kaldifeat.NumCols();
    }

    // read an entire utterance into an already allocated matrix
    // Matrix type needs to have operator(i,j)
    template <class MATRIX>
    void readNoAlloc(const parsedpath &ppath, const std::string &kindstr, const unsigned int period, MATRIX &feat)
    {
        // open the file and check dimensions
        size_t numframes = ppath.numframes();

        // read vectors from file and push to our target structure
        try
        {
            kaldi::Matrix<kaldi::BaseFloat> &kaldifeat = ppath.featuresection->read(ppath);
            size_t featdim = kaldifeat.NumCols();

            if (feat.cols() != numframes || feat.rows() != featdim)
            {
                throw std::logic_error("read: stripe read called with wrong dimensions");
            }
            copyKaldiToCntk(kaldifeat, feat);

#if 0
            std::wcout << (std::wstring)ppath << std::endl;
            for (int c=0; c<10; c++) {
                for (int r=0; r<10; r++) {
                    std::wcout << feat(r, c) << " ";
                }
                std::wcout << std::endl;
            }
            exit(1);
#endif
        }
        catch (...)
        {
            throw;
        }
    }

    // read an entire utterance into a virgen, allocatable matrix
    // Matrix type needs to have operator(i,j) and resize(n,m)
    template <class MATRIX>
    void readAlloc(const parsedpath &ppath, std::string &kindstr, unsigned int &period, MATRIX &feat)
    {
        // get the file
        size_t numframes = ppath.numframes();

        // read vectors from file and push to our target structure
        try
        {
            kaldi::Matrix<kaldi::BaseFloat> &kaldifeat = ppath.featuresection->read(ppath);
            size_t featdim = kaldifeat.NumCols();

            feat.resize(featdim, numframes); // result matrix--columns are features
            copyKaldiToCntk(kaldifeat, feat);
        }
        catch (...)
        {
            throw;
        }
    }
};

struct htkmlfentry
{
    unsigned int firstframe; // range [firstframe,firstframe+numframes)
    unsigned short numframes;
    // unsigned short classid;     // numeric state id
    unsigned int classid; // numeric state id - mseltzer changed from ushort to uint for untied cd phones > 2^16

public:
    // verify and save data
    void setdata(size_t ts, size_t te, size_t uid)
    {
        if (te < ts)
            throw std::runtime_error("htkmlfentry: end time below start time??");
        // save
        firstframe = (unsigned int) ts;
        numframes = (unsigned short) (te - ts);
        classid = (unsigned int) uid;
        // check for numeric overflow
        if (firstframe != ts || firstframe + numframes != te || classid != uid)
            throw std::runtime_error("htkmlfentry: not enough bits for one of the values");
    }
};

template <class ENTRY, class WORDSEQUENCE>
class htkmlfreader : public std::map<std::wstring, std::vector<ENTRY>> // [key][i] the data
{
    std::wstring curpath;                                      // for error messages
    std::unordered_map<std::string, size_t> statelistmap; // for state <=> index

    void strtok(char *s, const char *delim, std::vector<char *> &toks)
    {
        toks.resize(0);
        char *context = nullptr;
        for (char *p = strtok_s(s, delim, &context); p; p = strtok_s(NULL, delim, &context))
            toks.push_back(p);
    }
    void malformed(std::string what)
    {
        throw std::runtime_error(msra::strfun::strprintf("htkmlfreader: %s in '%S'", what.c_str(), curpath.c_str()));
    }

    std::vector<char *> readlines(const std::wstring &path, std::vector<char> &buffer)
    {
        // load it into RAM in one huge chunk
        auto_file_ptr f(fopenOrDie(path, L"rb"));
        size_t len = filesize(f);
        buffer.reserve(len + 1);
        freadOrDie(buffer, len, f);
        buffer.push_back(0); // this makes it a proper C string

        // parse into lines
        std::vector<char *> lines;
        lines.reserve(len / 20);
        strtok(&buffer[0], "\r\n", lines);
        return lines;
    }

public:
    // return if input statename is sil state (hard code to compared first 3 chars with "sil")
    bool issilstate(const std::string &statename) const // (later use some configuration table)
    {
        return (statename.size() > 3 && statename.at(0) == 's' && statename.at(1) == 'i' && statename.at(2) == 'l');
    }

    std::vector<bool> issilstatetable; // [state index] => true if is sil state (cached)

    // return if input stateid represent sil state (by table lookup)
    bool issilstate(const size_t id) const
    {
        assert(id < issilstatetable.size());
        return issilstatetable[id];
    }

    // constructor reads multiple MLF files
    htkmlfreader(const std::vector<std::wstring> &paths, const std::set<std::wstring> &restricttokeys, const std::wstring &stateListPath = L"", const double htkTimeToFrame = 100000.0, int targets_delay = 0)
    {
        // read state list
        if (stateListPath != L"")
            readstatelist(stateListPath);

        // read MLF(s) --note: there can be multiple, so this is a loop
        foreach_index (i, paths)
            read(paths[i], restricttokeys, htkTimeToFrame, targets_delay);
    }

    // note: this function is not designed to be pretty but to be fast
    void read(const std::wstring &path, const std::set<std::wstring> &restricttokeys, const double htkTimeToFrame, int targets_delay)
    {
        fprintf(stderr, "htkmlfreader: reading MLF file %S ...", path.c_str());
        curpath = path; // for error messages only

        std::string targets_rspecifier = trimmed(fileToStr(toStr(path)));

        kaldi::SequentialPosteriorReader targets_reader(targets_rspecifier);

        while (!targets_reader.Done())
        {

            std::wstring key = toWStr(targets_reader.Key());
            const kaldi::Posterior p = targets_reader.Value();

            std::vector<ENTRY> &entries = (*this)[key];
            if (!entries.empty())
                malformed(msra::strfun::strprintf("duplicate entry '%S'", key.c_str()));

            int num_rows = p.size(); // number of labels for this utterance

            entries.resize(num_rows);

            for (int row = 0; row < num_rows; row++)
            {
                int num_cols = p.at(row).size();
                if (num_cols != 1)
                {
                    std::wcout << "num_cols != 1: " << num_cols << std::endl;
                    exit(1);
                }
                int delay_row = 0;
                if (row - targets_delay >= 0)
                {
                    delay_row = row - targets_delay;
                }

                std::pair<int32, float> pair = p.at(delay_row).at(0);
                if (pair.second != 1)
                {
                    std::wcout << "pair.second != 1: " << pair.second << std::endl;
                    exit(1);
                }

                size_t ts = row;
                size_t te = row + 1;
                size_t target = pair.first;

                if (statelistmap.size() != 0)
                {
                    std::string target_str = std::to_string(target);
                    auto iter = statelistmap.find(target_str);
                    if (iter == statelistmap.end())
                    {
                        throw std::runtime_error(msra::strfun::strprintf("kaldi htkmlfentry: state %s not found in statelist", target_str.c_str()));
                    }
                    target = iter->second;
                }
                entries[row].setdata(ts, te, target);
            }

            targets_reader.Next();
        }

        curpath.clear();
        fprintf(stderr, " total %lu entries\n", this->size());
    }

    // read state list, index is from 0
    void readstatelist(const std::wstring &stateListPath = L"")
    {
        if (stateListPath != L"")
        {
            std::vector<char> buffer; // buffer owns the characters--don't release until done
            std::vector<char *> lines = readlines(stateListPath, buffer);
            size_t index;
            issilstatetable.reserve(lines.size());
            for (index = 0; index < lines.size(); index++)
            {
                statelistmap[lines[index]] = index;
                issilstatetable.push_back(issilstate(lines[index]));
            }
            if (index != statelistmap.size())
                throw std::runtime_error(msra::strfun::strprintf("readstatelist: lines (%d) not equal to statelistmap size (%d)", index, statelistmap.size()));
            if (statelistmap.size() != issilstatetable.size())
                throw std::runtime_error(msra::strfun::strprintf("readstatelist: size of statelookuparray (%d) not equal to statelistmap size (%d)", issilstatetable.size(), statelistmap.size()));
            fprintf(stderr, "total %lu state names in state list %S\n", statelistmap.size(), stateListPath.c_str());
        }
    }

    // return state num: varify the fintune layer dim
    size_t getstatenum() const
    {
        return statelistmap.size();
    }

    size_t getstateid(std::string statename) // added by Hang Su adaptation
    {
        return statelistmap[statename];
    }
};
};
}; // namespaces
