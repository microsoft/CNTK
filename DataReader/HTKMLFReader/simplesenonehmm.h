//
// <copyright file="simplesenonehmm.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// latticearchive.h -- managing lattice archives
//

#pragma once

#include "basetypes.h"
#include "fileutil.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>    // for find()
#include "simple_checked_arrays.h"

namespace msra { namespace asr {

// ===========================================================================
// simplesenonehmm -- simple senone-based CD-HMM
// ===========================================================================

class simplesenonehmm
{
public: // (TODO: better encapsulation)
    static const size_t MAXSTATES = 3;              // we use a fixed memory allocation since it's almost always 3 anyway
    struct transP;
    struct hmm
    {
        const char * name;                              // (this points into the key in the hash table to save memory)
        struct transP * transP;                         // underlying transition matrix
        unsigned char transPindex;                      // index of transP in struct transP
        unsigned char numstates;                        // number of states
        unsigned short senoneids[MAXSTATES];            // [0..numstates-1] senone indices

        const char * getname() const { return name; }   // (should be used for diagnostics only)
        size_t getsenoneid (size_t i) const { if (i < numstates) return (size_t) senoneids[i]; throw std::logic_error ("getsenoneid: out of bounds access"); }
        size_t getnumstates() const { return (size_t) numstates; }
        unsigned char gettransPindex() const { return transPindex;}
        const struct transP & gettransP() const { return *transP; }

        bool operator< (const hmm & other) const
        {
            return memcmp (this, &other, sizeof (other)) < 0;
        }
    };
    std::vector<hmm> hmms;                          // the set of HMMs
    std::unordered_map<std::string,size_t> symmap; // [name] -> index into hmms[]
    struct transP
    {
    private:
        size_t numstates;
        float loga[MAXSTATES+1][MAXSTATES+1];
        void check (int from, size_t to) const { if (from < -1 || from >= (int) numstates || to > numstates) throw std::logic_error ("transP: index out of bounds"); }
    public:
        void resize (size_t n) { if (n > MAXSTATES) throw std::runtime_error ("resize: requested transP that exceeds MAXSTATES"); numstates = n; }
        size_t getnumstates() const { return numstates; }
        // from = -1 and to = numstates are allowed, but we also allow 'from' to be size_t to avoid silly typecasts
        float &       operator() (int from,    size_t to)       { check (from, to);      return loga[from+1][to]; } // from >= -1
        const float & operator() (int from,    size_t to) const { check (from, to);      return loga[from+1][to]; } // from >= -1
        const float & operator() (size_t from, size_t to) const { check ((int)from, to); return loga[from+1][to]; } // from >= 0
        transP() : numstates (0) {}
    };
    std::vector<transP> transPs;                       // the transition matrices  --TODO: finish this
    std::hash_map<std::string,size_t> transPmap;    // [transPname] -> index into transPs[]
public:
    // get an hmm by index
    const hmm & gethmm (size_t i) const { return hmms[i]; }

    // get an hmm by name
    size_t gethmmid (const string & name) const
    {
        auto iter = symmap.find (name);
        if (iter == symmap.end())
            throw std::logic_error ("gethmm: unknown unit name: " + name);
        return iter->second;
    }

    // diagnostics: map state id to senone name
    std::vector<std::string> statenames;
    const char * getsenonename (size_t senoneid) const { return statenames[senoneid].c_str(); }

    // inverse lookup, for re-scoring the ground-truth path for sequence training
    // This may be ambiguous, but we know that for current setup, that's only the case for /sil/ and /sp/.
    std::vector<int> senoneid2transPindex;      // or -1 if ambiguous
    std::vector<int> senoneid2stateindex;       // 0..2, or -1 if ambiguous

    // construct from model files
    simplesenonehmm (const std::wstring & cdphonetyingpath, const std::wstring & statelistpath, const std::wstring & transPpath)
    {
        if (cdphonetyingpath.empty())   // no tying info specified --just leave an empty object
            return;
        fprintf (stderr, "simplesenonehmm: reading '%S', '%S', '%S'\n", cdphonetyingpath.c_str(), statelistpath.c_str(), transPpath.c_str());
        // read the state list
        vector<char> textbuffer;
        auto readstatenames = msra::files::fgetfilelines (statelistpath, textbuffer);
        foreach_index (s, readstatenames)
            statenames.push_back (readstatenames[s]);
        std::unordered_map<std::string,size_t> statemap; // [name] -> index
        statemap.rehash (readstatenames.size());
        foreach_index (i, readstatenames)
            statemap[readstatenames[i]] = i;
        // TRANSPNAME NUMSTATES (ROW_from[to])+
        msra::strfun::tokenizer toks (" \t", 5);
        auto transPlines = msra::files::fgetfilelines (transPpath, textbuffer);
        transPs.resize (transPlines.size());
        string key; key.reserve (100);
        foreach_index (i, transPlines)
        {
            toks = transPlines[i];
            if (toks.size() < 3)
                throw std::runtime_error ("simplesenonehmm: too few tokens in transP line: " + string (transPlines[i]));
            key = toks[0];  // transPname --using existing object to avoid malloc
            transPmap[key] = i;
            size_t numstates = msra::strfun::toint (toks[1]);
            if (numstates == 0)
                throw std::runtime_error ("simplesenonehmm: invalid numstates: " + string (transPlines[i]));
            auto & transP = transPs[i];
            transP.resize (numstates);
            size_t k = 2;   // index into tokens; transP values start at toks[2]
            for (int from = -1; from < (int) numstates; from++) for (size_t to = 0; to <= numstates; to++)
            {
                if (k >= toks.size())
                    throw std::runtime_error ("simplesenonehmm: not enough tokens on transP line: " + string (transPlines[i]));
                const char * sval = toks[k++];
                const double aij = msra::strfun::todouble (sval);
                if (aij > 1e-10)    // non-0
                    transP(from,to) = logf ((float) aij);   // we store log probs
                else
                    transP(from,to) = -1e30f;
            }
            if (toks.size() > k)
                throw std::runtime_error ("simplesenonehmm: unexpected garbage at endof transP line: " + string (transPlines[i]));
        }
        // allocate inverse lookup
        senoneid2transPindex.resize (readstatenames.size(), -2);
        senoneid2stateindex.resize (readstatenames.size(), -2);
        // read the cd-phone tying info
        // HMMNAME TRANSPNAME SENONENAME+
        auto lines = msra::files::fgetfilelines (cdphonetyingpath, textbuffer);
        hmms.reserve (lines.size());
        symmap.rehash (lines.size());
        // two tables: (1) name -> HMM; (2) HMM -> HMM index (uniq'ed)
        map<string,hmm> name2hmm;  // [name] -> unique HMM struct (without name)
        map<hmm,size_t> hmm2index; // [unique HMM struct] -> hmm index, hmms[i] contains full hmm
        foreach_index (i, lines)
        {
            toks = lines[i];
            if (toks.size() < 3)
                throw std::runtime_error ("simplesenonehmm: too few tokens in line: " + string (lines[i]));
            const char * hmmname = toks[0];
            const char * transPname = toks[1];
            // build the HMM structure
            hmm hmm;
            hmm.name = NULL;    // for use as key in hash tables, we keep this NULL
            // get the transP pointer
            // TODO: this becomes a hard lookup with failure
            key = transPname;   // (reuse existing memory)
            auto iter = transPmap.find (key);
            if (iter == transPmap.end())
                throw std::runtime_error ("simplesenonehmm: unknown transP name: " + string (lines[i]));
            size_t transPindex = iter->second;
            hmm.transPindex = (unsigned char) transPindex;
            hmm.transP = &transPs[transPindex];
            if (hmm.transPindex != transPindex)
                throw std::runtime_error ("simplesenonehmm: numeric overflow for transPindex field");
            // get the senones
            hmm.numstates = (unsigned char) (toks.size() - 2);    // remaining tokens
            if (hmm.numstates != transPs[transPindex].getnumstates())
                throw std::runtime_error ("simplesenonehmm: number of states mismatches that of transP: " + string (lines[i]));
            if (hmm.numstates > _countof (hmm.senoneids))
                throw std::runtime_error ("simplesenonehmm: hmm.senoneids[MAXSTATES] is too small in line: " + string (lines[i]));
            for (size_t s = 0; s < hmm.numstates; s++)
            {
                const char * senonename = toks[s+2];
                key = senonename;   // (reuse existing memory)
                auto iter = statemap.find (key);
                if (iter == statemap.end())
                    throw std::runtime_error ("simplesenonehmm: unrecognized senone name in line: " + string (lines[i]));
                hmm.senoneids[s] = (unsigned short) iter->second;
                if (hmm.getsenoneid(s) != iter->second)
                    throw std::runtime_error ("simplesenonehmm: not enough bits to store senone index in line: " + string (lines[i]));
                // inverse lookup
                if (senoneid2transPindex[hmm.senoneids[s]] == -2)   // no value yet
                    senoneid2transPindex[hmm.senoneids[s]] = hmm.transPindex;
                else if (senoneid2transPindex[hmm.senoneids[s]] != hmm.transPindex)
                    senoneid2transPindex[hmm.senoneids[s]] = -1;    // multiple inconsistent values
                if (senoneid2stateindex[hmm.senoneids[s]] == -2)
                    senoneid2stateindex[hmm.senoneids[s]] = (int) s;
                else if (senoneid2stateindex[hmm.senoneids[s]] != (int) s)
                    senoneid2stateindex[hmm.senoneids[s]] = -1;
            }
            for (size_t s = hmm.numstates; s < _countof (hmm.senoneids); s++)   // clear out the rest if needed
                hmm.senoneids[s] = USHRT_MAX;
            // add to name-to-HMM hash
            auto ir = name2hmm.insert (std::make_pair (hmmname, hmm));    // insert into hash table
            if (!ir.second) // not inserted
                throw std::runtime_error ("simplesenonehmm: duplicate unit name in line: " + string (lines[i]));
            // add to hmm-to-index hash
            // and update the actual lookup table
            size_t hmmindex = hmms.size();      // (assume it's a new entry)
            auto is = hmm2index.insert (std::make_pair (hmm, hmmindex));
            if (is.second)                      // was indeed inserted: add to hmms[]
            {
                // insert first, as this copies the name; we can then point to it
                auto it = symmap.insert (std::make_pair (hmmname, hmmindex)); // insert into hash table
                hmm.name = it.first->first.c_str(); // only use first name if multiple (the name is informative only anyway)
                hmms.push_back (hmm);
            }
            else                                // not inserted
            {
                hmmindex = is.first->second;    // use existing value
                symmap.insert (std::make_pair (hmmname, hmmindex)); // insert into hash table
            }
        }
        fprintf (stderr, "simplesenonehmm: %d units with %d unique HMMs, %d tied states, and %d trans matrices read\n",
                 (int)symmap.size(), (int)hmms.size(), (int)statemap.size(), (int)transPs.size());
    }

    // exposed so we can pass it to the lattice reader, which maps the symbol ids for us
    const std::unordered_map<std::string,size_t> & getsymmap() const { return symmap; }

    // inverse lookup --for scoring the ground-truth
    // Note: /sil/ and /sp/ will be ambiguous, so need to handle them as a special case.
    int senonetransP (size_t senoneid) const { return senoneid2transPindex[senoneid]; }
    int senonestate (size_t senoneid) const { return senoneid2stateindex[senoneid]; }
    const size_t getnumsenone () const {return senoneid2stateindex.size(); }
    const bool statebelongstohmm (const size_t senoneid, const hmm & hmm) const       // reutrn true if one of the states of this hmm == senoneid 
    {
        size_t numstates = hmm.getnumstates();
        for (size_t i = 0; i < numstates; i++)
            if (hmm.senoneids[i] == senoneid)
                return true;
        return false;
    }

};

};};
