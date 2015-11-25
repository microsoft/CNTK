//
// <copyright file="latticearchive.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//


#pragma once

#include "stdafx.h"
#include "basetypes.h"
#include "fileutil.h"
#include "htkfeatio.h"  // for MLF reading for numer lattices
#include "latticearchive.h"
#include "msra_mgram.h" // for MLF reading for numer lattices
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <set>
#include <hash_map>
#include <regex>

#pragma warning(disable : 4996)
namespace msra { namespace lattices {

// helper to write a symbol hash (string -> int) to a file
// File has two sections:
//  - physicalunitname     // line number is mapping, starting with 0
//  - logunitname physicalunitname   // establishes a mapping; logunitname will get the same numeric index as physicalunitname
template<class UNITMAP>
static void writeunitmap (const wstring & symlistpath, const UNITMAP & unitmap)
{
    std::vector<std::string> units;
    units.reserve (unitmap.size());
    std::vector<std::string> mappings;
    mappings.reserve (unitmap.size());
    for (auto iter = unitmap.cbegin(); iter != unitmap.cend(); iter++)  // why would 'for (auto iter : unitmap)' not work?
    {
        const std::string label = iter->first;
        const size_t unitid = iter->second;
        if (units.size() <= unitid)
            units.resize (unitid + 1);      // we grow it on demand; the result must be compact (all entries filled), we check that later
        if (!units[unitid].empty())         // many-to-one mapping: remember the unit; look it up while writing
            mappings.push_back (label);
        else
            units[unitid] = label;
    }

    auto_file_ptr flist = fopenOrDie (symlistpath, L"wb");
    // write (physical) units
    foreach_index (k, units)
    {
        if (units[k].empty())
            throw std::logic_error ("build: unitmap has gaps");
        fprintfOrDie (flist, "%s\n", units[k].c_str());
    }
    // write log-phys mappings
    foreach_index (k, mappings)
    {
        const std::string unit = mappings[k];               // logical name
        const size_t unitid = unitmap.find (unit)->second;  // get its unit id; this indexes the units array
        const std::string tounit = units[unitid];           // and get the name from tehre
        fprintfOrDie (flist, "%s %s\n", unit.c_str(), tounit.c_str());
    }
    fflushOrDie (flist);
}

// (little helper to do a map::find() with default value)
template<typename MAPTYPE, typename KEYTYPE, typename VALTYPE>
static size_t tryfind (const MAPTYPE & map, const KEYTYPE & key, VALTYPE deflt)
{
    auto iter = map.find (key);
    if (iter == map.end())
        return deflt;
    else
        return iter->second;
}

// archive format:
//  - output files of build():
//     - OUTPATH                --the resulting archive (a huge file), simple concatenation of binary blocks
//     - OUTPATH.toc            --contains keys and offsets; this is how content in archive is found
//       KEY=ARCHIVE[BYTEOFFSET]        // where ARCHIVE can be empty, meaning same as previous
//     - OUTPATH.symlist    --list of all unit names encountered, in order of numeric index used in archive (first = index 0)
//                                This file is suitable as an input to HHEd's AU command.
//  - in actual use,
//     - .toc files can be concatenated
//     - .symlist files must remain paired with the archive file
//  - for actual training, user also needs to provide, typically from an HHEd AU run:
//     - OUTPATH.tying          --map from triphone units to senone sequence by name; get full phone set from .symlist above
//       UNITNAME SENONE[2] SENONE[3] SENONE[4]
/*static*/ void archive::build (const std::vector<std::wstring> & infiles, const std::wstring & outpath,
                                const std::unordered_map<std::string,size_t> & modelsymmap,
                                const msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence> & labels,   // non-empty: build numer lattices
                                const msra::lm::CMGramLM & unigram, const msra::lm::CSymbolSet & unigramsymbols)  // for numer lattices
{
#if 0   // little unit test helper for testing the read function
    bool test = true;
    if (test)
    {
        archive a;
        a.open (outpath + L".toc");
        lattice L;
        std::hash_map<string,size_t> symmap;
        a.getlattice (L"sw2001_A_1263622500_1374610000", L, symmap);
        a.getlattice (L"sw2001_A_1391162500_1409287500", L, symmap);
        return;
    }
#endif

    const bool numermode = !labels.empty(); // if labels are passed then we shall convert the MLFs to lattices, and 'infiles' are regular keys

    const std::wstring tocpath = outpath + L".toc";
    const std::wstring symlistpath = outpath + L".symlist";

    // process all files
    std::set<std::wstring> seenkeys;        // (keep track of seen keys; throw error for duplicate keys)
    msra::files::make_intermediate_dirs (outpath);

    auto_file_ptr f = fopenOrDie (outpath, L"wb");
    auto_file_ptr ftoc = fopenOrDie (tocpath, L"wb");
    size_t brokeninputfiles = 0;
    foreach_index (i, infiles)
    {
        const std::wstring & inlatpath = infiles[i];
        fprintf (stderr, "build: processing lattice '%S'\n", inlatpath.c_str());

        // get key
        std::wstring key = regex_replace (inlatpath, wregex (L"=.*"), wstring());  // delete mapping
        key = regex_replace (key, wregex (L".*[\\\\/]"), wstring());                // delete path
        key = regex_replace (key, wregex (L"\\.[^\\.\\\\/:]*$"), wstring());        // delete extension (or not if none)
        if (!seenkeys.insert (key).second)
            throw std::runtime_error (msra::strfun::strprintf ("build: duplicate key for lattice '%S'", inlatpath.c_str()));

        // we fail all the time due to totally broken HDecode/copy process, OK if not too many files are missing
        bool latticeread = false;
        try
        {
            // fetch lattice
            lattice L;
            if (!numermode)
                L.fromhtklattice (inlatpath, modelsymmap);      // read HTK lattice
            else
                L.frommlf (key, modelsymmap, labels, unigram, unigramsymbols);       // read MLF into a numerator lattice
            latticeread = true;

            // write to archive
            uint64_t offset = fgetpos (f);
            L.fwrite (f);
            fflushOrDie (f);

            // write reference to TOC file   --note: TOC file is a headerless UTF8 file; so don't use fprintf %S format (default code page)
            fprintfOrDie (ftoc, "%s=%s[%llu]\n", msra::strfun::utf8 (key).c_str(), ((i - brokeninputfiles) == 0) ? msra::strfun::utf8 (outpath).c_str() : "", offset);
            fflushOrDie (ftoc);

            fprintf (stderr, "written lattice to offset %llu as '%S'\n", offset, key.c_str());
        }
        catch (const exception & e)
        {
            if (latticeread) throw;        // write failure
            // we ignore read failures
            fprintf (stderr, "ERROR: skipping unreadable lattice '%S': %s\n", inlatpath.c_str(), e.what());
            brokeninputfiles++;
        }
    }

    // write out the unit map
    // TODO: This is sort of redundant now--it gets the symmap from the HMM, i.e. always the same for all archives.
    writeunitmap (symlistpath, modelsymmap);

    fprintf (stderr, "completed %lu out of %lu lattices (%lu read failures, %.1f%%)\n", infiles.size(), infiles.size()-brokeninputfiles, brokeninputfiles, 100.0f * brokeninputfiles / infiles.size());
}

// helper to set a context value (left, right) with checking of uniqueness
void lattice::nodecontext::setcontext (int & lr, int val)
{
    if (lr == unknown)
        lr = val;
    else if (lr != val)
        lr = (signed short) ambiguous;
}

// helper for merge() to determine the unique node contexts
vector<lattice::nodecontext> lattice::determinenodecontexts (const msra::asr::simplesenonehmm & hset) const
{
    const size_t spunit = tryfind (hset.getsymmap(), "sp", SIZE_MAX);
    const size_t silunit = tryfind (hset.getsymmap(), "sil", SIZE_MAX);
    vector<lattice::nodecontext> nodecontexts (nodes.size());
    nodecontexts.front().left = nodecontext::start;
    nodecontexts.front().right = nodecontext::ambiguous;    // (should not happen, but won't harm either)
    nodecontexts.back().right = nodecontext::end;
    nodecontexts.back().left = nodecontext::ambiguous;      // (should not happen--we require !sent_end; but who knows)
    size_t multispseen = 0;                                 // bad entries with multi-sp
    foreach_index (j, edges)
    {
        const auto & e = edges[j];
        const size_t S = e.S;
        const size_t E = e.E;
        auto a = getaligninfo (j);
        if (a.size() == 0)  // !NULL edge
            throw std::logic_error ("determinenodecontexts: !NULL edges not allowed in merging, should be removed before");
        size_t A = a[0].unit;
        size_t Z = a[a.size()-1].unit;
        if (Z == spunit)
        {
            if (a.size() < 2)
                throw std::runtime_error ("determinenodecontexts: context-free unit (/sp/) found as a single-phone word");
            else
            {
                Z = a[a.size()-2].unit;
                if (Z == spunit)        // a bugg lattice --I got this from HVite, to be tracked down
                {
                    // search from end once again, to print a warning
                    int n;
                    for (n = (int) a.size() -1; n >= 0; n--)
                        if (a[n].unit != spunit)
                            break;
                    // ends with n = position of furthest non-sp
                    if (n < 0)  // only sp?
                        throw std::runtime_error ("determinenodecontexts: word consists only of /sp/");
                    fprintf (stderr, "determinenodecontexts: word with %lu /sp/ at the end found, edge %d\n", a.size() -1 - n, j);
                    multispseen++;
                    Z = a[n].unit;
                }
            }
        }
        if (A == spunit || Z == spunit)
        {
#if 0
            fprintf (stderr, "A=%d   Z=%d   fa=%d   j=%d/N=%d    L=%d  n=%d   totalalign=%d  ts/te=%d/%d\n", (int) A, (int) Z, (int) e.firstalign,(int) j, (int) edges.size(), (int) nodes.size(), (int) a.size(), (int) align.size(),
                    nodes[S].t, nodes[E].t);
            foreach_index (kk, a)
                fprintf (stderr, "a[%d] = %d\n", kk, a[kk].unit);
            dump (stderr, [&] (size_t i) { return hset.gethmm (i).getname(); });
#endif
            throw std::runtime_error ("determinenodecontexts: context-free unit (/sp/) found as a start phone or second last phone");
        }
        const auto & Ahmm = hset.gethmm (A);
        const auto & Zhmm = hset.gethmm (Z);
        int Aid = (int) Ahmm.gettransPindex();
        int Zid = (int) Zhmm.gettransPindex();
        nodecontexts[S].setright (Aid);
        nodecontexts[E].setleft (Zid);
    }
    if (multispseen > 0)
        fprintf (stderr, "determinenodecontexts: %lu broken edges in %lu with multiple /sp/ at the end seen\n", multispseen, edges.size());
    // check CI conditions and put in 't'
    // We make the hard assumption that there is only one CI phone, /sil/.
    const auto & silhmm = hset.gethmm (silunit);
    int silid = silhmm.gettransPindex();
    foreach_index (i, nodecontexts)
    {
        auto & nc = nodecontexts[i];
        if ((nc.left == nodecontext::unknown) ^ (nc.right == nodecontext::unknown))
            throw std::runtime_error ("determinenodecontexts: invalid dead-end node in lattice");
        if (nc.left == nodecontext::ambiguous && nc.right != silid && nc.right != nodecontext::end)
            throw std::runtime_error ("determinenodecontexts: invalid ambiguous left context (right context is not CI)");
        if (nc.right == nodecontext::ambiguous && nc.left != silid && nc.left != nodecontext::start)
            throw std::runtime_error ("determinenodecontexts: invalid ambiguous right context (left context is not CI)");
        nc.t = nodes[i].t;
    }
    return nodecontexts;    // (will this use a move constructor??)
}

// compar function for sorting and merging
bool lattice::nodecontext::operator< (const nodecontext & other) const
{
    // sort by t, left, right, i  --sort by i to make i appear before iother, as assumed in merge function
    int diff = (int) t - (int) other.t;
    if (diff == 0)
    {
        diff = left - other.left;
        if (diff == 0)
        {
            diff = right - other.right;
            if (diff == 0)
                return i < other.i; // (cannot use 'diff=' pattern since unsigned but may be SIZE_MAX)
        }
    }
    return diff < 0;
}

// remove that final !NULL edge
// We have that in HAPI lattices, but there can be only one at the end.
void lattice::removefinalnull()
{
    const auto & lastedge = edges.back();
    // last edge can be !NULL, recognized as having 0 alignment records
    if (lastedge.firstalign < align.size()) // has alignment records --not !NULL
        return;
    if (lastedge.S != nodes.size() -2 || lastedge.E != nodes.size() -1)
        throw std::runtime_error ("removefinalnull: malformed final !NULL edge");
    edges.resize (edges.size() -1); // remove it
    nodes.resize (nodes.size() -1); // its start node is now the new end node
    foreach_index (j, edges)
        if (edges[j].E >= nodes.size())
            throw std::runtime_error ("removefinalnull: cannot have final !NULL edge and other edges connecting to end node at the same time");
}

// merge a secondary lattice into the first
// With lots of caveats:
//  - this optimizes lattices to true unigram lattices where the only unique node condition is acoustic context
//  - no !NULL edge at the end, call removefinalnull() before
//  - this function returns an unsorted edges[] array, i.e. invalid. We sort in uniq'ed representation, which is easier.
// This function is not elegant at all, just hard labor!
void lattice::merge (const lattice & other, const msra::asr::simplesenonehmm & hset)
{
    if (!edges2.empty() || !other.edges2.empty())
        throw std::logic_error ("merge: lattice(s) must be in non-uniq'ed format (V1)");
    if (!info.numframes || !other.info.numframes)
        throw std::logic_error ("merge: lattice(s) must have identical number of frames");

    // establish node contexts
    auto contexts = determinenodecontexts (hset);
    auto othercontexts = other.determinenodecontexts (hset);

    // create joint node space and node mapping
    // This also collapses non-unique nodes.
    // Note the edge case sil-sil in one lattice which may be sil-ambiguous or ambiguous-sil on the other.
    // We ignore this, keeping such nodes unmerged. That's OK since middle /sil/ words have zero LM, and thus it's OK to keep them non-connected.
    foreach_index (i, contexts) contexts[i].i = i;
    foreach_index (i, othercontexts) othercontexts[i].iother = i;
    contexts.insert (contexts.end(), othercontexts.begin(), othercontexts.end());   // append othercontext
    sort (contexts.begin(), contexts.end());
    vector<size_t> nodemap (nodes.size(), SIZE_MAX);
    vector<size_t> othernodemap (other.nodes.size(), SIZE_MAX);
    int j = 0;
    foreach_index (i, contexts)     // merge identical nodes  --this is the critical step
    {
        if (j == 0 || contexts[j-1].t != contexts[i].t || contexts[j-1].left != contexts[i].left || contexts[j-1].right != contexts[i].right)
            contexts[j++] = contexts[i];            // entered a new one
        // node map
        if (contexts[i].i != SIZE_MAX)
            nodemap[contexts[i].i] = j-1;
        if (contexts[i].iother != SIZE_MAX)
            othernodemap[contexts[i].iother] = j-1;
    }
    fprintf (stderr, "merge: joint node space uniq'ed to %d from %d\n", j, contexts.size());
    contexts.resize (j);

    // create a new node array (just copy the contexts[].t fields)
    nodes.resize (contexts.size());
    foreach_index (inew, nodes)
        nodes[inew].t = (unsigned short) contexts[inew].t;
    info.numnodes = nodes.size();

    // incorporate the alignment records
    const size_t alignoffset = align.size();
    align.insert (align.end(), other.align.begin(), other.align.end());

    // map existing edges' S and E fields, and also 'firstalign'
    foreach_index (j, edges)
    {
        edges[j].S = nodemap[edges[j].S];
        edges[j].E = nodemap[edges[j].E];
    }
    auto otheredges = other.edges;
    foreach_index (j, otheredges)
    {
        otheredges[j].S = othernodemap[otheredges[j].S];
        otheredges[j].E = othernodemap[otheredges[j].E];
        otheredges[j].firstalign += alignoffset;    // that's where they are now
    }

    // at this point, a new 'nodes' array exists, and the edges already are w.r.t. the new node space and align space

    // now we are read to merge 'other' edges into this, simply by concatenation
    edges.insert (edges.end(), otheredges.begin(), otheredges.end());

    // remove acoustic scores --they are likely not identical if they come from different decoders
    // If we don't do that, this will break the sorting in builduniquealignments()
    info.hasacscores = 0;
    foreach_index (j, edges)
        edges[j].a = 0.0f;

    // Note: we have NOT sorted or de-duplicated yet. That is best done after conversion to the uniq'ed format.
}

// remove duplicates
// This must be called in uniq'ed format.
void lattice::dedup()
{
    if (edges2.empty())
        throw std::logic_error ("dedup: lattice must be in uniq'ed format (V2)");

    size_t k = 0;
    foreach_index (j, edges2)
    {
        if (k > 0 && edges2[k-1].S == edges2[j].S && edges2[k-1].E == edges2[j].E && edges2[k-1].firstalign == edges2[j].firstalign)
        {
            if (edges2[k-1].implysp != edges2[j].implysp)
                throw std::logic_error ("dedup: inconsistent 'implysp' flag for otherwise identical edges");
            continue;
        }
        edges2[k++] = edges2[j];
    }
    fprintf (stderr, "dedup: edges reduced to %d from %d\n", k, edges2.size());
    edges2.resize (k);
    info.numedges = edges2.size();
    edges.clear();  // (should already be, but isn't; make sure we no longer use it)
}

// load all lattices from a TOC file and write them to a new archive
// Use this to
//  - upgrade the file format to latest in case of format changes
//  - check consistency (read only; don't write out)
//  - dump to stdout
//  - merge two lattices (for merging numer into denom lattices)
// Input path is an actual TOC path, output is the stem (.TOC will be added). --yes, not nice, maybe fix it later
// Example command:
// convertlatticearchive --latticetocs dummy c:\smbrdebug\sw20_small.den.lats.toc.10 -w c:\smbrdebug\sw20_small.den.lats.converted --cdphonetying c:\smbrdebug\combined.tying --statelist c:\smbrdebug\swb300h.9304.aligned.statelist --transprobs c:\smbrdebug\MMF.9304.transprobs
// How to regenerate from my test lattices:
// buildlatticearchive c:\smbrdebug\sw20_small.den.lats.regenerated c:\smbrdebug\hvitelat\*lat
// We support two special output path syntaxs:
//  - empty ("") -> don't output, just check the format
//  - dash ("-") -> dump lattice to stdout instead
/*static*/ void archive::convert (const std::wstring & intocpath, const std::wstring & intocpath2, const std::wstring & outpath,
                                  const msra::asr::simplesenonehmm & hset)
{
    const auto & modelsymmap = hset.getsymmap();

    const std::wstring tocpath = outpath + L".toc";
    const std::wstring symlistpath = outpath + L".symlist";

    // open input archive
    // TODO: I find that HVite emits redundant physical triphones, and even HHEd seems so (in .tying file).
    //  Thus, we should uniq the units before sorting. We can do that here if we have the .tying file.
    //  And then use the modelsymmap to map them down.
    //  Do this directly in the hset module (it will be transparent).
    std::vector<std::wstring> intocpaths (1, intocpath);            // set of paths consisting of 1
    msra::lattices::archive archive (intocpaths, modelsymmap);

    // secondary archive for optional merging operation
    const bool mergemode = !intocpath2.empty();                     // true if merging two lattices
    std::vector<std::wstring> intocpaths2;
    if (mergemode)
        intocpaths2.push_back (intocpath2);
    msra::lattices::archive archive2 (intocpaths2, modelsymmap);    // (if no merging then this archive2 is empty)

    // read the intocpath file once again to get the keys in original order
    std::vector<char> textbuffer;
    auto toclines = msra::files::fgetfilelines (intocpath, textbuffer);

    auto_file_ptr f = NULL;
    auto_file_ptr ftoc = NULL;

    // process all files
    if (outpath != L"" && outpath != L"-")  // test for special syntaxes that bypass to actually create an output archive
    {
        msra::files::make_intermediate_dirs (outpath);
        f = fopenOrDie (outpath, L"wb");
        ftoc = fopenOrDie (tocpath, L"wb");
    }
    vector<const char *> invmodelsymmap;    // only used for dump() mode

    // we must parse the toc file once again to get the keys in original order
    size_t skippedmerges = 0;
    foreach_index (i, toclines)
    {
        const char * line = toclines[i];
        const char * p = strchr (line, '=');
        if (p == NULL)
            throw std::runtime_error ("open: invalid TOC line (no = sign): " + std::string (line));
        const std::wstring key = msra::strfun::utf16 (std::string (line, p - line));

        fprintf (stderr, "convert: processing lattice '%S'\n", key.c_str());

        // fetch lattice  --this performs any necessary format conversions already
        lattice L;
        archive.getlattice (key, L);

        lattice L2;
        if (mergemode)
        {
            if (!archive2.haslattice (key))
            {
                fprintf (stderr, "convert: cannot merge because lattice '%S' missing in secondary archive; skipping\n", key.c_str());
                skippedmerges++;
                continue;
            }
            archive2.getlattice (key, L2);

            // merge it in
            // This will connect each node with matching 1-phone context conditions; aimed at merging numer lattices.
            L.removefinalnull();    // get rid of that final !NULL headache
            L2.removefinalnull();
            L.merge (L2, hset);
            // note: we are left with dups due to true unigram merging (HTK lattices cannot represent true unigram lattices since id is on the nodes)
        }
        //L.removefinalnull();
        //L.determinenodecontexts (hset);

        // convert it  --TODO: once we permanently use the new format, do this in fread() for V1
        // Note: Merging may have left this in unsorted format; we need to be robust against that.
        const size_t spunit = tryfind (modelsymmap, "sp", SIZE_MAX);
        L.builduniquealignments (spunit);

        if (mergemode)
            L.dedup();

        if (f && ftoc)
        {
            // write to archive
            uint64_t offset = fgetpos (f);
            L.fwrite (f);
            fflushOrDie (f);
            
            // write reference to TOC file   --note: TOC file is a headerless UTF8 file; so don't use fprintf %S format (default code page)
            fprintfOrDie (ftoc, "%s=%s[%llu]\n", msra::strfun::utf8 (key).c_str(), (i == 0) ? msra::strfun::utf8 (outpath).c_str() : "", offset);
            fflushOrDie (ftoc);

            fprintf (stderr, "written converted lattice to offset %llu as '%S'\n", offset, key.c_str());
        }
        else if (outpath == L"-")
        {
            if (invmodelsymmap.empty()) // build this lazily
            {
                invmodelsymmap.resize (modelsymmap.size());
                for (auto iter = modelsymmap.begin(); iter != modelsymmap.end(); iter++)
                    invmodelsymmap[iter->second] = iter->first.c_str();
            }
            L.rebuildedges (false);
            L.dump (stdout, [&] (size_t i) { return invmodelsymmap[i]; } );
        }
    }   // end for (toclines)
    if (skippedmerges > 0)
        fprintf (stderr, "convert: %d out of %d merge operations skipped due to secondary lattice missing\n", skippedmerges, toclines.size());

    // write out the updated unit map
    if (f && ftoc)
        writeunitmap (symlistpath, modelsymmap);

    fprintf (stderr, "converted %d lattices\n", toclines.size());
}

// ---------------------------------------------------------------------------
// reading lattices from external formats (HTK lat, MLF)
// ---------------------------------------------------------------------------

// read an HTK lattice
// The lattice is expected to be freshly constructed (I did not bother to check).
void lattice::fromhtklattice (const wstring & path, const std::unordered_map<std::string,size_t> & unitmap)
{
    vector<char> textbuffer;
    auto lines = msra::files::fgetfilelines (path, textbuffer);
    if (lines.empty())
                throw std::runtime_error ("lattice: mal-formed lattice--empty input file (or all-zeroes)");
    auto iter = lines.begin();
    // parse out LMF and WP
    char dummychar = 0;     // dummy for sscanf() end checking
    for ( ; iter != lines.end() && strncmp (*iter, "N=", 2); iter++)
    {
        if (strncmp (*iter, "lmscale=", 8) == 0)    // note: HTK sometimes generates extra garbage space at the end of this line
            if (sscanf_s (*iter, "lmscale=%f wdpenalty=%f%c", &info.lmf, &info.wp, &dummychar, sizeof (dummychar)) != 2 && dummychar != ' ')
                throw std::runtime_error ("lattice: mal-formed lmscale/wdpenalty line in lattice: " + string (*iter));
    }
    
    // parse N and L
    if (iter != lines.end())
    {
        unsigned long N, L;
        if (sscanf_s (*iter, "N=%lu L=%lu %c", &N, &L, &dummychar, sizeof (dummychar)) != 2)
            throw std::runtime_error ("lattice: mal-formed N=/L= line in lattice: " + string (*iter));
        info.numnodes = N;
        info.numedges = L;
        iter++;
    }
    else
        throw std::runtime_error ("lattice: mal-formed before parse N=/L= line in lattice.");
    
    assert(info.numnodes > 0);
    nodes.reserve (info.numnodes);
    // parse the nodes
    for (size_t i = 0; i < info.numnodes; i++, iter++)
    {
        if (iter == lines.end())
            throw std::runtime_error ("lattice: not enough I lines in lattice");
        unsigned long itest;
        float t;
        if (sscanf_s (*iter, "I=%lu t=%f%c", &itest, &t, &dummychar, sizeof (dummychar)) < 2)
            throw std::runtime_error ("lattice: mal-formed node line in lattice: " + string (*iter));
        if (i != (size_t) itest)
            throw std::runtime_error ("lattice: out-of-sequence node line in lattice: " + string (*iter));
        nodes.push_back (nodeinfo ((unsigned int) (t / info.frameduration + 0.5)));
        info.numframes = max (info.numframes, (size_t) nodes.back().t);
    }
    // parse the edges
    assert(info.numedges > 0);
    edges.reserve (info.numedges);
    align.reserve (info.numedges * 10);  // 10 phones per word on av. should be enough
    std::string label;
    for (size_t j = 0; j < info.numedges; j++, iter++)
    {
        if (iter == lines.end())
            throw std::runtime_error ("lattice: not enough J lines in lattice");
        unsigned long jtest;
        unsigned long S, E;
        float a, l;
        char d[1024];
        // example:
        // J=12    S=1    E=13   a=-326.81   l=-5.090  d=:sil-t:s+k:e,0.03:dh:m-ax:m+sil,0.03:sil,0.02:
        int nvals = sscanf_s (*iter, "J=%lu S=%lu E=%lu a=%f l=%f d=%s", &jtest, &S, &E, &a, &l, &d, sizeof (d));
        if (nvals == 5 && j == info.numedges - 1)    // special case: last edge is a !NULL and thus may have the d= record missing
            strcpy (d, ":");
        else if (nvals != 6)
            throw std::runtime_error ("lattice: mal-formed edge line in lattice: " + string (*iter));
        if (j != (size_t) jtest)
            throw std::runtime_error ("lattice: out-of-sequence edge line in lattice: " + string (*iter));
        edges.push_back (edgeinfowithscores (S, E, a, l, align.size()));
        // build align array
        size_t edgeframes = 0;      // (for checking whether the alignment sums up right)
        const char * p = d;
        if (p[0] != ':' || (p[1] == 0 && j < info.numedges-1))    // last edge may be empty
            throw std::runtime_error ("lattice: alignment info must start with a colon and must have at least one entry: " + string (*iter));
        p++;
        while (*p)
        {
            // p points to an entry of the form TRIPHONE,DURATION
            const char * q = strchr (p, ',');
            if (q == NULL)
                throw std::runtime_error ("lattice: alignment entry lacking a comma: " + string (*iter));
            if (q == p)
                throw std::runtime_error ("lattice: alignment entry label empty: " + string (*iter));
            label.assign (p, q-p);  // the triphone label
            q++;
            char * ep;
            double duration = strtod (q, &ep); // (weird--returns a non-const ptr in ep to a const object)
            p = ep;
            if (*p != ':')
                throw std::runtime_error ("lattice: alignment entry not ending with a colon: " + string (*iter));
            p++;
            // create the alignment entry
            const size_t frames = (unsigned int) (duration / info.frameduration + 0.5);
            auto it = unitmap.find (label);
            if (it == unitmap.end())
                throw std::runtime_error ("lattice: unit in alignment that is not in model: " + label);
            const size_t unitid = it->second;
            //const size_t unitid = unitmap.insert (make_pair (label, unitmap.size())).first->second;  // may create a new entry with index = #entries
            align.push_back (aligninfo (unitid, frames));
            edgeframes += frames;
        }
        if (edgeframes != nodes[E].t - (size_t) nodes[S].t)
        {
            char msg[128];
            sprintf (msg, "\n-- where edgeframes=%d != (nodes[E].t - nodes[S].t=%d), the gap is %d.", edgeframes, nodes[E].t - (size_t) nodes[S].t, edgeframes + nodes[S].t - nodes[E].t);
            throw std::runtime_error ("lattice: alignment info duration mismatches edge duration: " + string (*iter) + msg);
        }
    }
    if (iter != lines.end())
        throw std::runtime_error ("lattice: unexpected garbage at end of lattice: " + string (*iter));
    checklattice();

    // create more efficient storage for alignments
    const size_t spunit = tryfind (unitmap, "sp", SIZE_MAX);
    builduniquealignments (spunit);

    showstats();
}

// construct a numerator lattice from an MLF entry
// The lattice is expected to be freshly constructed (I did not bother to check).
void lattice::frommlf (const wstring & key, const std::unordered_map<std::string,size_t> & unitmap,
                       const msra::asr::htkmlfreader<msra::asr::htkmlfentry,lattice::htkmlfwordsequence> & labels,
                       const msra::lm::CMGramLM & unigram, const msra::lm::CSymbolSet & unigramsymbols)
{
    const auto & transcripts = labels.allwordtranscripts(); // (TODO: we could just pass the transcripts map--does not really matter)

    // get the labels (state and word)
    auto iter = transcripts.find (key);
    if (iter == transcripts.end())
        throw std::runtime_error ("frommlf: no reference word sequence in MLF for lattice with key " + strfun::utf8 (key));
    const auto & transcript = iter->second;
    if (transcript.words.size() == 0)
        throw std::runtime_error ("frommlf: empty reference word sequence for lattice with key " + strfun::utf8 (key));

    // determine unigram scores for all words
    vector<float> lmscores (transcript.words.size());
    size_t silence = unigramsymbols["!silence"];
    size_t lmend = unigramsymbols["</s>"];
    size_t sentstart = unigramsymbols["!sent_start"];
    size_t sentend = unigramsymbols["!sent_end"];

    // create the lattice
    nodes.resize (transcript.words.size() +1);
    edges.resize (transcript.words.size());
    align.reserve (transcript.align.size());
    size_t numframes = 0;
    foreach_index (j, transcript.words)
    {
        const auto & w = transcript.words[j];
        nodes[j].t = w.firstframe;
        auto & e = edges[j];
        e.unused = 0;
        e.S = j;
        e.E = j+1;
        if (e.E != j+1)
            throw std::runtime_error (msra::strfun::strprintf ("frommlf: too many tokens to be represented as edgeinfo::E in label set: %S", key.c_str()));
        e.a = 0.0f; // no ac score

        // LM score
        // !sent_start and !silence are patched to LM score 0
        size_t wid = w.wordindex;
        if (wid == sentstart)
        {
            if (j != 0)
                throw std::logic_error ("frommlf: found an !sent_start token not at the first position");
        }
        else if (wid == sentend)
        {
            if (j != (int) transcript.words.size()-1)
                throw std::logic_error ("frommlf: found an !sent_end token not at the end position");
            wid = lmend;    // use </s> for score lookup
        }
        const int iwid = (int) wid;
        e.l = (wid != sentstart && wid != silence) ? (float) unigram.score (&iwid, 1) : 0.0f;

        // alignment
        e.implysp = 0;
        e.firstalign = align.size();
        auto a = transcript.getaligninfo (j);
        align.insert (align.end(), a.begin(), a.end());
        foreach_index (k, a)
            numframes += a[k].frames;
    }
    nodes[transcript.words.size()].t = (unsigned short) numframes;
    if (nodes[transcript.words.size()].t != numframes)
        throw std::runtime_error (msra::strfun::strprintf ("frommlf: too many frames to be represented as nodeinfo::t in label set: %S", key.c_str()));
    info.lmf = -1.0f;       // indicates not set
    info.wp = 0.0f;         // not set indicated by lmf < 0
    info.numedges = edges.size();
    info.numnodes = nodes.size();
    info.numframes = numframes;
    checklattice();

    // create more efficient storage for alignments
    const size_t spunit = tryfind (unitmap, "sp", SIZE_MAX);
    builduniquealignments (spunit);

    showstats();
}

};};
