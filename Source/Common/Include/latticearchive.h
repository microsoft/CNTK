//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// latticearchive.h -- managing lattice archives
//

#pragma once

#undef HACK_IN_SILENCE  // [v-hansu] hack to simulate DEL in the lattice
#define SILENCE_PENALTY // give penalty to added silence

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include "Basics.h"
#include "latticestorage.h"
#include "simple_checked_arrays.h"
#include "fileutil.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm> // for find()
#include "simplesenonehmm.h"
#include "Matrix.h"

namespace msra { namespace math {

class ssematrixbase;
template <class ssematrixbase>
class ssematrix;
template <class ssematrixbase>
class ssematrixstriperef;
};
};

namespace msra { namespace lm {

class CMGramLM;
class CSymbolSet;
};
}; // for numer-lattice building

namespace msra { namespace asr {

template <typename A, typename B>
class htkmlfreader;
struct htkmlfentry;
};
}; // for numer lattice building

namespace msra { namespace lattices {

typedef msra::math::ssematrixbase matrixbase;
typedef msra::math::ssematrix<matrixbase> matrix;
typedef msra::math::ssematrixstriperef<matrixbase> matrixstripe;
class littlematrixheap;

enum mbrclassdefinition // used to identify definition of class in minimum bayesian risk
{
    senone = 1, // senone is default, which means no mapping; sMBR
    // monophonestate = 2,
    monophone = 3, // pMBR?
};
// ===========================================================================
// lattice -- one lattice in memory
// ===========================================================================
class lattice
{
public: 
    struct header_v1_v2
    {
        size_t numnodes : 32;
        size_t numedges : 32;
        float lmf;
        float wp;
        double frameduration;        // in seconds
        size_t numframes : 32;       // number of frames
        size_t impliedspunitid : 31; // id of implied last unit (intended as /sp/); only used in V2
        size_t hasacscores : 1;      // if 1 then ac scores are embedded

        header_v1_v2()
            : numnodes(0), numedges(0), lmf(1.0f), wp(0.0f), frameduration(0.01 /*assumption*/), numframes(0), impliedspunitid(INT_MAX), hasacscores(1)
        {
        }
    };
    header_v1_v2 info; // information about the lattice
private:
    mutable int verbosity;
    static const unsigned int NOEDGE = 0xffffff; // 24 bits
    // static_assert (sizeof (nodeinfo) == 8, "unexpected size of nodeeinfo"); // note: int64_t required to allow going across 32-bit boundary
    // ensure type size as these are expected to be of this size in the files we read
    static_assert(sizeof(nodeinfo) == 2, "unexpected size of nodeeinfo"); // note: int64_t required to allow going across 32-bit boundary
    static_assert(sizeof(edgeinfowithscores) == 16, "unexpected size of edgeinfowithscores");
    static_assert(sizeof(aligninfo) == 4, "unexpected size of aligninfo");
    std::vector<nodeinfo> nodes;
    std::vector<edgeinfowithscores> edges;
    std::vector<aligninfo> align;
    // V2 lattices  --for a while, we will store both in RAM, until all code is updated
    static int fsgn(float f)
    {
        if (f > 0)
            return 1;
        else if (f < 0)
            return -1;
        else
            return 0;
    }                                                                // what's this function called??
    int comparealign(size_t j1, size_t j2, bool sortbyfinalsp) const // strcmp()-like function for comparing alignments
    {
        // sortbyfinalsp: This is for dealing with edges that only differ in a final zero-frame /sp/
        // These should be considered equal in merging, such that the one without /sp/ (MLFs don't have final /sp/)
        // gets merged away (since it is inconsistent with decoding).
        //  - sortbyfinalsp = true: use in sorting (the longer edge with /sp/ will go FIRST so that it is the one to survive uniq-ing)
        //  - sortbyfinalsp = false: use in uniq-ing; the edges will just be reported as identical
        if (edges[j1].implysp || edges[j2].implysp)
            LogicError("comparealign: must not operate on edges with implysp flag set");
        const auto a1 = getaligninfo(j1);
        const auto a2 = getaligninfo(j2);
        // sort by unit sequence first
        for (size_t k = 0; k < a1.size() && k < a2.size(); k++)
        {
            int diff = (int) a1[k].unit - (int) a2[k].unit;
            if (diff != 0)
                return diff;
        }
        // then by the alignment  --we want to keep similar alignments of the same sequence close by
        for (size_t k = 0; k < a1.size() && k < a2.size(); k++)
        {
            int diff = (int) a1[k].frames - (int) a2[k].frames;
            if (diff != 0)
                return diff;
        }
        // identical sequence up to here  --check if they only differ in a final 0-frame /sp/
        // This is for merging of MLFs with lattices, where MLFs don't have /sp/.
        if ((a2.size() == a1.size() + 1 && a2.back().frames == 0)     // a2 has extra 0-frame /sp/
            || (a1.size() == a2.size() + 1 && a1.back().frames == 0)) // a1 has extra 0-frame /sp/
        {
            if (!sortbyfinalsp) // 'false' -> report them equal (used in final merge)
                return 0;
            // 'true' -> the longer one (with /sp/) comes first, i.e. the sorting order is reverse
            return (int) a2.size() - (int) a1.size(); // (note a1 and a2 swapped)
        }
        // all identical--if length same, then identical; else length determines ordering
        return (int) a1.size() - (int) a2.size();
    }
    // sort order that is useful for uniq'ing alignments
    int uniqueorder(const edgeinfo& e1, const edgeinfo& e2) const
    {
        // first sort by start and end time (required for the scoring functions)
        int diff = (int) nodes[e1.S].t - (int) nodes[e2.S].t;
        if (diff != 0)
            return diff;
        diff = (int) nodes[e1.E].t - (int) nodes[e2.E].t;
        if (diff != 0)
            return diff;
        // now sort by alignment (and also a and l, which must be identical--but likely they are anyway)
        size_t j1 = e1.firstalign; // temporarily: these are the indices to the original edges
        size_t j2 = e2.firstalign;
        // now compare by alignment
        diff = comparealign(j1, j2, true);
        if (diff != 0)
            return diff;
        // With the above, we are sorted properly to detect alignment dups.
        // a and l are also stored in the uniq'ed storage
        // Note: When merging lattices, 'l' may have different precision.
        // We did sort by alignment first (above), so we can still detect dups if from this sort order if later we are lenient.
        diff = fsgn(edges[j1].l - edges[j2].l);
        if (diff != 0)
            return diff;
        diff = fsgn(edges[j1].a - edges[j2].a);
        if (diff != 0)
            return diff;
        // identical--these can be grouped
        // and sort identical edges by start and end node again
        // This is not really used, since we later sort once again according to 'latticeorder()'
        diff = (int) e1.S - (int) e2.S;
        if (diff != 0)
            return diff;
        diff = (int) e1.E - (int) e2.E;
        // if (diff != 0)
        return diff;
    }
    // lattice sort order --algorithms assume lattices are sorted by E, then by S
    int latticeorder(const edgeinfo& e1, const edgeinfo& e2) const
    {
        // sort identical edges by start and end node again
        int diff = (int) e1.E - (int) e2.E;
        if (diff != 0)
            return diff;
        diff = (int) e1.S - (int) e2.S;
        if (diff != 0)
            return diff;
        // within same S/E pair, sort by firstalign
        // Since end nodes represent word identities in HAPI, this should only ever happen when merging lattices, but let's not rely on HAPI's assumptions.
        // builduniquealignments() only dedups the alignment records, but in case of merging, we want to dedup edges altogether.
        // For that, it is necessary that within a given S/E pair, where we now may have several different words, these edges are sorted
        // to be able to dedup based on firstalign.
        diff = (int) e1.firstalign - (int) e2.firstalign;
        return diff;
    }
    // more compact lattice storage
    std::vector<edgeinfo> edges2;                 // TODO: rename these
    std::vector<aligninfo> uniquededgedatatokens; // [-1]: LM score; [-2]: ac score; [0..]: actual aligninfo records
    float& uniqueedgelmscore(size_t firstalign)
    {
        return *(float*) &uniquededgedatatokens.data()[firstalign - 1];
    }
    float& uniqueedgeacscore(size_t firstalign)
    {
        if (info.hasacscores)
            return *(float*) &uniquededgedatatokens.data()[firstalign - 2];
        else
            LogicError("uniqueedgeacscore: no ac scores stored in this lattice");
    }

public: // TODO: make private again once
    // construct from edges/align
    // This is also used for merging, where the edges[] array is not correctly sorted. So don't assume this here.
    void builduniquealignments(size_t spunit = SIZE_MAX /*fix this later*/)
    {
        // infer /sp/ unit if not given
        // BUGBUG: This sometimes leads to incorrect results. We currently post-fix it.
        if (spunit == SIZE_MAX)
        {
            // Using a very simple heuristics; take the last unit of the first non-silence edge. We know it works for our current setup, but otherwise it's tricky.
            foreach_index (j, edges)
            {
                const auto ai = getaligninfo(j);
                if (ai.size() < 2) // less than 2--must be /sil/
                    continue;
                spunit = ai[ai.size() - 1].unit;
                fprintf(stderr, "builduniquealignments: /sp/ unit inferred through heuristics as %d\n", (int) spunit);
                break;
            }
        }
        info.impliedspunitid = spunit;

        // edges2 array gets sorted to group edges with identical alignments together
        info.hasacscores = 0; // if we got any score != 0.0, we will set this
        edges2.resize(edges.size());
        foreach_index (j, edges)
        {
            if (edges[j].implysp)
                LogicError("builduniquealignments: original edges[] array must not have implied /sp/");
            edges2[j].S = edges[j].S;
            edges2[j].E = edges[j].E;
            edges2[j].unused = 0;
            edges2[j].implysp = 0;
            edges2[j].firstalign = j; // index into the original edges[] array before sorting, temporarily stored here to survive sorting
            checkoverflow(edges2[j].S, edges[j].S, "edgeinfo2::S");
            checkoverflow(edges2[j].E, edges[j].E, "edgeinfo2::E");
            checkoverflow(edges2[j].firstalign, j, "edgeinfo2::firstalign (j for sorting)");
            if (edges[j].a != 0.0f)
                info.hasacscores = 1;
        }

        // sort edges
        sort(edges2.begin(), edges2.end(), [&](const edgeinfo& e1, const edgeinfo& e2)
             {
                 return uniqueorder(e1, e2) < 0;
             });

        // create a uniq'ed version of the align[] array, into uniquededgedatatokens[]
        uniquededgedatatokens.resize(0);
        uniquededgedatatokens.reserve(align.size());

        size_t numuniquealignments = 0; // number of unique alignments (=number of edges with unique alignments)

        size_t prevj = SIZE_MAX; // this is an index into the original edges[] array before sorting
        size_t numimpliedsp = 0; // (statistics)
        foreach_index (j2, edges2)
        {
            size_t j = edges2[j2].firstalign; // index into the original edges[] array before sorting (was temporarily stored here)
            // allocate a new edge group if this edge differs from the previous
            const float lmargin = 1e-3f;                                                                                                                                                                                                                     // if merging then the same LM score may come from different ASCII sources with different precision. HTK lattices store 3 digits after the period.
#if 1                                                                                                                                                                                                                                                        // diagnostics on the merging of MLF and HTK inputs
            if (prevj != SIZE_MAX && fabs(edges[prevj].l - edges[j].l) <= lmargin && comparealign(prevj, j, false) == 0 && nodes[edges[prevj].S].t == nodes[edges[j].S].t && nodes[edges[prevj].E].t == nodes[edges[j].E].t && edges[prevj].l != edges[j].l) // some diagnostics
                fprintf(stderr, "build: merging edges %d and %d despite slightly different LM scores %.8f vs. %.8f, ts/te=%.2f/%.2f\n",
                        (int) prevj, (int) j, edges[prevj].l, edges[j].l, nodes[edges[prevj].S].t * 0.01f, nodes[edges[prevj].E].t * 0.01f);
#endif
            if (prevj == SIZE_MAX || fabs(edges[prevj].l - edges[j].l) > lmargin || (info.hasacscores && edges[prevj].a != edges[j].a) || comparealign(prevj, j, false) != 0)
            {
                // allocate a new alignment
                size_t currentfirstalign = uniquededgedatatokens.size() + 1;
                if (info.hasacscores)
                    currentfirstalign++;
                // inject the lm and ac scores
                uniquededgedatatokens.resize(currentfirstalign);
                uniqueedgelmscore(currentfirstalign) = edges[j].l;
                if (info.hasacscores)
                    uniqueedgeacscore(currentfirstalign) = edges[j].a;
                // and copy it
                edges2[j2].firstalign = currentfirstalign;                             // this is where it starts
                checkoverflow(edges2[j2].firstalign, currentfirstalign, "firstalign"); // this is also a sequence check

                const auto ai = getaligninfo(j);
                size_t nalign = ai.size();
                if (nalign == 0 && (size_t) j2 != edges.size() - 1)
                    RuntimeError("builduniquealignments: !NULL edges forbidden except for the very last edge");
                // special optimization: we do not store the /sp/ unit at the end
                if (nalign > 1 /*be robust against 1-unit edges that consist of spunit*/ && ai[nalign - 1].unit == spunit)
                {
                    nalign--;
                    edges2[j2].implysp = 1;
                    numimpliedsp++; // (diagnostics only)
                }
                else
                    edges2[j2].implysp = 0;
                // copy the tokens
                for (size_t k = 0; k < nalign; k++)
                {
                    auto a = ai[k];
                    if (a.last)
                        LogicError("builduniquealignments: unexpected 'last' flag already set in input aligns (numeric overflow in old format?)");
                    if (k == nalign - 1)
                        a.last = 1;
                    uniquededgedatatokens.push_back(a);
                }
                numuniquealignments++;
            }
            else // duplicate from previous
            {
                edges2[j2].firstalign = edges2[j2 - 1].firstalign;
                edges2[j2].implysp = edges2[j2 - 1].implysp;
            }
            prevj = j;
        }
        const size_t uniquealigntokens = uniquededgedatatokens.size() - (numuniquealignments * (info.hasacscores ? 2 : 1));
        const size_t nonuniquenonsptokens = align.size() - numimpliedsp;
        fprintf(stderr, "builduniquealignments: %d edges: %d unique alignments (%.2f%%); %d align tokens - %d implied /sp/ units = %d, uniqued to %d (%.2f%%)\n",
                (int) edges.size(), (int) numuniquealignments, 100.0f * numuniquealignments / edges.size(),
                (int) align.size(), (int) numimpliedsp, (int) nonuniquenonsptokens, (int) uniquealigntokens, 100.0f * uniquealigntokens / nonuniquenonsptokens);

        // sort it back into original order (sorted by E, then by S)
        sort(edges2.begin(), edges2.end(), [&](const edgeinfo& e1, const edgeinfo& e2)
             {
                 return latticeorder(e1, e2) < 0;
             });

        // TODO: be more consistent--we should clear out edges[] at this point!
    }

private:
    // infer ends in case of broken lattices with zero-token edges
    // This happened when /sp/ was wrongly inferred, and 0-token edges were generated, for which we have no 'last' flag.
    // Such lattices can no longer be generated, but we are stuck with old ones that have this problem.
    void inferends(std::vector<bool>& isend) const
    {
        isend.resize(uniquededgedatatokens.size() + 1, false);
        isend.back() = true;
        foreach_index (j, edges2)
        {
            size_t end = edges2[j].firstalign;
            end--; // LM score
            if (info.hasacscores)
                end--; // ac score
            // end is now the index of a unit right after a align sequence
            isend[end] = true;
        }
    }

public:
    // hack function to add a single-/sil/ edge, as well as a single /sp/, with LM score 0 to every unique (S,E) pair that doesn't already have a /sil/ and/or /sp/
    // This is to simulate DELetions. We observe a massive DEL problem. Hypothesis: Caused by too many strong positive obs for /sil/ and/or /sp/, and too few strong counter-weights.
    // Note: Adding /sil/ lowers the objective function a little (unexpected; maybe due to the hack), adding /sp/ lowers it more (really unexpected since no hack here).
    void hackinsilencesubstitutionedges(size_t silunit, size_t spunit, bool addsp)
    {
        std::vector<edgeinfowithscores> newedges;
        newedges.reserve(edges.size() * 5 / 2); // avoid realloc , this saves time :)
        std::vector<aligninfo> newalign;
        newalign.reserve(align.size() * 5 / 2); // avoid realloc
        // loop over all edges, and duplicate them, inserting /sil/ and /sp/ edges
        // Note that we assume that single-sil edges only (and always) exist at start and end, to avoid checking.
        // This is a hack anyway.
        // We exploit sortedness of the edges array by (S,E).
        foreach_index (j, edges)
        {
            auto e = edges[j];
            assert(e.unused == 0);
            e.unused = 0;
#ifdef SILENCE_PENALTY
            const float penaltyforsil = float(-1.8 / 12); // estimated on training MLF; assuming LMF=12
#else
            const float penaltyforsil = 0.0f;
#endif
            if (j > 0 && (e.S != edges[j - 1].S || e.E != edges[j - 1].E)) // new block entered
            {
                if (e.S != 0 && e.E != nodes.size() - 1) // and it's not a block that has !sent_start/end
                {
                    // create a new silence edge
                    // To make it perfectly clear: For /sil/, this is a HACK--the acoustic contexts are WRONG. Only a quick test. (For /sp/ there is no such problem though.)
                    const size_t numframes = nodes[e.E].t - nodes[e.S].t;
                    if (numframes > 0)
                    {
                        edgeinfowithscores sile = e;
                        sile.unused = 1;        // indicate that this is an added edge
                        sile.l = penaltyforsil; // sil is penalized
                        sile.implysp = 0;
                        sile.a = LOGZERO;                  // must not have this anyway
                        sile.firstalign = newalign.size(); // we create a new entry for this
                        if (sile.firstalign != newalign.size())
                            RuntimeError("hackinsilencesubstitutionedges: numeric bit-field overflow of .firstalign");
                        newedges.push_back(sile);
                        // create a new align entry
                        aligninfo asil(silunit, numframes);
                        newalign.push_back(asil);
                        if (addsp)
                        {
                            edgeinfowithscores spe = sile;
                            spe.firstalign = newalign.size();
                            if (spe.firstalign != newalign.size())
                                RuntimeError("hackinsilencesubstitutionedges: numeric bit-field overflow of .firstalign");
                            newedges.push_back(spe);
                            aligninfo asp(spunit, numframes);
                            newalign.push_back(asp);
                        }
                    }
                }
            }
            if (e.S != 0 && e.E != nodes.size() - 1) // add penalty to sil that appears by the end in a word edge
            {
                auto a = getaligninfo(j);
                if (a.back().unit == silunit && a.size() > 1)
                    e.l += penaltyforsil;
            }
            // copy the edge
            e.firstalign = newalign.size();
            if (e.firstalign != newalign.size())
                RuntimeError("hackinsilencesubstitutionedges: numeric bit-field overflow of .firstalign");
            newedges.push_back(e);
            // copy the align records
            auto a = getaligninfo(j);
            foreach_index (k, a)
                newalign.push_back(a[k]);
        }
        static int count = 0;
        if (count++ < 10) // (limit the log spam)
        {
            fprintf(stderr, "hackinsilencesubstitutionedges: added %d DEL (/sil/-%ssubstitution) edges (from %d to %d; align from %d to %d)\n",
                    (int) (newedges.size() - edges.size()),
                    addsp ? " and /sp/-" : "",
                    (int) edges.size(), (int) newedges.size(),
                    (int) align.size(), (int) newalign.size());
        }
        edges.swap(newedges);
        info.numedges = edges.size();
        align.swap(newalign);
        edges.shrink_to_fit(); // [v-hansu] might be useful when RAM is out of use
        align.shrink_to_fit();
    }
    // go back from V2 format to edges and align, so old code can still run
    // This will go away one we updated all code to use the new data structures.
    void rebuildedges(bool haszerotokenedges /*pass true for broken spunit that may have reduced edges to 0 entries*/)
    {
        // deal with broken (zero-token) edges
        std::vector<bool> isendworkaround;
        if (haszerotokenedges)
            inferends(isendworkaround);

        edges.resize(edges2.size());
        align.resize(0);
        align.reserve(uniquededgedatatokens.size() * 10); // should be enough
        foreach_index (j, edges)
        {
            edges[j].S = edges2[j].S;
            edges[j].E = edges2[j].E;
            edges[j].unused = 0;
            edges[j].implysp = 0;
            const size_t firstalign = edges2[j].firstalign;
            edges[j].a = info.hasacscores ? uniqueedgeacscore(firstalign) : -1e30f /*LOGZERO*/; // cannot reconstruct; not available
            edges[j].l = uniqueedgelmscore(firstalign);
            // expand the alignment tokens
            edges[j].firstalign = align.size();
            const size_t edgedur = nodes[edges2[j].E].t - nodes[edges2[j].S].t; // for checking and back-filling the implied /sp/
            size_t aligndur = 0;
            if (firstalign == uniquededgedatatokens.size() && (size_t) j != edges.size() - 1)
                RuntimeError("rebuildedges: !NULL edges forbidden except for the last edge");
            for (size_t k = firstalign; k < uniquededgedatatokens.size(); k++)
            {
                if (!isendworkaround.empty() && isendworkaround[k]) // secondary criterion to detect ends in broken lattices
                    break;
                aligninfo ai = uniquededgedatatokens[k];
                if (ai.unused != 0)
                    RuntimeError("rebuildedges: mal-formed uniquededgedatatokens[] array: 'unused' field must be 0");
                bool islast = ai.last != 0;
                ai.last = 0; // old format does not support this
                align.push_back(ai);
                aligndur += ai.frames;
                if (aligndur > edgedur)
                    RuntimeError("rebuildedges: mal-formed uniquededgedatatokens[] array: aligment longer than edge");
                if (islast)
                    break;
                if (k == uniquededgedatatokens.size() - 1)
                    RuntimeError("rebuildedges: mal-formed uniquededgedatatokens[] array: missing 'last' flag in last entry");
            }
            if (edges2[j].implysp)
            {
                if (info.impliedspunitid == SIZE_MAX)
                    RuntimeError("rebuildedges: edge requests implied /sp/ but none specified in lattice header");
                if (aligndur > edgedur)
                    RuntimeError("rebuildedges: edge alignment longer than edge duration");
                aligninfo ai(info.impliedspunitid, edgedur - aligndur /*frames: remaining frames are /sp/ */);
                align.push_back(ai);
            }
        }
        // fprintf (stderr, "rebuildedges: %d edges reconstructed to %d alignment tokens\n", edges.size(), align.size());    // [v-hansu] comment out because it takes up most of the log
        align.shrink_to_fit(); // free up unused memory (since we need it!!)
        // now get rid of the V2 data altogether
        uniquededgedatatokens.clear();
        uniquededgedatatokens.shrink_to_fit();
        edges2.clear();
        edges2.shrink_to_fit();
    }

public:
    class parallelstate;

    // a word sequence read from an MLF file
    struct htkmlfwordsequence
    {
        // a word entry read from an MLF file
        struct word // word info we are reading from the MLF file (if we want to add the ground-truth path)
        {
            static const unsigned int unknownwordindex = 0xfffff; // max value storable in 'wordindex'
            unsigned int wordindex : 20;                          // per mapping table; unknownwordindex denotes unknown word
            unsigned int firstalign : 12;                         // index into align record to first phoneme entry
            unsigned int firstframe : 16;                         // TODO: obsolete; once removed, we are back at 32 bits--yay
            word()
            {
            } // to keep compiler happy
            word(size_t wid, size_t ts, size_t as)
            {
                wordindex = (unsigned int) wid;
                firstframe = (unsigned int) ts;
                firstalign = (unsigned int) as;
                if (wordindex != wid)
                    RuntimeError("htkmlfwordentry: vocabulary size too large for bit field 'wordindex'");
                if (firstframe != ts)
                    RuntimeError("htkmlfwordentry: start frame too large for bit field 'firstframe'");
                if (firstalign != as)
                    RuntimeError("htkmlfwordentry: first-align index too large for bit field 'firstframe'");
            }
        };

        typedef msra::lattices::aligninfo aligninfo; // now we can access it as htkmlfwordsequence::aligninfo although it comes from some totally other corner of the system

        std::vector<word> words;
        std::vector<aligninfo> align;

        // get aligninfo array for a word
        const_array_ref<aligninfo> getaligninfo(size_t j) const
        {
            size_t begin = (size_t) words[j].firstalign;
            size_t end = j + 1 < words.size() ? (size_t) words[j + 1].firstalign : align.size();
            return const_array_ref<aligninfo>(align.data() + begin, end - begin);
        }
    };

private:
    struct edgealignments // struct to return alignments using an efficient long-vector storage
    {
        std::vector<unsigned int> alignoffsets;    // [j] index of first alignment in allalignments; one extra element for length of last entry
        std::vector<unsigned short> allalignments; // all alignments concatenated
    public:
        edgealignments(const lattice& L)
        {
            size_t alignbufsize = 0;
            alignoffsets.resize(L.edges.size() + 1); // one extra element so we can determine the length of last entry
            foreach_index (j, L.edges)
            {
                alignoffsets[j] = (unsigned int) alignbufsize;
                size_t edgenumframes = L.nodes[L.edges[j].E].t - L.nodes[L.edges[j].S].t;
                alignbufsize += edgenumframes;
            }
            alignoffsets[L.edges.size()] = (unsigned int) alignbufsize; // (TODO: remove if not actually needed)
        }
        // edgealignments[j][t] is the senone at frame offset t in edge j
        array_ref<unsigned short> operator[](size_t j)
        {
            if (allalignments.size() == 0)
                allalignments.resize(alignoffsets.back());
            size_t offset = alignoffsets[j];
            size_t numframes = alignoffsets[j + 1] - alignoffsets[j];
            if (numframes == 0)
                return array_ref<unsigned short>();
            return array_ref<unsigned short>(&allalignments[offset], numframes);
        }
        const_array_ref<unsigned short> operator[](size_t j) const
        {
            size_t offset = alignoffsets[j];
            size_t numframes = alignoffsets[j + 1] - alignoffsets[j];
            return const_array_ref<unsigned short>(&allalignments[offset], numframes);
        }
        // CUDA support
        const std::vector<unsigned int>& getalignoffsets() const
        {
            return alignoffsets;
        }
        std::vector<unsigned short>& getalignmentsbuffer()
        {
            allalignments.resize(alignoffsets.back());
            return allalignments;
        } // for retrieving it from the GPU
        const std::vector<unsigned short>& getalignmentsbuffer() const
        {
            if (allalignments.size() != alignoffsets.back())
                RuntimeError("getalignmentsbuffer: allalignments not allocated!\n");
            return allalignments;
        } // for retrieving it from the GPU
        size_t getalignbuffersize() const
        {
            return alignoffsets.back();
        }
    };

    struct backpointers
    {
        std::vector<size_t> backptroffsets;         // TODO: we could change this to 'unsigned int' to save some transfer time
        std::vector<unsigned short> backptrstorage; // CPU-side versions use this as the traceback buffer; CUDA code has its CUDA-side buffer
        size_t numofstates;                         // per sil hmm
        int verbosity;

    public:
        backpointers(const lattice& L, const msra::asr::simplesenonehmm& hset, int verbosity = 0)
            : numofstates(0)
        {
            size_t edgeswithsilence = 0; // (diagnostics only: number of edges with at least one /sil/)
            size_t backptrbufsize = 0;   // number of entries in buffer for silence backpointer array, used as cursor as we build it

            backptroffsets.resize(L.edges.size() + 1); // +1, so that the final entry determines the overall size of the allocated buffer
            const size_t silUnitId = hset.gethmmid("sil");
            numofstates = hset.gethmm(silUnitId).getnumstates();
            foreach_index (j, L.edges)
            {
                // for each edge, determine if it needs a backpointer buffer for silence
                // Multiple /sil/ in the same edge will share the same buffer, so we need to know the max length.
                const auto& aligntokens = L.getaligninfo(j); // get alignment tokens
                backptroffsets[j] = backptrbufsize;          // buffer for this edge begins here
                size_t maxsilframes = 0;                     // max #frames--we allocate this many for this edge
                size_t numsilunits = 0;                      // number of /sil/ units in this edge
                foreach_index (a, aligntokens)
                {
                    if (aligntokens[a].unit == silUnitId)
                    {
                        numsilunits++;                            // count
                        if (aligntokens[a].frames > maxsilframes) // determine max #frames
                            maxsilframes = aligntokens[a].frames;
                    }
                }
#if 1 // multiple /sil/ -> log this (as we are not sure whether this is actually proper--probably it is)
                if (numsilunits > 1)
                {
                    if (verbosity)
                    {
                        fprintf(stderr, "backpointers: lattice '%S', edge %d has %d /sil/ phonemes\n", L.getkey(), j, (int) numsilunits);
                        fprintf(stderr, "alignments: :");
                        foreach_index (a, aligntokens)
                        {
                            const auto& unit = aligntokens[a];
                            const auto& hmm = hset.gethmm(unit.unit);
                            fprintf(stderr, "%s,%.2f:", hmm.getname(), unit.frames / 100.0f);
                        }
                        fprintf(stderr, "\n");
                    }
                }
#endif
                if (numsilunits > 0)
                    edgeswithsilence++; // (for diagnostics message only)
                backptrbufsize += maxsilframes * numofstates;
            }
            backptroffsets[L.edges.size()] = backptrbufsize; // (TODO: remove if not actually needed)
            if (verbosity)
                fprintf(stderr, "backpointers: %.1f%% edges have at least one /sil/ unit inside\n", 100.0f * ((float) edgeswithsilence / L.edges.size()));
        }
        // CUDA support
        const std::vector<size_t>& getbackptroffsets() const
        {
            return backptroffsets;
        }
        std::vector<unsigned short>& getbackptrbuffer()
        {
            backptrstorage.resize(backptroffsets.back());
            return backptrstorage;
        } // for retrieving it from the GPU
        size_t getbackptrstoragesize() const
        {
            return backptroffsets.back();
        }
    };
    void forwardbackwardalign(parallelstate& parallelstate,
                              const msra::asr::simplesenonehmm& hset, const bool softalignstates,
                              const double minlogpp, const std::vector<double>& origlogpps,
                              std::vector<msra::math::ssematrixbase*>& abcs, littlematrixheap& matrixheap,
                              const bool returnsenoneids,
                              std::vector<float>& edgeacscores, const msra::math::ssematrixbase& logLLs,
                              edgealignments& thisedgealignments, backpointers& thisbackpointers, array_ref<size_t>& uids, const_array_ref<size_t> bounds) const;

    double forwardbackwardlatticesMBR(const std::vector<float>& edgeacscores, const msra::asr::simplesenonehmm& hset,
                                      const std::vector<double>& logalphas, const std::vector<double>& logbetas,
                                      const float lmf, const float wp, const float amf, const_array_ref<size_t>& uids,
                                      const edgealignments& thisedgealignments, std::vector<double>& Eframescorrect) const;

    void sMBRerrorsignal(parallelstate& parallelstate,
                         msra::math::ssematrixbase& errorsignal, msra::math::ssematrixbase& errorsignalneg,
                         const std::vector<double>& logpps, const float amf, double minlogpp,
                         const std::vector<double>& origlogpps, const std::vector<double>& logEframescorrect,
                         const double logEframescorrecttotal, const edgealignments& thisedgealignments) const;

    void mmierrorsignal(parallelstate& parallelstate, double minlogpp, const std::vector<double>& origlogpps,
                        std::vector<msra::math::ssematrixbase*>& abcs, const bool softalignstates,
                        const std::vector<double>& logpps, const msra::asr::simplesenonehmm& hset,
                        const edgealignments& thisedgealignments, msra::math::ssematrixbase& errorsignal) const;

    double bestpathlattice(const std::vector<float>& edgeacscores, std::vector<double>& logpps,
                           const float lmf, const float wp, const float amf) const;

    static float alignedge(const_array_ref<aligninfo> units, const msra::asr::simplesenonehmm& hset,
                           const msra::math::ssematrixbase& logLLs, msra::math::ssematrixbase& gammas,
                           size_t edgeindex, const bool returnsenoneids, array_ref<unsigned short> thisedgealignments);

    const_array_ref<aligninfo> getaligninfo(size_t j) const
    {
        size_t begin = (size_t) edges[j].firstalign;
        size_t end = j + 1 < edges.size() ? (size_t) edges[j + 1].firstalign : align.size();
        return const_array_ref<aligninfo>(align.data() + begin, end - begin);
    }

    static std::string gettranscript(const_array_ref<aligninfo> units, const msra::asr::simplesenonehmm& hset);

    void parallelforwardbackwardalign(parallelstate& parallelstate,
                                      const msra::asr::simplesenonehmm& hset, const msra::math::ssematrixbase& logLLs,
                                      std::vector<float>& edgeacscores, edgealignments& edgealignments, backpointers& backpointers) const;

    void parallelsMBRerrorsignal(parallelstate& parallelstate, const edgealignments& thisedgealignments,
                                 const std::vector<double>& logpps, const float amf,
                                 const std::vector<double>& logEframescorrect, const double logEframescorrecttotal,
                                 msra::math::ssematrixbase& errorsignal, msra::math::ssematrixbase& errorsignalneg) const;

    void parallelmmierrorsignal(parallelstate& parallelstate, const edgealignments& thisedgealignments,
                                const std::vector<double>& logpps, msra::math::ssematrixbase& errorsignal) const;

    double parallelforwardbackwardlattice(parallelstate& parallelstate, const std::vector<float>& edgeacscores,
                                          const edgealignments& thisedgealignments, const float lmf, const float wp,
                                          const float amf, const float boostingfactor, std::vector<double>& logpps, std::vector<double>& logalphas,
                                          std::vector<double>& logbetas, const bool returnEframescorrect,
                                          const_array_ref<size_t>& uids, std::vector<double>& logEframescorrect,
                                          std::vector<double>& Eframescorrectbuf, double& logEframescorrecttotal) const;

    static double scoregroundtruth(const_array_ref<size_t> uids, const_array_ref<htkmlfwordsequence::word> transcript,
                                   const std::vector<float>& transcriptunigrams, const msra::math::ssematrixbase& logLLs,
                                   const msra::asr::simplesenonehmm& hset, const float lmf, const float wp, const float amf);

    static float forwardbackwardedge(const_array_ref<aligninfo> units, const msra::asr::simplesenonehmm& hset,
                                     const msra::math::ssematrixbase& logLLs, msra::math::ssematrixbase& gammas,
                                     size_t edgeindex);

    double forwardbackwardlattice(const std::vector<float>& edgeacscores, parallelstate& parallelstate,
                                  std::vector<double>& logpps, std::vector<double>& logalphas, std::vector<double>& logbetas,
                                  const float lmf, const float wp, const float amf, const float boostingfactor, const bool sMBRmode,
                                  const_array_ref<size_t>& uids, const edgealignments& thisedgealignments,
                                  std::vector<double>& logEframescorrect, std::vector<double>& Eframescorrectbuf,
                                  double& logEframescorrecttotal) const;

public:
    // construct from a HTK lattice file
    void fromhtklattice(const std::wstring& path, const std::unordered_map<std::string, size_t>& unitmap);

    // construct from an MLF file (numerator lattice)
    void frommlf(const std::wstring& key, const std::unordered_map<std::string, size_t>& unitmap, const msra::asr::htkmlfreader<msra::asr::htkmlfentry, lattice::htkmlfwordsequence>& labels,
                 const msra::lm::CMGramLM& lm, const msra::lm::CSymbolSet& unigramsymbols);

    // check consistency
    //  - only one end node
    //  - only forward edges
    //  - nodes are sorted by time
    //  - edges are sorted by end node (they happen to come like this; so we can capitalize on it)
    void checklattice() const
    {
        // in/out counts to detect orphan nodes
        std::vector<size_t> numin(info.numnodes, 0), numout(info.numnodes, 0);
        // check edges' sortedness and count in/out
        for (size_t j = 0; j < info.numedges; j++)
        {
            const auto& e = edges[j];
            if (e.E <= e.S)
                RuntimeError("checklattice: lattice is not topologically sorted");
            if (nodes[e.E].t < nodes[e.S].t)
                RuntimeError("checklattice: lattice edge has negative time range");
            if (nodes[e.E].t == nodes[e.S].t && j < info.numedges - 1)
                RuntimeError("checklattice: 0-frame edges forbidden except for very last edge");
            if (j != (info.numedges - 1) && nodes[e.E].t == nodes[e.S].t) // last arc can be zero time range
                RuntimeError("checklattice: lattice edge has zero time range");
            if (j > 0 && e.E < edges[j - 1].E)
                RuntimeError("checklattice: lattice is not sorted by end node");
            if (j > 0 && e.E == edges[j - 1].E && e.S < edges[j - 1].S) // == also not allowed except for terminal edges
                RuntimeError("checklattice: lattice is not sorted by start node within the same end node");
            if (j > 0 && e.E == edges[j - 1].E && e.S == edges[j - 1].S)
            { // Note: same E means identical word on the edge, due to word id stored on node. Thus, the edge is redundant = forbidden.
                if (e.E != info.numnodes - 1)
                    RuntimeError("checklattice: lattice has duplicate edges");
                else // Exception: end of lattice, which happens rarely (2 examples found) and won't cause dramatic error, none in typical cases.
                    fprintf(stderr, "checklattice: WARNING: duplicate edge J=%d (S=%d -> E=%d) at end of lattice\n", (int) j, (int) e.S, (int) e.E);
            }
            numin[e.E]++;
            numout[e.S]++;
        }
        // check nodes and in/out counts
        if (nodes[0].t != 0.0f)
            RuntimeError("checklattice: lattice does not begin with time 0");
        for (size_t i = 0; i < info.numnodes; i++)
        {
            if (i > 0 && nodes[i].t < nodes[i - 1].t)
                RuntimeError("checklattice: lattice nodes not sorted by time");
            if ((numin[i] > 0) ^ (i > 0))
                RuntimeError("checklattice: found an orphaned start node");
            if ((numout[i] > 0) ^ (i < info.numnodes - 1))
                RuntimeError("checklattice: found an orphaned end node");
        }
    }

    void showstats() const // display stats info for a lattice
    {
        size_t totaledgeframes = 0;
        for (size_t j = 0; j < info.numedges; j++)
            totaledgeframes += nodes[edges[j].E].t - (size_t) nodes[edges[j].S].t;
        fprintf(stderr, "lattice: read %d nodes, %d edges, %d units, %d frames, %.1f edges/node, %.1f units/edge, %.1f frames/edge, density %.1f\n",
                (int) info.numnodes, (int) info.numedges, (int) align.size(), (int) info.numframes,
                info.numedges / (double) info.numnodes, align.size() / (double) info.numedges, totaledgeframes / (double) info.numedges, totaledgeframes / (double) info.numframes);
    }

    // merge a second lattice in --for use by convert()
private:
    // helper for merge()
    struct nodecontext
    {
        int left, right;
        static const signed short unknown = -1;   // not set yet
        static const signed short ambiguous = -2; // multiple --this is allowed if the other context is /sil/
        static const signed short start = -3;     // lattice start node
        static const signed short end = -4;       // lattice end node
        nodecontext()
        {
            left = unknown;
            right = unknown;
            t = SIZE_MAX;
            i = SIZE_MAX;
            iother = SIZE_MAX;
        }
        // helpers to set the values with uniq checks
    private:
        void setcontext(int& lr, int val);

    public:
        void setleft(int val)
        {
            setcontext(left, val);
        }
        void setright(int val)
        {
            setcontext(right, val);
        }
        // for building joint node space
        size_t t;      // frame index
        size_t i;      // original node index
        size_t iother; // original node index in 'other' lattice
        bool operator<(const nodecontext& other) const;
    };
    std::vector<nodecontext> determinenodecontexts(const msra::asr::simplesenonehmm& hset) const;

public:
    void removefinalnull(); // call this before merge on both lattices
    void merge(const lattice& other, const msra::asr::simplesenonehmm& hset);
    void dedup(); // call this after merge() after conversion to uniq'ed format

    template <typename HMMLOOKUPFUNCTION>
    void dump(FILE* f, const HMMLOOKUPFUNCTION& gethmmname) const // dump a lattice in HTK-like format
    {
        fprintf(f, "N=%lu L=%lu\n", (unsigned long)nodes.size(), (unsigned long)edges.size());
        // foreach_index (i, nodes)
        //    fprintf (f, "I=%d\tt=%.2f\n", i, nodes[i].t * 0.01f);
        foreach_index (j, edges)
        {
            const auto& e = edges[j];
            fprintf(f, "J=%d\tS=%d\tE=%d\tts=%.2f\tte=%.2f\ta=%.3f\tl=%.8f\td=:",
                    (int) j, (int) e.S, (int) e.E, (float) nodes[e.S].t * 0.01f, (float) nodes[e.E].t * 0.01f, (float) e.a, (float) e.l);
            const auto align2 = getaligninfo(j);
            foreach_index (k, align2) // e.g. d=:aa:m-ih:s+t:e,0.03:ow:e-t:m+sil,0.03:sil,0.21:
                fprintf(f, "%s,%.2f:", gethmmname(align2[k].unit), align2[k].frames * 0.01f);
            fprintf(f, "\n");
        }
    }

    size_t getnumframes() const
    {
        return info.numframes;
    }
    size_t getnumnodes() const
    {
        return info.numnodes;
    }
    size_t getnumedges() const
    {
        return info.numedges;
    }

    // write a tag, followed by an integer
    void fwritetag(FILE* f, const char* tag, size_t n)
    {
        fputTag(f, tag);
        fputint(f, (int) n);
    }

    template <class VECTOR>
    void fwritevector(FILE* f, const char* tag, const VECTOR& v)
    {
        fwritetag(f, tag, v.size());
        fwriteOrDie(v, f);
    }

    void fwrite(FILE* f)
    {
#if 1
        const size_t version = 2; // format version
        fwritetag(f, "LAT ", version);
        fwriteOrDie(&info, sizeof(info), 1, f);
        fwritevector(f, "NODS", nodes);
        fwritevector(f, "EDGS", edges2);                // uniqued edges
        fwritevector(f, "ALNS", uniquededgedatatokens); // uniqued alignments and scores
        fputTag(f, "END ");
#else
        const size_t version = 1; // format version
        fwritetag(f, "LAT ", version);
        fwriteOrDie(&info, sizeof(info), 1, f);
        fwritevector(f, "NODE", nodes);
        fwritevector(f, "EDGE", edges);
        fwritevector(f, "ALIG", align);
        fputTag(f, "END ");
#endif
    }

    // empty constructor, e.g. for use in minibatch source
    lattice()
    {
    }

    size_t freadtag(FILE* f, const char* tag)
    {
        fcheckTag(f, tag);
        return (unsigned int) fgetint(f);
    }

    template <class VECTOR>
    void freadvector(FILE* f, const char* tag, VECTOR& v, size_t expectedsize = SIZE_MAX)
    {
        const size_t sz = freadtag(f, tag);
        if (expectedsize != SIZE_MAX && sz != expectedsize)
            RuntimeError("freadvector: malformed file, number of vector elements differs from head, for tag %s", tag);
        freadOrDie(v, sz, f);
    }
    
    bool CheckTag(const char*& buffer, const std::string& expectedTag) 
    {
        std::string tag(buffer, expectedTag.length());
        if (tag != expectedTag)
            return false;
        buffer += expectedTag.length();
        return true;
    }
    
    int ReadTagFromBuffer(const char*& buffer, const std::string& expectedTag, size_t expectedSize = SIZE_MAX)
    {
        if (!CheckTag(buffer, expectedTag)) {
            // since lattice is packed densely by the reader, we may need to shift the buffer by 2 bytes.
            if (!CheckTag(buffer, expectedTag.substr(2)))
                RuntimeError("ReadTagFromBuffer: malformed file, missing expected tag: %s,", expectedTag.c_str());
        }
        int* sz = (int*)buffer;
        if (expectedSize != SIZE_MAX && *sz != expectedSize)
            RuntimeError("ReadTagFromBuffer: malformed file, number of vector elements differs from head, for tag %zu", expectedSize);

        buffer += 4;
        return *sz;
    }

    template <class T>
    void ReadVectorFromBuffer(const char*& buffer, const std::string& expectedTag, std::vector<T>& v, size_t expectedsize = SIZE_MAX)
    {
        int sz = ReadTagFromBuffer(buffer, expectedTag, expectedsize);
        v.resize(sz);
        for (size_t i = 0;i < sz;i++) {
            const T* element = reinterpret_cast<const T*>(buffer);
            v[i] = *element;
            buffer += sizeof(T);
        }
    }

    // read from a stream
    // This can be used on an existing structure and will replace its content. May be useful to avoid memory allocations (resize() will not shrink memory).
    // For efficiency, we will not check the inner consistency of the file here, but rather when we further process it.
    // (We already use the tag mechanism to check the rough structure.)
    // If this fails, the lattice is in unusable state, but it is OK to call fread() again to regain a usable object. I.e. this is safe to be used in retry loops.
    // This will also map the aligninfo entries to the new symbol table, through idmap.
    // V1 lattices will be converted. 'spsenoneid' is used in that process.
    template <class IDMAP>
    void fread(FILE* f, const IDMAP& idmap, size_t spunit)
    {
        size_t version = freadtag(f, "LAT ");
        if (version == 1)
        {
            freadOrDie(&info, sizeof(info), 1, f);
            freadvector(f, "NODE", nodes, info.numnodes);
            if (nodes.back().t != info.numframes)
                RuntimeError("fread: mismatch between info.numframes and last node's time");
            freadvector(f, "EDGE", edges, info.numedges);
            freadvector(f, "ALIG", align);
            fcheckTag(f, "END ");
            // map align ids to user's symmap  --the lattice gets updated in place here
            foreach_index (k, align)
                align[k].updateunit(idmap); // updates itself
        }
        else if (version == 2)
        {
            freadOrDie(&info, sizeof(info), 1, f);
            freadvector(f, "NODS", nodes, info.numnodes);
            if (nodes.back().t != info.numframes)
                RuntimeError("fread: mismatch between info.numframes and last node's time");
            freadvector(f, "EDGS", edges2, info.numedges); // uniqued edges
            freadvector(f, "ALNS", uniquededgedatatokens); // uniqued alignments
            fcheckTag(f, "END ");
            ProcessV2Lattice(spunit, info, uniquededgedatatokens, idmap);
        }
        else
            RuntimeError("fread: unsupported lattice format version");
    }

    // The same as fread above, but for buffer and only supporting lattice version 2.
    // Advances the buffer by reference.
    void ReadFromBuffer(const char* buffer, const std::vector<unsigned int>& idmap, size_t spunit)
    {
        ReadTagFromBuffer(buffer, "LAT ", 2);

        const header_v1_v2* pInfo = reinterpret_cast<const header_v1_v2*>(buffer);
        info = *pInfo;
        buffer += sizeof(header_v1_v2);

        ReadVectorFromBuffer(buffer, "NODS", nodes, info.numnodes);
        if (nodes.back().t != info.numframes)
            RuntimeError("ReadFromBuffer: mismatch between info.numframes and last node's time");
        ReadVectorFromBuffer(buffer, "EDGS", edges2, info.numedges); // uniqued edges
        ReadVectorFromBuffer(buffer, "ALNS", uniquededgedatatokens); // uniqued alignments
        CheckTag(buffer, "END ");
        ProcessV2Lattice(spunit, info, uniquededgedatatokens, idmap);
    }

    // Helper method to process v2 Lattice format
    template <class IDMAP>
    void ProcessV2Lattice(size_t spunit, header_v1_v2& info, std::vector<aligninfo>& uniquededgedatatokens, const IDMAP& idmap) 
    {
        // check if we need to map
        if (info.impliedspunitid != SIZE_MAX && info.impliedspunitid >= idmap.size()) // we have buggy lattices like that--what do they mean??
        {
            fprintf(stderr, "ProcessV2Lattice: detected buggy spunit id %d which is out of range (%d entries in map)\n", (int)info.impliedspunitid, (int)idmap.size());
            RuntimeError("ProcessV2Lattice: out of bounds spunitid");
        }

        // This is critical--we have a buggy lattice set that requires no mapping where mapping would fail
        bool needsmapping = false;
        foreach_index(k, idmap)
        {
            if (idmap[k] != (size_t)k
                && (k != (int)idmap.size() - 1 || idmap[k] != spunit) // that HACK that we add one more /sp/ entry at the end...
                )
            {
                needsmapping = true;
                break;
            }
        }
        // map align ids to user's symmap  --the lattice gets updated in place here
        if (needsmapping)
        {
            if (info.impliedspunitid != SIZE_MAX)
                info.impliedspunitid = idmap[info.impliedspunitid];

            // deal with broken (zero-token) edges
            std::vector<bool> isendworkaround;
            if (info.impliedspunitid != spunit)
            {
                fprintf(stderr, "ProcessV2Lattice: lattice with broken spunit, using workaround to handle potentially broken zero-token edges\n");
                inferends(isendworkaround);
            }

            size_t uniquealignments = 1;
            const size_t skipscoretokens = info.hasacscores ? 2 : 1;
            for (size_t k = skipscoretokens; k < uniquededgedatatokens.size(); k++)
            {
                if (!isendworkaround.empty() && isendworkaround[k]) // secondary criterion to detect ends in broken lattices
                {
                    k--; // don't advance, since nothing to advance over
                }
                else
                {
                    // this is a regular token: update it in-place
                    auto& ai = uniquededgedatatokens[k];
                    if (ai.unit >= idmap.size())
                        RuntimeError("ProcessV2Lattice: broken-file heuristics failed");
                    ai.updateunit(idmap); // updates itself
                    if (!ai.last)
                        continue;
                }
                // if last then skip over the lm and ac scores
                k += skipscoretokens;
                uniquealignments++;
            }
            fprintf(stderr, "ProcessV2Lattice: mapped %d unique alignments\n", (int)uniquealignments);
        }
        if (info.impliedspunitid != spunit)
        {
            // fprintf (stderr, "fread: inconsistent spunit id in file %d vs. expected %d; due to erroneous heuristic\n", info.impliedspunitid, spunit);    // [v-hansu] comment out becaues it takes up most of the log
            // it's actually OK, we can live with this, since we only decompress and then move on without any assumptions
            // RuntimeError("fread: mismatching /sp/ units");
        }
        // reconstruct old lattice format from this   --TODO: remove once we change to new data representation
        rebuildedges(info.impliedspunitid != spunit /*to be able to read somewhat broken V2 lattice archives*/);

    }

    // parallel versions (defined in parallelforwardbackward.cpp)
    class parallelstate
    {
        struct parallelstateimpl* pimpl;
        bool cpumode;

    public:
        parallelstate();
        ~parallelstate();
        bool enabled() const
        {
            return pimpl != NULL;
        }; // true if functions in here are available or not
        void copyalignments(edgealignments& edgealignments);
        void entercomputation(const class msra::asr::simplesenonehmm& hmms, const mbrclassdefinition mbrclassdef); // pass models in (to GPU)
        // no exitcomputation(); tear down the object instead
        struct parallelstateimpl* operator->()
        {
            return pimpl;
        } // to access the actual state (which are declared inside parallelstateimpl class)
        const struct parallelstateimpl* operator->() const
        {
            return pimpl;
        } // to access the actual state (which are declared inside parallelstateimpl class)
        const size_t getsilunitid();
        void getedgeacscores(std::vector<float>& edgeacscores);
        void getedgealignments(std::vector<unsigned short>& edgealignments);
        // to work with CNTK's GPU memory
        void setdevice(size_t DeviceId);
        size_t getdevice();
        void release(bool cpumode);
        void setloglls(const Microsoft::MSR::CNTK::Matrix<float>& loglls);
        void setloglls(const Microsoft::MSR::CNTK::Matrix<double>& loglls);
        void getgamma(Microsoft::MSR::CNTK::Matrix<float>& loglls);
        void getgamma(Microsoft::MSR::CNTK::Matrix<double>& loglls);
    };

    // forward-backward function
    // Note: logLLs and posteriors may be the same matrix (aliased).
    double forwardbackward(parallelstate& parallelstate, const class msra::math::ssematrixbase& logLLs, const class msra::asr::simplesenonehmm& hmms,
                           class msra::math::ssematrixbase& result, class msra::math::ssematrixbase& errorsignalbuf,
                           const float lmf, const float wp, const float amf, const float boostingfactor, const bool sMBRmode, array_ref<size_t> uids, const_array_ref<size_t> bounds = const_array_ref<size_t>(),
                           const_array_ref<htkmlfwordsequence::word> transcript = const_array_ref<htkmlfwordsequence::word>(), const std::vector<float>& transcriptunigrams = std::vector<float>()) const;

    std::wstring key; // (keep our own name (key) so we can identify ourselves for diagnostics messages)
    const wchar_t* getkey() const
    {
        return key.c_str();
    }

    void setverbosity(int veb) const
    {
        verbosity = veb;
    }
};

// ===========================================================================
// archive -- a disk-based archive of lattices
// Optimized for sequentially retrieving lattices in order of original archive
// building process.
// ===========================================================================

class archive
{
public:
    // set of phoneme mappings
    typedef std::vector<unsigned int> symbolidmapping;
    template <class SYMMAP>
    static void GetSymList(symbolidmapping& idmap, const std::wstring& symlistpath, const SYMMAP& symmap) 
    {
        std::vector<char> textbuffer;
        auto lines = msra::files::fgetfilelines(symlistpath, textbuffer);
        // establish mapping of each entry to the corresponding id in 'symmap'; this should fail if the symbol is not found
        idmap.reserve(lines.size() + 1); // last entry is a fake entry to return the /sp/ unit
        std::string symstring, tosymstring;
        foreach_index(i, lines)
        {
            char* sym = lines[i];
            // parse out a mapping  (log SPC phys)
            char* p = strchr(sym, ' ');
            if (p != NULL) // mapping: just verify that the supplied symmap has the same mapping
            {
                *p = 0;
                const char* tosym = p + 1;
                symstring = sym; // (reusing existing object to avoid malloc)
                tosymstring = tosym;
                if (getid(symmap, symstring) != getid(symmap, tosymstring))
                    RuntimeError("GetSymList: mismatching symbol id for %s vs. %s", sym, tosym);
            }
            else
            {
                if ((size_t)i != idmap.size()) // non-mappings must come first (this is to ensure compatibility with pre-mapping files)
                    RuntimeError("GetSymList: mixed up symlist file");
                symstring = sym; // (reusing existing object to avoid malloc)
                idmap.push_back((unsigned int)getid(symmap, symstring));
            }
        }
        // append a fixed-position entry: last entry means /sp/
        idmap.push_back((unsigned int)getid(symmap, "sp"));
    }

private:
    const std::unordered_map<std::string, size_t>& modelsymmap; // [triphone name] -> index used in model
    // set of lattice archive files referenced
    // Note that .toc files can be concatenated, i.e. one .toc file can reference multiple archive files.
    std::vector<std::wstring> archivepaths; // [archiveindex] -> archive path
    std::wstring prefixPathInToc;           // prefix path in a toc; using this to avoid pushd some path before start training
    mutable int verbosity;
    size_t getarchiveindex(const std::wstring& path) // get index of a path in archivepaths[]; create new entry if needed
    {
        auto iter = std::find(archivepaths.begin(), archivepaths.end(), path);
        size_t i = iter - archivepaths.begin();
        if (i == archivepaths.size())
            archivepaths.push_back(path);
        return i;
    }
    
    mutable std::vector<symbolidmapping> symmaps; // [archiveindex][unit] -> global unit map
    template <class SYMMAP>
    static size_t getid(const SYMMAP& symmap, const std::string& key)
    {
        auto iter = symmap.find(key);
        if (iter == symmap.end())
            RuntimeError("getcachedidmap: symbol not found in user-supplied symbol map: %s", key.c_str());
        return iter->second;
    }

    template <class SYMMAP>
    const symbolidmapping& getcachedidmap(size_t archiveindex, const SYMMAP& symmap /*[string] -> numeric id*/) const
    {
        symbolidmapping& idmap = symmaps[archiveindex];
        if (idmap.empty()) // TODO: delete this: && !modelsymmap.empty()/*no mapping; used in conversion*/)
        {                  // need to read the map and establish the mapping
            // get the symlist file
            const std::wstring symlistpath = archivepaths[archiveindex] + L".symlist";
            if (verbosity > 0)
                fprintf(stderr, "getcachedidmap: reading '%S'\n", symlistpath.c_str());
            archive::GetSymList(idmap, symlistpath, symmap);

        }
        return idmap;
    }
    // all lattices read so far
    struct latticeref
    {
        uint64_t offset : 48;
        uint64_t archiveindex : 16;
        latticeref(uint64_t offset, size_t archiveindex)
            : offset(offset), archiveindex(archiveindex)
        {
        }
    };
    static_assert(sizeof(latticeref) == 8, "unexpected byte size of struct latticeref");

    mutable size_t currentarchiveindex;               // which archive is open
    mutable auto_file_ptr f;                          // cached archive file handle of currentarchiveindex
    std::unordered_map<std::wstring, latticeref> toc; // [key] -> (file, offset)  --table of content (.toc file)
public:
    // construct = open the archive
    // archive() : currentarchiveindex (SIZE_MAX) {}
    void setverbosity(int veb) const
    {
        verbosity = veb;
    }
    // test if this object is loaded with anything (if not, an empty set of TOC paths was passed--meaning disable lattice mode)
    bool empty() const
    {
        return archivepaths.empty();
    }

    // construct from a list of TOC files
    archive(const std::vector<std::wstring>& tocpaths, const std::unordered_map<std::string, size_t>& modelsymmap, const std::wstring prefixPath = L"")
        : currentarchiveindex(SIZE_MAX), modelsymmap(modelsymmap), prefixPathInToc(prefixPath), verbosity(0)
    {
        if (tocpaths.empty()) // nothing to read--keep silent
            return;
        fprintf(stderr, "archive: opening %d lattice-archive TOC files ('%S' etc.)..", (int) tocpaths.size(), tocpaths[0].c_str());
        size_t onepercentage = tocpaths.size() / 100 ? tocpaths.size() / 100 : 1;
        foreach_index (i, tocpaths)
        {
            if ((i % onepercentage) == 0)
                fprintf(stderr, ".");
            open(tocpaths[i]);
        }
        fprintf(stderr, " %d total lattices referenced in %d archive files\n", (int) toc.size(), (int) archivepaths.size());
    }

    // open an archive
    // Can be called for multiple archives.
    // BUGBUG: NOT YET. We only really support one archive file at this point. Important to do that though.
    void open(const std::wstring& tocpath)
    {
        // BUGBUG: we only really support one archive file at this point
        // read the TOC in one swoop
        std::vector<char> textbuffer;
        auto toclines = msra::files::fgetfilelines(tocpath, textbuffer, 3);

        // parse it one by one
        size_t archiveindex = SIZE_MAX; // its index
        foreach_index (i, toclines)
        {
            const char* line = toclines[i];
            const char* p = strchr(line, '=');
            if (p == NULL)
                RuntimeError("open: invalid TOC line (no = sign): %s", line);
            const std::wstring key = msra::strfun::utf16(std::string(line, p - line));
            p++;
            const char* q = strchr(p, '[');
            if (q == NULL)
                RuntimeError("open: invalid TOC line (no [): %s", line);
            if (q != p)
            {
                std::wstring archivepath = msra::strfun::utf16(std::string(p, q - p));
                if (!prefixPathInToc.empty())
                {
                    archivepath = prefixPathInToc + L"/" + archivepath;
                }
                // TODO: should we allow paths relative to TOC file?
                archiveindex = getarchiveindex(archivepath);
            }
            if (archiveindex == SIZE_MAX)
                RuntimeError("open: invalid TOC line (empty archive pathname): %s", line);
            char c;
            uint64_t offset;
#ifdef _WIN32
            if (sscanf_s(q, "[%I64u]%c", &offset, &c, (unsigned int)sizeof(c)) != 1)
#else

            if (sscanf(q, "[%" PRIu64 "]%c", &offset, &c) != 1)
#endif
                RuntimeError("open: invalid TOC line (bad [] expression): %s", line);
            if (!toc.insert(make_pair(key, latticeref(offset, archiveindex))).second)
                RuntimeError("open: TOC entry leads to duplicate key: %s", line);
        }

        // initialize symmaps  --alloc the array, but actually read the symmap on demand
        symmaps.resize(archivepaths.size());
    }

    // check if a lattice for a given key is available  --do this during initial check ideally
    bool haslattice(const std::wstring& key) const
    {
        return toc.find(key) != toc.end();
    }

#if 0 // TODO: change design to keep the #frames in the TOC, so we can check for mismatches before entering the training iteration
    // return # frames for a key, or 0 if lattice not found (this combines the function of haslattice(), we save one lookup)
    size_t getlatticeframes (const std::wstring & key) const
    {
        auto iter = toc.find (key);
        if (iter == toc.end())
            return 0;
        else
            return iter->second->xyz;   // oops!
    }
#endif

    // get a lattice
    // This function is designed to be called from a retry loop due to the realistic chance of server disconnects or other server failures.
    // 'key' is supposed to be known to exist. Use haslattice() to ensure. This is because this function is called from a retry loop.
    // Lattices will have unit ids updated according to the modelsymmap.
    // V1 lattices will be converted. 'spsenoneid' is used in the conversion for optimizing storing 0-frame /sp/ aligns.
    void getlattice(const std::wstring& key, lattice& L,
                    size_t expectedframes = SIZE_MAX /*if unknown*/) const
    {
        auto iter = toc.find(key);
        if (iter == toc.end())
            LogicError("getlattice: requested lattice for non-existent key; haslattice() should have been used to check availability");
        // get the archive that the lattice lives in and its byte offset
        const size_t archiveindex = iter->second.archiveindex;
        const auto offset = iter->second.offset;
        // get id map (used below); this may lazily load a .symlist file. We do it here rather than later w.r.t. an outer retry loop.
        auto& idmap = getcachedidmap(archiveindex, modelsymmap); // at first time, this will load the .symlist file and create a mapping to the user SYMMAP
        const size_t spunit = idmap.back();                      // ugh--getcachedidmap() just appends it to the end
#if 1                                                            // prep for fixing the pushing of /sp/ at the end  --we actually can just look it up! Duh
        const size_t spunit2 = getid(modelsymmap, "sp");
        if (spunit2 != spunit)
            LogicError("getlattice: huh? same lookup of /sp/ gives different result?");
#endif
        // open archive file in case it is not the current one
        if (archiveindex != currentarchiveindex)
        {
            f = fopenOrDie(archivepaths[archiveindex], L"rbS"); // or throw (will close old 'f' iff succeeded)
            currentarchiveindex = archiveindex;
        }
        try // (for read operation)
        {
            // seek to start
            fsetpos(f, offset);
            // get it
            L.fread(f, idmap, spunit);
            L.setverbosity(verbosity);
#ifdef HACK_IN_SILENCE // hack to simulate DEL in the lattice
            const size_t silunit = getid(modelsymmap, "sil");
            const bool addsp = true;
            L.hackinsilencesubstitutionedges(silunit, spunit, addsp);
#endif
        }
        catch (...) // to retry a read error due to a disconnected file handle, we need to reopen the file
        {
            currentarchiveindex = SIZE_MAX;
            f = NULL; // this closes the file handle
            throw;
        }
        // check if number of frames is as expected
        if (expectedframes != SIZE_MAX && L.getnumframes() != expectedframes)
            LogicError("getlattice: number of frames mismatch between numerator lattice and features");
        // remember the latice key for diagnostics messages
        L.key = key;
    };

    // static method for building an archive
    static void build(const std::vector<std::wstring>& infiles, const std::wstring& outpath,
                      const std::unordered_map<std::string, size_t>& modelsymmap,
                      const msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence>& labels,
                      const msra::lm::CMGramLM& lm, const msra::lm::CSymbolSet& unigramsymbols);

    // static method for converting an archive to a new format
    // Extended features:
    //  - check consistency (don't write out)
    //  - dump to stdout
    //  - merge two lattices (for merging numer into denom lattices)
    static void convert(const std::wstring& intocpath, const std::wstring& intocpath2, const std::wstring& outpath,
                        const msra::asr::simplesenonehmm& hset);
};
};
};
