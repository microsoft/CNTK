// latticearchive.cpp -- managing lattice archives
//
// F. Seide, V-hansu

#include "Basics.h"
#include "simple_checked_arrays.h"
#include "latticearchive.h"
#include "simplesenonehmm.h"    // the model
#include "ssematrix.h"          // the matrices
#include "latticestorage.h"
#include <unordered_map>
#include <list>
#include <stdexcept>

using namespace std;

#define VIRGINLOGZERO (10 * LOGZERO)            // used for printing statistics on unseen states
#undef CPU_VERIFICATION

#ifdef _WIN32
int msra::numa::node_override = -1;     // for numahelpers.h
#endif

namespace msra { namespace lattices {

// ---------------------------------------------------------------------------
// helper class for allocation lots of small matrices, no free
// ---------------------------------------------------------------------------

class littlematrixheap
{
    static const size_t CHUNKSIZE;
    typedef msra::math::ssematrixfrombuffer matrixfrombuffer;
    std::list<std::vector<float>> heap;
    size_t allocatedinlast; // in last heap element
    size_t totalallocated;
    std::vector<matrixfrombuffer> matrices;
public:
    littlematrixheap (size_t estimatednumentries) : totalallocated (0), allocatedinlast (0) { matrices.reserve (estimatednumentries + 1); }
    msra::math::ssematrixbase & newmatrix (size_t rows, size_t cols)
    {
        const size_t elementsneeded = matrixfrombuffer::elementsneeded (rows, cols);
        if (heap.empty() || (heap.back().size() - allocatedinlast) < elementsneeded)
        {
            const size_t nelem = max (CHUNKSIZE, elementsneeded + 3/*+3 for SSE alignment*/);
            heap.push_back (std::vector<float> (nelem));
            allocatedinlast = 0;
            // make sure starting element is SSE-aligned (the constructor demands that)
            const size_t offelem = (((size_t)&heap.back()[allocatedinlast]) / sizeof (float)) % 4;
            if (offelem != 0)
                allocatedinlast += 4 - offelem;
        }
        auto & buffer = heap.back();
        if (elementsneeded > heap.back().size() - allocatedinlast)
            LogicError("newmatrix: allocation logic screwed up");
        // get our buffer into a handy vector-like thingy
        array_ref<float> vecbuffer (&buffer[allocatedinlast], elementsneeded);
        // allocate in the current heap location
        matrices.resize (matrices.size() + 1);
        if (matrices.size()+1 > matrices.capacity())
            LogicError("newmatrix: littlematrixheap cannot grow but was constructed with too small number of eements");
        auto & matrix = matrices.back();
        matrix = matrixfrombuffer (vecbuffer, rows, cols);
        allocatedinlast += elementsneeded;
        totalallocated += elementsneeded;
        return matrix;
    }
};

const size_t littlematrixheap::CHUNKSIZE = 256*1024; // 1 MB

// ---------------------------------------------------------------------------
// helpers for log-domain addition
// ---------------------------------------------------------------------------

#ifndef LOGZERO
#define LOGZERO -1e30f
#endif

// logadd (loga, logb) -> a += b, or loga = log [ exp(loga) + exp(logb) ]
static void logaddratio (float & loga, float diff)
{
    if (diff < -17.0f) return;      // log (2^-24), 23-bit mantissa -> cut of after 24th bit
    loga += logf (1.0f + expf (diff));
}
static void logaddratio (double & loga, double diff)
{
    if (diff < -37.0f) return;      // log (2^-53), 52-bit mantissa -> cut of after 53th bit
    loga += log (1.0 + exp (diff));
}
// loga <- log (exp (loga) + exp (logb)) = log (exp (loga) * (1.0 + exp (logb - loga)) = loga + log (1.0 + exp (logb - loga))
template<typename FLOAT> static void logadd (FLOAT & loga, FLOAT logb)
{
    if (logb > loga)            // we add smaller to bigger
        ::swap (loga, logb);
    if (loga <= LOGZERO)        // both are 0
        return;
    logaddratio (loga, logb - loga);
}
template<typename FLOAT> static void logmax (FLOAT & loga, FLOAT logb)  // for testing (max approx)
{
    if (logb > loga)
        loga = logb;
}

template<typename FLOAT> static FLOAT expdiff (FLOAT a, FLOAT b)  // for testing
{
    if (b > a)
        return exp(b) * (exp(a-b) - 1);
    else
        return exp(a) * (1 - exp(b-a));
}

template<typename FLOAT> static bool islogzero (FLOAT v) { return v < LOGZERO/2; }  // is this number to be considered 0

// ---------------------------------------------------------------------------
// other helpers go here
// ---------------------------------------------------------------------------

// helper to reconstruct the phonetic transcript
/*static*/ std::string lattice::gettranscript (const_array_ref<aligninfo> units, const msra::asr::simplesenonehmm & hset)
{
    std::string trans;
    foreach_index (k, units)    // we exploit that units have fixed boundaries
    {
        if (k > 0) trans.push_back (' ');
        trans.append (hset.gethmm (units[k].unit).getname());
    }
    return trans;
}

// ---------------------------------------------------------------------------
// forwardbackwardedge() -- perform state-level forward-backward on a single lattice edge
//
// Results:
//  - gammas(j,t) for valid time ranges (remaining areas are not initialized)
//  - return value is edge acoustic score
// Gammas matrix must have two extra columns as buffer.
// ---------------------------------------------------------------------------

/*static*/ float lattice::forwardbackwardedge (const_array_ref<aligninfo> units, const msra::asr::simplesenonehmm & hset, const msra::math::ssematrixbase & logLLs, 
                                               msra::math::ssematrixbase & loggammas, size_t edgeindex)
{
    // alphas and betas are stored in-place inside the loggammas matrix shifted by one?two columns
    assert (loggammas.cols() == logLLs.cols() + 2);
    msra::math::ssematrixstriperef<msra::math::ssematrixbase> logalphas (loggammas, 1, logLLs.cols());    // shifted views into gammas(,) for alphas and betas
    msra::math::ssematrixstriperef<msra::math::ssematrixbase> logbetas (loggammas, 2, logLLs.cols());

    // alphas(j,t) store the sum of all paths up to including state j at time t, including logLL(j,t)
    // betas(j,t) store the sum of all paths exiting from state j at time t, not including logLL(j,t)
    // gammas(j,t) = alphas(j,t) * betas(j,t) / totalLL

    // backward pass   --token passing
    size_t te = logbetas.cols();
    size_t je = logbetas.rows();
    float bwscore = 0.0f;                        // backward score
    for (size_t k = units.size() -1; k+1 > 0; k--)
    {
        const auto & hmm = hset.gethmm (units[k].unit);
        const size_t n = hmm.getnumstates();
        const auto & transP = hmm.gettransP();
        const size_t ts = te - units[k].frames;     // end time of current unit
        const size_t js = je - n;                   // range of state indices

        // pass in the transition score
        // t = ts: exit transition (last frame only or tee transition)
        float exitscore = 1e30f; // (something impossible)
        if (te == ts)                       // tee transition
        {
            exitscore = bwscore + transP(-1,n);
        }
        else                                // not tee: expand all last states
        {
            for (size_t from = 0/*no tee possible here*/; from < n; from++)
            {
                const size_t i = js + from; // origin trellis node
                logbetas(i,te-1) = bwscore + transP(from,n);
            }
        }

        // expand from states j at time t (not yet including LL) to time t-1
        for (size_t t = te -1; t+1 > ts/*note: cannot test t >= ts because t < 0 possible*/; t--)
        {
            for (size_t to = 0; to < n; to++)
            {
                const size_t j = js + to;               // source trellis node
                const size_t s = hmm.getsenoneid(to);   // senone id for state at position 'to' in the HMM
                const float acLL = logLLs(s,t);
                if (islogzero (acLL))
                    fprintf (stderr, "forwardbackwardedge: WARNING: edge J=%d unit %d (%s) frames [%d,%d) ac score(%d,%d) is zero (%d st, %d fr: %s)\n",
                    (int)edgeindex, (int) k, hmm.getname(), (int) ts, (int) te,
                    (int) s, (int) t,
                    (int) logbetas.rows(), (int) logbetas.cols(), gettranscript (units, hset).c_str());
                const float betajt = logbetas(j,t);     // sum over all all path exiting from (j,t) to end
                const float betajtpll = betajt + acLL;  // incorporate acoustic score
                if (t > ts) for (size_t from = 0/*no transition from entry state*/; from < n; from++)
                {
                    const size_t i = js + from;    // target trellis node
                    const float pathscore = betajtpll + transP(from,to);
                    if (to == 0)
                        logbetas(i,t-1/*propagate into preceding frame*/) = pathscore;
                    else
                        logadd (logbetas(i,t-1/*propagate into preceding frame*/), pathscore);
                }
                else                            // transition to entry state
                {
                    const float pathscore = betajtpll + transP(-1,to);
                    if (to == 0)
                        exitscore = pathscore;
                    else
                        logadd (exitscore, pathscore);  // propagate into preceding unit
                }
            }
        }

        bwscore = exitscore;
        if (islogzero (bwscore))
            fprintf (stderr, "forwardbackwardedge: WARNING: edge J=%d unit %d (%s) frames [%d,%d) bw score is zero (%d st, %d fr: %s)\n",
            (int)edgeindex, (int) k, hmm.getname(), (int) ts, (int) te, (int) logbetas.rows(), (int) logbetas.cols(), gettranscript (units, hset).c_str());

        te = ts;
        je = js;
    }
    assert (te == 0 && je == 0);
    const float totalbwscore = bwscore;

    // forward pass   --regular Viterbi
    // This also computes the gammas right away.
    size_t ts = 0;              // start frame for unit 'k'
    size_t js = 0;              // first row index of unit ' k'
    float fwscore = 0.0f;       // score passed across phone boundaries
    foreach_index (k, units)    // we exploit that units have fixed boundaries
    {
        const auto & hmm = hset.gethmm (units[k].unit);
        const size_t n = hmm.getnumstates();
        const auto & transP = hmm.gettransP();
        const size_t te = ts + units[k].frames;     // end time of current unit
        const size_t je = js + n;                   // range of state indices

        // expand from states j at time t (including LL) to time t+1
        for (size_t t = ts; t < te; t++)            // note: loop not entered for 0-frame units (tees)
        {
            for (size_t to = 0; to < n; to++)
            {
                const size_t j = js + to;           // target trellis node
                const size_t s = hmm.getsenoneid(to);
                const float acLL = logLLs(s,t);
                float alphajtnoll = LOGZERO;
                if (t == ts)                        // entering score
                {
                    const float pathscore = fwscore + transP(-1,to);
                    alphajtnoll = pathscore;
                }
                else for (size_t from = 0/*no entering possible*/; from < n; from++)
                {
                    const size_t i = js + from;     // origin trellis node
                    const float alphaitm1 = logalphas(i,t-1/*previous frame*/);
                    const float pathscore = alphaitm1 + transP(from,to);
                    logadd (alphajtnoll, pathscore);
                }
                logalphas(j,t) = alphajtnoll + acLL;
            }
            // update the gammas  --do it here because in next frame, betas get overwritten by alphas (they share memory)
            for (size_t j = js; j < je; j++)
            {
                if (!islogzero (totalbwscore))
                    loggammas(j,t) = logalphas(j,t) + logbetas(j,t) - totalbwscore;
                else        // 0/0 problem, can occur if an ac score is so bad that it is 0 after going through softmax
                    loggammas(j,t) = LOGZERO;
            }
        }
        // t = te: exit transition (last frame only or tee transition)
        float exitscore;
        if (te == ts)                       // tee transition
        {
            exitscore = fwscore + transP(-1,n);
        }
        else                                // not tee: expand all last states
        {
            exitscore = LOGZERO;
            for (size_t from = 0/*no tee possible here*/; from < n; from++)
            {
                const size_t i = js + from; // origin trellis node
                const float alphaitm1 = logalphas(i,te-1); // newly computed path score, transiting to t=te
                const float pathscore = alphaitm1 + transP(from,n);
                logadd (exitscore, pathscore);
            }
        }
        fwscore = exitscore;    // score passed on to next unit
        js = je;
        ts = te;
    }
    assert (js == logalphas.rows() && ts == logalphas.cols());
    const float totalfwscore = fwscore;

    // in extreme cases, we may have 0 ac probs, which lead to 0 path scores and division by 0 (subtracting LOGZERO)
    // These cases must be handled separately. If the whole path is 0 (0 prob is on the only path at some point) then skip the lattice.
    if (islogzero (totalbwscore) ^ islogzero (totalfwscore))
        fprintf (stderr, "forwardbackwardedge: WARNING: edge J=%d fw and bw 0 score %.10f vs. %.10f (%d st, %d fr: %s)\n",
        (int)edgeindex, (float) totalfwscore, (float) totalbwscore, (int) js, (int) ts, gettranscript (units, hset).c_str());
    if (islogzero (totalbwscore))
    {
        fprintf (stderr, "forwardbackwardedge: WARNING: edge J=%d has zero ac. score (%d st, %d fr: %s)\n",
                 (int)edgeindex, (int) js, (int) ts, gettranscript (units, hset).c_str());
        return LOGZERO;
    }

    if (fabsf (totalfwscore - totalbwscore) / ts > 1e-4f)
        fprintf (stderr, "forwardbackwardedge: WARNING: edge J=%d fw and bw score %.10f vs. %.10f (%d st, %d fr: %s)\n",
        (int)edgeindex, (float) totalfwscore, (float) totalbwscore, (int) js, (int) ts, gettranscript (units, hset).c_str());

    // we return the full path score
    return totalfwscore;

}

// ---------------------------------------------------------------------------
// alignedge() -- perform Viterbi alignment on a single edge
//
// This is an alternative to forwardbackwardedge() that just uses the best path.
// Results:
//  - if not returnsenoneids -> 'binary gammas(j,t)' for valid time ranges (remaining areas are not initialized); MMI-compatible
//  - if returnsenoneids ->  loggammas(0,t) will contain the senone ids directly instead (for sMBR mode)
//  - return value is edge acoustic score
// Gammas matrix must have two extra columns as buffer.
// ---------------------------------------------------------------------------

/*static*/ float lattice::alignedge (const_array_ref<aligninfo> units, const msra::asr::simplesenonehmm & hset, const msra::math::ssematrixbase & logLLs,
                                     msra::math::ssematrixbase & loggammas, size_t edgeindex/*for diagnostic messages*/, const bool returnsenoneids,
                                     array_ref<unsigned short> thisedgealignmentsj)
{
    // alphas and betas are stored in-place inside the loggammas matrix shifted by one?two columns
    assert (loggammas.cols() == logLLs.cols() + 2);
    msra::math::ssematrixstriperef<msra::math::ssematrixbase> backpointers (loggammas, 0, logLLs.cols());
    msra::math::ssematrixstriperef<msra::math::ssematrixbase> pathscores (loggammas, 2, logLLs.cols());

    // pathscores(j,t) store the sum of all paths up to including state j at time t, including logLL(j,t)
    // backpointers(j,t) are the relative states that it came from
    // gammas(j,t) <- 1 if on best path, 0 otherwise

    const int invalidbp = -2;

    // Viterbi alignment
    size_t ts = 0;              // start frame for unit 'k'
    size_t js = 0;              // first row index of unit 'k'
    float fwscore = 0.0f;       // score passed across phone boundaries
    int fwbackpointer = -1;     // bp passed across phone boundaries, -1 means start of utterance
    foreach_index (k, units)    // we exploit that units have fixed boundaries
    {
        const auto & hmm = hset.gethmm (units[k].unit);
        const size_t n = hmm.getnumstates();
        const auto & transP = hmm.gettransP();
        const size_t te = ts + units[k].frames;     // end time of current unit
        const size_t je = js + hmm.getnumstates();  // range of state indices

        // expand from states j at time t (including LL) to time t+1
        for (size_t t = ts; t < te; t++)            // note: loop not entered for 0-frame units (tees)
        {
            for (size_t j = js; j < je; j++)
            {
                const size_t to = j - js;           // relative state
                const size_t s = hmm.getsenoneid(to);
                pathscores(j,t) = LOGZERO;
                backpointers(j,t) = invalidbp;
                if (t == ts)                        // entering score
                {
                    const float pathscore = fwscore + transP(-1,to);
                    pathscores(j,t) = pathscore;
                    backpointers(j,t) = (float) fwbackpointer;
                }
                else for (size_t i = js; i < je; i++)
                {
                    const size_t from = i - js;
                    const float alphaitm1 = pathscores(i,t-1/*previous frame*/);
                    const float pathscore = alphaitm1 + transP(from,to);
                    if (pathscore > pathscores(j,t))
                    {
                        pathscores(j,t) = pathscore;
                        backpointers(j,t) = (float) i;
                    }
                }
                const float acLL = logLLs(s,t);
                pathscores(j,t) += acLL;
            }
        }
        // t = te: exit transition (last frame only or tee transition)
        float exitscore = LOGZERO;
        int exitbackpointer = invalidbp;
        if (te == ts)                       // tee transition
        {
            exitscore = fwscore + transP(-1,n);
            exitbackpointer = fwbackpointer;
        }
        else                                // not tee: expand all last states
        {
            for (size_t i = js; i < je; i++)
            {
                const size_t from = i - js;
                const float alphaitm1 = pathscores(i,te-1); // newly computed path score, transiting to t=te
                const float pathscore = alphaitm1 + transP(from,n);
                if (pathscore > exitscore)
                {
                    exitscore = pathscore;
                    exitbackpointer = (int) i;
                }
            }
        }
        if (exitbackpointer == invalidbp)
            LogicError("exitbackpointer came up empty");
        fwscore = exitscore;                // score passed on to next unit
        fwbackpointer = exitbackpointer;    // and accompanying backpointer
        js = je;
        ts = te;
    }
    assert (js == pathscores.rows() && ts == pathscores.cols());

    // in extreme cases, we may have 0 ac probs, which lead to 0 path scores and division by 0 (subtracting LOGZERO)
    // These cases must be handled separately. If the whole path is 0 (0 prob is on the only path at some point) then skip the lattice.
    if (islogzero (fwscore))
    {
        fprintf (stderr, "alignedge: WARNING: edge J=%d has zero ac. score (%d st, %d fr: %s)\n",
                 (int)edgeindex, (int) js, (int) ts, gettranscript (units, hset).c_str());
        return LOGZERO;
    }

    // traceback & gamma update
    size_t te = backpointers.cols();
    size_t je = backpointers.rows();
    int j = fwbackpointer;
    for (size_t k = units.size() -1; k+1 > 0; k--)  // go in units because we also need to clear out the column
    {
        const auto & hmm = hset.gethmm (units[k].unit);
        const size_t ts = te - units[k].frames;     // end time of current unit
        const size_t js = je - hmm.getnumstates();  // range of state indices
        for (size_t t = te -1; t + 1 > ts; t--)
        {
            if (j < (int) js || j >= (int) je)
                LogicError("invalid backpointer resulting in state index out of range");

            int bp = (int) backpointers(j,t);   // save the backpointer before overwriting it (gammas and backpointers are aliases of each other)
            if (!returnsenoneids)               // return binary gammas (for MMI; this mode is compatible with softalignmode)
                for (size_t i = js; i < je; i++)
                    loggammas(i,t) = ((int) i == j) ? 0.0f : LOGZERO;
            else                                // return senone id (for sMBR; note: NOT compatible with softalignmode; calling code must know this)
                thisedgealignmentsj[t] = (unsigned short) hmm.getsenoneid(j-js);

            if (bp == invalidbp)
                LogicError("deltabackpointer not initialized");
            j = bp; // trace back one step
        }

        te = ts;
        je = js;
    }
    if (j != -1)
        LogicError("invalid backpointer resulting in not reaching start of utterance when tracing back");
    assert (je == 0 && te == 0);

    // we return the full path score
    return fwscore;
}

// ---------------------------------------------------------------------------
// forwardbackwardlattice() -- lattice-level forward/backward
//
// This computes word posteriors, and also returns the per-node alphas and betas.
// Per-edge acoustic scores are passed in via a lambda, as this function is
// intended for use at multiple places with different scores.
// (Specifically, we also use it to determine a pruning threshold, based on
// the original lattice's ac. scores, before even bothering to compute the
// new ac. scores.)
// ---------------------------------------------------------------------------

double lattice::forwardbackwardlattice (const std::vector<float> & edgeacscores, parallelstate & parallelstate, std::vector<double> & logpps,
                                        std::vector<double> & logalphas, std::vector<double> & logbetas,
                                        const float lmf, const float wp, const float amf, const float boostingfactor, const bool sMBRmode,
                                        const_array_ref<size_t> & uids, const edgealignments & thisedgealignments, 
                                        std::vector<double> & logEframescorrect, std::vector<double> & Eframescorrectbuf, double & logEframescorrecttotal) const
{                                                                                                          // ^^ TODO: remove this
    // --- hand off to parallelized (CUDA) implementation if available
    if (parallelstate.enabled())
    {
        double totalfwscore = parallelforwardbackwardlattice (parallelstate, edgeacscores, thisedgealignments, lmf, wp, amf, boostingfactor, logpps, logalphas, logbetas, sMBRmode, uids, logEframescorrect, Eframescorrectbuf, logEframescorrecttotal);

        return totalfwscore;
    }
    // if we get here, we have no CUDA, and do it the good ol' way

    // allocate return values
    logpps.resize (edges.size());   // this is our primary return value

    // TODO: these are return values as well, but really shouldn't anymore; only used in some older baseline code we some day may want to compare against
    logalphas.assign (nodes.size(), LOGZERO);
    logalphas.front() = 0.0f;
    logbetas.assign (nodes.size(), LOGZERO);
    logbetas.back() = 0.0f;

    // --- sMBR version

    if (sMBRmode)
    {
        logEframescorrect.resize (edges.size());
        Eframescorrectbuf.resize (edges.size());

        std::vector<double> logaccalphas (nodes.size(), LOGZERO);   // [i] expected frames-correct count over all paths from start to node i
        std::vector<double> logaccbetas (nodes.size(), LOGZERO);    // [i] likewise
        std::vector<double> logframescorrectedge (edges.size());    // raw counts of correct frames in each edge

        // forward pass
        foreach_index (j, edges)
        {
            if (islogzero(edgeacscores[j]))                         // indicates that this edge is pruned
                continue;
            const auto & e = edges[j];
            const double inscore = logalphas[e.S];
            const double edgescore = (e.l * lmf + wp + edgeacscores[j]) / amf;
            const double pathscore = inscore + edgescore;
            logadd (logalphas[e.E], pathscore);

            size_t ts = nodes[e.S].t;
            size_t te = nodes[e.E].t;
            size_t framescorrect = 0;                               // count raw number of correct frames
            for (size_t t = ts; t < te; t++)
                framescorrect += (thisedgealignments[j][t-ts] == uids[t]);
            logframescorrectedge[j] = (framescorrect > 0) ? log ((double) framescorrect) : LOGZERO;          // remember for backward pass
            double loginaccs = logaccalphas[e.S] - logalphas[e.S];
            logadd (loginaccs, logframescorrectedge[j]);
            double logpathacc = loginaccs + logalphas[e.S] + edgescore;
            logadd (logaccalphas[e.E], logpathacc);
        }
        foreach_index (j, logaccalphas)
            logaccalphas[j] -= logalphas[j];

        const double totalfwscore = logalphas.back();
        const double totalfwacc = logaccalphas.back();
        if (islogzero (totalfwscore))
        {
            fprintf (stderr, "forwardbackward: WARNING: no path found in lattice (%d nodes/%d edges)\n", (int) nodes.size(), (int) edges.size());
            return LOGZERO;         // failed, do not use resulting matrix
        }

        // backward pass and computation of state-conditioned frames-correct count
        for (size_t j = edges.size() -1; j+1 > 0; j--)
        {
            if (islogzero(edgeacscores[j]))                         // indicates that this edge is pruned
                continue;
            const auto & e = edges[j];
            const double inscore = logbetas[e.E];
            const double edgescore = (e.l * lmf + wp + edgeacscores[j]) / amf;
            const double pathscore = inscore + edgescore;
            logadd (logbetas[e.S], pathscore);
            
            double loginaccs = logaccbetas[e.E] - logbetas[e.E];
            logadd (loginaccs, logframescorrectedge[j]);
            double logpathacc = loginaccs + logbetas[e.E] + edgescore;
            logadd (logaccbetas[e.S], logpathacc);

            // sum up to get final expected frames-correct count per state == per edge (since we assume hard state alignment)
            double logpp = logalphas[e.S] + edgescore + logbetas[e.E] - totalfwscore;
            if (logpp > 1e-2)
                fprintf (stderr, "forwardbackward: WARNING: edge J=%d log posterior %.10f > 0\n", (int) j, (float) logpp);
            if (logpp > 0.0)
                logpp = 0.0;
            logpps[j] = logpp;
            double tmplogeframecorrect = logframescorrectedge[j];
            logadd (tmplogeframecorrect, logaccalphas[e.S]);
            logadd (tmplogeframecorrect, logaccbetas[e.E] - logbetas[e.E]);
            Eframescorrectbuf[j] = exp(tmplogeframecorrect);
        }
        foreach_index (j, logaccbetas)
            logaccbetas[j] -= logbetas[j];
        const double totalbwscore = logbetas.front();
        const double totalbwacc = logaccbetas.front();
        if (fabs (totalfwscore - totalbwscore) / info.numframes > 1e-4)
            fprintf (stderr, "forwardbackward: WARNING: lattice fw and bw scores %.10f vs. %.10f (%d nodes/%d edges)\n", (float) totalfwscore, (float) totalbwscore, (int) nodes.size(), (int) edges.size());
            
        if (fabs (totalfwacc - totalbwacc) / info.numframes > 1e-4)
            fprintf (stderr, "forwardbackwardlatticesMBR: WARNING: lattice fw and bw accs %.10f vs. %.10f (%d nodes/%d edges)\n", (float) totalfwacc, (float) totalbwacc, (int) nodes.size(), (int) edges.size());

        logEframescorrecttotal = totalbwacc;
        return totalbwscore;
    }

    // --- MMI version

    // forward pass
    foreach_index (j, edges)
    {
        const auto & e = edges[j];
        const double inscore = logalphas[e.S];
        const double edgescore = (e.l * lmf + wp + edgeacscores[j]) / amf;  // note: edgeacscores[j] == LOGZERO if edge was pruned
        const double pathscore = inscore + edgescore;
        logadd (logalphas[e.E], pathscore);
    }
    const double totalfwscore = logalphas.back();
    if (islogzero (totalfwscore))
    {
        fprintf (stderr, "forwardbackward: WARNING: no path found in lattice (%d nodes/%d edges)\n", (int) nodes.size(), (int) edges.size());
        return LOGZERO;         // failed, do not use resulting matrix
    }

    // backward pass
    // this also computes the word posteriors on the fly, since we are at it
    for (size_t j = edges.size() -1; j+1 > 0; j--)
    {
        const auto & e = edges[j];
        const double inscore = logbetas[e.E];
        const double edgescore = (e.l * lmf + wp + edgeacscores[j]) / amf;
        const double pathscore = inscore + edgescore;
        logadd (logbetas[e.S], pathscore);

        // compute lattice posteriors on the fly since we are at it
        double logpp = logalphas[e.S] + edgescore + logbetas[e.E] - totalfwscore;
        if (logpp > 1e-2)
            fprintf (stderr, "forwardbackward: WARNING: edge J=%d log posterior %.10f > 0\n", (int) j, (float) logpp);
        if (logpp > 0.0)
            logpp = 0.0;
        logpps[j] = logpp;
    }

    const double totalbwscore = logbetas.front();
    if (fabs (totalfwscore - totalbwscore) / info.numframes > 1e-4)
        fprintf (stderr, "forwardbackward: WARNING: lattice fw and bw scores %.10f vs. %.10f (%d nodes/%d edges)\n", (float) totalfwscore, (float) totalbwscore, (int) nodes.size(), (int) edges.size());

    return totalfwscore;
}

// ---------------------------------------------------------------------------
// forwardbackwardlatticesMBR() -- compute expected frame-accuracy counts,
// both the conditioned one (corresponding to c(q) in Dan Povey's thesis)
// and the global one (which is the sMBR criterion to optimize).
//
// Outputs:
//  - Eframescorrect[j] == expected frames-correct count conditioned on a state of edge[j].
//    We currently assume a hard state alignment. With that, the value turns out
//    to be identical for all states of an edge, so we only store it once per edge.
//  - return value: expected frames-correct count for entire lattice
//
// Call forwardbackwardlattices() first to compute logalphas/betas.
// ---------------------------------------------------------------------------

double lattice::forwardbackwardlatticesMBR (const std::vector<float> & edgeacscores, const msra::asr::simplesenonehmm & hset,
                                            const std::vector<double> & logalphas, const std::vector<double> & logbetas,
                                            const float lmf, const float wp, const float amf, const_array_ref<size_t> & uids, 
                                            const edgealignments & thisedgealignments, std::vector<double> & Eframescorrect) const
{
    std::vector<double> accalphas (nodes.size(), 0);            // [i] expected frames-correct count over all paths from start to node i
    std::vector<double> accbetas (nodes.size(), 0);             // [i] likewise
    std::vector<size_t> maxcorrect (nodes.size(), 0);           // [i] max correct frames up to this node (oracle)

    std::vector<double> framescorrectedge (edges.size());       // raw counts of correct frames in each edge

    std::vector<int> backpointersformaxcorr(nodes.size(), -2);            // keep track of backpointer for the max corr
    backpointersformaxcorr.front() = -1;

    // forward pass
    foreach_index (j, edges)
    {
        if (islogzero(edgeacscores[j]))                         // indicates that this edge is pruned
            continue;
        const auto & e = edges[j];
        const double inaccs = accalphas[e.S];
        size_t ts = nodes[e.S].t;
        size_t te = nodes[e.E].t;

        size_t framescorrect = 0;                               // count raw number of correct frames
        for (size_t t = ts; t < te; t++)
            framescorrect += (thisedgealignments[j][t-ts] == uids[t]);
        framescorrectedge[j] = (double) framescorrect;          // remember for backward pass

        const double edgescore = (e.l * lmf + wp + edgeacscores[j]) / amf;
        // contribution to end node's path acc = start node's plus edge's correct count, weighted by LL, and divided by sum over LLs
        double pathacc = (inaccs + framescorrectedge[j]) * exp (logalphas[e.S] + edgescore - logalphas[e.E]);
        accalphas[e.E] += pathacc;
        // also keep track of max accuracy, so we can find out whether the lattice contains the correct path
        size_t oracleframescorrect = maxcorrect[e.S] + framescorrect;  // keep track of most correct path up to end of this edge
        if (oracleframescorrect > maxcorrect[e.E])
        {
            maxcorrect[e.E] = oracleframescorrect;
            backpointersformaxcorr[size_t(e.E)] = j; 
        }
    }
    const double totalfwacc = accalphas.back();

    hset;           // just for reference

    // report on ground-truth path
    // TODO: we will later have code that adds this path if needed
    size_t oracleframeacc = maxcorrect.back();
    if (oracleframeacc != info.numframes)
        fprintf (stderr, "forwardbackwardlatticesMBR: ground-truth path missing from lattice (most correct path: %d out of %d frames correct)\n", (unsigned int) oracleframeacc, (int) info.numframes);

    // backward pass and computation of state-conditioned frames-correct count
    for (size_t j = edges.size() -1; j+1 > 0; j--)
    {
        if (islogzero(edgeacscores[j]))                         // indicates that this edge is pruned
            continue;
        const auto & e = edges[j];
        const double inaccs = accbetas[e.E];
        const double edgescore = (e.l * lmf + wp + edgeacscores[j]) / amf;
        double pathacc = (inaccs + framescorrectedge[j]) * exp (logbetas[e.E] + edgescore - logbetas[e.S]);
        accbetas[e.S] += pathacc;

        // sum up to get final expected frames-correct count per state == per edge (since we assume hard state alignment)
        Eframescorrect[j] = (float) (accalphas[e.S] + accbetas[e.E] + framescorrectedge[j]);
    }

    const double totalbwacc = accbetas.front();

    if (fabs (totalfwacc - totalbwacc) / info.numframes > 1e-4)
        fprintf (stderr, "forwardbackwardlatticesMBR: WARNING: lattice fw and bw accs %.10f vs. %.10f (%d nodes/%d edges)\n", (float) totalfwacc, (float) totalbwacc, (int) nodes.size(), (int) edges.size());

    return totalbwacc;
}

// ---------------------------------------------------------------------------
// bestpathlattice() -- lattice-level "forward/backward" that only returns the
// best path, but in the form of word posteriors, which are 1 or 0, just like
// a real lattice-level forward/backward would do.
// We don't really use this; this was only for a contrast experiment.
// ---------------------------------------------------------------------------

double lattice::bestpathlattice (const std::vector<float> & edgeacscores, std::vector<double> & logpps,
                                 const float lmf, const float wp, const float amf) const
{
    // forward pass --sortnedness => regular Viterbi
    std::vector<double> logalphas (nodes.size(), LOGZERO);
    std::vector<int> backpointers(nodes.size(), -2);
    logalphas.front() = 0.0f;
    backpointers.front() = -1;
    foreach_index (j, edges)
    {
        const auto & e = edges[j];
        const double inscore = logalphas[e.S];
        const double edgescore = (e.l * lmf + wp + edgeacscores[j]) / amf;  // note: edgeacscores[j] == LOGZERO if edge was pruned
        const double pathscore = inscore + edgescore;
        if (pathscore > logalphas[e.E])
        {
            logalphas[e.E] = pathscore;
            backpointers[e.E] = j;
        }
    }

    const double totalfwscore = logalphas.back();
    if (islogzero (totalfwscore))
    {
        fprintf (stderr, "bestpathlattice: WARNING: no path found in lattice (%d nodes/%d edges)\n", (int) nodes.size(), (int) edges.size());
        return LOGZERO;         // failed, do not use resulting matrix
    }

    // traceback
    // We encode the result by storing log 1 in edges on the best path, and log 0 else;
    // this makes it naturally compatible with softalign mode
    logpps.resize (edges.size());
    foreach_index (j, edges) logpps[j] = LOGZERO;

    int backpos = backpointers[nodes.size()-1];
    while (backpos >= 0)
    {
        logpps[backpos] = 0.0f;               // edge is on best path -> PP = 1.0
        backpos = backpointers[edges[backpos].S];
    }
    assert (backpos == -1);

    return totalfwscore;
}

// ---------------------------------------------------------------------------
// forwardbackwardalign() -- compute the statelevel gammas or viterbi alignments
// the first phase of lattice::forwardbackward
// 
// Outputs:
// ---------------------------------------------------------------------------
void lattice::forwardbackwardalign (parallelstate & parallelstate,
                                    const msra::asr::simplesenonehmm & hset, const bool softalignstates, 
                                    const double minlogpp, const std::vector<double> & origlogpps, 
                                    std::vector<msra::math::ssematrixbase *> & abcs, littlematrixheap & matrixheap,
                                    const bool returnsenoneids,
                                    std::vector<float> & edgeacscores, const msra::math::ssematrixbase & logLLs,
                                    edgealignments & thisedgealignments, backpointers & thisbackpointers, array_ref<size_t> & uids, const_array_ref<size_t> bounds ) const
{   // NOTE: this will be removed and replaced by a proper representation of alignments someday
    // do forward-backward or alignment on a per-edge basis. This gives us:
    //  - per-edge gamma[j,t] = P(s(t)==s_j|edge) if forwardbackward, per-edge alignment thisedgealignments[j] if alignment
    //  - per-edge acoustic scores
    const size_t silunitid = hset.gethmmid("sil");      // shall be the same as parallelstate.getsilunitid()
    bool parallelsil = true;
    bool cpuverification = false;

#ifndef PARALLEL_SIL                                     // we use a define to make this marked
    parallelsil = false;
#endif
#ifdef CPU_VERIFICATION
    cpuverification = true;
#endif

    // Phase 1: abcs allocate
    if (!parallelstate.enabled() || !parallelsil || cpuverification)    // allocate abcs when 1.parallelstate not enabled (cpu mode); 2. enabled but not PARALLEL_SIL (silence need to be allocate); 3. cpuverfication
    {
        abcs.resize (edges.size(), NULL);                               // [edge index] -> alpha/beta/gamma matrices for each edge
        size_t countskip = 0;                                           // if pruning: count how many edges are pruned

        foreach_index (j, edges)
        {
            // determine number of frames
            // TODO: this is not efficient--we only use a block-diagonal-like structure, rest is empty (exploiting the fixed boundaries)
            const size_t edgeframes = nodes[edges[j].E].t - nodes[edges[j].S].t;
            if (edgeframes == 0)    // dummy !NULL edge at end of lattice
            {
                if ((size_t) j != edges.size() -1)
                    RuntimeError("forwardbackwardalign: unxpected 0-frame edge (only allowed at very end)");
                // note: abcs[j] is already initialized to be NULL in this case, which protects us from accidentally using it
            }
            else
            {
                // determine the number of states in an edge
                const auto & aligntokens = getaligninfo (j);    // get alignment tokens
                size_t edgestates = 0;

                bool edgehassil = false;
                foreach_index (i, aligntokens)
                    if (aligntokens[i].unit == silunitid)
                        edgehassil = true;

                if (!cpuverification && !edgehassil && parallelstate.enabled())      // !cpuverification, parallel & is non sil, we do not allocate
                {
                    abcs[j] = NULL;
                    continue;
                }

                foreach_index (k, aligntokens)
                    edgestates += hset.gethmm (aligntokens[k].unit).getnumstates();

                // allocate the matrix
                if (minlogpp > LOGZERO && origlogpps[j] < minlogpp)
                    countskip++;
                else
                    abcs[j] = &matrixheap.newmatrix (edgestates, edgeframes + 2);   // +2 to have one extra column for betas and one for gammas
            }
        }
        if (minlogpp > LOGZERO)
            fprintf(stderr, "forwardbackwardalign: %d of %d edges pruned\n", (int)countskip, (int)edges.size());
    }

    // Phase 2: alignment on CPU
    if (parallelstate.enabled() && !parallelsil)       // silence edge shall be process separately if not cuda and not PARALLEL_SIL
    {
        if (softalignstates)
            LogicError("forwardbackwardalign: parallelized version currently only handles hard alignments");
        if (minlogpp > LOGZERO)
            fprintf(stderr, "forwardbackwardalign: pruning not supported (we won't need it!) :)\n");
        edgeacscores.resize (edges.size());
        for (size_t j = 0; j < edges.size(); j++)
        {
            const auto & aligntokens = getaligninfo (j);    // get alignment tokens
            if (aligntokens.size() == 0)
                continue;
            bool edgehassil = false;
            foreach_index (i, aligntokens)
            {
                if (aligntokens[i].unit == silunitid)
                    edgehassil = true;
            }
            if (!edgehassil)                                 // only process sil
                continue;
            const edgeinfowithscores & e = edges[j];
            const size_t ts = nodes[e.S].t;
            const size_t te = nodes[e.E].t;
            const auto edgeLLs = msra::math::ssematrixstriperef<msra::math::ssematrixbase> (const_cast<msra::math::ssematrixbase &> (logLLs), ts, te - ts);
            edgeacscores[j] = alignedge (aligntokens, hset, edgeLLs, *abcs[j], j, true, thisedgealignments[j]);
        }
    }

    // Phase 3: alignment on GPU
    if (parallelstate.enabled())
        parallelforwardbackwardalign (parallelstate, hset, logLLs, edgeacscores, thisedgealignments, thisbackpointers);

    //zhaorui align to reference mlf
    if (bounds.size() > 0)
    {
        size_t framenum=bounds.size();
    
        msra::math::ssematrixbase *refabcs;
        size_t ts, te, t;
        ts = te = 0;
        
        
        aligninfo *refinfo;
        unsigned short *refalign;

        refinfo = (aligninfo*) malloc(sizeof(aligninfo)*1);
        refalign = (unsigned short *) malloc(sizeof(unsigned short ) * framenum);

        array_ref<aligninfo> refunits(refinfo,1);
        array_ref<unsigned short> refedgealignmentsj (refalign , framenum);

        while (te < framenum)
        {
            // found one phone's boundary (ts, te)
            t=ts+1;
            while (t <  framenum && bounds[t] == 0  )
                t++;
            te=t;
            
            
            //make one phone unit
            size_t phoneid = bounds[ts]-1;
            refunits[0].unit = phoneid;
            refunits[0].frames = te-ts;        
            
            size_t edgestates = hset.gethmm(phoneid).getnumstates();
            littlematrixheap refmatrixheap (1);        // for abcs
            refabcs = &refmatrixheap.newmatrix (edgestates, te-ts + 2);
            const auto edgeLLs = msra::math::ssematrixstriperef<msra::math::ssematrixbase> (const_cast<msra::math::ssematrixbase &> (logLLs), ts, te - ts);
            //do alignment
            alignedge ((const_array_ref<aligninfo>) refunits, hset, edgeLLs, *refabcs, 0, true,refedgealignmentsj);
            
            for(t=ts; t< te; t++)
            {
                uids[t] = (size_t)refedgealignmentsj[t-ts];
            }
            ts=te;

        }

        free(refinfo);
        free(refalign);
    }

    // Phase 4: alignment or forwardbackward on CPU for non parallel mode or verification

    if (!parallelstate.enabled() || cpuverification)        //non parallel mode or verification
    {
        edgeacscores.resize (edges.size());
        std::vector<float> edgeacscoresgpu;
        edgealignments thisedgealignmentsgpu (thisedgealignments);
        if (cpuverification)
        {
            parallelstate.getedgeacscores (edgeacscoresgpu);
            parallelstate.copyalignments (thisedgealignmentsgpu);
        }
        foreach_index (j, edges)
        {
            const edgeinfowithscores & e = edges[j];
            const size_t ts = nodes[e.S].t;
            const size_t te = nodes[e.E].t;
            if (ts == te)   // dummy !NULL edge at end
                edgeacscores[j] = 0.0f;
            else
            {
                const auto & aligntokens = getaligninfo (j);    // get alignment tokens
                const auto edgeLLs = msra::math::ssematrixstriperef<msra::math::ssematrixbase> (const_cast<msra::math::ssematrixbase &> (logLLs), ts, te - ts);
                if (minlogpp > LOGZERO && origlogpps[j] < minlogpp)
                    edgeacscores[j] = LOGZERO;              // will kill word level forwardbackward hypothesis
                else if (softalignstates)
                    edgeacscores[j] = forwardbackwardedge (aligntokens, hset, edgeLLs, *abcs[j], j);
                else
                    edgeacscores[j] = alignedge (aligntokens, hset, edgeLLs, *abcs[j], j, returnsenoneids, thisedgealignments[j]);
            }
            if (cpuverification)
            {
                const auto & aligntokens = getaligninfo (j);    // get alignment tokens
                bool edgehassil = false;
                foreach_index (i, aligntokens)
                {
                    if (aligntokens[i].unit == silunitid)
                        edgehassil = true;
                }
                if (fabs(edgeacscores[j] - edgeacscoresgpu[j]) > 1e-3)
                {
                    fprintf (stderr, "edge %d, sil ? %d, edgeacscores / edgeacscoresgpu MISMATCH %f v.s. %f, diff %e\n", 
                             j, edgehassil ? 1 : 0, (float)edgeacscores[j], (float)edgeacscoresgpu[j], 
                             (float)(edgeacscores[j] - edgeacscoresgpu[j]));
                    fprintf (stderr, "aligntokens: ");
                    foreach_index (i, aligntokens)
                        fprintf (stderr, "%d %d; ", i, aligntokens[i].unit);
                    fprintf (stderr, "\n");
                }
                for (size_t t = ts; t < te; t++)
                {
                    if (thisedgealignments[j][t-ts] != thisedgealignmentsgpu[j][t-ts])
                        fprintf (stderr, "edge %d, sil ? %d, time %d, alignment / alignmentgpu MISMATCH %d v.s. %d\n", j, edgehassil ? 1 : 0, (int)(t-ts), thisedgealignments[j][t-ts], thisedgealignmentsgpu[j][t-ts]);
                }
            }
        }
    }
}

// compute the error signal for sMBR mode
void lattice::sMBRerrorsignal (parallelstate & parallelstate,
                               msra::math::ssematrixbase & errorsignal, msra::math::ssematrixbase & errorsignalneg, // output
                               const std::vector<double> & logpps, const float amf,
                               double minlogpp, const std::vector<double> & origlogpps, const std::vector<double> & logEframescorrect, 
                               const double logEframescorrecttotal, const edgealignments & thisedgealignments) const
{
    if (parallelstate.enabled())    // parallel version
    {
        /*  time measurement for parallel sMBRerrorsignal
            errorsignalcompute: 19.871935 ms (cuda) v.s. 448.711444 ms (emu) */
        if (minlogpp > LOGZERO)
            fprintf(stderr, "sMBRerrorsignal: pruning not supported (we won't need it!) :)\n");
        parallelsMBRerrorsignal (parallelstate, thisedgealignments, logpps, amf, logEframescorrect, logEframescorrecttotal, errorsignal, errorsignalneg);
        return;
    }

    //  linear mode
    foreach_coord (i, j, errorsignal)
        errorsignal(i,j) = 0.0f; // Note: we don't actually put anything into the numgammas
    foreach_index (j, edges)
    {
        const auto & e = edges[j];
        if (nodes[e.S].t == nodes[e.E].t)                   // this happens for dummy !NULL edge at end of file
            continue;
        if (minlogpp > LOGZERO && origlogpps[j] < minlogpp) // this is pruned
            continue;

        size_t ts = nodes[e.S].t;
        size_t te = nodes[e.E].t;

        const double diff = logEframescorrect[j] - logEframescorrecttotal;
        // Note: the contribution of the states of an edge to their senones is the same for all states
        // so we compute it once and add it to all; this will not be the case without hard alignments.
        const double pp = exp (logpps[j]);      // edge posterior
        const float edgecorrect = (float) (pp * diff) / amf;
        for (size_t t = ts; t < te; t++)
        {
            const size_t s = thisedgealignments[j][t-ts];
            errorsignal(s,t) += edgecorrect;
        }
    }
}

// compute the error signal for MMI mode
void lattice::mmierrorsignal (parallelstate & parallelstate, double minlogpp, const std::vector<double> & origlogpps, 
                              std::vector<msra::math::ssematrixbase *> & abcs, const bool softalignstates, 
                              const std::vector<double> & logpps, const msra::asr::simplesenonehmm & hset, 
                              const edgealignments & thisedgealignments, msra::math::ssematrixbase & errorsignal) const
{
    if (parallelstate.enabled())
    {   
        if (minlogpp > LOGZERO)
            fprintf(stderr, "mmierrorsignal: pruning not supported (we won't need it!) :)\n");
        if (softalignstates)
            LogicError("mmierrorsignal: parallel version for softalignstates mode is not supported yet");
        parallelmmierrorsignal (parallelstate, thisedgealignments, logpps, errorsignal);
        return;
    }
    
    for (size_t j = 0; j < (errorsignal).cols(); j++) 
        for (size_t i = 0; i < (errorsignal).rows(); i++)
            errorsignal(i,j) = VIRGINLOGZERO;    // set to zero  --note: may be in-place with logLLs, which now get overwritten

    // size_t warnings = 0;   // [v-hansu] check code for mmi; search this comment to see all related codes
    foreach_index (j, edges)
    {
        const auto & e = edges[j];
        if (nodes[e.S].t == nodes[e.E].t)                   // this happens for dummy !NULL edge at end of file
            continue;
        if (minlogpp > LOGZERO && origlogpps[j] < minlogpp) // this is pruned
            continue;

        const auto & aligntokens = getaligninfo (j);        // get alignment tokens
        auto & loggammas = *abcs[j];

        const float edgelogP = (float) logpps[j];
        //if (islogzero (edgelogP))               // we had a 0 prob
        //    continue;

        // accumulate this edge's gamma matrix into target posteriors
        const size_t tedge = nodes[e.S].t;
        size_t ts = 0;                          // time index into gamma matrix
        size_t js = 0;                          // state index into gamma matrix
        foreach_index (k, aligntokens)          // we exploit that units have fixed boundaries
        {
            const auto & unit = aligntokens[k];
            const size_t te = ts + unit.frames;
            const auto & hmm = hset.gethmm (unit.unit); // TODO: inline these expressions
            const size_t n = hmm.getnumstates();
            const size_t je = js + n;
            // P(s) = P(s|e) * P(e)
            for (size_t t = ts; t < te; t++)
            {
                const size_t tutt = t + tedge;  // time index w.r.t. utterance
                //double logsum = LOGZERO;         // [v-hansu] check code for mmi; search this comment to see all related codes
                for (size_t i = 0; i < n; i++)
                {
                    const size_t j = js + i;                // state index for this unit in matrix
                    const size_t s = hmm.getsenoneid (i);   // state class index
                    const float gammajt = loggammas(j,t);
                    const float statelogP = edgelogP + gammajt;
                    logadd (errorsignal(s,tutt), statelogP);
                }
            }
            ts = te;
            js = je;
        }
        assert (ts + 2 == loggammas.cols() && js == loggammas.rows());
    }

    // check normalizedness (is that an actual English word?)
    // also count non-zero probs
    size_t nonzerostates = 0;
    foreach_column (t, errorsignal)
    {
        double logsum = LOGZERO;
        foreach_row (s, errorsignal)
        {
            if (islogzero (errorsignal(s,t)))
                nonzerostates++;
            else
                logadd (logsum, (double) errorsignal(s,t));
            // TODO: count VIRGINLOGZERO, print per frame
        }
        if (fabs (logsum) / errorsignal.rows() > 1e-6)
            fprintf (stderr, "forwardbackward: WARNING: overall posterior column(%d) sum = exp (%.10f) != 1\n", (int)t, logsum);
    }
    fprintf (stderr, "forwardbackward: %.3f%% non-zero state posteriors\n", 100.0f - nonzerostates * 100.0f / errorsignal.rows() / errorsignal.cols());

    // convert to non-log posterior  --that's what we return
    foreach_coord (i, j, errorsignal)
        errorsignal(i,j) = expf (errorsignal(i,j));
}

// compute ground truth's score
// It is critical to get all details consistent with the lattice, to avoid skewing the weights.
// ... TODO: we don't need this to be a class member, actually; try to just make it a 'static' function.
/*static*/ double lattice::scoregroundtruth (const_array_ref<size_t> uids, const_array_ref<htkmlfwordsequence::word> transcript, const std::vector<float> & transcriptunigrams,
                                             const msra::math::ssematrixbase & logLLs, const msra::asr::simplesenonehmm & hset, const float lmf, const float wp, const float amf)
{
    if (transcript[0].firstframe != 0)  // TODO: should we store the #frames instead? Then we can validate the total duration
        LogicError("scoregroundtruth: first transcript[] token does not start at frame 0");

    // get the silence models, since they are treated specially
    const size_t numframes = logLLs.cols();
    const auto & sil = hset.gethmm (hset.gethmmid ("sil"));
    const auto & sp = hset.gethmm (hset.gethmmid ("sp"));
    if (sp.numstates != 1 || sil.numstates != 3)
        RuntimeError("scoregroundtruth: only supports 1-state /sp/ and 3-state /sil/ tied to /sp/");
    const size_t silst = sp.senoneids[0];

    // loop over words
    double pathscore = 0.0;
    foreach_index (i, transcript)
    {
        size_t ts = transcript[i].firstframe;
        size_t te = ((size_t) i+1 < transcript.size()) ? transcript[i+1].firstframe : numframes;
        if (ts >= te)
            LogicError("scoregroundtruth: transcript[] tokens out of order");
        // acoustic score: loop over frames
        const msra::asr::simplesenonehmm::transP * prevtransP = NULL;   // previous transP
        int prevs = -1;                                                 // previous state index
        for (size_t t = ts; t < te; t++)
        {
            size_t senoneid = uids[t];
            // recover the transP and state index
            int s;
            const msra::asr::simplesenonehmm::transP * transP;
            if (senoneid == silst)
            {
                if (prevtransP == &sil.gettransP())     // "silst" may be tied to /sp/, and thus the /sil/ center state may be ambiguous
                {
                    transP = prevtransP;                // remain in /sil/
                    s = 1;  // note that this will fail if /sp/ can follow /sil/ and if /sil/ may end with "silst" (both not allowed currently)
                }
                else                                    // "silst" -> we are in /sp/
                {
                    transP = &sp.gettransP();
                    s = 0;
                }
            }
            else    // all others must be non-ambiguous
            {
                int transPindex = hset.senonetransP (senoneid);
                int sindex = hset.senonestate (senoneid);
                if (transPindex == -1 || sindex == -1)
                    RuntimeError("scoregroundtruth: failed to resolve ambiguous senone %s", hset.getsenonename (senoneid));
                transP = &hset.transPs[transPindex];
                s = sindex;
            }
            // if changing phoneme then add necessary enter/exit score
            if (transP != prevtransP)
            {
                if (prevtransP)         // previous exit transition
                    pathscore += (*prevtransP)(prevs,prevtransP->getnumstates());
                prevs = -1;             // enter transition
            }
            // add inter-state transP
            pathscore += (*transP)(prevs,s);
            // acoustic score
            pathscore += logLLs (senoneid,t);
            // remember state for next frame
            prevtransP = transP;
            prevs = s;
        }
        // add the last exit transition
        if (prevtransP)         // previous exit transition
            pathscore += (*prevtransP)(prevs,prevtransP->getnumstates());
        // need to add a /sp/ tee transition if we don't end in sp, since our transcript dictionary includes a /sp/ at the end of each entry
        if (uids[te-1] != sp.senoneids[0])
            pathscore += sp.gettransP()(-1,sp.numstates);
        // lm
        pathscore += transcriptunigrams[i] * lmf + wp;
    }
    if (islogzero (pathscore))
        fprintf (stderr, "scoregroundtruth: ground-truth path has zero probability; some model inconsistency, maybe?\n");
    // account for amf
    pathscore /= amf;
    fprintf (stderr, "scoregroundtruth: ground-truth score %.6f (%d frames)\n", pathscore, (int) numframes);
    return pathscore;
}

// ---------------------------------------------------------------------------
// sMBRdiagnostics() -- helper to print some diagnostics for analyzing sMBR results
// ---------------------------------------------------------------------------

// static  // with 'static', compiler will complain if the function is not used (we only compile it in sometimes for diagnostics)
void sMBRdiagnostics (const msra::math::ssematrixbase & errorsignal, const_array_ref<size_t> uids,
                      const_array_ref<size_t> bestpath, const vector<bool> & refseen, const msra::asr::simplesenonehmm & hset)
{
    // TODO:
    //  - print best positive runner-up
    //  - WARN tag if neg > pos
    //  - check the sum and warn if not 0
    //  - indicate whether best path state is correct or not
    size_t numcor = 0;
    size_t numnegbetter = 0;    // # frames the neg competitor is better
    size_t numposbetter = 0;    // # frames the pos competitor is better
    foreach_column (t, errorsignal)
    {
        const size_t sref = uids[t];
        const char * srefname = hset.getsenonename (sref);
        const size_t sbest = bestpath[t];
        const char * sbestname = hset.getsenonename (sbest);
        if (sref == sbest)
            numcor++;
        // for each frame, print error signal for ground truth and runner up (second largest abs value)
        size_t sneg = SIZE_MAX;     // competitor
        float eneg = 0.0f;
        size_t spos = SIZE_MAX;     // best postive competitor
        float epos = 0.0f;
        foreach_row (s, errorsignal)
        {
            if (s == sref)
                continue;
            if (errorsignal(s,t) < eneg)
            {
                sneg = s;
                eneg = errorsignal(s,t);
            }
            if (errorsignal(s,t) > epos)
            {
                spos = s;
                epos = errorsignal(s,t);
            }
        }
        if (fabs (errorsignal (sref, t)) > 0.0001f && errorsignal (sref, t) < -eneg)
            numnegbetter++;
        if (fabs (errorsignal (sref, t)) > 0.0001f && errorsignal (sref, t) < epos)
            numposbetter++;
        const char * snegname = sneg == SIZE_MAX ? "-" : hset.getsenonename (sneg);
        const char * sposname = spos == SIZE_MAX ? "-" : hset.getsenonename (spos);
        fprintf (stderr, "e(%d): ref %s: %.6f / %s: %.6f / %s: %.6f / top %s: %.6f%s%s%s%s%s\n",
                 (int) t, srefname, errorsignal (sref, t), snegname, eneg, sposname, epos, sbestname, errorsignal (sbest, t),
                 sbest == sref ? "" : " ERR",
                 fabs (errorsignal (sref, t)) > 0.0001f && errorsignal (sref, t) < 0 ? " INV!!" : "",
                 fabs (errorsignal (sref, t)) > 0.0001f && errorsignal (sref, t) < -eneg ? " WEAK" : "",
                 fabs (errorsignal (sref, t)) > 0.0001f && errorsignal (sref, t) < epos ? " 2ND" : "",
                 refseen[t] ? "" : " NOREF");
    }
    // print this to validate our bestpath computation
    fprintf (stderr, "sMBRdiagnostics: %d frames correct out of %d, %.2f%%, neg better in %d, pos in %d\n",
             (int) numcor, (int)errorsignal.cols(), 100.0f * numcor / errorsignal.cols(),
             (int) numnegbetter, (int) numposbetter);
}

// static  // with 'static', compiler will complain if the function is not used (we only compile it in sometimes for diagnostics)
void sMBRsuppressweirdstuff (msra::math::ssematrixbase & errorsignal, const_array_ref<size_t> uids)
{
    size_t numweird = 0;
    foreach_column (t, errorsignal)
    {
        const size_t sref = uids[t];
        // for each frame, print error signal for ground truth and runner up (second largest abs value)
        const float eref = errorsignal(sref,t);
        bool isweird = eref < 0.0f;             // negative for reference!?
        for (size_t s = 0; s < errorsignal.rows() && !isweird; s++)
        {
            if (s == sref)
                continue;
            if (fabs (errorsignal(s,t)) > eref)
                isweird = true;
        }
        if (isweird)
        {
            foreach_row (s, errorsignal)
                errorsignal(s,t) = 0.0f;
            numweird++;
        }
    }
    // print this to validate our bestpath computation
    fprintf (stderr, "sMBRsuppressweirdstuff: %d weird frames out of %d, %.2f%% were flattened\n",
             (int) numweird, (int) errorsignal.cols(), 100.0f * numweird / errorsignal.cols());
}


// ---------------------------------------------------------------------------
// forwardbackward() -- main function for MMI/sMBR
//
// This computes the lattice state-level statistics for sequence training using MMI or sMBR.
//
// Outputs, MMI mode:
//  - result = dengammas = denominator gammas (non-log form)
//  - returns log of sum over all paths' likelihoods (the denominator of the MMI objective)
//  - note: numgammas is not used/touched in MMI mode
//
// Outputs, sMBR mode:
// TODO: fix this comment
//  - result = errorsignal = (abs value of) negative contributions to error signal
//  - errorsignalbuf = for temporarily use to get errorsignal
//  - returns expected frames-correct count (the sMBR objective)
// ---------------------------------------------------------------------------

double lattice::forwardbackward (parallelstate & parallelstate, const msra::math::ssematrixbase & logLLs, const msra::asr::simplesenonehmm & hset,
                                 msra::math::ssematrixbase & result, msra::math::ssematrixbase & errorsignalbuf,
                                 const float lmf, const float wp, const float amf, const float boostingfactor,
                                 const bool sMBRmode, array_ref<size_t> uids, const_array_ref<size_t> bounds,
                                 const_array_ref<htkmlfwordsequence::word> transcript, const std::vector<float> & transcriptunigrams) const
{
    bool softalign = true;
    bool softalignstates = false;       // true if soft alignment within edges, currently we only support soft within edge in cpu mode
    bool softalignlattice = softalign;  // w.r.t. whole lattice

    edgealignments thisedgealignments (*this);          // alignments memory allocate for this lattice
    backpointers thisbackpointers (*this, hset);        // memory for forwardbackward

    if (info.numframes != logLLs.cols())
        LogicError("forwardbackward: #frames mismatch between lattice (%d) and LLs (%d)", (int) info.numframes, (int) logLLs.cols());
    // TODO: the following checks should throw, but I don't dare in case this will crash a critical job... if we never see this warning, then 
    if (info.numframes != uids.size())
        fprintf (stderr, "forwardbackward: #frames mismatch between lattice (%d) and uids (%d)\n", (int) info.numframes, (int) uids.size());
    if (info.numframes != result.cols())
        fprintf (stderr, "forwardbackward: #frames mismatch between lattice (%d) and result (%d)\n", (int) info.numframes, (int) result.cols());

    littlematrixheap matrixheap (info.numedges);        // for abcs

    // PHASE 0: fake word level forward backwards --only used when pruning enabled
    const double minlogpp = LOGZERO;                // pruning threshold  --LOGZERO means disabled
    std::vector<double> origlogpps;                 // word posterior from original lattice, for pruning decision

    // PHASE 1: per-edge forward backwards (="time alignments")
    
    // score the ground truth  --only if a transcript is provided, which happens if the user provides a language model
    // TODO: no longer used, remove this. 'transcript' parameter is no longer used in this function.
    transcript; transcriptunigrams;

    // allocate alpha/beta/gamma matrices (all are sharing the same memory in-place)
    std::vector<msra::math::ssematrixbase *> abcs;
    std::vector<float> edgeacscores; // [edge index] acoustic scores
    //funcation call for forwardbackward on edge level
    forwardbackwardalign (parallelstate, hset, softalignstates, minlogpp, origlogpps, abcs, matrixheap, sMBRmode/*returnsenoneids*/, edgeacscores, logLLs, thisedgealignments, thisbackpointers, uids,bounds);

    // PHASE 2: lattice-level forward backward

    // we exploit that the lattice is sorted by (end node, start node) for in-place processing

    // checklattice();      // comment out by v-hansu to save time
#ifdef PRINT_TIME_MEASUREMENT
    auto_timer latlevelfwbw;
#endif
    std::vector<double> logpps;
    std::vector<double> Eframescorrectbuf;     // this is used for compute the Eframescorrectdiff
    std::vector<double> logEframescorrect;      // this is the final output of PHASE 2
    std::vector<double> logalphas;
    std::vector<double> logbetas;
    double totalfwscore = 0;        // TODO: name no longer precise in sMBRmode
    double logEframescorrecttotal = LOGZERO;

    bool returnEframescorrect = sMBRmode;
    if (softalignlattice)
    {
        totalfwscore = forwardbackwardlattice (edgeacscores, parallelstate, logpps, logalphas, logbetas, lmf, wp, amf, boostingfactor, returnEframescorrect, (const_array_ref<size_t> &)uids, thisedgealignments, logEframescorrect, Eframescorrectbuf, logEframescorrecttotal);
        if (sMBRmode && !returnEframescorrect)
            logEframescorrecttotal = forwardbackwardlatticesMBR (edgeacscores, hset, logalphas, logbetas, lmf, wp, amf, (const_array_ref<size_t> &)uids, thisedgealignments, Eframescorrectbuf);
                //^^ BUGBUG not tested
    }
    else
        totalfwscore = bestpathlattice (edgeacscores, logpps, lmf, wp, amf);
#ifdef PRINT_TIME_MEASUREMENT
    latlevelfwbw.show("latlevelfwbw");  // 68.395682 ms
#endif
    if (islogzero (totalfwscore))
    {
        fprintf (stderr, "forwardbackward: WARNING: no path found in lattice (%d nodes/%d edges)\n", (int) nodes.size(), (int) edges.size());
        return LOGZERO;         // failed, do not use resulting matrix
    }
    
    // PHASE 3: compute final state-level posteriors (MMI mode)
    
    // compute expected frames correct in sMBRmode

    const size_t numframes = logLLs.cols();
    assert (numframes == info.numframes);
    //fprintf (stderr, "forwardbackward: total forward score %.6f (%d frames)\n", totalfwscore, (int) numframes);   // for now--while we are debugging the GPU port
    
    // MMI mode
    if(!sMBRmode)
    {
        // we first take the sum in log domain to avoid numerical issues
        auto & dengammas = result;  // result is denominator gammas
        mmierrorsignal (parallelstate, minlogpp, origlogpps, abcs, softalignstates, logpps, hset, thisedgealignments, dengammas);
        return totalfwscore / numframes;    // return value is av. posterior
    }
    // sMBR mode
    else
    {
        auto & errorsignal = result;
        sMBRerrorsignal (parallelstate, errorsignal, errorsignalbuf, logpps, amf, minlogpp, origlogpps, logEframescorrect, logEframescorrecttotal, thisedgealignments);

        static bool dummyvariable = (fprintf(stderr, "note: new version with kappa adjustment, kappa = %.2f\n", 1/amf ), true); // we only print once
        return exp (logEframescorrecttotal) / numframes;    // return value is av. expected frame-correct count
    }
}

};};
