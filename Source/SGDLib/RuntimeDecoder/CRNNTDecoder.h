#pragma once

typedef unsigned int token_t;
constexpr token_t InvalidToken = (token_t) -1;
typedef std::vector<token_t> token_seq_t;

class CTokenTrie
{
public:
    typedef unsigned int NodeId;
    static constexpr NodeId InvalidNodeId = (NodeId) -1;
    static constexpr NodeId Root = 0u;

    CTokenTrie()
    {
        clear();
    }

    void clear()
    {
        m_nodes.clear();
        #ifdef LINUXRUNTIMECODE
        m_nodes.emplace_back(NodeId(-1), (token_t) -1);
        #else
        m_nodes.emplace_back(InvalidNodeId, (token_t) -1);
        #endif
    }

    bool is_proper_prefix(NodeId x, NodeId y) const
    {
        if (x >= y)
            return false;

        auto p = m_nodes[y].parent;

        if (p == x)
            return true;

        rassert_op(p, !=, InvalidNodeId);
        return is_proper_prefix(x, p);
    }

    bool is_prefix(NodeId x, NodeId y) const
    {
        if (x > y)
            return false;

        if (x == y)
            return true;

        auto p = m_nodes[y].parent;

        rassert_op(p, !=, InvalidNodeId);
        return is_prefix(x, p);
    }

    void append_children(
        std::vector<std::pair<token_t, NodeId>>& result,
        NodeId x) const
    {
        for (auto n = m_nodes[x].child;
             n != InvalidNodeId;
             n = m_nodes[n].sibling)
            result.emplace_back(m_nodes[n].label, n);
    }

    NodeId walk(NodeId x, token_t k)
    {
        for (auto n = m_nodes[x].child;
             n != InvalidNodeId;
             n = m_nodes[n].sibling)
        {
            if (m_nodes[n].label == k)
                return n;

            if (m_nodes[n].sibling == InvalidNodeId)
            {
                auto y = (NodeId) m_nodes.size();
                m_nodes.emplace_back(x, k);

                m_nodes[n].sibling = y;
                return y;
            }
        }

        rassert_eq(m_nodes[x].child, InvalidNodeId);

        auto y = (NodeId) m_nodes.size();
        m_nodes.emplace_back(x, k);

        m_nodes[x].child = y;
        return y;
    }

    token_seq_t expand(NodeId x) const
    {
        token_seq_t result;
        for (auto n = x; n != Root; n = m_nodes[n].parent)
        {
            rassert_op(n, !=, InvalidNodeId);
            result.push_back(m_nodes[n].label);
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    NodeId parent_of(NodeId x) const
    {
        return m_nodes[x].parent;
    }

    token_t label_of(NodeId x) const
    {
        return m_nodes[x].label;
    }

private:
    struct Node
    {
        Node(NodeId p, token_t k)
            : parent(p),
              label(k),
              child(InvalidNodeId),
              sibling(InvalidNodeId)
        {
        }

        const NodeId parent;
        const token_t label;
        NodeId child;
        NodeId sibling;
    };

    std::vector<Node> m_nodes;
};

typedef CTokenTrie::NodeId token_seq_t_1;

struct BeamEntry
{
    BeamEntry(const token_seq_t_1& y, float logAlpha)
        : Y(y),
          LogAlpha(logAlpha),
          LogPr(NAN)
    {
    }

    token_seq_t_1 Y;
    float LogAlpha;
    float LogPr;

    bool operator<(const BeamEntry& that) const
    {
        return LogAlpha < that.LogAlpha;
    }
};

enum class RecombineKind
{
    Sum = 0,
    Max = 1
};

enum class BeamSortKind
{
    Prob = 0,
    Alpha = 1,
};

enum class ScoreNormalizationKind
{
    None = 0,
    WordPieceCnt = 1,
    WordPieceCntMinusOne = 2,
    SqrtWordPieceCnt = 3
};

float NormalizeScore(const token_seq_t& y, float logPr, ScoreNormalizationKind kind)
{
    rassert_op(y.size(), >, 0u);

    switch (kind)
    {
    case ScoreNormalizationKind::None:
        return logPr;

    case ScoreNormalizationKind::WordPieceCnt:
        return logPr / y.size();

    case ScoreNormalizationKind::WordPieceCntMinusOne:
        return y.size() <= 1 ? logPr : logPr / (y.size() - 1);

    case ScoreNormalizationKind::SqrtWordPieceCnt:
        return logPr / sqrtf((float) y.size());

    default:
        rfail("unknown score normalization kind %d\n", int(kind));
    }
};

// log(exp(log_x) + exp(log_y))
float log_add_exp(float log_x, float log_y)
{
    if (log_x < log_y)
        std::swap(log_x, log_y);
    return log_x + log1pf(expf(log_y - log_x));
    // return log_x > log_y ? log_x : log_y;
}

class CBeamB
{
public:
    CBeamB(size_t width, BeamSortKind beamSortKind)
        : m_width(width),
          m_beamSortKind(beamSortKind)
    {
    }

    void push(const BeamEntry& entry)
    {
        rassert_eq(isnan(entry.LogAlpha), false);
        rassert_eq(isnan(entry.LogPr), false);

        for (const auto& e : m_beam)
            rassert_op(e.Y, !=, entry.Y);

        m_beam.push_back(entry);

        switch (m_beamSortKind)
        {
        case BeamSortKind::Prob:
            std::sort(
                m_beam.begin(),
                m_beam.end(),
                [](const BeamEntry& x, const BeamEntry& y) { return x.LogPr > y.LogPr; });
            break;

        case BeamSortKind::Alpha:
            std::sort(m_beam.rbegin(), m_beam.rend());
            break;

        default:
            rfail("unknown beam sort kind %d\n", int(m_beamSortKind));
        }
        if (m_beam.size() > m_width)
        {
            rassert_eq(m_beam.size(), m_width + 1);
            m_beam.pop_back();
        }
    }

    size_t size() const
    {
        return m_beam.size();
    }

    const BeamEntry& top() const
    {
        return m_beam.front();
    }

    float threshold() const
    {
        if (m_beam.size() < m_width)
            return -std::numeric_limits<float>::infinity();

        switch (m_beamSortKind)
        {
        case BeamSortKind::Prob:
            return m_beam.back().LogPr;

        case BeamSortKind::Alpha:
            return m_beam.back().LogAlpha;

        default:
            rfail("unknown beam sort kind %d\n", int(m_beamSortKind));
        }
    }

    void clear()
    {
        m_beam.clear();
    }

    std::vector<BeamEntry>::iterator begin()
    {
        return m_beam.begin();
    }

    std::vector<BeamEntry>::iterator end()
    {
        return m_beam.end();
    }

    std::vector<BeamEntry>::const_iterator begin() const
    {
        return m_beam.begin();
    }

    std::vector<BeamEntry>::const_iterator end() const
    {
        return m_beam.end();
    }

    std::string list() const
    {
        std::string result;
        for (const auto& entry : m_beam)
        {
            if (!result.empty())
                result.push_back(' ');
            result.append(std::to_string(entry.LogPr));
        }

        return result;
    }

private:
    BeamSortKind m_beamSortKind;
    size_t m_width;
    std::vector<BeamEntry> m_beam;
};

template <typename T>
class priority_queue_ex : public std::priority_queue<T>
{
#ifndef LINUXRUNTIMECODE

public:
    void clear()
    {
        c.clear();
    }
#endif
};

class CBeamA
{
public:
    CBeamA(CTokenTrie& trie, size_t beamSize, BeamSortKind beamSortKind)
        : m_trie(trie)
    {
        switch (beamSortKind)
        {
        case BeamSortKind::Prob:
            m_tail = std::make_unique<CTailQueue>();
            break;

        case BeamSortKind::Alpha:
            m_tail = std::make_unique<CLimitedTailQueue>(beamSize);
            break;

        default:
            rfail("unknown beam sort kind %d\n", int(beamSortKind));
        }
    }

    // returns head nodes
    std::vector<token_seq_t_1> reset(CBeamB& B)
    {
        m_tail->reset(B.threshold());

        std::vector<token_seq_t_1> head_nodes;
        for (auto i = B.begin(); i != B.end(); i++)
        {
            auto has_prefix = std::any_of(
                B.begin(),
                B.end(),
                [this, i](const BeamEntry& j) { return m_trie.is_proper_prefix(j.Y, i->Y); });

            if (has_prefix)
                m_carry.emplace_back(i->Y, i->LogPr);
            else
            {
                head_nodes.push_back(i->Y);
                auto has_suffix = std::any_of(
                    B.begin(),
                    B.end(),
                    [this, i](const BeamEntry& j) { return m_trie.is_proper_prefix(i->Y, j.Y); });
                if (has_suffix)
                    m_head.emplace(i->Y, i->LogPr);
                else
                    m_tail->emplace(
                        m_trie.parent_of(i->Y),
                        m_trie.label_of(i->Y),
                        i->LogPr);
            }
        }

        return head_nodes;
    }

    bool try_pop(BeamEntry& entry, float threshold)
    {
        if (!m_head.empty())
        {
            entry = m_head.top();
            m_head.pop();
            return true;
        }

        rassert_eq(m_carry.size(), 0u);

        return m_tail->try_pop(entry, threshold, m_trie);
    }

    void recombine(
        const BeamEntry& star,
        const CVector& j,
        token_t blank,
        float B_threshold,
        RecombineKind kind,
        float insertionBoost)
    {
        m_trie.append_children(m_temp_children, star.Y);
        std::sort(
            m_temp_children.begin(),
            m_temp_children.end(),
            [](const std::pair<token_t, token_seq_t_1>& x,
               const std::pair<token_t, token_seq_t_1>& y) { return x.first < y.first; });
        m_temp_children.emplace_back(j.M, m_trie.InvalidNodeId);
        auto iter_y = m_temp_children.begin();

        rassert_eq(j.M - 1, blank);
        for (token_t k = 0; k < j.M - 1; k++)
        {
            auto logAlpha = star.LogAlpha + j[k];
            // rassert_op(logAlpha, <=, star.LogAlpha);
            logAlpha += insertionBoost;

            if (k == iter_y->first)
            {
                auto y = iter_y->second;
                iter_y++;
                auto iter = std::find_if(
                    m_carry.begin(),
                    m_carry.end(),
                    [&y](const BeamEntry& entry) { return entry.Y == y; });

                if (iter != m_carry.end())
                {
                    logAlpha = recombine_logAlpha(iter->LogAlpha, logAlpha, kind);
                    m_carry.erase(iter);
                }

                auto iter1 = std::find_if(
                    m_carry.begin(),
                    m_carry.end(),
                    [this, &y](const BeamEntry& entry) { return m_trie.is_proper_prefix(y, entry.Y); });

                if (iter1 != m_carry.end())
                    m_head.emplace(y, logAlpha);
                else if (logAlpha > B_threshold)
                    m_tail->emplace(star.Y, k, logAlpha);
            }
            else if (logAlpha > B_threshold)
                m_tail->emplace(star.Y, k, logAlpha);
        }

        m_temp_children.clear();
    }

private:
    float recombine_logAlpha(float logAlpha, float logAlpha2, RecombineKind kind)
    {
        switch (kind)
        {
        case RecombineKind::Sum:
            return log_add_exp(logAlpha, logAlpha2);

        case RecombineKind::Max:
            return logAlpha > logAlpha2 ? logAlpha : logAlpha2;

        default:
            rfail("unknown recombine kind %d\n", int(kind));
        }
    }

private:
    struct TailEntry
    {
        TailEntry()
            : Y(CTokenTrie::InvalidNodeId),
              K(InvalidToken),
              LogAlpha(-std::numeric_limits<float>::infinity())
        {
        }

        TailEntry(const token_seq_t_1& y, token_t k, float logAlpha)
            : Y(y),
              K(k),
              LogAlpha(logAlpha)
        {
        }

        token_seq_t_1 Y;
        token_t K;
        float LogAlpha;

        bool operator<(const TailEntry& that) const
        {
            return LogAlpha < that.LogAlpha;
        }
    };

    class ITailQueue
    {
    public:
        virtual ~ITailQueue() {}
        virtual void reset(float tier_line) = 0;
        virtual void emplace(
            const token_seq_t_1& y,
            token_t k,
            float logAlpha) = 0;
        virtual bool try_pop(BeamEntry& entry, float threshold, CTokenTrie& trie) = 0;
    };

    class CTailQueue : public ITailQueue
    {
    public:
        virtual void reset(float tier_line)
        {
#ifdef LINUXRUNTIMECODE
            m_tier_1 = priority_queue_ex<TailEntry>();
#else
            m_tier_1.clear();
#endif
            m_tier_2.clear();
            m_tier_line = tier_line;
        }

        virtual void emplace(
            const token_seq_t_1& y,
            token_t k,
            float logAlpha)
        {
            if (logAlpha >= m_tier_line)
                m_tier_1.emplace(y, k, logAlpha);
            else
                m_tier_2.emplace_back(y, k, logAlpha);
        }

        virtual bool try_pop(BeamEntry& entry, float threshold, CTokenTrie& trie)
        {
            if (!m_tier_1.empty())
            {
                if (m_tier_1.top().LogAlpha <= threshold)
                    return false;

                const auto& star = m_tier_1.top();
                entry = {trie.walk(star.Y, star.K), star.LogAlpha};
                m_tier_1.pop();
                return true;
            }

            if (!m_tier_2.empty())
            {
                // auto old_size = m_tier_2.size();
                auto rem_iter = m_tier_2.begin();
                auto max_iter = m_tier_2.begin();
                for (const auto& x : m_tier_2)
                {
                    if (x.LogAlpha > threshold)
                    {
                        *rem_iter = x;

                        if (x.LogAlpha > max_iter->LogAlpha)
                            max_iter = rem_iter;

                        rem_iter++;
                    }
                }

                // auto m = std::max_element(m_tier_2.begin(), rem_iter);
                // rassert_eq(m - m_tier_2.begin(), max_iter - m_tier_2.begin());

                bool ret;
                if (rem_iter != m_tier_2.begin())
                {
                    m_tier_line = max_iter->LogAlpha;
                    entry = {trie.walk(max_iter->Y, max_iter->K), max_iter->LogAlpha};

                    rem_iter--;
                    *max_iter = *rem_iter;

                    ret = true;
                }
                else
                {
                    m_tier_line = -std::numeric_limits<float>::infinity();
                    ret = false;
                }

                m_tier_2.erase(rem_iter, m_tier_2.end());

                // fprintf(stderr, "\nprune tier 2: %zu -> %zu (%d)\n", old_size, m_tier_2.size(), ret);

                return ret;
            }

            m_tier_line = -std::numeric_limits<float>::infinity();
            return false;
        }

    private:
        float m_tier_line = -std::numeric_limits<float>::infinity();
        priority_queue_ex<TailEntry> m_tier_1;
        std::vector<TailEntry> m_tier_2;
    };

    class CLimitedTailQueue : public ITailQueue
    {
    public:
        CLimitedTailQueue(size_t cnt)
            : m_pbegin(std::make_unique<TailEntry[]>(cnt)),
              m_cnt(cnt),
              m_pmin(&m_pbegin[0]),
              m_pend(&m_pbegin[cnt])
        {
        }

        virtual void reset(float tier_line)
        {
            tier_line;
            m_pend = &m_pbegin[m_cnt];
            for (auto p = &m_pbegin[0]; p < m_pend; p++)
                *p = {};
        }

        virtual void emplace(
            const token_seq_t_1& y,
            token_t k,
            float logAlpha)
        {
            if (m_pmin->LogAlpha < logAlpha)
            {
                *m_pmin = {y, k, logAlpha};
                m_pmin = std::min_element(&m_pbegin[0], m_pend);
            }
        }

        virtual bool try_pop(BeamEntry& entry, float threshold, CTokenTrie& trie)
        {
            if (m_pend == &m_pbegin[0])
                return false;

            auto pmax = std::max_element(&m_pbegin[0], m_pend);
            if (pmax->LogAlpha <= threshold)
                return false;

            // When popping, we shrink the queue so that the output from the
            // queue is limited by the given cnt
            entry = {trie.walk(pmax->Y, pmax->K), pmax->LogAlpha};
            *pmax = *(m_pend - 1);
            m_pend--;
            m_pmin = std::min_element(&m_pbegin[0], m_pend);

            return true;
        }

    private:
        std::unique_ptr<TailEntry[]> const m_pbegin;
        size_t m_cnt;
        TailEntry* m_pmin;
        TailEntry* m_pend;
    };

    CTokenTrie& m_trie;
    std::priority_queue<BeamEntry> m_head;
    std::vector<BeamEntry> m_carry;
    std::unique_ptr<ITailQueue> m_tail;

    std::vector<std::pair<token_t, token_seq_t_1>> m_temp_children;
};

class CPredictorCache
{
public:
    CPredictorCache(IPredictor& predictor, CTokenTrie& trie)
        : m_predictor(predictor),
          m_trie(trie)
    {
    }

    void Reset(token_t blank)
    {
        m_cache.clear();

        const auto& d = m_predictor.Forward(blank);
        auto s0 = m_predictor.DetachState();
        m_cache.emplace(
            m_trie.walk(m_trie.Root, blank),
            Entry{std::move(s0), d});
    }

    void Prune(const std::vector<token_seq_t_1>& head_nodes)
    {
        /*
static size_t max_size = 0;
auto size = m_cache.size();
if (size > max_size)
{
fprintf(stderr, "\nmax predictor cache size: %zu\n", max_size);
max_size = size;
}
*/
        for (auto iter = m_cache.begin(); iter != m_cache.end();)
        {
            bool prune = true;
            for (const auto& y : head_nodes)
            {
                if (m_trie.is_prefix(y, iter->first))
                {
                    prune = false;
                    break;
                }
            }

            if (prune)
                iter = m_cache.erase(iter);
            else
                iter++;
        }
    }

    const CVector& Ensure(const token_seq_t_1& y)
    {
        auto iter = m_cache.find(y);
        if (iter != m_cache.end())
            return iter->second.Output;

        auto y1 = m_trie.parent_of(y);
        auto iter1 = m_cache.find(y1);
        rassert_eq(false, iter1 == m_cache.end());

        auto k = m_trie.label_of(y);
        auto s1 = m_predictor.NewState();
        const auto& d = m_predictor.Forward(*s1, *iter1->second.State, k);
        /*
auto o1 = s1->s1->o.M * s1->s1->o.N;
auto c1 = s1->s1->ct.M * s1->s1->ct.N;
auto o2 = s1->s2->o.M * s1->s2->o.N;
auto c2 = s1->s2->ct.M * s1->s2->ct.N;
auto dd = d.M * d.N;
fprintf(stderr, "\nstate size: (%u + %u + %u + %u + %u) * %zu = %zu\n",
            o1,
            c1,
            o2,
            c2,
            dd,
            sizeof(float),
            (o1 + c1 + o2 + c2 + dd) * sizeof(float));
*/
        auto p = m_cache.emplace(
            y,
            Entry{std::move(s1), d});
        rassert_eq(p.second, true);

        return p.first->second.Output;
    }

private:
    struct Entry
    {
        Entry(
            std::unique_ptr<IPredictor::IState>&& s,
            const CVector& o)
            : State(std::move(s)),
              Output(o.M)
        {
            Output.Set(o);
        }

        std::unique_ptr<IPredictor::IState> State;
        CVector Output;
    };

    IPredictor& m_predictor;
    CTokenTrie& m_trie;
    std::map<token_seq_t_1, Entry> m_cache;
};

class CRNNTDecoder
{
public:
    CRNNTDecoder(CModelParams& params, token_t blank, uint32_t beamWidth, float insertionBoost, BeamSortKind beamSortKind)
        : m_evaluator(params),
          Blank(blank),
          m_insertionBoost(insertionBoost),
          P(m_evaluator.Predictor, token_trie),
          A(token_trie, beamWidth, beamSortKind),
          B(beamWidth, beamSortKind)
    {
    }

    void Reset()
    {
        m_evaluator.Reset();
        P.Reset(Blank);
        B.clear();
        token_trie.clear();

        BeamEntry b0(token_trie.walk(token_trie.Root, Blank), 0.0f);
        b0.LogPr = 0.0f;
        B.push(b0);
    }

    void Forward(const float* feats, size_t BaseFeatDim, RecombineKind recombineKind)
    {
        const auto& t = m_evaluator.Encoder.Forward(feats, BaseFeatDim);

        auto head_nodes = A.reset(B);
        P.Prune(head_nodes);
        B.clear();
        // fprintf(stderr, "%zu frames: A %u, B %u\n", ++frames, A.size(), B.size());
        // fprintf(stderr, "    A (top %f), B (%s)\n", A.top().LogPr, B.list().c_str());

        BeamEntry star(token_trie.InvalidNodeId, NAN);
        while (A.try_pop(star, B.threshold()))
        {
            const auto& j = m_evaluator.Joint.Forward_SoftMax(t, P.Ensure(star.Y));
            rassert_eq(j.M - 1, Blank);
            rassert_eq(isnan(star.LogPr), true);
            star.LogPr = star.LogAlpha + j[Blank];
            rassert_op(star.LogPr, <=, star.LogAlpha);
            B.push(star);

            A.recombine(star, j, Blank, B.threshold(), recombineKind, m_insertionBoost);

            // fprintf(stderr, "    A %u (top %f), B %u (top %f bottom %f)\n", A.size(), A.top().LogPr, B.size(), B.top().LogPr, B.bottom().LogPr);
            // fprintf(stderr, "    A (top %f), B (%s)\n", A.top().LogPr, B.list().c_str());
        }
    }

    struct ResultEntry
    {
        ResultEntry(const token_seq_t& y, float logPr, float normLogPr)
            : Y(y),
              LogPr(logPr),
              NormLogPr(normLogPr)
        {
        }

        token_seq_t Y;
        float LogPr;
        float NormLogPr;
    };

    std::vector<ResultEntry> GetResult(ScoreNormalizationKind scoreNormKind, float insertionBoostInFinalBeam = 0.24) const
    {
        std::vector<ResultEntry> result;
        rassert_eq(result.empty(), true);

        for (auto& entry : B)
        {
            auto y = token_trie.expand(entry.Y);
            auto logPr = entry.LogPr - (y.size() - 1) * m_insertionBoost;
            float normLogPr;

            normLogPr = logPr + (y.size() - 1) * insertionBoostInFinalBeam;
            normLogPr = NormalizeScore(y, normLogPr, scoreNormKind);

            result.emplace_back(y, logPr, normLogPr);
        }

        std::sort(
            result.begin(),
            result.end(),
            [](const ResultEntry& x, const ResultEntry& y) { return x.NormLogPr > y.NormLogPr; });
        return result;
    }

private:
    CModelEvaluator m_evaluator;
    token_t Blank;
    float m_insertionBoost;
    CTokenTrie token_trie;
    CPredictorCache P;
    CBeamA A;
    CBeamB B;
};
