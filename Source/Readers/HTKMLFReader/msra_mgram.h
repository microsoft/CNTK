//
// <copyright file="msra_mgram.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// msra_mgram.h -- simple ARPA LM read and access function
//

#pragma once

#include "Basics.h"
#include "fileutil.h" // for opening/reading the ARPA file
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm> // for various sort() calls
#include <math.h>

namespace msra { namespace lm {

// ===========================================================================
// core LM interface -- LM scores are accessed through this exclusively
// ===========================================================================

class ILM // generic interface -- mostly the score() function
{
public:
    virtual double score(const int *mgram, int m) const = 0;
    virtual bool oov(int w) const = 0; // needed for perplexity calculation
    // ... TODO (?): return true/false to indicate whether anything changed.
    // Intended as a signal to derived LMs that cache values.
    virtual void adapt(const int *data, size_t m) = 0; // (NULL,M) to reset, (!NULL,0) to flush

    // iterator for composing models --iterates in increasing order w.r.t. w
    class IIter
    {
    public:
        virtual operator bool() const = 0; // has iterator not yet reached end?
        // ... TODO: ensure iterators do not return OOVs w.r.t. user symbol table
        // (It needs to be checked which LM type's iterator currently does.)
        virtual void operator++() = 0; // advance by one
        // ... TODO: change this to key() or something like this
        virtual std::pair<const int *, int> operator*() const = 0; // current m-gram (mgram,m)
        virtual std::pair<double, double> value() const = 0;       // current (logP, logB)
    };
    virtual IIter *iter(int minM = 0, int maxM = INT_MAX) const = 0;
    virtual int order() const = 0;        // order, e.g. 3 for trigram
    virtual size_t size(int m) const = 0; // return #m-grams

    // diagnostics functions -- not all models implement these
    virtual int getLastLongestHistoryFound() const = 0;
    virtual int getLastLongestMGramFound() const = 0;
};

// ===========================================================================
// log-add helpers
// ===========================================================================

const double logzero = -1e30;

static inline double logadd(double x, double y)
{
    double diff = y - x;
    double sum = x; // x no longer used after this
    if (diff > 0)
    {
        sum = y;      // y no longer used after this
        diff = -diff; // that means we need to negate diff
    }
    if (diff > -24.0) // approx. from a constant from fmpe.h
        sum += log(1.0 + exp(diff));
    return sum;
}

// take the log, but clip to logzero
template <class FLOATTYPE> // float or double
static inline FLOATTYPE logclip(FLOATTYPE x)
{
    // ... TODO: use the proper constants here (slightly inconsistent)
    return x > (FLOATTYPE) 1e-30 ? log(x) : (FLOATTYPE) logzero;
}

// compute 1-P in logarithmic representation
static inline double invertlogprob(double logP)
{
    return logclip(1.0 - exp(logP));
}

// ===========================================================================
// CSymbolSet -- a simple symbol table
// ===========================================================================

// compare function to allow char* as keys (without, unordered_map will correctly
// compute a hash key from the actual strings, but then compare the pointers
// -- duh!)
struct less_strcmp : public binary_function<const char *, const char *, bool>
{ // this implements operator<
    bool operator()(const char *const &_Left, const char *const &_Right) const
    {
        return strcmp(_Left, _Right) < 0;
    }
};

class CSymbolSet : public std::unordered_map<const char *, int, std::hash<const char *>, less_strcmp>
{
    vector<const char *> symbols; // the symbols

    CSymbolSet(const CSymbolSet &);
    CSymbolSet &operator=(const CSymbolSet &);

public:
    CSymbolSet()
    {
        symbols.reserve(1000);
    }
    ~CSymbolSet()
    {
        clear();
    }

    void clear()
    {
        foreach_index (i, symbols)
            free((void *) symbols[i]);
        unordered_map::clear();
    }

    // operator[key] on a 'const' object
    // get id for an existing word, returns -1 if not existing
    int operator[](const char *key) const
    {
        unordered_map<const char *, int>::const_iterator iter = find(key);
        return (iter != end()) ? iter->second : -1;
    }

    // operator[key] on a non-'const' object
    // determine unique id for a word ('key')
    int operator[](const char *key)
    {
        unordered_map<const char *, int>::const_iterator iter = find(key);
        if (iter != end())
            return iter->second;

        // create
        const char *p = _strdup(key);
        if (!p)
            BadExceptionError("CSymbolSet:id string allocation failure");

        try
        {
            int id = (int) symbols.size();
            symbols.push_back(p); // we own the memory--remember to free it
            insert(make_pair(p, id));
            return id;
        }
        catch (...)
        {
            free((void *) p);
            throw;
        }
    }

    // return symbol string for a given id
    // Returned pointer is owned by this object.
    inline const char *operator[](int id) const
    {
        return symbols[id];
    }

    // overloads to be compatible with C++ strings and CSymMap
    int sym2existingId(const string &key) const
    {
        return (*this)[key.c_str()];
    }
    int sym2id(const string &key)
    {
        return (*this)[key.c_str()];
    }
    inline const char *id2sym(int id)
    {
        return (*this)[id];
    }

    // some helpers for writing and reading back a symbol set
    void write(FILE *f)
    {
        fputTag(f, "SYMS");       // header
        fputint(f, (int) size()); // symbol set
        foreach_index (k, symbols)
            fputstring(f, symbols[k]);
    }

    void read(FILE *f)
    {
        clear(); // clear out what was there before (typically nothing)
        fcheckTag(f, "SYMS");
        int numWords = fgetint(f);
        char buf[1000];
        for (int k = 0; k < numWords; k++)
        {
            fgetstring(f, buf);
            int id = (*this)[buf];
            if (id != k)
                RuntimeError("plsa: sequence error while reading vocabulary");
        }
    }
};

// ===========================================================================
// mgram_map -- lookup table for mgrams
// ===========================================================================

// variable naming convention for word ids:
//  - w   a word in user space
//        Defined by userSymMap::operator[](string) passed to read().
//        Data passed to score() and adapt() functions are in 'w' space.
//  - id  an id in internal LM space
//        E.g. defined by vocabulary in LM input file.
// All external LM accesses involve an implicit mapping, including:
//  w -> id  --for calls to score() and adapt()
//  id -> w  --for iterators (IIter++ orders by and *IIter returns keys in 'w' space)

// representation of LM in memory
// LMs are stored sparsely, i.e. only used elements are stored.
// For each m-gram, a score is stored. For each history, a back-off weight is stored.
// Both are stored in flat arrays, one per order, that are concatenations of
// individual arrays per history.
// The mgram_map provides a measure of locating these entries. For each level,
// it stores a flat array of 'firsts' which point to the first child entry in
// the next level (the next 'firsts' value denotes the end).
// The mgram_map also stores word ids, which are the indexes of the sparse
// elements.
// To access an m-gram score of back-off weight, the mgram_map structure is
// traversed, involving a binary search operation at each level.

// a compact vector to hold 24-bit vaulues
class int24_vector : std::vector<unsigned char>
{
public:
    // basic (non-tricky) operations --just multiply anything by 3
    int24_vector()
    {
    }
    int24_vector(size_t n)
        : std::vector<unsigned char>(n * 3)
    {
    }
    void resize(size_t n)
    {
        std::vector<unsigned char> &base = *this;
        base.resize(n * 3);
    }
    void reserve(size_t n)
    {
        std::vector<unsigned char> &base = *this;
        base.reserve(n * 3);
    }
    void swap(int24_vector &other)
    {
        std::vector<unsigned char> &base = *this;
        base.swap(other);
    }
    size_t size() const
    {
        const std::vector<unsigned char> &base = *this;
        return base.size() / 3;
    }
    bool empty() const
    {
        const std::vector<unsigned char> &base = *this;
        return base.empty();
    }

    // a reference to a 3-byte int (not a naked pointer as we cannot just assign to it)
    template <class T>
    class uint24_ref_t
    {
    protected:
        T p;
        friend class int24_vector; // only int24_vector may instantiate this
        __forceinline uint24_ref_t(T p)
            : p(p)
        {
        }

    public:
        // access
        __forceinline operator int() const
        {
            return (((((signed char) p[2]) << 8) + p[1]) << 8) + p[0];
        }
    };
    typedef uint24_ref_t<const unsigned char *> const_uint24_ref; // const version (only read)
    class uint24_ref : public uint24_ref_t<unsigned char *>       // non-const (read and assign)
    {
        static void overflow()
        {
            RuntimeError("uint32_ref: attempting to store value > 24 bits");
        }

    protected:
        friend class int24_vector; // only int24_vector may instantiate this
        __forceinline uint24_ref(unsigned char *p)
            : uint24_ref_t(p)
        {
        }

    public:
        // assignment operator
        __forceinline int operator=(int value)
        {
            if ((unsigned int) (value + 0x800000) > 0xffffff)
                overflow();
            p[0] = (unsigned char) value;
            p[1] = (unsigned char) (value >> 8);
            p[2] = (unsigned char) (value >> 16);
            assert(value == (int) *this);
            return value;
        }
    };

    // reading and writing
    __forceinline uint24_ref operator[](size_t i)
    {
        std::vector<unsigned char> &base = *this;
        return uint24_ref(&base[i * 3]);
    }
    __forceinline const_uint24_ref operator[](size_t i) const
    {
        const std::vector<unsigned char> &base = *this;
        return const_uint24_ref(&base[i * 3]);
    }
    __forceinline int back() const
    {
        const std::vector<unsigned char> &base = *this;
        return const_uint24_ref(&base[base.size() - 3]);
    }
    void push_back(int value)
    {
        std::vector<unsigned char> &base = *this;
        size_t cursize = base.size();
        size_t newsize = cursize + 3;
        if (newsize > base.capacity())
            base.reserve(newsize * 2); // double the size to ensure constant-time
        base.resize(newsize);
        uint24_ref r = uint24_ref(&base[cursize]);
        r = value;
        assert(value == back());
    }
};

// maps from m-grams to m-gram storage locations.
class mgram_map
{
    typedef unsigned int index_t; // (-> size_t when we really need it)
    //typedef size_t index_t;                   // (tested once, seems to work)
    static const index_t nindex; // invalid index
    // entry [m][i] is first index of children in level m+1, entry[m][i+1] the end.
    int M;                                    // order, e.g. M=3 for trigram
    std::vector<std::vector<index_t>> firsts; // [M][i] ([0] = zerogram = root)
    std::vector<int24_vector> ids;            // [M+1][i] ([0] = not used)
    bool level1nonsparse;                     // true: level[1] can be directly looked up
    std::vector<index_t> level1lookup;        // id->index for unigram level
    static void fail(const char *msg)
    {
        RuntimeError("mgram_map::%s", msg);
    }

    // mapping from w -> i -- users pass 'w', internally we use our own 'ids'
    std::vector<int> w2id; // w -> id
    std::vector<int> id2w; // id -> w
    int idmax;             // max id ever encountered by create()
    inline int map(int w) const
    {
        if (w < 0 || w >= (int) w2id.size())
            return -1;
        else
            return w2id[w];
    }

    // get index for 'id' in level m+1, as a child of index i in level m.
    // Returns -1 if not found.
    // This is a relatively generic binary search.
    inline index_t find_child(int m, index_t i, int id) const
    {
        // unigram level is a special case where we can avoid searching
        if (m == 0)
        {
            if (id < 0)
                return nindex;
            index_t i;
            if (level1nonsparse)
                i = (index_t) id;
            else // sparse: use a look-up table
            {
                if ((size_t) id >= level1lookup.size())
                    return nindex;
                i = level1lookup[id];
            }
            assert(i == nindex || ids[1][i] == id);
            return i;
        }
        index_t beg = firsts[m][i];
        index_t end = firsts[m][i + 1];
        const int24_vector &ids_m1 = ids[m + 1];
        while (beg < end)
        {
            index_t i = (beg + end) / 2;
            int v = ids_m1[i];
            if (id == v)
                return i; // found it
            else if (id < v)
                end = i; // id is left of i
            else
                beg = i + 1; // id is right of i
        }
        return nindex; // not found
    }

public:
    // --- allocation

    mgram_map()
    {
    }
    mgram_map(int p_M)
    {
        init(p_M);
    }

    // construct
    void init(int p_M)
    {
        clear();
        M = p_M;
        firsts.assign(M, std::vector<index_t>(1, 0));
        ids.assign(M + 1, int24_vector());
        ids[0].resize(1); // fake zerogram entry for consistency
        ids[0][0] = -1;
    }
    // reserve memory for a level
    void reserve(int m, size_t size)
    {
        if (m == 0)
            return; // cannot reserve level 0
        ids[m].reserve(size);
        if (m < M)
            firsts[m].reserve(size + 1);
        if (m == 1)
            level1lookup.reserve(size);
    }
    // allow to reduce M after the fact
    void resize(int newM)
    {
        if (newM > M)
            fail("resize() can only shrink");
        M = newM;
        firsts.resize(M);
        ids.resize(M + 1);
    }
    // destruct
    void clear()
    {
        M = 0;
        firsts.clear();
        ids.clear();
        w2id.clear();
        id2w.clear();
        idmax = -1;
    }
    // size
    inline int size(int m) const
    {
        return (int) ids[m].size();
    }
    // swap --used e.g. in merging
    void swap(mgram_map &other)
    {
        ::swap(M, other.M);
        firsts.swap(other.firsts);
        ids.swap(other.ids);
        ::swap(level1nonsparse, other.level1nonsparse);
        level1lookup.swap(other.level1lookup);
        w2id.swap(other.w2id);
        id2w.swap(other.id2w);
        ::swap(idmax, other.idmax);
    }

    // --- id mapping

    // test whether a word id is known in this model
    inline bool oov(int w) const
    {
        return map(w) < 0;
    }

    // return largest used word id (=last entry in unigram ids[])
    int maxid() const
    {
        return idmax;
    }

    // return largest used w (only after created())
    int maxw() const
    {
        return -1 + (int) w2id.size();
    }

    // map is indexed with a 'key'.
    // A key represents an m-gram by storing a pointer to the original array.
    // The key allows to remove predicted word (pop_w()) or history (pop_h()).
    class key
    {
    protected:
        friend class mgram_map;
        const int *mgram; // pointer to mgram array --key does not own that memory!
        int m;            // elements in mgram array
    public:
        // constructors
        inline key()
            : mgram(NULL), m(0)
        {
        } // required for use in std::vector
        inline key(const int *mgram, int m)
            : mgram(mgram), m(m)
        {
        }
        // manipulations
        inline key pop_h() const
        {
            if (m == 0)
                fail("key::pop_h() called on empty key");
            return key(mgram + 1, m - 1);
        }
        inline key pop_w() const
        {
            if (m == 0)
                fail("key::pop_w() called on empty key");
            return key(mgram, m - 1);
        }
        // access
        inline int back() const
        {
            if (m == 0)
                fail("key::back() called on empty key");
            return mgram[m - 1];
        }
        inline const int &operator[](int n) const
        {
            if (n < 0 || n >= m)
                fail("key::operator[] out of bounds");
            return mgram[n];
        }
        inline int order() const
        {
            return m;
        }
        // key comparison (used in sorting and merging)
        inline bool operator<(const key &other) const
        {
            for (int k = 0; k < m && k < other.m; k++)
                if (mgram[k] != other.mgram[k])
                    return mgram[k] < other.mgram[k];
            return m < other.m;
        }
        inline bool operator>(const key &other) const
        {
            return other < *this;
        }
        inline bool operator<=(const key &other) const
        {
            return !(*this > other);
        }
        inline bool operator>=(const key &other) const
        {
            return !(*this < other);
        }
        inline bool operator==(const key &other) const
        {
            if (m != other.m)
                return false;
            for (int k = 0; k < m; k++)
                if (mgram[k] != other.mgram[k])
                    return false;
            return true;
        }
        inline bool operator!=(const key &other) const
        {
            return !(*this == other);
        }
    };

    // 'coord' is an abstract coordinate of an m-gram. This is returned by
    // operator[], and is used as an index in our sister structure, mgram_data.
    struct coord
    {
        index_t i;        // index in that level -- -1 means not found
        unsigned short m; // level
        inline bool valid() const
        {
            return i != nindex;
        }
        inline void validate() const
        {
            if (!valid())
                fail("coord used but invalid");
        }
        void invalidate()
        {
            i = nindex;
        }
        inline int order() const
        {
            validate();
            return m;
        }
        inline coord(int m, index_t i)
            : m((unsigned short) m), i(i)
        {
        } // valid coord
        // ^^ this is where we'd test for index_t overflow if we ever need it
        inline coord(bool valid = true)
            : m(0), i(valid ? 0 : nindex)
        {
        } // root or invalid
    };

    // 'foundcoord' is an extended 'coord' as returned by operator[], with
    // information on whether it is valid or not, and whether it refers to
    // an m-gram or to a history only.
    class foundcoord : public /*<-want to get rid of this*/ coord
    {
        const short type;
        foundcoord &operator=(const foundcoord &);

    public:
        inline bool valid_w() const
        {
            return type > 0;
        }
        inline bool valid_h() const
        {
            return type == 0;
        }
        inline bool valid() const
        {
            return type >= 0;
        }
        inline operator const coord &() const
        {
            return *this;
        }
        inline foundcoord(short type, int m, index_t i)
            : type(type), coord(m, i)
        {
        }
        inline foundcoord(short type)
            : type(type), coord(type >= 0)
        {
        }
    };

    // search for an mgram -- given a 'key', return its 'coord.'
    // If m-gram is found, type=1. If only history found then type=0, and
    // coord represents the history token instead.
    // The given key may not be longer than our storage (we do not automatically
    // truncate because that would not be detectable by caller).
    __forceinline foundcoord operator[](const key &k) const
    {
        if (k.m > M) // call truncate() first with too long keys
            fail("operator[] called with too long key");
        if (k.m == 0)
            return foundcoord(1); // zerogram -> root

        // We traverse history one by one.
        index_t i = 0;
        for (int n = 1; n < k.m; n++)
        {
            int w = k[n - 1]; // may be -1 for unknown word
            int id = map(w);  // may still be -1
            //const char * sym = idToSymbol (id); sym;   // (debugging)
            i = find_child(n - 1, i, id);
            if (i == nindex)           // unknown history: fall back
                return foundcoord(-1); // indicates failure
            // found it: advance search by one history token
        }

        // Found history. Do we also find the prediced word?
        int w = k[k.m - 1]; // may be -1 for unknown word
        int id = map(w);    // may still be -1
        index_t i_m = find_child(k.m - 1, i, id);
        if (i_m == nindex) // not found
            return foundcoord(0, k.m - 1, i);
        else // found
            return foundcoord(1, k.m, i_m);
    }

    // truncate a key to the m-gram length supported by this
    inline key truncate(const key &k) const
    {
        if (k.m <= M)
            return k;
        else
            return key(k.mgram + (k.m - M), M);
    }

    // --- iterators
    //  - iterating over children of a history
    //  - deep-iterating over the entire tree

    // for (iterator iter (mgram_map, parent_coord); iter; ++iter) { mgram_data[iter]; w=*iter; }
    class iterator : public coord
    {
        index_t end;          // end index: i is invalid when it reaches this
        const mgram_map &map; // remembered for operator*
        void operator=(const iterator &);

    public:
        // bool: true if can use or increment
        inline operator bool() const
        {
            return i < end;
        }
        // increment
        inline void operator++()
        {
            if (i < end)
                i++;
            else
                fail("iterator used beyond end");
        }
        // retrieve word -- returns -1 if not used in user's w->id map, e.g. skipped word
        inline int operator*() const
        {
            if (i >= end)
                fail("iterator used beyond end");
            return map.id2w[map.ids[m][i]];
        }
        // construct 'coord' as first element
        iterator(const mgram_map &map, const coord &c)
            : map(map)
        {
            c.validate();
            // get the range
            index_t beg = map.firsts[c.m][c.i]; // first element of child
            end = map.firsts[c.m][c.i + 1];     // end = first of next entry
            // set the first child coordinate
            m = c.m + 1; // we iterate over the child level
            i = beg;     // first element
        }
        // alternative to loop over all m-grams of a level
        iterator(const mgram_map &map, int m)
            : map(map), coord(m, 0)
        {
            end = (m > 0) ? (index_t) map.ids[m].size() : 1; // loop over entire vector
        }
    };

    // for (deep_iterator iter (mgram_map, maxM); iter; ++iter) { mgram_data[iter]; key=*iter; }
    class deep_iterator : public coord
    {
    protected:
        int maxM;
        std::vector<index_t> pos; // current position [0..m]
        std::vector<int> mgram;   // current m-gram corresponding to 'pos'
        const mgram_map &map;     // remembered for operator*
        void operator=(const deep_iterator &);
        void validate() const
        {
            if (!valid())
                fail("iterator used beyond end");
        }

    public:
        // constructor
        deep_iterator(const mgram_map &map, int p_maxM = -1)
            : map(map), maxM(p_maxM), coord(map.firsts[0].size() >= 2)
        {
            if (maxM == -1)
                maxM = map.M;
            else if (maxM > map.M)
                fail("deep_iterator instantiated for invalid maximum depth");
            mgram.resize(maxM, -1);
            pos.resize(maxM + 1, 0);
        }
        // bool: true if can use or increment
        inline operator bool() const
        {
            return valid();
        }
        // increment
        inline void operator++()
        {
            validate();
            // if current position has a child then enter it
            if (m < maxM && m < map.M && map.firsts[m][pos[m]] < map.firsts[m][pos[m] + 1])
            {
                i = map.firsts[m][pos[m]];
                m++;
                pos[m] = i;
                mgram[m - 1] = map.id2w[map.ids[m][i]];
                return;
            }
            // advance vertically or step up one level
            for (; m > 0;)
            {
                // advance current position if still elements left
                i++;
                if (i < map.firsts[m - 1][pos[m - 1] + 1]) // not hit the end yet
                {
                    pos[m] = i;
                    mgram[m - 1] = map.id2w[map.ids[m][i]];
                    return;
                }
                // cannot enter or advance: step back one
                m--;
                i = pos[m]; // parent position
            }
            // reached the end
            invalidate(); // invalidates 'coord'--next call to bool() will return false
            return;
        }
        // retrieve keys -- returns -1 if not used in user's w->id map, e.g. skipped word
        // The key points into the iterator structure, i.e. it operator++ invalidates it!
        inline key operator*() const
        {
            validate();
            return key(&mgram[0], m);
        }
    };

    // for (reordering_iterator iter (mgram_map, wrank[], maxM); iter; ++iter) { mgram_data[iter]; key=*iter; }
    // Like deep_iterator, but iterates the map such that ws are returned in
    // increasing wrank[w] rather than in the original storage order.
    // Used for merging multiple models such as linear interpolation.
    class reordering_iterator : public deep_iterator
    {
        const std::vector<int> &wrank;             // assigns a rank to each w
        const char *i;                             // hide coord::i against accidental access
        std::vector<std::vector<index_t>> indexes; // coord::i <- indexes[m][this->i]
        std::vector<index_t> indexbase;            // indexes[m] is indexbase[m]-based
        inline index_t &index_at(int m, index_t i)
        {
            return indexes[m][i - indexbase[m]];
        }
        std::vector<std::pair<int, int>> sortTemp; // temp for creating indexes
        void operator=(const reordering_iterator &);

    public:
        // constructor
        reordering_iterator(const mgram_map &map, const std::vector<int> &wrank, int p_maxM = -1)
            : deep_iterator(map, p_maxM), wrank(wrank)
        {
            if (wrank.size() < map.w2id.size())
                fail("reordering_iterator: wrank has wrong dimension");
            indexes.resize(maxM + 1);
            indexes[0].push_back(0); // look-up table for root: only one item
            indexbase.resize(maxM + 1, 0);
            pos[0] = coord::i; // zerogram level: same i because no mapping there
            if (map.M >= 1)
                sortTemp.reserve(map.size(1));
        }
        // increment
        // We iterate through the map using (m, pos[m]) while user consumes (m, i)
        // i.e. for operator++(), coord::i is not iterated but a return value.
        inline void operator++()
        {
            validate();
            // if current position has a child then enter it
            // Note: We enter the item that coord::i points to, which is not pos[m]
            // but the mapped pos[m].
            if (m < maxM && m < map.M && map.firsts[m][index_at(m, pos[m])] < map.firsts[m][index_at(m, pos[m]) + 1])
            {
                // enter the level
                index_t beg = map.firsts[m][index_at(m, pos[m])]; // index range of sub-level
                index_t end = map.firsts[m][index_at(m, pos[m]) + 1];
                m++;
                pos[m] = beg;
                // build look-up table for returned values
                size_t num = end - beg;
                // we sort i by rank (and i, keeping original order for identical rank)
                sortTemp.resize(end - beg);
                foreach_index (k, sortTemp)
                {
                    index_t i = beg + k;
                    int id = map.ids[m][i];
                    int w = map.id2w[id];
                    sortTemp[k] = std::make_pair(wrank[w], i);
                }
                std::sort(sortTemp.begin(), sortTemp.end());
                // remember sorted i's
                indexbase[m] = beg; // used by index_at (m, *)
                indexes[m].resize(num);
                foreach_index (k, sortTemp)
                    index_at(m, k + beg) = sortTemp[k].second;
                // set up return values
                coord::i = index_at(m, pos[m]);
                mgram[m - 1] = map.id2w[map.ids[m][coord::i]];
                return;
            }
            // advance vertically or step up one level
            for (; m > 0;)
            {
                // advance current position if still elements left
                // use our own i (in pos[m]), then map to coord::i using sorted list
                pos[m]++;
                if (pos[m] < map.firsts[m - 1][index_at(m - 1, pos[m - 1]) + 1]) // not hit the end yet
                {
                    coord::i = index_at(m, pos[m]);
                    mgram[m - 1] = map.id2w[map.ids[m][coord::i]];
                    return;
                }
                // cannot enter or advance: step back one
                m--;
            }
            // reached the end
            invalidate(); // invalidates 'coord'--next call to bool() will return false
            return;
        }
    };

    // --- functions for building

    // 'unmapped_key' contains original 'id' rather than 'w' values. It is only
    // used for create()--at creation time, we use our private mapping.
    typedef key unmapped_key;

// create a new key (to be called in sequence).
// Only the last word given in the key is added. The history of the given
// mgram must already exist and must be the last.
// Important: Unlike operator[], create() takes an unmapped_key, i.e. the
// mapping is not applied.
// 'cache' is used for speed-up, it must be as large as key.m-1 and
// initialized to 0.
#pragma warning(push)           // known compiler bug: size_t (marked _w64) vs. unsigned...
#pragma warning(disable : 4267) // ...int (not marked) incorrectly flagged in templates
    typedef std::vector<index_t> cache_t;
    coord create(const unmapped_key &k, cache_t &cache)
    {
        if (k.m < 1)
            return coord(); // (root need not be created)
        // locate history (must exist), also updates cache[]
        bool prevValid = true;
        index_t i = 0; // index of history in level k.m-1
        if (cache.empty())
            cache.resize(M, nindex); // lazy initialization
        for (int m = 1; m < k.m; m++)
        {
            int thisid = k[m - 1];
            if (prevValid && cache[m - 1] != nindex && ids[m][cache[m - 1]] == thisid)
            {
                i = cache[m - 1]; // get from cache
                continue;
            }
            // need to actually search
            i = find_child(m - 1, i, thisid);
            if (i == nindex)
                fail("create() called with unknown history");
            cache[m - 1] = i;
            prevValid = false;
        }
        for (int m = k.m; m < M && cache[m - 1] != nindex; m++)
            cache[m - 1] = nindex; // clear upper entries (now invalid)
        // now i is the index of the id of the last history item
        // make the firsts entry if not there yet
        bool newHist = (firsts[k.m - 1].size() < (size_t) i + 2);
        while (firsts[k.m - 1].size() < (size_t) i + 2) // [i+1] is the end for this array
            firsts[k.m - 1].push_back((mgram_map::index_t) ids[k.m].size());
        if (firsts[k.m - 1].size() != (size_t) i + 2)
            fail("create() called out of order (history)");
        // create new word id
        int thisid = k[k.m - 1];
        if (!newHist && thisid <= ids[k.m].back())
            fail("create() called out of order");
        // keep track of idmax
        if (thisid > idmax)
            idmax = thisid;

        coord c(k.m, (index_t) ids[k.m].size());

        assert(firsts[k.m - 1].back() == (index_t) ids[k.m].size());
        ids[k.m].push_back(thisid); // create value
        firsts[k.m - 1].back() = (index_t) ids[k.m].size();
        if (firsts[k.m - 1].back() != (index_t) ids[k.m].size())
            fail("create() numeric overflow--index_t too small");
        assert(k.m == M || firsts[k.m].back() == (index_t) ids[k.m + 1].size());

        // optimization: level1nonsparse flag
        // If unigram level is entirely non-sparse, we can save the search
        // operation at that level, which is significantly slower than for the
        // much sparser higher levels.
        if (c.m == 1)
        {
            if (c.i == 0)
                level1nonsparse = true;                   // first entry
            level1nonsparse &= (c.i == (index_t) thisid); // no search needed
            level1lookup.resize(thisid + 1, nindex);
            level1lookup[thisid] = c.i;
        }

        return c;
    }
#pragma warning(pop)

    // call this at the end
    //  - establish the w->id mapping that is used in operator[]
    //  - finalize the firsts arrays
    // This function swaps the user-provided map and our current one.
    // We use swapping to avoid the memory allocation (noone else outside should
    // have to keep the map).
    // This function also builds our internal reverse map used in the iterator.
    void created(std::vector<int> &userToLMSymMap)
    {
        // finalize firsts arrays
        foreach_index (m, firsts)
            firsts[m].resize(ids[m].size() + 1, (int) ids[m + 1].size());
        foreach_index (m, firsts)
        {
            assert(firsts[m][0] == 0);
            foreach_index (i, ids[m])
                assert(firsts[m][i] <= firsts[m][i + 1]);
            assert((size_t) firsts[m].back() == ids[m + 1].size());
        }
        // id mapping
        // user-provided w->id map
        ::swap(w2id, userToLMSymMap);
        // reverse map
        id2w.assign(maxid() + 1, nindex);
        foreach_index (w, w2id)
        {
            int id = w2id[w];
            if (id < 0)
                continue; // invalid word
            if (id > maxid())
                continue; // id not in use
            id2w[id] = w;
        }
    }

    // helper for created()--return an identical map, as we have several
    // occasions where such a map is passed as userToLMSymMap to created().
    std::vector<int> identical_map(size_t n = SIZE_MAX) const
    {
        if (n == SIZE_MAX)
            n = maxid() + 1;
        std::vector<int> v(n);
        foreach_index (i, v)
            v[i] = i;
        return v;
    }

    // decide whether iterator will return in increasing w order
    bool inorder() const
    {
#if 0 // fix this: need access to w2id, or have an inorder() function in mgram_map
        bool inorder = true;
        for (int i = 1; inorder && i < (int) map.w2id.size(); i++)
            inorder &= (map.w2id[i+1] >= map.w2id[i]);
#endif
        return false;
    }
};

// ===========================================================================
// mgram_data -- data stored according to mgram_map
// Separate from mgram_map, so that we can share the same map for multiple data.
// ===========================================================================

template <class DATATYPE>
class mgram_data
{
    std::vector<std::vector<DATATYPE>> data;
    static void fail(const char *msg)
    {
        RuntimeError("mgram_data::%s", msg);
    }

public:
    mgram_data()
    {
    }
    mgram_data(int M)
    {
        init(M);
    }
    // for an M-gram, indexes [0..M] are valid thus data[] has M+1 elements
    void init(int M)
    {
        data.assign(M + 1, std::vector<DATATYPE>());
    }
    void reserve(int m, size_t size)
    {
        data[m].reserve(size);
    }
    void resize(int M)
    {
        if ((size_t) M + 1 <= data.size())
            data.resize(M + 1);
        else
            fail("resize() can only shrink");
    }
    size_t size(int m) const
    {
        return data[m].size();
    }
    size_t size() const
    {
        size_t sz = 0;
        foreach_index (m, data)
            sz += size(m);
        return sz;
    }
    void clear()
    {
        data.clear();
    }
    void swap(mgram_data &other)
    {
        data.swap(other.data);
    }
    // access existing elements. Usage:
    // DATATYPE & element = mgram_data[mgram_map[mgram_map::key (mgram, m)]]
    __forceinline DATATYPE &operator[](const mgram_map::coord &c)
    {
        c.validate();
        return data[c.m][c.i];
    }
    __forceinline const DATATYPE &operator[](const mgram_map::coord &c) const
    {
        c.validate();
        return data[c.m][c.i];
    }
    // create entire vector (for random-access situations).
    void assign(int m, size_t size, const DATATYPE &value)
    {
        data[m].assign(size, value);
    }
    // create an element. We can only append.
    inline void push_back(const mgram_map::coord &c, const DATATYPE &val)
    {
        c.validate();
        if (data[c.m].size() != (size_t) c.i)
            fail("push_back() only allowed for last entry");
        data[c.m].push_back(val);
    }
};

// ===========================================================================
// CMGramLM -- a back-off M-gram language model in memory, loaded from an ARPA file
// ===========================================================================

class CMGramLM : public ILM
{
protected:
#if 0
    void clear()        // release all memory --object unusable after this
    {
        M = -1;
        map.clear();
        logP.clear();
        logB.clear();
    }
#endif
    int M; // e.g. M=3 for trigram
    // ^^ TODO: can we do away with this entirely and replace it by map.order()/this->order()
    mgram_map map;
    mgram_data<float> logP; // [M+1][i] probabilities
    mgram_data<float> logB; // [M][i] back-off weights (stored for histories only)
    friend class CMGramLMIterator;

    // diagnostics of previous score() call
    mutable int longestMGramFound;   // longest m-gram (incl. predicted token) found
    mutable int longestHistoryFound; // longest history (excl. predicted token) found

    // this function is for reducing M after the fact, e.g. during estimation
    // ... TODO: rethink the resize business. It is for shrinking only.
    void resize(int newM)
    {
        M = newM;
        map.resize(M);
#if 0 // ... BUGBUG: we call this before logP/logB exist
        logP.resize (M);
        logB.resize (M-1);
#endif
    }

public:
    CMGramLM()
        : M(-1)
    {
    } // needs explicit initialization through read() or init()

    virtual int getLastLongestHistoryFound() const
    {
        return longestHistoryFound;
    }
    virtual int getLastLongestMGramFound() const
    {
        return longestMGramFound;
    }

    // -----------------------------------------------------------------------
    // score() -- compute an m-gram score (incl. back-off and fallback)
    // -----------------------------------------------------------------------
    // mgram[m-1] = word to predict, tokens before that are history
    // m=3 means trigram
    virtual double score(const int *mgram, int m) const
    {
        longestHistoryFound = 0; // (diagnostics)

        double totalLogB = 0.0; // accumulated back-off

        for (mgram_map::key key = map.truncate(mgram_map::key(mgram, m));; key = key.pop_h())
        {
            // look up the m-gram
            const mgram_map::foundcoord c = map[key];

            // (diagnostics -- can be removed if not used)
            if (c.valid() && key.order() - 1 > longestHistoryFound)
                longestHistoryFound = key.order() - 1;
            if (c.valid_w())
                longestMGramFound = key.order();

            // full m-gram found -> return it (zerogram always considered found)
            if (c.valid_w())
                return totalLogB + logP[c];

            // history found but predicted word not -> back-off
            if (c.valid_h())          // c is coordinate of parent instead
                totalLogB += logB[c]; // and continue like fall back

            // history not found -> fall back
        } // and go again with the shortened history
    }

    // same as score() but without optimizations (for reference)
    // ... this is really no longer needed
    virtual double score_unoptimized(const int *mgram, int m) const
    {
        return score_unoptimized(map.truncate(mgram_map::key(mgram, m)));
    }

    inline double score_unoptimized(const mgram_map::key &key) const
    {
        // look up the m-gram
        const mgram_map::foundcoord c = map[key];

        // full m-gram found -> return it
        if (c.valid_w())
            return logP[c];

        // history found but predicted word not -> back-off
        else if (c.valid_h()) // c is coordinate of patent instead
            return logB[c] + score_unoptimized(key.pop_h());

        // history not found -> fall back
        else
            return score_unoptimized(key.pop_h());
    }

    // test for OOV word (OOV w.r.t. LM)
    virtual bool oov(int w) const
    {
        return map.oov(w);
    }

    virtual void adapt(const int *, size_t)
    {
    } // this LM does not adapt

private:
    // keep this for debugging
    std::wstring filename; // input filename
    struct SYMBOL
    {
        string symbol; // token
        int id;        // numeric id in LM space (index of word read)
        bool operator<(const SYMBOL &other) const
        {
            return symbol < other.symbol;
        }
        SYMBOL(int p_id, const char *p_symbol)
            : id(p_id), symbol(p_symbol)
        {
        }
    };
    std::vector<SYMBOL> lmSymbols; // (id, word) symbols used in LM
    std::vector<int> idToSymIndex; // map LM id to index in lmSymbols[] array

    // search for a word in the sorted word array.
    // Only use this after sorting, i.e. after full 1-gram section has been read.
    // Only really used in read().
    inline int symbolToId(const char *word) const
    {
        int beg = 0;
        int end = (int) lmSymbols.size();
        while (beg < end)
        {
            int i = (beg + end) / 2;
            const char *v = lmSymbols[i].symbol.c_str();
            int cmp = strcmp(word, v);
            if (cmp == 0)
                return lmSymbols[i].id; // found it
            else if (cmp < 0)
                end = i; // id is left of i
            else
                beg = i + 1; // id is right of i
        }
        return -1; // not found
    }

    inline const char *idToSymbol(int id) const
    {
        if (id < 0)
            return NULL; // empty string for unknown ids
        int i = idToSymIndex[id];
        return lmSymbols[i].symbol.c_str();
    }

private:
    // type cast to const char*, to allow write() to use both const char* and string
    static const char *const_char_ptr(const char *p)
    {
        return p;
    }
    static const char *const_char_ptr(const string &s)
    {
        return s.c_str();
    }

public:
    // write model out as an ARPA (text) file.
    // symbols can be anything that has symbols[w] -> std::string& or const char*
    template <class SYMMAP>
    void write(FILE *outf, const SYMMAP &symbols, int M = INT_MAX) const
    {
        if (M > this->M)
            M = this->M; // clip; also covers default value
        if (M < 1 || map.size(1) == 0)
            RuntimeError("write: attempting to write empty model");

        // output header
        //  \data\
        //  ngram 1=58289
        //  ngram 2=956100
        //  ...
        fprintfOrDie(outf, "\\data\\\n");
        for (int m = 1; m <= M; m++)
        {
            fprintfOrDie(outf, "ngram %d=%d\n", m, map.size(m));
        }
        fflushOrDie(outf);

        // output m-grams themselves
        // M-gram sections
        const double log10 = log(10.0);
        for (int m = 1; m <= M; m++)
        {
            fprintf(stderr, "estimate: writing %d %d-grams..", map.size(m), m);
            int step = (int) logP.size(m) / 100;
            if (step == 0)
                step = 1;
            int numMGramsWritten = 0;

            // output m-gram section
            fprintfOrDie(outf, "\n\\%d-grams:\n", m);
            for (mgram_map::deep_iterator iter(map, m); iter; ++iter)
            {
                if (iter.order() != m) // a parent
                    continue;

                const mgram_map::key key = *iter;
                assert(m == key.order());

                // --- output m-gram to ARPA file
                fprintfOrDie(outf, "%.4f", logP[iter] / log10);
                for (int k = 0; k < m; k++)
                { // the M-gram words
                    int wid = key[k];
                    const char *w = const_char_ptr(symbols[wid]);
                    fprintfOrDie(outf, " %s", w);
                }

                if (m < M)
                { // back-off weight (not for highest order)
                    fprintfOrDie(outf, " %.4f", logB[iter] / log10);
                }
                fprintfOrDie(outf, "\n");

                // progress
                if (numMGramsWritten % step == 0)
                {
                    fprintf(stderr, ".");
                }
                numMGramsWritten++;
            }
            fflushOrDie(outf);
            assert(numMGramsWritten == map.size(m));
            fprintf(stderr, "\n");
        }

        fprintfOrDie(outf, "\n\\end\\\n");
        fflushOrDie(outf);
    }

    // get TopM Ngram probability
    // GangLi add this function to do probability pruning
    double KeepTopMNgramThreshold(int topM, int ngram)
    {
        // initial return as a very low value
        double probThrshold = -99;

        // check if nessary to prune
        if (map.size(ngram) > topM)
        {
            std::vector<std::pair<int, float>> probArray;
            probArray.reserve(map.size(ngram));
        }

        return probThrshold;
    }

protected:
    // replace zerogram prob by one appropriate for OOVs
    // We use the minimum of all unigram scores (assuming they represent singleton
    // events, which are closest to a zerogram--a better choice may be a leaving-
    // one-out estimate?).
    // Back-off weight is reset to 1.0 such that there is no extra penalty on it.
    void updateOOVScore()
    {
        float unknownLogP = 0.0f;
        for (mgram_map::iterator iter(map, mgram_map::coord()); iter; ++iter)
        {
            if (logP[iter] < -98.0f)
                continue; // disabled token, such as <s>, does not count
            if (logP[iter] < unknownLogP)
                unknownLogP = logP[iter];
        }
        logP[mgram_map::coord()] = unknownLogP;
        logB[mgram_map::coord()] = 0.0f;
    }

public:
    // read an ARPA (text) file.
    // Words do not need to be sorted in the unigram section, but the m-gram
    // sections have to be in the same order as the unigrams.
    // The 'userSymMap' defines the vocabulary space used in score().
    // If 'filterVocabulary' then LM entries for words not in userSymMap are skipped.
    // Otherwise the userSymMap is updated with the words from the LM.
    // 'maxM' allows to restrict the loading to a smaller LM order.
    // SYMMAP can be e.g. CSymMap or CSymbolSet.
    template <class SYMMAP>
    void read(const std::wstring &pathname, SYMMAP &userSymMap, bool filterVocabulary, int maxM)
    {
        int lineNo = 0;
        auto_file_ptr f(fopenOrDie(pathname, L"rbS"));
        fprintf(stderr, "read: reading %ls", pathname.c_str());
        filename = pathname; // (keep this info for debugging)

        // --- read header information

        // search for header line
        char buf[1024];
        lineNo++, fgetline(f, buf);
        while (strcmp(buf, "\\data\\") != 0 && !feof(f))
            lineNo++, fgetline(f, buf);
        lineNo++, fgetline(f, buf);

        // get the dimensions
        std::vector<int> dims;
        dims.reserve(4);

        while (buf[0] == 0 && !feof(f))
            lineNo++, fgetline(f, buf);

        int n, dim;
        dims.push_back(1); // dummy zerogram entry
        while (sscanf(buf, "ngram %d=%d", &n, &dim) == 2 && n == (int) dims.size())
        {
            dims.push_back(dim);
            lineNo++, fgetline(f, buf);
        }

        M = (int) dims.size() - 1;
        if (M == 0)
            RuntimeError("read: mal-formed LM file, no dimension information (%d): %ls", lineNo, pathname.c_str());
        int fileM = M;
        if (M > maxM)
            M = maxM;

        // allocate main storage
        map.init(M);
        logP.init(M);
        logB.init(M - 1);
        for (int m = 0; m <= M; m++)
        {
            map.reserve(m, dims[m]);
            logP.reserve(m, dims[m]);
            if (m < M)
                logB.reserve(m, dims[m]);
        }
        lmSymbols.reserve(dims[0]);

        logB.push_back(mgram_map::coord(), 0.0f); // dummy logB for backing off to zg
        logP.push_back(mgram_map::coord(), 0.0f); // zerogram score -- gets updated later

        std::vector<bool> skipWord; // true: skip entry containing this word
        skipWord.reserve(lmSymbols.capacity());

        // --- read main sections

        const double ln10xLMF = log(10.0);                // ARPA scores are strangely scaled
        msra::strfun::tokenizer tokens(" \t\n\r", M + 1); // used in tokenizing the input line
        for (int m = 1; m <= M; m++)
        {
            while (buf[0] == 0 && !feof(f))
                lineNo++, fgetline(f, buf);

            if (sscanf(buf, "\\%d-grams:", &n) != 1 || n != m)
                RuntimeError("read: mal-formed LM file, bad section header (%d): %ls", lineNo, pathname.c_str());
            lineNo++, fgetline(f, buf);

            std::vector<int> mgram(m + 1, -1);     // current mgram being read ([0]=dummy)
            std::vector<int> prevmgram(m + 1, -1); // cache to speed up symbol lookup
            mgram_map::cache_t mapCache;           // cache to speed up map.create()

            // read all the m-grams
            while (buf[0] != '\\' && !feof(f))
            {
                if (buf[0] == 0)
                {
                    lineNo++, fgetline(f, buf);
                    continue;
                }

                // -- parse the line
                tokens = &buf[0];
                if ((int) tokens.size() != ((m < fileM) ? m + 2 : m + 1))
                    RuntimeError("read: mal-formed LM file, incorrect number of tokens (%d): %ls", lineNo, pathname.c_str());
                double scoreVal = atof(tokens[0]);     // ... use sscanf() instead for error checking?
                double thisLogP = scoreVal * ln10xLMF; // convert to natural log

                bool skipEntry = false;
                for (int n = 1; n <= m; n++)
                {
                    const char *tok = tokens[n];
                    // map to id
                    int id;
                    if (m == 1) // unigram: build vocab table
                    {
                        id = (int) lmSymbols.size(); // unique id for this symbol
                        lmSymbols.push_back(SYMBOL(id, tok));
                        bool toSkip = false;
                        if (userSymMap.sym2existingId(lmSymbols.back().symbol) == -1)
                        {
                            if (filterVocabulary)
                                toSkip = true; // unknown word
                            else
                                userSymMap.sym2id(lmSymbols.back().symbol); // create it in user's space
                        }
                        skipWord.push_back(toSkip);
                    }
                    else // mgram: look up word in vocabulary
                    {
                        if (prevmgram[n] >= 0 && strcmp(idToSymbol(prevmgram[n]), tok) == 0)
                            id = prevmgram[n]; // optimization: most of the time, it's the same
                        else
                        {
                            id = symbolToId(tok);
                            if (id == -1)
                                RuntimeError("read: mal-formed LM file, m-gram contains unknown word (%d): %ls", lineNo, pathname.c_str());
                        }
                    }
                    mgram[n] = id;             // that's our id
                    skipEntry |= skipWord[id]; // skip entry if any token is unknown
                }

                double thisLogB = 0.0;
                if (m < M && !skipEntry)
                {
                    double boVal = atof(tokens[m + 1]); // ... use sscanf() instead for error checking?
                    thisLogB = boVal * ln10xLMF;        // convert to natural log
                }

                lineNo++, fgetline(f, buf);

                if (skipEntry) // word contained unknown vocabulary: skip entire entry
                    goto skipMGram;

                // -- enter the information into our data structure
                // Note that the mgram_map/mgram_data functions are highly efficient
                // because they can only be called in sorted order.

                // locate the corresponding entries
                {                                                   // (local block because we 'goto' over this)
                    mgram_map::key key(&mgram[1], m);               // key to locate this m-gram
                    mgram_map::coord c = map.create(key, mapCache); // create it & gets its location

                    // enter into data structure
                    logP.push_back(c, (float) thisLogP); // prob value
                    if (m < M)                           // back-off weight
                        logB.push_back(c, (float) thisLogB);
                }

            skipMGram:
                // remember current mgram for next iteration
                ::swap(mgram, prevmgram);
            }

            // fix the symbol set -- now we can binary-search in them with symbolToId()
            if (m == 1)
            {
                std::sort(lmSymbols.begin(), lmSymbols.end());
                idToSymIndex.resize(lmSymbols.size(), -1);
                for (int i = 0; i < (int) lmSymbols.size(); i++)
                {
                    idToSymIndex[lmSymbols[i].id] = i;
                }
            }

            fprintf(stderr, ", %d %d-grams", map.size(m), m);
        }
        fprintf(stderr, "\n");

        // check end tag
        if (M == fileM)
        { // only if caller did not restrict us to a lower order
            while (buf[0] == 0 && !feof(f))
                lineNo++, fgetline(f, buf);
            if (strcmp(buf, "\\end\\") != 0)
                RuntimeError("read: mal-formed LM file, no \\end\\ tag (%d): %ls", lineNo, pathname.c_str());
        }

        // update zerogram score by one appropriate for OOVs
        updateOOVScore();

        // establish mapping of word ids from user to LM space.
        // map's operator[] maps mgrams using this map.
        std::vector<int> userToLMSymMap(userSymMap.size());
        for (int i = 0; i < (int) userSymMap.size(); i++)
        {
            const char *sym = userSymMap.id2sym(i);
            int id = symbolToId(sym); // may be -1 if not found
            userToLMSymMap[i] = id;
        }
        map.created(userToLMSymMap);
    }

protected:
    // sort LM such that iterators will iterate in increasing order w.r.t. w2id[w]
    // This is achieved by replacing all internal ids by w2id[w].
    // This function is expensive: it makes a full temporary copy and involves sorting.
    // w2id[] gets destroyed by this function.
    void sort(std::vector<int> &w2id)
    {
        // create a full copy of logP and logB in the changed order
        mgram_map sortedMap(M);
        mgram_data<float> sortedLogP(M);
        mgram_data<float> sortedLogB(M - 1);

        for (int m = 1; m <= M; m++)
        {
            sortedMap.reserve(m, map.size(m));
            sortedLogP.reserve(m, logP.size(m));
            if (m < M)
                sortedLogB.reserve(m, logB.size(m));
        }

        // iterate in order of w2id
        // Order is determined by w2id[], i.e. entries with lower new id are
        // returned first.
        std::vector<int> mgram(M + 1, -1); // unmapped key in new id space
        mgram_map::cache_t createCache;
        for (mgram_map::reordering_iterator iter(map, w2id); iter; ++iter)
        {
            int m = iter.order();
            mgram_map::key key = *iter; // key in old 'w' space
            // keep track of an unmapped key in new id space
            if (m > 0)
            {
                int w = key.back();
                int newid = w2id[w]; // map to new id space
                mgram[m - 1] = newid;
            }
            for (int k = 0; k < m; k++)
                assert(mgram[k] == w2id[key[k]]);
            // insert new key into sortedMap
            mgram_map::coord c = sortedMap.create(mgram_map::unmapped_key(&mgram[0], m), createCache);
            // copy over logP and logB
            sortedLogP.push_back(c, logP[iter]);
            if (m < M)
                sortedLogB.push_back(c, logB[iter]);
        }

        // finalize sorted map
        sortedMap.created(w2id);

        // replace LM by sorted LM
        map.swap(sortedMap);
        logP.swap(sortedLogP);
        logB.swap(sortedLogB);
    }

public:
    // sort LM such that internal ids are in lexical order
    // After calling this function, iterators will iterate in lexical order,
    // and writing to an ARPA file creates a lexicographically sorted file.
    // Having sorted files is useful w.r.t. efficiency when iterating multiple
    // models in parallel, e.g. interpolating or otherwise merging models,
    // because then IIter can use the efficient deep_iterator (which iterates
    // in our internal order and therefore does not do any sorting) rather than
    // the reordering_iterator (which involves sort operations).
    template <class SYMMAP>
    void sort(const SYMMAP &userSymMap)
    {
        // deterine sort order
        // Note: This code copies all strings twice.
        std::vector<pair<std::string, int>> sortTemp(userSymMap.size()); // (string, w)
        foreach_index (w, sortTemp)
            sortTemp[w] = make_pair(userSymMap[w], w);
        std::sort(sortTemp.begin(), sortTemp.end());
        std::vector<int> w2id(userSymMap.size(), -1); // w -> its new id
        foreach_index (id, w2id)
            w2id[sortTemp[id].second] = id;

        // sort w.r.t. new id space
        sort(w2id);
    }

    // iterator to enumerate all known m-grams
    // This is used when creating whole models at once.
    template <class ITERATOR>
    class TIter : public ILM::IIter
    {
        int minM;               // minimum M we want to iterate (skip all below)
        const CMGramLM &lm;     // the underlying LM (for value())
        std::vector<int> wrank; // sorting criterion
        ITERATOR iter;          // the iterator used in this interface
        void findMinM()
        {
            while (iter && iter.order() < minM)
                ++iter;
        }

    public:
        // constructors
        TIter(const CMGramLM &lm, int minM, int maxM)
            : minM(minM), lm(lm), iter(lm.map, maxM)
        {
            findMinM();
        }
        TIter(const CMGramLM &lm, bool, int minM, int maxM)
            : minM(minM), lm(lm), wrank(lm.map.identical_map(lm.map.maxw() + 1)), iter(lm.map, wrank, maxM)
        {
            findMinM();
        }
        // has iterator not yet reached end?
        virtual operator bool() const
        {
            return iter;
        }
        // advance by one
        virtual void operator++()
        {
            ++iter;
            findMinM();
        }
        // current m-gram (mgram,m)
        virtual std::pair<const int *, int> operator*() const
        {
            mgram_map::key key = *iter;
            return std::make_pair(key.order() == 0 ? NULL : &key[0], key.order());
        }
        // current value (logP, logB)
        // No processing here--read out the logP/logB values directly from the data structure.
        virtual std::pair<double, double> value() const
        {
            if (iter.order() < lm.M)
                return std::make_pair(lm.logP[iter], lm.logB[iter]);
            else
                return std::make_pair(lm.logP[iter], 0.0);
        }
    };
    virtual IIter *iter(int minM, int maxM) const
    {
        if (maxM == INT_MAX)
            maxM = M; // default value
        // if no sorting needed, then we can use the efficient deep_iterator
        if (map.inorder())
            return new TIter<mgram_map::deep_iterator>(*this, minM, maxM);
        // sorting needed: use reordering_iterator
        return new TIter<mgram_map::reordering_iterator>(*this, true, minM, maxM);
    }

    virtual int order() const
    {
        return M;
    }
    virtual size_t size(int m) const
    {
        return (int) logP.size(m);
    }

protected:
    // computeSeenSums -- compute sum of seen m-grams, store at their history coord
    // If islog then P is logP, otherwise linear (non-log) P.
    template <class FLOATTYPE>
    static void computeSeenSums(const mgram_map &map, int M, const mgram_data<float> &P,
                                mgram_data<FLOATTYPE> &PSum, mgram_data<FLOATTYPE> &backoffPSum,
                                bool islog)
    {
        // dimension the accumulators and initialize them to 0
        PSum.init(M - 1);
        for (int m = 0; m <= M - 1; m++)
            PSum.assign(m, map.size(m), 0);

        backoffPSum.init(M - 1);
        for (int m = 0; m <= M - 1; m++)
            backoffPSum.assign(m, map.size(m), 0);

        // iterate over all seen m-grams
        msra::basetypes::fixed_vector<mgram_map::coord> histCoord(M); // index of history mgram
        for (mgram_map::deep_iterator iter(map, M); iter; ++iter)
        {
            int m = iter.order();
            if (m < M)
                histCoord[m] = iter;
            if (m == 0)
                continue;

            const mgram_map::key key = *iter;
            assert(m == key.order());

            float thisP = P[iter];
            if (islog)
            {
                if (thisP <= logzero)
                    continue; // pruned or otherwise lost
                thisP = exp(thisP);
            }
            else
            {
                if (thisP == 0.0f)
                    continue; // a pruned or otherwise lost m-gram
            }

            // parent entry
            const mgram_map::coord j = histCoord[m - 1]; // index of parent entry

            // accumulate prob in B field (temporarily misused)
            PSum[j] += thisP;

            // the mass of the back-off distribution covered by higher-order seen m-grams.
            // This must exist, as any sub-sequence of any seen m-mgram exists
            // due to the way we count the tokens.
            const mgram_map::key boKey = key.pop_h();
            const mgram_map::foundcoord c = map[boKey];
            if (!c.valid_w())
                RuntimeError("estimate: malformed data: back-off value not found"); // must exist
            // look it up
            float Pc = P[c];
            backoffPSum[j] += islog ? exp(Pc) : Pc;
        }
    }

    // computeBackoff -- compute back-off weights
    // Set up or update logB[] based on P[].
    // logB[] is an output from this function only.
    // If islog then P is logP, otherwise linear (non-log) P.
    static void computeBackoff(const mgram_map &map, int M,
                               const mgram_data<float> &P, mgram_data<float> &logB,
                               bool islog)
    {
        mgram_data<float> backoffPSum; // accumulator for the probability mass covered by seen m-grams

        // sum up probabilities of seen m-grams
        //  - we temporarily use the B field for the actual seen probs
        //  - and backoffSum for their prob pretending we are backing off
        computeSeenSums(map, M, P, logB, backoffPSum, islog);
        // That has dimensioned logB as we need it.

        // derive the back-off weight from it
        for (mgram_map::deep_iterator iter(map, M - 1); iter; ++iter)
        {
            double seenMass = logB[iter]; // B field misused: sum over all seen children
            if (seenMass > 1.0)
            {
                if (seenMass > 1.0001) // (a minor round-off error is acceptable)
                    fprintf(stderr, "estimate: seen mass > 1.0: %8.5f --oops??\n", seenMass);
                seenMass = 1.0; // oops?
            }

            // mass covered by seen m-grams is unused -> take out
            double coveredBackoffMass = backoffPSum[iter];
            if (coveredBackoffMass > 1.0)
            {
                if (coveredBackoffMass > 1.0001) // 1.0 for unigrams, sometimes flags this
                    fprintf(stderr, "estimate: unseen backoff mass < 0: %8.5f --oops??\n", 1.0 - coveredBackoffMass);
                coveredBackoffMass = 1.0; // oops?
            }

            // redistribute such that
            //      seenMass + bow * usedBackoffMass = 1
            //  ==> bow = (1 - seenMass) / usedBackoffMass
            double freeMass = 1.0 - seenMass;
            double accessibleBackoffMass = 1.0 - coveredBackoffMass; // sum of all backed-off items

            // back-off weight is just the free probability mass
            double bow = (accessibleBackoffMass > 0) ? freeMass / accessibleBackoffMass : 1.0;
            // A note on the curious choice of bow=1.0 for accessibleBackoffMass==0:
            // If accessibleBackoffMass==0, we are in undefined territory.
            // Because this means we never back off. Problem is that we have
            // already discounted the probabilities, i.e. there is probability
            // mass missing (distribution not normalized). Possibilities for
            // remedying the normalization issue are:
            //  1. use linear interpolation instead generally
            //  2. use linear interpolation only for such distributions
            //  3. push mass into <UNK> class if available
            //  4. ignore the normalization problem.
            // We choose 2. for the unigram distribution (enforced outside of this
            // function), and 4. for all other cases.
            // A second question arises for OOV words in this case. With OOVs,
            // accessibleBackoffMass should no longer be 0, but we don't know its
            // value. Be Poov the mass of all OOV words, then
            //  bow = (1 - seenMass) / Poov
            // Further, if seenMass was not discounted (as in our unigram case),
            // it computes to 1, but if we had accounted for Poov, it would
            // compute as (1-Poov) instead. Thus,
            //  bow = (1 - (1-Poov)) / Poov = 1
            // Realistically, this case happens for the unigram distribution.
            // Practically it means fallback instead of back-off for OOV words.
            // Also, practically, Poov is very small, so is the error.
            logB[iter] = logclip((float) bow);
        }
    }
};

// ===========================================================================
// CMGramLMIterator -- a special-purpose class that allows for direct iteration.
// ===========================================================================

class CMGramLMIterator : public msra::lm::mgram_map::iterator
{
    const CMGramLM &lm;

public:
    CMGramLMIterator(const CMGramLM &lm, mgram_map::coord c)
        : lm(lm), msra::lm::mgram_map::iterator(lm.map, c)
    {
    }
    float logP() const
    {
        return lm.logP[*this];
    }
    float logB() const
    {
        return lm.logB[*this];
    }
    float logB(mgram_map::coord c) const
    {
        return lm.logB[c];
    }
    msra::lm::mgram_map::coord locate(const int *mgram, int m) const
    {
        msra::lm::mgram_map::foundcoord c = lm.map[msra::lm::mgram_map::key(mgram, m)];
        if (!c.valid_w())
            LogicError("locate: attempting to locate a non-existing history");
        return c;
    }
};

#if 0
// ===========================================================================
// CMGramLMEstimator -- estimator for CMGramLM
// Implements Kneser-Ney discounting with Goodman/Chen modification, as well
// as Kneser-Ney back-off.
// ===========================================================================

class CMGramLMEstimator : public CMGramLM
{
    mgram_data<unsigned int> counts;    // [M+1][i] counts
    mgram_map::cache_t mapCache;        // used in map.create()
    std::vector<int> adaptBuffer;       // adapt() pushes data in here
    std::vector<int> adaptBufferHead;   // first M-1 tokens for cyclic closure
    std::vector<unsigned int> minObs;   // GangLi: prune each gram by obs occur

public:
    // calling sequence:
    //  - init()
    //  - push_back() for each count, in increasing order
    //  - estimate() -- heavy lifting happens here
    //  - writeARPA() to file
    // ... missing: forms of score-based pruning should happen here

    // construct
    void init (int p_M)
    {
        // dimensions
        M = p_M;
        map.init (M);
        logP.clear();
        logB.clear();
        counts.init (M);

        if ((int) minObs.size() != M)
        {// first time initial
            minObs.resize(M, 0);
            if (M > 2) minObs[2] = 2; // GangLi: prune trigram if Obs < 2, this is default value
            fprintf (stderr, "Set miniObs to 0 0 2.\n");
        }
        else
        {
            fprintf (stderr, "Not reset miniObs because it has already been set.\n");
        }

        for (int m = 1; m <= M; m++) counts.reserve (m, 1000000);   // something to start with
    }

    // set std::vector<unsigned int> minObs
    void setMinObs(const std::vector<unsigned int> & setMinObs)
    {
        if (minObs.size() != setMinObs.size())
            RuntimeError("In setMinObs: setMinObs size (%d) is not for %d-gram.", setMinObs.size(), minObs.size());
        minObs = setMinObs;
    }

    // call count() repeatedly to add counts, then call estimate() when done.
    // Include history counts. Probabilities are based on provided history
    // counts, rather than the sum of seen m-grams, to allow for count pruning.
    void push_back (const int * mgram, int m, unsigned int count)
    {
        if (m > M) RuntimeError("push_back: called with m-gram longer than M");
        // add to mgram_map & get location
        mgram_map::coord c = map.create (mgram_map::unmapped_key (mgram, m), mapCache);
        // save the count
        counts.push_back (c, count);
    };

protected:

    // add all tokens from adaptBuffer to counts[].
    // This is an expensive operation involving recreating the map, so use
    // this only for large chunks of data at a time.
    void merge()
    {
        // we need at least one M-gram
        int ntoks = (int) adaptBuffer.size() - (M -1);
        if (ntoks < 1)
            return;

        // create sorted set of counts and create merged counts
        mgram_map mmap (M);
        mgram_data<unsigned int> mcounts (M);
        mcounts.push_back (mgram_map::coord(), ntoks);  // zerogram count
        std::vector<int> keybuf (M+1);
        // do one order after another (to save memory)
        fprintf (stderr, "merge: adding %d tokens...", ntoks);
        for (int m = 1; m <= M; m++)
        {
            mgram_map::cache_t mmapCache;
            // enumerate all m-grams of this order
            std::vector<mgram_map::key> keys (ntoks);
            foreach_index (j, keys) keys[j] = mgram_map::key (&adaptBuffer[j], m);
            // sort them
            std::sort (keys.begin(), keys.end());
            // pre-allocate
            size_t alloc = counts.size (m);
            alloc++;                    // count first key
            for (int j = 1; j < ntoks; j++)
                if (keys[j] > keys[j-1])
                    alloc++;            // count unique keys
            mmap.reserve (m, alloc);    // worst case: no overlap
            mcounts.reserve (m, alloc);
            // merge with existing counts
            // Typical merge-sort operation with two iterators.
            mgram_map::deep_iterator iter (map, m);
            int i = 0;
            while (i < ntoks || iter)
            {
                if (iter && iter.m != m)
                {
                    ++iter;
                    continue;
                }
                if (iter)
                {
                    // regular case (neither has reached the end)
                    if (i < ntoks && iter)
                    {
                        // Note: *iter is a 'mapped' key, while create() expects an
                        // unmapped one. During training, both are identical.
                        mgram_map::unmapped_key oldkey = (mgram_map::unmapped_key) *iter;
                        if (oldkey < keys[i])   // key exists in old counts but not new
                        {
                            unsigned int count = counts[iter];
                            mcounts.push_back (mmap.create (oldkey, mmapCache), count);// store 'count' under 'key'
                            ++iter;             // watch out: this invalidates oldkey
                        }
                        else
                        {
                            // a range of new m-grams
                            mgram_map::unmapped_key newkey = keys[i];
                            unsigned int count = 1;
                            i++;
                            while (i < ntoks && newkey == keys[i])
                            {   // consume new tokens with the same key
                                count++;
                                i++;
                            }
                            if (oldkey == newkey)       // if old mgram matches then consume it
                            {
                                count += counts[iter];  // sum both up
                                ++iter;
                            }
                            mcounts.push_back (mmap.create (newkey, mmapCache), count);
                        }
                    }
                    else // if (i == ntoks && iter)
                    {   // final old counts
                        unsigned int count = counts[iter];
                        mgram_map::unmapped_key oldkey = (mgram_map::unmapped_key) *iter;
                        mcounts.push_back (mmap.create (oldkey, mmapCache), count);
                        ++iter;
                    }
                }
                else // if (i < ntoks && !iter)
                {   // final new counts
                    mgram_map::unmapped_key newkey = keys[i];
                    unsigned int count = 1;
                    i++;
                    while (i < ntoks && newkey == keys[i])
                    {   // consume new tokens with the same key
                        count++;
                        i++;
                    }
                    mcounts.push_back (mmap.create (newkey, mmapCache), count); // store 'count' under 'key'
                }
            }
            fprintf (stderr, " %d %d-grams", (int)mcounts.size (m), m);
        }

        // remove used up tokens from the buffer
        adaptBuffer.erase (adaptBuffer.begin(), adaptBuffer.begin() + ntoks);

        // Establish w->id mapping -- mapping is identical (w=id) during estimation.
        std::vector<int> w2id (mmap.maxid() +1);
        foreach_index (i, w2id) w2id[i] = i;
        //std::vector<int> w2id (mmap.identical_map());

        // close down creation of new tokens, so we can random-access
        mmap.created (w2id);

        // and swap
        map.swap (mmap);
        counts.swap (mcounts);

        fprintf (stderr, "\n");

        // destructor will delete previous counts and map (now in mcount/mmap)
    }

public:

    // training by pushing data in
    // special modes:
    //  - data=NULL -> reset; m=LM order from now on
    //  - data not NULL but m=0 -> taken as end indicator
    virtual void adapt (const int * data, size_t m)
    {
        // special call for reset
        if (data == NULL)
        {
            if (m == 0) RuntimeError("adapt: must pass LM order");
            init ((int) m);     // clear out current LM
            adaptBuffer.clear();
            adaptBufferHead.clear();
            return;
        }

        // special call to flush (incl. cyclic closure)
        if (m == 0)
        {
            // cyclicaly close the data set if final
            adaptBuffer.insert (adaptBuffer.end(), adaptBufferHead.begin(), adaptBufferHead.end());
            adaptBufferHead.clear();
            // merge the remaining tokens in
            merge();
            adaptBuffer.clear();    // the cyclically closed tokens remain->clear
            return;
        }

        // regular call: pushing word tokens in
        const size_t countChunkSize = 10000000;    // 10 million
        adaptBuffer.reserve (countChunkSize);

        // insert into our buffer
        adaptBuffer.insert (adaptBuffer.end(), data, data + m);

        // remember initial tokens for cyclic closure
        while (m > 0 && (int) adaptBufferHead.size() < M-1)
        {
            adaptBufferHead.push_back (*data);
            data++;
            m--;
        }

        // flush the buffer
        if (adaptBuffer.size() > countChunkSize)
            merge();
    }

#if 0 // debugging code -- rename adapt() above to adapt1()
    virtual void adapt (const int * data, size_t m)
    {
        while (m > 2)
        {
            adapt1 (data, 2);
            data += 2;
            m -= 2;
        }
        while (m > 0)
        {
            adapt1 (data, 1);
            data++;
            m--;
        }
    }
#endif

protected:
    // read one file
    // If dropId != -1 then do not create userSymMap but look up entries, and
    // use dropId for all unknown ones.
    template<class SYMMAP>
    int read (FILE * f, SYMMAP & userSymMap, int startId, int endId, int dropId)
    {
        const SYMMAP & constSymMap = userSymMap;

        // fgetline will check the line length, so enlarge the buf.size
        std::vector<char> buf (5000000);
        std::vector<int> ids;
        ids.reserve (buf.size() / 4);
        msra::strfun::tokenizer tokens (" \t", ids.capacity());
        int totalTokens = 0;    // for visual feedback
        while (!feof (f))
        {
            // fgetline will check the line length, so enlarge the buf.size
            tokens = fgetline (f, &buf[0], (int) buf.size());
            if (tokens.empty()) continue;
            ids.resize (0);
            ids.push_back (startId);
            foreach_index (i, tokens)
            {
                const char * p = tokens[i];
                int id = dropId == -1 ? userSymMap[p] : constSymMap[p];
                ids.push_back (id);

                if (totalTokens++ % 100000 == 0) fprintf (stderr, ".");
            }
            ids.push_back (endId);
            totalTokens += 2;
            adapt (&ids[0], ids.size());
        }
        return totalTokens;
    }

public:

    // 'read' here means read text.
    // filterVocabulary:
    //   false - no filter. The userSymMapis built (or augmented) by this function,
    //           incl. sentence boundary markers <s> and </s>
    //   true  - remove all words that are not in userSymMap. The userSymMap is
    //           not modified. If <UNK> is present, unknown words are mapped to
    //            it. Otherwise, m-grams involving OOV words are pruned.
    template<class SYMMAP>
    void read (const std::wstring & pathname, SYMMAP & userSymMap, bool filterVocabulary, int maxM)
    {
        if (!filterVocabulary)
        {   // create <s> and </s>
            userSymMap["<s>"];
            userSymMap["</s>"];
        }
        const SYMMAP & constSymMap = userSymMap;
        int startId = constSymMap["<s>"];   // or -1 -- but later enforce it is not
        int endId   = constSymMap["</s>"];  // or -1 -- dito.
        int unkId   = constSymMap["<UNK>"]; // or -1 -- -1 is OK

        if (startId == -1 || endId == -1)   // if filtering, these must be given
            RuntimeError("read: <s> and/or </s> missing in vocabulary");

        // if filtering but no <UNK>, we use (vocabsize) as the id, and have
        // estimate() prune it
        int dropId = filterVocabulary ? unkId != -1 ? unkId : userSymMap.size() : -1;

        if (filterVocabulary)
            RuntimeError ("CMGramLMEstimator::read() not tested for filterVocabulary==true");

        // reset adaptation
        adapt (NULL, maxM);         // pass dimension here

        // read all material and push into counts
        msra::basetypes::auto_file_ptr f = fopenOrDie (pathname, L"rbS");
        std::string tag = fgetline (f);
        if (tag == "#traintext")
        {
            read (f, userSymMap, startId, endId, dropId);
        }
        else if (tag == "#trainfiles")
        {
            while (!feof (f))
            {
                string thispath = fgetline (f);
                if (thispath.empty() || thispath[0] == '#') continue;   // comment
                msra::basetypes::auto_file_ptr thisf = fopenOrDie (thispath, "rbS");
                fprintf (stderr, "read: ingesting training text from %s ..", thispath.c_str());
                int numTokens = read (thisf, userSymMap, startId, endId, dropId);
                fprintf (stderr, "%d tokens\n", numTokens);
            }
        }
        else if (!tag.empty() && tag[0] == '#')
        {
            RuntimeError ("read: unknown tag '%s'", tag.c_str());
        }
        else    // no tag: just load the file directly
        {
            rewind (f);
            read (f, userSymMap, startId, endId, dropId);
        }

        // finalize
        adapt (&maxM, 0);

        // estimate
        vector<bool> dropWord (userSymMap.size(), false);
        dropWord.push_back (true);  // filtering but no <UNK>: 
        assert (!filterVocabulary || unkId != -1 || dropWord[dropId]);

        //std::vector<unsigned int> minObs (2, 0);
        //std::vector<unsigned int> iMinObs (3, 0);
        //iMinObs[1] = 3;  // remove singleton 2+-grams
        //iMinObs[2] = 3;  // remove singleton 3+-grams

        //// set prune value to 0 3 3
        //setMinObs (iMinObs);

        // TODO: Re-enable when MESSAGE definition is provided (printf?)
        // for (size_t i = 0; i < minObs.size(); i++)
        // {
        //     MESSAGE("minObs %d: %d.", i, minObs[i]);
        // }

        estimate (startId, minObs, dropWord);

#if 0 // write it out for debugging
        vector<string> syms (userSymMap.size());
        foreach_index (i, syms) syms[i] = userSymMap[i];
        auto_file_ptr outf = fopenOrDie ("d:/debug.lm", "wbS");
        write (outf, syms);
#endif
    }

protected:

    // reduce M
    void resize (int newM)
    {
        CMGramLM::resize (newM);
        counts.resize (newM);
    }

public:

    // -----------------------------------------------------------------------
    // estimate() -- estimate a back-off m-gram language model.
    // -----------------------------------------------------------------------
    //  - Kneser-Ney absolute discounting
    //  - Goodman-Shen count-specific discounting values
    //  - Kneser-Ney back-off
    // minObs is 0-based, i.e. minObs[0] is the cut-off for unigrams.
    void estimate (int startId, const std::vector<unsigned int> & minObs, vector<bool> dropWord)
    {
        if (!adaptBuffer.empty())
            RuntimeError("estimate: adaptation buffer not empty, call adapt(*,0) to flush buffer first");

        // Establish w->id mapping -- mapping is identical (w=id) during estimation.
        std::vector<int> w2id (map.maxid() +1);
        foreach_index (i, w2id) w2id[i] = i;
        //std::vector<int> w2id (map.identical_map());

        // close down creation of new tokens, so we can random-access
        map.created (w2id);

        // ensure M reflects the actual order of read data
        while (M > 0 && counts.size (M) == 0) resize (M-1);

        for (int m = 1; m <= M; m++)
            fprintf (stderr, "estimate: read %d %d-grams\n", (int)counts.size (m), m);

        // === Kneser-Ney smoothing
        // This is a strange algorithm.

#if 1 // Kneser-Ney back-off
        // It seems not to work for fourgram models (applied to the trigram).
        // But if it is only applied to bigram and unigram, there is a gain
        // from the fourgram. So we are not applying it to trigram and above.
        // ... TODO: use a constant to define the maximum KN count level,
        // and then do not allocate memory above that.
        mgram_data<unsigned int> KNCounts;      // [shifted m-gram] (*,v,w)
        mgram_data<unsigned int> KNTotalCounts; // [shifted, shortened m-gram] (*,v,*)
        if (M >= 2)
        {
            fprintf (stderr, "estimate: allocating Kneser-Ney counts...\n");

            KNCounts.init (M-1);
            for (int m = 0; m <= M-1; m++) KNCounts.assign (m, counts.size (m), 0);
            KNTotalCounts.init (M-2);
            for (int m = 0; m <= M-2; m++) KNTotalCounts.assign (m, counts.size (m), 0);

            fprintf (stderr, "estimate: computing Kneser-Ney counts...\n");

            // loop over all m-grams to determine KN counts
            for (mgram_map::deep_iterator iter (map); iter; ++iter)
            {
                const mgram_map::key key = *iter;
                if (key.order() < 2) continue;    // undefined for unigrams
                const mgram_map::key key_w = key.pop_h();
                const mgram_map::foundcoord c_w = map[key_w];
                if (!c_w.valid_w())
                    RuntimeError("estimate: invalid shortened KN m-gram");
                KNCounts[c_w]++;              // (u,v,w) -> count (*,v,w)
                const mgram_map::key key_h = key_w.pop_w();
                mgram_map::foundcoord c_h = map[key_h];
                if (!c_h.valid_w())
                    RuntimeError("estimate: invalid shortened KN history");
                KNTotalCounts[c_h]++;         // (u,v,w) -> count (*,v,w)
            }
        }
#else // regular back-off: just use regular counts instad
        mgram_data<unsigned int> & KNCounts = counts;
        mgram_data<unsigned int> & KNTotalCounts = counts;
        // not 'const' so we can later clear() them... this is only for testng anyway
#endif

        // === estimate "modified Kneser-Ney" discounting values
        // after Chen and Goodman: An empirical study of smoothing techniques for
        // language modeling, CUED TR-09-09 -- a rich resource about everything LM!

        std::vector<double> d1 (M+1, 0.0);
        std::vector<double> d2 (M+1, 0.0);
        std::vector<double> d3 (M+1, 0.0);
        fprintf (stderr, "estimate: discounting values:");

        {
            // actually estimate discounting values
            std::vector<int> n1 (M+1, 0);   // how many have count=1, 2, 3, 4
            std::vector<int> n2 (M+1, 0);
            std::vector<int> n3 (M+1, 0);
            std::vector<int> n4 (M+1, 0);

            for (mgram_map::deep_iterator iter (map); iter; ++iter)
            {
                int m = iter.order();
                if (m == 0) continue;   // skip the zerogram

                unsigned int count = counts[iter];

                // Kneser-Ney smoothing can also be done for back-off weight computation
                if (m < M && m < 3)      // for comments see where we estimate the discounted probabilities
                {   //    ^^ seems not to work for 4-grams...
                    const mgram_map::key key = *iter;   // needed to check for startId
                    assert (key.order() == m);

                    if (m < 2 || key.pop_w().back() != startId)
                    {
                        count = KNCounts[iter];
                        if (count == 0)  // must exist
                            RuntimeError("estimate: malformed data: back-off value not found (numerator)");
                    }
                }

                if (count == 1)      n1[m]++;
                else if (count == 2) n2[m]++;
                else if (count == 3) n3[m]++;
                else if (count == 4) n4[m]++;
            }

            for (int m = 1; m <= M; m++)
            {
                if (n1[m] == 0) RuntimeError(msra::strfun::strprintf ("estimate: error estimating discounting values: n1[%d] == 0", m));
                if (n2[m] == 0) RuntimeError(msra::strfun::strprintf ("estimate: error estimating discounting values: n2[%d] == 0", m));
                //if (n3[m] == 0) RuntimeError ("estimate: error estimating discounting values: n3[%d] == 0", m);
                double Y = n1[m] / (n1[m] + 2.0 * n2[m]);
                if (n3[m] ==0 || n4[m] == 0)
                {
                    fprintf (stderr, "estimate: n3[%d] or n4[%d] is 0, falling back to unmodified discounting\n", m, m);
                    d1[m] = Y;
                    d2[m] = Y;
                    d3[m] = Y;
                }
                else
                {
                    d1[m] = 1.0 - 2.0 * Y * n2[m] / n1[m];
                    d2[m] = 2.0 - 3.0 * Y * n3[m] / n2[m];
                    d3[m] = 3.0 - 4.0 * Y * n4[m] / n3[m];
                }
                // ... can these be negative??
                fprintf (stderr, " (%.3f, %.3f, %.3f)", d1[m], d2[m], d3[m]);
            }
            fprintf (stderr, "\n");
        }

        // === threshold against minimum counts (set counts to 0)
        // this is done to save memory, but it has no impact on the seen probabilities
        // ...well, it does, as pruned mass get pushed to back-off distribution... ugh!

        fprintf (stderr, "estimate: pruning against minimum counts...\n");

        // prune unigrams first (unigram cut-off can be higher than m-gram cut-offs,
        // as a means to decimate the vocabulary)

        unsigned int minUniObs = minObs[0];     // minimum unigram count
        int removedWords = 0;
        for (mgram_map::iterator iter (map, 1); iter; ++iter)
        {   // unigram pruning is special: may be higher than m-gram threshold
            if (counts[iter] >= minUniObs) continue;
            int wid = *iter;
            dropWord[wid] = true;       // will throw out all related m-grams
            removedWords++;
        }
        fprintf (stderr, "estimate: removing %d too rare vocabulary entries\n", removedWords);

        // now prune m-grams against count cut-off

        std::vector<int> numMGrams (M+1, 0);
        msra::basetypes::fixed_vector<mgram_map::coord> histCoord (M);  // index of history mgram
        for (int m = 1; m <= M; m++)
        {
            for (mgram_map::deep_iterator iter (map); iter; ++iter)
            {
                if (iter.order() != m) continue;
                bool prune = counts[iter] < minObs[m-1];    // prune if count below minimum
                // prune by vocabulary
                const mgram_map::key key = *iter;
                for (int k = 0; !prune && k < m; k++)
                {
                    int wid = key[k];
                    prune |= dropWord[wid];
                }
                if (prune)
                {
                    counts[iter] = 0;   // pruned: this is how we remember it
                    continue;
                }
                // for remaining words, check whether the structure is still intact
                if (m < M) histCoord[m] = iter;
                mgram_map::coord j = histCoord[m-1];    // parent
                if (counts[j] == 0)
                    RuntimeError ("estimate: invalid pruning: a parent m-gram got pruned away");
                    //RuntimeError("estimate: invalid pruning: a parent m-gram got pruned away");
                numMGrams[m]++;
            }
        }

        for (int m = 1; m <= M; m++)
        {
            fprintf (stderr, "estimate: %d-grams after pruning: %d out of %d (%.1f%%)\n", m,
                     numMGrams[m], (int)counts.size (m),
                     100.0 * numMGrams[m] / max (counts.size (m), size_t(1)));
        }

        // ensure M reflects the actual order of read data after pruning
        while (M > 0 && numMGrams[M] == 0) resize (M-1);    // will change M

        // === compact memory after pruning

        // naw... this is VERY tricky with the mgram_map architecture to keep all data in sync
        // So for now we just skip those in all subsequent steps (i.e. we don't save memory)

        // === estimate M-gram

        fprintf (stderr, "estimate: estimating probabilities...\n");

        // dimension the m-gram store
        mgram_data<float> P (M);            // [M+1][i] probabilities
        for (int m = 1; m <= M; m++) P.reserve (m, numMGrams[m]);

        // compute discounted probabilities (uninterpolated except, later, for unigram)

        // We estimate into a new map so that pruned items get removed.
        // For large data sets, where strong pruning is used, there is a net
        // memory gain from doing this (we gain if pruning cuts more than half).
        mgram_map Pmap (M);
        for (int m = 1; m <= M; m++) Pmap.reserve (m, numMGrams[m]);
        mgram_map::cache_t PmapCache;   // used in map.create()

        // m-grams
        P.push_back (mgram_map::coord(), 0.0f); // will be updated later
        for (int m = 1; m <= M; m++)
        {
            fprintf (stderr, "estimate: estimating %d %d-gram probabilities...\n", numMGrams[m], m);

            // loop over all m-grams of level 'm'
            msra::basetypes::fixed_vector<mgram_map::coord> histCoord (m);
            for (mgram_map::deep_iterator iter (map, m); iter; ++iter)
            {
                if (iter.order() != m)
                {
                    // a parent: remember how successors can find their history
                    // (files are nested like a tree)
                    histCoord[iter.order()] = iter;
                    continue;
                }

                const mgram_map::key key = *iter;
                assert (key.order() == iter.order());   // (remove this check once verified)

                // get history's count
                const mgram_map::coord j = histCoord[m-1];  // index of parent entry
                double histCount = counts[j];               // parent count --before pruning
                //double histCount = succCount[j];        // parent count --actuals after pruning

                // estimate probability for this M-gram
                unsigned int count = counts[iter];
                // this is 0 for pruned entries

                // count = numerator, histCount = denominator

                // Kneser-Ney smoothing --replace all but the highest-order
                // distribution with that strange Kneser-Ney smoothed distribution.
                if (m < M && m < 3 && count > 0)     // all non-pruned items except highest order
                {   //    ^^ seems not to work for 4-gram
                    // We use a normal distribution if the history is the sentence
                    // start, as there we fallback without back-off. [Thanks to
                    // Yining Chen for the tip.]
                    if (m < 2 || key.pop_w().back() != startId)
                    {
                        count = KNCounts[iter];             // (u,v,w) -> count (*,v,w)
                        if (count == 0)                     // must exist
                            RuntimeError ("estimate: malformed data: back-off value not found (numerator)");

                        const mgram_map::key key_h = key.pop_w();
                        mgram_map::foundcoord c_h = map[key_h];
                        if (!c_h.valid_w())
                            RuntimeError("estimate: invalid shortened KN history");
                        histCount = KNTotalCounts[c_h];     // (u,v,w) -> count (*,v,*)
                        if (histCount == 0)                 // must exist
                            RuntimeError ("estimate: malformed data: back-off value not found (denominator)");
                        assert (histCount >= count);
                    }
                }

                double dcount;
                double dP;

                // pruned case
                if (count == 0)         // this entry was pruned before
                    goto skippruned;

                // <s> does not count as an event, as it is never emitted.
                // For now we prune it, but later we put the unigram back with -99.0.
                if (key.back() == startId)
                {   // (u, v, <s>)
                    if (m > 1)          // do not generate m-grams
                        goto skippruned;
                    count = 0;          // unigram is kept in structure
                }
                else if (m == 1)
                {   // unigram non-<s> events
                    histCount--;        // do not count <s> in denominator either
                    // For non-unigrams, we don't need to care because m-gram
                    // histories of <s> always ends in </s>, and we never ask for such an m-gram
                    // ... TODO: actually, is subtracting 1 the right thing to do here?
                    // shouldn't we subtract the unigram count of <s> instead?
                }

                // Histories with any token before <s> are not valuable, and
                // actually need to be removed for consistency with the above
                // rule of removing m-grams predicting <s> (if we don't we may
                // create orphan m-grams).
                for (int k = 1; k < m-1; k++)
                {   // ^^ <s> at k=0 and k=m-1 is OK; anywhere else -> useless m-gram
                    if (key[k] == startId)
                        goto skippruned;
                }

                // estimate discounted probability
                dcount = count;                      // "modified Kneser-Ney" discounting
                if (count >= 3)      dcount -= d3[m];
                else if (count == 2) dcount -= d2[m];
                else if (count == 1) dcount -= d1[m];
                if (dcount < 0.0)   // 0.0 itself is caused by <s>
                    RuntimeError("estimate: negative discounted count value");

                if (histCount == 0)
                    RuntimeError ("estimate: unexpected 0 denominator");
                dP = dcount / histCount;
                // and this is the discounted probability value
                {
                    // Actually, 'key' uses a "mapped" word ids, while create()
                    // expects unmapped ones. However, we have established an
                    // identical mapping at the start of this function, such that
                    // we can be sure that key=unmapped key.
                    mgram_map::coord c = Pmap.create ((mgram_map::unmapped_key) key, PmapCache);
                    P.push_back (c, (float) dP);
                }

skippruned:;    // m-gram was pruned
            }
        }
        // the distributions are not normalized --discount mass is missing
        fprintf (stderr, "estimate: freeing memory for counts...\n");
        KNCounts.clear();       // free some memory
        KNTotalCounts.clear();

        // the only items used below are P and Pmap.
        w2id.resize (Pmap.maxid() +1);
        foreach_index (i, w2id) w2id[i] = i;
        //std::vector<int> w2id (Pmap.identical_map());
        Pmap.created (w2id);    // finalize and establish mapping for read access
        map.swap (Pmap);        // install the new map in our m-gram
        Pmap.clear();           // no longer using the old one

        counts.clear();         // counts also no longer needed

        // zerogram
        int vocabSize = 0;
        for (mgram_map::iterator iter (map, 1); iter; ++iter)
            if (P[iter] > 0.0)  // (note: this excludes <s> and all pruned items)
                vocabSize++;
        P[mgram_map::coord()] = (float) (1.0 / vocabSize);  // zerogram probability

        // interpolating the unigram with the zerogram
        // This is necessary as there is no back-off path from the unigram
        // except in the OOV case. I.e. probability mass that was discounted
        // from the unigrams is lost. We fix it by using linear interpolation
        // instead of strict discounting for the unigram distribution.
        double unigramSum = 0.0;
        for (mgram_map::iterator iter (map, 1); iter; ++iter)
            unigramSum += P[iter];
        double missingUnigramMass = 1.0 - unigramSum;
        if (missingUnigramMass > 0.0)
        {
            float missingUnigramProb = (float) (missingUnigramMass * P[mgram_map::coord()]);
            fprintf (stderr, "estimate: distributing missing unigram mass of %.2f to %d unigrams\n",
                     missingUnigramMass, vocabSize);
            for (mgram_map::iterator iter (map, 1); iter; ++iter)
            {
                if (P[iter] == 0.0f) continue;      // pruned
                P[iter] += missingUnigramProb;      // add it in
            }
        }

        // --- M-gram sections --back-off weights

        fprintf (stderr, "estimate: determining back-off weights...\n");
        computeBackoff (map, M, P, logB, false);
        // now the LM is normalized assuming the ARPA back-off computation

        // --- take logs and push estimated values into base CMGramLM structure

        // take logs in place
        for (int m = 0; m <= M; m++)
            for (mgram_map::iterator iter (map, m); iter; ++iter)
                P[iter] = logclip (P[iter]);    // pruned entries go to logzero
        P.swap (logP);      // swap into base language model

        // --- final housekeeping to account for idiosyncrasies of the ARPA format

        // resurrect sentence-start symbol with log score -99
        const mgram_map::foundcoord cs = map[mgram_map::key (&startId, 1)];
        if (cs.valid_w())
            logP[cs] = -99.0f * log (10.0f);

        // update zerogram prob
        // The zerogram will only be used in the OOV case--the non-OOV case has
        // been accounted for above by interpolating with the unigram. Thus, we
        // replace the zerogram by a value appropriate for an OOV word. We
        // choose the minimum unigram prob. This value is not stored in the ARPA
        // file, but instead recomputed when loading it. We also reset the
        // corresponding back-off weight to 1.0 such that we actually get the
        // desired OOV score.
        updateOOVScore();

        fprintf (stderr, "estimate: done");
        for (int m = 1; m <= M; m++) fprintf (stderr, ", %d %d-grams", (int)logP.size (m), m);
        fprintf (stderr, "\n");
    }
};
#endif

#if 0
// ===========================================================================
// CMGramLMClone -- create CMGramLM from sub-LMs through ILM and ILM::IIter
//  - create in memory into a CMGramLM
//  - write to ARPA file (static function)
// ===========================================================================

class CMGramLMClone : public CMGramLM
{
public:
    // create an LM in memory iterating through an underlying model
    // This uses IILM::IIter and the score() function, i.e. it works for all
    // derivative LM types such as linear interpolation.
    // Back-off weights are recomputed in this function. I.e. even if applied
    // to a plain m-gram, results may be different if tricks were played with
    // the back-off weights in the original model.
    // The dropWord[] vector, if not empty, specifies a set of words that
    // should be dropped (m-grams that contain such a word are skipped).
    // Underlying models are assumed to have m-gram property, otherwise the
    // resulting LM will explode.
    void clone (const ILM & lm, int p_M = INT_MAX, const vector<bool> & dropWord = vector<bool>())
    {
        if (p_M > lm.order())
            p_M = lm.order();
        M = p_M;
        map.init (M);
        logP.init (M);
        logB.init (M-1);

        // allocate the memory
        for (int m = 0; m <= M; m++)
        {
            size_t size_m = lm.size (m);
            map.reserve (m, size_m);
            logP.reserve (m, size_m);
            if (m < M)
                logB.reserve (m, size_m);
        }

        // compute the scores
        // Iterator will iterate in increasing order of word ids as returned
        // by *iter.
        bool filterWords = !dropWord.empty();
        mgram_map::cache_t mapCache;
        auto_ptr<IIter> piter (lm.iter (0, M));
        for (IIter & iter = *piter; iter; ++iter)
        {
            // get key (mgram[], m) for current iter position
            std::pair<const int*,int> keyp = *iter;
            const int * mgram = keyp.first;
            int m = keyp.second;
            mgram_map::unmapped_key key (mgram, m);
            // skip if we filter against a dropWord[] list
            if (filterWords)
            {
                // if any of the dropWord[mgram[]] is set then skip
                for (int i = 0; i < key.order(); i++)
                {
                    int w = key[i];
                    if (dropWord[w])
                        goto skipMGram;//skipMGram
                }
            }
            // local block for get rid of: warning C4533: initialization of 'c' is skipped by 'goto skipMGram'
            // (local block because we 'goto' over this)
            {
                // create map entry
                mgram_map::coord c = map.create (key, mapCache);
                // create probability entry
                double thisLogP = lm.score (mgram, m);
                logP.push_back (c, (float) thisLogP);
            }
skipMGram:
            filterWords = filterWords;
        }
        // finalize map and establish w->id mapping (identical)
        std::vector<int> w2id (map.identical_map());
        map.created (w2id);

        // create back-off data
        computeBackoff (map, M, logP, logB, true);

        // and replace zerogram score by the OOV score
        updateOOVScore();
    }

    // static function to clone a model and write it out as an ARPA (text) file.
    // Symbols can be anything that has symbols[w] -> std::string& .
    // A future version may do this more efficiently.
    template<class SYMMAP>
    static void write (const ILM & lm, int M, FILE * outf, const SYMMAP & symbols)
    {
        fprintf (stderr, "write: cloning...\n");
        CMGramLMClone outlm;
        outlm.clone (lm, M);
        fprintf (stderr, "write: saving...\n");
        ((const CMGramLM&) outlm).write (outf, symbols);
    }

    // read and parse a #clone file
    static void read (const wstring & clonepath, wstring & lmpath)
    {
        wstring dir, file;
        splitpath (clonepath, dir, file);   // we allow relative paths in the file

        msra::basetypes::auto_file_ptr f(fopenOrDie (clonepath, L"rbS"));
        std::string line = fgetline (f);
        if (line != "#clone")
            RuntimeError("read: invalid header line " + line);
        std::string lmpath8 = fgetline (f); // only one item: the pathname
        if (lmpath8.empty())
            RuntimeError("read: pathname missing");
        lmpath = msra::strfun::utf16 (lmpath8);
    }
};
#endif

#if 0
// ===========================================================================
// CPerplexity -- helper to measure perplexity
// ===========================================================================

class CPerplexity
{
    double logPAcc;     // accumulated logP
    int numTokensAcc;   // tokens accumulated
    int numOOVTokens;   // OOVs have been skipped
    int numUtterances;
    const ILM & lm;
    int startId, endId;
    std::vector<int> buf;   // temp buffer to insert <s> and  </s>
    CPerplexity & operator= (const CPerplexity &);  // inaccessible
public:
    CPerplexity (const ILM & p_lm, int p_startId, int p_endId) : lm (p_lm), startId (p_startId), endId (p_endId)
    { buf.reserve (1000); reset(); }

    // reset perplexity accumulation (clear all that's been passed)
    void reset() { logPAcc = 0.0; numTokensAcc = numOOVTokens = numUtterances = 0; }

    // Add perplexity for an utterance. Ids are in the same numeric id
    // space as was used to read() the language model. Only the actual words
    // should be included in ids[], do not include sentence start/end markers.
    // These are implied by this function.
    template<class SYMMAP>
    void addUtterance (const std::vector<int> & ids, const SYMMAP & symMap)
    {
        buf.assign (1, startId);
        buf.insert (buf.end(), ids.begin(), ids.end());
        buf.push_back (endId);
        for (int i = 1; i < (int) buf.size(); i++)
        {
            if (lm.oov (buf[i]))    // silently skip words unknown to the LM
            {
                numOOVTokens++;
                continue;
            }
            double logP = lm.score (&buf[0], i +1); // use full history
            if (logP <= -1e20)
            {
#if 0 // should really not happen
                fprintf (stderr, "skipping poor-scoring %s (%.2f)\n", symMap[buf[i]], logP);
#endif
                numOOVTokens++;
                continue;
            }
#if 0 // analysis of back-off etc.
            // dump some interesting information
            int mseenhist = lm.getLastLongestHistoryFound();
            int mseen = lm.getLastLongestMGramFound();
            int order = lm.order();
            if (order > i+1)    // limit to what we've requested
                order = i+1;
            char pbuf[20];
            sprintf (pbuf, "%7.5f", exp (logP));
            for (int k = 2; pbuf[k]; k++) if (pbuf[k] == '0') pbuf[k] = '.'; else break;
            char smseenhist[20];  // fallback=length of actual history
            smseenhist[order-1] = 0;
            for (int k = 0; k < order -1; k++) smseenhist[k] = (k >= order-1 - mseenhist) ? '.' : 'X';
            char smseen[20];
            smseen[order] = 0;
            for (int k = 0; k < order; k++) smseen[k] = (k >= order - mseen) ? '.' : 'X';
            char seq[100] = { 0 };
            for (int i1 = i - (order-1); i1 <= i; i1++)
            {
                strcat (seq, "_");
                strcat (seq, symMap[buf[i1]]);
            }
            fprintf (stderr, "=%-22s\t%6.2f\t%s\t%s %s\n", seq+1, logP, pbuf +1, smseenhist, smseen);
#else
            symMap;
#endif
#if 0 // testing of optimization
            double logP1 = lm.score_unoptimized (&buf[0], i +1); // use full history
            if (fabs (logP - logP1) > 1e-3)
                RuntimeError ("bug in optimized score()");
#endif
            logPAcc += logP;
            numTokensAcc++;
        }
        numUtterances++;
    }

    // return perplexity of words accumulated so far
    double getPerplexity() const
    {
        double avLogP = logPAcc / max (numTokensAcc, 1);
        double PPL = exp (-avLogP);
        return PPL;
    }

    // return number of words passed in, including OOV tokens (but not implied sentence ends)
    int getNumWords() const { return numTokensAcc + numOOVTokens - numUtterances; }

    // return number of OOV tokens
    int getNumOOV() const { return numOOVTokens; }

    // return number of utterances
    int getNumUtterances() const { return numUtterances; }
};
#endif
};
}; // namespace
