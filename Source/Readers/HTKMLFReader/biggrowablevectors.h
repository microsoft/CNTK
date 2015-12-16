//
// <copyright file="biggrowablevectors.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// biggrowablevectors.h -- big growable vector that uses two layers and optionally a disk backing store for paging

#pragma once

namespace msra { namespace dbn {

// ---------------------------------------------------------------------------
// growablevectorbase -- helper for two-layer growable random-access array
// This allows both a fully allocated vector (with push_back()), e.g. for uids,
// as well as a partially allocated one (content managed by derived class), for features and lattice blocks.
// TODO:
//  - test this (make copy of binary first before full compilation; or rebuild the previous version)
//  - fully move in-mem range here, test again
//  - then we can move towards paging from archive directly (biggrowablevectorarray gets tossed)
// ---------------------------------------------------------------------------
template<class BLOCKTYPE> class growablevectorbase
{
protected:  // fix this later
    const size_t elementsperblock;
    size_t n;                                           // number of elements
    std::vector<std::unique_ptr<BLOCKTYPE>> blocks;     // the data blocks
    void operator= (const growablevectorbase &);        // (non-assignable)
    void check (size_t t) const { if (t >= n) LogicError("growablevectorbase: out of bounds"); }   // bounds check helper

    // resize intermediate level, but do not allocate blocks
    // (may deallocate if shrinking)
    void resize_without_commit (size_t T)
    {
        blocks.resize ((T + elementsperblock-1) / elementsperblock);
        n = T;
        // TODO: update allocated range
    }

    // commit memory
    // begin/end must be block boundaries
    void commit (size_t begin, size_t end, BLOCKTYPE * blockdata)
    {
        auto blockptr = getblock (begin, end);  // memory leak: if this fails (logic error; should never happen)
        blockptr.set (blockdata);               // take ownership of the block
        // TODO: update allocated range  --also enforce consecutiveness
    }

    // flush a block
    // begin/end must be block boundaries
    void flush (size_t begin, size_t end)
    {
        auto blockptr = getblock (begin, end);  // memory leak: if this fails (logic error; should never happen)
        blockptr.reset();                       // release it
        // TODO: update allocated range  --also enforce consecutiveness
    }

    // helper to get a block pointer, with block referenced as its entire range
    std::unique_ptr<BLOCKTYPE> & getblockptr (size_t t) // const
    {
        check (t);
        return blocks[t / elementsperblock];
    }

    // helper to get a block pointer, with block referenced as its entire range
    std::unique_ptr<BLOCKTYPE> & getblockptr (size_t begin, size_t end) const
    {
        // BUGBUG: last block may be shorter than elementsperblock
        if (end - begin != elementsperblock || getblockt (begin) != 0)
            LogicError("growablevectorbase: non-block boundaries passed to block-level function");
        return getblockptr (begin);
    }
public:
    growablevectorbase (size_t elementsperblock) : elementsperblock (elementsperblock), n (0) { blocks.reserve (1000); }
    size_t size() const { return n; }       // number of frames
    bool empty() const { return size() == 0; }

    // to access an element t -> getblock(t)[getblockt(t)]
    BLOCKTYPE & getblock (size_t t) const
    {
        check (t);
        const size_t blockid = t / elementsperblock;
        return *blocks[blockid].get();
    }

    size_t getblockt (size_t t) const
    {
        check (t);
        return t % elementsperblock;
    }
};

// ---------------------------------------------------------------------------
// biggrowablevector -- big vector we can push_back to
// ---------------------------------------------------------------------------
template<class ELEMTYPE> class biggrowablevector : public growablevectorbase<std::vector<ELEMTYPE>>
{
public:
    biggrowablevector() : growablevectorbase<std::vector<ELEMTYPE>>::growablevectorbase (65536) { }

    template<typename VALTYPE> void push_back (VALTYPE e)   // VALTYPE could be an rvalue reference
    {
        size_t i = this->size();
        this->resize_without_commit (i + 1);
        auto & block = this->getblockptr (i);
        if (block.get() == NULL)
            block.reset (new std::vector<ELEMTYPE> (this->elementsperblock));
        (*block)[this->getblockt (i)] = e;
    }

          ELEMTYPE & operator[] (size_t t)       { return this->getblock(t)[this->getblockt (t)]; }    // get an element
    const ELEMTYPE & operator[] (size_t t) const { return this->getblock(t)[this->getblockt (t)]; }    // get an element

    void resize (const size_t n)
    {
        this->resize_without_commit (n);
        foreach_index (i, this->blocks)
            if (this->blocks[i].get() == NULL)
                this->blocks[i].reset (new std::vector<ELEMTYPE> (this->elementsperblock));
    }
};

};};
