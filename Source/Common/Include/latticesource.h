#pragma once

#include <vector>
#include <memory>
#include "latticearchive.h"

namespace msra { namespace dbn {

// ---------------------------------------------------------------------------
// latticesource -- manages loading of lattices for MMI (in pairs for numer and denom)
// ---------------------------------------------------------------------------

class latticepair : public std::pair<msra::lattices::lattice, msra::lattices::lattice>
{
public:
    // NOTE: we don't check numerator lattice now
    size_t getnumframes() const
    {
        return second.getnumframes();
    }
    size_t getnumnodes() const
    {
        return second.getnumnodes();
    }
    size_t getnumedges() const
    {
        return second.getnumedges();
    }
    std::wstring getkey() const
    {
        return second.getkey();
    }
};

class latticesource
{
    const msra::lattices::archive numlattices, denlattices;
    int verbosity;

public:
    typedef msra::dbn::latticepair latticepair;
    latticesource(std::pair<std::vector<std::wstring>, std::vector<std::wstring>> latticetocs, const std::unordered_map<std::string, size_t>& modelsymmap, std::wstring RootPathInToc)
        : numlattices(latticetocs.first, modelsymmap, RootPathInToc), denlattices(latticetocs.second, modelsymmap, RootPathInToc), verbosity(0)
    {
    }

    bool empty() const
    {
#ifndef NONUMLATTICEMMI // TODO:set NUM lattice to null so as to save memory
        if (numlattices.empty() ^ denlattices.empty())
            RuntimeError("latticesource: numerator and denominator lattices must be either both empty or both not empty");
#endif
        return denlattices.empty();
    }

    bool haslattice(std::wstring key) const
    {
#ifdef NONUMLATTICEMMI
        return denlattices.haslattice(key);
#else
        return numlattices.haslattice(key) && denlattices.haslattice(key);
#endif
    }

    void getlattices(const std::wstring& key, std::shared_ptr<const latticepair>& L, size_t expectedframes, std::set<int>& specialwordids) const
    {
        std::shared_ptr<latticepair> LP(new latticepair);
        denlattices.getlattice(key, LP->second, specialwordids, expectedframes); // this loads the lattice from disk, using the existing L.second object
        L = LP;
    }

    void setverbosity(int veb)
    {
        verbosity = veb;
        numlattices.setverbosity(veb);
        denlattices.setverbosity(veb);
    }
};
} }
