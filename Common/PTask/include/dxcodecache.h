//--------------------------------------------------------------------------------------
// File: dxcodecache.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _DX_CODE_CACHE_H_
#define _DX_CODE_CACHE_H_

#include "primitive_types.h"
#include "ptdxhdr.h"
#include <map>

namespace PTask {

    class DXCodeCache {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        DXCodeCache();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~DXCodeCache();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Cache get. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="szFile">   [in,out] If non-null, the file. </param>
        /// <param name="szFunc">   [in,out] If non-null, the func. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        ID3D11ComputeShader*	CacheGet(char * szFile, char * szFunc);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Cache put. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="szFile">   [in,out] If non-null, the file. </param>
        /// <param name="szFunc">   [in,out] If non-null, the func. </param>
        /// <param name="p">        [in,out] If non-null, the p. </param>
        ///-------------------------------------------------------------------------------------------------

        void					CachePut(char * szFile, char * szFunc, ID3D11ComputeShader* p);
    protected:
        struct ltstr {
            bool operator()(std::string s1, std::string s2) const {
                return strcmp(s1.c_str(), s2.c_str()) < 0;
            }
        };
        std::map<std::string, ID3D11ComputeShader*, ltstr> m_cache;
    };

};

#endif