//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <stdexcept>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <algorithm>
#include <stdlib.h>

#include "Basics.h"
#include "Matrix.h"
#include "ComputationNode.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
struct MemRequestInfo
{
    DEVICEID_TYPE deviceId;                     // which device to allocate data 
    std::vector<shared_ptr<Matrix<ElemType>>*> pMatrixPtrs;    // memory pointers 
    size_t matrixSize;                          // memory size 
    bool mbScale;                               // whether the memory shall be scaled by minibatch size 
    bool isWorkSpace;                           // workspace memory or not, by workspace we indicate whether a memory space will be released very shortly after allocation 
    int allocStep;                              // at what step counter memory allocation is requested 
    int releaseStep;                            // at what step counter memory release is requested  
    int memoryId;                               // integer indexing the memory buffer ID 
    MemRequestInfo(DEVICEID_TYPE deviceId, shared_ptr<Matrix<ElemType>>*pMatrixPtr, size_t matrixSize, bool mbScale, bool isWorkSpace, int allocStep)
        :deviceId(deviceId), matrixSize(matrixSize), mbScale(mbScale), isWorkSpace(isWorkSpace), allocStep(allocStep), releaseStep(INT_MAX), memoryId(-1)
    {
        pMatrixPtrs.push_back(pMatrixPtr);
    }
    void SetReleaseStep(int step) { releaseStep = step; }
    void SetMemoryId(int id) { memoryId = id;  }
};

template <class ElemType>
struct greater_than_mem_req_size
{
    inline bool operator() (const MemRequestInfo<ElemType>& info1, const MemRequestInfo<ElemType>& info2)
    {
        return (info1.matrixSize > info2.matrixSize);
    }
};

struct MemAllocInfo
{
    int memoryId; 
    size_t memorySize; 
    vector<pair<int, int>> occupancy; 
    MemAllocInfo(int memoryId, size_t memorySize, vector<pair<int, int>> occ)
        :memoryId(memoryId), memorySize(memorySize), occupancy(occ)
    {
    }
};

// MatrixPool -- class to support memory sharing
// Despite the gather general name of this class, it is specifically designed to support the memory sharing of ComputationNodes.
// Note: see #define SUPRESS_MEMSHARING below as for how to temporarily disable memory sharing altogether, for debugging
class MatrixPool
{
public:
    typedef const void* AliasNodePtr; // use as an identifier in place of ComputationNodeBasePtr to avoid include order issue

protected:
    vector<MemRequestInfo<float>> m_memRequestInfoFloatVec; 
    vector<MemRequestInfo<double>> m_memRequestInfoDoubleVec;
    vector<MemRequestInfo<half>> m_memRequestInfoHalfVec;
    set<DEVICEID_TYPE> m_deviceIDSet; 
    int m_stepCounter; 

    template <class ElemType>
    vector<MemRequestInfo<ElemType>>& GetMemRequestInfoVec();

    // MatrixPool allows a bunch of node to share one matrix

    struct AliasInfo
    {
        void* pMatrixPtr;
        size_t totalCount;
        size_t releaseCount;

        AliasInfo(size_t total = 0)
            : pMatrixPtr(nullptr), totalCount(total), releaseCount(0)
        {
        }
    };
    unordered_map<AliasNodePtr, AliasInfo> m_aliasGroups;
    unordered_map<AliasNodePtr, AliasNodePtr> m_aliasLookup;

public:

    void Reset()
    {
        m_stepCounter = 0;
        m_aliasGroups.clear();
        m_aliasLookup.clear();
    };

    template <class ElemType>
    MemRequestInfo<ElemType>* GetMemInfo(shared_ptr<Matrix<ElemType>> *pMatrixPtr)
    {
        vector<MemRequestInfo<ElemType>>& memInfoVec = GetMemRequestInfoVec<ElemType>();
        // iterate through the vector and find the pointer memInfo
        for (auto& memInfo : memInfoVec)
        {
            if (memInfo.pMatrixPtrs[0] == pMatrixPtr)
                return &memInfo;
        }
        return nullptr;
    }

    template <class ElemType>
    void RequestRelease(shared_ptr<Matrix<ElemType>> *pMatrixPtr)
    {
        auto memInfo = GetMemInfo(pMatrixPtr);
        if (memInfo != nullptr)
        {
            memInfo->SetReleaseStep(m_stepCounter);
        }
        m_stepCounter++; 
    }

    // isWorkSpace is a flag indicating a memory is temporary and will be released very shortly. In the current implementation, all workspace
    // memories will have their own pool. This is a design proven to be useful for the workspace memory in convolution. 
    // matrixSize is an estimate of the required memory to be allocated. Note we don't allocate any memory at the time of request. Instead, a 
    // global memory allocation optimziation is run to improve memory efficiency 
    // mbScale is another flag indicating if the size of the memory will scale w.r.t. the minibatch size. Unfortunately, at the time of memory
    // request and pointer assignment, we don't known the minibatch size. Thus our memory sharing algorithm is sub-optimal. 
    template <class ElemType>
    void RequestAllocate(DEVICEID_TYPE deviceId, shared_ptr<Matrix<ElemType>>*pMatrixPtr, size_t matrixSize, bool mbScale, bool isWorkSpace)
    {
        vector<MemRequestInfo<ElemType>>& memInfoVec = GetMemRequestInfoVec<ElemType>(); 
        MemRequestInfo<ElemType> memInfo(deviceId, pMatrixPtr, matrixSize, mbScale, isWorkSpace, m_stepCounter);
        memInfoVec.push_back(memInfo); 
        m_deviceIDSet.insert(deviceId); 
        m_stepCounter++; 

        // assign some temporary pointer, they will be replaced later unless the matrix is sparse
        *pMatrixPtr = make_shared<Matrix<ElemType>>(deviceId);
    }

    void OptimizedMemoryAllocation()
    {
        // MatrixPool is not templated, so we call both float and double versions here 
        OptimizedMemoryAllocationFunc<float>(); 
        OptimizedMemoryAllocationFunc<double>();
        OptimizedMemoryAllocationFunc<half>();
        return; 
    }

    void SetAliasInfo(
        const unordered_map<AliasNodePtr, unordered_set<AliasNodePtr>>& groupMap,
        const unordered_map<AliasNodePtr, AliasNodePtr>& rootLookupMap)
    {
        m_aliasLookup.clear();
        for (const auto& pair : groupMap)
        {
            m_aliasGroups.insert(std::make_pair(pair.first, AliasInfo(pair.second.size())));

            for (const auto& child : pair.second)
            {
                if (rootLookupMap.find(child) == rootLookupMap.end())
                    InvalidArgument("group nodes should be in lookupMap");
            }
        }

        for (const auto& pair : rootLookupMap)
        {
            if (groupMap.find(pair.second) == groupMap.end())
                InvalidArgument("lookup root should be group key");
        }
        m_aliasLookup = rootLookupMap;
    }

    template <class ElemType>
    void RequestAliasedRelease(AliasNodePtr node)
    {
        const auto iter = m_aliasLookup.find(node);
        if (iter == m_aliasLookup.end())
            LogicError("node not aliased");

        auto parent = iter->second;
        auto& aliasInfo = m_aliasGroups[parent];
        if (aliasInfo.pMatrixPtr == nullptr)
            LogicError("double releasing aliased matrix, or releasing before any allocation for the matrix");

        if (aliasInfo.releaseCount >= aliasInfo.totalCount)
            LogicError("number of alias instances exceeded expectation");

        aliasInfo.releaseCount++;

        if (aliasInfo.releaseCount == aliasInfo.totalCount)
        {
            RequestRelease((shared_ptr<Matrix<ElemType>>*)aliasInfo.pMatrixPtr);
            aliasInfo.pMatrixPtr = nullptr;
        }
    }

    template <class ElemType>
    void RequestAliasedAllocate(DEVICEID_TYPE deviceId, AliasNodePtr node, shared_ptr<Matrix<ElemType>>*pMatrixPtr, size_t matrixSize, bool mbScale)
    {
        const auto iter = m_aliasLookup.find(node);
        if (iter == m_aliasLookup.end())
            LogicError("node not aliased");

        auto parent = iter->second;
        auto& aliasInfo = m_aliasGroups[parent];
        if (aliasInfo.pMatrixPtr == nullptr)
        {
            // first allocation for the group
            aliasInfo.pMatrixPtr = pMatrixPtr;
            RequestAllocate(deviceId, pMatrixPtr, matrixSize, mbScale, false);
        }
        else
        {
            auto aliasRootMatrixPtr = (shared_ptr<Matrix<ElemType>>*)aliasInfo.pMatrixPtr;
            *pMatrixPtr = *aliasRootMatrixPtr;
            GetMemInfo<ElemType>(aliasRootMatrixPtr)->pMatrixPtrs.push_back(pMatrixPtr);
        }
    }

private: 
    bool CheckOverlap(pair<int, int>occ, vector<pair<int, int>>&occVec)
    {
        bool bRet = false;
        for (auto& o : occVec)
        {
            if (occ.first <= o.second && occ.second >= o.first)
            {
                bRet = true;
                break;
            }
        }
//#define SUPRESS_MEMSHARING // #define this to disable memory sharing by always return true 
// TODO: Make this a runtime option.
#ifdef SUPRESS_MEMSHARING
        bRet = true; 
#endif
        return bRet;
    }

    template <class ElemType>
    void OptimizedMemoryAllocationFunc()
    {
        vector<MemRequestInfo<ElemType>>& memInfoVec = GetMemRequestInfoVec<ElemType>();
        if (memInfoVec.empty())
            return; 

        // remove all requests that has been marked as sparse matrices, those will not participate in memory sharing 
        for (auto iter = memInfoVec.begin(); iter != memInfoVec.end(); )
        {
            bool hasSparse = false;
            for (auto matPtr : iter->pMatrixPtrs)
            {
                if ((*matPtr)->GetMatrixType() == SPARSE)
                {
                    hasSparse = true;
                    break;
                }
            }

            if (hasSparse)
                memInfoVec.erase(iter);
            else
                iter++; 
        }

        // sort the memory request from largest size to smallest 
        std::sort(memInfoVec.begin(), memInfoVec.end(), greater_than_mem_req_size<ElemType>());

        std::vector<bool> workspaceFlagVec = {true, false};
        for (auto& devId : m_deviceIDSet)
        {
            for (auto wsFlag : workspaceFlagVec)   // we allocate the workspace memory pointers first, and they are not shared with the non-workspace memory requests
            {
                // memAllocInfoVec is a sorted list of memory allocations from smallest to largest in memory size 
                vector<MemAllocInfo> memAllocInfoVec;
                int memoryCounter = 0;
                // we start with memory request that is scalable with minibatch size(usually those require larger memory size)
                for (auto& memInfo : memInfoVec)
                {
                    // check if it's the proper device
                    if (memInfo.deviceId != devId || memInfo.isWorkSpace != wsFlag || !memInfo.mbScale)
                        continue;

                    if (!memAllocInfoVec.empty())
                    {
                        // since we assign from highest memory to lowest, every memory that has been allocated can accommodate the 
                        // current memory request, unless there is a conflict (overlap) 
                        auto iter = memAllocInfoVec.begin();
                        while (iter != memAllocInfoVec.end() && CheckOverlap(make_pair(memInfo.allocStep, memInfo.releaseStep), iter->occupancy))
                            iter++;
                        if (iter == memAllocInfoVec.end())
                        {
                            // no current memory can be assigned, need to create a new one 
                            vector<pair<int, int>> occ;
                            occ.push_back(make_pair(memInfo.allocStep, memInfo.releaseStep));
                            MemAllocInfo ma(memoryCounter, memInfo.matrixSize, occ);
                            // insert in the front of the vector to maintain sorted order 
                            memAllocInfoVec.insert(memAllocInfoVec.begin(), ma);
                            memInfo.SetMemoryId(memoryCounter);
                            memoryCounter++;
                        }
                        else
                        {
                            iter->occupancy.push_back(make_pair(memInfo.allocStep, memInfo.releaseStep));
                            memInfo.SetMemoryId(iter->memoryId);
                        }
                    }
                    else
                    {
                        vector<pair<int, int>> occ;
                        occ.push_back(make_pair(memInfo.allocStep, memInfo.releaseStep));
                        MemAllocInfo ma(memoryCounter, memInfo.matrixSize, occ);
                        memAllocInfoVec.push_back(ma);
                        memInfo.SetMemoryId(memoryCounter);
                        memoryCounter++;
                    }
                }

                // rescan the request list and this time allocate for those that doesn't depend on minibatch size 
                for (auto& memInfo : memInfoVec)
                {
                    // check if it's the proper device
                    if (memInfo.deviceId != devId || memInfo.isWorkSpace != wsFlag || memInfo.mbScale)
                        continue;

                    if (!memAllocInfoVec.empty())
                    {
                        // the memory allocation vector is sorted by size. We find the largest available buffer that doesn't have time overlap
                        auto workingAlloc = memAllocInfoVec.end();
                        for (auto iter = memAllocInfoVec.begin(); iter != memAllocInfoVec.end(); iter++)
                        {
                            if (!CheckOverlap(make_pair(memInfo.allocStep, memInfo.releaseStep), iter->occupancy))
                                workingAlloc = iter;
                        }
                        if (workingAlloc == memAllocInfoVec.end())  // nothing works 
                        {
                            vector<pair<int, int>> occ;
                            occ.push_back(make_pair(memInfo.allocStep, memInfo.releaseStep));
                            MemAllocInfo ma(memoryCounter, memInfo.matrixSize, occ);
                            memAllocInfoVec.push_back(ma);  // add as the last one 
                            memInfo.SetMemoryId(memoryCounter);
                            memoryCounter++;
                        }
                        else
                        {
                            workingAlloc->occupancy.push_back(make_pair(memInfo.allocStep, memInfo.releaseStep));
                            memInfo.SetMemoryId(workingAlloc->memoryId);
                        }
                    }
                    else
                    {
                        vector<pair<int, int>> occ;
                        occ.push_back(make_pair(memInfo.allocStep, memInfo.releaseStep));
                        MemAllocInfo ma(memoryCounter, memInfo.matrixSize, occ);
                        memAllocInfoVec.push_back(ma);
                        memInfo.SetMemoryId(memoryCounter);
                        memoryCounter++;
                    }
                }

                // now assign the actual pointers 
                for (int i = 0; i < memoryCounter; i++)
                {
                    auto matrixPtr = make_shared<Matrix<ElemType>>(devId);
                    if (!matrixPtr) // this can't really happen, because we haven't started allocating memory yet
                        LogicError("MatrixPool: failed to get a valid matrix.");
                    for (auto& memInfo : memInfoVec)
                    {
                        if (memInfo.deviceId == devId && memInfo.isWorkSpace == wsFlag && memInfo.memoryId == i)
                        {
                            for (auto pOutMatrixPtr : memInfo.pMatrixPtrs)
                            {
                                *pOutMatrixPtr = matrixPtr;
                            }
                        }
                    }
                }
            }
        }
    }
};

}}}
