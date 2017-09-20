#pragma once

#ifdef USE_MKL

#include "mkl_dnn.h"

namespace Microsoft { namespace MSR { namespace CNTK {

static void CHECK_MKL(dnnError_t err)
{
    if (err != E_SUCCESS)
        RuntimeError("mkl err (%d)\n", err);
}


// adapter that converts data between user layout and primitive required layout
class MKLDnnResourceAdapter
{
    dnnLayout_t userLayout = nullptr;
    dnnLayout_t primLayout = nullptr;
    dnnPrimitive_t convertPrim = nullptr;
    bool isInput;
    void* tempBuffer = nullptr;
    dnnResourceType_t resourceType;
public:
    void Create(dnnLayout_t ltUser, dnnLayout_t ltPrim, dnnResourceType_t rt, bool userToPrim)
    {
        Clear();
        convertPrim = nullptr;
        tempBuffer = nullptr;
        isInput = userToPrim;
        resourceType = rt;
        if (!dnnLayoutCompare_F32(ltUser, ltPrim))
        {
            userLayout = ltUser;
            primLayout = ltPrim;
            dnnLayout_t from = userToPrim ? ltUser : ltPrim;
            dnnLayout_t to = userToPrim ? ltPrim : ltUser;
            CHECK_MKL(dnnConversionCreate_F32(&convertPrim, from, to));
            CHECK_MKL(dnnAllocateBuffer_F32(&tempBuffer, ltPrim)); // always allocate temp buffer for primLayout
        }
    }

    void PrepareForExecution(void* userData, void* resources[dnnResourceNumber])
    {
        if (isInput)
        {
            if (convertPrim)
            {
                CHECK_MKL(dnnConversionExecute_F32(convertPrim, userData, tempBuffer));
                resources[resourceType] = tempBuffer;
            }
            else
                resources[resourceType] = userData;
        }
        else
        {
            resources[resourceType] = convertPrim ? tempBuffer : userData;
        }
    }

    void ConvertOutput(void* userData)
    {
        if (isInput)
            RuntimeError("Cannot execute output ResourceAdapter for input");

        if (convertPrim)
            CHECK_MKL(dnnConversionExecute_F32(convertPrim, tempBuffer, userData));
    }

    void Clear()
    {
        if (convertPrim) { dnnDelete_F32(convertPrim); convertPrim = nullptr; }
        if (userLayout) { dnnLayoutDelete_F32(userLayout); userLayout = nullptr; }
        if (primLayout) { dnnLayoutDelete_F32(primLayout); primLayout = nullptr; }
        if (tempBuffer) { dnnReleaseBuffer_F32(tempBuffer); tempBuffer = nullptr; }
    }

    ~MKLDnnResourceAdapter()
    {
        Clear();
    }
};

} } }

#endif