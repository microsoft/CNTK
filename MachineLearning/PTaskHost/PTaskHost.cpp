
#include "stdafx.h"
#include "PTaskHost.h"

#ifdef USE_PTASK

PTASKHOST_API void __stdcall
HostTask(LPDEPENDENTCONTEXT depContext)
{
    printf("Host task called...\n");

    /*
    assert(depContext->nArguments == 4 || depContext->nArguments == 5); // some versions pass a sync variable too
    assert(depContext->ppArguments != NULL);
                
    if(depContext->pbIsDependentBinding[0]) {

        void * pvStream = depContext->pStreams[0];
        cudaStream_t hStream = reinterpret_cast<cudaStream_t>(pvStream);
        size_t device = reinterpret_cast<size_t>(depContext->pDependentDevices[0]);
        onstream override(hStream);
        onptaskdevice overridedev(device);

        // in this case the depContext->ppArguments[*]  are device pointers
        GEMM_PARAMS * pParams = (GEMM_PARAMS*) depContext->ppArguments[depContext->nArguments-1];
        MARKRANGEENTER(L"SGemm-A,B,C ctors");
        cudamatrixops A(pParams->ADim.rows, pParams->ADim.cols, pParams->ADim.colstride, (float *)depContext->ppArguments[0], "GEMM-A");
        cudamatrixops B(pParams->BDim.rows, pParams->BDim.cols, pParams->BDim.colstride, (float *)depContext->ppArguments[1], "GEMM-B");
        cudamatrixops C(pParams->CDim.rows, pParams->CDim.cols, pParams->CDim.colstride, (float *)depContext->ppArguments[2]);
        MARKRANGEEXIT();
        if (pParams->scale < 1e-12 && pParams->scale != 0.0f) {
            fprintf(stderr, "!!Scale small, but not equal to zero in GEMM!!\n");
            pParams->scale = 0.0f;
        }
        else
        {
            DEPMATRIX3NANCHECK(depContext, SGemm, A, B, C);
        }
        MARKRANGEENTER(L"gemm");
        C.gemm(pParams->scale, A, pParams->Aistransposed, B, pParams->Bistransposed, pParams->ABweight);
        MARKRANGEEXIT();
        // TODO: check that ABweight and thisscale can be parameters to the function and don't need to be available when the function actually runs on GPU
        //cublasDestroy(handle);
    } else {
        // in this case the depContext->ppArguments[*]  are host pointers                 
        // call host version if it exists
        assert(false);  // no native version
    }
    */
}

#else
#include <stdio.h>
PTASKHOST_API void __stdcall
DummyTask()
{
    printf("This should never be called.\n");
}
#endif
