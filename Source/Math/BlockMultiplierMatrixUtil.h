//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full licence information.
//
#pragma once
#define NOMINMAX
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <string.h>//for memset
#include "BlockMultiplierPlatform.h"


namespace Microsoft { namespace MSR { namespace CNTK {

    template<typename ScalarT> void DumpMatrix(ScalarT* pDumpMe, int rows, int cols, std::ostream* pStream, int rowMax = std::numeric_limits<int>::max(),
                                               int colMax = std::numeric_limits<int>::max())
    {
        for (int r = 0; r < std::min(rows, rowMax); ++r)
        {
            for (int c = 0; c < std::min(cols, colMax); ++c)
            {
                (*pStream) << pDumpMe[r * cols + c] << " ";
            }
            (*pStream) << std::endl;
        }
    }

    // Turn a row+col into an absolute offset
    FORCEINLINE int RowColToOffset(int idxRow, int idxCol, int numCols)
    {
        return idxRow * numCols + idxCol;
    }

    template<typename ScalarT>struct TransposeArgs
    {
        int r;
        ScalarT* transposeMe;
        ScalarT* transposed;
        int origRows;
        int origCols;
    };


    template<class ScalarT>void TransposeThread(TransposeArgs<ScalarT> ta)
    {
        for (int c = 0; c < ta.origCols; ++c)
        {
            //new c,r = old r,c
            int oldOffset = RowColToOffset(ta.r, c, ta.origCols);
            int newOffset = RowColToOffset(c, ta.r, ta.origRows);
            ta.transposed[newOffset] = ta.transposeMe[oldOffset];
        }
    }

    template<typename ScalarT> class TransposeThreadType
    {
        public:
            void operator()(TransposeArgs<ScalarT> ta)
            {
                TransposeThread<ScalarT>(ta);
            }
    };


    template<class ScalarT> void Transpose(ScalarT* transposeMe, ScalarT* transposed, int origRows, int origCols)
    {
#pragma omp parallel for
        for (int r = 0; r < origRows; ++r)
        {
            for (int c = 0; c < origCols; ++c)
            {
                int oldOffset = RowColToOffset(r, c, origCols);
                int newOffset = RowColToOffset(c, r, origRows);
                transposed[newOffset] = transposeMe[oldOffset];
            }
        }
    }

    template<typename ScalarT> ScalarT* CreateAlignedMatrix(int m, int n, ScalarT initVal, int alignment = 64)
    {
        ScalarT* ret = (ScalarT*)ALIGNED_ALLOC(sizeof(ScalarT) * (m * n), alignment);


        if (initVal != 0)
        {
            for (int i = 0; i < m * n; ++i)
            {
                ret[i] = initVal;// +i;
            }
        }
        else
        {
            memset(ret, 0, sizeof(ScalarT) * m * n);
        }

        return ret;
    }

    template<typename ScalarT> void FreeAlignedMatrix(ScalarT* destroyMe)
    {
        ALIGNED_FREE(destroyMe);
    }

    template<typename ScalarT> double MeanSquaredError(ScalarT* lhs, ScalarT* rhs, int m, int n)
    {
        double accumulatedError = 0.0;
        for (int r = 0; r < m; ++r)
        {
            for(int c = 0; c < n; ++c)
            {
                double err = ((double)lhs[RowColToOffset(r, c, n)] - (double)rhs[RowColToOffset(r, c, n)]);
                err = err * err;
                accumulatedError += err;
            }
        }
        return accumulatedError / (double)(m * n);
    }


    template<typename ScalarT> void RandInitIntMatrix(ScalarT* initMe, int m, int n, ScalarT bound)
    {
        ScalarT* curr = initMe;
        for (int i = 0; i < m * n; ++i)
        {
            *curr++ = rand() % bound;
        }
    }

    //Helper fn for tests
    template<typename ScalarT>static void RandInitFloatMatrix(ScalarT* initMe, int m, int n, ScalarT min, ScalarT max)
    {
        for (int i = 0; i < m * n; ++i)
        {
            initMe[i] = min + ((max - min) * ((ScalarT)rand() / RAND_MAX));
        }
    }

    //Viewing matrices and troubleshooting is a lot easier in Octave.
    //Utility fn for exporting to Octave format
    template<typename ScalarT>void DumpMatrixToOctaveFormat(const ScalarT* dumpMe, int rows, int cols, const char* fileName, const char* id)
    {
        std::ofstream ofs(fileName);
        ofs << "# Created by gemmbenchmark" << std::endl <<
            "# name: " << id << std::endl <<
            "# type: matrix" << std::endl <<
            "# rows: " << rows << std::endl <<
            "# columns: " << cols << std::endl;

        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                ofs << ' ' << (ScalarT)(dumpMe[(cols * r) + c]);
            }
            ofs << std::endl;
        }
    }

}}} //End namespaces
