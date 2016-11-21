//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// EvalMultithreads.cpp : Sample application shows how to evaluate a model in multiple threading environment. 
//
#include <iostream>
#include <stdio.h>
#include <string>

class CntkNetParser;

class BG_DataBlob
{
    friend class CntkNetParser;

public:
    BG_DataBlob(const void* data, size_t length, size_t length2, const std::wstring& name);

private:
    bool Reformat(const std::wstring &);

    const void* _data;
    size_t _length;
    size_t _length2;
    std::wstring _name;
};

class BG_Node
{
    friend class CntkNetParser;

public:
    BG_Node(const std::wstring& name);
    const std::wstring &Name() const { return name; }


private:
    bool SetAttribute(const std::wstring& name, const std::wstring& value);
    bool SetRank(size_t rank);
    bool SetDim(size_t dim, size_t value);
    bool AddTag(const std::wstring& value);
    bool SetOp(const std::wstring& op);

    std::map<std::wstring, std::wstring> attribs;
    std::vector<size_t> dims;
    std::vector<std::wstring> tags;

    std::wstring name;
    std::wstring op;
    BG_DataBlob *data;
};

class BG_Graph
{
    friend class CntkNetParser;

public:
    BG_Graph();

    BG_Node *LookupNode(const std::wstring& name);
    bool Serialize(FILE* fOut);

private:
    bool AddNode(BG_Node* node);
    bool AddArc(const std::wstring& to_nodeName, const std::wstring& from_childName);
    bool AddTag(const std::wstring& value);

    std::wstring BaseFilename(const std::wstring& filename);
    bool SetName(const std::wstring& name);
    bool SetVersion(const std::wstring& version);
    bool SetToolkitName(const std::wstring& name);
    bool SetToolkitVersion(const std::wstring& version);
    bool AddModelInfo(const std::wstring& name, const std::wstring& value);

    std::vector<BG_Node*> nodes;
    std::vector<std::wstring> tags;
    std::vector<std::tuple<std::wstring, std::wstring>> arcs;

    std::wstring graphName;
    std::wstring graphVersion;
    std::wstring toolkitName;
    std::wstring toolkitVersion;
    std::map<std::wstring, std::wstring> attribs;
};

// A class to parse a CNTK model file, and translate it into XML.
class CntkNetParser {
public:
    CntkNetParser();

    static const size_t CNTK_MAXIMUM_VERSION_SUPPORTED;
    static const size_t maxCntkStringLen;

    // Matrix format description (lifted from CommonMatrix.h)
    enum MatrixFlagBitPosition
    {
        bitPosRowMajor = 0,         // row major matrix
        bitPosSparse = 1,           // sparse matrix (COO if uncompressed)
        bitPosCompressed = 2,       // a compressed sparse format (CSC/CSR)
        bitPosDontOwnBuffer = 3,    // buffer is not owned by this matrix
        bitPosSetValueOnDevice = 4, // in a setValue situation, the copy from buffer is already on the device
    };

    enum MatrixFormat
    {
        matrixFormatDense = 0,                          // default is dense
        matrixFormatColMajor = 0,                       // default is column major
        matrixFormatRowMajor = 1 << bitPosRowMajor,     // row major matrix
        matrixFormatSparse = 1 << bitPosSparse,         // sparse matrix
        matrixFormatCompressed = 1 << bitPosCompressed, // a compressed sparse format (CSC/CSR/COO)
        matrixFormatDenseColMajor = matrixFormatDense + matrixFormatColMajor,
        matrixFormatDenseRowMajor = matrixFormatDense + matrixFormatRowMajor,
        matrixFormatSparseCSC = matrixFormatSparse + matrixFormatColMajor + matrixFormatCompressed,
        matrixFormatSparseCSR = matrixFormatSparse + matrixFormatRowMajor + matrixFormatCompressed,
        matrixFormatSparseOther = matrixFormatSparse + matrixFormatRowMajor,                   // currently used for CPU sparse format, will change to CSC/CSR eventually
        matrixFormatMask = matrixFormatRowMajor + matrixFormatSparse + matrixFormatCompressed, // mask that covers all the
        matrixFormatSparseBlockCol,                                                            // col block based sparse matrix
        matrixFormatSparseBlockRow,                                                            // row block based sparse matrix
    };

    // Translate the CNTK model file format into XML:BrainGraph format.
    BG_Graph *Net2Bg(std::wstring filename, FILE *fOut, wchar_t **modelInfo, bool verbose = false);

private:
    // Unparse a MatrixFormat to readable text
    std::wstring BuildMatrixFormat(MatrixFormat format);

    //NB: expects zero-termination
    char * CheckInt32(char *data, int32_t value);
    char * CheckInt64(char *data, int64_t value);
    char * CheckMarker(char *data, wchar_t *name);
    bool TryCheckMarker(char *&data, wchar_t *name);

    char * GetBool(char *data, bool &value);
    char * GetChar(char *data, char &value);
    char * GetInt32(char *data, int32_t &value);
    char * GetInt64(char *data, int64_t &value);
    char * GetUint64(char *data, uint64_t &value);
    char * GetPtrDiff(char *data, ptrdiff_t &value);
    char * GetFloat(char *data, float &value);
    char * GetWstring(char *data, std::wstring& value, size_t maxLen);

    bool SerializeData(BG_Node *node, const uint8_t *source, size_t nBytes);

    // ComputationNode::LoadValue -> Matrix::Read
    char *GetMatrix(char *data, BG_Node *node, bool verbose);
    char *ReadTagList(char *data, wchar_t *eTag, BG_Graph *g);
    char *ReadTensor(char *data, BG_Node *node);
    bool Sanitize(wchar_t *badName);

    // public, jic
    uint64_t versionFound;
};
