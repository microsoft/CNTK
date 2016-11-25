//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// EvalMultithreads.cpp : Sample application shows how to evaluate a model in multiple threading environment. 
//
#include <functional>
#include <thread>
#include <iostream>
#include "CNTKLibrary.h"
#include "LSTM/LstmGraphNode.h"

#pragma warning(push, 0)
#include <graphid.pb.h>
#pragma warning(pop)

extern "C"
{
#include <b64/cencode.h>
}

using namespace CNTK;

#define MAX_BASE64_EXPORT_LENGTH 100

template <typename FunctionType>
void Traverse(const FunctionPtr& rootFunction, std::unordered_set<FunctionPtr>& visitedFunctions, const FunctionType& functor)
{
    visitedFunctions.insert(rootFunction);
    functor(rootFunction);

    std::vector<Variable> rootFunctionInputs = rootFunction->Inputs();
    for (const auto& rootInput : rootFunctionInputs)
    {
        if (rootInput.IsOutput() && visitedFunctions.find(rootInput.Owner()) == visitedFunctions.end())
        {
            const auto& function = rootInput.Owner();
            Traverse(function, visitedFunctions, functor);
        }
    }
}

std::string ConstructUniqueName(const std::wstring& uid, const std::wstring& name)
{
    return std::string(uid.begin(), uid.end()) + "/" + std::string(name.begin(), name.end());
}

std::string ConstructUniqueName(const std::wstring& name)
{
    return std::string(name.begin(), name.end());
}

static std::string EncodeBase64(const char *buf, size_t len)
{
	base64_encodestate state;
	char *sout = new char[len * 2];
	memset(sout, 0, len * 2);

	base64_init_encodestate(&state);
	base64_encode_block(buf, (int)len, sout, &state);
	base64_encode_blockend(sout, &state);

	// TODO: remove once we export everything.
    static_assert(MAX_BASE64_EXPORT_LENGTH > 10, "base64 export should at least be 10");
    if (strlen(sout) > MAX_BASE64_EXPORT_LENGTH)
	{
		strcpy_s(sout + MAX_BASE64_EXPORT_LENGTH - 8, len * 2 - MAX_BASE64_EXPORT_LENGTH, "...");
	}

	auto result = std::string(sout);

	delete[] sout;

	return result;
}

graphIR::Graph* CntkGraphToGraphIr(std::wstring filename, FunctionPtr evalFunc)
{
	auto graphInfo = new graphIR::GraphInfo();
    graphInfo->set_framework_name("CNTK");
    graphInfo->set_framework_version("2.0beta3.0"); // TODO: call cntk function to retrieve version string
    graphInfo->set_graph_version("0.1");
    graphInfo->set_description("Exported by the Graph Ir Exporter from CNTK");
    graphInfo->set_model_name(std::string(filename.begin(), filename.end()));

	auto graph = new graphIR::Graph();
	graph->set_allocated_graph_info(graphInfo);

	std::unordered_set<FunctionPtr> functions;
	Traverse(evalFunc->RootFunction(), functions,
		[&graph](const FunctionPtr& f)
	{
		fprintf(stderr, "now at %S opcode %S\n", f->Uid().c_str(), f->OpName().c_str());

		graphIR::Node *node = graph->add_nodes();

		node->set_name(ConstructUniqueName(f->Uid(), f->Name()));

		auto name = f->OpName();
		node->set_op(std::string(name.begin(), name.end()));

		auto d = f->Attributes();
		std::stringstream strstr(std::ios_base::in | std::ios_base::out | std::ios_base::binary);
		strstr << d;
		auto where = strstr.tellp();
		auto str = strstr.str();

		graphIR::InitArg initArg;
		initArg.set_dbytes(4); // fp32 is 4 bytes per entry

		(*node->mutable_ext_attrs())["##CNTK##NODE##"] = EncodeBase64(str.c_str(), str.length());

		for (auto out : f->Placeholders())
		{
			fprintf(stderr, "Placeholder not expected: %S\n", out.Name().c_str());
		}

		for (auto out : f->Inputs())
		{
			auto input = node->add_inputs();

			input->set_name(ConstructUniqueName(out.Uid(), out.Name()));

			name = L"fp32";
			input->set_dtype(std::string(name.begin(), name.end()));

			input->set_dbytes(4); // fp32 is 4 bytes per entry

			fprintf(stderr, "    <= %S type %d [", out.Name().c_str(), out.GetDataType());

			int rank = 0;
			for (auto dims : out.Shape().Dimensions())
			{
				input->add_shape((int)dims);

				if (rank++ != 0) fprintf(stderr, ", ");
				fprintf(stderr, "%lu", (unsigned long)dims);
			}

			fprintf(stderr, "]\n");
		}

		for (auto out : f->Parameters())
		{
			const auto& buf = out.Value()->DataBuffer<float>();

			size_t rank = 1;
			for (auto dims : out.Shape().Dimensions())
			{
				rank *= dims;
			}

			graphIR::InitArg initArg2;
			initArg2.set_dbytes(4); // fp32 is 4 bytes per entry
			initArg2.set_data_base64(EncodeBase64((char*)buf, rank * 4));

			(*node->mutable_init_attrs())[ConstructUniqueName(out.Uid(), out.Name())] = initArg2;

			fprintf(stderr, "    == %S type %d value %f\n", out.Name().c_str(), out.GetDataType(), buf[0]);
		}

		for (auto out : f->Constants())
		{
			const auto& buf = out.Value()->DataBuffer<float>();

			size_t rank = 1;
			for (auto dims : out.Shape().Dimensions())
			{
				rank *= dims;
			}

			graphIR::InitArg initArg3;
			initArg3.set_dbytes(4); // fp32 is 4 bytes per entry
			initArg3.set_data_base64(EncodeBase64((char *)buf, rank * 4));

			(*node->mutable_init_attrs())[ConstructUniqueName(out.Uid(), out.Name())] = initArg3;

			fprintf(stderr, "    == %S type %d value %f\n", out.Name().c_str(), out.GetDataType(), buf[0]);
		}

		for (auto iter = f->Attributes().begin(); iter != f->Attributes().end(); iter++)
		{
			DictionaryValue value = iter->second;


			std::wstring resultValue = L"";

			switch (value.ValueType())
			{
			case DictionaryValue::Type::Bool:
				resultValue = std::to_wstring(iter->second.Value<bool>());
				break;

			case DictionaryValue::Type::Int:
				resultValue = std::to_wstring(iter->second.Value<int>());
				break;

			case DictionaryValue::Type::SizeT:
				resultValue = std::to_wstring(iter->second.Value<size_t>());
				break;

			case DictionaryValue::Type::Double:
				resultValue = std::to_wstring(iter->second.Value<double>());
				break;

			case DictionaryValue::Type::String:
				resultValue = iter->second.Value<std::wstring>();
				break;

			case DictionaryValue::Type::Float:
				resultValue = std::to_wstring(iter->second.Value<float>());
				break;

			default:
				resultValue = std::wstring(L"<<unsupported>>");
				break;
			}

			(*node->mutable_ext_attrs())[ConstructUniqueName(iter->first)] = std::string(resultValue.begin(), resultValue.end());
		}

		// Combine nodes are special, they just reflect their input to their output
		if (f->OpName() != L"Combine")
		{
			for (auto out : f->Outputs())
			{
				auto output = node->add_outputs();
				output->set_name(ConstructUniqueName(out.Uid(), out.Name()));

				name = L"fp32";
				output->set_dtype(std::string(name.begin(), name.end()));

				output->set_dbytes(4); // fp32 is 4 bytes per entry

				fprintf(stderr, "    => %S type %d [", out.Name().c_str(), out.GetDataType());

				int rank = 0;
				for (auto dims : out.Shape().Dimensions())
				{
					output->add_shape((int)dims);

					if (rank++ != 0) fprintf(stderr, ", ");
					fprintf(stderr, "%lu", (unsigned long)dims);
				}

				fprintf(stderr, "]\n");
			}
		}
	});


    fprintf(stderr, "\n\n");
    for (auto func : functions)
    {
        fprintf(stderr, "X uid %S, op %S\n", func->Uid().c_str(), func->OpName().c_str());
    }

    return graph;
}

CNTK::FunctionPtr GraphIrToCntkGraph(graphIR::Graph* /*graphIrPtr*/, CNTK::FunctionPtr /*modelFuncPtr*/)
{
    return nullptr;
}



#include "GraphIrExporter.h"

BG_DataBlob::BG_DataBlob(const void* data, size_t length, size_t length2, const std::wstring& name)
{
    _data = data;
    _length = length;
    _length2 = length2;
    _name = name;
}

bool BG_DataBlob::Reformat(const std::wstring &)
{
    return true;
}


BG_Node::BG_Node(const std::wstring& name2)
{
    name = name2;
}

bool BG_Node::SetAttribute(const std::wstring& attrib, const std::wstring& value)
{
    attribs[attrib] = value;
    return true;
}

bool BG_Node::SetRank(size_t rank)
{
    dims.resize(rank);
    return true;
}

bool BG_Node::SetDim(size_t dim, size_t value)
{
    dims[dim] = value;
    return true;
}

bool BG_Node::AddTag(const std::wstring& value)
{
    tags.push_back(value);
    return true;
}

bool BG_Node::SetOp(const std::wstring& value)
{
    op = value;
    return true;
}



BG_Graph::BG_Graph()
{
}

BG_Node *BG_Graph::LookupNode(const std::wstring& name2)
{
    for (auto node : nodes)
    {
        if (node->Name() == name2)
        {
            return node;
        }
    }

    return nullptr;
}

bool BG_Graph::AddNode(BG_Node* node)
{
    if (node != nullptr)
    {
        nodes.push_back(node);
    }
    return true;
}

bool BG_Graph::AddArc(const std::wstring& to_nodeName, const std::wstring& from_childName)
{
    arcs.push_back({ to_nodeName, from_childName });
    return true;
}


bool BG_Graph::AddTag(const std::wstring& value)
{
    tags.push_back(value);
    return true;
}

std::wstring BG_Graph::BaseFilename(const std::wstring& filename)
{
    return filename;
}

bool BG_Graph::SetName(const std::wstring& value)
{
    graphName = value;
    return true;
}

bool BG_Graph::SetVersion(const std::wstring& version)
{
    graphVersion = version;
    return true;
}

bool BG_Graph::SetToolkitName(const std::wstring& name)
{
    toolkitName = name;
    return true;
}

bool BG_Graph::SetToolkitVersion(const std::wstring& version)
{
    toolkitVersion = version;
    return true;
}

bool BG_Graph::AddModelInfo(const std::wstring& name, const std::wstring& value)
{
    attribs[name] = value;
    return true;
}

bool BG_Graph::Serialize(FILE* /*fOut*/)
{
    return true;
}




const size_t CntkNetParser::CNTK_MAXIMUM_VERSION_SUPPORTED = 15;
const size_t CntkNetParser::maxCntkStringLen = 256;

CntkNetParser::CntkNetParser() : versionFound(0)
{
}

// Unparse a MatrixFormat to readable text
std::wstring CntkNetParser::BuildMatrixFormat(MatrixFormat format)
{
    if (format == matrixFormatSparseBlockCol)
    {
        return L"Sparse Block ColumnMajor";
    }
    if (format == matrixFormatSparseBlockRow)
    {
        return L"Sparse Block RowMajor";
    }
    //bits
    std::wstring fmt;
    if (format & matrixFormatSparse)
        fmt += L"Sparse";
    else
        fmt += L"Dense";
    if (format & matrixFormatRowMajor)
        fmt += L" RowMajor";
    else
        fmt += L" ColumnMajor";
    if (format & matrixFormatCompressed)
        fmt += L" Compressed";
    return fmt;
}

//NB: expects zero-termination
char * CntkNetParser::CheckMarker(char *data, wchar_t *name)
{
    wchar_t *p = (wchar_t *)data;
    wchar_t *s = name;
    for (;; p++, s++)
    {
        if (*p != *s)
            goto bad;
        if (*s == 0)
            break;
    }
    return (char *)(p + 1);
bad:
    printf("Missing marker %ls\n", name);
    return 0;
}

//NB: expects zero-termination
bool CntkNetParser::TryCheckMarker(char *&data, wchar_t *name)
{
    wchar_t *p = (wchar_t *)data;
    wchar_t *s = name;
    for (;; p++, s++)
    {
        if (*p != *s)
            goto bad;
        if (*s == 0)
            break;
    }
    data = (char *)(p + 1);
    return true;
bad:
    return false;
}

char * CntkNetParser::CheckInt32(char *data, int32_t value)
{
    int32_t *p = (int32_t *)data;
    if (*p != value)
        goto bad;
    return (char *)(p + 1);
bad:
    printf("Wrong value32 %d\n", value);
    return 0;
}

char * CntkNetParser::CheckInt64(char *data, int64_t value)
{
    int64_t *p = (int64_t *)data;
    if (*p != value)
        goto bad;
    return (char *)(p + 1);
bad:
    printf("Missing value64 %lu\n", (unsigned long)value);
    return 0;
}

char * CntkNetParser::GetBool(char *data, bool &value)
{
    bool *p = (bool *)data;
    value = *p;
    return (char *)(p + 1);
}

char * CntkNetParser::GetChar(char *data, char &value)
{
    char *p = (char *)data;
    value = *p;
    return (char *)(p + 1);
}

char * CntkNetParser::GetInt32(char *data, int32_t &value)
{
    int32_t *p = (int32_t *)data;
    value = *p;
    return (char *)(p + 1);
}

char * CntkNetParser::GetInt64(char *data, int64_t &value)
{
    int64_t *p = (int64_t *)data;
    value = *p;
    return (char *)(p + 1);
}

char * CntkNetParser::GetUint64(char *data, uint64_t &value)
{
    uint64_t *p = (uint64_t *)data;
    value = *p;
    return (char *)(p + 1);
}

char * CntkNetParser::GetPtrDiff(char *data, ptrdiff_t &value)
{
    ptrdiff_t *p = (ptrdiff_t *)data;
    value = *p;
    return (char *)(p + 1);
}

char * CntkNetParser::GetFloat(char *data, float &value)
{
    float *p = (float *)data;
    value = *p;
    return (char *)(p + 1);
}

char * CntkNetParser::GetWstring(char *data, std::wstring& value, size_t maxLen)
{
    wchar_t *p = (wchar_t *)data;
    for (;; p++)
    {
        if (*p == 0)
            break;
        if (--maxLen == 0)
            goto bad;
        value.append(1, (wchar_t)*p);
    }

    return (char *)(p + 1);

bad:
    printf("String too long\n");
    return 0;
}

bool CntkNetParser::SerializeData(BG_Node *node, const uint8_t *source, size_t nBytes)
{
    // Create a blob for it
    BG_DataBlob *b = new BG_DataBlob((void*)source, nBytes, nBytes, L"binary");
    if (!b) {
        printf("Cant create new blob??\n");
        return false;
    }

    // For very small buffers no point using Zlib
    bool isSmall = (nBytes <= 1024);
    const std::wstring fmt = (isSmall) ? L"binary.base64" : L"binary.zlib.base64";

    // Compress it
    if (!b->Reformat(fmt))
    {
        printf("Cant create new blob??\n");
        return false;
    }

    node->data = b;
    return true;
}

#define CHECK(x) {x;if (p==0) goto done;}

// ComputationNode::LoadValue -> Matrix::Read
char *CntkNetParser::GetMatrix(char *data, BG_Node *node, bool /*verbose*/)
{
    char *p = (char*)data;
    char typeByte;
    std::wstring matName;
    CHECK(p = GetChar(p, typeByte));
    if ((typeByte != 'd') && (typeByte != 's'))
    {
        printf("bad type byte %c\n", typeByte);
        goto done;
    }
    // We already said what MatrixFormat was, so drop this

    // CPUMatrix:: operator >>
    CHECK(p = CheckMarker(p, L"BMAT"));
    uint64_t elSize;
    CHECK(p = GetUint64(p, elSize));
    node->SetAttribute(L"elementSize", std::to_wstring(elSize));

    CHECK(p = GetWstring(p, matName, maxCntkStringLen));
    // Drop this "unnamed" matName, we already have a name for this node.

    int32_t mFormat;
    CHECK(p = GetInt32(p, mFormat));
    MatrixFormat format = (MatrixFormat)mFormat;
    node->SetAttribute(L"matrixFormat", BuildMatrixFormat(format));

    //bool isRowMajor = ((format == matrixFormatSparseBlockRow) ||
    //    (format & matrixFormatRowMajor));

    uint64_t numRows, numCols;
    CHECK(p = GetUint64(p, numRows));
    CHECK(p = GetUint64(p, numCols));
    // NB: These two attributes are kept only for debugging reasons
    node->SetAttribute(L"numberOfRows", std::to_wstring(numRows));
    node->SetAttribute(L"numberOfColumns", std::to_wstring(numCols));

    // Set the tensor info
    // Identify the case of rank=1
    if ((numRows == 1) || (numCols == 1))
    {
        node->SetRank(1);
        node->SetDim(0, numCols*numRows);
    }
    else
    {
        node->SetRank(2);
        node->SetDim(0, numCols);
        node->SetDim(1, numRows);
    }

    size_t nBytes = numRows*numCols*elSize;

    if (!SerializeData(node, (const uint8_t*)p, nBytes))
        goto done;

    p += nBytes;
    CHECK(p = CheckMarker(p, L"EMAT"));
    return p;

done:
    printf("GetMatrix failed\n");
    return 0;
}

char *CntkNetParser::ReadTagList(char *data, wchar_t *eTag, BG_Graph *g)
{
    char *p = (char *)data;

    // NB: We just skip the "E" and reuse the tag
    const std::wstring listName = eTag + 1;

    // NB: we drop the count info
    uint64_t numNodes;
    CHECK(p = GetUint64(p, numNodes));

    for (uint64_t j = 0; j < numNodes; j++)
    {
        std::wstring nodeName;
        CHECK(p = GetWstring(p, nodeName, maxCntkStringLen));
        BG_Node *node = g->LookupNode(nodeName);
        if (!node)
        {
            printf("Could not find node %ls in graph %ls\n", nodeName.c_str(), g->graphName.c_str());
            goto done;
        }

        node->AddTag(listName);
    }

    CHECK(p = CheckMarker(p, eTag));
    return p;

done:
    return 0;
}

char *CntkNetParser::ReadTensor(char *data, BG_Node *node)
{
    char *p = (char *)data;

    // BUGBUG: Should we cleanup the case of "B0[1 x N]" to "B0[N]"?
    int32_t rank, dim;
    CHECK(p = GetInt32(p, rank));
    node->SetRank(rank);

    for (int32_t j = 0; j < rank; j++)
    {
        CHECK(p = GetInt32(p, dim));
        node->SetDim(j, dim);
    }

    return p;

done:
    printf("Failed to read tensor info for node %ls\n", node->name.c_str());
    return 0;
}

// Unused, left as documentation
bool CntkNetParser::Sanitize(wchar_t *badName)
{
    bool wasOk = true;
    for (size_t i = 0;; i++)
    {
        wchar_t c = badName[i];
        if (c == 0)
            break;

        if ((c == L'.') || (c == L'*') || (c == L'-') ||
            (c == L'[') || (c == L']')) // more??
        {
            wasOk = false;
            badName[i] = L'_';
        }
    }

    return wasOk;
}

// Translate the CNTK model file format into XML:BrainGraph format.
BG_Graph *CntkNetParser::Net2Bg(std::wstring filename, FILE* /*fOut*/, wchar_t **modelInfo, bool verbose)
{
    // Read the whole file in
    FILE *fIn = 0;

    _wfopen_s(&fIn, filename.c_str(), L"rb");
    if (!fIn)
    {
        printf("Cant open %ls\n", filename.c_str());
        return 0;
    }

    fseek(fIn, 0, SEEK_END);
    size_t dataLen = ftell(fIn);
    fseek(fIn, 0, SEEK_SET);

    char *dataMap = new char[dataLen];
    fread(dataMap, 1, dataLen, fIn);
    fclose(fIn);
    fIn = 0;

    // Create an empty graph
    BG_Graph *g = new BG_Graph;

    // Extract the model name from the filename; CNTK has no notion of a model name.
    std::wstring modelName = g->BaseFilename(filename);
    g->SetName(modelName);
    g->SetVersion(L"0.1");

    // Result is bad unless.
    bool ok = false;

    // Start with parsing the header
    char *p = dataMap;

    CHECK(p = CheckMarker(p, L"BCN"));

    // Version check
    CHECK(p = CheckMarker(p, L"BVersion"));
    CHECK(p = GetUint64(p, versionFound));
    if (versionFound > CNTK_MAXIMUM_VERSION_SUPPORTED)
    {
        printf("Saved CNTK model version %lu is not supported.\n", (unsigned long)versionFound);
        goto done;
    }

    g->SetToolkitName(L"CNTK");
    g->SetToolkitVersion(std::to_wstring(versionFound));

    CHECK(p = CheckMarker(p, L"EVersion"));

    // Add any model information we might have
    if (modelInfo)
    {
        // The Windows shell strips all quotes, so its not trivial
        for (; *modelInfo; modelInfo++)
        {
            wchar_t *name = *modelInfo;

            wchar_t *value = wcschr(name, L'=');
            if (!value)
            {
                printf("bad modelInfo clause '%ls'. Should be in the form <name>=\"<value>\"\n", name);
                goto done;
            }
            *value++ = 0; //zero-terminate the name part

                            //validate value
            if (*value == 0)
            {
                printf("bad modelInfo clause '%ls'. Missing value\n", name);
                goto done;
            }

            //make sure no quotes, or properly quoted
            {
                if (*value == L'"')
                    value++;
                size_t len = wcslen(value);
                if (len == 0)
                {
                Malformed:
                    printf("bad modelInfo clause '%ls'. Malformed value\n", name);
                    goto done;
                }

                if (value[len - 1] == L'"')
                {
                    value[len - 1] = 0;
                    if (--len == 0)
                        goto Malformed;
                }

                wchar_t *p2 = value;
                for (; *p2; p2++)
                {
                    if ((*p2 == L'"') && (p2[-1] != L'\\'))
                        goto Malformed;
                }
            }

            g->AddModelInfo(name, value);
        }
    }

    // Node list
    uint64_t nNodes;
    CHECK(p = GetUint64(p, nNodes));
    CHECK(p = CheckMarker(p, L"BNodeList"));

    for (uint64_t i = 0; i < nNodes; i++)
    {
        BG_Node *curNode = 0;
        std::wstring precision;

        CHECK(p = GetWstring(p, precision, 12));
        if ((precision != L"float") && (precision != L"double"))
        {
            printf("bad precision %ls\n", precision.c_str());
            goto done;
        }

        std::wstring operation;
        CHECK(p = GetWstring(p, operation, maxCntkStringLen));

        std::wstring nodeName;
        CHECK(p = GetWstring(p, nodeName, maxCntkStringLen));

        curNode = new BG_Node(nodeName);
        if (!curNode)
        {
            printf("Internal error: cannot create new node\n");
            goto done;
        }

        curNode->SetOp(operation);
        curNode->SetAttribute(L"dataType", precision);
        g->AddNode(curNode);

        if (verbose)
        {
            fprintf(stderr, "Found %ls its a %ls\n", nodeName.c_str(), operation.c_str());
        }

        // Do we have additional data ?
        if (operation == L"LearnableParameter")
        {
            float lrMultiplier;
            CHECK(p = GetFloat(p, lrMultiplier));
            curNode->SetAttribute(L"learningRateMultiplier", std::to_wstring(lrMultiplier));

            // TensorShape::Load
            CHECK(p = ReadTensor(p, curNode));
            CHECK(p = GetMatrix(p, curNode, verbose));

        }
        else if ((operation == L"InputValue") || (operation == L"SparseInputValue"))
        {
            uint64_t numRows, numCols;
            CHECK(p = GetUint64(p, numRows));
            CHECK(p = GetUint64(p, numCols));

#if 0
            // This is always 0, apparently.
            curNode->SetAttribute(L"numberOfRows", std::to_wstring(numRows));
            curNode->SetAttribute(L"numberOfColumns", std::to_wstring(numCols));
#endif

            // TensorShape::Load
            CHECK(p = ReadTensor(p, curNode));

            //since modelVersion is >= 8
            int32_t nrAxes;
            CHECK(p = GetInt32(p, nrAxes));
            curNode->SetAttribute(L"numberOfAxes", std::to_wstring(nrAxes));

            if (nrAxes == 1)
            {
                std::wstring dynAxisName;
                CHECK(p = GetWstring(p, dynAxisName, maxCntkStringLen));
                curNode->SetAttribute(L"dynAxis", dynAxisName);
            }

            if (versionFound > 10)
            {
                float learningRateMultiplier;
                CHECK(p = GetFloat(p, learningRateMultiplier));
                curNode->SetAttribute(L"learningRateMultiplier", std::to_wstring(learningRateMultiplier));
            }

        }
        else if ((operation == L"InvStdDev") || (operation == L"Mean"))
        {
            bool hasComputed;
            CHECK(p = GetBool(p, hasComputed));

            curNode->SetAttribute(L"hasComputed",
                (hasComputed) ? L"true" : L"false");
            CHECK(p = GetMatrix(p, curNode, verbose));

        }
        else if (operation == L"Times")
        {
            uint64_t outputRank;
            CHECK(p = GetUint64(p, outputRank));
            curNode->SetAttribute(L"outputRank", std::to_wstring(outputRank));

            if (versionFound >= 12)
            {
                int32_t inferInputRankToMap;
                CHECK(p = GetInt32(p, inferInputRankToMap));
                curNode->SetAttribute(L"inferiorInputRank", std::to_wstring(inferInputRankToMap));
            }
        }
        else if ((operation == L"PastValue") || (operation == L"FutureValue"))
        {
            int32_t timeStep;
            CHECK(p = GetInt32(p, timeStep));
            curNode->SetAttribute(L"direction", std::to_wstring(timeStep));

            // TensorShape::Load
            CHECK(p = ReadTensor(p, curNode));

            float activationValue;
            CHECK(p = GetFloat(p, activationValue));
            curNode->SetAttribute(L"activationValue", std::to_wstring(activationValue));

        }
        else if (operation == L"Slice")
        {
            ptrdiff_t beginIndex, length;
            CHECK(p = GetPtrDiff(p, beginIndex));
            CHECK(p = GetPtrDiff(p, length));
            curNode->SetAttribute(L"beginIndex", std::to_wstring(beginIndex));
            curNode->SetAttribute(L"length", std::to_wstring(length));

            int32_t axis;
            CHECK(p = GetInt32(p, axis));
            curNode->SetAttribute(L"axis", std::to_wstring(axis));

        }
        else if (operation == L"RowStack")
        {
            int32_t spliceDim;
            CHECK(p = GetInt32(p, spliceDim));
            curNode->SetAttribute(L"spliceDimension", std::to_wstring(spliceDim));

        }
        else
        {
            if (verbose)
                fprintf(stderr, "Assuming no data for operation %ls\n", operation.c_str());
        }
    }

    CHECK(p = CheckMarker(p, L"ENodeList"));

    // Relations (aka arcs of the graph)
    CHECK(p = CheckMarker(p, L"BRelation"));

    for (uint64_t i = 0; i < nNodes; i++)
    {
        std::wstring nodeName;
        uint64_t nChildren;

        CHECK(p = GetWstring(p, nodeName, maxCntkStringLen));
        CHECK(p = GetUint64(p, nChildren));

        for (uint64_t j = 0; j < nChildren; j++)
        {
            std::wstring childName;
            CHECK(p = GetWstring(p, childName, maxCntkStringLen));
            CHECK(g->AddArc(/*to*/nodeName, /*from*/childName));
        }
    }
    CHECK(p = CheckMarker(p, L"ERelation"));

    // "RootNodes" e.g. nodes that share a common tag
    // They may or may not be actual roots of the graph
    CHECK(p = CheckMarker(p, L"BRootNodes"));

    if (TryCheckMarker(p, L"BFeatureNodes"))
    {
        CHECK(p = ReadTagList(p, L"EFeatureNodes", g));
    }
    if (TryCheckMarker(p, L"BLabelNodes"))
    {
        CHECK(p = ReadTagList(p, L"ELabelNodes", g));
    }
    if (TryCheckMarker(p, L"BCriterionNodes"))
    {
        CHECK(p = ReadTagList(p, L"ECriterionNodes", g));
    }
    if (TryCheckMarker(p, L"BCriteriaNodes"))
    {
        CHECK(p = ReadTagList(p, L"ECriteriaNodes", g));
    }
    if (TryCheckMarker(p, L"BNodesReqMultiSeqHandling"))
    {
        CHECK(p = ReadTagList(p, L"ENodesReqMultiSeqHandling", g));
    }
    if (TryCheckMarker(p, L"BEvalNodes"))
    {
        CHECK(p = ReadTagList(p, L"EEvalNodes", g));
    }
    if (TryCheckMarker(p, L"BOutputNodes"))
    {
        CHECK(p = ReadTagList(p, L"EOutputNodes", g));
    }
    if (TryCheckMarker(p, L"BPairNodes"))
    {
        CHECK(p = ReadTagList(p, L"EPairNodes", g));
    }

    CHECK(p = CheckMarker(p, L"ERootNodes"));

    // End of the model
    CHECK(p = CheckMarker(p, L"ECN"));

    ok = true;

done:
    delete dataMap;

    if (!ok)
    {
        delete g;
        g = nullptr;
    }

    return g;
}
#undef CHECK




// Driver program
int wmain(int argc, wchar_t **argv)
{
    bool debug = false;

    if (argc < 2) {
    Usage:
        printf("Usage: %ls [options] <cntk saved net filename> [<xml filename> [modelInfo clauses]]\n", argv[0]);
        printf("Options:\n");
        printf(" -verbose\tTrace intermediate steps\n");
        printf(" [modelInfo]\tShould be in the form name=\"value\"\n");
        printf("Example: %ls simple.dnn simple.dnn.xml author=\"The CNTK folks\"\n", argv[0]);
        return -1;
    }

    while ((argc > 2) && (argv[1][0] == L'-'))
    {
        switch (argv[1][1]) {

        case L'v':
            debug = true;
            break;

        default:
            printf("Unknown option %ls\n", argv[1]);
            goto Usage;
        }
        argc--, argv++;
    }

    // Where does the output go
    FILE *fOut = 0;
    if (argc > 2)
    {
        _wfopen_s(&fOut, argv[2], L"wb");
        if (fOut == 0)
        {
            printf("Could not create file %ls\n", argv[2]);
            return 2;
        }
    }
    else
        fOut = stdout;

    // Do the parsing and translating
    if (debug)
        fprintf(stderr, "Translating net %ls...\n", argv[1]);

    CntkNetParser parser;
    bool ok = false;
    BG_Graph *g = parser.Net2Bg(argv[1], fOut, (argc>3) ? argv + 3 : 0, debug);
    ok = g != 0;
    if (!ok || debug)
        fprintf(stderr, "Translation %s\n", (ok) ? "ok" : "failed");

    if (ok)
        ok = g->Serialize(fOut);

    if (!ok || debug)
        fprintf(stderr, "Serialization %s\n", (ok) ? "ok" : "failed");

    return (ok) ? 0 : 1;
}
