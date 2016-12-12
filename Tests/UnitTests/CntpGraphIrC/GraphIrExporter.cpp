//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// EvalMultithreads.cpp : Sample application shows how to evaluate a model in multiple threading environment. 
//
#define _CRT_SECURE_NO_WARNINGS
#include <functional>
#include <thread>
#include <iostream>
#include <fstream>
#include "GraphIrExporter.h"

extern "C"
{
#include <b64/cencode.h>
#include <b64/cdecode.h>
}

using namespace CNTK;

static int MAX_BASE64_EXPORT_LENGTH = 100;

std::string EncodeBase64(const char *buf, size_t len)
{
	base64_encodestate state;
	char *sout = new char[len * 2 + 4];

    char *temp = sout;
	base64_init_encodestate(&state);
	temp += base64_encode_block(buf, (int)len, temp, &state);
	temp += base64_encode_blockend(temp, &state);
    *temp++ = '\0';

	// TODO: remove once we export everything.
    assert(MAX_BASE64_EXPORT_LENGTH > 10); // base64 export should at least be 10
    if (strlen(sout) > MAX_BASE64_EXPORT_LENGTH)
	{
		strcpy_s(sout + MAX_BASE64_EXPORT_LENGTH - 8, len * 2 - MAX_BASE64_EXPORT_LENGTH, "...");
	}

	auto result = std::string(sout);

	delete[] sout;

	return result;
}

std::vector<char> DecodeBase64(const std::string str)
{
    base64_decodestate state;
    char *sout = new char[str.length()];

    base64_init_decodestate(&state);
    auto len = base64_decode_block(str.c_str(), (int)str.size(), sout, &state);

    return std::vector<char>(sout, sout + len);
}

std::wstring TransformCntkToGraphIr(const std::string& name, const CNTK::DeviceDescriptor& device)
{
    auto filename = name;
    auto filenameW = std::wstring(filename.begin(), filename.end());

    // The model file will be trained and copied to the current runtime directory first.
    auto modelFuncPtr = CNTK::Function::LoadModel(filenameW, device);

    // construct default output name by stripping extension.
    auto idx = filename.find_last_of('.');
    if (idx != std::string::npos && idx > 0)
    {
        filename = filename.substr(0, idx);
    }

    // json dump does not contain entire raw array data
    // because the output would be too big.
    MAX_BASE64_EXPORT_LENGTH = 100;
    auto jsonfilename = filename + std::string(".graphir_json");
    auto message = GRAPHIR::Serialize(modelFuncPtr);
    {
        std::string jsonstring;
        auto serialized = google::protobuf::util::MessageToJsonString(*message, &jsonstring);
        auto fp = fopen(jsonfilename.c_str(), "w+");
        auto written = fwrite(jsonstring.c_str(), sizeof(char), jsonstring.length(), fp);

        assert(written == jsonstring.length());
        fclose(fp);
    }

    // re-serialize with entire array buffers to dump
    // in binary format. this includes full array data.
    MAX_BASE64_EXPORT_LENGTH = INT_MAX;
    auto pbfilename = filename + std::string(".graphir_model");
    auto message2 = GRAPHIR::Serialize(modelFuncPtr);
    {
        std::fstream fs;
        fs.open(pbfilename, std::ios::in | std::ios::out | std::ios::trunc | std::ios::binary);

        message2->SerializeToOstream(&fs);
        fs.close();
    }

    // return the name of the saved graph-ir model.
    return std::wstring(pbfilename.begin(), pbfilename.end());
}

std::wstring TransformGraphIrToCntk(const std::string& filename, const CNTK::DeviceDescriptor& device)
{
    auto filenameW = std::wstring(filename.begin(), filename.end());

    std::fstream fs;
    fs.open(filenameW, std::ios::in | std::ios::binary);

    //
    // Import the original function
    //

    graphIR::Graph message2;
    if (!message2.ParseFromIstream(&fs))
    {
        throw std::string("Cannot load/parse.");
    }

    // construct default output name by stripping extension.
    auto idx = filenameW.find_last_of(L'.');
    if (idx != std::string::npos && idx > 0)
    {
        filenameW = filenameW.substr(0, idx);
    }

    // note: must use the binary serialization since json
    // does not contain full array data.
    auto importedFunction = GRAPHIR::Deserialize(&message2);

    fs.close();

    auto cntkfilename = filenameW + L".cntk_model";
    importedFunction->SaveModel(cntkfilename);

    // return the name of the saved cntk model.
    return cntkfilename;
}
