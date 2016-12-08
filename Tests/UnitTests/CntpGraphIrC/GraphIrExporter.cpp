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

int MAX_BASE64_EXPORT_LENGTH = 100;

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
    assert(MAX_BASE64_EXPORT_LENGTH > 10, "base64 export should at least be 10");
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

void DumpAsJson(const google::protobuf::Message& message, const std::string& filename)
{
    // save it out to disk in json format.
    std::string jsonstring;
    auto serialized = google::protobuf::util::MessageToJsonString(message, &jsonstring);
    auto fp = fopen((filename + std::string(".pb.json")).c_str(), "w+");
    auto written = fwrite(jsonstring.c_str(), sizeof(char), jsonstring.length(), fp);

    assert(written == jsonstring.length());
    fclose(fp);
}

void DumpAsBinary(const google::protobuf::Message& message, const std::string& filename)
{
    auto filePath = (filename + std::string(".pb"));

    std::fstream fs;
    fs.open(filePath, std::fstream::in | std::fstream::out | std::fstream::trunc);

    message.SerializeToOstream(&fs);

    fs.close();
}

