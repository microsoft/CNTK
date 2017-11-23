#pragma once
#include <iostream>
#include <vector>
#include <string>
using namespace std;

typedef unsigned int ZipState;
#define ZipState_OK 0
#define ZipState_Error 1
#define ZipState_Not_Opened 2

struct ZipSeqInfo
{
	std::string zip_seq_name;
	size_t zip_seq_index;
	size_t zip_seq_size;

	ZipSeqInfo(std::string, size_t, size_t);
};

class ZipDecoder
{
public:
	ZipState zip_state;
	std::string zip_path;
	FILE* zip_file;
	std::vector<ZipSeqInfo>ZipInfo;

	ZipDecoder();

	ZipDecoder(std::string);

	ZipDecoder(FILE*);

	void OpenZip(std::string);

	void zip_get_central_directory(size_t&, size_t&, size_t&);

	size_t zip_get_entries();

	void zip_get_seq_info(size_t, ZipSeqInfo&);

	void decode();
};
