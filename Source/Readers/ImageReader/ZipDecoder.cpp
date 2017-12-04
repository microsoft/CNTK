#include "stdafx.h"
#include <iostream>
#include <assert.h>
#include <vector>
#include <string>
#include "ZipDecoder.h"
using namespace std;

inline size_t hex2dec(char* str, size_t size)
{
    //assert(size <= 8);
    size_t result = 0;
    for (size_t i(0); i < size; ++i)
        memcpy(((unsigned char*)(&result)) + i, str + i, 1);

    return result;
}

bool isImageFormat(char* str, size_t size)
{
    for (size_t i(0); i < size; ++i)
    {
        if (str[i] == '.')
        {
            string extension = string(str + i + 1, size - i - 1);
            if (extension == "jpg"
                || extension == "JPG"
                || extension == "jpeg"
                || extension == "JPEG"
                || extension == "png"
                || extension == "PNG")
                return true;
            else
                return false;
        }
    }

    return false;
}

ZipSeqInfo::ZipSeqInfo(std::string name, size_t index, size_t size)
    :zip_seq_name(name), zip_seq_index(index), zip_seq_size(size)
{}

void ZipDecoder::OpenZip(std::string path)
{
    zip_path = path;
    zip_file = fopen(zip_path.c_str(), "rb");
    zip_state = ZipState_Error;
    assert(nullptr != zip_file);
    zip_state = ZipState_OK;
    decode();
}

ZipDecoder::ZipDecoder()
{
    zip_state = ZipState_Not_Opened;
}

ZipDecoder::ZipDecoder(std::string path) :zip_path(path)
{
    zip_path = path;
    zip_file = fopen(zip_path.c_str(), "rb");
    zip_state = ZipState_Error;
    assert(nullptr != zip_file);
    zip_state = ZipState_OK;
    decode();
}

ZipDecoder::ZipDecoder(FILE* file)
{
    zip_file = file;
    zip_state = ZipState_Error;
    assert(nullptr != zip_file);
    zip_state = ZipState_OK;
    decode();
}

size_t ZipDecoder::zip_get_entries()
{
    return ZipInfo.size();
}

void ZipDecoder::zip_get_seq_info(size_t index, ZipSeqInfo& zsi)
{
    assert(index >= 0 && index < ZipInfo.size());
    zsi = ZipInfo[index];
}

void ZipDecoder::zip_get_central_directory(size_t& entries, size_t& size, size_t& offset)
{
    long len = 21;
    const size_t buffer_size = 1 << 12; //4KB buffer size
    auto buffer = new char[buffer_size];
	auto ptr = buffer;
    size_t bytesRead;

    fseek(zip_file, -21, SEEK_END);
    do {
        ++len;
        fseek(zip_file, -1, SEEK_CUR);
        bytesRead = fread(buffer, len, 1, zip_file);
        fseek(zip_file, -len, SEEK_END);
    } while (buffer[0] != 0x50 || buffer[1] != 0x4b || buffer[2] != 0x05 || buffer[3] != 0x06);
    entries = hex2dec(buffer + 10, 2);
    size = hex2dec(buffer + 12, 4);
    offset = hex2dec(buffer + 16, 4);
    if (0xffff == entries || 0xffffffff == size || 0xffffffff == offset)
    {
        len += 75;
        fseek(zip_file, -len, SEEK_END);
        do {
            ++len;
            fseek(zip_file, -1, SEEK_CUR);
            bytesRead = fread(buffer, 56, 1, zip_file);
            fseek(zip_file, -56, SEEK_CUR);
        } while (buffer[0] != 0x50 || buffer[1] != 0x4b || buffer[2] != 0x06 || buffer[3] != 0x06);
        if (0xffff == entries)
            entries = hex2dec(buffer + 32, 8);
        if (0xffffffff == size)
            size = hex2dec(buffer + 40, 8);
        if (0xffffffff == offset)
            offset = hex2dec(buffer + 48, 8);
    }
    
    delete[] ptr;
}

void ZipDecoder::decode()
{
    size_t central_directory_entries;
    size_t central_directory_offset;
    size_t central_directory_size;
    zip_get_central_directory(central_directory_entries, central_directory_size, central_directory_offset);

    const size_t buffer_size = central_directory_size;
    auto buffer = new char[buffer_size];
	auto ptr = buffer;

    _fseeki64(zip_file, central_directory_offset, SEEK_SET);
    size_t bytesRead = fread(buffer, central_directory_size, 1, zip_file);

    size_t offset;
    size_t compressed_size;
    size_t file_name_length;
    size_t extra_field_length;
    size_t file_comment_length;

    for (size_t i(0); i < central_directory_entries; ++i)
    {
        buffer += 20;
        compressed_size = hex2dec(buffer, 4);
        buffer += 8;
        file_name_length = hex2dec(buffer, 2);
        buffer += 2;
        extra_field_length = hex2dec(buffer, 2);
        buffer += 2;
        file_comment_length = hex2dec(buffer, 2);
        buffer += 10;
        offset = hex2dec(buffer, 4);
        buffer += 4;
        ZipInfo.push_back(ZipSeqInfo(string(buffer, file_name_length), offset + 30 + file_name_length, compressed_size));
        buffer += file_name_length + extra_field_length + file_comment_length;
    }
    
    delete[] ptr;
}
