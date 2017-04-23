#pragma once
/*!
* \file util.h
* \brief Struct Option stores many input arguments 
*/

#include <cstring>
#include <cstdlib>
#include <random>
#include <cassert>
#include <exception>

struct Option
{
    const char* train_file;
    const char* save_vocab_file;
    int min_count;

    void ParseArgs(int argc, char *argv[]);
}; 