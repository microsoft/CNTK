/*!
* \file word_count.cpp
* \brief word_frequency generator on the basis of train_file
*  Usage:
*    [-train_file <train_file>] [-save_vocab <vocab for saving>] [-min_count <number>]
*/
#define _CRT_SECURE_NO_WARNINGS
#include <map>   
#include <fstream>   
#include <iostream>   
#include <string>   

#include "util.h"

using namespace std;

void display_map(map<string, int> &wmap, FILE * file_,Option * option_)
{
    map<string, int>::const_iterator map_it;
    for (map_it = wmap.begin(); map_it != wmap.end(); map_it++)
    {
        if (map_it->second >= option_->min_count) 
        {
            fprintf(file_, "%s   %d\n", (map_it->first).c_str(), map_it->second);
        }
    
    }
}

int main(int argc, char *argv[])
{
    Option *option_= new Option();
    FILE * output_file;
    option_->ParseArgs(argc, argv);
    output_file = fopen(option_->save_vocab_file, "w");
    ifstream ifs(option_->train_file);
    string szTemp;
    map<string, int> wmap;

    while (ifs >> szTemp)
        wmap[szTemp]++;

    display_map(wmap,output_file,option_);

    return false;
}

