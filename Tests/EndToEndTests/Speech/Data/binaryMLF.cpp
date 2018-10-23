#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <assert.h>    
#include <unordered_map>
using namespace std;


typedef unsigned int uint;
typedef unsigned short ushort;

const uint MAX_UTT_ID = numeric_limits<uint>::max();
const uint MAX_SENONE_COUNT = numeric_limits<ushort>::max();
const string MLF_BIN_LABEL = "MLF";
const short MODEL_VERSION = 2;
const size_t SENONE_ZEROS = 100000;

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

vector<string> split(const string &s, char delim) {
    std::vector<string> elems;
    split(s, delim, back_inserter(elems));
    return elems;
}

string trim(string s)
{
    auto res = s.substr(s.find_first_not_of(" \n\r\t\""));
    res.erase(res.find_last_not_of(" \n\r\t\"") + 1);
    return res;
}

unordered_map<string, ushort>  GenerateLabelToIdMap(string filePath)
{
    unordered_map<string, ushort> labelToId;
    ushort id = 0;
    ifstream stateListFile(filePath);

    string str;
    while (getline(stateListFile, str))
    {
        str = trim(str);
        pair<string, ushort> entry(str, id);
        labelToId.insert(entry);
        id++;
    }
    return labelToId;
}

int main(int argc, char * argv[])
{
    if (argc != 4)
    {
        cout << "USAGE: MLF_file State_list_file Output_file" << endl;
        exit(EXIT_FAILURE);
    }
    cout << argv[0] << endl;
    ifstream mlfFile(argv[1]);
    auto labelToId = GenerateLabelToIdMap(argv[2]);

    string str;
    getline(mlfFile, str);
    str = trim(str);
    if (str != "#!MLF!#")
    {
        cout << "Warning: Input file is missing MLF header. It may be malformed." << endl;
    }

    string outFilePath = argv[3];

    //write header
    ofstream outFile;
    outFile.open(outFilePath, ios::out | ios::binary);
    outFile << MLF_BIN_LABEL;
    outFile.write((char*)&MODEL_VERSION, sizeof(short));
    vector<ushort> sequence;
    // Read MLF and write body
    bool first = true;
    uint senEnd = 0;
    while (getline(mlfFile, str))
    {
        // Write utt id
        auto sp = split(str, ' ');
        sp[0] = trim(sp[0]);
        if (sp.size() == 1 && sp[0] == "#!MLF!#")
        {
            first = true;
        }
        else if (sp.size() == 1 && sp[0] != ".") {
            assert(sequence.size() % 2 == 0);
            ushort seqSize = (ushort)(sequence.size()/2);

            if (seqSize == 0)
            {
                if (!first) {
                    cout << "ERROR: empty MLF sequence";
                    exit(EXIT_FAILURE);
                }
                first = false;
            }
            else {
                outFile.write((char*)&senEnd, sizeof(uint));
                outFile.write((char*)&seqSize, sizeof(ushort));
                outFile.write((char*)sequence.data(), sizeof(ushort) * sequence.size());
                sequence.clear();
            }
            if (MODEL_VERSION == 1)
            {
                auto utId = stoul(sp[0]);
                assert(utId < MAX_UTT_ID);
                outFile.write((char*)&utId, sizeof(uint));
            }
            else {
                auto utId = sp[0];
                utId = utId.substr(0, utId.find("."));
                ushort len = utId.size();

                outFile.write((char*)&len, sizeof(ushort));
                outFile << utId;
            }

        }
        else if (sp[0] != "." && sp[0] != "#!MLF!#") {
            // write label and count
            sp[1] = trim(sp[1]);
            auto senStart = stoul(sp[0]) / SENONE_ZEROS;
            senEnd = stoul(sp[1]) / SENONE_ZEROS;
            auto key = trim(sp[2]);
            size_t senCount = (senEnd - senStart);
            //cout << senCount << "  " << labelToId.at(key) << endl;
            if (senCount > MAX_SENONE_COUNT) {
                cout << "ERROR: senone count is greater than limit " << MAX_SENONE_COUNT << endl;
                exit(EXIT_FAILURE);
            }
            short senCountShort = (short)senCount;
            sequence.push_back(labelToId.at(key));
            sequence.push_back(senCountShort);
        }
    }
    // Write the last entry
    assert(sequence.size() % 2 == 0);
    ushort seqSize = (ushort)(sequence.size()/2);
    outFile.write((char*)&senEnd, sizeof(uint));
    outFile.write((char*)&seqSize, sizeof(ushort));
    outFile.write((char*)sequence.data(), sizeof(ushort) * sequence.size());
    sequence.clear();

    outFile.close();

    cout << "Finished." << endl;
}