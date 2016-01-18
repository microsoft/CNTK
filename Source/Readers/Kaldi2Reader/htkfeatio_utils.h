#pragma once

#include <fstream>
#include <sstream>

namespace msra { namespace asr {

inline std::string toStr(std::wstring w)
{
    return std::string(w.begin(), w.end());
}

inline std::wstring toWStr(std::string s)
{
    return std::wstring(s.begin(), s.end());
}

inline std::string fileToStr(std::string fname)
{
    std::ifstream t(fname, std::ifstream::in);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

inline std::string trimmed(std::string str)
{
    auto found = str.find_first_not_of(" \t\n");
    if (found == string::npos)
    {
        str.erase(0);
        return str;
    }
    str.erase(0, found);
    found = str.find_last_not_of(" \t\n");
    if (found != string::npos)
        str.erase(found + 1);

    return str;
}
}
}