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
    if (found == std::string::npos)
    {
        str.erase(0);
        return str;
    }
    str.erase(0, found);
    found = str.find_last_not_of(" \t\n");
    if (found != std::string::npos)
        str.erase(found + 1);

    return str;
} 

} }

struct MatchPathSeparator
{
    bool operator()(char ch) const
    {
        return ch == '\\' || ch == '/';
    }
};

inline std::string basename(std::string const &pathname)
{
    return std::string(std::find_if(pathname.rbegin(), pathname.rend(), MatchPathSeparator()).base(), pathname.end());
}

inline std::wstring basename(std::wstring const &pathname)
{
    return std::wstring(std::find_if(pathname.rbegin(), pathname.rend(), MatchPathSeparator()).base(), pathname.end());
}

inline std::string removeExtension(std::string const &filename)
{
    size_t lastindex = filename.find_first_of(".");
    return filename.substr(0, lastindex);
}

inline std::wstring removeExtension(std::wstring const &filename)
{
    size_t lastindex = filename.find_first_of(L".");
    return filename.substr(0, lastindex);
}
