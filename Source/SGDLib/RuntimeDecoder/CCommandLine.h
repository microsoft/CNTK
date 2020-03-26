#pragma once

#include <string>
#include <map>

class CCommandLine
{
public:
    CCommandLine(int argc, wchar_t* argv[]);
    const std::wstring& GetAction() const { return m_action; }
    const std::wstring& GetOption(const std::wstring& name) const;
    const std::wstring GetOption(const std::wstring& name, const std::wstring& _default) const;

private:
    std::wstring m_action;
    std::map<std::wstring, std::wstring> m_optionMap;
};
