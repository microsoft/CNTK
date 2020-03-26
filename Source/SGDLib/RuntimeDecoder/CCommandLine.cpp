#include "stdafx.h"
#include "CCommandLine.h"

CCommandLine::CCommandLine(int argc, wchar_t* argv[])
{
    rassert_op(argc, >=, 2);

    m_action = argv[1];
    for (int i = 2; i < argc; i++)
    {
        std::wstring arg(argv[i]);
        std::wstring key, value;
        bool before_equal = true;
        for (auto c : arg)
        {
            if (before_equal)
            {
                if (c == _T('='))
                    before_equal = false;
                else
                    key.push_back(c);
            }
            else
                value.push_back(c);
        }

        // duplicate key
        rassert_eq(m_optionMap.find(key) == m_optionMap.end(), true);
        m_optionMap.insert(make_pair(key, value));
    }
}

const std::wstring& CCommandLine::GetOption(const std::wstring& key) const
{
    auto iter = m_optionMap.find(key);
    // not found key
    rassert_eq(iter == m_optionMap.end(), false);
    return iter->second;
}

const std::wstring CCommandLine::GetOption(const std::wstring& key, const std::wstring& _default) const
{
    auto iter = m_optionMap.find(key);
    if (iter != m_optionMap.end())
        return iter->second;
    else
        return _default;
}
