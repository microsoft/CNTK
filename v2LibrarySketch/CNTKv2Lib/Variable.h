#pragma once

#include <vector>
#include "Value.h"
#include "Function.h"

namespace CNTK
{
    class Variable
    {
    public:
        enum class Type
        {
            Const,
            Input,
            Computed
        };

    public:
        // Create an input Variable
        Variable(std::vector<long long> shape, const std::wstring& name = L"");

        // Create a const Variable
        Variable(Value constValue, const std::wstring& name = L"");

        std::vector<long long> Shape() const;
        Type Type() const;
        const std::wstring& Name() const;

        // Function that owns this variable. nullptr for Const and Input variables
        Function Owner() const;
    };
}
