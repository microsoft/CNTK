//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <stack>
#include <functional>
#include <mutex>

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// conc_stack -- very simple version of thread-safe stack. Add other functions as needed.
// Kept in a separate header because it pulls in some large headers that are not super-commonly needed otherwise.
// -----------------------------------------------------------------------

template <typename T>
class conc_stack
{
public:
    typedef typename std::stack<T>::value_type value_type;

    conc_stack()
    {
    }

    value_type pop_or_create(std::function<value_type()> factory)
    {
        {
            std::lock_guard<std::mutex> g(m_locker);
            if (m_stack.size() != 0)
            {
                auto res = std::move(m_stack.top());
                m_stack.pop();
                return res;
            }
        }

        return factory();
    }

    void push(const value_type& item)
    {
        std::lock_guard<std::mutex> g(m_locker);
        m_stack.push(item);
    }

    void push(value_type&& item)
    {
        std::lock_guard<std::mutex> g(m_locker);
        m_stack.push(std::forward<value_type>(item));
    }

public:
    conc_stack(const conc_stack&) = delete;
    conc_stack& operator=(const conc_stack&) = delete;
    conc_stack(conc_stack&&) = delete;
    conc_stack& operator=(conc_stack&&) = delete;

private:
    std::stack<value_type> m_stack;
    std::mutex m_locker;
};
} } }
