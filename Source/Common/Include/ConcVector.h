//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include <functional>
#include <mutex>

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// conc_vector -- very simple version of thread-safe vector. Add other functions as needed.
// Kept in a separate header because it pulls in some large headers that are not super-commonly needed otherwise.
// -----------------------------------------------------------------------

template <typename T>
class conc_vector
{
public:
    typedef typename std::vector<T>::value_type value_type;

    conc_vector()
    {
    }

    value_type at_or_create(int i, std::function<value_type(int)> factory)
    {
        std::lock_guard<std::mutex> g(m_locker);
        while (i >= m_vector.size())
            m_vector.emplace_back(factory(static_cast<int>(m_vector.size())));
        return std::move(m_vector[i]);
    }

    void assignTo(int i, const value_type& item)
    {
        std::lock_guard<std::mutex> g(m_locker);
        m_vector[i] = item;
    }

    void assignTo(int i, value_type&& item)
    {
        std::lock_guard<std::mutex> g(m_locker);
        m_vector[i] = std::forward<value_type>(item);
    }

public:
    conc_vector(const conc_vector&) = delete;
    conc_vector& operator=(const conc_vector&) = delete;
    conc_vector(conc_vector&&) = delete;
    conc_vector& operator=(conc_vector&&) = delete;

private:
    std::vector<value_type> m_vector;
    std::mutex m_locker;
};
} } }
