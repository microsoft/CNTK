//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <exception>
#include <mutex>

namespace CNTK {

// Class that allows to capture/rethrow exceptions happened on different threads.
class ExceptionCapture
{
public:
    // This method executes f and catches the first happened exception.
    // Thread-safe.
    template <typename Function, typename... Parameters>
    void SafeRun(Function f, Parameters... params)
    {
        try
        {
            f(params...);
        }
        catch (...)
        {
            Capture();
        }
    }

    // Should be called from the master thread. Throws if exception happened.
    void RethrowIfHappened()
    {
        if (m_exception)
        {
            std::rethrow_exception(m_exception);
        }
    }

private:
    // Captures the exception.
    // Thread-safe.
    void Capture()
    {
        std::unique_lock<std::mutex> guard(m_lock);
        // Capturing only the first exception
        if (!m_exception)
        {
            m_exception = std::current_exception();
        }
    }

    std::exception_ptr m_exception;
    std::mutex m_lock;
};

}