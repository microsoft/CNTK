//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// This is the main header of the CNTK library API containing the entire public API definition. 
//

// Swig specific utility classes, the file should be used only from cntk_py.i
#pragma once

#include <memory>

namespace CNTK
{
#define StartExceptionHandling                                                                  \
    try                                                                                         \
    {

#define EndExceptionHandling                                                                    \
    }                                                                                           \
    catch (const Swig::DirectorException &e) { SWIG_exception(SWIG_RuntimeError, e.what()); }   \
    catch (const Microsoft::MSR::CNTK::IExceptionWithCallStackBase& err)                        \
    {                                                                                           \
        auto msg = std::string(dynamic_cast<const std::exception&>(err).what());                \
        msg = msg + "\n" + err.CallStack();                                                     \
                                                                                                \
        if (dynamic_cast<const std::invalid_argument*>(&err))                                   \
        {                                                                                       \
            SWIG_exception(SWIG_ValueError, msg.c_str());                                       \
        }                                                                                       \
                                                                                                \
        if (dynamic_cast<const std::logic_error*>(&err) ||                                      \
            dynamic_cast<const std::runtime_error*>(&err))                                      \
        {                                                                                       \
            SWIG_exception(SWIG_RuntimeError, msg.c_str());                                     \
        }                                                                                       \
                                                                                                \
        SWIG_exception(SWIG_UnknownError, msg.c_str());                                         \
    }                                                                                           \
    catch (const std::runtime_error &e) { SWIG_exception(SWIG_RuntimeError, e.what()); }        \
    catch (const std::invalid_argument &e) { SWIG_exception(SWIG_ValueError, e.what()); }       \
    catch (const std::logic_error &e) { SWIG_exception(SWIG_RuntimeError, e.what()); }          \
    catch (const std::exception &e) { SWIG_exception(SWIG_UnknownError, e.what()); }            \
    catch (...) { SWIG_exception(SWIG_UnknownError, "Runtime exception"); }

    class AllowThreadsGuard
    {
        PyThreadState* m_state;
    public:
        AllowThreadsGuard()
        {
            m_state = PyEval_SaveThread();
        }

        ~AllowThreadsGuard()
        {
            PyEval_RestoreThread(m_state);
        }

    private:
        AllowThreadsGuard(const AllowThreadsGuard&) = delete; AllowThreadsGuard& operator=(const AllowThreadsGuard&) = delete;
        AllowThreadsGuard& operator=(AllowThreadsGuard&&) = delete; AllowThreadsGuard(AllowThreadsGuard&& other) = delete;
    };

    class GilStateGuard final
    {
        PyGILState_STATE m_state;

    public:
        GilStateGuard() : m_state(PyGILState_Ensure())
        {}

        ~GilStateGuard()
        {
            PyGILState_Release(m_state); 
        }

    private:
        GilStateGuard(const GilStateGuard&) = delete; GilStateGuard& operator=(const GilStateGuard&) = delete;
        GilStateGuard& operator=(GilStateGuard&&) = delete; GilStateGuard(GilStateGuard&& other) = delete;
    };

    inline void PyDecRef(PyObject* p)
    {
        Py_XDECREF(p);
    }

    typedef std::unique_ptr<PyObject, decltype(&PyDecRef)> PyObjectPtr;
}