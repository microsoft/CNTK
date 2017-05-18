//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKExceptionHandling.i -- exception handling common for Python, C# and Java
//

//This file contains exception handling common for Python, C# and Java
%{
    #include "ExceptionWithCallStack.h"
%}
//
// Exception handling
//
%exception {
    try { $action }
    catch (const Swig::DirectorException &e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
    catch (const Microsoft::MSR::CNTK::IExceptionWithCallStackBase& err)
    {
        auto msg = std::string(dynamic_cast<const std::exception&>(err).what());
        msg = msg + "\n" + err.CallStack();
        
        if (dynamic_cast<const std::invalid_argument*>(&err)) 
        {
           SWIG_exception(SWIG_ValueError, msg.c_str()); 
        }
        
        if (dynamic_cast<const std::logic_error*>(&err) || 
            dynamic_cast<const std::runtime_error*>(&err)) 
        {
           SWIG_exception(SWIG_RuntimeError, msg.c_str()); 
        }
       
        SWIG_exception(SWIG_UnknownError, msg.c_str()); 
    }
    catch (const std::runtime_error &e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
    catch (const std::invalid_argument &e) { SWIG_exception(SWIG_ValueError, e.what()); }
    catch (const std::logic_error &e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
    catch (const std::exception &e) { SWIG_exception(SWIG_UnknownError, e.what()); }
    catch (...) { SWIG_exception(SWIG_UnknownError,"Runtime exception"); }
}

