//This file contains exception handling common for Python and C#

//
// Exception handling
//
// TODO: print out C++ call stack trace info.
// It is currently not available as a part of exception message.
%exception {
    try { $action }
    catch (const Swig::DirectorException &e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
    catch (const std::runtime_error &e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
    catch (const std::invalid_argument &e) { SWIG_exception(SWIG_ValueError, e.what()); }
    catch (const std::logic_error &e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
    catch (const std::exception &e) { SWIG_exception(SWIG_UnknownError, e.what()); }
    catch (...) { SWIG_exception(SWIG_UnknownError,"Runtime exception"); }
}

