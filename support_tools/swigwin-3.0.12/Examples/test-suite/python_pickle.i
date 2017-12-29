%module python_pickle


%include <std_string.i>

%extend PickleMe {
#if 0
// Note: %pythoncode can't be used with -builtin
%pythoncode %{
def __reduce__(self):
    print "In Python __reduce__"
    return (type(self), (self.msg, ))
%}
#else
  // Equivalent to Python code above
  PyObject *__reduce__() {
    if (debug)
      std::cout << "In C++ __reduce__" << std::endl;
    PyObject *args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, SWIG_From_std_string(self->msg));

    swig_type_info *ty = SWIGTYPE_p_PickleMe;
    SwigPyClientData *data = (SwigPyClientData *)ty->clientdata;
#if defined(SWIGPYTHON_BUILTIN)
    PyObject *callable = (PyObject *)data->pytype;
#else
    PyObject *callable = data->klass;
#endif
    Py_INCREF(callable);

    PyObject *ret = PyTuple_New(2);
    PyTuple_SetItem(ret, 0, callable);
    PyTuple_SetItem(ret, 1, args);
    return ret;
  }
#endif
}

%inline %{
#include <iostream>

bool debug = false;

struct PickleMe {
  std::string msg;
  PickleMe(const std::string& msg) : msg(msg) {
    if (debug)
      std::cout << "In C++ constructor " << " [" << msg << "]" << std::endl;
  }
};

struct NotForPickling {
  std::string msg;
  NotForPickling(const std::string& msg) : msg(msg) {}
};
%}
