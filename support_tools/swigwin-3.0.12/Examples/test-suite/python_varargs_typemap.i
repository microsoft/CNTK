%module python_varargs_typemap

 /* The typemap and action are taken from the "Variable length arguments"
  * chapter of the SWIG manual.
  */

%typemap(in) (...)(char *vargs[10]) {
  int i;
  Py_ssize_t argc;
  for (i = 0; i < 10; i++) vargs[i] = 0;
  argc = PyTuple_Size(varargs);
  if (argc > 10) {
    PyErr_SetString(PyExc_ValueError, "Too many arguments");
    SWIG_fail;
  }
  for (i = 0; i < argc; i++) {
    PyObject *pyobj = PyTuple_GetItem(varargs, i);
    char *str = 0;
%#if PY_VERSION_HEX>=0x03000000
    PyObject *pystr;
    if (!PyUnicode_Check(pyobj)) {
       PyErr_SetString(PyExc_ValueError, "Expected a string");
       SWIG_fail;
    }
    pystr = PyUnicode_AsUTF8String(pyobj);
    str = strdup(PyBytes_AsString(pystr));
    Py_XDECREF(pystr);
%#else  
    if (!PyString_Check(pyobj)) {
       PyErr_SetString(PyExc_ValueError, "Expected a string");
       SWIG_fail;
    }
    str = PyString_AsString(pyobj);
%#endif
    vargs[i] = str;
  }
  $1 = (void *)vargs;
}

%feature("action") testfunc {
  char **vargs = (char **) arg3;
  result = testfunc(arg1, arg2, vargs[0], vargs[1], vargs[2], vargs[3], vargs[4],
                    vargs[5], vargs[6], vargs[7], vargs[8], vargs[9], NULL);
}

%typemap(freearg) (...) {
%#if PY_VERSION_HEX>=0x03000000
  int i;
  for (i = 0; i < 10; i++) {
    free(vargs$argnum[i]);
  }
%#endif
}

%inline {
char* testfunc (int arg1, double arg2, ...)
{
  va_list ap;
  char *c;
  static char buffer[1024];
  buffer[0] = 0;
  va_start(ap, arg2);
  while ((c = va_arg(ap, char *))) {
    strcat(buffer, c);
  }
  va_end(ap);
  return buffer;
}
}

%inline %{
char *doublecheck(char *inputval) { return inputval; }
%}

