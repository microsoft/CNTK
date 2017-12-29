/* File : example.i */
%module example

%{
#include <unistd.h>
#include <ffi.h>
%}

/* A wrapper for execlp() using libffi to handle an arbitrary
   number of arguments */

%typemap(in) (...) {
   char **argv;
   int    argc;
   int    i;

   argc = PyTuple_Size(varargs);
   argv = (char **) malloc(sizeof(char *)*(argc+1));
   for (i = 0; i < argc; i++) {
      PyObject *o = PyTuple_GetItem(varargs,i);
      if (!PyString_Check(o)) {
          PyErr_SetString(PyExc_ValueError,"Expected a string");
          SWIG_fail;
      }
      argv[i] = PyString_AsString(o);
   }
   argv[i] = NULL;
   $1 = (void *) argv;
}

/* Rewrite the function call, using libffi */    
%feature("action") execlp {
  int       i, vc;
  ffi_cif   cif;
  ffi_type  **types;
  void      **values;
  char      **args;

  vc = PyTuple_Size(varargs);
  types  = (ffi_type **) malloc((vc+3)*sizeof(ffi_type *));
  values = (void **) malloc((vc+3)*sizeof(void *));
  args   = (char **) arg3;

  /* Set up path parameter */
  types[0] = &ffi_type_pointer;
  values[0] = &arg1;
  
  /* Set up first argument */
  types[1] = &ffi_type_pointer;
  values[1] = &arg2;

  /* Set up rest of parameters */
  for (i = 0; i <= vc; i++) {
    types[2+i] = &ffi_type_pointer;
    values[2+i] = &args[i];
  }
  if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, vc+3,
                   &ffi_type_uint, types) == FFI_OK) {
    ffi_call(&cif, (void (*)()) execlp, &result, values);
  } else {
    free(types);
    free(values);
    free(arg3);
    PyErr_SetString(PyExc_RuntimeError, "Whoa!!!!!");
    SWIG_fail;
  }
  free(types);
  free(values);
  free(arg3);
}

int execlp(const char *path, const char *arg1, ...);


/* A wrapper for printf() using libffi */

%{
  typedef struct {
    int type;
    union {
      int    ivalue;
      double dvalue;
      void   *pvalue;
    } val;
  } vtype;
  enum { VT_INT, VT_DOUBLE, VT_POINTER };
  %}

%typemap(in) (const char *fmt, ...) {
  vtype *argv;
  int    argc;
  int    i;

  $1 = PyString_AsString($input);

  argc = PyTuple_Size(varargs);
  argv = (vtype *) malloc(argc*sizeof(vtype));
  for (i = 0; i < argc; i++) {
    PyObject *o = PyTuple_GetItem(varargs,i);
    if (PyInt_Check(o)) {
      argv[i].type = VT_INT;
      argv[i].val.ivalue = PyInt_AsLong(o);
    } else if (PyFloat_Check(o)) {
      argv[i].type = VT_DOUBLE;
      argv[i].val.dvalue = PyFloat_AsDouble(o);
    } else if (PyString_Check(o)) {
      argv[i].type = VT_POINTER;
      argv[i].val.pvalue = (void *) PyString_AsString(o);
    } else {
      free(argv);
      PyErr_SetString(PyExc_ValueError,"Unsupported argument type");
      SWIG_fail;
    }
  }

  $2 = (void *) argv;
}

/* Rewrite the function call, using libffi */    
%feature("action") printf {
  int       i, vc;
  ffi_cif   cif;
  ffi_type  **types;
  void      **values;
  vtype     *args;

  vc = PyTuple_Size(varargs);
  types  = (ffi_type **) malloc((vc+1)*sizeof(ffi_type *));
  values = (void **) malloc((vc+1)*sizeof(void *));
  args   = (vtype *) arg2;

  /* Set up fmt parameter */
  types[0] = &ffi_type_pointer;
  values[0] = &arg1;

  /* Set up rest of parameters */
  for (i = 0; i < vc; i++) {
    switch(args[i].type) {
    case VT_INT:
      types[1+i] = &ffi_type_uint;
      values[1+i] = &args[i].val.ivalue;
      break;
    case VT_DOUBLE:
      types[1+i] = &ffi_type_double;
      values[1+i] = &args[i].val.dvalue;
      break;
    case VT_POINTER:
      types[1+i] = &ffi_type_pointer;
      values[1+i] = &args[i].val.pvalue;
      break;
    default:
      abort();    /* Whoa! We're seriously hosed */
      break;   
    }
  }
  if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, vc+1,
                   &ffi_type_uint, types) == FFI_OK) {
    ffi_call(&cif, (void (*)()) printf, &result, values);
  } else {
    free(types);
    free(values);
    free(args);
    PyErr_SetString(PyExc_RuntimeError, "Whoa!!!!!");
    SWIG_fail;
  }
  free(types);
  free(values);
  free(args);
}

int printf(const char *fmt, ...);


  


