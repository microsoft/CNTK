%module langobj


#ifndef SWIG_Object
#define SWIG_Object void *
#endif


%inline %{

#ifdef SWIGTCL
#define SWIG_Object Tcl_Obj *
#endif

#ifdef SWIGPYTHON
#define SWIG_Object PyObject *
#endif

#ifdef SWIGRUBY
#define SWIG_Object VALUE
#endif

#ifndef SWIG_Object
#define SWIG_Object void *
#endif

%}


%inline {

  SWIG_Object identity(SWIG_Object x) {
#ifdef SWIGPYTHON
    Py_XINCREF(x);
#endif
    return x;    
  }

}

  
