%module scilab_pointer_conversion_functions

%warnfilter(SWIGWARN_TYPEMAP_SWIGTYPELEAK_MSG) pfoo; /* Setting a pointer/reference variable may leak memory. */

%inline %{

void *getNull() { return NULL; }
bool isNull(void *p) { return p == NULL; }

int foo = 3;
int *pfoo = &foo;

double getFooAddress() { return (double) (unsigned long) pfoo; }
bool equalFooPointer(void *p) { return p == pfoo; }

%}

%typemap(out, noblock=1) struct structA* {
  if (SwigScilabPtrFromObject(pvApiCtx, SWIG_Scilab_GetOutputPosition(), $1, SWIG_Scilab_TypeQuery("struct structA *"), 0, NULL) != SWIG_OK) {
    return SWIG_ERROR;
  }
  SWIG_Scilab_SetOutput(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + SWIG_Scilab_GetOutputPosition());
}

%inline %{

struct structA {
  int x;
};

%}
