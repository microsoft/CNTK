%module funcptr_cpp

%{
#if defined(__SUNPRO_CC)
#pragma error_messages (off, badargtype2w) /* Formal argument ... is being passed extern "C" ... */
#endif
%}

%inline %{

int addByValue(const int &a, int b) { return a+b; }
int * addByPointer(const int &a, int b) { static int val; val = a+b; return &val; }
int & addByReference(const int &a, int b) { static int val; val = a+b; return val; }

int call1(int (*d)(const int &, int), int a, int b) { return d(a, b); }
int call2(int * (*d)(const int &, int), int a, int b) { return *d(a, b); }
int call3(int & (*d)(const int &, int), int a, int b) { return d(a, b); }
%}

%constant int (*ADD_BY_VALUE)(const int &, int) = addByValue;
%constant int * (*ADD_BY_POINTER)(const int &, int) = addByPointer;
%constant int & (*ADD_BY_REFERENCE)(const int &, int) = addByReference;
%constant int (* const ADD_BY_VALUE_C)(const int &, int) = addByValue;

%inline %{
typedef int AddByValueTypedef(const int &a, int b);
typedef int * AddByPointerTypedef(const int &a, int b);
typedef int & AddByReferenceTypedef(const int &a, int b);
void *typedef_call1(AddByValueTypedef *& precallback, AddByValueTypedef * postcallback) { return 0; }
void *typedef_call2(AddByPointerTypedef *& precallback, AddByPointerTypedef * postcallback) { return 0; }
void *typedef_call3(AddByReferenceTypedef *& precallback, AddByReferenceTypedef * postcallback) { return 0; }
%}

