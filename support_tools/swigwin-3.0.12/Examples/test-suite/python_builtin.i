// Test customizing slots when using the -builtin option

%module python_builtin

%{
#if defined(_MSC_VER)
  #pragma warning(disable: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif
%}

%inline %{
#ifdef SWIGPYTHON_BUILTIN
bool is_python_builtin() { return true; }
#else
bool is_python_builtin() { return false; }
#endif
%}

// Test 0 for default tp_hash
%inline %{
struct ValueStruct {
  int value;
  ValueStruct(int value) : value(value) {}
  static ValueStruct *inout(ValueStruct *v) {
    return v;
  }
};
%}

// Test 1a for tp_hash
#if defined(SWIGPYTHON_BUILTIN)
%feature("python:tp_hash") SimpleValue "SimpleValueHashFunction"
#endif

%inline %{
struct SimpleValue {
  int value;
  SimpleValue(int value) : value(value) {}
};
%}

%{
#if PY_VERSION_HEX >= 0x03020000
Py_hash_t SimpleValueHashFunction(PyObject *v)
#else
long SimpleValueHashFunction(PyObject *v)
#endif
{
  SwigPyObject *sobj = (SwigPyObject *) v;
  SimpleValue *p = (SimpleValue *)sobj->ptr;
  return p->value;
}
hashfunc test_hashfunc_cast() {
    return SimpleValueHashFunction;
}
%}

// Test 1b for tp_hash
#if defined(SWIGPYTHON_BUILTIN)
%feature("python:slot", "tp_hash", functype="hashfunc") SimpleValue2::HashFunc;
#endif

%inline %{
struct SimpleValue2 {
  int value;
  SimpleValue2(int value) : value(value) {}
#if PY_VERSION_HEX >= 0x03020000
  typedef Py_hash_t HashType;
#else
  typedef long HashType;
#endif
  HashType HashFunc() { return (HashType)value; }
};
%}

// Test 2 for tp_hash
#if defined(SWIGPYTHON_BUILTIN)
%feature("python:slot", "tp_hash", functype="hashfunc") BadHashFunctionReturnType::bad_hash_function;
#endif

%inline %{
struct BadHashFunctionReturnType {
    static const char * bad_hash_function() {
      return "bad hash function";
    }
};
%}

// Test 3 for tp_hash
#if defined(SWIGPYTHON_BUILTIN)
%feature("python:slot", "tp_hash", functype="hashfunc") ExceptionHashFunction::exception_hash_function;
#endif

%catches(const char *) exception_hash_function;

%inline %{
#if PY_VERSION_HEX < 0x03020000
  #define Py_hash_t long
#endif
struct ExceptionHashFunction {
    static Py_hash_t exception_hash_function() {
      throw "oops";
    }
};
%}

// Test 4 for tp_dealloc (which is handled differently to other slots in the SWIG source)
#if defined(SWIGPYTHON_BUILTIN)
%feature("python:tp_dealloc") Dealloc1 "Dealloc1Destroyer"
%feature("python:tp_dealloc") Dealloc2 "Dealloc2Destroyer"
%feature("python:slot", "tp_dealloc", functype="destructor") Dealloc3::Destroyer;
#endif

%inline %{
static int Dealloc1CalledCount = 0;
static int Dealloc2CalledCount = 0;
static int Dealloc3CalledCount = 0;

struct Dealloc1 {
};
struct Dealloc2 {
  ~Dealloc2() {}
};
struct Dealloc3 {
  void Destroyer() {
    Dealloc3CalledCount++;
    delete this;
  }
};
%}

%{
void Dealloc1Destroyer(PyObject *v) {
  SwigPyObject *sobj = (SwigPyObject *) v;
  Dealloc1 *p = (Dealloc1 *)sobj->ptr;
  delete p;
  Dealloc1CalledCount++;
}
void Dealloc2Destroyer(PyObject *v) {
  SwigPyObject *sobj = (SwigPyObject *) v;
  Dealloc2 *p = (Dealloc2 *)sobj->ptr;
  delete p;
  Dealloc2CalledCount++;
}
%}

// Test 5 for python:compare feature
%feature("python:compare", "Py_LT") MyClass::lessThan;

%inline %{
  class MyClass {
  public:
    MyClass(int val = 0) : val(val) {}
    bool lessThan(const MyClass& other) const {
      less_than_counts++;
      return val < other.val;
    }
    int val;
    static int less_than_counts;
  };
  int MyClass::less_than_counts = 0;
%}

// Test 6 add in container __getitem__ to support basic sequence protocol
// Tests overloaded functions being used for more than one slot (mp_subscript and sq_item)
%include <exception.i>
%include <std_except.i>
%apply int {Py_ssize_t}
%typemap(in) PySliceObject * {
  if (!PySlice_Check($input))
    SWIG_exception(SWIG_TypeError, "in method '$symname', argument $argnum of type '$type'");
  $1 = (PySliceObject *)$input;
}
%typemap(typecheck,precedence=300) PySliceObject* {
  $1 = PySlice_Check($input);
}

%feature("python:slot", "mp_subscript", functype="binaryfunc") SimpleArray::__getitem__(PySliceObject *slice);
%feature("python:slot", "sq_item", functype="ssizeargfunc") SimpleArray::__getitem__(Py_ssize_t n);
%feature("python:slot", "sq_length", functype="lenfunc") SimpleArray::__len__;
%inline %{
  class SimpleArray {
    Py_ssize_t size;
    int numbers[5];
  public:
    SimpleArray(Py_ssize_t size) : size(size) {
      for (Py_ssize_t x = 0; x<size; ++x)
        numbers[x] = (int)x*10;
    }

    Py_ssize_t __len__() {
      return size;
    }

    int __getitem__(Py_ssize_t n) throw (std::out_of_range) {
      if (n >= (int)size)
        throw std::out_of_range("Index too large");
      return numbers[n];
    }

    SimpleArray __getitem__(PySliceObject *slice) throw (std::out_of_range, std::invalid_argument) {
      if (!PySlice_Check(slice))
        throw std::invalid_argument("Slice object expected");
      Py_ssize_t i, j, step;
#if PY_VERSION_HEX >= 0x03020000
      PySlice_GetIndices((PyObject *)slice, size, &i, &j, &step);
#else
      PySlice_GetIndices((PySliceObject *)slice, size, &i, &j, &step);
#endif
      if (step != 1)
        throw std::invalid_argument("Only a step size of 1 is implemented");

      {
        Py_ssize_t ii = i<0 ? 0 : i>=size ? size-1 : i;
        Py_ssize_t jj = j<0 ? 0 : j>=size ? size-1 : j;
        if (ii > jj)
          throw std::invalid_argument("getitem i should not be larger than j");
        SimpleArray n(jj-ii);
        for (Py_ssize_t x = 0; x<size; ++x)
          n.numbers[x] = numbers[x+ii];
        return n;
      }
    }
  };
%}
