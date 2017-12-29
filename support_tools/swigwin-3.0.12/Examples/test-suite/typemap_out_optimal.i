// Test the optimal attribute in the out typemap
%module typemap_out_optimal

// Just the following languages tested
#if defined (SWIGCSHARP) || defined (SWIGD)
%typemap(out, optimal="1") SWIGTYPE %{
  $result = new $1_ltype((const $1_ltype &)$1);
%}
#elif defined (SWIGJAVA)
%typemap(out, optimal="1") SWIGTYPE %{ 
  *($&1_ltype*)&$result = new $1_ltype((const $1_ltype &)$1);
%}
#elif defined (SWIGUTL)
%typemap(out,noblock="1", optimal="1") SWIGTYPE {
  %set_output(SWIG_NewPointerObj(%new_copy($1, $ltype), $&descriptor, SWIG_POINTER_OWN | %newpointer_flags));
}
#endif

%ignore XX::operator=;

#ifdef SWIGD
%rename(trace) XX::debug;
#endif

%inline %{
#include <iostream>
using namespace std;

struct XX {
  XX() { if (debug) cout << "XX()" << endl; }
  XX(int i) { if (debug) cout << "XX(" << i << ")" << endl; }
  XX(const XX &other) { if (debug) cout << "XX(const XX &)" << endl; }
  XX& operator =(const XX &other) { if (debug) cout << "operator=(const XX &)" << endl; return *this; }
  ~XX() { if (debug) cout << "~XX()" << endl; }
  static XX create() { 
    return XX(123);
  }
  static bool debug;
};
bool XX::debug = true;
%}

