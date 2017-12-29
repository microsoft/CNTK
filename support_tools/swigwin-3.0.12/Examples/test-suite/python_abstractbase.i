%module python_abstractbase
%include <pyabc.i>
%include <std_map.i>
%include <std_multimap.i>
%include <std_set.i>
%include <std_multiset.i>
%include <std_list.i>
%include <std_vector.i>

namespace std
{
  %template(Mapii) map<int, int>;
  %template(Multimapii) multimap<int, int>;
  %template(IntSet) set<int>;
  %template(IntMultiset) multiset<int>;
  %template(IntVector) vector<int>;
  %template(IntList) list<int>;
}

%inline %{
#ifdef SWIGPYTHON_BUILTIN
bool is_python_builtin() { return true; }
#else
bool is_python_builtin() { return false; }
#endif
%}

#ifdef SWIGPYTHON_PY3 // set when using -py3
#define is_swig_py3 1
#else
#define is_swig_py3 0
#endif
