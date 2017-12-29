%module("templatereduce") li_std_multimap

%feature("trackobjects");

%include std_pair.i
%include std_multimap.i

%inline %{
struct A{
    int val;
    
    A(int v = 0): val(v)
    {
    }

};
%}

namespace std
{
  %template(pairA) pair<int, A*>;
  %template(multimapA) multimap<int, A*>;
}
