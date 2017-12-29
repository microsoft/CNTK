/* This interface file tests whether whitespace in angle brackets
   affects the SWIG types. SF Bug #221917, reported by
   burchanb@cs.tamu.edu. */

%module template_whitespace

%{
template<class T> class vector {
};
template<class T, class U> class map {
};
%}

//%typemap(in) vector<int> "$target = new vector<int>();";
//%typemap(in) vector<unsigned int> "$target = new vector<unsigned int>();";
//%typemap(in) map<int,int> "$target = new map<int, int>();";

%inline %{
void foo(vector<int > v) {}
void bar(vector<unsigned  int> v) {}
void baz(map < int , int > p) {}
%}
