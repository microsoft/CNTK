// valuewrapper.i
%module valuewrapper

%inline %{
template <typename T> struct X {
   X(int) {}
};
 
template <typename T> struct Y {
   Y() {}
   int spam(T t = T(0)) { return 0; }
};
%}
 
%template(Xi) X<int>;
%template(YXi) Y< X<int> >;

