%module xxx

template<typename T> T blah(T x) { };

%template(blahi) blah<int,double>;
%template(blahf) blah<>;




