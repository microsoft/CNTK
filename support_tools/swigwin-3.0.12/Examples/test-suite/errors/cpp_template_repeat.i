%module xxx

template<class T> T blah(T x) { };

%template(iblah) blah<int>;
%template(iiblah) blah<int>;
// The second %template instantiation above should surely be ignored with a warning, but doesn't atm
