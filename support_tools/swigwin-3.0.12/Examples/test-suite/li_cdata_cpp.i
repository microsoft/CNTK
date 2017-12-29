%module li_cdata_cpp

%include <cdata.i>

%cdata(int);
%cdata(double);

void *malloc(size_t size);
