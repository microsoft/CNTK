%module li_cdata

%include <cdata.i>

%cdata(int);
%cdata(double);

void *malloc(size_t size);
