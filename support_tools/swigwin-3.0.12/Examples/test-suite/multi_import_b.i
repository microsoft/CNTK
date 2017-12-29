%module multi_import_b

%include multi_import_c.i

%{
#include "multi_import.h"
%}

class YYY : public XXX
{
public:
        int testy();
};
