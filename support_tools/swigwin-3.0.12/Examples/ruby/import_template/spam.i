%module spam
%{
#include "spam.h"
%}

%import bar.i
%include "spam.h"

%template(IntSpam) Spam<int>;

