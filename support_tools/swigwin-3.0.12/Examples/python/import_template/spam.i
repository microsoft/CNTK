%module spam
%{
#include "spam.h"
%}

%import bar.i
%include "spam.h"

%template(intSpam) Spam<int>;

