/* File : example.i */
%module Example

%{
#include "example.h"
%}

%insert(m3wrapintf) %{
EXCEPTION E(Error);
%}
%insert(m3wrapimpl) %{
IMPORT Ctypes AS C;
%}

%pragma(modula3) enumitem="enum=error;int;srcstyle=underscore;Error";

%typemap("m3rawintype")   double & %{C.double%};
%typemap("m3wrapintype")  double & %{LONGREAL%};

%typemap("m3wraprettype") error ""
%typemap("m3wrapretvar") error "rawerr: C.int;"
%typemap("m3wrapretraw")  error "rawerr"
%typemap("m3wrapretcheck:throws") error "E"
%typemap("m3wrapretcheck") error
%{VAR err := VAL(rawerr, Error);
BEGIN
IF err # Error.ok THEN
RAISE E(err);
END;
END;%}

%typemap("m3rawintype")              errorstate & %{C.int%};
%typemap("m3wrapintype",numinputs=0) errorstate & %{%};
%typemap("m3wrapargvar")             errorstate & %{err:C.int:=ORD(Error.ok);%};
%typemap("m3wrapoutcheck:throws")    errorstate & "E";
%typemap("m3wrapoutcheck")           errorstate &
%{IF VAL(err,Error) # Error.ok THEN
RAISE E(VAL(err,Error));
END;%}

/* Let's just grab the original header file here */

%include "example.h"
