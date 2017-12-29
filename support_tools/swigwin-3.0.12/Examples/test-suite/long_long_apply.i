/* This interface file tests whether SWIG handles the Microsoft __int64 type through the use of 
applying the long long typemaps. The generated code should not have any instances of long long. */

%module long_long_apply

%{
#ifdef _MSC_VER /* Visual C++ */
    typedef __int64 LongLong;
    typedef unsigned __int64 UnsignedLongLong;
#else
    typedef long long LongLong;
    typedef unsigned long long UnsignedLongLong;
#endif
%}

%apply long long { LongLong };
%apply unsigned long long { UnsignedLongLong };

%apply const long long & { const LongLong & };
%apply const unsigned long long & { const UnsignedLongLong & };

%inline %{
// pass by value
LongLong value1(LongLong x) { return x; }
UnsignedLongLong value2(UnsignedLongLong x) { return x; }

// pass by reference
const LongLong &ref1(const LongLong &x) { return x; }
const UnsignedLongLong &ref2(const UnsignedLongLong &x) { return x; }

// global variables
LongLong global1;
UnsignedLongLong global2;

// global reference variables
const LongLong& global_ref1 = global1;
const UnsignedLongLong& global_ref2 = global2;
%}


// typemaps library
%include "typemaps.i"
%apply long long *INPUT { LongLong *INPUT };
%apply unsigned long long *INPUT { UnsignedLongLong *INPUT };

%apply long long *OUTPUT { LongLong *OUTPUT };
%apply unsigned long long *OUTPUT { UnsignedLongLong *OUTPUT };

%apply long long *INOUT { LongLong *INOUT };
%apply unsigned long long *INOUT { UnsignedLongLong *INOUT };

%apply long long &INPUT { LongLong &INPUT };
%apply unsigned long long &INPUT { UnsignedLongLong &INPUT };

%apply long long &OUTPUT { LongLong &OUTPUT };
%apply unsigned long long &OUTPUT { UnsignedLongLong &OUTPUT };

%apply long long &INOUT { LongLong &INOUT };
%apply unsigned long long &INOUT { UnsignedLongLong &INOUT };

%inline %{
LongLong in_longlong(LongLong *INPUT) { return *INPUT; }
UnsignedLongLong in_ulonglong(UnsignedLongLong *INPUT) { return *INPUT; }
LongLong inr_longlong(LongLong &INPUT) { return INPUT; }
UnsignedLongLong inr_ulonglong(UnsignedLongLong &INPUT) { return INPUT; }

void out_longlong(LongLong x, LongLong *OUTPUT) {  *OUTPUT = x; }
void out_ulonglong(UnsignedLongLong x, UnsignedLongLong *OUTPUT) {  *OUTPUT = x; }
void outr_longlong(LongLong x, LongLong &OUTPUT) {  OUTPUT = x; }
void outr_ulonglong(UnsignedLongLong x, UnsignedLongLong &OUTPUT) {  OUTPUT = x; }

void inout_longlong(LongLong *INOUT) {  *INOUT = *INOUT; }
void inout_ulonglong(UnsignedLongLong *INOUT) {  *INOUT = *INOUT; }
void inoutr_longlong(LongLong &INOUT) {  INOUT = INOUT; }
void inoutr_ulonglong(UnsignedLongLong &INOUT) {  INOUT = INOUT; }
%}

