%module li_typemaps_apply

// Test %apply to global primitive type references/pointers to make sure the return types are still okay... mainly for the strongly typed languages.

%include "typemaps.i"

#if !defined(SWIGJAVA) // Needs asymmetric type marshalling support for this testcase to work

%define TMAPS(PRIMTYPE, NAME)
%apply PRIMTYPE *INPUT { PRIMTYPE * }
%apply PRIMTYPE &INPUT { PRIMTYPE & }
%inline %{
PRIMTYPE *input_ptr_##NAME(PRIMTYPE *v) { static PRIMTYPE stat; stat = *v; return &stat; }
PRIMTYPE &input_ref_##NAME(PRIMTYPE &v) { static PRIMTYPE stat; stat = v; return stat; }
%}

%apply PRIMTYPE *OUTPUT { PRIMTYPE * }
%apply PRIMTYPE &OUTPUT { PRIMTYPE & }
%inline %{
PRIMTYPE *output_ptr_##NAME(PRIMTYPE x, PRIMTYPE *v) { static PRIMTYPE stat; stat = x; *v = x; return &stat; }
PRIMTYPE &output_ref_##NAME(PRIMTYPE x, PRIMTYPE &v) { static PRIMTYPE stat; stat = x; v = x; return stat; }
%}

%apply PRIMTYPE *INOUT { PRIMTYPE * }
%apply PRIMTYPE &INOUT { PRIMTYPE & }
%inline %{
PRIMTYPE *inout_ptr_##NAME(PRIMTYPE *v) { static PRIMTYPE stat; stat = *v; *v = *v; return &stat; }
PRIMTYPE &inout_ref_##NAME(PRIMTYPE &v) { static PRIMTYPE stat; stat = v; v = v; return stat; }
%}
%enddef

TMAPS(bool,               bool)
TMAPS(int,                int)
TMAPS(short,              short)
TMAPS(long,               long)
TMAPS(unsigned int,       uint)
TMAPS(unsigned short,     ushort)
TMAPS(unsigned long,      ulong)
TMAPS(unsigned char,      uchar)
TMAPS(signed char,        schar)
TMAPS(float,              float)
TMAPS(double,             double)
TMAPS(long long,          longlong)
TMAPS(unsigned long long, ulonglong)

#endif
