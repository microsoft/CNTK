%module li_math
#ifdef SWIGPHP
// PHP already provides these functions with the same names, so just kill that
// warning.
%warnfilter(SWIGWARN_PARSE_KEYWORD);
#endif
%include math.i
