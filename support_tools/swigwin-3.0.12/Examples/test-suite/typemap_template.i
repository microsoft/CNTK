%module typemap_template

/* Test bug in 1.3.40 where the presence of a generic/unspecialized typemap caused the incorrect specialized typemap to be used */

%typemap(in) SWIGTYPE "_this_will_not_compile_SWIGTYPE_ \"$type\" "
%typemap(in) const SWIGTYPE & "_this_will_not_compile_const_SWIGTYPE_REF_\"$type\" "

%typemap(in) const TemplateTest1 & {$1 = (TemplateTest1<YY> *)0; /* in typemap generic for $type */}
%typemap(in) const TemplateTest1< ZZ > & {$1 = (TemplateTest1<ZZ> *)0; /* in typemap ZZ for $type */}
%typemap(in) const TemplateTest1< int > & {$1 = (TemplateTest1<int> *)0; /* in typemap int for $type */}

%inline %{
template<typename T> struct TemplateTest1 {
  void setT(const TemplateTest1& t) {}
  typedef double Double;
};
%}

%inline %{
  struct YY {};
  struct ZZ {};
%}


%template(TTYY) TemplateTest1< YY >;
%template(TTZZ) TemplateTest1< ZZ >;
%template(TTint) TemplateTest1< int >;

%inline %{
  void extratest(const TemplateTest1< YY > &t, 
                 const TemplateTest1< ZZ > &tt,
                 const TemplateTest1< int > &ttt)
  {}
%}

%typemap(in) TemplateTest1 "_this_will_not_compile_TemplateTest_ \"$type\" "

%inline %{
  void wasbug(TemplateTest1< int >::Double wbug) {}
%}
