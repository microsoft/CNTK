%module template_base_template

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) traits<double, double>; /* Ruby, wrong class name */

%warnfilter(SWIGWARN_PARSE_EXPLICIT_TEMPLATE);

%inline %{
  template <class ArgType, class ResType>
  struct traits
  {
    typedef ArgType arg_type;
    typedef ResType res_type;
  };
 
  template <class ArgType, class ResType>
  struct Funktion
  {
  };

  // Egad!
  template <class AF, class AG>
  struct Klass
    : Funktion<typename traits<AF, AG>::arg_type,
                                typename traits<AF, AG>::res_type>
  {
  };
%}

%{
 template struct Funktion <double, double>;
 template struct Klass <double, double>;
%}
 
%template(traits_dd) traits <double, double>;
%template(Funktion_dd) Funktion <double, double>;
%template(Klass_dd) Klass <double, double>;





