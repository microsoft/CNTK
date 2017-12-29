%module template_tbase_template

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) traits<Double, Double>;	/* Ruby, wrong class name */

%warnfilter(SWIGWARN_PARSE_EXPLICIT_TEMPLATE);

%inline %{
  typedef double Double;

 
  template <class ArgType, class ResType>
  struct Funktion
  {
	char *test() { return (char *) "test"; }
  };

  template <class ArgType, class ResType>
  struct traits
  {
    typedef ArgType arg_type;
    typedef ResType res_type;
    typedef Funktion<ArgType, double> base;	
  };

  // Egad!
  template <class AF, class AG>
  struct Class_
    : Funktion<typename traits<AF, AG>::arg_type,
                                typename traits<AF, AG>::res_type>
  {
  };
 
  template <class AF, class RF>
  typename traits<AF, RF>::base
  make_Class()
  {
    return Class_<AF, RF>();
  }

%}
%{
  template struct Funktion <Double, Double>;
  template struct Class_ <Double, Double>; 
%}
 
%template(traits_dd) traits <Double, Double>;
%template(Funktion_dd) Funktion <Double, Double>;
%template(Class_dd) Class_ <Double, Double>;
%template(make_Class_dd) make_Class<Double,Double>;
