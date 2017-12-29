%module template_specialization_enum

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Hello;	/* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Hi;	/* Ruby, wrong class name */

%warnfilter(SWIGWARN_GO_NAME_CONFLICT);                       /* Ignoring 'hello due to Go name ('Hello) conflict with 'Hello' */

%inline %{

  enum Hello 
  {
    hi, hello
  };
  

  template <Hello, class A>
    struct C
    {
    };
  

  template <Hello, class BB>
    struct Base
    {
    };  
  
  
  template <class A>
    struct C<hello , A>  : Base<hello, A>
    {
      int fhello()
      {
	return hello;
      }
      
    protected:
      C()
      {
      }
    };
  

  template <class A>
    struct C<hi , A> : Base<hi, A>
    {
      int fhi()
      {
	return hi;
      }

    protected:
      C()
      {
      }
    };
  
      
%}

%template(Base_dd) Base<hi, int>;
%template(Base_ii) Base<hello, int>;

%template(C_i) C<hi, int>;
%template(C_d) C<hello, int>;
