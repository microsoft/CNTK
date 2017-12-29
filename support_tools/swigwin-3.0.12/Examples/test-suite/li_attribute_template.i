%module li_attribute_template

%include <exception.i>

//#define SWIG_ATTRIBUTE_TEMPLATE
%include <attribute.i>
%include <std_string.i>

%inline
{
  class Foo {
  public:
        Foo( int _value ) { value = _value; }
        int value;
  };

  template< class T1, class T2>
  struct pair{
     pair( T1 t1, T2 t2 ):
        first(t1), second(t2) {;}

     T1 first;
     T2 second;
  };

  template< class T1, class T2>
  struct C
  {
    C(int a, int b, int c) :
        a(a), b(b), c(c), d(a), _e(b),
        _f(a,b), _g(b,c)
    {

/*
        _f.first = a;
        _f.second = b;

        _g.first = b;
        _g.second = c;
*/

    }

    int get_value() const
    {
      return a;
    }

    void set_value(int aa)
    {
      a = aa;
    }

    /* only one ref method */
    int& get_ref()
    {
      return b;
    }

    Foo get_class_value() const { return d; }
    void set_class_value( Foo foo) { d = foo; }

    const Foo& get_class_ref() const { return _e; }
    void set_class_ref( const Foo& foo ) { _e = foo; }

    pair<T1,T2> get_template_value() const { return _f; }
    void set_template_value( const pair<T1,T2> f ) { _f = f; }

    const pair<T1,T2>& get_template_ref() const { return _g; }
    void set_template_ref( const pair<T1,T2>& g ) {  _g = g; }

    std::string get_string() { return str; }
    void set_string(std::string other) { str = other; }

  private:
    int a;
    int b;
    int c;
    Foo d;
    Foo _e;
    pair<T1,T2> _f;
    pair<T1,T2> _g;

    std::string str;
  };

}

%define %instantiate_C( T1, T2 )
%template (pair_ ## T1 ## T2 ) pair<T1,T2>;
// Primitive types
%attribute( %arg(C<T1,T2>), int, a, get_value, set_value );
%attributeref( %arg(C<T1,T2>), int, b, get_ref );

// Strings
%attributestring(%arg(C<T1,T2>), std::string, str, get_string, set_string);

// Class types
%attributeval( %arg(C<T1,T2>), Foo, d, get_class_value, set_class_value  );
%attribute2( %arg(C<T1,T2>), Foo, e, get_class_ref, set_class_ref  );

// Moderately templated types
%attributeval( %arg(C<T1,T2>), %arg(pair<T1,T2>), f, get_template_value, set_template_value );
%attribute2( %arg(C<T1,T2>), %arg(pair<T1,T2>), g, get_template_ref, set_template_ref );

%template (C ## T1 ## T2) C<T1,T2>;
%enddef


%instantiate_C(int,int);
