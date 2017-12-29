use Inline SWIG => <<"END_CODE", SWIG_ARGS => '-c++ -proxy', CC => 'g++', LD=>'g++';
  class Foo {
  public:
    int meaning() { return 42; };
  };
END_CODE

my $o = new Foo();
print $o->meaning(),"\n";

use Inline SWIG => ' ', SWIG_INTERFACE => <<"END_CODE", SWIG_ARGS => '-c++', CC => "g++", LD => "g++";
%include std_string.i
%inline {
  template <class Type>
  class Bar {
    Type _val;
  public: 
    Bar(Type v) : _val(v) {}
    Type meaning() { return _val; }  
  };
}
%template(Bar_i) Bar<int>;
%template(Bar_d) Bar<double>;
%template(Bar_s) Bar<std::string>;
END_CODE

my $o = new Bar_i(1);
print $o->meaning(),"\n";

my $o = new Bar_d(2);
print $o->meaning(),"\n";

my $o = new Bar_s("hello");
print $o->meaning(),"\n";
