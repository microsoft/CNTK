%module features

%warnfilter(SWIGWARN_LANG_IDENTIFIER,SWIGWARN_IGNORE_OPERATOR_PLUSEQ);

// This testcase checks that %feature is working for templates and non user supplied constructors/destructors and is just generally working

// If the default %exception is used it will not compile. It shouldn't get used.
%exception "this_will_not_compile";

// Test 1: Test for no user supplied constructors and destructor
%exception Simple::Simple(const Simple&) "$action /*Simple::Simple*/";
%exception Simple::Simple() "$action /*Simple::Simple*/";
%exception Simple::~Simple() "$action /*Simple::~Simple*/";

%inline %{
class Simple {};
%}


%exception NS::SimpleNS::SimpleNS(const NS::SimpleNS&) "$action /*NS::SimpleNS::SimpleNS*/";
%exception NS::SimpleNS::SimpleNS() "$action /*NS::SimpleNS::SimpleNS*/";
%exception NS::SimpleNS::~SimpleNS() "$action /*NS::SimpleNS::~SimpleNS*/";
// method tests
%exception NS::SimpleNS::method()       "_failed_ /*NS::Simple::method() const*/";
%exception NS::SimpleNS::method() const "$action /*NS::Simple::method() const*/";
%exception NS::SimpleNS::afunction() "$action /*NS::Simple::afunction()*/";

%inline %{
  namespace NS 
  {
    
    class SimpleNS {
    public:
      void method() const {}
      void afunction() {}
    };
  }
  
%}

// Test 2: Test templated functions
%exception foobar "caca";
%exception foobar<int>(int) "$action /*foobar<int>*/";

%inline %{
template<class T> void foobar(T t) {}
%}

%template(FooBarInt) foobar<int>;

// Test 3: Test templates with no user supplied constructors and destructor
%exception SimpleTemplate<int>::SimpleTemplate(const SimpleTemplate<int>&) "$action /*SimpleTemplate<int>::SimpleTemplate<int>*/";
%exception SimpleTemplate<int>::SimpleTemplate() "$action /*SimpleTemplate<int>::SimpleTemplate<int>*/";
%exception SimpleTemplate<int>::~SimpleTemplate() "$action /*SimpleTemplate<int>::~SimpleTemplate*/";

%inline %{
template<class T> class SimpleTemplate {
 public:
};
 
%}

%template(SimpleInt) SimpleTemplate<int>;

// Test 4: Test templates with user supplied constructors and destructor
%exception Template<int>::Template() "$action /*Template<int>::Template<int>*/";
%exception Template<int>::Template(const Template&) "$action /*Template<int>::Template<int>(const Template&)*/";
%exception Template<int>::~Template() "$action /*Template<int>::~Template*/";
// method tests
%exception Template<int>::foo "$action /*Template<int>::foo*/";
%exception Template::get "$action /*Template<int>::get*/";
%exception Template<int>::set(const int &t) "$action /*Template<int>::set(const int &t)*/";
%exception Template<int>::bar(const int &t)       "_failed_ /*Template<int>::bar(const int &t) const*/";
%exception Template<int>::bar(const int &t) const "$action /*Template<int>::bar(const int &t) const*/";

%inline %{
template<class T> class Template {
public:
  Template(){}

  Template(const Template&){}
  ~Template(){}
  void foo(){}
  void bar(const int &t) const {}
#ifdef SWIG
    %extend {
      T& get(int i) const {
        throw 1;
      }
      void set(const T &t) {}
    }
#endif
};
%}

%template(TemplateInt) Template<int>; 

// Test 5: wildcards
%exception Space::WildCards::WildCards(const Space::WildCards&) "$action /* Space::WildCards::WildCards() */";
%exception Space::WildCards::WildCards() "$action /* Space::WildCards::WildCards() */";
%exception Space::WildCards::~WildCards() "$action /* Space::WildCards::WildCards() */";
%exception *::incy              "_failure_ /* *::incy */";
%exception *::incy(int a)       "_failure_ /* *::incy(int a) */";
%exception *::incy(int a) const "$action /* *::incy(int a) const */";
%exception *::wincy(int a) "$action /* *::wincy(int a) */";
%exception *::spider "$action /* *::spider */";
%exception *::spider(int a) "_failure_ /* *::spider(int a)  */";

%inline %{
namespace Space {
  struct WildCards {
    virtual ~WildCards() {}
    virtual WildCards* incy(int a) const { return 0; }
    virtual WildCards* wincy(int a) { return 0; }
    virtual WildCards* spider(int a) const { return 0; }
  };
}
%}

// Test 6: default arguments
%exception Space::Animals::Animals(const Space::Animals&) "$action /* Space::Animals::Animals(int a = 0, double d = 0.0) */";
%exception Space::Animals::Animals(int a = 0, double d = 0.0) "$action /* Space::Animals::Animals(int a = 0, double d = 0.0) */";
%exception Space::Animals::~Animals() "$action /* Space::Animals::~Animals() */";
%exception Space::Animals::lions(int a = 0, double d = 0.0) const "$action /* Space::Animals::lions(int a = 0, double d = 0.0) const */";
%exception Space::Animals::leopards(int a = 0, double d = 0.0) "$action /* Space::Animals::leopards(int a = 0, double d = 0.0) */";
%exception *::cheetahs(int a = 0, double d = 0.0) const "$action /* *::cheetahs(int a = 0, double d = 0.0) const */";
%exception *::jackal(int a = 0, double d = 0.0) "$action /* *::jackal(int a = 0, double d = 0.0) */";
%inline %{
namespace Space {
  struct Animals {
    Animals(int a = 0, double d = 0.0) {}
    void* lions(int a = 0, double d = 0.0) const { return 0; }
    void* leopards(int a = 0, double d = 0.0) { return 0; }
    int cheetahs(int a = 0, double d = 0.0) const { return 0; }
    int jackal(int a = 0, double d = 0.0) { return 0; }
  };
}
%}

// Test 7: inheritance
%exception Space::Base::Base(const Space::Base&) "$action /* Space::Base::Base() */";
%exception Space::Base::Base() "$action /* Space::Base::Base() */";
%exception Space::Base::~Base() "$action /* Space::Base::~Base() */";
%exception Space::Derived::Derived(const Space::Derived&) "$action /* Space::Derived::Derived() */";
%exception Space::Derived::Derived() "$action /* Space::Derived::Derived() */";
%exception Space::Derived::~Derived() "$action /* Space::Derived::~Derived() */";
// The following should apply to both Base and Derived
%exception Space::Base::virtualmethod(int a) const "$action /* Space::Base::virtualmethod(int a) const */";

%exception Space::Base::operator+=(int) "$action /* Space::Base::Base() */";

%inline %{
namespace Space {
  struct Base {
    int operator+=(int) { return 0; }    
    virtual const char** virtualmethod(int a) const { return 0; }
    virtual ~Base() {}
  };
  struct Derived : Base {
    virtual const char** virtualmethod(int a) const { return 0; }
  };
}
%}

// Test 8 conversion operators
%rename(opbool) operator bool;
%rename(opuint) operator unsigned int;

%exception ConversionOperators::ConversionOperators() "$action /* ConversionOperators::ConversionOperators() */";
%exception ConversionOperators::~ConversionOperators() "$action /* ConversionOperators::~ConversionOperators() */";
%exception ConversionOperators::operator bool "$action /* ConversionOperators::operator bool */";
%exception ConversionOperators::operator unsigned int "$action /* ConversionOperators::unsigned int*/";

%inline %{
  class ConversionOperators {
  public:
    operator bool() { return false; }
    operator unsigned int() { return 0; }
  };
%}

