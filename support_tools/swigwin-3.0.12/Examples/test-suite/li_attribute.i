%module li_attribute

%include <exception.i>

//#define SWIG_ATTRIBUTE_TEMPLATE
%include <attribute.i>

%{
// forward reference needed if using SWIG_ATTRIBUTE_TEMPLATE
struct A;
struct MyFoo; // %attribute2 does not work with templates
%}

%attribute(A, int, a, get_a, set_a);
%attributeref(A, int, b);

%attributeref(Param<int>, int, value);


%attribute(A, int, c, get_c);  /* read-only */
%attributeref(A, int, d, b);   /* renames accessor method 'b' to attribute name 'd' */

%attributeref(B, A*, a)

%inline
{
  struct A
  {
    A(int a, int b, int c) : _a(a), _b(b), _c(c)
    {
    }
    
    int get_a() const 
    {
      return _a;
    }
    
    void set_a(int aa) 
    {
      _a = aa;
    }

    /* only one ref method */
    int& b() 
    {
      return _b;
    }    

    int get_c() const 
    {
      return _c;
    }
  private:
    int _a;
    int _b;
    int _c;
  };

  template <class C>
  struct Param 
  {
    Param(C v) : _v(v)
    {
    }

    const int& value() const 
    {
      return _v;
    }
    
    int& value() 
    {
      return _v;
    }    
  private:
    C _v;
  }; 
  
  
  struct B {
    B(A *a) : mA(a)
    {
    }
    
    A*& a() { return mA; }
    
  protected:
    A*  mA;
  };
 
}

%template(Param_i) Param<int>;


// class/struct attribute with get/set methods using return/pass by reference
%attribute2(MyClass, MyFoo, Foo, GetFoo, SetFoo);
%inline %{
  struct MyFoo { 
    MyFoo() : x(-1) {}
    int x;
  };
  class MyClass {
    MyFoo foo;
  public:
    MyFoo& GetFoo() { return foo; }
    void SetFoo(const MyFoo& other) { foo = other; }
  };
%} 


// class/struct attribute with get/set methods using return/pass by value
%attributeval(MyClassVal, MyFoo, ReadWriteFoo, GetFoo, SetFoo);
%attributeval(MyClassVal, MyFoo, ReadOnlyFoo, GetFoo);
%inline %{
  class MyClassVal {
    MyFoo foo;
  public:
    MyFoo GetFoo() { return foo; }
    void SetFoo(MyFoo other) { foo = other; }
  };
%} 


// string attribute with get/set methods using return/pass by value
%include <std_string.i>
%attributestring(MyStringyClass, std::string, ReadWriteString, GetString, SetString);
%attributestring(MyStringyClass, std::string, ReadOnlyString, GetString);
%inline %{
  class MyStringyClass {
    std::string str;
  public:
    MyStringyClass(const std::string &val) : str(val) {}
    std::string GetString() { return str; }
    void SetString(std::string other) { str = other; }
  };
%} 


