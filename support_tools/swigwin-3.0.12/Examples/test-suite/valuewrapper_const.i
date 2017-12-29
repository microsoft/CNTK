%module valuewrapper_const

%inline %{
// B requires swig to use the SwigValueWrapper 
class B 
{ 
private: 
  B() { } 
public: 
  B(const B&){ } 
}; 
 
// A returns a B and a const B 
class A 
{ 
  B b;
public: 
  A(const B& b) : b(b) { }
  // this one works    SwigValueWrapper< B > result; 
  B GetB() {
        return b;
  } 
  // this one is incorrect     B result; 
  const B GetBconst() const {
        return b;
  }
  ::B GetBGlobalQualifier() {
        return b;
  }
  const ::B GetBconstGlobalGlobalQualifier() const {
        return b;
  }
}; 

%}
 
