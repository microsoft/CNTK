/*
This testcase checks that a virtual destructor with void as a parameter is 
correctly handled.
Also tests a class with protected destructor derived from a class with a
public destructor.
*/

%module virtual_destructor

%inline %{

class VirtualVoidDestructor {
public:
  VirtualVoidDestructor() {};
  virtual ~VirtualVoidDestructor(void) { };
};

class Derived : public VirtualVoidDestructor {
protected:
  virtual ~Derived() {};
};
%}
