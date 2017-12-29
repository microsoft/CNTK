/* This test checks whether SWIG correctly parses the new constructor
   inheritance.
*/
%module cpp11_inheriting_constructors

%inline %{
// Delegating constructors
class BaseClass {
private:
  int _val;
public:
  BaseClass(int iValue) { _val = iValue; }
};

// Constructor inheritance via using declaration
class DerivedClass: public BaseClass {
public:
  using BaseClass::BaseClass; // Adds DerivedClass(int) constructor
};

// Member initialization at the site of the declaration
class SomeClass {
public:
    SomeClass() {}
    explicit SomeClass(int new_value) : value(new_value) {}

    int value = 5;
};
%}
