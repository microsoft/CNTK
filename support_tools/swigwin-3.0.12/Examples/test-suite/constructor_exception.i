%module constructor_exception

%inline %{
class MyError {
};

class SomeClass {
public:
   SomeClass(int x) {
       if (x < 0) {
           throw MyError();
       }
   }
};

class Test {
  SomeClass o;
public:
  Test(int x) try : o(x) { }
  catch (MyError &) {
  } 
  catch (int) {
  }
  catch (...) {
  }
};
%}
