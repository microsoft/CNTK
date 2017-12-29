/* File : example.h */

#include <iostream>

using namespace std;

class Bar
{
 public:
   virtual void bar () {
     cout << "bar" << endl;
   }
   virtual ~Bar() {}
};

class Foo
{
 public:
   virtual void foo () {
     cout << "foo" << endl;
   }
   virtual ~Foo() {}
};

class Foo_Bar : public Foo, public Bar
{
 public:
   virtual void fooBar () {
     cout << "foobar" << endl;
   }
};
