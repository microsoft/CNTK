/* File : example.h */

enum color { RED, BLUE, GREEN };

class Foo {
 public:
  Foo() { }
  enum speed { IMPULSE, WARP, LUDICROUS };
  void enum_test(speed s);
};

void enum_test(color c, Foo::speed s);

