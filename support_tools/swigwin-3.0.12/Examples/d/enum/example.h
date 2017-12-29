/* File : example.h */

enum color { RED, BLUE, GREEN };

class Foo {
 public:
  Foo() { }
  enum speed { IMPULSE=10, WARP=20, LUDICROUS=30 };
  void enum_test(speed s);
};

void enum_test(color c, Foo::speed s);

