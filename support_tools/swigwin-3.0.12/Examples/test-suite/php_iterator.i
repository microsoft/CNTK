/* php_iterator.i - PHP-specific testcase for wrapping to a PHP Iterator */
%module php_iterator

%typemap("phpinterfaces") MyIterator "Iterator";

%inline %{

class MyIterator {
  int i, from, to;
public:
  MyIterator(int from_, int to_)
    : i(from_), from(from_), to(to_) { }
  void rewind() { i = from; }
  bool valid() const { return i != to; }
  int key() const { return i - from; }
  int current() const { return i; }
  void next() { ++i; }
};

%}
