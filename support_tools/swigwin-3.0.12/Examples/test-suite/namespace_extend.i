%module namespace_extend

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) bar; /* Ruby, wrong class name */

%{
namespace foo {
   class bar {
   public:
   };
}
foo::bar *new_foo_bar() {
   return new foo::bar;
}
void     delete_foo_bar(foo::bar *self) {
   delete self;
}
int foo_bar_blah(foo::bar *self, int x) {
   return x;
}
%}

namespace foo {
    class bar {
    public:
    %extend {
        bar();
       ~bar();
        int blah(int x);
    };
  };
}



